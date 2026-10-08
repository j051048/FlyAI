"""Bring up the gpt-oss-120B swarm (even OR VRAM-aware uneven split) and drive it with the
MODEL-FREE n-gram spec-decode coordinator — the long-context path (no draft model, no draft
KV, so it survives past 100k where the 20b vLLM draft OOMs).

Reuses launch_oss's instance/ssh helpers; launches stages tail-first with --fast
--direct-return. The coordinator is CPU-only (tokenizer + n-gram drafter + ring drive), so it
runs ON the head stage box: in-region 0ms coord->head, no separate coord box.

  # N=3 heterogeneous (fat 48GB node holds 18 layers, two 24GB nodes 9 each), 100k+ ctx:
  python launch_ngram.py --stages 42195546,42195544,42195547 --layers 18,9,9 \
      --max-ctx 131072 --prompt-file /root/prompt_long.txt --prefill-chunk 4096 \
      --depth 4 --K 4 --ngram-n 3 --max-new 256

--layers gives the per-stage layer counts (must sum to the model's layer_count); omit for an
even split. Stage order is taken as given (head first); the fat node should usually be the head.
Teardown is manual (vastai destroy).
"""
import argparse

from launch_oss import ep, fire, instances, rssh, warm_stage, M120, PORT, PSK, managed_command, remote_config, process_command  # noqa: F401


def launch_stage_uneven(inst, stage, nstages, nxt_ep, served_head, lo, hi, max_ctx, timeout, window=False, receipts=False, sync_send=False):
    """launch_oss.launch_stage + explicit --lo/--hi (uneven split) + a longer edge timeout for
    the multi-minute long-context prefill. window=True sets FV_WINDOW=1 so sliding-attention
    layers read only their 128-key window at decode (O(window) not O(ctx)) — the long-context
    speed lever. sync_send=True sets SHARD_SYNC_SEND=1 (force the old synchronous forward send, the
    TTFT A/B baseline). NEVER pkill -f specpipe (self-match); kill by GPU pid + port."""
    argv = ["python3", "phase0/specpipe.py", "--legacy-protocol", "--stage", str(stage), "--nstages", str(nstages),
            "--model", M120, "--listen-port", str(PORT), "--fast", "--direct-return", "--lo", str(lo), "--hi", str(hi),
            "--max-ctx", str(max_ctx), "--timeout", str(timeout)]
    if nxt_ep: argv += ["--next", nxt_ep]
    if served_head: argv.append("--served-head")
    env = {"SHARD_PSK": PSK}
    if window: env["FV_WINDOW"] = "1"
    if receipts: env["SHARD_RECEIPTS"] = "1"
    if sync_send: env["SHARD_SYNC_SEND"] = "1"
    fire(inst, managed_command(f"specpipe.stage{stage}", argv, env))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", required=True, help="comma ids, head first")
    ap.add_argument("--layers", default="", help="per-stage layer counts e.g. 18,9,9 (sum=layer_count); else even")
    ap.add_argument("--max-ctx", type=int, default=131072)
    ap.add_argument("--prompt-file", default="/root/prompt_long.txt")
    ap.add_argument("--prefill-chunk", type=int, default=4096)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--ngram-n", type=int, default=3)
    ap.add_argument("--reasoning", default="low", help="gpt-oss reasoning_effort (low cuts the analysis channel)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--edge-timeout", type=int, default=1800, help="per-edge timeout (long prefill needs >600s)")
    ap.add_argument("--window", action="store_true", help="FV_WINDOW=1: sliding layers read only their 128-key window at decode (long-ctx speed)")
    ap.add_argument("--sync-send", action="store_true", help="SHARD_SYNC_SEND=1: force the OLD synchronous inter-stage send (TTFT A/B baseline)")
    ap.add_argument("--receipts", action="store_true", help="SHARD_RECEIPTS=1: each stage signs a per-block receipt; coordinator collects + verifies (PROVE)")
    ap.add_argument("--no-launch", action="store_true", help="ring already warm; just run the coordinator")
    a = ap.parse_args()

    sids = [int(x) for x in a.stages.split(",")]
    insts = instances()
    stages = [insts[i] for i in sids]
    nstages = len(stages)
    eps = [ep(s) for s in stages]

    from shard.pipeline_plan import parse_split, model_layers
    config = remote_config(stages[0], M120)
    ranges = parse_split(a.layers, model_layers(config), nstages)

    head = stages[0]
    for k, s in enumerate(stages):
        ip, hp = eps[k]
        lo, hi = ranges[k]
        print(f"  stage{k} {s['id']} ({s.get('geolocation')}) {ip}:{hp} layers[{lo}:{hi}]"
              f"{' [head+coord]' if k == 0 else ''}{' [tail]' if k == nstages - 1 else ''}", flush=True)

    if not a.no_launch:
        print("[launch] stages tail-first (--fast --direct-return, flex prefill / graphed eager decode)...", flush=True)
        for k in range(nstages - 1, -1, -1):
            nxt = f"{eps[k + 1][0]}:{eps[k + 1][1]}" if k < nstages - 1 else None
            lo, hi = ranges[k]
            launch_stage_uneven(stages[k], k, nstages, nxt, served_head=(k == 0),
                                lo=lo, hi=hi, max_ctx=a.max_ctx, timeout=a.edge_timeout, window=a.window,
                                receipts=a.receipts, sync_send=a.sync_send)
            label, ok = warm_stage(stages[k], f"stage{k} {stages[k]['id']}")
            print(f"  {'OK ' if ok else 'FAIL '}{label}", flush=True)
            if not ok:
                print("[abort] stage failed to warm", flush=True)
                return

    head_ep = f"127.0.0.1:{PORT}"                       # coordinator runs ON the head box -> localhost to stage 0
    tail_ep = f"{eps[nstages - 1][0]}:{eps[nstages - 1][1]}"
    print(f"\n[coord] n-gram coordinator on head {head['id']}: --next {head_ep} --tail {tail_ep}", flush=True)
    renv = " SHARD_RECEIPTS=1" if a.receipts else ""
    argv = ["python3", "phase0/specpipe.py", "--legacy-protocol", "--coordinator", "--nstages", str(nstages),
            "--model", M120, "--ngram-draft", "--ngram-n", str(a.ngram_n), "--pipe", "--depth", str(a.depth), "--K", str(a.K),
            "--next", head_ep, "--direct-return",
            "--tail", tail_ep, "--prompt-file", a.prompt_file,
            "--prefill-chunk", str(a.prefill_chunk), "--max-ctx", str(a.max_ctx), "--max-new", str(a.max_new),
            "--reasoning", a.reasoning, "--timeout", str(a.edge_timeout), "--dump", "/root/run.json"]
    env = {"SHARD_PSK":PSK}
    if a.receipts: env['SHARD_RECEIPTS']='1'
    if a.sync_send: env['SHARD_SYNC_SEND']='1'
    r = rssh(head, process_command(argv, env), timeout=a.edge_timeout + 600)
    if r.returncode: raise SystemExit('coordinator failed')
    print(r.stdout[-3500:], flush=True)
    if r.stderr.strip():
        print("[stderr]", r.stderr[-1200:], flush=True)
    print("\n[done] ring still warm; teardown: vastai destroy instance <id>", flush=True)


if __name__ == "__main__":
    main()
