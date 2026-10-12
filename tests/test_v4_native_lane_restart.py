"""Real process/SQLite lease restart cannot turn durable output into migrated KV.

The child runs the original tiny CPU Stage and signed native control handler.
Socket epoch handover is covered separately in test_v4_native_lane_socket.py;
this test deliberately destroys every Python object between the two attempts.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


COHORT = "c" * 64
NODE, GPU, PRINCIPAL = "restart-node", "GPU-cpu-math-fixture", "local-controller"


def _ledger(root):
    from shard.leases import LeaseLedger
    return LeaseLedger(root / "leases.sqlite", node_id=NODE,
        authorize=lambda principal, action, binding: PRINCIPAL if principal == PRINCIPAL else None)


def _child(root, phase):
    import hashlib
    import secrets
    import socket
    import torch
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    import v4_ref_cpu as R
    import v4_pipe as VP
    from test_v4_conversation_cache import stage, forward
    from v4_native_lanes import NativeLanes, CONTRACT
    from v4_native_lane_protocol import handle, SCHEMA, DOMAIN, ACK_DOMAIN
    from v4_conversation_protocol import _sign, _verify, canonical
    from shard.job_journal import JobJournal
    from shard.leased_runtime import install_stage_lease_checks
    from shard.leases import LeaseRequest, LeaseResources, LeaseExpired
    from shard.pipeline_plan import build_plan
    from shard.pipeline_session import SessionConfig, SessionRegistry
    from shard.receipt import pub_b64, verify_coverage, ReceiptError

    torch.set_num_threads(1)
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
        dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
    model = args, R.build_oracle(args, 7)
    st, control = stage(model), stage(model)
    # Fixed test identities survive restart; runtime/session ownership must not.
    node_key = Ed25519PrivateKey.from_private_bytes(bytes([11]) * 32)
    coordinator_key = Ed25519PrivateKey.from_private_bytes(bytes([12]) * 32)
    plan = build_plan({"n_layers": 4}, ring_id="restart-ring", cohort_id=COHORT,
        endpoints=["127.0.0.1:19999"], node_ids=[NODE], gpu_uuids=[GPU])
    plan["stages"][0]["signer_pubkey"] = pub_b64(node_key)
    plan["coordinator"]["signer_pubkey"] = pub_b64(coordinator_key)
    session = SessionConfig.from_plan(plan, 0, caller_key=node_key)
    registry = SessionRegistry(session, node_key)
    owner_socket, peer = socket.socketpair()
    grant = registry.acquire(secrets.token_hex(16), owner_socket)
    assert session.verify_grant(grant) == grant
    epoch = f"{grant['boot_id']}:{grant['fence']}"

    ledger = _ledger(root)
    ledger.register_capacity("host", available_ram_bytes=64 << 20, pinnable_ram_bytes=0,
                             gpu_capacity_bytes={GPU: 64 << 20})
    request = LeaseRequest("restart-ring", COHORT, NODE, GPU, "host",
                           LeaseResources(1, 32 << 20), 120)
    lease = ledger.prepare(request, principal=PRINCIPAL, idempotency_key=phase)
    lease = ledger.commit(lease.lease_id, lease.fencing_token, principal=PRINCIPAL)
    guard = ledger.guard(lease.lease_id, lease.fencing_token, principal=PRINCIPAL,
                         ring_id=plan["ring_id"], model_cohort_sha256=COHORT)
    resident = guard.begin_work("resident-stage-" + phase)
    install_stage_lease_checks(st, guard)
    st._conversation_lease_guard = guard
    runner = st._native_lanes = NativeLanes(st, max_jobs=2, host_bytes=32 << 20,
        state_directory=root / "private-prefix", disk_bytes=16 << 20)
    identity = hashlib.sha256(b"tenant-a/native-restart-fixture").hexdigest()
    binding = {"job_id": "job-" + phase, "nonce": secrets.token_hex(32), "swarm_id": plan["ring_id"]}

    def call(action, parameters, *, owner_epoch=epoch, expect=True):
        request = {"schema": SCHEMA, "contract": CONTRACT, "action": action,
            "request_id": secrets.token_hex(16), "job": binding, "identity_sha256": identity,
            "owner_epoch": owner_epoch, "cohort_id": COHORT, "horizon": 64, "parameters": parameters}
        frame = {"op": "reset" if action == "open" else "native_lane",
                 "request": _sign(coordinator_key, DOMAIN, request)}
        if action == "open":
            frame["native_lane_open"] = True
        response = handle(st, frame, stage_index=0, session=session, signer_key=node_key,
                          vp=VP, owner_grant=grant)
        ack = _verify(pub_b64(node_key), ACK_DOMAIN, response["acks"][0])
        assert ack["request_sha256"] == hashlib.sha256(canonical(request)).hexdigest()
        assert ack["job"] == binding and ack["stage"] == 0 and ack["ok"] is expect, ack
        return ack["result"], {"request": request, "ack": response["acks"][0]}

    journal = JobJournal(root / "private-jobs" / "jobs.sqlite")
    journal_owner = journal.claim_owner(ttl_s=120)
    prior = None
    rejected = None
    if phase == "after":
        prior = json.loads((root / "before.json").read_text(encoding="utf-8"))
        assert lease.fencing_token > prior["lease"]["fencing_token"]
        assert grant["boot_id"] != prior["grant"]["boot_id"]
        try:
            ledger.assert_fence(prior["lease"]["lease_id"], prior["lease"]["fencing_token"], principal=PRINCIPAL)
        except LeaseExpired:
            pass
        else:
            raise AssertionError("stopped process's lease remained executable")
        # Even a still-valid old signature cannot claim the new live owner.
        try:
            call("open", {}, owner_epoch=prior["epoch"])
        except ValueError as error:
            assert "owner epoch" in str(error)
        else:
            raise AssertionError("old authenticated owner crossed a process restart")
        entry = prior["entry"]
        restore = {"transaction_id": secrets.token_hex(16), "prefix": entry["prefix"],
            "identity_sha256": identity, "stages": [{"stage": 0, "pubkey": pub_b64(node_key),
                "entry_id": entry["entry_id"], "state_digest": entry["state_digest"]}]}
        rejected, _ = call("open", {"restore": restore}, expect=False)
        assert "expired" in rejected["error"]
        assert runner.store.status()["loaded"] == 0 and not runner.active
        persisted = journal.get(journal_owner, tenant="a", job_id=prior["binding"]["job_id"])
        assert persisted["state"] == "completed" and persisted["tokens"] == prior["tokens"]
        # The completed response remains readable; it is not injected as live KV.
        prompt = prior["prompt"] + prior["tokens"] + [19, 20]
    else:
        prompt = list(range(1, 14))

    opened, open_evidence = call("open", {})
    assert opened["pos"] == 0 and opened["restored"] is False and opened["restore_tier"] is None
    generated, expected, consumed = [], [], []
    maximum = 3
    for pos in range(len(prompt) + maximum - 1):
        token = prompt[pos] if pos < len(prompt) else generated[-1]
        actual, _ = call("step", {"pos": pos, "token": token})
        _, logits = forward(control, torch.tensor([[token]]), pos)
        wanted = int(logits.reshape(-1, args.vocab_size)[-1].argmax())
        assert actual["token"] == wanted
        consumed.append(token)
        if pos >= len(prompt) - 1:
            generated.append(actual["token"]); expected.append(wanted)
    result, _ = call("receipt", {})
    receipt = result["receipt"]
    verify_coverage([receipt], 4, expected_by_signer={pub_b64(node_key): (0, 4)},
                    expected_nonce=binding["nonce"], check_chain=True)
    assert receipt["n_chunks"] == len(prompt) + maximum - 1
    assert "conversation_restore" not in receipt and generated == expected
    if prior:
        try:
            verify_coverage([prior["receipt"]], 4, expected_by_signer={pub_b64(node_key): (0, 4)},
                            expected_nonce=binding["nonce"])
        except ReceiptError:
            pass
        else:
            raise AssertionError("old committed output reused an old execution receipt")
    prefix = {"token_count": len(consumed), "digest": hashlib.sha256(canonical(consumed)).hexdigest()}
    closed, _ = call("close", {"retain": True, "prefix": prefix})
    assert runner.store.status()["written"] == 1
    record = {"id": binding["job_id"], "tenant": "a", "payload": {"prompt_ids": prompt},
        "context": {"model_id": "native-cpu-fixture", "cohort_id": COHORT},
        "prompt_tokens": len(prompt), "max_new": maximum, "timeout_s": 120,
        "fingerprint": hashlib.sha256(canonical(prompt)).hexdigest(), "idempotency_key": phase}
    journal.admit(journal_owner, record)
    journal.running(journal_owner, record["id"])
    journal.append_tokens(journal_owner, record["id"], 0, generated)
    journal.finish(journal_owner, record["id"], "completed", result={"tokens": generated, "receipt": receipt})
    if prior:
        assert journal.get(journal_owner, tenant="a", job_id=prior["binding"]["job_id"])["tokens"] == prior["tokens"]
    journal.release_owner(journal_owner)
    report = {"pid": os.getpid(), "lease": lease.to_dict(), "resident": resident.to_dict(),
        "grant": grant, "epoch": epoch, "binding": binding, "prompt": prompt, "tokens": generated,
        "entry": closed["cache"], "receipt": receipt, "open_evidence": open_evidence,
        "restore_rejection": rejected, "disk_loaded": runner.store.status()["loaded"]}
    (root / (phase + ".json")).write_text(json.dumps(report), encoding="utf-8")
    registry.release(grant); peer.close()
    if phase == "after":
        guard.end_work(resident)
        ledger.release(lease.lease_id, lease.fencing_token, principal=PRINCIPAL)
    # The first child deliberately leaves resident work recorded. Only its
    # parent's observed process exit may authorize local ledger cleanup.


def test_native_prefix_survives_on_disk_but_is_not_migrated_across_process_and_lease_restart(tmp_path):
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith("V4_"):
            env.pop(name)
    env.update(V4_KERNELS="cpu", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    script = str(Path(__file__).resolve())
    before = subprocess.run([sys.executable, script, "--child", str(tmp_path), "before"],
                            env=env, capture_output=True, text=True, timeout=60)
    assert before.returncode == 0, before.stdout + before.stderr
    old = json.loads((tmp_path / "before.json").read_text(encoding="utf-8"))
    assert old["pid"] != os.getpid() and old["entry"]
    assert list((tmp_path / "private-prefix").glob("*.safetensors"))
    ledger = _ledger(tmp_path)
    retired = ledger.release(old["lease"]["lease_id"], old["lease"]["fencing_token"], principal=PRINCIPAL)
    assert retired.state == "draining" and retired.active_work == 1
    checked = []
    def confirm_stopped(work):
        checked.append(work.to_dict())
        return before.returncode == 0 and work.to_dict() == old["resident"]
    assert ledger.recover_stopped_work(confirm_stopped) == 1
    assert checked == [old["resident"]]
    after = subprocess.run([sys.executable, script, "--child", str(tmp_path), "after"],
                           env=env, capture_output=True, text=True, timeout=60)
    assert after.returncode == 0, after.stdout + after.stderr
    fresh = json.loads((tmp_path / "after.json").read_text(encoding="utf-8"))
    assert fresh["pid"] != os.getpid()
    assert fresh["lease"]["fencing_token"] > old["lease"]["fencing_token"]
    assert fresh["grant"]["boot_id"] != old["grant"]["boot_id"]
    assert fresh["restore_rejection"] and fresh["disk_loaded"] == 0
    assert fresh["receipt"]["n_chunks"] == len(fresh["prompt"]) + len(fresh["tokens"]) - 1
    assert "conversation_restore" not in fresh["receipt"]


if __name__ == "__main__":
    assert sys.argv[1] == "--child"
    repo = Path(__file__).resolve().parents[1]
    for path in (repo, repo / "tests", repo / "engines" / "deepseek_v4", repo / "phase0"):
        sys.path.insert(0, str(path))
    _child(Path(sys.argv[2]), sys.argv[3])
