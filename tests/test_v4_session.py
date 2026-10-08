"""Strict signed sessions around the actual tiny V4 runtime/reference, CPU only."""
import threading

import pytest

torch = pytest.importorskip("torch")
import v4_pipe as VP
from shard.pipeline_plan import build_plan
from shard.pipeline_session import SessionConfig, SessionBusy
from shard.receipt import gen_key, pub_b64, verify_coverage, wire_receipt


def test_strict_full_v4_ring_preserves_reference_tokens_receipt_chain_and_remote_owner(tmp_path, monkeypatch):
    R = pytest.importorskip("v4_ref_cpu")
    pytest.importorskip("v4_stage")
    args = R.cpu_args(); oracle = R.build_oracle(args)
    VP._write_tiny_checkpoint(str(tmp_path), args, oracle)
    prompt, maximum = (168, 15, 493, 72, 22), 6
    reference = VP._reference_tokens(oracle, prompt, maximum)
    keys = [gen_key(), gen_key(), gen_key()]
    # This fixture injects existing keys explicitly; key-file ACL/read policy has
    # independent tests and is not weakened in the application.
    monkeypatch.setattr(VP, "load_or_make_node_key", lambda path: keys[int(str(path)[-1])])
    ports = VP._free_ports(2)
    plan = build_plan({"n_layers": args.n_layers}, ring_id="v4-strict-test", cohort_id="a" * 64,
        endpoints=[f"127.0.0.1:{port}" for port in ports], split=f"3,{args.n_layers-3}")
    for index in range(2):
        plan["stages"][index]["signer_pubkey"] = pub_b64(keys[index])
    plan["coordinator"]["signer_pubkey"] = pub_b64(keys[2])
    configs = [SessionConfig.from_plan(plan, index, caller_key=keys[index]) for index in range(2)]
    coord = SessionConfig.from_plan(plan, -1, caller_key=keys[2])
    events, errors, threads = [threading.Event(), threading.Event()], [], []
    def run(index):
        row = plan["stages"][index]
        try:
            VP.serve_stage(index, 2, row["lo"], row["hi"], ports[index],
                nxt=row["next_endpoint"], ckpt_dir=str(tmp_path), args=args, device="cpu",
                receipts=True, key_path="key" + str(index), ready=events[index],
                session_config=configs[index], timeout=5)
        except Exception as error:
            errors.append(error)
            events[index].set()
    for index in reversed(range(2)):
        thread = threading.Thread(target=run, args=(index,), daemon=True); thread.start(); threads.append(thread)
    assert all(event.wait(20) for event in events) and not errors, errors
    head, tail = plan["coordinator"]["head"], plan["coordinator"]["tail"]
    pipe, ret = VP.connect_ring(head, tail, timeout=5, retry_s=5, session_config=coord)
    try:
        with pytest.raises(SessionBusy):
            VP.connect_ring(head, tail, timeout=1, retry_s=0.5, session_config=coord)
        assignments = {pub_b64(keys[index]): (row["lo"], row["hi"]) for index, row in enumerate(plan["stages"])}
        result = VP.coordinate(pipe, ret, prompt, maximum, receipts=True, layer_count=args.n_layers,
            nonce="1" * 64, swarm_id=plan["ring_id"], job_id="strict-job", expected_by_signer=assignments,
            strict_job_binding=True, timeout=5)
        assert result["tokens"] == reference and result["receipts_ok"] is True
        verify_coverage([wire_receipt(row) for row in result["receipts"]], args.n_layers,
            expected_by_signer=assignments, expected_nonce="1" * 64, check_chain=True)
        VP.send_msg(pipe, {"op": "stop"})
    finally:
        pipe.close(); ret.close()
        for thread in threads:
            thread.join(10)
    assert not errors and not any(thread.is_alive() for thread in threads), errors
