"""Real CPU V4 sockets: sealed IDs preserve decode and signed BF16/FP8 chains."""
import collections
import tempfile
import threading

import pytest

from test_v4_pipe import tiny, VP, PROMPT, wire_receipt, verify_coverage
from v4_privacy import TokenPrivacy, wire_token_bytes

SECRET = bytes(range(32))
NONCE = "a9" * 32


class _PrivacyRing:
    def __init__(self, checkpoint, args, *, codecs=None, dspark=False):
        # Hash layers span two trusted stages; the third is an opaque middle.
        self.ranges = [(0, 1), (1, 2), (2, 5), (5, args.n_layers)]
        self.args = args
        self.errors = []
        self.threads = []
        ports = VP._free_ports(len(self.ranges))
        events = [threading.Event() for _ in self.ranges]
        keys = tempfile.mkdtemp(prefix="privacy-ring-", dir=checkpoint)
        def run(kwargs):
            try:
                VP.serve_stage(**kwargs)
            except Exception as exc:
                # Keep only text: retaining a traceback also retains live sockets.
                self.errors.append((type(exc).__name__, str(exc)))
        for i, (lo, hi) in enumerate(self.ranges):
            kwargs = dict(stage=i, nstages=len(self.ranges), lo=lo, hi=hi, port=ports[i],
                          nxt=f"127.0.0.1:{ports[i + 1]}" if i < len(self.ranges) - 1 else None,
                          ckpt_dir=checkpoint, device="cpu", receipts=True, key_path=f"{keys}/s{i}.key",
                          dspark=dspark, ready=events[i], timeout=5.0)
            if codecs is not None:
                kwargs["token_privacy"] = codecs[i]
            thread = threading.Thread(target=run, args=(kwargs,), daemon=True)
            thread.start()
            self.threads.append(thread)
        for event in events:
            assert event.wait(15), f"stage startup failed: {self.errors}"
        self.pipe, self.ret = VP.connect_ring(f"127.0.0.1:{ports[0]}", f"127.0.0.1:{ports[-1]}", timeout=5)

    def coordinate(self, mode, *, timeout=8.0):
        kwargs = dict(nonce=NONCE, receipts=True, layer_count=self.args.n_layers, timeout=timeout)
        if mode == "greedy":
            return VP.coordinate(self.pipe, self.ret, list(PROMPT), 8, **kwargs)
        if mode == "spec":
            return VP.coordinate_spec(self.pipe, self.ret, list(PROMPT), 8,
                                      K=3, drafter=VP._RepeatDrafter(), **kwargs)
        if mode == "dspark":
            return VP.coordinate_dspark(self.pipe, self.ret, list(PROMPT), 8, **kwargs)
        return VP.coordinate_dspark_pipelined(self.pipe, self.ret, list(PROMPT), 8,
                                              lazy=True, depth=4, **kwargs)

    def close(self):
        try:
            VP.send_msg(self.pipe, {"op": "stop"})
        except OSError:
            pass
        self.pipe.close()
        for thread in self.threads:
            thread.join(timeout=1.0)
        self.ret.close()


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("mode", ["greedy", "spec", "dspark", "pipelined"])
def test_sealed_real_ring_matches_legacy_and_preserves_signed_chain(tiny, monkeypatch, fp8, mode):
    checkpoint, args, _reference = tiny
    monkeypatch.setattr(VP, "V4_FP8_WIRE", fp8)
    monkeypatch.setattr(VP, "V4_SEALED_IDS", False)
    monkeypatch.delenv("SHARD_V4_TOKEN_KEY_FILE", raising=False)
    legacy = _PrivacyRing(checkpoint, args, dspark=mode in ("dspark", "pipelined"))
    try:
        baseline = legacy.coordinate(mode)
    finally:
        legacy.close()
    assert not legacy.errors

    trusted = TokenPrivacy(SECRET)
    ordinary = TokenPrivacy(key_id=trusted.key_id)
    monkeypatch.setattr(VP, "V4_TOKEN_PRIVACY_KEY_ID", trusted.key_id)
    monkeypatch.setattr(VP, "V4_SEALED_IDS", True)
    observed = []
    original = VP._recv_hids
    def observe(msg, signer, **kwargs):
        result = original(msg, signer, **kwargs)
        if kwargs.get("token_privacy") is not None:
            observed.append(dict(lo=kwargs["stage_lo"], is_tail=kwargs.get("is_tail", False),
                                 has_plain_ids="ids" in msg, has_plain_hint="dnxt" in msg,
                                 dprev=msg.get("dprev"), envelope=msg["sealed_ids"],
                                 local_ids=result[1].tolist()))
        return result
    monkeypatch.setattr(VP, "_recv_hids", observe)
    sealed = _PrivacyRing(checkpoint, args, codecs=[trusted, trusted, ordinary, trusted],
                          dspark=mode in ("dspark", "pipelined"))
    try:
        actual = sealed.coordinate(mode)
        assert actual["tokens"] == baseline["tokens"], (mode, fp8)
    finally:
        sealed.close()
    assert not sealed.errors
    assert actual["receipts_ok"] is True
    wired = sorted((wire_receipt(receipt) for receipt in actual["receipts"]), key=lambda x: x["layer_start"])
    verify_coverage(wired, args.n_layers, expected_nonce=NONCE, check_chain=True)
    for left, right in zip(wired, wired[1:]):
        assert left["out_root"] == right["in_root"]
    assert observed
    assert not any(item["has_plain_ids"] or item["has_plain_hint"] for item in observed)
    middle = [item for item in observed if item["lo"] == 2]
    assert middle and all(not any(token for row in item["local_ids"] for token in row) for item in middle)
    assert all(item["dprev"] in (None, True) for item in observed)
    # Every frame arrives at the three receivers with exactly the same envelope.
    envelopes = collections.Counter(wire_token_bytes(item["envelope"]) for item in observed)
    assert set(envelopes.values()) == {3}
    if mode == "pipelined":
        assert any(item["envelope"]["has_next_token"] for item in observed), "lazy dnxt never exercised"
        assert actual["drafted"] > 0
    if mode in ("dspark", "pipelined"):
        assert actual["drafted"] > 0, "real MTP drafter did not run"


@pytest.mark.parametrize("bad_member", ["legacy", "wrong_key"])
def test_mixed_ring_fails_before_token_execution(tiny, monkeypatch, bad_member):
    checkpoint, args, _reference = tiny
    trusted = TokenPrivacy(SECRET)
    codecs = [trusted, trusted, None if bad_member == "legacy" else TokenPrivacy(key_id="01" * 32), trusted]
    # Explicit None starts a legacy ordinary member; the coordinator is sealed.
    monkeypatch.setattr(VP, "V4_SEALED_IDS", False)
    ring = _PrivacyRing(checkpoint, args, codecs=codecs)
    monkeypatch.setattr(VP, "V4_SEALED_IDS", True)
    monkeypatch.setattr(VP, "V4_TOKEN_PRIVACY_KEY_ID", trusted.key_id)
    try:
        with pytest.raises((OSError, EOFError, RuntimeError)):
            ring.coordinate("greedy", timeout=0.5)
    finally:
        ring.close()
    assert any("legacy" in reason or "mode/key mismatch" in reason for _name, reason in ring.errors)


def test_stale_head_frame_exposes_no_tokens_and_preserves_warm_ring(tiny, monkeypatch):
    checkpoint, args, _reference = tiny
    trusted = TokenPrivacy(SECRET)
    monkeypatch.setattr(VP, "V4_SEALED_IDS", True)
    monkeypatch.setattr(VP, "V4_TOKEN_PRIVACY_KEY_ID", trusted.key_id)
    fenced_frames = []
    original_send = VP._KeepWarm.send
    def observe(self, msg):
        if isinstance(msg, dict) and msg.get("fenced"):
            fenced_frames.append(dict(msg))
        return original_send(self, msg)
    monkeypatch.setattr(VP._KeepWarm, "send", observe)
    ring = _PrivacyRing(checkpoint, args,
                        codecs=[trusted, trusted, TokenPrivacy(key_id=trusted.key_id), trusted])
    def drive(nonce, inject):
        VP.send_msg(ring.pipe, {"op": "reset", "swarm_id": "swarm", "job_id": "job", "nonce": nonce,
                                "token_privacy": trusted.descriptor(), "temp": 0.0,
                                "spec": True, "dspark": False})
        assert VP.recv_msg(ring.ret) == "ok"
        VP.send_msg(ring.pipe, {"op": "step", "ids": [list(PROMPT)], "start_pos": 0, "epoch": 1})
        first = VP.recv_msg(ring.ret)["token"]
        if inject:
            VP.send_msg(ring.pipe, {"op": "step", "ids": [[7]], "dnxt": 129279,
                                    "start_pos": len(PROMPT) - 1, "epoch": 0})
            assert VP.recv_msg(ring.ret) == {"fenced": True, "epoch": 0, "pos": len(PROMPT) - 1}
        VP.send_msg(ring.pipe, {"op": "step", "ids": [[first]], "start_pos": len(PROMPT), "epoch": 1})
        second = VP.recv_msg(ring.ret)["token"]
        receipts, ok = VP._sweep_receipts(ring.pipe, ring.ret, args.n_layers, nonce)
        assert ok is True
        verify_coverage([wire_receipt(item) for item in receipts], args.n_layers,
                        expected_nonce=nonce, check_chain=True)
        return [first, second]
    try:
        clean = drive(NONCE, False)
        after_stale = drive("b8" * 32, True)
    finally:
        ring.close()
    assert clean == after_stale
    assert not ring.errors
    assert len(fenced_frames) == 3
    assert all(set(frame) <= {"op", "start_pos", "epoch", "cpos", "fenced"} for frame in fenced_frames)
