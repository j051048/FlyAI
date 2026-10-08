import types

import pytest
import v4_pipe as vp


def test_messages_use_actual_vendored_renderer_and_tokenizer():
    from pathlib import Path
    import sys
    enc = Path(vp._vendored("deepseek_v4_ref")) / "encoding"
    sys.path.insert(0, str(enc))
    from encoding_dsv4 import encode_messages
    job = {"messages": [{"role": "user", "content": "Write a hello-world program."}]}
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return list(text.encode())
    ids = vp._encode_prompt(Tokenizer(), job)
    assert bytes(ids).decode() == encode_messages(job["messages"], "chat")
    assert vp._encode_prompt(Tokenizer(), {"promptIds": [1, 2]}) == [1, 2]


@pytest.mark.parametrize("method", [vp.coordinate, vp.coordinate_spec,
                                    vp.coordinate_dspark, vp.coordinate_dspark_pipelined])
def test_cancel_before_reset_does_not_touch_ring(monkeypatch, method):
    sent = []
    monkeypatch.setattr(vp, "send_msg", lambda *args: sent.append(args))
    def cancelled():
        raise TimeoutError("request cancelled")
    with pytest.raises(TimeoutError, match="cancelled"):
        method(None, None, [1, 2], 3, cancel_check=cancelled)
    assert sent == []


def test_identity_reset_barrier_discards_old_tokens_then_validates_every_reply(monkeypatch):
    identity = {"job_id": "new", "nonce": "newnonce", "swarm_id": "s"}
    incoming = iter([{"token": 12, "job_id": "old"},
                     {"op": "reset_ok", "ok": True, **identity},
                     {"token": 42, **identity}, {"token": 99, "job_id": "old"}])
    outgoing = []
    monkeypatch.setattr(vp, "send_msg", lambda sock, msg: outgoing.append(msg))
    monkeypatch.setattr(vp, "recv_msg", lambda sock: next(incoming))
    send, recv, _ = vp._coord_io(None, None, identity)
    send(None, {"op": "reset"})
    assert outgoing[0]["reply_binding"] == 1
    assert recv(None)["op"] == "reset_ok"
    assert recv(None)["token"] == 42
    with pytest.raises(RuntimeError, match="bound"):
        recv(None)


def test_new_service_rejects_legacy_reset_ack_instead_of_silent_downgrade(monkeypatch):
    monkeypatch.setattr(vp, "send_msg", lambda *args: None)
    monkeypatch.setattr(vp, "recv_msg", lambda *args: "ok")
    send, recv, _ = vp._coord_io(None, None, {"job_id": "j", "nonce": "n", "swarm_id": "s"})
    send(None, {"op": "reset"})
    with pytest.raises(RuntimeError, match="bound"):
        recv(None)


def test_public_codec_does_not_read_inherited_private_key(monkeypatch):
    from v4_privacy import TokenPrivacy
    key_id = TokenPrivacy(b"k" * 32).key_id
    monkeypatch.setattr(vp, "V4_SEALED_IDS", True)
    monkeypatch.setattr(vp, "V4_TOKEN_PRIVACY_KEY_ID", key_id)
    monkeypatch.setenv("SHARD_V4_TOKEN_KEY_FILE", "not-a-real-key-file")
    codec = vp._configured_token_privacy()
    assert codec.key_id == key_id
    with pytest.raises(ValueError, match="secret"):
        codec.seal([[1]], {"job_nonce": "a" * 64, "start_pos": 0, "epoch": 0})
