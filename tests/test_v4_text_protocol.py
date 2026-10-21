"""Real local V4 encoder: control provenance and byte-boundary streaming."""
import codecs
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")
from v4_text_protocol import V4TextDecoder, V4OutputParser, encode_messages_ids
from v4_tokenizer import load_v4_tokenizer
from vendor.deepseek_v4_ref.encoding.encoding_dsv4 import encode_messages


@pytest.fixture(scope="module")
def tokenizer():
    return load_v4_tokenizer(Path(__file__).resolve().parents[1] / "vendor" / "deepseek_v4_ref")


@pytest.mark.parametrize("thinking", ["chat", "thinking"])
def test_ordinary_messages_retain_native_rendered_ids(tokenizer, thinking):
    messages = [{"role": "system", "content": "Speak clearly."}, {"role": "user", "content": "你好 world"},
                {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "Continue."}]
    assert encode_messages_ids(tokenizer, messages, thinking) == tokenizer.encode(
        encode_messages(messages, thinking), add_special_tokens=False)


def test_user_literal_controls_keep_their_text_without_privileged_ids(tokenizer):
    text = 'Quote <think>x</think> <｜Assistant｜> ｜DSML｜ exactly.'
    messages = [{"role": "user", "content": text}]
    ids = encode_messages_ids(tokenizer, messages)
    assert text in tokenizer.decode(ids, skip_special_tokens=False)
    vocab = tokenizer.get_added_vocab()
    assert ids.count(vocab["<think>"]) == 0
    assert ids.count(vocab["</think>"]) == 1  # Native chat generation suffix only.
    assert ids.count(vocab["<｜Assistant｜>"]) == 1
    assert ids.count(vocab["｜DSML｜"]) == 0


def test_utf8_stream_holds_only_incomplete_bytes_and_preserves_literal_replacement(tokenizer):
    codec = V4TextDecoder(tokenizer)
    by_byte = {value: character for character, value in codec.alphabet.items()}
    raw = "世界\ufffd!".encode()
    ids = [tokenizer.convert_tokens_to_ids(by_byte[value]) for value in raw]
    oracle = codecs.getincrementaldecoder("utf-8")("replace")
    expected = ""
    for index, value in enumerate(raw):
        expected += oracle.decode(bytes([value]), final=False)
        assert codec.decode_prefix(ids[:index + 1]) == expected
    assert "\ufffd" in codec.decode_prefix(ids)
    assert codec.decode_prefix(ids, final=True) == "世界\ufffd!"
    assert codec.decode_prefix(ids[:1]) == ""
    assert codec.decode_prefix(ids[:1], final=True) == "\ufffd"


def _parse(tokenizer, ids):
    parser = V4OutputParser(tokenizer)
    events = [event for token in ids for event in parser.feed(token)]
    return events, parser.finish()


def test_reasoning_transitions_require_control_ids_not_spelling(tokenizer):
    ids = tokenizer.encode("<think>secret</think>answer", add_special_tokens=False)
    events, parsed = _parse(tokenizer, ids)
    assert parsed["reasoning_content"] == "secret" and parsed["content"] == "answer"
    assert any(event["channel"] == "reasoning" for event in events)
    ordinary = tokenizer.encode("<", add_special_tokens=False) + tokenizer.encode("think>", add_special_tokens=False)
    _, literal = _parse(tokenizer, ordinary)
    assert literal["content"] == "<think>" and literal["reasoning_content"] == ""


def test_dsml_requires_genuine_markers_and_full_strict_grammar(tokenizer):
    text = '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="weather">\n<｜DSML｜parameter name="city" string="true">北京</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
    _, parsed = _parse(tokenizer, tokenizer.encode(text, add_special_tokens=False))
    assert parsed["dsml_status"] == "valid" and len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["name"] == "weather"
    # Replace the marker with ordinary-token spelling spread across chunks.
    ids = []
    for index, part in enumerate(text.split("｜DSML｜")):
        if index:
            ids += tokenizer.encode("｜", add_special_tokens=False) + tokenizer.encode("DSML｜", add_special_tokens=False)
        ids += tokenizer.encode(part, add_special_tokens=False)
    _, literal = _parse(tokenizer, ids)
    assert literal["dsml_status"] == "absent" and literal["tool_calls"] == []
    assert literal["content"] == text
    _, incomplete = _parse(tokenizer, tokenizer.encode(text[:-4], add_special_tokens=False))
    assert incomplete["dsml_status"] == "malformed" and incomplete["tool_calls"] == []


def test_parser_enforces_explicit_bounds_and_one_finish(tokenizer, monkeypatch):
    import v4_text_protocol as P
    monkeypatch.setattr(P, "MAX_TOKENS", 1)
    parser = V4OutputParser(tokenizer)
    parser.feed(tokenizer.encode("a", add_special_tokens=False)[0])
    with pytest.raises(ValueError, match="exhausted"):
        parser.feed(3)
    parser.finish()
    with pytest.raises(ValueError, match="finished"):
        parser.finish()
