"""Token-provenance V4 text protocol and bounded UTF-8 stream decoding.

The native renderer owns control tokens. Identical spelling inside message text
is encoded as ordinary bytes. Parsing tools is diagnostic only: the text gateway
does not advertise tool execution or turn arbitrary strings into API tool calls.
"""
from __future__ import annotations

import codecs
import copy
import json
import re

MAX_TOKENS = 16384
MAX_TEXT_BYTES = 4 << 20
CONTROL_TEXT = frozenset(("<think>", "</think>", "｜DSML｜"))


def _reserved(text):
    return text in CONTROL_TEXT or text.startswith("<｜") and text.endswith("｜>")


def _renderer():
    # Repo import first, deployed flat tree second. No dynamic tokenizer code.
    try:
        from vendor.deepseek_v4_ref.encoding import encoding_dsv4
    except ImportError:
        from deepseek_v4_ref.encoding import encoding_dsv4
    return encoding_dsv4


def encode_messages_ids(tokenizer, messages, thinking_mode="chat", *, reasoning_effort=None):
    """Render supported text messages while keeping literal controls unprivileged.

    Ordinary messages retain the original rendered encoding. Only a message
    containing a reserved spelling takes the segmented path. A detached BPE
    encoder prevents mutable added-token options racing another request.
    """
    if (not isinstance(messages, list) or not messages or
            any(not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or
                not isinstance(m.get("content"), str) or set(m) - {"role", "content", "reasoning_content"}
                or "reasoning_content" in m and not isinstance(m["reasoning_content"], str)
                for m in messages)):
        raise ValueError("V4 token protocol requires supported text messages")
    if sum(len(m.get(k, "").encode()) for m in messages for k in ("content", "reasoning_content")) > MAX_TEXT_BYTES:
        raise ValueError("V4 text protocol exceeds message byte budget")
    render = _renderer().encode_messages
    added = tokenizer.get_added_vocab() if callable(getattr(tokenizer, "get_added_vocab", None)) else {}
    controls = {text: token for text, token in added.items() if _reserved(text)}
    if not controls or not any(text in m.get(k, "") for m in messages
                               for k in ("content", "reasoning_content") for text in controls):
        return tokenizer.encode(render(messages, thinking_mode, reasoning_effort=reasoning_effort),
                                add_special_tokens=False)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("literal V4 controls require the supported local Fast encoder")
    from tokenizers import Tokenizer
    spec = json.loads(backend.to_str())
    spec["added_tokens"] = [row for row in spec.get("added_tokens", []) if row["content"] not in controls]
    ordinary = Tokenizer.from_str(json.dumps(spec))
    ordinary.no_padding(); ordinary.no_truncation()
    control_ids = set(controls.values())
    pattern = re.compile("(" + "|".join(re.escape(text) for text in sorted(controls, key=len, reverse=True)) + ")")
    def literal(text):
        ids = []
        for part in pattern.split(text):
            # Splitting the spelling prevents BPE from reconstructing a reserved
            # vocabulary entry even when it was also present in model.vocab.
            chunks = (part[:1], part[1:]) if part in controls else (part,)
            for chunk in chunks:
                ids.extend(ordinary.encode(chunk, add_special_tokens=False).ids)
        if control_ids.intersection(ids):
            raise ValueError("literal text unexpectedly encoded a V4 control ID")
        return ids
    detached, values = copy.deepcopy(messages), {}
    all_text = json.dumps(messages, ensure_ascii=False)
    nonce = 0
    while f"\ue000V4TEXT{nonce}:" in all_text:
        nonce += 1
    for message in detached:
        for key in ("content", "reasoning_content"):
            if key in message:
                marker = f"\ue000V4TEXT{nonce}:{len(values)}\ue001"
                values[marker], message[key] = message[key], marker
    template = render(detached, thinking_mode, reasoning_effort=reasoning_effort)
    split = re.compile("(" + "|".join(re.escape(v) for v in values) + ")")
    ids = []
    for part in split.split(template):
        ids.extend(literal(values[part]) if part in values else
                   tokenizer.encode(part, add_special_tokens=False))
    return ids


def _byte_alphabet():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars, extra = list(values), 0
    for value in range(256):
        if value not in values:
            values.append(value); chars.append(256 + extra); extra += 1
    return {chr(char): value for value, char in zip(values, chars)}


class V4TextDecoder:
    """Decode ByteLevel tokens with a real incremental UTF-8 boundary.

    A valid literal U+FFFD is retained; only an incomplete byte suffix is held.
    Non-ByteLevel test/adaptor tokenizers retain their original decode contract.
    The parser/decoder never infers controls from user-visible text spelling.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        backend = getattr(tokenizer, "backend_tokenizer", None)
        spec = json.loads(backend.to_str()) if backend is not None else {}
        self.byte_level = (spec.get("decoder") or {}).get("type") == "ByteLevel"
        self.alphabet = _byte_alphabet()
        self.added = {row["id"]: row["content"] for row in spec.get("added_tokens", [])}
        self.special = set(getattr(tokenizer, "all_special_ids", ()))
        self.controls = {text: token for token, text in self.added.items() if _reserved(text)}

    def token_bytes(self, token):
        if type(token) is not int or token < 0:
            raise ValueError("nonnegative integer output token required")
        if token in self.added:
            return self.added[token].encode("utf-8")
        text = self.tokenizer.convert_ids_to_tokens(token)
        if not isinstance(text, str):
            raise ValueError("output token is outside the loaded tokenizer vocabulary")
        try:
            return bytes(self.alphabet[c] for c in text)
        except KeyError as error:
            raise ValueError("unsupported non-byte V4 vocabulary entry") from error

    def decode_prefix(self, tokens, *, final=False):
        if len(tokens) > MAX_TOKENS:
            raise ValueError("V4 output token budget exceeded")
        if not self.byte_level:
            text = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return text if final else text.rstrip("\ufffd")
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        parts, size = [], 0
        for token in tokens:
            if token in self.special:
                continue
            data = self.token_bytes(token)
            size += len(data)
            if size > MAX_TEXT_BYTES:
                raise ValueError("V4 output text budget exceeded")
            parts.append(decoder.decode(data, final=False))
        parts.append(decoder.decode(b"", final=final))
        return "".join(parts)


class V4OutputParser:
    """Incremental token-aware reasoning events and strict final DSML parsing.

    Text made from ordinary token IDs never starts reasoning or tool markup,
    even if it spells the reserved marker. Every genuine DSML marker is retained
    in plain_text; unsupported/malformed calls are never fabricated as valid.
    """
    def __init__(self, tokenizer, *, thinking=False):
        self.codec = V4TextDecoder(tokenizer)
        if not self.codec.byte_level:
            raise ValueError("V4 protocol parser requires the native ByteLevel tokenizer")
        self.channel = "reasoning" if thinking else "content"
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.parts, self.count, self.size, self.closed = [], 0, 0, False

    def feed(self, token):
        if self.closed or self.count >= MAX_TOKENS:
            raise ValueError("closed or exhausted V4 output parser")
        self.count += 1
        marker = self.codec.added.get(token)
        events = []
        if marker in ("<think>", "</think>"):
            tail = self.decoder.decode(b"", final=True)
            if tail:
                self.parts.append((self.channel, tail, False)); events.append({"channel": self.channel, "text": tail})
            self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
            self.channel = "reasoning" if marker == "<think>" else "content"
            return events
        if token in self.codec.special:
            return events
        data = self.codec.token_bytes(token)
        self.size += len(data)
        if self.size > MAX_TEXT_BYTES:
            raise ValueError("V4 protocol text budget exceeded")
        text = self.decoder.decode(data, final=False)
        if text:
            genuine = marker == "｜DSML｜"
            self.parts.append((self.channel, text, genuine))
            events.append({"channel": "dsml_marker" if genuine else self.channel, "text": text})
        return events

    def finish(self):
        if self.closed:
            raise ValueError("V4 output parser already finished")
        self.closed = True
        tail = self.decoder.decode(b"", final=True)
        if tail:
            self.parts.append((self.channel, tail, False))
        content = "".join(t for c, t, _ in self.parts if c == "content")
        reasoning = "".join(t for c, t, _ in self.parts if c == "reasoning")
        # Protect ordinary-token spelling from the upstream string parser. Pick
        # a placeholder absent from the complete bounded stream and restore it
        # in parsed arguments. Provenance, not a regex match, grants authority.
        placeholder = "\ue000"
        while placeholder in content:
            placeholder += "\ue000"
        genuine_placeholder = placeholder + "V4DSML"
        while genuine_placeholder in content:
            genuine_placeholder += "X"
        protected = "".join(genuine_placeholder if genuine else t
                            for c, t, genuine in self.parts if c == "content")
        protected = protected.replace("｜DSML｜", placeholder).replace(genuine_placeholder, "｜DSML｜")
        start = "<｜DSML｜tool_calls"
        at = protected.find(start)
        result = {"content": content, "reasoning_content": reasoning,
                  "tool_calls": [], "dsml_status": "absent"}
        if at >= 0:
            try:
                end, stop, calls = _renderer().parse_tool_calls(at + len(start), protected)
                if stop != "</｜DSML｜tool_calls>" or protected[end:].strip():
                    raise ValueError("incomplete DSML tool block or unexpected suffix")
                result["tool_calls"] = json.loads(json.dumps(calls, ensure_ascii=False).replace(placeholder, "｜DSML｜"))
                result["content"] = protected[:at].replace(placeholder, "｜DSML｜")
                result["dsml_status"] = "valid"
            except (ValueError, AssertionError, TypeError, json.JSONDecodeError):
                result["dsml_status"] = "malformed"
        return result
