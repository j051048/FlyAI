"""Real local tokenizer assets; no GPU/model download and no native config rewrite."""
import hashlib
import json
from pathlib import Path

import pytest

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")
from engines.deepseek_v4.v4_tokenizer import load_v4_tokenizer


@pytest.fixture
def encoders(tmp_path):
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"<bos>": 0, "<eos>": 1, "<unk>": 2, "hello": 3, "world": 4, "世界": 5}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    config = {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<bos>", "eos_token": "<eos>",
              "pad_token": "<eos>", "unk_token": "<unk>", "model_max_length": 4096,
              "clean_up_tokenization_spaces": False,
              "chat_template": "{% for message in messages %}{{ message['content'] }} {% endfor %}{{ eos_token }}"}
    directories = []
    for name, model in (("native", {"n_layers": 43, "dim": 4096, "dtype": "fp8", "expert_dtype": "fp4"}),
                        ("hf", {"model_type": "llama", "dtype": "bfloat16"})):
        path = tmp_path / name; path.mkdir()
        tokenizer.save(str(path / "tokenizer.json"))
        (path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "config.json").write_text(json.dumps(model), encoding="utf-8")
        directories.append(path)
    return directories


def test_native_fp8_encoder_ids_attributes_and_chat_match_legal_hf_view(encoders):
    native, hf = encoders
    config_bytes = (native / "config.json").read_bytes()
    expected = transformers.AutoTokenizer.from_pretrained(hf, local_files_only=True, trust_remote_code=False)
    actual = load_v4_tokenizer(native)
    for text in ("hello world", "世界 hello", "<bos> hello <eos>", "unseen world"):
        assert actual.encode(text, add_special_tokens=False) == expected.encode(text, add_special_tokens=False)
        assert actual.decode(actual.encode(text)) == expected.decode(expected.encode(text))
    assert (actual.bos_token_id, actual.eos_token_id, actual.pad_token_id, actual.unk_token_id) == (0, 1, 1, 2)
    assert actual.special_tokens_map == expected.special_tokens_map
    assert actual.model_max_length == expected.model_max_length == 4096
    assert actual.chat_template == expected.chat_template
    messages = [{"role": "user", "content": "hello 世界"}]
    assert actual.apply_chat_template(messages, tokenize=True) == expected.apply_chat_template(messages, tokenize=True)
    assert (native / "config.json").read_bytes() == config_bytes
    assert hashlib.sha256((native / "config.json").read_bytes()).digest() == hashlib.sha256(config_bytes).digest()


def test_encoder_never_parses_native_model_config_or_calls_network(encoders, monkeypatch):
    native, _ = encoders
    def forbidden(*args, **kwargs):
        pytest.fail("local encoder attempted model config discovery or network access")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", forbidden)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    import requests
    monkeypatch.setattr(requests.Session, "send", forbidden)
    # Even a config the HF model loader cannot parse is irrelevant to encoding.
    (native / "config.json").write_bytes(b"not an HF model config")
    assert load_v4_tokenizer(native).encode("hello world") == [3, 4]


@pytest.mark.parametrize("class_name", [None, "AutoTokenizer", "LlamaTokenizer", "UnknownTokenizer"])
def test_unknown_encoder_class_cannot_silently_fall_back(encoders, class_name):
    native, _ = encoders
    config = json.loads((native / "tokenizer_config.json").read_text())
    config["tokenizer_class"] = class_name
    (native / "tokenizer_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unsupported V4 tokenizer_class"):
        load_v4_tokenizer(native)


def test_dynamic_tokenizer_code_and_duplicate_config_fields_are_refused(encoders):
    native, _ = encoders
    path = native / "tokenizer_config.json"
    path.write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "auto_map": {"AutoTokenizer": "evil.Class"}}))
    with pytest.raises(ValueError, match="remote/dynamic"):
        load_v4_tokenizer(native)
    path.write_text('{"tokenizer_class":"Unknown","tokenizer_class":"PreTrainedTokenizerFast"}')
    with pytest.raises(ValueError, match="duplicate"):
        load_v4_tokenizer(native)


def test_missing_local_assets_never_resolve_a_hub_model(tmp_path):
    with pytest.raises(ValueError, match="local tokenizer"):
        load_v4_tokenizer(tmp_path)
    with pytest.raises(ValueError, match="existing local directory"):
        load_v4_tokenizer(tmp_path / "no-model")


def test_helper_is_part_of_frozen_engine_source_identity():
    import v4_benchmark
    identity = v4_benchmark.source_identity()
    assert any(row["name"] == "engines/deepseek_v4/v4_tokenizer.py" for row in identity["files"])
