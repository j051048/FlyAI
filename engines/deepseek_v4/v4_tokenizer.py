"""Local V4 encoder loading, isolated from native ModelArgs/config dtype.

The native engine needs dtype='fp8'; Transformers AutoConfig interprets that
as a Torch dtype before consulting tokenizer_class. Select the known encoder
class directly and preserve the original encoder assets/config unchanged.
"""
import json
from pathlib import Path


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate V4 tokenizer configuration field")
        value[key] = item
    return value


def load_v4_tokenizer(directory):
    """Load the supported local Fast encoder; no AutoConfig/remote code/network.

    Tokenizer-base loading retains AddedToken attributes, chat templates and
    special-token configuration. No regex, vocabulary or rendering rewrite is
    applied, and native config.json is neither parsed nor modified here.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("V4 tokenizer requires an existing local directory")
    config_file, tokenizer_file = directory / "tokenizer_config.json", directory / "tokenizer.json"
    if not config_file.is_file() or not tokenizer_file.is_file():
        raise ValueError("V4 requires local tokenizer.json and tokenizer_config.json")
    if config_file.stat().st_size > 128 << 20:
        raise ValueError("V4 tokenizer configuration exceeds metadata limit")
    with config_file.open("r", encoding="utf-8-sig") as stream:
        config = json.load(stream, object_pairs_hook=_unique)
    if not isinstance(config, dict) or config.get("tokenizer_class") != "PreTrainedTokenizerFast":
        raise ValueError("unsupported V4 tokenizer_class; require PreTrainedTokenizerFast")
    if config.get("auto_map"):
        raise ValueError("V4 encoder does not support remote/dynamic tokenizer code")
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(str(directory),
        tokenizer_file=str(tokenizer_file), local_files_only=True, trust_remote_code=False)
