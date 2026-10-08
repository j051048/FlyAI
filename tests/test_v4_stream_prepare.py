"""Streaming conversion uses real packed bytes and the reference conversion math."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from shard import weight_artifacts as A
from v4_stream_prepare import HFSource, LocalTarget, convert_hf, native_name


@pytest.fixture
def hf_source(tmp_path):
    root = tmp_path / "hf"; root.mkdir()
    config = {"n_layers": 3, "n_mtp_layers": 3, "dspark_target_layer_ids": [2], "expert_dtype": "fp4"}
    weights = {f"model.layers.{i}.mlp.experts.0.w1.weight": torch.arange(64, dtype=torch.int8).reshape(2, 32)
               for i in range(3)}
    weights.update({f"mtp.{i}.weight": torch.ones(4) for i in range(3)})
    weights.update({name: torch.ones(4) for name in ("model.embed.weight", "model.head.weight",
        "model.norm.weight", "model.hc_head_fn", "model.hc_head_base", "model.hc_head_scale")})
    weights["mtp.0.embed.weight"] = torch.ones(4)  # reference conversion omits this alias.
    weights["model.layers.0.self_attn.wo_a.weight"] = torch.arange(256*256).reshape(256,256).remainder(8).to(torch.float8_e4m3fn)
    scales = {"model.layers.0.self_attn.wo_a.weight_scale_inv": torch.ones(2,2).to(torch.float8_e8m0fnu)}
    save_file(weights, str(root / "first.safetensors")); save_file(scales, str(root / "scale.safetensors"))
    (root / "tokenizer.json").write_text('{"fixture": true}')
    (root / "config.json").write_text(json.dumps(config))
    return root, config, {**weights, **scales}


def test_streamed_files_equal_upstream_mp1_conversion_tensor_bytes(hf_source, tmp_path):
    root, config, values = hf_source
    source = HFSource(tmp_path / "input-cache", directory=root)
    out = tmp_path / "streamed"
    result = convert_hf(source, config, [(LocalTarget(out), None)], model_id="test/v4",
                        max_file_bytes=4096, chunk_bytes=19, scratch=tmp_path)
    # Execute the original vendor converter on tiny tensors, then compare every
    # logical dtype/shape/payload against our bounded converter's actual files.
    reference = Path(__file__).resolve().parents[1] / "vendor/deepseek_v4_ref/inference/convert.py"
    spec = importlib.util.spec_from_file_location("reference_v4_convert", reference)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    expected = tmp_path / "reference"
    module.main(str(root), str(expected), 1, 1, "fp4")
    got = {}
    for path in out.glob("*.safetensors"): got.update(load_file(str(path)))
    gold = load_file(str(expected / "model0-mp1.safetensors"))
    assert set(got) == set(gold)
    for name in gold:
        assert got[name].dtype == gold[name].dtype and got[name].shape == gold[name].shape
        assert torch.equal(got[name].view(torch.uint8), gold[name].view(torch.uint8)), name
    verified = A.verify_weight_pack(out, expected_checkpoint_id=result["checkpoint_id"])
    assert verified["payload_integrity_verified"]
    assert len(list(out.glob("*.safetensors"))) > 1


def test_direct_stage_distribution_never_requires_a_complete_output_on_one_node(hf_source, tmp_path):
    root, config, _ = hf_source
    targets = []
    for index in range(3):
        roles = dict(lo=index, hi=index+1, head=index==0, tail=index==2, dspark=index==2)
        targets.append((LocalTarget(tmp_path / f"stage{index}"), roles))
    result = convert_hf(HFSource(tmp_path / "cache", directory=root), config, targets,
                        max_file_bytes=4096, chunk_bytes=23, scratch=tmp_path)
    all_names, global_roots = set(), set()
    for index in range(3):
        verified = A.verify_stage_artifacts(tmp_path / f"stage{index}", lo=index, hi=index+1)
        global_roots.add(verified["checkpoint_id"]); all_names.update(verified["tensor_names"])
        assert all(not n.startswith(f"layers.{j}.") for n in verified["tensor_names"] for j in range(3) if j!=index)
    assert global_roots == {result["checkpoint_id"]}
    assert "mtp.2.weight" in all_names
    assert not any(p.suffix == ".safetensors" for p in tmp_path.iterdir())


def test_missing_destination_coverage_fails_before_writing_weights(hf_source, tmp_path):
    root, config, _ = hf_source
    sink = LocalTarget(tmp_path / "only-head")
    with pytest.raises(A.ArtifactError, match="full converted"):
        convert_hf(HFSource(tmp_path / "cache", directory=root), config,
            [(sink, dict(lo=0,hi=1,head=True,tail=False,dspark=False))])
    assert not (tmp_path / "only-head").exists()


def test_native_name_preserves_reference_alias_and_scale_semantics():
    assert native_name("model.layers.1.self_attn.wq_a.weight_scale_inv") == "layers.1.attn.wq_a.scale"
    assert native_name("mtp.2.head.weight") is None
    assert native_name("model.layers.0.mlp.gate.e_score_correction_bias") == "layers.0.ffn.gate.bias"


def test_source_cache_refuses_a_single_shard_larger_than_its_budget(monkeypatch, tmp_path):
    from get_model import DownloadSpec
    import get_model
    # Actual metadata seam, no external network: the file cannot be downloaded
    # because its bounded disk budget is rejected before the downloader runs.
    spec = DownloadSpec("one.safetensors", 100, "sha256", "a" * 64)
    monkeypatch.setattr("phase0.get_model.resolve_files", lambda *a,**k:("b"*40,[spec]))
    with pytest.raises(A.ArtifactError, match="cache budget"):
        HFSource(tmp_path / "cache", repo="test/model", max_cache_bytes=99,
                 downloader=lambda *a,**k:pytest.fail("oversized input downloaded"))


def test_target_disk_failure_precedes_any_output_payload(hf_source, tmp_path, monkeypatch):
    root, config, _ = hf_source
    sink = LocalTarget(tmp_path / "small-node")
    source = HFSource(tmp_path / "cache", directory=root)
    monkeypatch.setattr(sink, "preflight", lambda size: (_ for _ in ()).throw(A.ArtifactError("disk ceiling")))
    with pytest.raises(A.ArtifactError, match="disk ceiling"):
        convert_hf(source, config, [(sink, None)], scratch=tmp_path)
    assert not list(sink.staging.glob("*.safetensors"))


def test_verified_source_mutation_is_not_relabelled_as_original(hf_source, tmp_path):
    root, _, _ = hf_source
    source = HFSource(tmp_path / "cache", directory=root)
    path = root / "first.safetensors"; data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(A.ArtifactError, match="changed"):
        source.file("first.safetensors")


def test_verified_input_replacement_between_lookup_and_open_is_rejected(hf_source, tmp_path, monkeypatch):
    import os
    root, _, _ = hf_source
    source = HFSource(tmp_path / "cache", directory=root)
    original = source.file
    replaced = False
    def race(name):
        nonlocal replaced
        path = original(name)
        if not replaced:
            replacement = path.with_suffix(".replacement")
            raw = bytearray(path.read_bytes()); raw[-1] ^= 1; replacement.write_bytes(raw)
            os.replace(replacement, path); replaced = True
        return path
    monkeypatch.setattr(source, "file", race)
    name = next(iter(source.weight_map))
    with pytest.raises(A.ArtifactError, match="replaced"):
        source.read(name, 0, min(10, source.info(name)["storage_bytes"]))


def test_shared_filesystem_budget_includes_output_scratch_not_only_final_stage(hf_source, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import v4_stream_prepare as stream
    root, config, _ = hf_source
    source = HFSource(tmp_path / "cache", directory=root)
    sink = LocalTarget(tmp_path / "small-output")
    monkeypatch.setattr(stream.shutil, "disk_usage", lambda path: SimpleNamespace(free=310000))
    with pytest.raises(A.ArtifactError, match="shared filesystem"):
        convert_hf(source, config, [(sink, None)], scratch=tmp_path)
    assert not sink.staging.exists() and not sink.root.exists()


def test_upload_failure_cleans_only_owned_unpublished_staging(hf_source, tmp_path, monkeypatch):
    root, config, _ = hf_source
    source = HFSource(tmp_path / "cache", directory=root)
    protected = tmp_path / "running-old-model"; protected.mkdir(); (protected / "weights").write_bytes(b"old")
    sink = LocalTarget(tmp_path / "new-output")
    original = sink.put
    count = 0
    def fail(name, path, size, sha):
        nonlocal count
        original(name, path, size, sha); count += 1
        if count == 2: raise A.ArtifactError("injected transfer failure")
    monkeypatch.setattr(sink, "put", fail)
    with pytest.raises(A.ArtifactError, match="transfer failure"):
        convert_hf(source, config, [(sink, None)], scratch=tmp_path)
    assert not sink.staging.exists() and not sink.root.exists()
    assert (protected / "weights").read_bytes() == b"old"


def test_small_scale_is_read_once_when_weight_and_scale_shards_cannot_coexist(hf_source, tmp_path, monkeypatch):
    from get_model import DownloadSpec
    import v4_stream_prepare as stream
    root, config, _ = hf_source
    paths = [p for p in root.iterdir() if p.is_file()]
    specs = [DownloadSpec(p.name, p.stat().st_size, "sha256", A.hash_file(p)[0]) for p in paths]
    monkeypatch.setattr("phase0.get_model.resolve_files", lambda *a,**k:("b"*40,specs))
    downloads = []
    def download(cache, spec, url, **kwargs):
        target = Path(cache) / spec.path; target.write_bytes((root / spec.path).read_bytes())
        downloads.append(spec.path); return spec.digest
    maximum = max(spec.size for spec in specs)
    source = HFSource(tmp_path / "cache", repo="test/immutable", max_cache_bytes=maximum, downloader=download)
    convert_hf(source, config, [(LocalTarget(tmp_path / "new"), None)], scratch=tmp_path)
    # Header scan, early boundary tensors, then the continuous weight pass:
    # the latter must not re-download once per 128-row conversion block.
    assert downloads.count("first.safetensors") <= 3
    assert downloads.count("scale.safetensors") <= 2
