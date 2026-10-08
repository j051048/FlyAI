"""Native multi-file/partial V4 math and immutable loading on small CPU modules.

Plain-directory fixtures here exercise the legacy numerical loader. Signed/global
artifact tests below use weight_artifacts verification, never header-only hashes.
"""
import dataclasses
import json
import os
from pathlib import Path
import shutil

import pytest

torch = pytest.importorskip("torch")
ST = pytest.importorskip("safetensors.torch")
REF = pytest.importorskip("v4_ref_cpu")
V4 = pytest.importorskip("v4_stage")
DS = pytest.importorskip("v4_dspark_draft")
from shard import weight_artifacts as ART


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def native(tmp_path):
    args = REF.cpu_args(n_layers=4, n_routed_experts=6, n_activated_experts=2,
        compress_ratios=(0, 4, 8, 0, 0, 0), dspark_target_layer_ids=(1, 2, 3))
    oracle = REF.build_oracle(args, 17)
    weights = {name: tensor.detach().cpu().clone().contiguous()
               for name, tensor in oracle.state_dict().items()
               if not (name.startswith("mtp.") and (".embed." in name or ".head." in name))}
    def write(name, selected, count=2):
        directory = tmp_path / name; directory.mkdir()
        (directory / "config.json").write_text(json.dumps(dataclasses.asdict(args)), encoding="utf-8")
        names = sorted(selected)
        for index in range(count):
            part = {key: selected[key] for key in names[index::count]}
            if part:
                ST.save_file(part, str(directory / f"model{index:03d}-mp1.safetensors"))
        return directory
    complete = write("full", weights)
    head = write("head", {name: value for name, value in weights.items()
                         if name.startswith("layers.0.") or name == "embed.weight"})
    tail = write("tail", {name: value for name, value in weights.items()
                         if not name.startswith("layers.0.")})
    return args, weights, complete, head, tail, write


def forward(stages, ids, start):
    h = stages[0].embed(ids)
    for stage in stages:
        h = stage.forward(h, ids, start)
    return stages[-1].logits_all(h, full_logits=False)


def assert_math(args, complete, head_dir, tail_dir):
    baseline = V4.Stage(0, 4, args, head=True, tail=True, dspark=True, device="cpu").load(complete)
    head = V4.Stage(0, 1, args, head=True, device="cpu").load(head_dir)
    tail = V4.Stage(1, 4, args, tail=True, dspark=True, device="cpu").load(tail_dir)
    original_draft, local_draft = DS.DSparkTail(baseline).load(complete), DS.DSparkTail(tail).load(tail_dir)
    assert len(V4.weight_map(head_dir)) < len(V4.weight_map(complete))
    assert all(block.embed is tail.embed_tokens and block.head is tail.lm_head for block in local_draft.mtp)
    ids = torch.tensor([[3, 5, 7, 9, 11]])
    wanted, got = forward([baseline], ids, 0), forward([head, tail], ids, 0)
    assert torch.equal(got, wanted)
    tokens = got.argmax(-1)
    original_draft.prefill(tokens, baseline.tail_main_hidden())
    local_draft.prefill(tokens, tail.tail_main_hidden())
    current = tokens.unsqueeze(1)
    for position in range(ids.shape[1], ids.shape[1] + 4):
        wanted, got = forward([baseline], current, position), forward([head, tail], current, position)
        assert torch.equal(got, wanted)
        assert torch.equal(tail.tail_main_hidden(), baseline.tail_main_hidden())
        current = got.argmax(-1).unsqueeze(1)
        left = original_draft.advance_and_draft(current, baseline.tail_main_hidden(), start_pos=position)
        right = local_draft.advance_and_draft(current, tail.tail_main_hidden(), start_pos=position)
        assert all(torch.equal(a, b) for a, b in zip(left, right))
        assert torch.equal(original_draft.last_spec[1], local_draft.last_spec[1])


def test_native_small_files_and_layer_only_directories_preserve_prefill_decode_and_mtp(native):
    args, _, complete, head_dir, tail_dir, _ = native
    assert_math(args, complete, head_dir, tail_dir)


@pytest.fixture
def packaged(native, tmp_path):
    args, _, complete, _, _, _ = native
    # Synthetic encoder bytes are only hash-validation assets in this CPU test;
    # no tokenizer/model download or hardware acceptance is being represented.
    (complete / "tokenizer.json").write_text("{}", encoding="utf-8")
    (complete / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    from v4_pipe import V4_MODEL_ID
    catalog, packing = ART.catalogue_directory(complete, V4_MODEL_ID, source={"fixture": "CPU native math"})
    ART.write_metadata(complete, catalog, packing)
    sizes = []
    def read_range(row, offset, size):
        sizes.append(size)
        with (complete / row["path"]).open("rb") as stream:
            stream.seek(offset); return stream.read(size)
    head, tail = tmp_path / "packed-head", tmp_path / "packed-tail"
    for directory, lo, hi, flags in ((head, 0, 1, {"head": True}),
                                     (tail, 1, 4, {"tail": True, "dspark": True})):
        descriptor = ART.select_stage_artifacts(catalog, packing, lo, hi, **flags)
        ART.repack_stage(catalog, packing, descriptor, directory, read_range,
                         max_file_bytes=128 << 10, chunk_bytes=251)
    cohort = {"model_id": catalog["model_id"], "checkpoint_id": catalog["checkpoint_id"],
              "manifest_sha256": catalog["manifest_sha256"], "config_sha256": catalog["config_sha256"],
              "runtime_abi": catalog["runtime_abi"], "n_layers": args.n_layers}
    return args, complete, head, tail, catalog, cohort, sizes


def test_verified_stream_repacked_stages_preserve_real_v4_and_drafter_math(packaged):
    args, complete, head, tail, catalog, _, sizes = packaged
    assert max(sizes) <= 251
    assert_math(args, complete, head, tail)
    full = V4.verify_checkpoint_artifacts(complete)
    partial = V4.verify_checkpoint_artifacts(tail, lo=1, hi=4, head=False, tail=True, dspark=True)
    assert full["checkpoint_id"] == partial["checkpoint_id"] == catalog["checkpoint_id"]
    assert full["verification_scope"] == "complete packed payload hashes"
    assert partial["verification_scope"] == "assigned stage payload hashes"
    assert set(partial["pack"]["weight_map"]) == set(partial["tensor_names"])
    assert not any(name.startswith("layers.0.") for name in partial["tensor_names"])


def test_partial_artifact_refuses_wrong_span_role_or_global_identity(packaged):
    args, _, head, tail, _, _, _ = packaged
    with pytest.raises(ART.ArtifactError, match="assignment differs"):
        V4.Stage(0, 1, args, head=True, device="cpu").load(tail)
    with pytest.raises(ART.ArtifactError, match="pinned model"):
        V4.Stage(0, 1, args, head=True, device="cpu").load(head, expected_checkpoint_id="tensor-sha256:" + "0" * 64)
    V4.verify_checkpoint_artifacts(head, lo=0, hi=1, head=True, tail=False, dspark=False)
    with pytest.raises(ART.ArtifactError, match="assignment differs"):
        V4.verify_checkpoint_artifacts(head, lo=0, hi=1, head=False)


def test_partial_payload_tamper_fails_full_hash_before_parameter_loading(packaged):
    args, _, head, _, _, _, _ = packaged
    path = next(head.glob("*.safetensors"))
    with path.open("r+b") as stream:
        stream.seek(-1, 2); byte = stream.read(1)
        stream.seek(-1, 2); stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ART.ArtifactError, match="payload hash mismatch"):
        V4.Stage(0, 1, args, head=True, device="cpu").load(head)


def test_drafter_cannot_load_a_different_global_checkpoint(packaged, native):
    args, complete, _, _, _, _, _ = packaged
    stage = V4.Stage(0, 4, args, head=True, tail=True, dspark=True, device="cpu").load(complete)
    _, weights, _, _, _, write = native
    alternate = write("alternate-model", {name: value + 1 if value.is_floating_point() else value
        for name, value in weights.items()})
    (alternate / "tokenizer.json").write_text("{}")
    catalog, packing = ART.catalogue_directory(alternate, ART.read_json(complete / ART.GLOBAL_FILE)["model_id"])
    ART.write_metadata(alternate, catalog, packing)
    with pytest.raises(RuntimeError, match="MTP checkpoint differs"):
        DS.DSparkTail(stage).load(alternate)


def test_coordinator_assets_only_prove_global_identity_without_claiming_local_weights(packaged, tmp_path):
    _, complete, _, _, catalog, cohort, _ = packaged
    import v4_pipe
    directory = tmp_path / "coordinator"; directory.mkdir()
    shutil.copyfile(complete / ART.GLOBAL_FILE, directory / ART.GLOBAL_FILE)
    for name in catalog["assets"]:
        shutil.copyfile(complete / name, directory / name)
    proof = v4_pipe.verify_cohort_directory(directory, cohort, production=False)
    assert proof["checkpoint_id"] == catalog["checkpoint_id"]
    assert proof["payload_integrity_verified"] is False
    assert proof["verification_scope"] == "pinned catalogue and local model assets only"
    assert not list(directory.glob("*.safetensors"))
    with pytest.raises(ART.ArtifactError):
        v4_pipe.verify_cohort_directory(directory, cohort, weights=True, production=False)
    (directory / "special_tokens_map.json").write_text("{}")
    with pytest.raises(ART.ArtifactError, match="unverified local encoder"):
        v4_pipe.verify_cohort_directory(directory, cohort, production=False)


def test_verified_artifact_metadata_is_immutable_and_json_is_not_a_witness(packaged):
    _, _, head, _, _, _, _ = packaged
    proof = V4.verify_checkpoint_artifacts(head)
    with pytest.raises(ART.ArtifactError, match="actual unchanged"):
        ART.verified_stage_artifact_descriptor(dict(proof))
    path = head / ART.STAGE_FILE
    path.write_text(path.read_text() + " ")
    with pytest.raises(RuntimeError, match="immutable checkpoint artifact metadata changed"):
        V4.verify_checkpoint_artifacts(head)


@pytest.mark.parametrize("field,value", [("runtime_abi", "unknown"), ("wire_version", "unknown"),
    ("numeric_contract", "unknown"), ("quantization", "fp16")])
def test_production_native_cohort_rejects_unsupported_contracts(field, value):
    from v4_artifact_contract import validate_native_cohort
    config = json.loads((Path(__file__).resolve().parents[1] / "vendor/deepseek_v4_ref/inference/config.json").read_text())
    cohort = {"model_id": "deepseek-ai/DeepSeek-V4-Flash", "runtime_abi": "deepseek-v4-native/1",
              "wire_version": "shard-pipeline-session/1", "numeric_contract": "greedy-native-fp4-fp8/1",
              "quantization": "fp4-fp8", "n_layers": 43}
    assert validate_native_cohort(cohort, config) == cohort
    cohort[field] = value
    with pytest.raises(ValueError, match="unsupported V4 cohort"):
        validate_native_cohort(cohort, config)


@pytest.mark.parametrize("field,value", [("n_layers", 40), ("dim", 5120), ("n_routed_experts", 384),
    ("dtype", "bf16"), ("expert_dtype", "fp8"), ("scale_fmt", None), ("dspark_target_layer_ids", [37, 38, 39])])
def test_new_architecture_or_quantization_is_not_accepted_as_old_native_flash(field, value):
    from v4_artifact_contract import validate_native_config
    config = json.loads((Path(__file__).resolve().parents[1] / "vendor/deepseek_v4_ref/inference/config.json").read_text())
    config[field] = value
    with pytest.raises(ValueError, match="unsupported V4"):
        validate_native_config(config, model_id="deepseek-ai/DeepSeek-V4-Flash", runtime_abi="deepseek-v4-native/1")


def test_streamed_native_fp4_weights_and_scale_aliases_survive_the_real_cache_loader(tmp_path):
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("packed byte loader needs recent PyTorch formats")
    args = REF.cpu_args(n_layers=1, n_mtp_layers=0, n_routed_experts=6, n_activated_experts=2,
        expert_dtype="fp4", compress_ratios=(0,), dspark_target_layer_ids=())
    model = V4.ref(); V4._set_globals(model, args)
    with torch.device("cpu"), model.set_dtype(torch.bfloat16):
        layer = model.Block(0, args)
    with torch.no_grad():
        for index, parameter in enumerate(layer.parameters()):
            if parameter.dtype == torch.float4_e2m1fn_x2:
                view = parameter.view(torch.uint8)
                view.copy_((torch.arange(view.numel()) + index).remainder(256).to(torch.uint8).reshape(view.shape))
            elif parameter.dtype == torch.float8_e8m0fnu:
                parameter.view(torch.uint8).fill_(127)
            elif parameter.dtype in (torch.int32, torch.int64):
                parameter.zero_()
            else:
                parameter.fill_(.02)
    original = {"layers.0." + name: value.detach().clone().contiguous() for name, value in layer.state_dict().items()}
    source = tmp_path / "packed-source"; source.mkdir()
    (source / "config.json").write_text(json.dumps(dataclasses.asdict(args)), encoding="utf-8")
    ST.save_file(original, str(source / "model0-mp1.safetensors"))
    catalog, packing = ART.catalogue_directory(source, "test/tiny-packed-native")
    descriptor = ART.select_stage_artifacts(catalog, packing, 0, 1)
    def read(row, offset, size):
        with (source / row["path"]).open("rb") as stream:
            stream.seek(offset); return stream.read(size)
    destination = tmp_path / "packed-stage"
    ART.repack_stage(catalog, packing, descriptor, destination, read, chunk_bytes=251)
    loaded = V4.Stage(0, 1, args, device="cpu", expert_placement="ram",
        expert_cache_reference=True, expert_cache_slots=2).load(destination)
    runtime = loaded.layers[0].ffn._hybrid_runtime
    with runtime.cache.acquire([0, 1]) as lease:
        lease.wait_on()
        for eid in (0, 1):
            canonical = loaded.layers[0].ffn.experts[eid]
            cached = runtime.cache.experts[lease.mapping[eid]]
            for kind in ("w1", "w2", "w3"):
                for member in ("weight", "scale"):
                    wanted = original[f"layers.0.ffn.experts.{eid}.{kind}.{member}"].view(torch.uint8)
                    assert torch.equal(getattr(getattr(canonical, kind), member).view(torch.uint8), wanted)
                    assert torch.equal(getattr(getattr(cached, kind), member).view(torch.uint8), wanted)
                assert getattr(canonical, kind).weight.scale is getattr(canonical, kind).scale
                assert getattr(cached, kind).weight.scale is getattr(cached, kind).scale


def test_duplicate_native_tensor_names_never_silently_override(native):
    _, weights, _, _, _, write = native
    directory = write("duplicates", weights, count=1)
    name = next(iter(weights))
    ST.save_file({name: weights[name]}, str(directory / "model999-mp1.safetensors"))
    with pytest.raises(RuntimeError, match="duplicate tensor"):
        V4.weight_map(directory)


def test_cached_config_is_not_mutated_by_serving_overrides(native):
    _, _, directory, _, _, _ = native
    one = V4.config(directory)
    expected = one.max_seq_len
    one.max_seq_len += 100
    assert V4.config(directory).max_seq_len == expected
    config_path = directory / "config.json"
    body = json.loads(config_path.read_text()); body["max_seq_len"] += 1
    config_path.write_text(json.dumps(body))
    with pytest.raises(RuntimeError, match="immutable checkpoint config changed"):
        V4.config(directory)


def test_cached_weight_map_detects_in_place_file_mutation(native):
    _, weights, _, _, _, write = native
    directory = write("mutation", weights, count=1)
    V4.weight_map(directory)
    path = directory / "model000-mp1.safetensors"
    selected = dict(weights); name = next(iter(selected)); selected[name] = torch.zeros_like(selected[name])
    ST.save_file(selected, str(path))
    with pytest.raises(RuntimeError, match="immutable checkpoint .*changed"):
        V4.weight_map(directory)


def test_cached_raw_handle_refuses_replacement_stamp_instead_of_old_mmap(native):
    _, weights, directory, _, _, _ = native
    name = next(iter(weights))
    assert torch.equal(V4.raw(name, directory), weights[name])
    path = directory / V4.weight_map(directory)[name]
    stamp = path.stat()
    # Windows cannot always unlink a live mmap. A changed stamp is enough to
    # prove the cache must refuse rather than returning its old mapped tensor.
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
    with pytest.raises(RuntimeError, match="immutable checkpoint weight changed"):
        V4.raw(name, directory)


def test_missing_partial_mtp_parameter_is_not_replaced_by_random_initialization(native):
    args, weights, _, _, _, write = native
    selected = {name: value for name, value in weights.items() if not name.startswith("layers.0.")
                and name != "mtp.0.attn.wq_a.weight"}
    directory = write("missing-mtp", selected)
    stage = V4.Stage(1, 4, args, tail=True, dspark=True, device="cpu").load(directory)
    with pytest.raises(RuntimeError, match="missing"):
        DS.DSparkTail(stage).load(directory)
