"""Resource contracts fail closed before any memory-aware scheduler is enabled."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shard.node import LayerRange, ModelRuntime
from shard.plan import plan_ring
from shard.resources import (CalibrationProvenance, GpuRequirements, HostRequirements,
                             NodeResources, PlacementRequirements, ResourceError,
                             evaluate_fit, linux_cgroup_headroom, linux_host_resources)
from engines.deepseek_v4 import v4_resources as vr


def requirements(**gpu):
    return PlacementRequirements(vr.MODEL_ID, 0, 1,
        GpuRequirements(resident_weights_bytes=100, **gpu),
        HostRequirements(routed_experts_bytes=200, kv_offload_bytes=50, pinned_bytes=200),
        CalibrationProvenance("metadata-sha256:" + "a" * 64, "b" * 64,
                              "2026-10-06T12:00:00+08:00", "node-1", "isolated CUDA probe", "local-report.json"))


def test_gpu_ram_and_pinned_are_independent_not_added_twice():
    req = requirements(expert_cache_bytes=20, load_peak_extra_bytes=10)
    assert req.gpu.peak_bytes == 130 and req.host.peak_bytes == 250
    assert evaluate_fit(req, NodeResources(130, 250, 200))["fits"]
    assert evaluate_fit(req, NodeResources(129, 249, 199))["insufficient"] == ["vram", "ram", "pinned"]
    unknown = evaluate_fit(req, NodeResources(130, 250, None))
    assert unknown["status"] == "unknown" and unknown["unknown"] == ["pinned"]


@pytest.mark.parametrize("invalid", [-1, 1.5, True, "100", None])
def test_byte_components_reject_invalid_numbers(invalid):
    with pytest.raises(ResourceError):
        GpuRequirements(kv_hot_bytes=invalid)
    with pytest.raises(ResourceError):
        HostRequirements(routed_experts_bytes=invalid)


def test_serialized_requirements_need_explicit_components_and_measured_provenance():
    body = requirements().to_dict()
    assert PlacementRequirements.from_dict(json.loads(json.dumps(body))) == requirements()
    body["gpu"].pop("graph_bytes")
    with pytest.raises(ResourceError, match="explicitly"):
        PlacementRequirements.from_dict(body)
    body = requirements().to_dict()
    body["provenance"]["kind"] = "storage_estimate"
    with pytest.raises(ResourceError, match="measured"):
        PlacementRequirements.from_dict(body)
    body = requirements().to_dict()
    body["provenance"]["runtime_config_sha256"] = 3
    with pytest.raises(ResourceError, match="SHA-256"):
        PlacementRequirements.from_dict(body)
    body = requirements().to_dict()
    body["gpu"] = asdict(GpuRequirements())
    with pytest.raises(ResourceError, match="nonzero"):
        PlacementRequirements.from_dict(body)
    body = requirements().to_dict()
    body.pop("schema")
    with pytest.raises(ResourceError, match="schema"):
        PlacementRequirements.from_dict(body)


def test_routed_pool_pinning_is_subset_of_host_budget():
    with pytest.raises(ResourceError, match="pinned-memory"):
        HostRequirements(routed_experts_bytes=100, pinned_bytes=50)
    with pytest.raises(ResourceError, match="exceed"):
        HostRequirements(routed_experts_bytes=100, pinned_bytes=101)


def cgroup_reader(version=2, namespace_root="/"):
    if version == 2:
        files = {"/proc/self/cgroup": "0::/pool/worker\n",
                 "/proc/self/mountinfo": f"29 23 0:26 {namespace_root} /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
                 "/sys/fs/cgroup/memory.max": "max", "/sys/fs/cgroup/memory.current": "400",
                 "/sys/fs/cgroup/pool/memory.max": "1000", "/sys/fs/cgroup/pool/memory.current": "700",
                 "/sys/fs/cgroup/pool/worker/memory.max": "max", "/sys/fs/cgroup/pool/worker/memory.current": "100"}
    else:
        files = {"/proc/self/cgroup": "5:memory:/worker\n",
                 "/proc/self/mountinfo": "29 23 0:26 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes": str(2**63),
                 "/sys/fs/cgroup/memory/worker/memory.limit_in_bytes": "600",
                 "/sys/fs/cgroup/memory/worker/memory.usage_in_bytes": "100"}
    files.update({"/proc/meminfo": "MemAvailable: 10 kB\n", "/proc/self/status": "VmLck: 1 kB\n"})
    def read(path):
        if path not in files:
            raise FileNotFoundError(path)
        return files[path]
    return files, read


def test_nested_parent_cgroup_limit_binds_and_pinnable_is_not_os_memlock():
    _, read = cgroup_reader()
    assert linux_cgroup_headroom(read)["available_bytes"] == 300
    report = linux_host_resources(read, 2048)
    assert report["host_available_ram_bytes"] == 10240
    assert report["available_ram_bytes"] == 300
    assert report["os_lock_headroom_bytes"] == 1024
    assert report["pinnable_ram_bytes"] is None


def test_cgroup_v1_and_unknown_usage_and_namespace_mount_root():
    files, read = cgroup_reader(1)
    assert linux_cgroup_headroom(read)["available_bytes"] == 500
    del files["/sys/fs/cgroup/memory/worker/memory.usage_in_bytes"]
    assert linux_cgroup_headroom(read)["status"] == "unknown"
    assert linux_host_resources(read, 2048)["available_ram_bytes"] is None
    files, read = cgroup_reader()
    files["/proc/self/cgroup"] = "0::/\n"
    files["/proc/self/mountinfo"] = "29 23 0:26 /pool/worker /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
    files["/sys/fs/cgroup/memory.max"] = "600"
    files["/sys/fs/cgroup/memory.current"] = "400"
    assert linux_cgroup_headroom(read)["available_bytes"] == 200


def test_all_structural_inventory_and_measurement_dicts_are_refused_by_legacy_planner():
    for profile in (vr.structural_profile(), {"schema": "v4-storage-inventory/1"},
                    {"schema": "v4-resource-measurement/1"}, {"calibration_status": "unmeasured"}):
        with pytest.raises(ValueError, match="calibrated"):
            plan_ring([], [], profile)
    with pytest.raises(NotImplementedError, match="measured"):
        ModelRuntime("v4", LayerRange(0, 1)).placement_requirements()


def write_checkpoint(directory):
    config = {"n_layers": 43, "dim": 4096, "hc_mult": 4,
              "dspark_target_layer_ids": [40, 41, 42], "n_mtp_layers": 3}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {f"layers.{i}.attn.weight": ("BF16", [2, 2]) for i in (0, 40, 41, 42)}
    tensors.update({"layers.40.ffn.experts.0.w1.weight": ("F4", [2, 4]),
                    "layers.40.ffn.experts.0.w1.scale": ("F8_E8M0", [2, 1]),
                    "layers.40.ffn.shared_experts.w1.weight": ("F4", [2, 4]),
                    "embed.weight": ("BF16", [2, 2]), "head.weight": ("BF16", [2, 2]),
                    "norm.weight": ("BF16", [2]), "hc_head_fn": ("F32", [4, 4]),
                    "hc_head_base": ("F32", [4]), "hc_head_scale": ("F32", [1])})
    for i in range(3):
        tensors[f"mtp.{i}.ffn.experts.0.w1.weight"] = ("F4", [2, 4])
        tensors[f"mtp.{i}.markov_head.markov_w2.weight"] = ("BF16", [2, 2])
        tensors[f"mtp.{i}.confidence_head.proj.weight"] = ("BF16", [1, 2])
    write_tensors(directory / "model0-mp1.safetensors", tensors)
    return directory


def write_tensors(path, tensors):
    header, cursor = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = math_product(shape) * vr._BITS[dtype] // 8
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + size]}
        cursor += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(cursor))


def math_product(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def test_header_sizes_scales_shared_experts_and_fp32_promotions(tmp_path):
    inventory = vr.inspect_checkpoint(write_checkpoint(tmp_path))
    tensors = inventory["tensors"]
    assert tensors["head.weight"]["storage_bytes"] == 8
    assert tensors["head.weight"]["runtime_parameter_floor_bytes"] == 16
    assert tensors["mtp.0.markov_head.markov_w2.weight"]["runtime_parameter_floor_bytes"] == 16
    assert tensors["mtp.0.confidence_head.proj.weight"]["runtime_parameter_floor_bytes"] == 8
    assert tensors["layers.40.ffn.shared_experts.w1.weight"]["kind"] == "resident"
    result = vr.stage_storage_inventory(inventory, 40, 43, tail=True, dspark=True)
    assert result["routed_expert_storage_bytes"] == 6 + 3 * 4
    assert result["runtime_peak_bytes"] is None
    assert inventory["payload_integrity_verified"] is False
    assert result["aliases"]["mtp.*.head.weight"] == "head.weight"
    with pytest.raises(ResourceError, match="identity"):
        vr.inspect_checkpoint(tmp_path, expected_checkpoint_id="another checkpoint")


def test_dspark_tail_and_checkpoint_completeness_are_checked(tmp_path):
    inventory = vr.inspect_checkpoint(write_checkpoint(tmp_path))
    with pytest.raises(ResourceError, match="all target"):
        vr.stage_storage_inventory(inventory, 41, 43, tail=True, dspark=True)
    with pytest.raises(ResourceError, match="every assigned"):
        vr.stage_storage_inventory(inventory, 0, 2, head=True)
    del inventory["tensors"]["mtp.2.ffn.experts.0.w1.weight"]
    del inventory["tensors"]["mtp.2.markov_head.markov_w2.weight"]
    del inventory["tensors"]["mtp.2.confidence_head.proj.weight"]
    with pytest.raises(ResourceError, match="MTP"):
        vr.stage_storage_inventory(inventory, 40, 43, tail=True, dspark=True)


def test_missing_declared_structure_is_not_filled_from_reference_defaults(tmp_path):
    write_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    del config["n_layers"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ResourceError, match="explicitly declare"):
        vr.inspect_checkpoint(tmp_path)


@pytest.mark.parametrize("failure", ["gpu_pool", "host", "pinned", "unknown_allocator", "missing_draft"])
def test_stage_calibration_checks_live_allocator_host_and_draft_evidence(tmp_path, monkeypatch, failure):
    inventory = vr.inspect_checkpoint(write_checkpoint(tmp_path))
    stage = SimpleNamespace(args=SimpleNamespace(n_layers=43), lo=40, hi=43, head=False,
                            tail=True, _dspark=True, dtype="bf16", device="cuda:0",
                            _resource_checkpoint_dir=str(tmp_path))
    req = PlacementRequirements(vr.MODEL_ID, 40, 43, GpuRequirements(resident_weights_bytes=100),
        HostRequirements(prefill_bytes=100, pinned_bytes=80),
        CalibrationProvenance(inventory["checkpoint_id"], vr.runtime_config_identity(stage),
                              "2026-10-06T00:00:00Z", "test-node", "validator fixture", "fixture"))
    report = {"module_storage": {"gpu_bytes": 50, "host_bytes": 100, "host_pinned_bytes": 80},
              "allocator": {"allocated_bytes": 80, "reserved_bytes": 100},
              "draft_measurement_status": "measured"}
    if failure == "gpu_pool":
        report["allocator"]["reserved_bytes"] = 101
    elif failure == "host":
        report["module_storage"]["host_bytes"] = 101
    elif failure == "pinned":
        report["module_storage"]["host_pinned_bytes"] = 81
    elif failure == "unknown_allocator":
        report["allocator"]["allocated_bytes"] = None
    else:
        report["draft_measurement_status"] = "missing"
    monkeypatch.setattr(vr, "measure_stage_resources", lambda *args, **kwargs: report)
    with pytest.raises(ResourceError):
        vr.placement_requirements_for_stage(stage, req.to_dict())


@pytest.mark.parametrize("case", ["truncated", "overlap", "bad_shape", "unknown_dtype", "duplicate", "bad_fp4_alignment"])
def test_header_corruption_is_rejected(tmp_path, case):
    path = tmp_path / "model0-mp1.safetensors"
    if case == "truncated":
        path.write_bytes(struct.pack("<Q", 50) + b"{}")
    else:
        entry = {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}
        header = {"a": entry}
        payload = bytes(2)
        if case == "overlap":
            header["b"] = dict(entry)
        elif case == "bad_shape":
            entry["shape"] = [True]
        elif case == "unknown_dtype":
            entry["dtype"] = "guess"
        elif case == "bad_fp4_alignment":
            entry.update(dtype="F4", shape=[3])
        blob = json.dumps(header).encode()
        if case == "duplicate":
            blob = b'{"a":' + json.dumps(entry).encode() + b',"a":' + json.dumps(entry).encode() + b'}'
        path.write_bytes(struct.pack("<Q", len(blob)) + blob + payload)
    with pytest.raises(ResourceError):
        vr.read_safetensors_header(path)


def test_inventory_cli_is_stdlib_only_outside_repo_cwd(tmp_path):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    write_checkpoint(directory)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    script = Path(vr.__file__).resolve()
    result = subprocess.run([sys.executable, "-S", str(script), "inventory", "--checkpoint", str(directory),
                             "--lo", "40", "--hi", "43", "--tail", "--dspark"],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["routed_expert_storage_bytes"] == 18


def test_cuda_probe_unavailable_has_unknown_bandwidth():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this test pins the unavailable branch without allocating on a live GPU")
    from shard.probe import measure_transfer_resources
    report = measure_transfer_resources()
    assert report["status"] == "unavailable"
    assert report["h2d_bytes_per_second"] is None
    assert report["tested_pinned_bytes"] is None


def test_host_resource_cli_needs_no_torch_or_model(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-S", "-m", "shard.probe", "--resources"],
                            cwd=repo, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "shard-node-resource-probe/1"
    assert report["transfer"] is None
    assert report["host"]["pinnable_ram_bytes"] is None
    assert "available_disk_bytes" in report["host"]


def test_live_storage_inventory_deduplicates_aliases_and_reports_unknown_gpu_peaks():
    torch = pytest.importorskip("torch")
    layer = torch.nn.Linear(2, 2, bias=False)
    layer.register_buffer("kv_cache", torch.zeros(4))
    stage = SimpleNamespace(args=SimpleNamespace(n_layers=43), lo=40, hi=43, head=False, tail=True,
                            _dspark=True, dtype=torch.float32, device="cpu",
                            layers=torch.nn.ModuleList([layer]), embed_tokens=torch.nn.Embedding(2, 2),
                            norm=None, lm_head=None)
    stage.lm_head = stage.embed_tokens  # Synthetic alias: counted once.
    stage.rollback = [layer.kv_cache, torch.zeros(2)]
    report = vr.measure_stage_resources(stage, checkpoint_id="fixture")
    assert report["module_storage"]["host_bytes"] == 16 + 16 + 16 + 8
    assert report["allocator"]["peak_allocated_bytes"] is None
    assert report["unattributed_components"]["workspace_bytes"] is None
    assert report["placement_requirements"] is None
    with pytest.raises(NotImplementedError, match="measured"):
        vr.placement_requirements_for_stage(stage)
    with pytest.raises(ResourceError, match="CPU Stage"):
        vr.placement_requirements_for_stage(stage, requirements().to_dict())


def test_real_safetensors_fp4_logical_shape(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    packed = torch.empty(2, 2, dtype=torch.float4_e2m1fn_x2)
    path = tmp_path / "actual.safetensors"
    save_file({"packed": packed}, path)
    tensor = vr.read_safetensors_header(path)["tensors"]["packed"]
    assert tensor["dtype"] == "F4" and tensor["shape"] == [2, 4]
    assert tensor["storage_bytes"] == packed.untyped_storage().nbytes() == 4


def test_storage_requirements_and_evaluate_fit_with_disk():
    from shard.resources import StorageRequirements
    storage = StorageRequirements(
        model_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        layer_start=0,
        layer_end=11,
        storage_bytes=38 * 1024 * 1024 * 1024,
        files=("model-00001-mp1.safetensors", "model-00002-mp1.safetensors"),
        manifest_sha256="c" * 64,
    )
    req = requirements()
    # Fits when disk is sufficient
    fit_res = evaluate_fit(req, NodeResources(130, 250, 200, 50 * 1024 * 1024 * 1024), storage=storage)
    assert fit_res["fits"] is True
    assert fit_res["required_disk_bytes"] == 38 * 1024 * 1024 * 1024

    # Insufficient disk
    tight_res = evaluate_fit(req, NodeResources(130, 250, 200, 10 * 1024 * 1024 * 1024), storage=storage)
    assert tight_res["fits"] is False
    assert "disk" in tight_res["insufficient"]

    # Unknown disk
    unknown_res = evaluate_fit(req, NodeResources(130, 250, 200, None), storage=storage)
    assert unknown_res["fits"] is False
    assert unknown_res["status"] == "unknown"
    assert "disk" in unknown_res["unknown"]
    assert "measure available disk space" in unknown_res["action"]
