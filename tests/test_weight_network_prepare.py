"""Offers -> bounded planning -> signed preparation RPC -> verified CPU startup.

GPU/latency capacities here are synthetic control-plane fixtures. Actual stage
files, tensor hashes, quota accounting, signatures and child processes are real.
This is not a DeepSeek numerical or GPU throughput test.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import pytest
import torch
from safetensors.torch import save_file
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from shard import weight_artifacts as W
from shard.control_plane import NodeLeaseAgent, LeaseRPCClient, _identity
from shard.deployment import config_digest
from shard.leased_runtime import configured_stage_factory
from shard.leases import LeaseLedger
from shard.manifest import save_key
from shard.network_service import OpenNetworkService
from shard.offers import ModelCohort, OfferRegistry, sign_offer
from shard.resources import (GpuRequirements, HostRequirements, PlacementRequirements,
    CalibrationProvenance, StorageFile, StorageRequirements, PreparationOption, STORAGE_PREPARE_SCHEMA)
from shard.receipt import verify_coverage
from shard.weight_prepare import configured_prepare_factory
from test_control_plane import serving
from test_ring_pool import Backend


def test_small_disk_node_is_planned_then_range_repacked_before_real_signed_startup(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("V4_"): monkeypatch.delenv(name)
    source = tmp_path / "source"; source.mkdir()
    (source / "config.json").write_text(json.dumps({"n_layers": 3, "n_mtp_layers": 0}))
    values = {f"layers.{i}.weight": torch.arange(8192, dtype=torch.float32) + i for i in range(3)}
    values.update({name: torch.ones(4) for name in ("embed.weight", "head.weight", "norm.weight",
        "hc_head_fn", "hc_head_base", "hc_head_scale")})
    save_file(values, str(source / "model0-mp1.safetensors"))
    catalog, pack = W.catalogue_directory(source, "fixture/native")
    W.write_metadata(source, catalog, pack)
    model = ModelCohort(catalog["model_id"], catalog["manifest_sha256"], catalog["checkpoint_id"],
        catalog["config_sha256"], "fp32-fixture", catalog["runtime_abi"], "cpu-fixture-wire/1", "fixture/1", 3)
    quota = 60000
    assert (source / "model0-mp1.safetensors").stat().st_size > quota
    registry = OfferRegistry(tmp_path / "registry.sqlite")
    control_key = Ed25519PrivateKey.generate()
    sidecar = tmp_path / "controller.key"
    sidecar.write_bytes(b"\x08\x01\x12\x40" + control_key.private_bytes_raw() + control_key.public_key().public_bytes_raw())
    nonce = "a1" * 32
    records, agents, markers = [], {}, []
    service = None
    with ExitStack() as stack:
        try:
            for index in range(3):
                node_key = Ed25519PrivateKey.generate()
                node_id, gpu = _identity(node_key)[0] + f"/GPU-{index}", f"GPU-{index}"
                folder = tmp_path / f"node{index}"; folder.mkdir()
                cache = folder / "cache"; cache.mkdir()
                fs, domain = f"fs-{index}", f"host-{index}"
                selected = W.select_stage_artifacts(catalog, pack, index, index + 1,
                    head=index == 0, tail=index == 2)
                bound = W.repack_storage_bound(catalog, pack, selected, max_file_bytes=4096, chunk_bytes=1024)
                assert bound["disk_peak_bytes"] < quota
                cfg = {"lo": index, "hi": index + 1, "head": index == 0, "tail": index == 2,
                       "stage": index, "nstages": 3, "max_ctx": 128, "environment": {}}
                req = PlacementRequirements(model.model_id, index, index + 1,
                    GpuRequirements(resident_weights_bytes=100), HostRequirements(),
                    CalibrationProvenance(model.checkpoint_id, config_digest(cfg),
                        datetime.now(timezone.utc).isoformat(), node_id, "CPU control fixture", "no GPU measurement"))
                source_bytes = sum(row["size"] for row in selected["files"])
                fetch = PreparationOption("fetch", fs, source_bytes + 20000, 8 << 20)
                repack = PreparationOption("range_repack", fs, bound["disk_peak_bytes"], 8 << 20)
                storage = StorageRequirements(model.model_id, index, index + 1, source_bytes,
                    files=tuple(row["path"] for row in selected["files"]),
                    manifest_sha256=model.manifest_sha256, schema=STORAGE_PREPARE_SCHEMA,
                    file_records=tuple(StorageFile(row["path"], row["size"], row["sha256"]) for row in selected["files"]),
                    checkpoint_id=model.checkpoint_id, artifact_id=selected["artifact_id"], filesystem_id=fs,
                    head=index == 0, tail=index == 2, preparation_options=(fetch, repack))
                now = time.time()
                body = sign_offer({"gpu_uuid": gpu, "memory_domain_id": domain, "host_id": domain,
                    "endpoints": [f"/ip4/127.0.0.1/tcp/{21000 + index}"], "region": "fixture",
                    "resources": {"available_vram_bytes": 1 << 20, "available_ram_bytes": 64 << 20,
                        "pinnable_ram_bytes": 0, "available_disk_bytes": quota, "measured_at": now,
                        "filesystems": {fs: {"available_disk_bytes": quota, "measured_at": now}}},
                    "models": [{"cohort": model.to_dict(), "profile": {"layer_ms": 1, "cap_layers": 1},
                        "measured_at": now, "calibrations": [{"requirements": req.to_dict(),
                            "runtime_config": cfg, "storage": storage.to_dict()}]}],
                    "issued_at": now, "ttl_s": 120, "sequence": 1}, node_key)
                registry.announce(body)
                ledger_path = folder / "leases.sqlite"
                ledger = LeaseLedger(ledger_path, node_id=node_id, authorize=lambda p, a, b: p)
                ledger.register_capacity(domain, available_ram_bytes=64 << 20, pinnable_ram_bytes=0,
                                         gpu_capacity_bytes={gpu: 1 << 20})
                ledger.register_filesystem(fs, folder, available_disk_bytes=quota)
                rows = [{"cohort_id": model.cohort_id, "catalog": str(source / W.GLOBAL_FILE),
                    "pack": str(source / W.PACK_FILE), "cache_root": str(cache), "jobs_db": str(folder / "jobs.sqlite"),
                    "provider": {"kind": "local", "root": str(source)},
                    "resources": {"filesystem_id": fs}, "repack_resources": {"filesystem_id": fs},
                    "repack": {"max_file_bytes": 4096, "chunk_bytes": 1024}}]
                key_path, marker = folder / "key.json", folder / "started.json"
                save_key(node_key, str(key_path)); markers.append(marker)
                script = ("import sys,time,json,base64; from pathlib import Path; "
                    "from shard.weight_artifacts import verify_stage_artifacts; "
                    "from shard.receipt import ReceiptSigner; "
                    "from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey; "
                    "v=verify_stage_artifacts(sys.argv[1]); "
                    "k=Ed25519PrivateKey.from_private_bytes(base64.b64decode(Path(sys.argv[2]).read_text())); "
                    "s=ReceiptSigner(k,'ring','cpu-warmup',v['lo'],v['hi'],sys.argv[4]); "
                    "s.observe(b'cpu-fixture-activation',b'cpu-fixture-activation'); "
                    "Path(sys.argv[3]).write_text(json.dumps({'directory':v['directory'],'receipt':s.finalize()})); "
                    "time.sleep(30)")
                template = {"cohort_id": model.cohort_id, "lo": index, "hi": index + 1,
                    "head": index == 0, "tail": index == 2, "requirements": req.to_dict(), "runtime_config": cfg,
                    "storage": storage.to_dict(),
                    "argv": [sys.executable, "-c", script.replace("{", "{{").replace("}", "}}"),
                        "{model_dir}", str(key_path), str(marker), nonce]}
                agent = NodeLeaseAgent(ledger, node_key, cohorts=[model.cohort_id],
                    stage_factory=configured_stage_factory(ledger_path, [template],
                        weight_preparer=configured_prepare_factory(ledger_path, rows)))
                address = stack.enter_context(serving(lambda path, body, agent=agent: agent.dispatch(body)))
                agents[node_id] = LeaseRPCClient(address, node_id, control_key)
                records.append((body, storage, ledger))
            nodes = registry.snapshot(model.cohort_id)
            assert len(nodes) == 3
            assert all([span["preparation_mode"] for span in node["allowed_spans"]] == ["range_repack"] for node in nodes)
            mesh = {"schema": "shard-link-measurements/1", "edges": [
                {"src": a["id"], "dst": b["id"], "rtt_ms": 1, "measured_at": time.time(), "ttl_s": 120}
                for a in nodes for b in nodes if a is not b]}
            profile = {"n_layers": 3, "layer_vram_mb": .0001, "kv_mb_per_layer": 0, "reserve_mb": 0,
                "head_reserve_mb": 0, "tail_reserve_mb": 0, "cap_layers": 1,
                "layer_ms_base": 1, "head_layer_ms_mult": 1}
            config = {"schema": "shard-open-network/1", "controller_sidecar_key": str(sidecar),
                "registry_db": str(tmp_path / "registry.sqlite"), "formations": [{"ring_id": "ring",
                    "dir": str(source), "cohort": model.to_dict(), "profile": profile, "measurements": mesh,
                    "head": "127.0.0.1:21000", "tail": "127.0.0.1:21002", "warmup_timeout_s": 20,
                    "objective": "serial", "weight_prepare_timeout_s": 20}]}
            config_path = tmp_path / "network.json"; config_path.write_text(json.dumps(config))
            def factory(directory, plan, cohort, row, contracts):
                backend = Backend(); backend.model_id = cohort.model_id; backend.layers = cohort.n_layers
                def warmup(timeout_s=20):
                    deadline = time.monotonic() + timeout_s
                    while not all(marker.exists() for marker in markers) and time.monotonic() < deadline: time.sleep(.01)
                    values = [json.loads(marker.read_text()) for marker in markers]
                    expected = {body["public_key"]: (i, i + 1) for i, (body, _, _) in enumerate(records)}
                    verify_coverage([value["receipt"] for value in values], 3, expected, nonce, check_chain=True)
                    backend.healthy = True
                    return {"proof_verified": True, "committed_tokens": 1}
                backend.warmup = warmup
                return backend
            service = OpenNetworkService(config_path, factory, registry=registry,
                agent_factory=lambda offer: agents[offer["node_id"]])
            result = service.form_all()[0]
            assert result["ready"] and len(result["plan"]["stages"]) == 3
            assert all(stage["preparation_mode"] == "range_repack" for stage in result["plan"]["stages"])
            ring = service.pool.rings()[0]
            assert ring.state.value == "READY"
            assert len(ring.backend.preparations) == 3
            for state in ring.backend.preparations.values():
                assert state["ready"] and "converting" in state["history"]
                assert state["preparation_mode"] == state["planned_preparation_mode"] == "range_repack"
                verified = W.verify_stage_artifacts(state["directory"])
                assert sum(row["size"] for row in verified["files"]) < quota
                assert state["artifact_id"] != state["source_artifact_id"]
        finally:
            if service is not None:
                service.close()
                deadline = time.monotonic() + 5
                while service.controller._formations and time.monotonic() < deadline: time.sleep(.01)
            registry.close()
