"""Actual byte-hashed stage files, durable job progress and preparation boundaries."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch
from safetensors.torch import save_file, load_file

from shard.fetch import LocalDirProvider, MirrorProvider
from shard.leases import LeaseConflict
from shard import weight_artifacts as W
from shard.weight_prepare import WeightPrepareManager, RangeReader, PreparationError, RoutedProvider

COHORT = "1" * 64


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"; root.mkdir()
    (root / "config.json").write_text(json.dumps({"n_layers": 3, "n_mtp_layers": 3,
        "dspark_target_layer_ids": [2]}))
    (root / "tokenizer.json").write_text('{"fixture":true}')
    values = {f"layers.{i}.weight": torch.arange(16, dtype=torch.float32) + i for i in range(3)}
    values.update({name: torch.ones(4) for name in ("embed.weight", "head.weight", "norm.weight",
        "hc_head_fn", "hc_head_base", "hc_head_scale")})
    values.update({f"mtp.{i}.weight": torch.arange(4, dtype=torch.float32) for i in range(3)})
    # Files already partitioned by layer allow a genuine subset fetch test.
    for index in range(3):
        save_file({name: tensor for name, tensor in values.items() if name.startswith(f"layers.{index}.")},
                  str(root / f"model{index}-mp1.safetensors"))
    save_file({name: tensor for name, tensor in values.items() if not name.startswith("layers.")},
              str(root / "model3-mp1.safetensors"))
    catalog, pack = W.catalogue_directory(root, "fixture/native")
    return root, catalog, pack, values


class Guard:
    node_id = "node/GPU"
    memory_domain_id = "host"
    live = True
    def assert_live(self):
        if not self.live: raise RuntimeError("expired")
        return self


class Reservation:
    def __init__(self): self.active = 0; self.finished = None; self.released = False
    def assert_live(self): return self
    def begin_work(self, name): self.active += 1; return name
    def end_work(self, name): self.active -= 1
    def renew(self, **kwargs): return self
    def release(self): assert self.active == 0; self.released = True
    def finish(self, witness):
        assert self.active == 0
        assert W.verified_stage_artifact_descriptor(witness)
        self.finished = witness


def assignment(catalog, pack, **options):
    roles = dict(lo=0, hi=1, head=True, tail=False, dspark=False); roles.update(options)
    selected = W.select_stage_artifacts(catalog, pack, **roles)
    return {**roles, "cohort_id": COHORT, "node_id": Guard.node_id,
            "weight_artifacts": {name: selected[name] for name in ("artifact_id", "checkpoint_id", "manifest_sha256")}}


def manager(tmp_path, source, **options):
    root, catalog, pack, _ = source
    reservations = []
    def reserve(*_args):
        result = Reservation(); reservations.append(result); return result
    return WeightPrepareManager(tmp_path / "cache", tmp_path / "jobs.db", catalog=catalog,
        pack=pack, cohort_id=COHORT, provider=LocalDirProvider(str(root)), reserve=reserve,
        resources={"filesystem_id": "fs", "disk_peak_bytes": 100000, "ram_bytes": 16 << 20, "pinned_bytes": 0},
        **options), reservations


def test_verified_subset_publishes_and_survives_restart(source, tmp_path):
    first, reservations = manager(tmp_path, source)
    request = assignment(source[1], source[2])
    result = first.prepare(request, Guard())
    assert result["state"] == "ready" and result["payload_integrity_verified"]
    assert result["history"][-5:] == ["reserved", "fetching", "verifying", "published", "ready"]
    got = W.verify_stage_artifacts(result["directory"], lo=0, hi=1, head=True)
    assert reservations[0].finished is not None and reservations[0].active == 0
    assert (Path(result["directory"]) / "model0-mp1.safetensors").exists()
    assert not (Path(result["directory"]) / "model1-mp1.safetensors").exists()
    # A new manager uses persisted status; repeated loading re-verifies bytes.
    second, _ = manager(tmp_path, source)
    assert second.status(result["job_id"])["ready"]
    assert second.prepare(request, Guard())["directory"] == result["directory"]


def test_wrong_pins_fail_before_reservation(source, tmp_path):
    prep, reservations = manager(tmp_path, source)
    request = assignment(source[1], source[2]); request["weight_artifacts"]["checkpoint_id"] = "forged"
    with pytest.raises(PreparationError, match="identity"): prep.prepare(request, Guard())
    assert reservations == [] and not list((tmp_path / "cache").glob(".partial-*"))


def test_bad_download_is_failed_and_does_not_publish(source, tmp_path):
    prep, reservations = manager(tmp_path, source)
    class Bad:
        def fetch(self, row, dest): Path(dest).write_bytes(b"bad")
    prep.provider = Bad()
    request = assignment(source[1], source[2])
    with pytest.raises(PreparationError, match="SHA256"): prep.prepare(request, Guard())
    job = prep._binding(request)[0]
    assert prep.status(job)["state"] == "failed"
    assert not (prep.root / job).exists()
    assert reservations[0].released and not reservations[0].active


def test_failed_fetch_resumes_verified_files_without_refetch(source, tmp_path):
    prep, _ = manager(tmp_path, source)
    calls = []
    class Drop(LocalDirProvider):
        failed = False
        def fetch(self, row, dest):
            calls.append(row["path"])
            if row["path"].endswith(".safetensors") and not self.failed:
                self.failed = True; raise OSError("disconnected")
            return super().fetch(row, dest)
    prep.provider = Drop(str(source[0]))
    request = assignment(source[1], source[2])
    with pytest.raises(OSError): prep.prepare(request, Guard())
    assert prep.prepare(request, Guard())["ready"]
    assert calls.count("config.json") == 1


def test_corrupt_published_files_are_never_replaced(source, tmp_path):
    prep, _ = manager(tmp_path, source)
    request = assignment(source[1], source[2]); ready = prep.prepare(request, Guard())
    path = Path(ready["directory"]) / "model0-mp1.safetensors"
    corrupted = bytearray(path.read_bytes()); corrupted[-1] ^= 1; path.write_bytes(corrupted)
    with pytest.raises(W.ArtifactError): prep.prepare(request, Guard())
    assert path.read_bytes() == corrupted
    assert prep.status(ready["job_id"])["state"] == "failed"


def test_capacity_failure_uses_bounded_raw_repack_same_checkpoint(source, tmp_path):
    calls, reservations = [], []
    def reserve(stage, guard, resources, ttl):
        calls.append(resources["disk_peak_bytes"])
        if len(calls) == 1: raise LeaseConflict("insufficient disk")
        result = Reservation(); reservations.append(result); return result
    def convert(dest, catalog, pack, selected, provider, check):
        return W.repack_stage(catalog, pack, selected, dest, RangeReader(provider, chunk_bytes=13),
                              chunk_bytes=13, max_file_bytes=40, cancel_check=check)
    prep, _ = manager(tmp_path, source, convert=convert, repack={"max_file_bytes": 40, "chunk_bytes": 13},
        repack_resources={"filesystem_id": "fs", "disk_peak_bytes": 10000, "ram_bytes": 16 << 20, "pinned_bytes": 0})
    prep.reserve = reserve
    ready = prep.prepare(assignment(source[1], source[2]), Guard())
    assert calls == [100000, 10000]
    assert ready["artifact_id"] != ready["source_artifact_id"]
    assert ready["checkpoint_id"] == source[1]["checkpoint_id"]
    assert "converting" in ready["history"]
    weights = {}
    for path in Path(ready["directory"]).glob("*.safetensors"): weights.update(load_file(str(path)))
    assert set(weights) == {"embed.weight", "layers.0.weight"}
    for name, actual in weights.items(): assert torch.equal(actual, source[3][name])


def test_unknown_capacity_does_not_retry_a_converter(source, tmp_path):
    prep, _ = manager(tmp_path, source, convert=lambda *_: pytest.fail("must not convert"),
        repack_resources={"filesystem_id": "fs", "disk_peak_bytes": 50, "ram_bytes": 1})
    prep.reserve = lambda *_: (_ for _ in ()).throw(ValueError("capacity unknown"))
    with pytest.raises(ValueError, match="unknown"): prep.prepare(assignment(source[1], source[2]), Guard())


def test_async_submit_is_idempotent_for_running_job(source, tmp_path):
    prep, _ = manager(tmp_path, source)
    started, proceed = threading.Event(), threading.Event()
    original = prep.provider.fetch
    def slow(row, dest): started.set(); assert proceed.wait(3); return original(row, dest)
    prep.provider.fetch = slow
    request = assignment(source[1], source[2])
    first = prep.submit(request, Guard()); assert started.wait(1)
    second = prep.submit(request, Guard()); assert first["job_id"] == second["job_id"]
    worker = prep._threads[first["job_id"]]
    proceed.set()
    worker.join(3)
    assert prep.status(first["job_id"])["ready"]


@pytest.mark.parametrize("offset,length", [(-1,1),(0,-1),(0,8),(7,3),(False,1)])
def test_range_bounds_are_not_provider_instructions(source, offset, length):
    reader = RangeReader(LocalDirProvider(str(source[0])), chunk_bytes=4)
    with pytest.raises(PreparationError): reader({"path": "config.json", "size": 8}, offset, length)


def test_http_ranges_require_exact_206_never_accept_whole_big_file():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Length", "100000"); self.end_headers()
        def log_message(self, *_): pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        reader = RangeReader(MirrorProvider(f"http://127.0.0.1:{server.server_port}"), chunk_bytes=4)
        with pytest.raises(PreparationError, match="exact requested"):
            reader({"path": "large.safetensors", "size": 100000}, 7, 4)
    finally:
        server.shutdown(); server.server_close(); thread.join(2)


def test_real_shared_ledger_finish_and_restart_adoption_do_not_double_charge(source, tmp_path):
    from shard.leases import LeaseLedger, LeaseRequest, LeaseResources
    from shard.weight_prepare import configured_prepare_factory
    root, catalog, pack, _ = source
    W.write_metadata(root, catalog, pack)
    ledger_path = tmp_path / "host.sqlite"
    ledger = LeaseLedger(ledger_path, node_id=Guard.node_id, authorize=lambda p, a, b: p)
    ledger.register_capacity("host", available_ram_bytes=64 << 20, pinnable_ram_bytes=8 << 20,
                             gpu_capacity_bytes={"GPU": 1000})
    ledger.register_filesystem("fs", tmp_path, available_disk_bytes=1000000)
    req = LeaseRequest("ring", COHORT, Guard.node_id, "GPU", "host", LeaseResources(100, 1 << 20), 120)
    lease = ledger.prepare(req, principal="controller", idempotency_key="stage")
    ledger.commit(lease.lease_id, lease.fencing_token, principal="controller")
    guard = ledger.guard(lease.lease_id, lease.fencing_token, principal="controller", ring_id="ring", model_cohort_sha256=COHORT)
    rows = [{"cohort_id": COHORT, "catalog": str(root / W.GLOBAL_FILE), "pack": str(root / W.PACK_FILE),
        "cache_root": str(tmp_path / "cache"), "jobs_db": str(tmp_path / "jobs.sqlite"),
        "provider": {"kind": "local", "root": str(root)}, "resources": {"filesystem_id": "fs"}}]
    hook = configured_prepare_factory(ledger_path, rows)
    first = hook(assignment(catalog, pack), guard)
    assert first["ready"] and first["payload_integrity_verified"]
    with ledger._transaction() as (conn, _):
        charged = [tuple(row) for row in conn.execute("SELECT directory,artifact,bytes FROM artifact_cache")]
        assert len(charged) == 1 and charged[0][2] > 0
    second = configured_prepare_factory(ledger_path, rows)(assignment(catalog, pack), guard)
    assert second["directory"] == first["directory"]
    with ledger._transaction() as (conn, _):
        assert [tuple(row) for row in conn.execute("SELECT directory,artifact,bytes FROM artifact_cache")] == charged
        assert conn.execute("SELECT COUNT(*) FROM artifact_preparations WHERE state IN ('prepared','draining')").fetchone()[0] == 0


def test_underdeclared_disk_and_ram_fail_before_any_file_io(source, tmp_path):
    prep, reservations = manager(tmp_path, source)
    prep.resources["disk_peak_bytes"] = 1
    prep.provider.fetch = lambda *_: pytest.fail("must reject before fetching")
    with pytest.raises(LeaseConflict, match="pre-write"): prep.prepare(assignment(source[1], source[2]), Guard())
    assert not reservations
    prep.resources["disk_peak_bytes"] = 100000
    prep.resources["ram_bytes"] = 1
    with pytest.raises(LeaseConflict, match="pre-write"): prep.prepare(assignment(source[1], source[2]), Guard())
    assert not reservations


def test_node_signed_rpc_prepares_real_files_before_local_process_launch(source, tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import os
    import sys
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from shard.control_plane import NodeLeaseAgent, LeaseRPCClient, _identity
    from shard.leases import LeaseLedger, LeaseRequest, LeaseResources
    from shard.leased_runtime import configured_stage_factory
    from shard.weight_prepare import configured_prepare_factory
    from shard.resources import PlacementRequirements, GpuRequirements, HostRequirements, CalibrationProvenance
    from shard.deployment import config_digest
    from test_control_plane import serving

    for name in list(os.environ):
        if name.startswith("V4_"): monkeypatch.delenv(name)
    node_key = Ed25519PrivateKey.generate()
    node_id = _identity(node_key)[0] + "/GPU"
    ledger_path = tmp_path / "rpc-host.sqlite"
    ledger = LeaseLedger(ledger_path, node_id=node_id, authorize=lambda p, a, b: p)
    ledger.register_capacity("host", available_ram_bytes=64 << 20, pinnable_ram_bytes=8 << 20,
                             gpu_capacity_bytes={"GPU": 1000})
    ledger.register_filesystem("fs", tmp_path, available_disk_bytes=1000000)
    root, catalog, pack, _ = source
    W.write_metadata(root, catalog, pack)
    rows = [{"cohort_id": COHORT, "catalog": str(root / W.GLOBAL_FILE), "pack": str(root / W.PACK_FILE),
        "cache_root": str(tmp_path / "cache"), "jobs_db": str(tmp_path / "jobs.sqlite"),
        "provider": {"kind": "local", "root": str(root)}, "resources": {"filesystem_id": "fs"}}]
    cfg = {"lo": 0, "hi": 1, "head": True, "tail": False, "environment": {}}
    requirements = PlacementRequirements("fixture/native", 0, 1,
        GpuRequirements(resident_weights_bytes=1), HostRequirements(),
        CalibrationProvenance(catalog["checkpoint_id"], config_digest(cfg),
            datetime.now(timezone.utc).isoformat(), node_id, "cpu-fixture", "cpu-test"))
    marker = tmp_path / "started.json"
    # A real CPU process checks real hashes before announcing its launch marker.
    script = ("import sys,time,json; from pathlib import Path; "
        "from shard.weight_artifacts import verify_stage_artifacts; "
        "v=verify_stage_artifacts(sys.argv[1],lo=0,hi=1,head=True,tail=False); "
        "Path(sys.argv[2]).write_text(json.dumps({'directory':v['directory'],'verified':v['payload_integrity_verified']})); "
        "time.sleep(30)")
    template = {"cohort_id": COHORT, "lo": 0, "hi": 1, "head": True, "tail": False,
        "runtime_config": cfg, "requirements": requirements.to_dict(),
        "argv": [sys.executable, "-c", script.replace("{", "{{").replace("}", "}}"), "{model_dir}", str(marker)]}
    factory = configured_stage_factory(ledger_path, [template],
        weight_preparer=configured_prepare_factory(ledger_path, rows))
    agent = NodeLeaseAgent(ledger, node_key, cohorts=[COHORT], stage_factory=factory)
    with serving(lambda path, body: agent.dispatch(body)) as address:
        client = LeaseRPCClient(address, node_id, Ed25519PrivateKey.generate())
        lease = client.prepare(LeaseRequest("ring", COHORT, node_id, "GPU", "host", LeaseResources(100, 1 << 20), 120),
                               idempotency_key="node-stage")
        lease = client.operation("commit", lease)
        request = {**assignment(catalog, pack), "node_id": node_id, "gpu_uuid": "GPU", "ring_id": "ring",
                   "stage": 0, "nstages": 2, "next": "127.0.0.1:12345"}
        state = client.operation("prepare_stage", lease, assignment=request)
        deadline = time.monotonic() + 5
        while state["state"] not in ("ready", "failed") and time.monotonic() < deadline:
            time.sleep(.01)
            state = client.operation("prepare_status", lease, assignment=request, job_id=state["job_id"])
        assert state["state"] == "ready" and not marker.exists()
        try:
            started = client.operation("start_stage", lease, assignment=request)
            assert started["pid"] > 0
            while not marker.exists() and time.monotonic() < deadline: time.sleep(.01)
            observed = json.loads(marker.read_text())
            assert observed["verified"] and observed["directory"] == state["directory"]
            assert client.operation("stage_status", lease)["resident_work_held"]
        finally:
            client.operation("stop_stage", lease)
        assert not client.operation("stage_status", lease)["resident_work_held"]


def test_per_file_routed_provider_uses_partial_local_sources_for_new_stage(source, tmp_path):
    import shutil
    root, catalog, pack, _ = source
    assets, head, other = (tmp_path / name for name in ("assets", "head-source", "other-source"))
    for folder in (assets, head, other): folder.mkdir()
    for name in catalog["assets"]: shutil.copyfile(root / name, assets / name)
    for name in ("model0-mp1.safetensors", "model3-mp1.safetensors"):
        shutil.copyfile(root / name, head / name)
    for name in ("model1-mp1.safetensors", "model2-mp1.safetensors"):
        shutil.copyfile(root / name, other / name)
    provider = RoutedProvider({name: {"kind": "local", "root": str(folder)} for name, folder in (
        ("model0-mp1.safetensors", head), ("model3-mp1.safetensors", head),
        ("model1-mp1.safetensors", other), ("model2-mp1.safetensors", other))},
        default={"kind": "local", "root": str(assets)})
    prep, _ = manager(tmp_path, source)
    prep.provider = provider
    ready = prep.prepare(assignment(catalog, pack, lo=1, hi=2, head=False), Guard())
    assert ready["ready"] and (Path(ready["directory"]) / "model1-mp1.safetensors").exists()
    file = next(row for row in pack["files"] if row["path"] == "model1-mp1.safetensors")
    assert RangeReader(provider, chunk_bytes=4)(file, file["size"] - 4, 4) == (other / file["path"]).read_bytes()[-4:]
    with pytest.raises(PreparationError, match="no locally configured"):
        RoutedProvider({}).for_file(file)


def test_real_persisted_active_worker_is_recovered_only_with_its_exact_os_job_lock(source, tmp_path):
    from shard.leases import LeaseLedger, LeaseRequest, LeaseResources, PrepareRequest, PrepareResources
    from shard.weight_prepare import configured_prepare_factory
    root, catalog, pack, _ = source
    W.write_metadata(root, catalog, pack)
    ledger_path = tmp_path / "resume.sqlite"
    ledger = LeaseLedger(ledger_path, node_id=Guard.node_id, authorize=lambda p, a, b: p)
    ledger.register_capacity("host", available_ram_bytes=64 << 20, pinnable_ram_bytes=0,
                             gpu_capacity_bytes={"GPU": 1000})
    ledger.register_filesystem("fs", tmp_path, available_disk_bytes=1000000)
    lease = ledger.prepare(LeaseRequest("ring", COHORT, Guard.node_id, "GPU", "host", LeaseResources(100, 1 << 20), 120),
                           principal="controller", idempotency_key="stage")
    ledger.commit(lease.lease_id, lease.fencing_token, principal="controller")
    guard = ledger.guard(lease.lease_id, lease.fencing_token, principal="controller", ring_id="ring", model_cohort_sha256=COHORT)
    request = assignment(catalog, pack)
    prep, _ = manager(tmp_path, source)
    job = prep._binding(request)[0]
    old = ledger.prepare_artifact(PrepareRequest(request["weight_artifacts"]["artifact_id"], COHORT,
        Guard.node_id, "host", PrepareResources("fs", 100000, 48 << 20), 120),
        principal="controller", idempotency_key="interrupted")
    work = old.begin_work("artifact-prepare-" + job + "-previous-worker")
    # The old in-process worker has ended; its durable handle is still active.
    rows = [{"cohort_id": COHORT, "catalog": str(root / W.GLOBAL_FILE), "pack": str(root / W.PACK_FILE),
        "cache_root": str(tmp_path / "cache"), "jobs_db": str(tmp_path / "jobs.sqlite"),
        "provider": {"kind": "local", "root": str(root)}, "resources": {"filesystem_id": "fs"}}]
    ready = configured_prepare_factory(ledger_path, rows)(request, guard)
    assert ready["ready"]
    with ledger._transaction() as (conn, _):
        row = conn.execute("SELECT state,active FROM artifact_preparations WHERE id=?", (old.lease_id,)).fetchone()
        assert tuple(row) == ("released", 0)
        assert conn.execute("SELECT ended FROM artifact_work WHERE reservation=? AND id=?", (old.lease_id, work.work_id)).fetchone()[0] is not None


def test_explicit_planned_fetch_does_not_silently_fall_back_to_another_strategy(source, tmp_path):
    prep, _ = manager(tmp_path, source, convert=lambda *_: pytest.fail("not the selected strategy"),
        repack_resources={"filesystem_id": "fs"})
    prep.reserve = lambda *_: (_ for _ in ()).throw(LeaseConflict("insufficient disk"))
    request = {**assignment(source[1], source[2]), "preparation_mode": "fetch"}
    with pytest.raises(LeaseConflict): prep.prepare(request, Guard())
    assert prep.status(prep._binding(request)[0])["error_code"] == "resource_conflict"


def test_explicit_planned_range_repack_is_used_even_if_fetch_would_fit(source, tmp_path):
    def convert(dest, catalog, pack, selected, provider, check):
        W.repack_stage(catalog, pack, selected, dest, RangeReader(provider, chunk_bytes=13),
                       chunk_bytes=13, max_file_bytes=40, cancel_check=check)
    prep, reservations = manager(tmp_path, source, convert=convert,
        repack={"max_file_bytes": 40, "chunk_bytes": 13}, repack_resources={"filesystem_id": "fs"})
    prep.provider.fetch = lambda *_: pytest.fail("the plan selected raw range reads")
    request = {**assignment(source[1], source[2]), "preparation_mode": "range_repack"}
    ready = prep.prepare(request, Guard())
    assert len(reservations) == 1 and ready["preparation_mode"] == ready["planned_preparation_mode"] == "range_repack"
    assert "converting" in ready["history"] and "fetching" not in ready["history"]
    # A repeated load reserves verification RAM and adopts the same packing.
    again = prep.prepare(request, Guard())
    assert again["directory"] == ready["directory"] and again["preparation_mode"] == "range_repack"


@pytest.mark.parametrize("mode", ["range_repack", "unapproved", True])
def test_unconfigured_or_invalid_planned_strategy_is_rejected_before_reservation(source, tmp_path, mode):
    prep, reservations = manager(tmp_path, source)
    with pytest.raises(PreparationError): prep.prepare({**assignment(source[1], source[2]), "preparation_mode": mode}, Guard())
    assert not reservations
