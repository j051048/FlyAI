"""Production seams exercised with real tiny modules; no GPU speed claims."""
import pytest

torch = pytest.importorskip("torch")
REF = pytest.importorskip("v4_ref_cpu")
V4 = pytest.importorskip("v4_stage")
VP = pytest.importorskip("v4_pipe")
from test_v4_stage import stage_from_oracle
from shard.runtime_profile import RuntimeProfiler


def test_profiled_stage_keeps_outputs_and_rollback_state_exact(monkeypatch):
    args = REF.cpu_args()
    oracle = REF.build_oracle(args, 7)
    monkeypatch.setattr(V4, "V4_PROFILE_RUNTIME", False)
    plain = stage_from_oracle(oracle, args, 0, 2, tail=False)
    monkeypatch.setattr(V4, "V4_PROFILE_RUNTIME", True)
    watched = stage_from_oracle(oracle, args, 0, 2, tail=False)
    for stage in (plain, watched):
        stage._spec = True
    for position, ids in ((0, [[1, 2, 3, 4]]), (4, [[5, 6]]), (5, [[8]])):
        ids = torch.tensor(ids)
        expected = plain.forward(plain.embed(ids), ids, position)
        actual = watched.forward(watched.embed(ids), ids, position)
        assert torch.equal(actual, expected)
    phases = watched.runtime_metrics()["performance"]["phases"]
    assert phases["main.prefill.layer.host"]["count"] == 2
    assert "main.replay.layer.host" in phases
    assert not any(name.endswith(".gpu") for name in phases)
    watched.reset()
    assert watched.runtime_metrics()["performance"]["phases"] == {}


@pytest.mark.parametrize("name", ["runtime_profile", "expert_prefetch"])
@pytest.mark.parametrize("value", ["false", "0", 1])
def test_explicit_switches_reject_truthy_non_booleans_before_model_build(monkeypatch, name, value):
    monkeypatch.setattr(V4, "ref", lambda: pytest.fail("model allocation preceded switch validation"))
    with pytest.raises(ValueError, match=name):
        V4.Stage(0, 1, device="cpu", **{name: value})


def test_router_staging_storage_is_in_the_resource_inventory_and_config_is_bound():
    from types import SimpleNamespace
    import v4_resources as resources
    layer = torch.nn.Linear(2, 2, bias=False)
    ids, slots = torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int32)
    layer._hybrid_runtime = SimpleNamespace(router_scratch_tensors=lambda: iter([
        ("ids", ids), ("ids_alias", ids[:1]), ("slots", slots)]))
    stage = SimpleNamespace(args=SimpleNamespace(n_layers=43), lo=40, hi=43, head=False,
        tail=True, _dspark=False, dtype=torch.float32, device="cpu", layers=torch.nn.ModuleList([layer]))
    report = resources.measure_stage_resources(stage, checkpoint_id="fixture")
    assert report["module_storage"]["by_kind"]["host_staging_bytes"] == 24
    assert report["module_storage"]["host_bytes"] == 16 + 24
    before = resources.runtime_config_identity(stage)
    stage._expert_prefetch = True
    assert resources.runtime_config_identity(stage) != before


def test_profile_only_timer_does_not_introduce_device_synchronization(monkeypatch):
    monkeypatch.setattr(VP, "V4_TIMING", False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: pytest.fail("profiling synchronized a token"))
    profile = RuntimeProfiler()
    timer = VP._timer("s", ("fwd", "out"), "cuda:0", profile)
    timer.sync()
    timer.lap("fwd")
    timer.lap("out")
    stats = profile.snapshot()["phases"]
    assert stats["stage.fwd.host"]["count"] == 1
    assert stats["stage.out.host"]["count"] == 1


def test_production_flags_are_carried_to_every_local_gpu(monkeypatch):
    keys = {"V4_PROFILE_RUNTIME": "1", "V4_PROFILE_GPU_EVERY": "32", "V4_EXPERT_PREFETCH": "1",
            "V4_EXPERT_PREFETCH_SLOTS": "2", "V4_EXPERT_PREFETCH_WARMUP": "3", "V4_WIRE_FUSED": "1"}
    monkeypatch.delenv("V4_CUDA_GRAPH", raising=False)
    for key, value in keys.items():
        monkeypatch.setenv(key, value)
    plan = VP.box_ring_launch([dict(id="miner", lo=0, hi=8)], 4)
    for stage in plan["stages"]:
        for key, value in keys.items():
            assert f"{key}={value} " in stage["cmd"]


def test_dspark_observation_does_not_call_partial_coverage_fully_engaged(monkeypatch):
    from types import SimpleNamespace
    import v4_levers as levers
    module = SimpleNamespace(V4_DSPARK_MOE=True, coverage_details=lambda tail: {
        0: {"cuda_grouped_gemms": 4, "declined": {}},
        1: {"cuda_grouped_gemms": 0, "declined": {"no-bank": 1}}})
    monkeypatch.setattr(levers, "_mod", lambda name: module if name == "v4_dspark_moe" else None)
    ctx = SimpleNamespace(stage=SimpleNamespace(_runtime_draft=object()))
    requested, observed, verdict = levers._check_dspark_moe(ctx)
    assert requested == "on" and observed.startswith("partial/1-of-2/") and verdict is False


def test_shared_observer_sees_mtp_instance_binding_without_a_class_patch(monkeypatch):
    from types import SimpleNamespace
    import v4_levers as levers
    module = SimpleNamespace(V4_FP8_SHARED=True, shared_status=lambda: "armed",
        drafter_shared_coverage=lambda tail: {0: dict(banked=True, instance_bound=True, cuda_steps=0)})
    monkeypatch.setattr(levers, "_mod", lambda name: module if name == "v4_fp8_gemv" else None)
    ctx = SimpleNamespace(stage=SimpleNamespace(_runtime_draft=object(), _shared_banked=0),
                          mod=SimpleNamespace(Expert=SimpleNamespace(forward=lambda: None)))
    requested, observed, verdict = levers._check_fp8_shared(ctx)
    assert requested == "on" and "draft-banked-1" in observed and verdict is None


def test_prefetch_runs_before_real_attention_and_keeps_reference_output(tmp_path, monkeypatch):
    from test_v4_hybrid import checkpoint, stage
    args = REF.cpu_args(n_layers=4, n_routed_experts=6, n_activated_experts=2,
        compress_ratios=(0, 4, 8, 0, 0, 0), dspark_target_layer_ids=(1, 2, 3))
    directory, _ = checkpoint(tmp_path / "model", args, force_routes=True)
    monkeypatch.setattr(V4, "V4_EXPERT_PREFETCH", False)
    baseline = stage(directory, args, hybrid=True, slots=3)
    monkeypatch.setattr(V4, "V4_EXPERT_PREFETCH", True)
    watched = stage(directory, args, hybrid=True, slots=3, metrics=True)
    events, handles = [], []
    for i, block in enumerate(watched.layers):
        runtime = block.ffn._hybrid_runtime
        original = runtime.prefetch_for_attention
        def predict(i=i, original=original):
            events.append(("prefetch", i))
            return original()
        monkeypatch.setattr(runtime, "prefetch_for_attention", predict)
        handles.append(block.attn.register_forward_pre_hook(
            lambda m, inp, i=i: events.append(("attention", i))))
    ids = torch.tensor([[1, 2, 3, 4]])
    assert torch.equal(watched.forward(watched.embed(ids), ids, 0),
                       baseline.forward(baseline.embed(ids), ids, 0))
    assert events == [entry for i in range(4) for entry in (("prefetch", i), ("attention", i))]
    for position in range(4, 9):
        ids = torch.tensor([[position + 1]])
        assert torch.equal(watched.forward(watched.embed(ids), ids, position),
                           baseline.forward(baseline.embed(ids), ids, position))
    assert watched.runtime_metrics()["totals"]["dma_misses"] == 0  # explicit CPU emulation
    # Rollback replay must not manufacture prediction work for discarded futures.
    watched._replaying = True
    events.clear()
    watched._run(watched.embed(ids), ids, 9, {})
    assert not any(kind == "prefetch" for kind, _ in events)
    for handle in handles:
        handle.remove()
