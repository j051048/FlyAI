"""Integration of explicit Stage options with actual exclusive weights and state."""
import pytest
import torch
import v4_stage as V
import v4_dual_runtime as D
from test_v4_hybrid import args, checkpoint, ids, assert_equal_stages


def test_stage_segment_reservations_and_ready_first_execute_real_exclusive_weights(tmp_path, args, monkeypatch):
    monkeypatch.setattr(V, "V4_EXPERT_SEGMENT_GRAPH_MIB", 2)
    monkeypatch.setattr(V, "V4_EXPERT_READY_FIRST", True)
    monkeypatch.setattr(V, "V4_EXPERT_READY_MIB", 1)
    directory, _ = checkpoint(tmp_path / "weights", args)
    candidate = V.Stage(0, args.n_layers, args, head=True, tail=True, device="cpu",
        expert_placement="ram", expert_cache_reference=True, expert_cache_slots=3,
        expert_storage="exclusive", expert_storage_host_bytes=8 << 20).load(directory)
    monkeypatch.setattr(V, "V4_EXPERT_SEGMENT_GRAPH_MIB", 0)
    monkeypatch.setattr(V, "V4_EXPERT_READY_FIRST", False)
    control = V.Stage(0, args.n_layers, args, head=True, tail=True, device="cpu").load(directory)
    tokens = ids(args, 19)
    config_before = D.config(candidate)
    with torch.no_grad():
        assert_equal_stages(candidate, control, tokens[:, :7], 0)
        for pos in range(7, 19):
            assert_equal_stages(candidate, control, tokens[:, pos:pos + 1], pos)
    observed = D.observation(candidate)["expert_storage"]["execution"]
    assert observed and all(row["segments"]["reference_calls"] > 0 for row in observed)
    assert all(row["segments"]["captures"] == row["segments"]["replays"] == 0 for row in observed)
    assert all(row["ready_observations"]["batches"] > 0 for row in observed)
    assert D.config(candidate) == config_before
    assert any(name.startswith("expert_graph.") for name, _ in D.tensors(candidate))
    reserve = D.reservation(candidate)
    assert reserve["remaining_host_reserved_bytes"] >= 1 << 20
    candidate.reset(); control.reset()
    with torch.no_grad():
        assert_equal_stages(candidate, control, tokens[:, :9], 0)


def test_segment_experiment_requires_ram_and_valid_budget(args, monkeypatch):
    monkeypatch.setattr(V, "V4_EXPERT_SEGMENT_GRAPH_MIB", -1)
    with pytest.raises(ValueError, match="bounded"):
        V.Stage(0, 1, args, device="cpu")
    monkeypatch.setattr(V, "V4_EXPERT_SEGMENT_GRAPH_MIB", 1)
    with pytest.raises(ValueError, match="RAM expert"):
        V.Stage(0, 1, args, device="cpu")
