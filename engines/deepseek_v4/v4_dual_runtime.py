"""Static identity, bounded observations and reservations for opt-in local tiers.

Counters never enter the calibrated configuration identity. Driver IPC memory
and shared-memory witnesses are outside Torch's allocator and must be charged
separately. These declarations are not hardware attestation or speed evidence.
"""
def tail_of(stage, draft=None):
    draft = draft or getattr(stage, "_resource_drafter", None) or getattr(stage, "_runtime_draft", None)
    return getattr(draft, "tail", draft)


def config(stage):
    from v4_host_transport import public_config as transport_config
    from v4_draft_adapter import public_config as adapter_config
    import v4_dspark_fast as fast
    host = getattr(stage, "_host_compute", None)
    chain = {"enabled": fast.V4_DSPARK_CHAIN_GRAPH and bool(getattr(stage, "tail", False))}
    if chain["enabled"]:
        chain.update(graph_bytes=fast.V4_DSPARK_CHAIN_MIB << 20,
                     max_graphs=fast.V4_DSPARK_CHAIN_LIMIT,
                     reference=fast.V4_DSPARK_CHAIN_REFERENCE,
                     coverage="committed_advance_and_full_mtp", uncommitted_taps=False)
    return {"schema": "v4-local-execution-config/1",
            "expert_storage": getattr(stage, "_expert_storage", "canonical"),
            "expert_storage_host_bytes": getattr(stage, "_expert_storage_host_bytes", 0),
            "expert_execution": {"segment_graph_bytes": getattr(stage, "_expert_segment_bytes", 0),
                "ready_output_bytes": getattr(stage, "_expert_ready_bytes", 0)},
            "host_compute": host.config() if host else {"enabled": False},
            "host_transport": transport_config(), "draft_chain": chain,
            "draft_adapter": adapter_config() if getattr(stage, "tail", False) else {"enabled": False}}


def observation(stage):
    tail = tail_of(stage)
    chain = getattr(tail, "_draft_chain", None)
    host = getattr(stage, "_host_compute", None)
    transport = getattr(stage, "_host_transport", None)
    manager = getattr(stage, "_expert_cache", None)
    return {"schema": "v4-local-execution-observation/1", "scope": "stage_lifetime_not_per_job",
            "performance_qualified": False,
            "host_compute": host.status() if host else {"enabled": False},
            "host_transport": transport.status() if transport else {"enabled": False},
            "expert_storage": {"mode": getattr(manager, "storage_mode", "canonical"),
                               "host_bytes": manager.host_bytes if manager else 0,
                               "preallocated_device_bytes": manager.preallocated_device_bytes if manager else 0,
                               "execution": [b.ffn._hybrid_runtime.execution_status()
                                   for b in getattr(stage, "_hybrid_blocks", [])]},
            "draft_chain": chain.status() if chain else {"enabled": False},
            "draft_adapter": getattr(tail, "_draft_adapter_status", {"enabled": False})}


def tensors(stage, draft=None):
    for index, block in enumerate(getattr(stage, "_hybrid_blocks", [])):
        graphs = block.ffn._hybrid_runtime.segment_graphs
        if graphs is not None:
            for name, tensor in graphs.tensors():
                yield f"expert_graph.{index}.{name}", tensor
    manager = getattr(stage, "_expert_cache", None)
    if manager is not None and hasattr(manager, "storage_tensors"):
        for name, tensor in manager.storage_tensors():
            yield "expert_storage." + str(name), tensor
    chain = getattr(tail_of(stage, draft), "_draft_chain", None)
    if chain is not None:
        yield from chain.tensors()


def reservation(stage, draft=None):
    """Total external bytes plus still-unallocated bounded graph/adapter space."""
    import v4_dspark_fast as fast
    from v4_host_transport import public_config
    from v4_draft_adapter import public_config as adapter_config
    transport = getattr(stage, "_host_transport", None)
    if transport is not None:
        external = transport.reservation()
    else:
        declared = public_config()
        size, slots = declared.get("max_bytes", 0), declared.get("slots", 1)
        # Before the signed session is installed, conservatively reserve one
        # incoming and one outgoing channel. Loaded sessions refine this bound.
        digest_scratch = 8 * (size // slots) + 131072 if declared.get("digest_backend") == "device_sha256" else 5 * (size // slots)
        external = {"host_reserved_bytes": size + 4 * (size // slots) + slots * 65536 if size else 0,
                    "device_reserved_bytes": size + digest_scratch if size and str(stage.device).startswith("cuda") else 0,
                    "driver_device_allocated_bytes": 0}
    tail = tail_of(stage, draft)
    chain = getattr(tail, "_draft_chain", None)
    declared_graph = (fast.V4_DSPARK_CHAIN_MIB << 20) if fast.V4_DSPARK_CHAIN_GRAPH and getattr(stage, "tail", False) else 0
    accounted = (chain.used + chain.retained) if chain else 0
    remaining_graph = max(0, declared_graph - accounted)
    expert_accounted = sum(b.ffn._hybrid_runtime.segment_graphs.used_bytes
        for b in getattr(stage, "_hybrid_blocks", []) if b.ffn._hybrid_runtime.segment_graphs is not None)
    remaining_expert = max(0, getattr(stage, "_expert_segment_bytes", 0) - expert_accounted)
    ready_bytes = getattr(stage, "_expert_ready_bytes", 0)
    adapter = adapter_config() if getattr(stage, "tail", False) else {"enabled": False}
    # Adapter load's CPU read/model/move working set is a separate peak reserve.
    # Device parameters, once installed, already appear in MTP's inventory.
    remaining_adapter = adapter.get("device_reserved_bytes", 0) if not getattr(tail, "_draft_adapter_status", None) else 0
    reference = fast.V4_DSPARK_CHAIN_REFERENCE
    actual_driver = external["driver_device_allocated_bytes"]
    return {"host_reserved_bytes": external["host_reserved_bytes"] + adapter.get("host_reserved_bytes", 0),
            "remaining_host_reserved_bytes": external["host_reserved_bytes"] + adapter.get("host_reserved_bytes", 0)
                + (remaining_graph if reference else 0)
                + (remaining_expert + ready_bytes if getattr(stage, "_expert_cache_reference", False) else 0),
            "remaining_device_reserved_bytes": max(0, external["device_reserved_bytes"] - actual_driver)
                + (remaining_graph if not reference else 0) + remaining_adapter
                + (remaining_expert + ready_bytes if not getattr(stage, "_expert_cache_reference", False) else 0),
            "driver_device_allocated_bytes": actual_driver,
            "draft_chain_remaining_bytes": remaining_graph,
            "external_transport": external}
