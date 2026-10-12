"""Offline cross-report identity binding; no GPU, subprocess or remote actions.

These checks supplement independent numeric/benchmark/soak verification. They
never trust a saved passing status and never qualify throughput on their own.
Every public function returns errors and leaves the supplied evidence unchanged.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re


def _failure(label, error):
    return [f"{label} binding invalid: {type(error).__name__}: {error}"]


def _source_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("complete frozen source file rows required")
    result = {}
    for row in rows:
        name, sha = row["name"], row["sha256"]
        if (not isinstance(name, str) or not name or name in result or "\\" in name or ":" in name
                or PurePosixPath(name).is_absolute() or any(part in ("", ".", "..") for part in name.split("/"))
                or not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None):
            raise ValueError("invalid or duplicate frozen source identity")
        result[name] = sha
    return result


def numeric_binding_errors(numeric_plan, frozen_campaign, checkpoint, frozen_input_path):
    """Bind a verified numerical plan to this campaign's model and exact input."""
    errors = []
    try:
        from tools import v4_acceptance_numeric as N
        from phase0 import v4_benchmark as B
        if (numeric_plan.get("schema") != N.SCHEMA or numeric_plan.get("sha256") != N.digest(
                {k: v for k, v in numeric_plan.items() if k != "sha256"})):
            errors.append("numeric plan schema/hash differs")
        if (frozen_campaign.get("schema") != "flyai-v4-5090-campaign-plan/1"
                or frozen_campaign.get("sha256") != B.digest({k: v for k, v in frozen_campaign.items() if k != "sha256"})):
            errors.append("campaign schema/hash differs")
        if (frozen_campaign.get("numeric_max_tokens") != 8192 or frozen_campaign.get("numeric_input_tokens") != 4104
                or numeric_plan.get("fixture") is not False or numeric_plan.get("device") != "cuda:0"):
            errors.append("numeric input/device contract differs or uses a fixture")
        if checkpoint.get("identity_kind") != "logical-tensor-catalogue":
            raise ValueError("native logical checkpoint identity required")
        proof = numeric_plan["artifact_preview"]
        for key in ("checkpoint_id", "manifest_sha256", "config_sha256"):
            if not checkpoint.get(key) or proof.get(key) != checkpoint[key]:
                errors.append("numeric " + key + " differs from benchmark checkpoint")
        if numeric_plan.get("config_asset_sha256") != checkpoint.get("config_sha256"):
            errors.append("numeric config asset differs from benchmark config")
        cfg = {**N.model_defaults(), **checkpoint["global_catalog"]["config"],
               "max_seq_len": 8192, "max_batch_size": 1}
        ratios = cfg["compress_ratios"]
        if cfg["n_layers"] != 43 or len(ratios) < 43 or cfg["index_topk"] < 1:
            raise ValueError("native 43-layer compression/indexer geometry required")
        hi = next((i + 1 for i in range(43) if {4, 128} <= set(ratios[:i + 1])), None)
        if hi is None:
            raise ValueError("actual head prefix lacks ratio-4 indexer and ratio-128 layers")
        if (type(numeric_plan.get("lo")) is not int or type(numeric_plan.get("hi")) is not int
                or numeric_plan.get("lo") != 0 or numeric_plan.get("hi") != hi
                or numeric_plan.get("roles") != {"head": True, "tail": False, "dspark": False}
                or any(proof["assigned"].get(key) != value for key, value in
                       {"lo": 0, "hi": hi, "head": True, "tail": False, "dspark": False}.items())):
            errors.append("numeric assigned head prefix/roles differ from actual compression geometry")
        if (numeric_plan["limits"].get("tokens") != 8192
                or numeric_plan.get("runtime_model_args_sha256") != N.digest(cfg)
                or N.digest(numeric_plan["runtime_model_args"]) != N.digest(cfg)):
            errors.append("numeric model arguments or 8192-token capacity differ")
        current = N.input_descriptor(frozen_input_path, cfg, 0, 8192)
        original = numeric_plan["input"]
        if (current["tokens"] != 4104 or set(current["tensors"]) != {"token_ids"}
                or {k: v for k, v in current.items() if k != "path"} !=
                   {k: v for k, v in original.items() if k != "path"}):
            errors.append("numeric original 4104 frozen token-ID bytes differ")
        expected_environment = dict(N.NUMERIC_ENV, V4_MAX_SEQ="8192")
        campaign_env = frozen_campaign["config"]["ring"]["environment"]
        for key in expected_environment:
            if key != "V4_EXECUTION_PROFILE" and key in campaign_env:
                expected_environment[key] = campaign_env[key]
        if (numeric_plan.get("environment") != expected_environment
                or numeric_plan.get("environment_sha256") != N.digest(expected_environment)):
            errors.append("numeric numerical recipe differs from the campaign")
        core = _source_rows(frozen_campaign["source"]["files"])
        kit = _source_rows(frozen_campaign["kit"])
        if any(name in core and core[name] != sha for name, sha in kit.items()):
            raise ValueError("conflicting frozen core/helper source identities")
        known = {**core, **kit}
        files = numeric_plan["source_files"]
        required = {name for name in core if
            (name.startswith("engines/deepseek_v4/v4_") and name.endswith(".py"))
            or (name.startswith("shard/") and len(Path(name).parts) == 2 and name.endswith(".py"))
            or (name.startswith("vendor/deepseek_v4_ref/") and name.endswith(".py"))}
        required |= {"phase0/v4_numeric_diagnose.py", "tools/v4_acceptance_numeric.py"}
        if (not isinstance(files, dict) or set(files) != required
                or any(name not in known or known[name] != sha for name, sha in files.items())
                or numeric_plan.get("source_sha256") != N.digest(files)):
            errors.append("numeric executed source inventory differs from frozen core/helpers")
    except Exception as error:
        errors.extend(_failure("numeric", error))
    return errors


def _plan_protocol_errors(plan, protocol):
    from phase0 import v4_benchmark as B
    from shard.pipeline_session import SessionConfig
    from shard.weight_artifacts import catalog_model_cohort
    errors = list(B.protocol_errors(protocol))
    session = SessionConfig.from_plan(plan)
    plan = session.plan
    hardware, stages = protocol["hardware"], plan["stages"]
    expected = [(n["node_id"], n["gpu_uuid"], n["signer_pubkey"], n["layer_start"], n["layer_end"]) for n in hardware]
    actual = [(n["node_id"], n["gpu_uuid"], n["signer_pubkey"], n["lo"], n["hi"]) for n in stages]
    if expected != actual or plan["n_layers"] != 43:
        errors.append("strict deployment node/GPU/signer/layer assignment differs from protocol")
    checkpoint = protocol["checkpoint"]
    if checkpoint.get("identity_kind") != "logical-tensor-catalogue":
        errors.append("strict deployment requires the native logical checkpoint")
    else:
        cohort = catalog_model_cohort(checkpoint["global_catalog"])
        if plan.get("model_cohort") != cohort:
            errors.append("strict deployment model cohort differs from benchmark checkpoint/config")
    if plan["coordinator"]["signer_pubkey"] in {n["signer_pubkey"] for n in stages}:
        errors.append("coordinator signing identity must be distinct from stage signers")
    return errors, plan


def deployment_binding_errors(plan, protocol, ring_config):
    """Bind the strict deployed ring to the frozen ports, roles and identities."""
    errors = []
    try:
        problems, plan = _plan_protocol_errors(plan, protocol)
        errors.extend(problems)
        rows, stages = ring_config["stages"], plan["stages"]
        if plan["ring_id"] != ring_config["ring_id"]:
            errors.append("strict deployment ring ID differs from campaign")
        if ring_config.get("model_id") is not None and ring_config["model_id"] != protocol["model_id"]:
            errors.append("strict deployment model ID differs from campaign")
        expected = [(n["node_id"], n["lo"], n["hi"], f"127.0.0.1:{n['port']}") for n in rows]
        actual = [(n["node_id"], n["lo"], n["hi"], n["endpoint"]) for n in stages]
        if actual != expected:
            errors.append("strict deployment spans/nodes/loopback ports differ from campaign")
        endpoints = [row[3] for row in expected]
        if ([n["next_endpoint"] for n in stages] != endpoints[1:] + [None]
                or plan["coordinator"]["head"] != endpoints[0]
                or plan["coordinator"]["tail"] != endpoints[-1]
                or plan["coordinator"].get("node_id") != ring_config["ring_id"] + "-coordinator"):
            errors.append("strict forward/direct return/coordinator routes differ from campaign")
        max_seq = int(ring_config["environment"]["V4_MAX_SEQ"])
        if (protocol["env"] != ring_config["environment"]
                or plan.get("execution", {}).get("max_context") != max_seq
                or any(type(n.get("context_limit")) is not int or n["context_limit"] != max_seq for n in stages)):
            errors.append("strict per-stage context/environment differs from campaign")
    except Exception as error:
        errors.extend(_failure("deployment", error))
    return errors


def cold_binding_errors(loaded, protocol, previous_process_ids):
    """Bind zero-request loaded observations to newly booted measured workers.

    The caller retains the process IDs after this returns no errors. This
    function never changes that collection or authenticates unsigned boot logs.
    Completed benchmark receipts authenticate the same pinned runtime identities.
    """
    errors = []
    try:
        from phase0 import v4_benchmark as B
        errors.extend(B.protocol_errors(protocol))
        if (loaded.get("schema") != "flyai-v4-acceptance-ring-loaded/1"
                or loaded.get("cold_open") is not True or loaded.get("ready") is not False
                or type(loaded.get("request_count")) is not int or loaded["request_count"] != 0):
            errors.append("cold run lacks a zero-request load-only boot record")
        if (not isinstance(previous_process_ids, (set, frozenset, list, tuple))
                or any(not isinstance(value, str) or not value for value in previous_process_ids)):
            raise ValueError("prior process IDs must be an explicit collection of nonempty IDs")
        seen = set(previous_process_ids)
        observations = loaded["observations"]
        hardware = protocol["hardware"]
        if (not isinstance(observations, list) or len(observations) != len(hardware)
                or [row.get("node_id") for row in observations] != [row["node_id"] for row in hardware]):
            raise ValueError("one ordered loaded observation per actual stage required")
        process_ids = set()
        for node, observation in zip(hardware, observations):
            process_id = observation.get("process_run_id")
            if not isinstance(process_id, str) or not process_id or process_id in seen or process_id in process_ids:
                errors.append("cold stage process ID was missing, duplicated or reused from another cell")
            process_ids.add(process_id)
            if observation.get("phase") != "loaded":
                errors.append("cold boot observation has already completed model work")
            if not node.get("runtime_config_sha256"):
                errors.append("cold stage lacks a pinned actual runtime configuration")
            problems = B.runtime_observation_errors(observation, node, protocol)
            errors.extend(problem for problem in problems if problem !=
                          "runtime observation must describe the measured completed job")
    except Exception as error:
        errors.extend(_failure("cold", error))
    return errors


def warmup_binding_errors(proof, protocol, plan):
    """Reverify warmup signatures and captured layers for this exact warm ring."""
    errors = []
    try:
        from phase0 import v4_benchmark as B
        from shard.receipt import verify_coverage, wire_receipt
        from engines.deepseek_v4.v4_execution_profiles import require_profile_warmup
        problems, plan = _plan_protocol_errors(plan, protocol)
        errors.extend(problems)
        if (proof.get("verified") is not True or not isinstance(proof.get("job_id"), str) or not proof["job_id"]
                or not isinstance(proof.get("nonce"), str) or len(proof["nonce"]) < 32):
            raise ValueError("warmup job/nonce/verification declaration incomplete")
        nodes = {node["signer_pubkey"]: node for node in protocol["hardware"]}
        assignments = {key: (node["layer_start"], node["layer_end"]) for key, node in nodes.items()}
        receipts = [wire_receipt(row) for row in proof["receipts"]]
        verify_coverage(receipts, 43, expected_by_signer=assignments, expected_nonce=proof["nonce"], check_chain=True)
        for receipt in receipts:
            if receipt["job_id"] != proof["job_id"] or receipt["swarm_id"] != plan["ring_id"]:
                errors.append("warmup signed job/ring binding differs")
            observation = receipt.get("runtime_observation")
            if observation is None:
                errors.append("warmup lacks a signed actual runtime observation")
            else:
                errors.extend(B.runtime_observation_errors(observation, nodes[receipt["pubkey"]], protocol))
        profile = protocol["env"].get("V4_EXECUTION_PROFILE")
        if profile != "resident-whole":
            errors.append("campaign warmup requires its frozen resident-whole profile")
        require_profile_warmup(receipts, {node["node_id"]: "resident-whole" for node in protocol["hardware"]})
    except Exception as error:
        errors.extend(_failure("warmup", error))
    return errors
