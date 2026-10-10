"""Authenticated prompt-boundary state-cache barriers over the existing ring.

Only control acknowledgements and opaque prefix bindings cross nodes. The cache
does not transport weights/KV, reuse old receipts, or turn a restore into a forward.
"""
from collections import OrderedDict
import base64
from copy import deepcopy
import hashlib
import json
import math
import os
import secrets
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

CONTROL = "v4-conversation-control/1"
ACK = "v4-conversation-ack/1"
MATH_REPLY = {"token", "tokens", "acc", "draft", "conf", "n", "d2"}
V4_CONVERSATION_CACHE_MIB = int(os.environ.get("V4_CONVERSATION_CACHE_MIB", "0"))
V4_CONVERSATION_CACHE_GPU_MIB = int(os.environ.get("V4_CONVERSATION_CACHE_GPU_MIB", "0"))
V4_CONVERSATION_CACHE_ENTRIES = int(os.environ.get("V4_CONVERSATION_CACHE_ENTRIES", "4"))
V4_CONVERSATION_CACHE_TTL_S = float(os.environ.get("V4_CONVERSATION_CACHE_TTL_S", "300"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sign(key, domain, value):
    return {**value, "signature": base64.b64encode(key.sign(domain + canonical(value))).decode()}


def _verify(public, domain, value):
    if not isinstance(value, dict) or "signature" not in value:
        raise ValueError("unsigned conversation control/evidence")
    body = {k: v for k, v in value.items() if k != "signature"}
    Ed25519PublicKey.from_public_bytes(base64.b64decode(public, validate=True)).verify(
        base64.b64decode(value["signature"], validate=True), domain + canonical(body))
    return body


def configured_cache(stage):
    from v4_conversation_cache import CacheQuotas, StageConversationCache
    if V4_CONVERSATION_CACHE_MIB < 0 or V4_CONVERSATION_CACHE_GPU_MIB < 0:
        raise ValueError("conversation cache budgets must be nonnegative")
    quotas = CacheQuotas(host_bytes=V4_CONVERSATION_CACHE_MIB << 20,
        gpu_bytes=V4_CONVERSATION_CACHE_GPU_MIB << 20,
        max_entries=V4_CONVERSATION_CACHE_ENTRIES, ttl_s=V4_CONVERSATION_CACHE_TTL_S)
    return StageConversationCache(stage, quotas, enabled=V4_CONVERSATION_CACHE_MIB > 0)


def cache_config(stage=None):
    return {"enabled": V4_CONVERSATION_CACHE_MIB > 0,
        "host_reserved_bytes": V4_CONVERSATION_CACHE_MIB << 20,
        "restore_gpu_reserved_bytes": V4_CONVERSATION_CACHE_GPU_MIB << 20,
        "entries": V4_CONVERSATION_CACHE_ENTRIES, "ttl_s": V4_CONVERSATION_CACHE_TTL_S,
        "scope": "node-local exact-repeat state snapshots; extended prefixes require per-request shadow"}


def execution_binding(message):
    """Request-dependent arithmetic, including RefSlim's history decision."""
    if any(name in message and type(message[name]) is not bool for name in ("spec", "dspark", "pipelined")):
        raise ValueError("cache execution mode flags must be booleans")
    if type(message.get("temp", 0)) not in (int, float):
        raise ValueError("cache execution temperature must be numeric")
    result = {"max_pos": message.get("max_pos"), "spec": bool(message.get("spec", False)),
        "dspark": bool(message.get("dspark", False)), "pipelined": bool(message.get("pipelined", False)),
        "temp": float(message.get("temp", 0)), "seed": message.get("seed", 0)}
    if result["max_pos"] is not None and (type(result["max_pos"]) is not int or result["max_pos"] < 1):
        raise ValueError("invalid cache execution horizon")
    if not math.isfinite(result["temp"]) or type(result["seed"]) is not int:
        raise ValueError("invalid cache sampling execution binding")
    return result


def _restore_shadow_flags(stage, shadow):
    saved = shadow.pop("step_flags", None)
    if saved is not None:
        stage._spec, stage._replaying = saved


def reset_shadow(stage):
    """Reset/disconnect cleanup without touching the already-owned model buffers."""
    shadow = getattr(stage, "_conversation_shadow", None)
    if shadow is not None:
        _restore_shadow_flags(stage, shadow)
    stage._conversation_shadow = None
    stage._conversation_restored_reply = None


def shadow_before_step(stage, message):
    """Only marked, sequential s=1 suffix frames run with rollback disabled."""
    transaction = message.get("conversation_shadow")
    if transaction is None:
        return False
    shadow = getattr(stage, "_conversation_shadow", None)
    if (shadow is None or transaction != shadow["transaction_id"] or shadow.get("failed")
            or message.get("start_pos") != shadow["next_pos"] or shadow["next_pos"] >= shadow["full_count"]):
        raise ValueError("shadow suffix frame is not bound to the active sequential transaction")
    if "step_flags" in shadow:
        raise ValueError("overlapping shadow suffix step")
    shadow["step_flags"] = (stage._spec, stage._replaying)
    stage._spec, stage._replaying = False, True
    return True


def shadow_after_step(stage, message, out=None):
    """Finally hook; tap retention is idempotent when the tail supplies its reply."""
    transaction = message.get("conversation_shadow")
    if transaction is None:
        return out
    shadow = getattr(stage, "_conversation_shadow", None)
    if shadow is None or transaction != shadow["transaction_id"]:
        raise ValueError("shadow suffix completion has no matching transaction")
    _restore_shadow_flags(stage, shadow)
    pos = message.get("start_pos")
    if shadow.get("last_stored_pos") != pos:
        if stage._pos != pos + 1:
            shadow["failed"] = True
            return out
        for layer, pieces in shadow["taps"].items():
            tap = stage._last_tap.get(layer)
            if tap is None or tap.shape[1] != 1:
                shadow["failed"] = True
                raise ValueError("shadow suffix did not produce one main tap per owned target layer")
            pieces.append(tap)
        shadow["last_stored_pos"] = pos
        shadow["next_pos"] = pos + 1
    if out is not None:
        if not isinstance(out, dict) or type(out.get("token")) is not int:
            raise ValueError("shadow tail has no mathematical token reply")
        if shadow.get("last_reply_pos") != pos:
            shadow["suffix_replies"].append({k: deepcopy(out[k]) for k in MATH_REPLY if k in out})
            shadow["last_reply_pos"] = pos
    return out


def _shadow_control(stage, request, *, cache, identity, prefix, drafter):
    action, transaction = request["action"], request["transaction_id"]
    parameters = request.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("shadow parameters must be an object")
    if action == "shadow_begin":
        if set(parameters) != {"full_prefix_count", "parent_count"}:
            raise ValueError("shadow begin needs exact prefix counts")
        full, parent = parameters["full_prefix_count"], parameters["parent_count"]
        if (type(full) is not int or type(parent) is not int or not 0 < parent < full
                or full != prefix.token_count or full > stage.args.max_seq_len or stage._pos != parent
                or getattr(stage, "_conversation_shadow", None) is not None):
            raise ValueError("shadow begin is not a drained parent-prefix boundary")
        import torch
        taps = {}
        if stage._dspark:
            for layer in stage._tap_ids:
                tap = stage._last_tap.get(layer)
                if tap is None or tap.shape[0] != 1 or tap.shape[1] != parent:
                    raise ValueError("restored parent has no complete DSpark target taps")
                taps[layer] = [tap]
        # Retained pieces, concatenated taps and the MTP hidden concatenation
        # are bounded explicitly. Ordinary model-forward workspace is covered
        # by the existing runtime calibration, not invented by this quota.
        scratch = sum(t[0].numel() // parent * full * t[0].element_size() * 3 for t in taps.values())
        if torch.device(stage.device).type == "cuda":
            if scratch > cache.quotas.gpu_bytes:
                raise ValueError("shadow tap workspace exceeds explicit conversation GPU quota")
            free, _ = torch.cuda.mem_get_info(stage.device)
            if scratch > free:
                raise ValueError("shadow tap workspace exceeds current free device bytes")
        elif cache.status()["host_bytes"] + scratch > cache.quotas.host_bytes:
            raise ValueError("shadow tap workspace exceeds explicit conversation host quota")
        stage._conversation_shadow = {"transaction_id": transaction, "full_count": full,
            "parent_count": parent, "next_pos": parent, "taps": taps, "suffix_replies": [],
            "identity": request["identity"], "prefix": request["prefix"], "job": request["job"],
            "parent_reply": deepcopy(getattr(stage, "_conversation_restored_reply", None)),
            "spec": stage._spec, "replaying": stage._replaying}
        stage._conversation_restore_note = None
        return {"ok": True, "reason": "shadow_suffix_armed"}
    if action == "shadow_finish":
        shadow = getattr(stage, "_conversation_shadow", None)
        if (shadow is None or shadow["transaction_id"] != transaction or shadow.get("failed")
                or shadow["identity"] != request["identity"] or shadow["prefix"] != request["prefix"]
                or shadow["job"] != request["job"] or stage._pos != prefix.token_count
                or shadow["next_pos"] != prefix.token_count or parameters):
            raise ValueError("shadow suffix is not fully completed and bound")
        _restore_shadow_flags(stage, shadow)
        import torch
        if shadow["taps"]:
            stage._last_tap = {layer: torch.cat(pieces, dim=1) for layer, pieces in shadow["taps"].items()}
        shadow["taps"] = {}  # Release pieces before MTP allocates main_hidden.
        reply = None
        if stage.tail:
            rows = shadow["suffix_replies"]
            if len(rows) != shadow["full_count"] - shadow["parent_count"]:
                raise ValueError("shadow tail suffix replies are incomplete")
            reply = {"token": rows[-1]["token"]}
            if shadow["spec"]:
                old = shadow["parent_reply"]
                if not isinstance(old, dict) or len(old.get("tokens", ())) != shadow["parent_count"]:
                    raise ValueError("shadow speculative parent omits per-position model tokens")
                reply["tokens"] = list(old["tokens"]) + [row["token"] for row in rows]
            if drafter is not None:
                # RingDrafter's start_pos=0 branch only uses IDs' shape and the
                # final predicted token. Actual prompt IDs remain coordinator-local.
                reply.update(drafter.on_chunk({"ids": [[0] * prefix.token_count], "start_pos": 0}, stage, reply))
            stage._conversation_prefill_reply = deepcopy(reply)
        stage._conversation_shadow = None
        return {"ok": True, "reason": "shadow_candidate_ready", "tail_reply": reply}
    if action == "shadow_compare":
        if parameters:
            raise ValueError("unexpected shadow comparison parameters")
        entry = request["entries"].get(str(request["stage_index"]))
        if not entry:
            raise ValueError("shadow comparison omits local candidate")
        result = cache.shadow_compare(entry, identity=identity, prefix_ids=prefix,
            tail_reply=getattr(stage, "_conversation_prefill_reply", None) if stage.tail else None)
        return {"ok": True, "reason": "shadow_compared", "shadow_passed": result["passed"],
                "shadow_scope": result["scope"]}
    if action == "shadow_abort":
        reset_shadow(stage)
        cache.reset_current()
        stage._conversation_restore_note = None
        return {"ok": True, "reason": "shadow_aborted_and_reset"}
    raise ValueError("unsupported shadow control")


def handle_stage_control(stage, message, *, stage_index, session, signer_key, drafter=None):
    """Return one signed acknowledgement; a rejected operation leaves a live ring.

    The coordinator verifies every stage acknowledgement before commit/publication.
    Authentication/binding failures never write cache state.
    """
    if session is None or signer_key is None:
        raise ValueError("conversation caching requires a strict authenticated session")
    request = _verify(session.plan["coordinator"]["signer_pubkey"], b"v4-conversation-control/1\0", message["request"])
    expected = {"schema", "action", "transaction_id", "identity", "prefix", "entries", "job", "execution"}
    if set(request) not in (expected, expected | {"parameters"}) or request["schema"] != CONTROL:
        raise ValueError("invalid conversation control fields")
    if request["action"] not in ("capture", "capture_abort", "restore_prepare", "restore_commit", "abort",
                                  "shadow_begin", "shadow_finish", "shadow_compare", "shadow_abort"):
        raise ValueError("unsupported conversation action")
    if "parameters" in request and not request["action"].startswith("shadow_"):
        raise ValueError("unexpected conversation control parameters")
    if (not isinstance(request["entries"], dict) or
            request["execution"] != execution_binding(getattr(stage, "_conversation_reset_parameters", {}))):
        raise ValueError("conversation control differs from the current reset arithmetic")
    job = request["job"]
    if (not isinstance(job, dict) or set(job) != {"job_id", "nonce", "swarm_id"}
            or job != getattr(stage, "_conversation_job_binding", None)):
        raise ValueError("conversation command is not bound to the current reset")
    from v4_conversation_cache import ConversationIdentity, PrefixBinding
    identity = ConversationIdentity.from_dict(request["identity"])
    prefix = PrefixBinding.from_dict(request["prefix"])
    if identity.ring_id != session.plan["ring_id"] or identity.cohort_id != session.plan["cohort_id"]:
        raise ValueError("conversation identity belongs to another ring/cohort")
    if identity.ring_epoch != getattr(stage, "_conversation_session_epoch", None):
        raise ValueError("conversation identity belongs to another owner/boot generation")
    guard = getattr(stage, "_conversation_lease_guard", None)
    fences = dict(identity.lease_fences)
    node = session.stage(stage_index)["node_id"]
    if guard is not None:
        guard.assert_live()
        if fences.get(node) != guard.fencing_token:
            raise ValueError("conversation lease fence differs from the executing stage")
    elif fences.get(node) != 1:
        raise ValueError("manual cache requires the explicit non-leased fence marker")
    transaction_id = request["transaction_id"]
    if not isinstance(transaction_id, str) or not 1 <= len(transaction_id) <= 128:
        raise ValueError("invalid conversation transaction identity")
    ack = {"schema": ACK, "action": request["action"], "transaction_id": transaction_id,
        "stage": stage_index, "job": job, "prefix": request["prefix"], "ok": False,
        "identity_sha256": hashlib.sha256(canonical(request["identity"])).hexdigest(),
        "execution_sha256": hashlib.sha256(canonical(request["execution"])).hexdigest(),
        "parameters_sha256": hashlib.sha256(canonical(request.get("parameters", {}))).hexdigest(),
        "entry_id": None, "entry_digest": None, "state_digest": None, "tail_reply": None, "reason": "disabled"}
    cache = getattr(stage, "_conversation_cache", None)
    if cache is not None and cache.enabled:
        cache.bind_drafter(drafter)
        pending = getattr(stage, "_conversation_pending", {})
        stage._conversation_pending = pending
        try:
            action = request["action"]
            if action.startswith("shadow_"):
                ack.update(_shadow_control(stage, dict(request, stage_index=stage_index), cache=cache,
                    identity=identity, prefix=prefix, drafter=drafter))
            elif action == "capture":
                reply = getattr(stage, "_conversation_prefill_reply", None) if stage.tail else None
                entry_id = cache.capture(identity, prefix, committed_frontier=prefix.token_count,
                    drained=True, tail_reply=reply)
                if entry_id is None:
                    ack["reason"] = "capacity_or_disabled"
                else:
                    ack.update(ok=True, entry_id=entry_id, entry_digest=cache.entry_digest(entry_id), reason="captured")
            elif action == "restore_prepare":
                if len(pending) >= cache.quotas.max_entries * 2:
                    raise ValueError("prepared conversation transaction capacity exceeded")
                ticket = cache.prepare_restore(identity, prefix, entry_id=request["entries"].get(str(stage_index)))
                if ticket is not None:
                    if transaction_id in pending:
                        cache.abort_restore(ticket)
                        raise ValueError("conversation transaction already prepared")
                    pending[transaction_id] = {"ticket": ticket,
                        "binding": hashlib.sha256(canonical({k: request[k] for k in ("identity", "prefix", "entries", "job", "execution")})).hexdigest()}
                    ack.update(ok=True, entry_id=ticket["entry_id"], reason="prepared")
                else:
                    ack["reason"] = "miss"
            elif action == "restore_commit":
                prepared = pending.pop(transaction_id, None)
                if prepared is None:
                    raise ValueError("conversation commit has no prepared ticket")
                ticket = prepared["ticket"]
                if prepared["binding"] != hashlib.sha256(canonical({k: request[k] for k in ("identity", "prefix", "entries", "job", "execution")})).hexdigest():
                    cache.abort_restore(ticket)
                    raise ValueError("conversation commit differs from its prepare binding")
                result = cache.commit_restore(ticket)
                stage._conversation_restored_reply = deepcopy(result["tail_reply"]) if stage.tail else None
                stage._conversation_restore_note = {"schema": "v4-conversation-restore/1",
                    "transaction_id": transaction_id, "prefix": request["prefix"],
                    "identity_sha256": hashlib.sha256(canonical(request["identity"])).hexdigest(),
                    "entry_id": ticket["entry_id"], "state_digest": result["state_digest"],
                    "scope": "restored prefix state; receipt chunks count only new forwards"}
                ack.update(ok=True, entry_id=ticket["entry_id"], state_digest=result["state_digest"],
                           entry_digest=result["entry_content_sha256"],
                           tail_reply=result["tail_reply"] if stage.tail else None, reason="restored")
            else:
                prepared = pending.pop(transaction_id, None)
                if prepared is not None:
                    cache.abort_restore(prepared["ticket"])
                entry = request["entries"].get(str(stage_index))
                if entry:
                    cache.invalidate(entry)
                if action == "abort":
                    cache.reset_current()
                    stage._conversation_restore_note = None
                ack.update(ok=True, reason="aborted_and_reset" if action == "abort" else "capture_discarded")
        except Exception as error:
            ack["reason"] = type(error).__name__
    return _sign(signer_key, b"v4-conversation-ack/1\0", ack)


class ConversationCoordinator:
    """Serial owner-local index of fully published, node-local snapshots."""
    def __init__(self, *, enabled=False, max_entries=128, ttl_s=300, extended_shadow=False,
                 max_prefix_tokens=8192):
        if (type(enabled) is not bool or type(extended_shadow) is not bool or type(max_entries) is not int
                or max_entries < 1 or type(ttl_s) not in (float, int) or not math.isfinite(ttl_s) or ttl_s <= 0
                or type(max_prefix_tokens) is not int or max_prefix_tokens < 1):
            raise ValueError("bounded conversation coordinator configuration required")
        self.enabled, self.max_entries, self.ttl_s = enabled, max_entries, ttl_s
        self.extended_shadow, self.max_prefix_tokens = extended_shadow, max_prefix_tokens
        self.entries = OrderedDict()

    def request(self, identity, prefix, *, max_new):
        if type(max_new) is not int or max_new < 1:
            raise ValueError("positive request token budget required")
        return ConversationRequest(self, identity, prefix, enabled=self.enabled and max_new > 1)


class ConversationRequest:
    def __init__(self, owner, identity, prefix, *, enabled):
        self.owner, self.identity, self.prefix, self.enabled = owner, deepcopy(identity), deepcopy(prefix), enabled
        self.base_identity = deepcopy(identity)
        self.key = hashlib.sha256(canonical([identity, prefix])).hexdigest()
        self.pending_reply = None
        self.prefill_waiting = False
        self.job = None
        self.reset_message = None
        self.execution = execution_binding({})
        self.raw_prefix = None
        self.eos_ids = set()
        self.shadow = None
        self.cancel_check = None
        self.record = {"enabled": enabled, "hit": False, "mode": "exact_repeat", "reason": "disabled" if not enabled else "miss"}

    def bind_tokens(self, prompt_ids, *, eos_ids=()):
        """Coordinator-private token index; IDs never enter control packets."""
        ids = tuple(prompt_ids)
        if (not ids or len(ids) != self.prefix["token_count"] or
                any(type(v) is not int or v < 0 for v in ids)):
            raise ValueError("request IDs do not match its opaque exact prefix")
        eos_ids = tuple(eos_ids)
        if any(type(v) is not int or v < 0 for v in eos_ids):
            raise ValueError("invalid cache request EOS IDs")
        self.eos_ids = set(eos_ids)
        if len(ids) <= self.owner.max_prefix_tokens:
            self.raw_prefix = ids

    def _control(self, pipe, ret, send, recv, action, transaction, entries, *,
                 identity=None, prefix=None, parameters=None):
        cfg = getattr(pipe, "config", None)
        if cfg is None or cfg.caller_key is None:
            raise ValueError("state caching requires a pinned strict coordinator key")
        identity = self.identity if identity is None else identity
        prefix = self.prefix if prefix is None else prefix
        body = {
            "schema": CONTROL, "action": action, "transaction_id": transaction,
            "identity": identity, "prefix": prefix, "entries": entries, "job": self.job,
            "execution": self.execution}
        if parameters is not None:
            body["parameters"] = parameters
        request = _sign(cfg.caller_key, b"v4-conversation-control/1\0", body)
        self._guard()
        send(pipe, {"op": "conversation_control", "request": request, "acks": []})
        result = recv(ret)
        self._guard()
        if not isinstance(result, dict) or result.get("op") != "conversation_ack" or result.get("job") != self.job:
            raise ValueError("conversation barrier has an invalid current-job binding")
        rows = result.get("acks")
        if not isinstance(rows, list) or len(rows) != cfg.plan["nstages"]:
            raise ValueError("conversation barrier omits stage acknowledgements")
        valid = []
        for index, row in enumerate(rows):
            body = _verify(cfg.stage(index)["signer_pubkey"], b"v4-conversation-ack/1\0", row)
            if any(body.get(k) != v for k, v in {"schema": ACK, "stage": index, "action": action,
                "transaction_id": transaction, "job": self.job, "prefix": prefix,
                "identity_sha256": hashlib.sha256(canonical(identity)).hexdigest(),
                "execution_sha256": hashlib.sha256(canonical(self.execution)).hexdigest(),
                "parameters_sha256": hashlib.sha256(canonical(parameters or {})).hexdigest()}.items()):
                raise ValueError("conversation acknowledgement belongs to another stage/request")
            valid.append(body)
        return valid

    def _guard(self):
        if self.cancel_check is not None:
            self.cancel_check()

    def after_reset(self, pipe, ret, send, recv, message):
        if not self.enabled:
            return
        self.job = {k: message[k] for k in ("job_id", "nonce", "swarm_id")}
        self.reset_message = dict(message)
        self.execution = execution_binding(message)
        # Preserve the static runtime hash and bind request-dependent recurrence
        # separately into the effective cache identity. Nonces stay per job.
        self.identity = dict(self.base_identity, config_sha256=hashlib.sha256(
            canonical([self.base_identity["config_sha256"], self.execution])).hexdigest())
        self.key = hashlib.sha256(canonical([self.identity, self.prefix])).hexdigest()
        if message.get("temp", 0) != 0:
            raise ValueError("conversation state reuse currently requires greedy temperature zero")
        entry = self.owner.entries.get(self.key)
        if entry is not None and self.raw_prefix is not None and entry.get("raw_prefix") not in (None, self.raw_prefix):
            # A buggy caller cannot hide differing IDs behind one opaque binding.
            entry = None
        if entry is None:
            self._try_extended_shadow(pipe, ret, send, recv)
            return
        if entry["expires"] <= time.monotonic():
            self.owner.entries.pop(self.key, None)
            self._try_extended_shadow(pipe, ret, send, recv)
            return
        if entry.get("first_token") in self.eos_ids:
            self.record["reason"] = "cached_eos_requires_real_prefill_receipt"
            return
        tx = secrets.token_hex(16)
        rows = self._control(pipe, ret, send, recv, "restore_prepare", tx, entry["entries"])
        if all(row["ok"] is True for row in rows):
            rows = self._control(pipe, ret, send, recv, "restore_commit", tx, entry["entries"])
            if all(row["ok"] is True and row["entry_digest"] == entry["digests"][str(row["stage"])] for row in rows):
                reply = rows[-1]["tail_reply"]
                if not isinstance(reply, dict) or "token" not in reply:
                    raise ValueError("cached prompt has no bound tail result")
                self.pending_reply = {**reply, **self.job}
                self.owner.entries.move_to_end(self.key)
                self.record.update(hit=True, reason="all_stages_restored", transaction_id=tx)
                return
        self._control(pipe, ret, send, recv, "abort", tx, entry["entries"])
        self.owner.entries.pop(self.key, None)
        send(pipe, message)
        reset = recv(ret)
        if not isinstance(reset, dict) or reset.get("op") != "reset_ok" or not reset.get("ok"):
            raise RuntimeError("full-prefill fallback reset failed")
        if any(reset.get(k) != v for k, v in self.job.items()):
            raise RuntimeError("fallback reset belongs to another job")
        self.record["reason"] = "restore_miss_full_prefill"

    def _native_reset(self, pipe, ret, send, recv):
        self._guard()
        send(pipe, self.reset_message)
        reset = recv(ret)
        self._guard()
        if (not isinstance(reset, dict) or reset.get("op") != "reset_ok" or reset.get("ok") is not True
                or any(reset.get(k) != v for k, v in self.job.items())):
            raise RuntimeError("reference shadow/full-prefill reset barrier failed")

    def _try_extended_shadow(self, pipe, ret, send, recv):
        if not self.owner.extended_shadow or self.raw_prefix is None:
            return
        now = time.monotonic()
        same_mode = {k: self.execution[k] for k in ("spec", "dspark", "pipelined", "temp", "seed")}
        parents = [entry for entry in self.owner.entries.values() if
            entry.get("base_identity") == self.base_identity and entry["expires"] > now
            and entry.get("raw_prefix") is not None and 0 < len(entry["raw_prefix"]) < len(self.raw_prefix)
            and self.raw_prefix[:len(entry["raw_prefix"])] == entry["raw_prefix"]
            and {k: entry.get("execution", {}).get(k) for k in same_mode} == same_mode]
        if not parents:
            return
        parent = max(parents, key=lambda entry: len(entry["raw_prefix"]))
        tx, candidate = secrets.token_hex(16), {}
        parent_count = len(parent["raw_prefix"])
        self.record.update(mode="extended_prefix_reference_shadow", reference_shadow=True,
            parent_prefix_tokens=parent_count, extra_suffix_steps=0, all_stage_shadow_passed=False,
            scope="per-request candidate/full-prefill comparison; current request saves no full prefill")
        self.shadow = {"transaction_id": tx, "candidate": candidate}
        try:
            rows = self._control(pipe, ret, send, recv, "restore_prepare", tx, parent["entries"],
                identity=parent["identity"], prefix=parent["prefix"])
            if not all(row["ok"] is True for row in rows):
                raise _OptionalShadowMiss("parent_prepare_miss")
            rows = self._control(pipe, ret, send, recv, "restore_commit", tx, parent["entries"],
                identity=parent["identity"], prefix=parent["prefix"])
            if not all(row["ok"] is True and row["entry_digest"] == parent["digests"][str(row["stage"])] for row in rows):
                raise _OptionalShadowMiss("parent_restore_miss")
            rows = self._control(pipe, ret, send, recv, "shadow_begin", tx, {}, parameters={
                "full_prefix_count": len(self.raw_prefix), "parent_count": parent_count})
            if not all(row["ok"] is True for row in rows):
                raise _OptionalShadowMiss("shadow_workspace_or_state_refused")
            for pos in range(parent_count, len(self.raw_prefix)):
                self._guard()
                send(pipe, {"op": "step", "ids": [[self.raw_prefix[pos]]], "start_pos": pos,
                            "conversation_shadow": tx})
                reply = recv(ret)
                self._guard()
                if not isinstance(reply, dict) or any(reply.get(k) != v for k, v in self.job.items()):
                    raise ValueError("shadow suffix reply belongs to another job")
                self.record["extra_suffix_steps"] += 1
            rows = self._control(pipe, ret, send, recv, "shadow_finish", tx, {}, parameters={})
            if not all(row["ok"] is True for row in rows):
                raise _OptionalShadowMiss("shadow_finish_refused")
            rows = self._control(pipe, ret, send, recv, "capture", tx, {})
            candidate.update({str(row["stage"]): row["entry_id"] for row in rows if row["entry_id"]})
            if not all(row["ok"] is True for row in rows):
                raise _OptionalShadowMiss("shadow_candidate_quota_refused")
            self._native_reset(pipe, ret, send, recv)
            self.record["reason"] = "awaiting_original_full_prefill_shadow"
        except _OptionalShadowMiss as error:
            self._control(pipe, ret, send, recv, "shadow_abort", tx, {})
            if candidate:
                self._control(pipe, ret, send, recv, "capture_abort", tx, candidate)
            # Release uncommitted parent restore tickets too. Do not discard the
            # stable parent checkpoint merely because a new shadow quota failed.
            self._control(pipe, ret, send, recv, "abort", tx, {})
            self._native_reset(pipe, ret, send, recv)
            self.shadow = None
            self.record["reason"] = str(error) + "_original_full_prefill"

    def before_send(self, message):
        if self.enabled and message.get("op") == "step" and message.get("start_pos") == 0:
            if self.pending_reply is not None:
                return False
            self.prefill_waiting = True
        return True

    def cached_reply(self):
        reply, self.pending_reply = self.pending_reply, None
        return reply

    def after_prefill(self, pipe, ret, send, recv, reply):
        if not self.enabled or not self.prefill_waiting:
            return
        self.prefill_waiting = False
        shadow = self.shadow
        if shadow is not None:
            rows = self._control(pipe, ret, send, recv, "shadow_compare", shadow["transaction_id"],
                shadow["candidate"], parameters={})
            passed = all(row["ok"] is True and row.get("shadow_passed") is True for row in rows)
            self._control(pipe, ret, send, recv, "capture_abort", shadow["transaction_id"], shadow["candidate"])
            self.shadow = None
            self.record.update(all_stage_shadow_passed=passed,
                shadow_result="passed" if passed else "state_or_reply_mismatch_reference_kept")
        tx = secrets.token_hex(16)
        rows = self._control(pipe, ret, send, recv, "capture", tx, {})
        entries = {str(row["stage"]): row["entry_id"] for row in rows if row["entry_id"]}
        if all(row["ok"] is True for row in rows):
            self.owner.entries[self.key] = {"entries": entries,
                "digests": {str(row["stage"]): row["entry_digest"] for row in rows},
                "expires": time.monotonic() + self.owner.ttl_s, "identity": deepcopy(self.identity),
                "prefix": deepcopy(self.prefix), "base_identity": deepcopy(self.base_identity),
                "execution": deepcopy(self.execution), "raw_prefix": self.raw_prefix,
                "first_token": reply.get("token")}
            self.owner.entries.move_to_end(self.key)
            while len(self.owner.entries) > self.owner.max_entries:
                self.owner.entries.popitem(last=False)
            self.record.update(captured=True, reason="all_stages_captured")
            if shadow is not None:
                self.record["reason"] = "shadow_checked_original_full_prefill_retained"
        elif entries:
            self._control(pipe, ret, send, recv, "capture_abort", tx, entries)
            self.record["reason"] = "capture_capacity_miss_normal_prefill"


class _OptionalShadowMiss(RuntimeError):
    """A known local policy/capacity miss; authenticated transport errors propagate."""
