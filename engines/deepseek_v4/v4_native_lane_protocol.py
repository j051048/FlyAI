"""Strict signed control and independent receipts for native scalar lanes."""
import hashlib
import time

import torch

from v4_conversation_protocol import _sign, _verify, canonical
from v4_native_lanes import CONTRACT
from shard.receipt import ReceiptSigner, pub_b64

SCHEMA = "v4-native-lane/1"
DOMAIN = b"v4-native-lane/1\0"
ACK_DOMAIN = b"v4-native-lane-ack/1\0"


def runtime_binding(config):
    # The resource lease uses these same two deployment-path exclusions. All
    # numerical knobs, source hashes, state reservations and runtime versions stay.
    body = dict(config)
    body["environment"] = {k: v for k, v in body.get("environment", {}).items()
                           if k not in ("V4_DIR", "V4_DEV")}
    return hashlib.sha256(canonical(body)).hexdigest()


def _job(value):
    if not isinstance(value, dict) or set(value) != {"job_id", "nonce", "swarm_id"}:
        raise ValueError("complete native job binding required")
    if any(not isinstance(v, str) or not 1 <= len(v) <= 256 for v in value.values()):
        raise ValueError("invalid native job identity")
    if len(value["nonce"]) != 64 or any(c not in "0123456789abcdef" for c in value["nonce"]):
        raise ValueError("fresh native job nonce required")
    return value


def handle(st, message, *, stage_index, session, signer_key, vp, owner_grant=None):
    """Execute a signed scalar operation and return a forwardable frame.

    Every operation returns a fresh per-stage signed acknowledgement. A failed
    stage prevents downstream forward execution; the coordinator drains this
    barrier before releasing affected lanes or rejecting the ring.
    """
    if session is None or signer_key is None:
        raise ValueError("native lanes require strict authenticated sessions")
    request = _verify(session.plan["coordinator"]["signer_pubkey"], DOMAIN, message["request"])
    fields = {"schema", "contract", "action", "request_id", "job", "identity_sha256",
              "owner_epoch", "cohort_id", "horizon", "parameters"}
    if set(request) != fields or request["schema"] != SCHEMA or request["contract"] != CONTRACT:
        raise ValueError("invalid native lane request contract")
    if message.get("op") == "reset":
        if message.get("native_lane_open") is not True or request["action"] != "open":
            raise ValueError("native reset barrier only permits a signed open")
    elif message.get("op") != "native_lane" or request["action"] == "open":
        raise ValueError("native open must use the strict owner reset barrier")
    job = _job(request["job"])
    if job["swarm_id"] != session.plan["ring_id"] or request["cohort_id"] != session.plan["cohort_id"]:
        raise ValueError("native lane request belongs to another ring/cohort")
    if (not isinstance(request["request_id"], str) or len(request["request_id"]) != 32
            or not isinstance(request["parameters"], dict)):
        raise ValueError("invalid native lane transaction")
    runner = getattr(st, "_native_lanes", None)
    if runner is None:
        raise ValueError("node was not calibrated with native lanes")
    if (not owner_grant or request["owner_epoch"] !=
            f"{owner_grant['boot_id']}:{owner_grant['fence']}"):
        raise ValueError("native request owner epoch differs from the authenticated live connection")
    runner.bind_epoch(request["owner_epoch"])
    if getattr(st, "_native_runtime_binding", None) is None:
        from v4_resources import runtime_config_payload
        st._native_runtime_binding = runtime_binding(runtime_config_payload(st))
    if runner.store is not None:
        row = session.plan["stages"][stage_index]
        # The artifact inventory can describe thousands of tensors. Hash it once
        # per loaded stage, never serialize that full catalogue per token/write.
        bound = getattr(st, "_native_storage_runtime", None)
        if bound is None:
            bound = st._native_storage_runtime = {
                "session_contract": session.contract_digest, "cohort_id": session.plan["cohort_id"],
                "runtime_config_sha256": row.get("runtime_config_sha256"),
                "stage": {key: row[key] for key in ("node_id", "lo", "hi", "signer_pubkey")},
                "artifacts_sha256": hashlib.sha256(canonical(getattr(st, "_artifact_verification", None))).hexdigest()}
        if bound["session_contract"] != session.contract_digest or bound["runtime_config_sha256"] != row.get("runtime_config_sha256"):
            raise ValueError("durable native loaded stage identity changed")
        runner.bind_storage_identity(bound)
    contexts = getattr(st, "_native_lane_contexts", None)
    if contexts is None or getattr(st, "_native_lane_epoch", None) != request["owner_epoch"]:
        contexts = st._native_lane_contexts = {}
        st._native_lane_epoch = request["owner_epoch"]
    action, params, key = request["action"], request["parameters"], job["job_id"]
    result = {}
    try:
        if message.get("lane_error"):
            raise ValueError("an earlier stage rejected this native operation")
        guard = getattr(st, "_conversation_lease_guard", None)
        if guard is not None:
            guard.assert_live()
        vp._set_job_horizon(request["horizon"])
        if action == "open":
            if set(params) - {"restore"} or key in contexts:
                raise ValueError("invalid or duplicate native lane open")
            restore = params.get("restore")
            entry = None
            if restore:
                if set(restore) != {"transaction_id", "prefix", "identity_sha256", "stages"}:
                    raise ValueError("invalid native restore binding")
                rows = restore["stages"]
                if len(rows) != session.plan["nstages"]:
                    raise ValueError("native restore omits a stage")
                row = rows[stage_index]
                if row["stage"] != stage_index or row["pubkey"] != pub_b64(signer_key):
                    raise ValueError("native restore names another stage")
                if restore["identity_sha256"] != request["identity_sha256"]:
                    raise ValueError("native restore identity changed")
                entry = row["entry_id"]
            started = time.perf_counter()
            cached = runner.open(key, identity=request["identity_sha256"],
                                 horizon=request["horizon"], restore_entry=entry)
            note = None
            if restore:
                if cached["prefix"] != restore["prefix"] or cached["state_digest"] != row["state_digest"]:
                    runner.close(key)
                    raise ValueError("native prefix content/frontier differs")
                note = {"schema": "v4-conversation-restore/1", "transaction_id": restore["transaction_id"],
                        "prefix": restore["prefix"], "identity_sha256": restore["identity_sha256"],
                        "entry_id": entry, "state_digest": cached["state_digest"],
                        "scope": "restored prefix state; receipt chunks count only new forwards"}
            contexts[key] = {"job": job, "identity": request["identity_sha256"],
                             "signer": ReceiptSigner(signer_key, job["swarm_id"], key, st.lo, st.hi,
                                                     nonce=job["nonce"]),
                             "restore": note, "receipted": False}
            result = {"pos": runner.active[key]["pos"], "restored": cached is not None,
                      "restore_tier": runner.active[key]["restore_tier"],
                      "runtime_binding_sha256": st._native_runtime_binding,
                      "restore_seconds": time.perf_counter() - started if cached else None,
                      "capability": runner.config()}
        elif action == "close" and key not in contexts:
            # Cleanup is idempotent after a partially successful open barrier.
            result = {"released": True, "cache": None}
        else:
            ctx = contexts.get(key)
            if ctx is None or ctx["job"] != job or ctx["identity"] != request["identity_sha256"]:
                raise ValueError("native operation does not own this lane")
            if action == "step":
                if set(params) != {"pos", "token"} or ctx["receipted"]:
                    raise ValueError("invalid native forward or already finalized lane")
                token, pos = params["token"], params["pos"]
                if type(token) is not int or not 0 <= token < st.args.vocab_size:
                    raise ValueError("native token exceeds the model vocabulary")
                runner.activate(key, pos)
                ids = torch.tensor([[token]], dtype=torch.long, device=st.device)
                if st.head:
                    h = st.embed(ids)
                    incoming = vp._payload_bytes(h, ids)
                else:
                    h, received, incoming = vp._recv_hids(message, ctx["signer"], device=st.device)
                    if received.shape != ids.shape or not torch.equal(received.to(ids.device), ids):
                        raise ValueError("native input token changed between authenticated stages")
                with torch.no_grad():
                    h = st.forward(h, ids, pos)
                runner.finish_step(key)
                frame, outgoing = vp._make_step_frame(h, ids, pos, ctx["signer"])
                ctx["signer"].observe(incoming, outgoing)
                result = {"pos": pos + 1}
                if st.tail:
                    result["token"] = int(st.logits_all(h, full_logits=False).reshape(-1, st.args.vocab_size)[-1].argmax())
                else:
                    message = {**frame, "op": "native_lane", "request": message["request"],
                               "acks": message.get("acks", [])}
            elif action == "receipt":
                if params or ctx["receipted"]:
                    raise ValueError("native receipt may be finalized exactly once")
                from v4_observability import runtime_observation
                observation = runtime_observation(st, session_config=session, phase="job_complete")
                result = {"receipt": ctx["signer"].finalize(runtime_observation=observation,
                                                          conversation_restore=ctx["restore"])}
                ctx["receipted"] = True
            elif action == "close":
                if set(params) != {"retain", "prefix"} or type(params["retain"]) is not bool:
                    raise ValueError("invalid native release fields")
                if params["retain"] and not ctx["receipted"]:
                    raise ValueError("unverified incomplete work may not become a conversation prefix")
                entry = runner.close(key, retain=params["retain"], prefix=params["prefix"])
                contexts.pop(key)
                result = {"released": True, "cache": ({k: v for k, v in entry.items() if k != "expires"}
                                                       if entry else None),
                          "cache_write_error": runner.last_cache_write_error}
            else:
                raise ValueError("unknown native lane action")
        ok = True
    except (ValueError, RuntimeError, KeyError, TypeError) as error:
        message["lane_error"] = True
        result, ok = {"error": str(error)[:256]}, False
    ack = {"schema": SCHEMA, "request_id": request["request_id"], "request_sha256": hashlib.sha256(canonical(request)).hexdigest(),
           "job": job, "stage": stage_index, "pubkey": pub_b64(signer_key), "ok": ok,
           "contract": CONTRACT, "result": result}
    message.setdefault("acks", []).append(_sign(signer_key, ACK_DOMAIN, ack))
    return message
