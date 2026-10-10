"""Signed cache control barriers; real tiny strict-ring tests are below."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("cryptography")
P = pytest.importorskip("v4_conversation_protocol")
from shard.receipt import gen_key, pub_b64
from v4_conversation_cache import ConversationIdentity, PrefixBinding


JOB = {"job_id": "request-1", "nonce": "11" * 32, "swarm_id": "test-swarm"}
PREFIX = PrefixBinding(5, "2" * 64).to_dict()


class RecordingCache:
    """Only control sequencing is mocked; cryptographic signatures are real."""
    enabled = True
    def __init__(self, index, events):
        self.index, self.events = index, events
        self.entries, self.tickets = {}, {}
        self.position, self.drafter = 5, None
        self.fail_capture = self.fail_prepare = self.fail_commit = False
        self.resets = 0
        self.quotas = SimpleNamespace(max_entries=4)
    def bind_drafter(self, value):
        self.drafter = value
    def capture(self, identity, prefix, **kwargs):
        self.events.append((self.index, "capture"))
        if self.fail_capture:
            raise RuntimeError("quota refused")
        key = f"entry-{self.index}"
        self.entries[key] = (identity, prefix, deepcopy(kwargs["tail_reply"]))
        return key
    def entry_digest(self, key):
        assert key in self.entries
        return "3" * 64
    def prepare_restore(self, identity, prefix, *, entry_id=None):
        self.events.append((self.index, "prepare"))
        if self.fail_prepare or entry_id not in self.entries:
            return None
        assert self.entries[entry_id][:2] == (identity, prefix)
        ticket = {"ticket": f"ticket-{self.index}", "entry_id": entry_id,
                  "entry_content_sha256": "3" * 64, "state_digest": "4" * 64}
        self.tickets[ticket["ticket"]] = ticket
        return ticket
    def commit_restore(self, ticket):
        self.events.append((self.index, "commit"))
        if self.fail_commit:
            raise RuntimeError("copy failed")
        self.tickets.pop(ticket["ticket"])
        self.position = 5
        return {"entry_id": ticket["entry_id"], "entry_content_sha256": "3" * 64,
                "state_digest": "4" * 64, "restored_state_sha256": "4" * 64,
                "tail_reply": self.entries[ticket["entry_id"]][2]}
    def abort_restore(self, ticket):
        if isinstance(ticket, dict) and "ticket" in ticket:
            self.tickets.pop(ticket["ticket"], None)
    def invalidate(self, key=None):
        self.events.append((self.index, "invalidate"))
        if key is None:
            self.entries.clear()
        else:
            self.entries.pop(key, None)
    def reset_current(self):
        self.events.append((self.index, "reset"))
        self.resets += 1
        self.position = 0


@pytest.fixture
def ring():
    keys = [gen_key(), gen_key()]
    coordinator_key = gen_key()
    plan = {"ring_id": "test-ring", "cohort_id": "a" * 64, "nstages": 2,
            "coordinator": {"signer_pubkey": pub_b64(coordinator_key)},
            "stages": [{"node_id": f"node-{i}", "signer_pubkey": pub_b64(key)}
                       for i, key in enumerate(keys)]}
    config = SimpleNamespace(plan=plan, caller_key=coordinator_key,
                             stage=lambda i: plan["stages"][i])
    events = []
    stages = [SimpleNamespace(tail=i == 1, _conversation_cache=RecordingCache(i, events),
        _conversation_job_binding=dict(JOB), _conversation_session_epoch="boot:1",
        _conversation_restore_note=None, _conversation_prefill_reply={"token": 17, "acc": True})
        for i in range(2)]
    identity = ConversationIdentity("tenant-hmac", "a" * 64, "bit-native/1", "tensor-sha256:" + "b" * 64,
        "c" * 64, "d" * 64, "e" * 64, "test-ring", "boot:1", (("node-0", 1), ("node-1", 1))).to_dict()
    return SimpleNamespace(keys=keys, cfg=config, stages=stages, identity=identity, events=events)


def command(ring, action="capture", transaction="transaction-1", *, entries=None, identity=None, prefix=None, job=None):
    body = {"schema": P.CONTROL, "action": action, "transaction_id": transaction,
            "identity": deepcopy(identity or ring.identity), "prefix": deepcopy(prefix or PREFIX),
            "entries": entries or {}, "job": dict(job or JOB), "execution": P.execution_binding({})}
    return {"op": "conversation_control", "request": P._sign(ring.cfg.caller_key,
        b"v4-conversation-control/1\0", body), "acks": []}


def ack(ring, message, index=0):
    signed = P.handle_stage_control(ring.stages[index], message, stage_index=index,
        session=ring.cfg, signer_key=ring.keys[index])
    return P._verify(pub_b64(ring.keys[index]), b"v4-conversation-ack/1\0", signed)


def test_stage_capture_is_signed_and_commits_opaque_identity(ring):
    evidence = ack(ring, command(ring))
    assert evidence["ok"] is True and evidence["entry_id"] == "entry-0"
    assert evidence["prefix"] == PREFIX
    assert evidence["entry_digest"] == "3" * 64
    assert evidence["job"] == JOB
    assert "ids" not in str(evidence)
    import hashlib
    assert evidence["identity_sha256"] == hashlib.sha256(P.canonical(ring.identity)).hexdigest()


@pytest.mark.parametrize("mutation", [
    lambda message: message["request"].update(action="restore_commit"),
    lambda message: message["request"]["identity"].update(tenant="other-tenant"),
    lambda message: message["request"]["job"].update(nonce="22" * 32),
    lambda message: message["request"]["prefix"].update(digest="f" * 64),
])
def test_tampered_coordinator_command_never_mutates_cache(ring, mutation):
    message = command(ring)
    mutation(message)
    with pytest.raises(Exception):
        ack(ring, message)
    assert ring.events == []


def test_freshly_signed_wrong_job_or_epoch_and_lease_fence_reject(ring):
    with pytest.raises(ValueError):
        ack(ring, command(ring, job=dict(JOB, nonce="22" * 32)))
    with pytest.raises(ValueError):
        ack(ring, command(ring, identity=dict(ring.identity, ring_epoch="old-boot:1")))
    with pytest.raises(ValueError):
        ack(ring, command(ring, identity=dict(ring.identity, lease_fences=[["node-0", 2], ["node-1", 1]])))
    assert ring.events == []


@pytest.mark.parametrize("mutation", [
    lambda identity, prefix, job: identity.update(tenant="different-tenant"),
    lambda identity, prefix, job: prefix.update(digest="f" * 64),
])
def test_commit_must_match_prepared_transaction_identity(ring, mutation):
    entries = {"0": "entry-0"}
    assert ack(ring, command(ring))["ok"]
    assert ack(ring, command(ring, "restore_prepare", entries=entries))["ok"]
    identity, prefix, job = deepcopy(ring.identity), dict(PREFIX), dict(JOB)
    mutation(identity, prefix, job)
    rejected = ack(ring, command(ring, "restore_commit", entries=entries, identity=identity, prefix=prefix, job=job))
    assert rejected["ok"] is False
    assert (0, "commit") not in ring.events


def test_capture_abort_preserves_completed_prefill_state(ring):
    assert ack(ring, command(ring))["ok"]
    evidence = ack(ring, command(ring, "capture_abort", entries={"0": "entry-0"}))
    assert evidence["ok"] is True
    assert ring.stages[0]._conversation_cache.position == 5
    assert ring.stages[0]._conversation_cache.resets == 0
    assert not ring.stages[0]._conversation_cache.entries


class SimulatedWire:
    def __init__(self, ring):
        self.ring = ring
        self.pipe = SimpleNamespace(config=ring.cfg)
        self.responses = []
        self.corrupt_stage = None
        self.wrong_identity_stage = None
    def send(self, channel, message):
        if message.get("op") == "reset":
            for stage in self.ring.stages:
                stage._conversation_cache.reset_current()
                stage._conversation_job_binding = {k: message[k] for k in JOB}
                stage._conversation_reset_parameters = dict(message)
            self.responses.append({"op": "reset_ok", "ok": True, **{k: message[k] for k in JOB}})
            return
        assert message["op"] == "conversation_control"
        rows = [P.handle_stage_control(stage, message, stage_index=i, session=self.ring.cfg,
            signer_key=self.ring.keys[i]) for i, stage in enumerate(self.ring.stages)]
        if self.corrupt_stage is not None:
            rows[self.corrupt_stage]["ok"] = not rows[self.corrupt_stage]["ok"]
        if self.wrong_identity_stage is not None:
            index = self.wrong_identity_stage
            body = {k: v for k, v in rows[index].items() if k != "signature"}
            body["identity_sha256"] = "f" * 64
            rows[index] = P._sign(self.ring.keys[index], b"v4-conversation-ack/1\0", body)
        self.responses.append({"op": "conversation_ack", "job": deepcopy(message["request"]["job"]), "acks": rows})
    def recv(self, channel):
        return self.responses.pop(0)


def request(ring, owner=None):
    first = owner is None
    owner = owner or P.ConversationCoordinator(enabled=True)
    req = owner.request(ring.identity, PREFIX, max_new=3)
    req.job = dict(JOB)
    if first:
        req.after_reset(SimpleNamespace(config=ring.cfg), object(), lambda *_: None, lambda *_: None,
                        {"op": "reset", **JOB})
    return req


def capture_ring(ring, wire, req):
    req.prefill_waiting = True
    req.after_prefill(wire.pipe, object(), wire.send, wire.recv, {"token": 17, **JOB})


def test_coordinator_prepare_all_then_commit_all_with_fresh_job_binding(ring):
    wire, req = SimulatedWire(ring), request(ring)
    capture_ring(ring, wire, req)
    ring.events.clear()
    new = request(ring, req.owner)
    fresh = dict(JOB, nonce="33" * 32, job_id="request-2")
    for stage in ring.stages:
        stage._conversation_job_binding = dict(fresh)
        stage._conversation_cache.position = 0
    new.after_reset(wire.pipe, object(), wire.send, wire.recv, {"op": "reset", **fresh})
    assert ring.events == [(0, "prepare"), (1, "prepare"), (0, "commit"), (1, "commit")]
    assert new.record["hit"] is True
    assert new.cached_reply() == {"token": 17, "acc": True, **fresh}
    for stage in ring.stages:
        assert stage._conversation_restore_note["scope"].startswith("restored prefix state")


@pytest.mark.parametrize("failure", ["prepare", "commit"])
def test_partial_restore_resets_all_stages_and_falls_back(ring, failure):
    wire, req = SimulatedWire(ring), request(ring)
    capture_ring(ring, wire, req)
    setattr(ring.stages[1]._conversation_cache, "fail_" + failure, True)
    following = request(ring, req.owner)
    following.after_reset(wire.pipe, object(), wire.send, wire.recv, {"op": "reset", **JOB})
    assert following.pending_reply is None and following.record["hit"] is False
    assert following.record["reason"] == "restore_miss_full_prefill"
    assert all(stage._conversation_cache.position == 0 for stage in ring.stages)
    assert not req.owner.entries


def test_partial_capture_is_optional_and_preserves_prefill(ring):
    wire, req = SimulatedWire(ring), request(ring)
    ring.stages[1]._conversation_cache.fail_capture = True
    capture_ring(ring, wire, req)
    assert req.record["hit"] is False and not req.owner.entries
    assert all(stage._conversation_cache.position == 5 for stage in ring.stages)
    assert all(stage._conversation_cache.resets == 0 for stage in ring.stages)
    assert not ring.stages[0]._conversation_cache.entries


def test_coordinator_rejects_tampered_signed_stage_evidence(ring):
    wire, req = SimulatedWire(ring), request(ring)
    wire.corrupt_stage = 1
    with pytest.raises(Exception):
        capture_ring(ring, wire, req)
    assert not req.owner.entries


def test_valid_stage_signature_with_wrong_identity_is_not_publishable(ring):
    wire, req = SimulatedWire(ring), request(ring)
    wire.wrong_identity_stage = 1
    with pytest.raises(ValueError):
        capture_ring(ring, wire, req)
    assert not req.owner.entries


def test_one_token_job_and_default_disabled_do_not_restore_cache(ring):
    assert request(ring).owner.request(ring.identity, PREFIX, max_new=1).enabled is False
    assert P.ConversationCoordinator().request(ring.identity, PREFIX, max_new=3).enabled is False


@pytest.mark.parametrize("ttl", [float("nan"), float("inf"), True, 0, -1])
def test_coordinator_ttl_is_finite_positive_and_never_boolean(ttl):
    with pytest.raises(ValueError):
        P.ConversationCoordinator(ttl_s=ttl)


class NumericalWire:
    """Real CPU V4 state/model; synchronous transport for gate unit tests."""
    def __init__(self, *, draft=False, quota_bytes=20_000_000):
        import torch
        import v4_ref_cpu as R
        import v4_dspark_draft as DS
        from test_v4_conversation_cache import stage
        from v4_conversation_cache import StageConversationCache, CacheQuotas
        self.torch, self.draft_enabled = torch, draft
        args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
            dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
        self.oracle = R.build_oracle(args, 7)
        self.stage = stage((args, self.oracle), draft=draft)
        self.tail = DS.DSparkTail(self.stage) if draft else None
        if draft:
            for i, layer in enumerate(self.tail.mtp):
                layer.load_state_dict({k: v for k, v in self.oracle.mtp[i].state_dict().items()
                                      if k not in DS.ALIAS_KEYS}, strict=False)
        self.drafter = DS.RingDrafter(self.tail) if draft else None
        self.stage._conversation_cache = StageConversationCache(self.stage, CacheQuotas(quota_bytes),
                                                               enabled=True, drafter=self.drafter)
        self.stage._conversation_session_epoch = "boot:1"
        self.stage._conversation_pending = {}
        self.key, self.caller_key = gen_key(), gen_key()
        plan = {"ring_id": "test-ring", "cohort_id": "a" * 64, "nstages": 1,
                "coordinator": {"signer_pubkey": pub_b64(self.caller_key)},
                "stages": [{"node_id": "node-0", "signer_pubkey": pub_b64(self.key)}]}
        self.config = SimpleNamespace(plan=plan, caller_key=self.caller_key, stage=lambda i: plan["stages"][i])
        self.pipe = SimpleNamespace(config=self.config)
        self.responses, self.control_messages = [], []
        self.normal_prefills = self.shadow_steps = 0
    def send(self, channel, message):
        st, torch = self.stage, self.torch
        if message["op"] == "reset":
            st._conversation_job_binding = {k: message[k] for k in JOB}
            st._conversation_reset_parameters = dict(message)
            P.reset_shadow(st)
            for prepared in st._conversation_pending.values():
                st._conversation_cache.abort_restore(prepared["ticket"])
            st._conversation_pending = {}
            st.reset()
            st._spec = bool(message.get("spec", False))
            st._dspark = bool(message.get("dspark", False))
            if self.tail is not None:
                import v4_dspark_draft as DS
                self.tail.reset()
                self.drafter = DS.RingDrafter(self.tail)
                self.drafter.pipelined = bool(message.get("pipelined", False))
            self.responses.append({"op": "reset_ok", "ok": True, **st._conversation_job_binding})
        elif message["op"] == "conversation_control":
            self.control_messages.append(deepcopy(message))
            evidence = P.handle_stage_control(st, message, stage_index=0, session=self.config,
                signer_key=self.key, drafter=self.drafter)
            self.responses.append({"op": "conversation_ack", "job": dict(st._conversation_job_binding),
                                   "acks": [evidence]})
        else:
            assert message["op"] == "step"
            marked = P.shadow_before_step(st, message)
            ids = torch.tensor(message["ids"], dtype=torch.long)
            try:
                with torch.no_grad():
                    hidden = st.forward(st.embed(ids), ids, message["start_pos"])
            finally:
                P.shadow_after_step(st, message)
            with torch.no_grad():
                logits = st.logits_all(hidden, full_logits=bool(st._spec))
                tokens = logits.argmax(-1).reshape(-1).tolist()
            out = {"token": int(tokens[-1])}
            if st._spec:
                out["tokens"] = tokens
            if self.drafter is not None and not marked:
                out.update(self.drafter.on_chunk(message, st, out))
            P.shadow_after_step(st, message, out=out)
            if message["start_pos"] == 0:
                st._conversation_prefill_reply = {k: deepcopy(v) for k, v in out.items() if k in P.MATH_REPLY}
                self.normal_prefills += 1
            if marked:
                self.shadow_steps += 1
            self.responses.append({**out, **st._conversation_job_binding})
    def recv(self, channel):
        return self.responses.pop(0)


def numerical_identity():
    return ConversationIdentity("tenant-hmac", "a" * 64, "bit-native/1", "tiny-cpu-fixture",
        "c" * 64, "d" * 64, "e" * 64, "test-ring", "boot:1", (("node-0", 1),)).to_dict()


def numerical_prefill(wire, owner, ids, number, *, maximum=4, eos_ids=()):
    import hashlib
    import hmac
    opaque = PrefixBinding(len(ids), hmac.new(b"synthetic-test-tenant-secret", P.canonical(ids), hashlib.sha256).hexdigest()).to_dict()
    request = owner.request(numerical_identity(), opaque, max_new=maximum)
    request.bind_tokens(ids, eos_ids=eos_ids)
    reset = {"op": "reset", "job_id": f"job-{number}", "nonce": f"{number:064x}",
        "swarm_id": "unit-swarm", "max_pos": len(ids) + maximum,
        "spec": wire.draft_enabled, "dspark": wire.draft_enabled, "pipelined": wire.draft_enabled}
    wire.send(wire.pipe, reset)
    wire.recv(None)
    request.after_reset(wire.pipe, None, wire.send, wire.recv, reset)
    step = {"op": "step", "ids": [ids], "start_pos": 0}
    if request.before_send(step):
        wire.send(wire.pipe, step)
        reply = wire.recv(None)
        request.after_prefill(wire.pipe, None, wire.send, wire.recv, reply)
    else:
        reply = request.cached_reply()
    return reply, request


@pytest.mark.parametrize("draft", [False, True])
def test_real_v4_extended_suffix_shadow_rebuilds_taps_and_keeps_original_prefill(draft):
    from v4_conversation_cache import _buffers, _metadata, _same
    wire, reference = NumericalWire(draft=draft), NumericalWire(draft=draft)
    owner = P.ConversationCoordinator(enabled=True, extended_shadow=True)
    numerical_prefill(wire, owner, list(range(5)), 1)
    observed, request = numerical_prefill(wire, owner, list(range(13)), 2)
    expected, _ = numerical_prefill(reference, P.ConversationCoordinator(), list(range(13)), 2)
    assert {k: v for k, v in observed.items() if k in P.MATH_REPLY} == {
        k: v for k, v in expected.items() if k in P.MATH_REPLY}
    assert _same(_buffers(wire.stage, wire.drafter), _buffers(reference.stage, reference.drafter))
    assert _same(_metadata(wire.stage, wire.drafter), _metadata(reference.stage, reference.drafter))
    assert request.record["reference_shadow"] is True and request.record["hit"] is False
    assert request.record["extra_suffix_steps"] == wire.shadow_steps == 8
    assert type(request.record["all_stage_shadow_passed"]) is bool
    assert wire.normal_prefills == 2  # Candidate has not replaced the reference prefill.
    assert wire.stage._conversation_shadow is None and wire.stage._replaying is False
    assert all("ids" not in message["request"] for message in wire.control_messages)
    again, repeated = numerical_prefill(wire, owner, list(range(13)), 3)
    assert repeated.record["hit"] is True and wire.normal_prefills == 2
    assert again["token"] == expected["token"]


def test_cached_eos_and_different_job_horizon_use_real_prefill():
    wire = NumericalWire()
    owner = P.ConversationCoordinator(enabled=True)
    first, _ = numerical_prefill(wire, owner, list(range(5)), 1)
    eos, request = numerical_prefill(wire, owner, list(range(5)), 2, eos_ids=(first["token"],))
    assert request.record["hit"] is False and wire.normal_prefills == 2
    assert eos["token"] == first["token"]
    longer, request = numerical_prefill(wire, owner, list(range(5)), 3, maximum=10)
    assert request.record["hit"] is False and wire.normal_prefills == 3
    assert longer["token"] == first["token"]


def test_extended_shadow_workspace_refusal_resets_then_original_prefill():
    wire = NumericalWire(draft=True)
    owner = P.ConversationCoordinator(enabled=True, extended_shadow=True)
    numerical_prefill(wire, owner, list(range(5)), 1)
    # Keep the snapshot but leave no extra quota for tap concatenation.
    from dataclasses import replace
    cache = wire.stage._conversation_cache
    charge = cache.status()["host_bytes"]
    cache.quotas = replace(cache.quotas, host_bytes=charge, tenant_host_bytes=charge)
    reply, request = numerical_prefill(wire, owner, list(range(13)), 2)
    assert request.record["hit"] is False and request.record["reference_shadow"] is True
    assert request.record["extra_suffix_steps"] == 0 and wire.normal_prefills == 2
    assert wire.stage._pos == 13 and wire.stage._replaying is False
    assert wire.stage._conversation_shadow is None and "token" in reply


@pytest.fixture
def strict_tiny_ring(tmp_path, monkeypatch, request):
    """Real V4 loader, sockets, HELLO owner grants and signed receipt chain."""
    import threading
    import traceback
    torch = pytest.importorskip("torch")
    R = pytest.importorskip("v4_ref_cpu")
    VP = pytest.importorskip("v4_pipe")
    from shard.pipeline_plan import build_plan
    from shard.pipeline_session import SessionConfig
    # In-memory genuine keys avoid NTFS's POSIX-mode emulation. Production key
    # permission/reload policy has separate tests; HELLO/signatures remain real.
    stage_keys = [gen_key(), gen_key()]
    key_provider = {str(tmp_path / f"stage-{i}.key"): key for i, key in enumerate(stage_keys)}
    monkeypatch.setattr(VP, "load_or_make_node_key", lambda path: key_provider[str(path)])
    coordinator_key = gen_key()
    draft = bool(getattr(request, "param", False))
    ranges = [(0, 1), (1, 4)] if draft else [(0, 2), (2, 4)]
    args = R.cpu_args(n_layers=4, compress_ratios=(0, 4, 8, 0, 0, 0),
        dspark_target_layer_ids=(1, 2, 3), window_size=8, max_seq_len=64)
    oracle = R.build_oracle(args, 7)
    VP._write_tiny_checkpoint(str(tmp_path), args, oracle)
    prompt, max_new = [3, 9, 17, 2, 41], 4
    reference = VP._reference_tokens(oracle, prompt, max_new)
    ports = VP._free_ports(2)
    addresses = [f"127.0.0.1:{port}" for port in ports]
    plan = build_plan({"n_layers": args.n_layers}, ring_id="conversation-tiny",
        cohort_id="a" * 64, endpoints=addresses, split=[hi - lo for lo, hi in ranges])
    for index, key in enumerate(stage_keys):
        plan["stages"][index]["signer_pubkey"] = pub_b64(key)
    plan["coordinator"]["signer_pubkey"] = pub_b64(coordinator_key)
    configs = [SessionConfig.from_plan(plan, index, caller_key=stage_keys[index], ttl_s=120)
               for index in range(2)]
    coordinator = SessionConfig.from_plan(plan, -1, caller_key=coordinator_key, ttl_s=120)
    monkeypatch.setattr(P, "V4_CONVERSATION_CACHE_MIB", 32)
    monkeypatch.setattr(P, "V4_CONVERSATION_CACHE_GPU_MIB", 0)
    actual_caches = {}
    configured = P.configured_cache
    def record_cache(stage):
        value = configured(stage)
        actual_caches[stage.lo] = value
        return value
    monkeypatch.setattr(P, "configured_cache", record_cache)
    errors, threads, ready = [], [], [threading.Event(), threading.Event()]
    def run(index):
        try:
            lo, hi = ranges[index]
            VP.serve_stage(index, 2, lo, hi, ports[index],
                addresses[index + 1] if index == 0 else None, ckpt_dir=str(tmp_path),
                device="cpu", receipts=True, key_path=str(tmp_path / f"stage-{index}.key"),
                timeout=20, ready=ready[index], session_config=configs[index], dspark=draft)
        except Exception:
            errors.append(traceback.format_exc())
            ready[index].set()
    pipe = ret = None
    try:
        for index in (1, 0):
            thread = threading.Thread(target=run, args=(index,), daemon=True)
            threads.append(thread)
            thread.start()
            assert ready[index].wait(20), "strict stage did not listen"
            assert not errors, errors
        pipe, ret = VP.connect_ring(*addresses, timeout=20, retry_s=3, session_config=coordinator)
        epoch = f"{pipe.grant['boot_id']}:{pipe.grant['fence']}"
        opaque = PrefixBinding(len(prompt), "2" * 64).to_dict()
        identity = ConversationIdentity("tenant-a-hmac", "a" * 64, "bit-native/1", "tiny-cpu-fixture",
            "c" * 64, "d" * 64, "e" * 64, plan["ring_id"], epoch,
            tuple((row["node_id"], 1) for row in plan["stages"])).to_dict()
        expected = {pub_b64(key): ranges[index] for index, key in enumerate(stage_keys)}
        def reconnect():
            nonlocal pipe, ret
            pipe.close()
            ret.close()
            pipe, ret = VP.connect_ring(*addresses, timeout=20, retry_s=3, session_config=coordinator)
            return pipe, ret
        yield SimpleNamespace(VP=VP, pipe=pipe, ret=ret, owner=P.ConversationCoordinator(enabled=True),
            identity=identity, prefix=opaque, prompt=prompt, max_new=max_new, reference=reference,
            expected=expected, caches=actual_caches, errors=errors, coordinator=coordinator,
            reconnect=reconnect, draft=draft, args=args)
    finally:
        if pipe is not None:
            try:
                VP.send_msg(pipe, {"op": "stop"})
            except (OSError, ValueError):
                pass
        for thread in threads:
            thread.join(5)
        for channel in (pipe, ret):
            if channel is not None:
                channel.close()


def actual_job(ring, number, *, identity=None, max_new=None, eos_ids=()):
    maximum = ring.max_new if max_new is None else max_new
    req = ring.owner.request(identity or ring.identity, ring.prefix, max_new=maximum)
    function = ring.VP.coordinate_dspark_pipelined if ring.draft else ring.VP.coordinate
    result = function(ring.pipe, ring.ret, ring.prompt, maximum,
        nonce=f"{number:064x}", swarm_id="conversation-test", job_id=f"job-{number}",
        layer_count=4, receipts=True, timeout=20, expected_by_signer=ring.expected,
        strict_job_binding=True, conversation=req, eos_ids=eos_ids)
    assert not ring.errors, ring.errors
    return result, req


def test_actual_strict_ring_repeat_restores_state_and_signs_only_new_work(strict_tiny_ring):
    ring = strict_tiny_ring
    first, first_request = actual_job(ring, 1)
    repeat, repeat_request = actual_job(ring, 2)
    assert first["tokens"] == repeat["tokens"] == ring.reference
    assert first["receipts_ok"] is repeat["receipts_ok"] is True
    assert first_request.record.get("captured") is True
    assert repeat_request.record["hit"] is True
    assert {row["nonce"] for row in repeat["receipts"]} == {f"{2:064x}"}
    assert all(row["n_chunks"] == ring.max_new for row in first["receipts"])
    assert all(row["n_chunks"] == ring.max_new - 1 for row in repeat["receipts"])
    assert all("conversation_restore" in row for row in repeat["receipts"])
    for cache in ring.caches.values():
        assert cache.status()["hits"] == 1


def test_actual_strict_ring_one_stage_miss_resets_all_then_full_prefill(strict_tiny_ring):
    ring = strict_tiny_ring
    first, _ = actual_job(ring, 3)
    ring.caches[2].invalidate()
    fallback, request = actual_job(ring, 4)
    assert fallback["tokens"] == first["tokens"] == ring.reference
    assert fallback["receipts_ok"] is True and request.record["hit"] is False
    assert request.record["reason"] in ("restore_miss_full_prefill", "all_stages_captured")
    assert all(row["n_chunks"] == ring.max_new for row in fallback["receipts"])
    assert all("conversation_restore" not in row for row in fallback["receipts"])


def test_actual_strict_ring_same_prefix_other_tenant_and_one_token_never_hit(strict_tiny_ring):
    ring = strict_tiny_ring
    actual_job(ring, 5)
    other, request = actual_job(ring, 6, identity=dict(ring.identity, tenant="tenant-b-hmac"))
    assert request.record["hit"] is False and other["tokens"] == ring.reference
    assert all(row["n_chunks"] == ring.max_new for row in other["receipts"])
    short, request = actual_job(ring, 7, max_new=1)
    assert short["tokens"] == ring.reference[:1] and short["receipts_ok"] is True
    assert request.enabled is False
    assert all(row["n_chunks"] == 1 for row in short["receipts"])


def test_actual_strict_ring_prepared_old_owner_disconnect_then_reset_cleans_tickets(strict_tiny_ring):
    ring = strict_tiny_ring
    first, previous = actual_job(ring, 8)
    entry = previous.owner.entries[previous.key]
    rows = previous._control(ring.pipe, ring.ret, ring.VP.send_msg, ring.VP.recv_msg,
                             "restore_prepare", "unfinished-old-owner", entry["entries"])
    assert all(row["ok"] for row in rows)
    assert all(cache.status()["prepared"] == 1 for cache in ring.caches.values())
    ring.pipe, ring.ret = ring.reconnect()
    old_epoch = ring.identity["ring_epoch"]
    ring.identity["ring_epoch"] = f"{ring.pipe.grant['boot_id']}:{ring.pipe.grant['fence']}"
    assert ring.identity["ring_epoch"] != old_epoch
    following, request = actual_job(ring, 9)
    assert following["tokens"] == first["tokens"] == ring.reference
    assert following["receipts_ok"] is True and request.record["hit"] is False
    for cache in ring.caches.values():
        assert cache.status()["prepared"] == 0
        assert cache.stage._conversation_pending == {}


@pytest.mark.parametrize("strict_tiny_ring", [True], indirect=True)
def test_actual_strict_ring_dspark_mtp_repeat_matches_original_greedy(strict_tiny_ring):
    ring = strict_tiny_ring
    first, request = actual_job(ring, 10)
    repeat, request = actual_job(ring, 11)
    assert first["tokens"] == repeat["tokens"] == ring.reference
    assert first["receipts_ok"] is repeat["receipts_ok"] is True
    assert request.record["hit"] is True
    assert all(cache.status()["hits"] == 1 for cache in ring.caches.values())
    assert all("conversation_restore" in row for row in repeat["receipts"])


def test_actual_strict_ring_cached_first_eos_has_real_nonzero_work_receipt(strict_tiny_ring):
    ring = strict_tiny_ring
    initial, _ = actual_job(ring, 12)
    eos, request = actual_job(ring, 13, eos_ids=(initial["tokens"][0],))
    assert eos["tokens"] == initial["tokens"][:1]
    assert request.record["hit"] is False and eos["receipts_ok"] is True
    assert all(row["n_chunks"] == 1 for row in eos["receipts"])
    assert all("conversation_restore" not in row for row in eos["receipts"])


@pytest.mark.parametrize("strict_tiny_ring", [False, True], indirect=True)
def test_actual_strict_ring_extended_shadow_is_signed_and_retains_full_reference(strict_tiny_ring):
    import v4_ref_cpu as R
    ring = strict_tiny_ring
    ring.owner = P.ConversationCoordinator(enabled=True, extended_shadow=True)
    actual_job(ring, 14)
    ring.prompt = ring.prompt + list(range(10, 18))
    ring.prefix = PrefixBinding(len(ring.prompt), "6" * 64).to_dict()
    reference = ring.VP._reference_tokens(R.build_oracle(ring.args, 7), ring.prompt, ring.max_new)
    observed, request = actual_job(ring, 15)
    assert observed["tokens"] == reference and observed["receipts_ok"] is True
    assert request.record["hit"] is False and request.record["reference_shadow"] is True
    assert request.record["extra_suffix_steps"] == 8
    assert type(request.record["all_stage_shadow_passed"]) is bool
    assert all("conversation_restore" not in row for row in observed["receipts"])
    assert all(cache.stage._conversation_shadow is None for cache in ring.caches.values())
    repeated, request = actual_job(ring, 16)
    assert repeated["tokens"] == reference and repeated["receipts_ok"] is True
    assert request.record["hit"] is True
