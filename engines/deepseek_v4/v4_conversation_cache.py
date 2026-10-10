"""Node-local V4 state checkpoints, with prepare/commit restore transactions.

This is a state cache, never a generated-output cache. An exact prompt repeat
restores every persistent attention/draft buffer and its prompt-boundary reply.
Extending a prefix is refused unless a *per-request* full-prefill shadow comparison
has passed. No snapshots, weights, or old signatures are sent between nodes.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import sys
import threading
import time
import uuid

import torch


class ConversationCacheError(RuntimeError):
    pass


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


@dataclass(frozen=True)
class PrefixBinding:
    """Opaque exact-prefix identity, normally coordinator tenant-keyed HMAC.

    Authentication of this binding belongs to the signed control protocol. A
    stage never receives plaintext prefix IDs through the cache control path.
    """
    token_count: int
    digest: str

    def __post_init__(self):
        if type(self.token_count) is not int or self.token_count < 1:
            raise ValueError("positive prefix token count required")
        if not isinstance(self.digest, str) or len(self.digest) != 64 or any(c not in "0123456789abcdef" for c in self.digest):
            raise ValueError("opaque prefix digest must be lowercase SHA256/HMAC")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"token_count", "digest"}:
            raise ValueError("prefix binding requires exactly token_count and digest")
        return cls(**value)

    def to_dict(self):
        return {"token_count": self.token_count, "digest": self.digest}


PrefixKey = PrefixBinding


def _prefix_count(value):
    return value.token_count if isinstance(value, PrefixBinding) else len(value)


@dataclass(frozen=True)
class ConversationIdentity:
    tenant: str
    cohort_id: str
    numeric_contract: str
    source_id: str
    config_sha256: str
    tokenizer_sha256: str
    template_sha256: str
    ring_id: str
    ring_epoch: str
    lease_fences: tuple

    def __post_init__(self):
        for name in ("tenant", "cohort_id", "numeric_contract", "source_id", "config_sha256",
                     "tokenizer_sha256", "template_sha256", "ring_id", "ring_epoch"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 512:
                raise ValueError(f"{name} must be a bounded nonempty identity")
        for name in ("cohort_id", "config_sha256", "tokenizer_sha256", "template_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA256")
        if not isinstance(self.lease_fences, tuple) or not self.lease_fences:
            raise ValueError("ordered node lease fences required")
        if any(not isinstance(v, tuple) or len(v) != 2 or not isinstance(v[0], str) or not v[0]
               or type(v[1]) is not int or v[1] < 1 for v in self.lease_fences):
            raise ValueError("lease fences must be (node identity, positive integer) pairs")
        if len({v[0] for v in self.lease_fences}) != len(self.lease_fences):
            raise ValueError("duplicate lease owner")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("conversation identity must be an object")
        data = dict(value)
        data["lease_fences"] = tuple(tuple(v) for v in data.get("lease_fences", ()))
        return cls(**data)

    def to_dict(self):
        return _json(self.__dict__)


@dataclass(frozen=True)
class CacheQuotas:
    host_bytes: int
    pinned_bytes: int = 0
    gpu_bytes: int = 0
    max_entries: int = 4
    ttl_s: float = 300.0
    max_tenant_entries: int = 4
    tenant_host_bytes: int | None = None

    def __post_init__(self):
        for name in ("host_bytes", "pinned_bytes", "gpu_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative bytes")
        for name in ("max_entries", "max_tenant_entries"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if type(self.ttl_s) not in (float, int) or not math.isfinite(self.ttl_s) or self.ttl_s <= 0:
            raise ValueError("cache TTL must be finite and positive")
        if self.pinned_bytes > self.host_bytes:
            raise ValueError("pinned quota is a subset of host quota")
        if self.tenant_host_bytes is not None and (
                type(self.tenant_host_bytes) is not int or not 0 <= self.tenant_host_bytes <= self.host_bytes):
            raise ValueError("tenant host quota must fit the stage quota")


def _prefix(value):
    if isinstance(value, dict):
        return PrefixBinding.from_dict(value)
    if isinstance(value, PrefixBinding):
        return value
    if torch.is_tensor(value):
        if value.dtype != torch.long or value.dim() != 2 or value.shape[0] != 1:
            raise ValueError("conversation cache supports exact single-sequence int64 token IDs")
        value = value[0].tolist()
    if not isinstance(value, (list, tuple)) or not value or any(type(v) is not int or v < 0 for v in value):
        raise ValueError("nonempty nonnegative token prefix required")
    return tuple(value)


def _signature(t):
    return (str(t.device), str(t.dtype), tuple(t.shape), tuple(t.stride()),
            t.storage_offset(), t.untyped_storage().data_ptr())


def _walk(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple, deque)):
        for item in value:
            yield from _walk(item)


def _size(value):
    tensors = {id(t): t for t in _walk(value)}
    return sum(t.numel() * t.element_size() for t in tensors.values())


def _python_bytes(value, seen=None):
    """Conservative retained Python metadata, counted before copying tensors."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_python_bytes(k, seen) + _python_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, deque)):
        size += sum(_python_bytes(v, seen) for v in value)
    return size


def _copy_tree(value, *, pin=False, memo=None, devices=None, restore=False):
    memo = {} if memo is None else memo
    if torch.is_tensor(value):
        if id(value) not in memo:
            target = devices.get(id(value), "cpu") if restore else "cpu"
            copy = torch.empty_like(value, device=target,
                                    pin_memory=bool(pin and str(target) == "cpu"))
            copy.copy_(value.detach())
            memo[id(value)] = copy
            if devices is not None and not restore:
                devices[id(copy)] = str(value.device)
        return memo[id(value)]
    if isinstance(value, dict):
        return {k: _copy_tree(v, pin=pin, memo=memo, devices=devices, restore=restore)
                for k, v in value.items()}
    if isinstance(value, deque):
        return deque((_copy_tree(v, pin=pin, memo=memo, devices=devices, restore=restore)
                      for v in value), maxlen=value.maxlen)
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_tree(v, pin=pin, memo=memo, devices=devices, restore=restore) for v in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ConversationCacheError(f"unsupported state metadata: {type(value).__name__}")


def _logical_rows(value):
    """Visit one last-axis row with bounded Python state, even at long context.

    itertools.product materializes its input ranges; Tensor iteration unbinds
    an entire dimension. Indexed recursion retains only one view per dimension.
    """
    if value.dim() <= 1:
        yield value
    else:
        for index in range(value.shape[0]):
            yield from _logical_rows(value[index])


def _byte_chunks(tensor, *, workspace=None, chunk_bytes=1 << 20):
    """Literal bytes in logical C order, never a full non-contiguous pack."""
    value = tensor.detach()
    elements = max(1, chunk_bytes // value.element_size())
    if value.is_contiguous():
        rows = (value.reshape(-1),)
    else:
        rows = _logical_rows(value)
    for row in rows:
        for start in range(0, row.numel(), elements):
            part = row[start:start + elements]
            block = part.contiguous().reshape(-1).view(torch.uint8)
            if block.device.type == "cuda":
                target = (torch.empty(block.numel(), dtype=torch.uint8, device="cpu") if workspace is None
                          else workspace[:block.numel()])
                target.copy_(block, non_blocking=False)
                # Do not keep GPU packing storage across the yield/next copy.
                del block
                yield target
                del target
            elif block.device.type == "cpu":
                yield block
                del block
            else:
                raise ConversationCacheError("state hashing supports CPU/CUDA only")


def _same(a, b, *, workspace=None, chunk_bytes=1 << 20):
    if torch.is_tensor(a):
        # torch.equal treats NaN differently and omits the sign of zero. Byte
        # comparison is the numerical contract, including -inf score states.
        if not torch.is_tensor(b) or a.dtype != b.dtype or a.shape != b.shape:
            return False
        # Keep the left side independent when comparing two CUDA sources.
        left = _byte_chunks(a, chunk_bytes=chunk_bytes)
        right = _byte_chunks(b, workspace=workspace, chunk_bytes=chunk_bytes)
        x = y = None
        x_offset = y_offset = 0
        while True:
            if x is None:
                x = next(left, None)
                x_offset = 0
            if y is None:
                y = next(right, None)
                y_offset = 0
            if x is None or y is None:
                return x is None and y is None
            count = min(x.numel() - x_offset, y.numel() - y_offset)
            if not torch.equal(x[x_offset:x_offset + count], y[y_offset:y_offset + count]):
                return False
            x_offset += count
            y_offset += count
            if x_offset == x.numel():
                x = None
            if y_offset == y.numel():
                y = None
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(
            _same(a[k], b[k], workspace=workspace, chunk_bytes=chunk_bytes) for k in a)
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple, deque)):
        return len(a) == len(b) and all(_same(x, y, workspace=workspace, chunk_bytes=chunk_bytes) for x, y in zip(a, b))
    return a == b


def _digest(value, *, workspace=None, chunk_bytes=1 << 20):
    """Canonical structural hash plus literal tensor bytes, at cache barriers."""
    result = hashlib.sha256()
    def feed(data):
        result.update(len(data).to_bytes(8, "big"))
        result.update(data)
    def visit(item):
        if torch.is_tensor(item):
            feed(b"tensor")
            feed(json.dumps([str(item.dtype), list(item.shape)], separators=(",", ":")).encode())
            for block in _byte_chunks(item, workspace=workspace, chunk_bytes=chunk_bytes):
                # A view of this bounded chunk; never a full-size byte string.
                result.update(memoryview(block.numpy()))
        elif isinstance(item, dict):
            feed(b"dict")
            feed(str(len(item)).encode())
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple, deque)):
            feed(type(item).__name__.encode())
            feed(str(len(item)).encode())
            if isinstance(item, deque):
                visit(item.maxlen)
            for part in item:
                visit(part)
        else:
            feed(type(item).__name__.encode())
            feed(json.dumps(item, allow_nan=False, separators=(",", ":")).encode())
    visit(value)
    return result.hexdigest()


def _drafter_tail(drafter):
    return getattr(drafter, "tail", drafter)


def _buffers(stage, drafter):
    runtime = getattr(stage, "_kv_runtime", None)
    if runtime is not None:
        if runtime._active is not None:
            raise ConversationCacheError("KV workspace is active; drain before snapshot/restore")
        runtime.synchronize_host()
    result = OrderedDict()
    for i, layer in enumerate(stage.layers):
        attn = layer.attn
        entry = runtime.entries.get(id(attn)) if runtime is not None else None
        result[f"main.{i}.kv"] = entry["ring"] if entry is not None else attn.kv_cache
        if entry is not None:
            result[f"main.{i}.history"] = entry["history"]
        indexer = getattr(attn, "indexer", None)
        if indexer is not None:
            result[f"main.{i}.index"] = entry["index_history"] if entry is not None else indexer.kv_cache
        for name, owner in (("attention", attn), ("indexer", indexer)):
            compressor = getattr(owner, "compressor", None)
            if compressor is not None:
                for field in ("kv_state", "score_state"):
                    result[f"main.{i}.{name}.{field}"] = getattr(compressor, field)
    tail = _drafter_tail(drafter)
    if tail is not None:
        for i, layer in enumerate(tail.mtp):
            result[f"draft.{i}.kv"] = layer.attn.kv_cache
    return result


def _metadata(stage, drafter):
    main = {name: getattr(stage, name) for name in ("_pos", "_spec", "_dspark", "_last_tap", "_spec_ckpts")}
    result = {"main": main}
    tail = _drafter_tail(drafter)
    if tail is not None:
        result["tail"] = {"_pos": tail._pos, "last_spec": tail.last_spec}
        if tail is not drafter:
            result["drafter"] = {name: getattr(drafter, name) for name in
                                 ("_done", "pipelined", "_cfront", "_mfront", "_last")}
    return result


def _model_stamp(stage, drafter):
    modules = list(stage._owned_modules())
    tail = _drafter_tail(drafter)
    if tail is not None:
        modules.append(tail.mtp)
    params = [p for m in modules for p in m.parameters()]
    params += [getattr(stage, name) for name in ("hc_head_fn", "hc_head_base", "hc_head_scale")
               if torch.is_tensor(getattr(stage, name, None))]
    parameters = tuple((id(p), _signature(p), p._version) for p in {id(p): p for p in params}.values())
    ropes = tuple((_signature(layer.attn.freqs_cis), layer.attn.freqs_cis._version)
                  for m in (stage.layers, getattr(tail, "mtp", ())) for layer in m)
    static = (stage.lo, stage.hi, stage.head, stage.tail, stage._dspark_capable,
              stage._spec_depth, stage._tap_ids, id(getattr(stage, "_kv_runtime", None)),
              getattr(tail, "block_size", None), getattr(tail, "temperature", None))
    return parameters, ropes, static


@dataclass
class _Entry:
    id: str
    identity: ConversationIdentity
    prefix: tuple | PrefixBinding
    buffers: dict
    metadata: dict
    signatures: dict
    model_stamp: tuple
    metadata_devices: dict
    tail_reply: dict | None
    host_bytes: int
    pinned_bytes: int
    restore_gpu_bytes: int
    expires: float
    generation: int
    content_digest: str
    state_digest: str
    hash_gpu_scratch_bytes: int
    tickets: set = field(default_factory=set)


class StageConversationCache:
    """Host snapshots, bounded per stage and tenant; disabled unless explicit.

    Caller owns the stage execution lock and the all-ring drain/barrier. This
    class locks its own entries, and verifies the boundary again at both phases.
    Snapshot/restore may synchronize at these boundaries, never once per token.
    """
    def __init__(self, stage, quotas, *, enabled=False, drafter=None, pinned=False, clock=time.monotonic):
        if type(enabled) is not bool or type(pinned) is not bool or not isinstance(quotas, CacheQuotas):
            raise ValueError("explicit boolean cache configuration and validated quotas required")
        if pinned and not torch.cuda.is_available():
            raise ValueError("pinned conversation cache requires a CUDA runtime")
        self.stage, self.quotas, self.enabled, self.drafter = stage, quotas, enabled, drafter
        self.pinned, self.clock = pinned, clock
        self._entries, self._tickets = OrderedDict(), {}
        self._generation, self._lock = 0, threading.RLock()
        self._hash_chunk_bytes = min(1 << 20, quotas.host_bytes // 8) if enabled else 0
        self._hash_host_reserved_bytes = 3 * self._hash_chunk_bytes
        self._hash_workspace = None
        self._counts = {"captures": 0, "hits": 0, "misses": 0, "evictions": 0,
                        "restore_failures": 0, "shadow_passes": 0, "shadow_failures": 0}

    def _ensure_hash_workspace(self):
        if self._hash_chunk_bytes < 1:
            raise ConversationCacheError("conversation host quota cannot reserve bounded hash workspace")
        if self._hash_workspace is None:
            self._hash_workspace = torch.empty(self._hash_chunk_bytes, dtype=torch.uint8, device="cpu")

    def _state_digest(self, value):
        self._ensure_hash_workspace()
        return _digest(value, workspace=self._hash_workspace, chunk_bytes=self._hash_chunk_bytes)

    def hash_workspace_reservation(self):
        return {"hash_chunk_bytes": self._hash_chunk_bytes,
                "hash_workspace_host_reserved_bytes": self._hash_host_reserved_bytes,
                "hash_workspace_gpu_max_reserved_bytes": max(
                    (e.hash_gpu_scratch_bytes for e in self._entries.values()), default=0)}

    def bind_drafter(self, drafter):
        """Rebind a per-job wrapper, retaining cache only for the same MTP owner."""
        with self._lock:
            if _drafter_tail(drafter) is not _drafter_tail(self.drafter):
                self.invalidate()
            self.drafter = drafter

    def _boundary(self, frontier=None):
        if getattr(self.stage, "_replaying", False):
            raise ConversationCacheError("cannot capture or restore during rollback replay")
        if frontier is not None and (type(frontier) is not int or frontier != self.stage._pos):
            raise ConversationCacheError("snapshot frontier is not the stage's committed position")
        device = torch.device(self.stage.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return _buffers(self.stage, self.drafter)

    def _prune(self):
        for key, entry in tuple(self._entries.items()):
            if self.clock() >= entry.expires:
                self.invalidate(key)
                self._counts["evictions"] += 1

    def _room(self, tenant, host, pin):
        host_limit = self.quotas.host_bytes - self._hash_host_reserved_bytes
        tenant_limit = (min(self.quotas.tenant_host_bytes, host_limit)
                        if self.quotas.tenant_host_bytes is not None else host_limit)
        if host > tenant_limit or pin > self.quotas.pinned_bytes:
            raise ConversationCacheError("conversation snapshot exceeds declared stage/tenant quota")
        while True:
            entries = list(self._entries.values())
            own = [e for e in entries if e.identity.tenant == tenant]
            tenant_full = (len(own) >= self.quotas.max_tenant_entries or
                           sum(e.host_bytes for e in own) + host > tenant_limit)
            all_full = (len(entries) >= self.quotas.max_entries or
                        sum(e.host_bytes for e in entries) + host > host_limit or
                        sum(e.pinned_bytes for e in entries) + pin > self.quotas.pinned_bytes)
            if not tenant_full and not all_full:
                return
            victim = next((e for e in entries if not e.tickets and (not tenant_full or e.identity.tenant == tenant)), None)
            if victim is None:
                raise ConversationCacheError("conversation quota held by prepared restore tickets")
            self._entries.pop(victim.id)
            self._counts["evictions"] += 1

    def capture(self, identity, prefix_ids, *, committed_frontier, drained, tail_reply=None):
        if not self.enabled:
            return None
        if not isinstance(identity, ConversationIdentity) or drained is not True:
            raise ConversationCacheError("validated identity and drained committed boundary required")
        prefix = _prefix(prefix_ids)
        if committed_frontier != _prefix_count(prefix):
            raise ConversationCacheError("prefix must cover exactly the committed stage frontier")
        if _prefix_count(prefix) > self.stage.args.max_seq_len or (
                not isinstance(prefix, PrefixBinding) and max(prefix) >= self.stage.args.vocab_size):
            raise ConversationCacheError("prefix exceeds loaded model vocabulary/context")
        # Reply carries only fresh mathematical prefill results, never proof or
        # request identity. Protocol adds the new nonce and new signer afterward.
        reply = _json(tail_reply) if tail_reply is not None else None
        if reply is not None and (not isinstance(reply, dict) or set(reply) - {"token", "tokens", "draft", "conf", "acc", "n", "d2"}):
            raise ConversationCacheError("cached tail reply may not contain signatures or job metadata")
        with self._lock:
            self._prune()
            buffers = self._boundary(committed_frontier)
            metadata = _metadata(self.stage, self.drafter)
            stamp = _model_stamp(self.stage, self.drafter)
            signatures = {name: _signature(value) for name, value in buffers.items()}
            tree = {"buffers": buffers, "metadata": metadata}
            tensor_bytes = _size(tree)
            # Extra room for the copied dict/deque/tensor wrappers and the
            # entry/device maps; no unbounded token list escapes the RAM quota.
            host = tensor_bytes + 2 * _python_bytes((tree, prefix, reply, stamp, signatures, identity.to_dict())) + 4096
            gpu = sum(t.numel() * t.element_size() for t in {id(t): t for t in _walk(metadata)}.values()
                      if t.device.type == "cuda")
            hash_gpu = self._hash_chunk_bytes if any(t.device.type == "cuda" and not t.is_contiguous()
                                                    for t in _walk(tree)) else 0
            gpu += hash_gpu
            if gpu > self.quotas.gpu_bytes:
                raise ConversationCacheError("restoring dynamic taps/rollback/draft metadata exceeds GPU quota")
            self._room(identity.tenant, host, tensor_bytes if self.pinned else 0)
            self._ensure_hash_workspace()
            devices, memo = {}, {}
            saved = _copy_tree(buffers, pin=self.pinned, memo=memo)
            meta = _copy_tree(metadata, pin=self.pinned, memo=memo, devices=devices)
            if stamp != _model_stamp(self.stage, self.drafter):
                raise ConversationCacheError("weights changed while snapshotting")
            self._generation += 1
            key = uuid.uuid4().hex
            state_digest = self._state_digest({"buffers": saved, "metadata": meta, "tail_reply": reply})
            content_digest = _digest({"state_digest": state_digest, "identity": identity.to_dict(),
                "prefix": prefix.to_dict() if isinstance(prefix, PrefixBinding) else prefix})
            self._entries[key] = _Entry(key, identity, prefix, saved, meta, signatures, stamp, devices,
                reply, host, tensor_bytes if self.pinned else 0, gpu, self.clock() + self.quotas.ttl_s, self._generation,
                content_digest, state_digest, hash_gpu)
            self._counts["captures"] += 1
            return key

    def prepare_restore(self, identity, prefix_ids, *, entry_id=None):
        if not self.enabled:
            return None
        if not isinstance(identity, ConversationIdentity):
            raise ValueError("validated conversation identity required")
        prefix = _prefix(prefix_ids)
        with self._lock:
            self._prune()
            entry = next((e for e in reversed(tuple(self._entries.values())) if
                          (entry_id is None or e.id == entry_id) and e.identity == identity and
                          e.prefix == prefix and self.clock() < e.expires), None)
            if entry is None:
                self._counts["misses"] += 1
                return None
            self._validate_entry(entry)
            if len(self._tickets) >= self.quotas.max_entries * 2:
                raise ConversationCacheError("bounded prepared restore capacity reached")
            token = uuid.uuid4().hex
            entry.tickets.add(token)
            self._tickets[token] = (entry.id, entry.generation)
            return {"ticket": token, "entry_id": entry.id, "frontier": _prefix_count(entry.prefix),
                    "mode": "exact_repeat", "host_bytes": entry.host_bytes,
                    "pinned_bytes": entry.pinned_bytes, "restore_gpu_bytes": entry.restore_gpu_bytes,
                    "entry_content_sha256": entry.content_digest}

    def _validate_entry(self, entry):
        state_digest = self._state_digest({"buffers": entry.buffers, "metadata": entry.metadata, "tail_reply": entry.tail_reply})
        content_digest = _digest({"state_digest": state_digest, "identity": entry.identity.to_dict(),
            "prefix": entry.prefix.to_dict() if isinstance(entry.prefix, PrefixBinding) else entry.prefix})
        if state_digest != entry.state_digest or content_digest != entry.content_digest:
            raise ConversationCacheError("cached state content was modified")
        buffers = self._boundary()
        if self.clock() >= entry.expires or _model_stamp(self.stage, self.drafter) != entry.model_stamp:
            raise ConversationCacheError("expired checkpoint or changed model parameters")
        if {k: _signature(t) for k, t in buffers.items()} != entry.signatures:
            raise ConversationCacheError("state storage changed; captured graph pointers cannot be replaced")
        if set(buffers) != set(entry.buffers):
            raise ConversationCacheError("state inventory changed")
        return buffers

    def _ticket(self, ticket):
        token = ticket.get("ticket") if isinstance(ticket, dict) else ticket
        binding = self._tickets.get(token)
        entry = self._entries.get(binding[0]) if binding else None
        if entry is None or entry.generation != binding[1]:
            raise ConversationCacheError("unknown or consumed restore ticket")
        return token, entry

    def commit_restore(self, ticket):
        with self._lock:
            token, entry = self._ticket(ticket)
            try:
                buffers = self._validate_entry(entry)
                # Allocate dynamic rollback/tap objects *before* touching buffers.
                metadata = _copy_tree(entry.metadata, devices=entry.metadata_devices, restore=True)
                with torch.no_grad():
                    for name, target in buffers.items():
                        target.copy_(entry.buffers[name])
                for name, value in metadata["main"].items():
                    setattr(self.stage, name, value)
                tail = _drafter_tail(self.drafter)
                if tail is not None:
                    for name, value in metadata["tail"].items():
                        setattr(tail, name, value)
                    for name, value in metadata.get("drafter", {}).items():
                        setattr(self.drafter, name, value)
                self._boundary(_prefix_count(entry.prefix))
                state_digest = self._state_digest({"buffers": _buffers(self.stage, self.drafter),
                    "metadata": _metadata(self.stage, self.drafter), "tail_reply": entry.tail_reply})
                if state_digest != entry.state_digest:
                    raise ConversationCacheError("restored state byte hash differs from checkpoint")
                self._entries.move_to_end(entry.id)
                self._counts["hits"] += 1
                return {"entry_id": entry.id, "frontier": _prefix_count(entry.prefix), "prefix_tokens": _prefix_count(entry.prefix),
                        "mode": "exact_repeat", "tail_reply": _json(entry.tail_reply),
                        "entry_content_sha256": entry.content_digest,
                        "restored_state_sha256": state_digest, "state_digest": state_digest}
            except Exception:
                self._counts["restore_failures"] += 1
                self.invalidate(entry.id)
                self.reset_current()
                raise
            finally:
                self.abort_restore(token)

    def abort_restore(self, ticket):
        token = ticket.get("ticket") if isinstance(ticket, dict) else ticket
        with self._lock:
            binding = self._tickets.pop(token, None)
            if binding and binding[0] in self._entries:
                self._entries[binding[0]].tickets.discard(token)

    def invalidate(self, entry_id=None):
        with self._lock:
            ids = tuple(self._entries) if entry_id is None else (entry_id,)
            for key in ids:
                entry = self._entries.pop(key, None)
                if entry:
                    for token in entry.tickets:
                        self._tickets.pop(token, None)

    def reset_current(self):
        self.stage.reset()
        tail = _drafter_tail(self.drafter)
        if tail is not None:
            tail.reset()
            if self.drafter is not tail:
                self.drafter._done = False
                self.drafter._cfront = self.drafter._mfront = self.drafter._last = None

    def shadow_compare(self, candidate_entry_id, *, identity, prefix_ids, tail_reply):
        """Compare a saved incremental candidate to CURRENT full-prefill state.

        Root must execute the original full prompt prefill after candidate capture.
        Reference state stays current on pass/fail. A pass authorizes only this
        exact request; it is not a shape/general incremental-parity certificate.
        Capturing reference state then enables a future *exact repeat* cheaply.
        """
        prefix = _prefix(prefix_ids)
        with self._lock:
            entry = self._entries.get(candidate_entry_id)
            if entry is None or entry.identity != identity or entry.prefix != prefix:
                raise ConversationCacheError("shadow candidate identity/prefix mismatch")
            buffers = self._validate_entry(entry)
            passed = (_same(entry.buffers, buffers, workspace=self._hash_workspace, chunk_bytes=self._hash_chunk_bytes)
                      and _same(entry.metadata, _metadata(self.stage, self.drafter),
                                workspace=self._hash_workspace, chunk_bytes=self._hash_chunk_bytes)
                      and _same(entry.tail_reply, _json(tail_reply)))
            self._counts["shadow_passes" if passed else "shadow_failures"] += 1
            return {"passed": passed, "scope": "per_request_full_prefill_state_and_reply_bytes",
                    "prefix_tokens": _prefix_count(prefix), "incremental_reuse_authorized": passed}

    def status(self):
        with self._lock:
            self._prune()
            return {"schema": "v4-conversation-cache/1", "enabled": self.enabled,
                    "mode": "exact_repeat", "extended_prefix": "per_request_shadow_required",
                    "entries": len(self._entries), "host_bytes": self._hash_host_reserved_bytes + sum(e.host_bytes for e in self._entries.values()),
                    "pinned_bytes": sum(e.pinned_bytes for e in self._entries.values()), "gpu_cache_bytes": 0,
                    "prepared": len(self._tickets), **self.hash_workspace_reservation(), **self._counts}

    def snapshot_tensors(self):
        """Read-only inventory for node resource accounting, not a restore API."""
        with self._lock:
            self._prune()
            if self._hash_workspace is not None:
                yield "conversation.hash_workspace", self._hash_workspace
            for entry in self._entries.values():
                seen = set()
                for i, tensor in enumerate(_walk({"buffers": entry.buffers, "metadata": entry.metadata})):
                    identity = (str(tensor.device), tensor.untyped_storage().data_ptr())
                    if identity not in seen:
                        seen.add(identity)
                        yield f"conversation.{entry.id}.{i}", tensor

    def storage_inventory(self):
        with self._lock:
            tensors = list(self.snapshot_tensors())
            host = sum(t.untyped_storage().nbytes() for _, t in tensors if t.device.type == "cpu")
            pin = sum(t.untyped_storage().nbytes() for _, t in tensors if t.device.type == "cpu" and t.is_pinned())
            return {"host_tensor_bytes": host, "pinned_tensor_bytes": pin,
                    "gpu_tensor_bytes": sum(t.untyped_storage().nbytes() for _, t in tensors if t.device.type == "cuda"),
                    "host_budget_charge_bytes": self._hash_host_reserved_bytes + sum(e.host_bytes for e in self._entries.values()),
                    **self.hash_workspace_reservation(),
                    "scope": "retained snapshots; budget charge additionally bounds Python metadata"}

    def entry_digest(self, entry_id):
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                raise ConversationCacheError("unknown conversation entry")
            self._validate_entry(entry)
            return entry.content_digest


class FullRingRestoreTransaction:
    """Local test/control adapter for the same all-stage wire transaction barrier."""
    def __init__(self, caches):
        self.caches = tuple(caches)
        if not self.caches:
            raise ValueError("all-stage cache inventory required")

    def restore(self, identity, prefix_ids, *, entry_ids=None):
        prepared = []
        try:
            for i, cache in enumerate(self.caches):
                ticket = cache.prepare_restore(identity, prefix_ids,
                    entry_id=None if entry_ids is None else entry_ids[i])
                if ticket is None:
                    raise ConversationCacheError("one stage missed the exact conversation prefix")
                prepared.append((cache, ticket))
            # No inference is allowed before every commit completes.
            replies = [cache.commit_restore(ticket) for cache, ticket in prepared]
            return {"restored": True, "frontier": _prefix_count(_prefix(prefix_ids)), "stages": replies}
        except Exception as error:
            cleanup = []
            for cache, ticket in prepared:
                cache.abort_restore(ticket)
            for cache in self.caches:
                try:
                    cache.invalidate()
                    cache.reset_current()
                except Exception as reset_error:
                    cleanup.append(type(reset_error).__name__)
            if cleanup:
                raise ConversationCacheError("all-ring reset failed; inference must remain fenced: " + ",".join(cleanup)) from error
            return {"restored": False, "reason": type(error).__name__, "fallback": "full_prefill"}
