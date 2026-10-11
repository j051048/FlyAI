"""Internal stateful single-token continuation, not new-chat prefill equivalence.

The trusted caller supplies the exact already-consumed token trace and owns the
stage execution lock. The numerical oracle is uninterrupted execution of THAT
prefill and subsequent single-token decode steps. No public HTTP/wire capability
is enabled here; ordinary extended-prefix requests still require reference shadow.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import sys
import threading
import uuid

import torch
from v4_conversation_cache import ConversationCacheError, ConversationIdentity, _prefix, _model_stamp, _buffers, _signature

CONTRACT = "native-single-token-continuation/1"


@dataclass(frozen=True)
class TraceCheckpoint:
    entry_id: str
    trace_id: str
    frontier: int
    horizon: int
    prefix: tuple
    identity: ConversationIdentity
    _cache_identity: ConversationIdentity
    _issuer: object


class NativeTraceStore:
    """Use an existing explicitly budgeted cache; allocate no second state pool."""
    def __init__(self, cache):
        if not cache.enabled:
            raise ConversationCacheError("native trace requires an explicitly enabled bounded state cache")
        self.cache, self.stage = cache, cache.stage
        self._issuer, self._lock, self._live = object(), threading.RLock(), None

    def _greedy(self):
        if self.stage._spec or self.stage._dspark or self.stage._replaying or self.stage._spec_ckpts:
            raise ConversationCacheError("native trace v1 requires drained greedy execution without MTP/speculation")
        slim = sys.modules.get("v4_ref_slim")
        if slim is not None and (getattr(slim, "V4_REF_SLIM", False) or getattr(slim, "V4_REF_SLIM_NOQAT", False)):
            raise ConversationCacheError("history-dependent reference bypass is outside native trace v1")
        if any(getattr(layer.attn, name, False) for layer in self.stage.layers
               for name in ("_slim_skip_indexer", "_ref_slim_skip_indexer")):
            raise ConversationCacheError("history-dependent reference bypass is outside native trace v1")

    def _stamp(self):
        return (_model_stamp(self.stage, None), self.stage._pos,
                tuple((name, _signature(value), value._version)
                      for name, value in _buffers(self.stage, None, synchronize=False).items()))

    def capture(self, identity, prefix_ids, *, horizon, drained, trace_id=None, tail_reply=None):
        with self._lock:
            self._greedy()
            prefix = _prefix(prefix_ids)
            if (not isinstance(prefix, tuple) or type(horizon) is not int or
                    not len(prefix) <= horizon <= self.stage.args.max_seq_len):
                raise ConversationCacheError("native trace needs raw exact consumed IDs and its fixed execution horizon")
            trace_id = trace_id or uuid.uuid4().hex
            if not isinstance(trace_id, str) or not 1 <= len(trace_id) <= 128:
                raise ConversationCacheError("bounded native execution trace identity required")
            digest = hashlib.sha256(json.dumps([CONTRACT, trace_id, horizon], separators=(",", ":")).encode()).hexdigest()
            # Namespace the internal trace so normal exact-repeat/full-prefill
            # lookup cannot serve a decode-generated state as a fresh prefill.
            effective = replace(identity, numeric_contract=identity.numeric_contract + "|trace:" + digest)
            entry = self.cache.capture(effective, prefix, committed_frontier=len(prefix),
                                       drained=drained, tail_reply=tail_reply)
            checkpoint = TraceCheckpoint(entry, trace_id, len(prefix), horizon, prefix, identity, effective, self._issuer)
            self._live = checkpoint, self._stamp()
            return checkpoint

    def restore(self, checkpoint, *, identity, prefix_ids, horizon):
        with self._lock:
            if (not isinstance(checkpoint, TraceCheckpoint) or checkpoint._issuer is not self._issuer or
                    identity != checkpoint.identity or _prefix(prefix_ids) != checkpoint.prefix or horizon != checkpoint.horizon):
                raise ConversationCacheError("native trace owner/tenant/cohort/prefix/horizon differs")
            ticket = self.cache.prepare_restore(checkpoint._cache_identity, checkpoint.prefix,
                                               entry_id=checkpoint.entry_id)
            if ticket is None:
                raise ConversationCacheError("native trace expired, evicted or unavailable")
            result = self.cache.commit_restore(ticket)
            self._greedy()
            self._live = checkpoint, self._stamp()
            return {**result, "mode": "native_trace_continuation", "contract": CONTRACT,
                    "scope": "same native execution trace; not ordinary full-prefill equality",
                    "gpu_qualified": False}

    def advance(self, ids, *, hidden=None):
        """One original native step. Caller must snapshot the new frontier explicitly."""
        with self._lock:
            self._greedy()
            if self._live is None or self._live[1] != self._stamp():
                raise ConversationCacheError("stage changed outside the owned trace; restore before continuing")
            checkpoint = self._live[0]
            if self.stage._pos >= checkpoint.horizon:
                raise ConversationCacheError("native trace fixed horizon exhausted")
            if torch.is_tensor(ids):
                if ids.device.type != "cpu" or ids.dtype != torch.long or ids.numel() != 1:
                    raise ConversationCacheError("native trace input must be one host int64 token")
                value = int(ids.item())
            else:
                value = ids[0] if isinstance(ids, (list, tuple)) and len(ids) == 1 else ids
            if type(value) is not int or not 0 <= value < self.stage.args.vocab_size:
                raise ConversationCacheError("native trace v1 advances exactly one sequence and one token")
            token = torch.tensor([[value]], dtype=torch.long, device=self.stage.device)
            if hidden is None:
                if not self.stage.head:
                    raise ConversationCacheError("non-head native stage needs its original incoming hidden payload")
                hidden = self.stage.embed(token)
            if tuple(hidden.shape) != (1, 1, self.stage.args.hc_mult, self.stage.args.dim):
                raise ConversationCacheError("native trace hidden payload has changed the original step shape")
            with torch.no_grad():
                result = self.stage.forward(hidden, token, self.stage._pos)
            self._live = checkpoint, self._stamp()
            return result
