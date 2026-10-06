"""per-node runtime: serve one contiguous block of a model's layers.

`ModelRuntime` is the FIREWALL between shard's in-house moat (ring, transport,
spec-decode, receipts, verification — all model-agnostic) and the commodity model
layer (per-architecture forward pass + quant kernels). The orchestration above
(coordinate_pipe) only moves token-ids + activations + argmax over sockets, so it
rides on ANY implementation of this interface unchanged. See docs/MODEL_RUNTIME.md.

A node = a block of layers + a transport endpoint + a heartbeat. Implementations:
  - M25Runtime   — the tuned betanet fast-path (phase0/m25_stage.py, hand-rolled).
  - VllmRuntime  — generic: registry -> model class -> slice layers -> block forward
                   (to build). Inheriting the model zoo is what makes this "one
                   engine, every model."

Contract notes (from the proven m25 serve loop):
  - KV cache lives per-node for THIS block's layers; it crops to `start_pos` so a
    spec-decode rollback needs no extra bookkeeping (a re-prefill at an earlier
    start overwrites stale speculative KV).
  - rotary tables, attention impl, MoE routing, quant — all model-specific, hidden
    behind the implementation. The interface speaks only hidden-states and token-ids.
  - head stage (`is_head`) embeds token-ids; tail stage (`is_tail`) applies the final
    norm + lm_head and returns logits. middle stages only forward hidden-states.
"""

from dataclasses import dataclass


@dataclass
class LayerRange:
    start: int  # inclusive
    end: int    # exclusive


class ModelRuntime:
    """serves layers [layer_range] of `model`, plus the head/tail pieces if assigned.

    model-agnostic by construction: every model-specific concern is an implementation
    detail behind these methods. swappable per model (one adapter per backend), so the
    blast radius of upstream churn or an exotic architecture is contained here.
    """

    def __init__(self, model: str, layer_range: LayerRange,
                 is_head: bool = False, is_tail: bool = False, device: str = "cuda:0"):
        self.model = model
        self.layer_range = layer_range
        self.is_head = is_head      # owns the embedding
        self.is_tail = is_tail      # owns the final norm + lm_head
        self.device = device

    # ---- lifecycle ----
    def load_shard(self) -> None:
        """Pull only this block's weights and its assigned boundary pieces, then
        load them into the memory tiers declared by placement_requirements().
        A backend may use full GPU residency or a measured GPU/host placement;
        the contract itself does not implement weight offloading. Weight key
        names derive from the manifest weight_map and model config."""
        raise NotImplementedError

    def reset(self) -> None:
        """drop this block's KV cache (or logically reset it). called on a new request
        or a spec-decode rollback past the cached span."""
        raise NotImplementedError

    def heartbeat(self) -> dict:
        """Liveness and measured resource availability, reported to the scheduler.
        Unavailable RAM, pinned-memory or transfer measurements remain unknown."""
        raise NotImplementedError

    def placement_requirements(self):
        """Measured shard.resources.PlacementRequirements for this exact block/config.

        Checkpoint storage sizes are insufficient to calibrate runtime peaks. A
        backend must measure its resident weights, state, graph/workspace and
        loading peaks; absence of calibration must remain an explicit failure.
        """
        raise NotImplementedError("this runtime has no measured placement resource contract")

    # ---- forward (the hot path) ----
    def embed(self, token_ids):
        """head only: token-ids -> hidden states for layer 0."""
        raise NotImplementedError

    def forward(self, hidden_states, start_pos: int):
        """run this block's layers over `hidden_states` starting at absolute position
        `start_pos` (KV managed internally, cropped to start_pos for rollback). returns
        the activations to hand to the next stage."""
        raise NotImplementedError

    def logits(self, hidden_states):
        """tail only: final norm + lm_head -> logits [.., vocab]."""
        raise NotImplementedError

    # ---- continuous batching (optional; B requests per ring traversal) ----
    def forward_prefill_stream(self, hidden_states, stream: int, start_pos: int):
        """prefill one stream into its batch row (per-stream, == solo). optional: a
        backend that doesn't batch may leave this unimplemented and serve B=1."""
        raise NotImplementedError

    def forward_decode_batch(self, hidden_states, starts):
        """batched decode across all active rows; each stream's output byte-identical to
        solo (per-stream attention + determinism-pinned MoE)."""
        raise NotImplementedError
