"""Model-free n-gram / prompt-lookup drafter for the long-context spec-decode path.

Why this exists: over WAN, tok/s = g / (hops*RTT), where g = committed tokens per ring
traversal. A draft MODEL is the usual way to raise g, but the gpt-oss-20b draft OOMs at
100k context (its own KV blows the card), so spec-decode dies exactly where long context
needs it most. An n-gram drafter raises g with ZERO model and ZERO KV: it proposes the
continuation that followed the most recent earlier occurrence of the current suffix in the
text so far. At 100k context that text is huge and (for code / documents / structured data,
the latency-tolerant long-context demand) highly self-similar, so g climbs — for free.

Losslessness: this only PROPOSES. The distributed target verifies every proposed token in
the ring and greedy-commits the accepted prefix + one correction, so the output is bit-for-bit
the same greedy decode no matter how good or bad the proposals are. Proposal quality moves g
(speed), never correctness — so we're free to be heuristic here.

Fixed-shape contract: the fast-verify CUDA graph needs a fixed K+1-token chunk, so propose()
ALWAYS returns exactly k tokens (padded if the match runs short, or no match exists — a pad
token just gets rejected and the round still commits the one free verify token, i.e. it
degrades to plain decode, never worse).

Interface mirrors the async draft-socket the pipelined coordinator already speaks
(request -> fetch, one outstanding), so it drops into coordinate_pipe as `local_draft`:
    d.request(ids, k)   # snapshot the conditioning prefix (cheap)
    ds = d.fetch()      # exactly k proposed token ids
"""


import os


class NgramDrafter:
    """Incrementally-indexed prompt-lookup. `table` maps an ng-token suffix -> the latest
    earlier position whose continuation we copy. The index is grown forward as the committed
    sequence grows (O(appended) per round); on a divergence (the coordinator rewinds a short
    speculative tail) it re-indexes only from the divergence point — stale entries pointing
    past the new end are harmless (a bad proposal is just rejected). The whole search is a
    dict lookup, so it stays ~O(1) per round even at 100k context, fully hidden behind the
    WAN verify."""

    def __init__(self, ng=3, max_ext=64, max_cand=48, margin=256, min_match=1, adaptive=None):
        self.ng = ng                 # anchor suffix length to index on (the lookup key)
        self.max_ext = max_ext       # cap how far back we extend a match (bounds per-round cost)
        self.max_cand = max_cand     # only weigh the most-recent N occurrences of an anchor
        self.min_match = min_match
        
        # P0-1: Adaptive margin handling to prevent index starvation on short context (<=256 tok).
        # When margin is the default 256 or NGRAM_MARGIN is 'auto', dynamic derive min(cap, ctx // 4).
        # Explicit non-default margin (like margin=64 in fake-ring regression suites) is preserved fixed.
        env_margin = os.environ.get("NGRAM_MARGIN", "").strip().lower()
        if env_margin:
            if env_margin == "auto":
                self.margin_cap = 256
                self.adaptive = True
            else:
                try:
                    self.margin_cap = int(env_margin.removeprefix("fixed:"))
                    self.adaptive = False
                except ValueError:
                    raise ValueError("NGRAM_MARGIN requires auto, fixed:<nonnegative integer>, or a legacy integer") from None
        else:
            self.margin_cap = margin
            self.adaptive = (adaptive if adaptive is not None else (margin == 256))
        if type(self.margin_cap) is not int or self.margin_cap < 0:
            raise ValueError("n-gram margin must be a nonnegative integer")

        self.margin = self.margin_cap
        self.indexed = 0             # committed positions < this are in `table`
        self.table = {}              # ng-token anchor -> LIST of continuation-start positions (chronological)
        self._pending = None         # snapshotted (ids, k) between request() and fetch()
        self.matched = False         # did the LAST propose() find a real longest-match? (HybridDrafter routing)
        
        # P1-1: Workload telemetry to quantify copy-only vs novel text efficacy
        self.ngram_rounds = 0        # total propose rounds
        self.ngram_silent = 0        # fallback placeholder rounds (no match)
        self.ngram_matched = 0       # valid proposals
        self.accepted_history = []   # feedback record of accepted tokens

    def effective_margin(self, seq_len: int) -> int:
        """Derive active margin from current sequence length to avoid table starvation."""
        if not self.adaptive:
            return self.margin_cap
        if seq_len <= self.margin_cap:
            return min(self.margin_cap, max(0, seq_len // 4))
        return self.margin_cap

    @property
    def silent_ratio(self) -> float:
        return self.ngram_silent / max(1, self.ngram_rounds)

    def note_accepted(self, count: int):
        """Record accepted token count from verification round."""
        self.accepted_history.append(int(count))

    def metrics(self) -> dict:
        return {
            "ngram_rounds": self.ngram_rounds,
            "ngram_silent": self.ngram_silent,
            "ngram_matched": self.ngram_matched,
            "silent_ratio": round(self.silent_ratio, 4),
            "table_entries": len(self.table),
            "mean_accept": (round(sum(self.accepted_history) / max(1, len(self.accepted_history)), 3)
                            if self.accepted_history else 0.0),
        }

    # ---- async-draft-socket shim ------------------------------------------------
    def request(self, ids, k):
        self._pending = (list(ids), k)          # snapshot: coordinator may mutate its prefix

    def fetch(self):
        ids, k = self._pending
        return self.propose(ids, k)

    def cancel(self):
        self._pending = None                    # drop a stale request without computing the proposal

    # ---- the drafter ------------------------------------------------------------
    def _sync(self, seq):
        """Index newly-committed anchors only — positions in [indexed, len-eff_margin)."""
        eff_margin = self.effective_margin(len(seq))
        stable = len(seq) - eff_margin
        if stable <= self.indexed:
            return
        ng, tbl = self.ng, self.table
        for p in range(max(self.indexed, ng), stable):       # anchor seq[p-ng:p], continuation start p
            tbl.setdefault(tuple(seq[p - ng:p]), []).append(p)
        self.indexed = stable

    def propose(self, seq, k):
        self._sync(seq)
        n = len(seq)
        self.matched = False                     # default: no real match (HybridDrafter -> EAGLE on a miss)
        self.ngram_rounds += 1

        if n < self.ng:                          # not enough context yet -> plain decode
            self.ngram_silent += 1
            return [seq[-1] if seq else 0] * k
        cands = self.table.get(tuple(seq[n - self.ng:n]))
        if not cands:                            # the suffix never occurred in committed text -> plain decode
            self.ngram_silent += 1
            return [seq[-1]] * k
        # LONGEST-MATCH: among all earlier occurrences of the anchor, pick the one whose preceding
        # context matches the current position the longest -> at large context this disambiguates
        # which file/region we're truly copying (a generic 2-gram has many homes; a long match has one).
        ng, me = self.ng, self.max_ext
        best_p, best_len = None, -1
        for p in cands[-self.max_cand:][::-1]:   # most recent first (ties -> most recent wins)
            if p >= n:
                continue
            L = 0
            while L < me and p - ng - 1 - L >= 0 and seq[p - ng - 1 - L] == seq[n - ng - 1 - L]:
                L += 1
            if L > best_len:
                best_len, best_p = L, p
                if L == me:
                    break
        if best_p is None:
            self.ngram_silent += 1
            return [seq[-1]] * k
        self.matched = best_len >= self.min_match   # real longest-match -> draftable (see min_match)
        self.ngram_matched += 1
        cont = seq[best_p:best_p + k]
        if len(cont) < k:                        # ran off the end -> pad (pads get rejected, harmless)
            cont = cont + [cont[-1] if cont else seq[-1]] * (k - len(cont))
        return cont


def simulate_g(seq_ids, prompt_len, ng=3, k=4):
    """Offline upper-bound on g for a fixed greedy output `seq_ids` (prompt + generation):
    walk the generation region simulating greedy n-gram spec-decode and count committed
    tokens per traversal. Faithful because the verify always lands >=1 true token and the
    accepted prefix is exactly what real greedy spec-decode commits. Returns (g, traversals)."""
    d = NgramDrafter(ng=ng, margin=0)            # offline: no speculative tail, index everything
    i = prompt_len
    end = len(seq_ids)
    traversals = 0
    while i < end:
        ds = d.propose(seq_ids[:i], k)
        acc = 0
        for j in range(min(k, end - i - 1)):
            if ds[j] == seq_ids[i + j]:
                acc += 1
            else:
                break
        i += acc + 1                              # accepted drafts + the free verify token
        traversals += 1
    return (end - prompt_len) / max(traversals, 1), traversals
