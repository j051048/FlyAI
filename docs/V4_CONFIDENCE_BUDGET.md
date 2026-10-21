# Calibrated confidence prefix budgets

This local opt-in extends `round_speculation`. It keeps the loaded DSpark block,
rollback depth, refill floor, verification frames, greedy accept rule, reset and
receipt barriers fixed. It chooses only a not-yet-submitted prefix at a settled
floor-one refill. Existing configurations retain the existing controller.

V4's confidence head returns raw logits, including negative values. The new
calibrator fits a separate isotonic/PAV acceptance curve at each reached draft
position. It never assumes a V4.1 sigmoid or multiplies raw logits. Entire request
IDs are split deterministically before fitting (75% training, 25% held out).
Artifacts retain held-out Brier error, ECE, reliability bins, observation counts
and training score support. Unreached positions after a rejection are censored;
unsent candidates are never labeled as rejected.

Static prefix experiments automatically retain a bounded `confidence_probe` in
their coordinator result, which the benchmark stores in its raw
`coordinator_stats`. The probe includes only observed reached-position labels and
actual refill counts. The default runtime does not collect it. Explicit
`confidence_probe=True` is available for local correctness/probe runs. A truncated
probe cannot be promoted. The legacy serial `conf_probe` callback now receives
only the sent confidence prefix, with its accepted count; positions after the
first rejection remain censored.

The CLI consumes completed raw benchmark reports and their independently frozen
protocols, plus the service's exported selected binding:

```powershell
python phase0/v4_confidence_calibration.py `
  --reports baseline-report.json short-report.json `
  --protocols baseline-protocol.json short-protocol.json `
  --binding C:/private/round.json.binding.json `
  --max-p95-ttft-s 4 --max-p95-tpot-s 0.2 `
  --out C:/private/confidence.json
```

Output defaults to `mode=shadow`. Use the existing private round slot with
`--publish-slot`, or install its operator-local configuration normally. The
configuration includes `confidence_calibration` alongside the ordinary full round
calibration. Service admission rebuilds both from raw reports; a client request
cannot install an artifact or select a new policy.

Statistical fit and execution qualification are separate. A collection of score
rows alone cannot qualify hardware. Promotion re-verifies every stage's raw
signed receipts, exact source/runtime bindings, static prefix experiments,
unchanged greedy outputs, workload protocol and frozen network contract. GPU
enforcement additionally requires an authenticated logical checkpoint identity
and GPU work metrics from every measured stage; CPU/missing metrics remain
statistical only. Signed worker observations do not prove hardware honesty, and
coordinator confidence labels and wall clocks are observations, not remote
execution attestations.

At runtime, the artifact is bound to the selected model cohort, source bytes,
runtime configuration, ring and verified lease generation, loaded width and fixed
depth/floor. It expires within one day. Its workload must match the actual prompt
token digest and generation limit exactly. Generalization to arbitrary prompts
is intentionally not assumed.

For each measured cap, expected committed tokens are one guaranteed greedy token
plus the sum of calibrated position-conditional survival estimates. The product
of these conditional estimates is a statistical model of survival. The objective
divides this expectation by the actual warm wall time per observed refill,
including WAN, queue gaps, rejected work and terminal drain. It uses only frozen
costs, without teaching a shadow run the speed of an unexecuted prefix. A proposed
gain must exceed the existing minimum-gain setting, and the candidate must pass
the frozen P95 TTFT/TPOT bounds. An optional DMA constraint requires complete
all-stage measurements; unknown DMA stays `null` and never becomes zero.

Missing/stale/unverified evidence, an unknown workload, an unsupported raw score,
insufficient held-out coverage, excessive calibration error, or missing required
baseline/P95/DMA measurements keeps the unchanged prefix. Shadow records the
proposed cap while sending the baseline. CPU tests cover genuine negative model
scores, real socket cancellation, strict signed receipts and repeated resets;
they establish correctness and make no GPU speed or production qualification
claim. No live artifact is included in the repository.
