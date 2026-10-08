# V4 foundation validation — 2026-10-06

This is the historical validation snapshot of the 2026-10-06 foundation change,
not the current feature inventory. That change implemented the first three items of the local expert-cache plan:
the frozen benchmark/evidence contract, optional signed runtime observations,
and measured GPU/RAM/pinned-memory resource contracts and probes.

At that foundation snapshot the V4 loader remained fully GPU resident, and local
expert offloading/cache replacement, KV paging and memory-aware placement were later work.
They subsequently gained opt-in implementations; current contracts and limits are in
[V4_HYBRID_RUNTIME.md](V4_HYBRID_RUNTIME.md) and [V4_NEXT_PHASE.md](V4_NEXT_PHASE.md).
Neither four-card 40 tok/s nor six-card 30 tok/s has been measured in this run.

## Environment

Validation used an ignored workspace `.venv` on Windows with Python 3.11.15,
PyTorch 2.14.1+cpu, pytest 9.1.1, safetensors 0.8.0 and cryptography 50.0.2.
No checkpoint download, GPU workload or fleet rental was performed.

For reproducible local CPU tests:

```powershell
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
```

## Passed checks

| Suite | Result |
|---|---|
| New benchmark, runtime metrics, V4 observer and resource-contract tests | 78 passed |
| Those new tests plus receipt binding/coverage, plan, probe, scheduler, engine boundaries and packaging hygiene | 217 passed, including the 78 above |
| Existing V4 Stage and DSpark tests | 230 passed |
| V4 ring/protocol tests | 147 passed, 1 skipped |
| V4 lever/launch audit tests | 23 passed |

The four nonoverlapping larger suites total **617 passed, 1 skipped**. The ring
skip is an existing test's explicit refusal to grade a degenerate toy oracle
fingerprint on this build. Tests retained the original token, hidden-state,
KV/rollback, signature, signer, nonce and chain assertions.

`compileall`, `git diff --check`, benchmark/resource CLI help, and the host probe
were also checked. The H2D probe reports unavailable CUDA in this CPU-only test
environment, with unknown bandwidth/pinnable capacity rather than guessed zero.
The Windows host probe separately reports physical free RAM; the current
process is in a Windows job whose remaining memory limit is unknown, so it does
not claim that all physical free RAM can be admitted to a node.

Two portability corrections were included: high-resolution `perf_counter`
measurement for short pipeline-fill intervals, and explicit UTF-8 source reads
for the lever audit. The keep-warm test observes sender completion timestamps
instead of treating a previously queued packet as a post-stop write. Independent
test rings now get separate node-key directories, while jobs on one ring retain
their signing identities. Production key permission checks were not relaxed.

## Historical full-stack limitation and subsequent resolution

The broader `tests/test_v4_full_stack.py` check was not green at the foundation snapshot:
its offline selftest rejects the fixed toy prompt because the reference returns
`[388, 388, 388, 388, 388, 388]` rather than the required diverse fingerprint.
Loading the original `HEAD` version of `v4_pipe.py` in the same environment and
running its selftest reproduces this failure and the same token stream.
The selftest's ring/reference, speculative decoding, DSpark, rollback and receipt
checks pass, but its fingerprint gate was not weakened or reported as passed.
The subsequent fix depth-scaled only synthetic oracle residual-output initialization;
real checkpoint parameters and the diversity/parity gates were not weakened. The current
CPU full-stack check passes as described in [V4_NEXT_PHASE.md](V4_NEXT_PHASE.md).
The 2026-10-08 `c2ab623` selected CPU/socket regression records 985 passed, 3 skipped;
that is neither a complete repository CI run nor GPU hardware acceptance.
Before a GPU deployment, run the existing full-stack checks in the deployment's
intended environment as well as the benchmark's hardware/evidence gates.

## Interfaces

- [V4_BENCHMARK.md](V4_BENCHMARK.md): local checkpoint/source inventory, frozen
  prompts/settings, existing-ring runner, raw evidence verification and comparison.
- [RUNTIME_METRICS.md](RUNTIME_METRICS.md): optional signed observations and their
  work/memory denominators.
- [RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md): byte-unit resource requirements,
  host/H2D probes, V4 header inventory and real Stage calibration commands.

Header inventory and incomplete measurement reports cannot be used as an
automatic placement calibration or silently inherit the M25 profile.
