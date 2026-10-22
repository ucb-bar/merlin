# Inactive instruction-refill candidate replay

The opt-in adapter in Spike commit `dc30824` was replayed against unchanged public ELFs and retained physical labels. It is **not selected or qualified**. No new physical simulation was run.

| Frontend scope | Previous | Candidate | Physical |
|---|---:|---:|---:|
| Repeated invalidated, both layouts (4 scopes) | 125 | 161 | 177 |
| Prewarmed (6 scopes) | 122 | 122 | 126 |
| First use, both layouts (2 scopes) | 173 | 173 | 177 |

All 12 replay executions check all 18 calls: 4,608 matrix values and 576 guards each. All timing scopes complete without unresolved observations. The repeated scopes apply the new modeled L2-hit refill three times. First-use scopes retain the original lower-miss route. Remaining frontend error is explicit; no aggregate residual was fitted.

The 24 queue-capacity replays also complete numerically, but apply this adapter zero times and retain every previous prediction, including failed first-use/paired-throughput conditions. They are a legacy-path regression check, not evidence that the new path improves queue timing. The queued batch had already been submitted when priority changed to frontend cases; it drained without terminating running simulations.

`comparison.json` retains every result and approximation. `retention.json` pins compact original timing reports and numerical stdout (lossless gzip), plus source/build/model identities. Absolute paths in readable metadata are portable `retained/tasks/` references. No ELF, toolchain, private data, or active campaign state is copied.

The initial replay refused timing because its model still named the prior observer source identity. Those receipts remain under `prior-refused-attempt/`; `source-binding-correction-v2.json` records the sole correction before successful replay. Existing ordinary numerical evidence is reused only for the same pinned ELF; every annotated run independently executes the embedded full-output checks.
