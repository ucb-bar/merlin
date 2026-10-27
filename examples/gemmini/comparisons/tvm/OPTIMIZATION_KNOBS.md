# Optimization knobs and ownership

[FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md) owns the current workload and budget plan.
Start at fusion=2, constant prepacking on, proved bias preload on and CSE off;
then search existing graph alternatives within the same 4h/1h/1h tuning allowances.
Stop new submissions at exhaustion, finish active runs and record completion time
beyond the allowance. Connected confirmation consumes tuning time; final E2E is separate.

| Area | Existing search | Boundary |
| --- | --- | --- |
| Graph | Fusion levels 0–3, constant prepacking, bias preload, CSE | Deduplicate legal plans and re-extract TensorIR tasks after graph changes. |
| Contraction schedules | Legal rectangles/tails, M/N/K reuse, traversal, OS/WS, hoisting, pipeline, staging, epilogue placement and existing unrolling | Stock MetaSchedule decisions/traces; memory and tensorization legality checked. |
| Memory correctness | SRAM allocation/lifetimes, config shadowing, hazard fences, completion and adjacent-load coalescing | Fixed lowering rules, not new independent DSE dimensions. |
| New host scheduling or bounded input packing | Research only | No new compiler passes, CPU scheduling integration or cross-function residency is selected. |

Keep all numerical choices fixed by the selected 8ec protocol and captured
weights/scales. Do not change precision, calibration or rounding to improve a
timing or make an intrinsic fit. Use exact structural equality for record reuse;
model names and captured node IDs never select production lowering.

Historical timing observations and their limits are in
[PERFORMANCE_AUDIT.md](PERFORMANCE_AUDIT.md) and [FASED_SWEEPS.md](FASED_SWEEPS.md).
Host mcycle attribution includes stalls and does not establish compute-bound
behavior or array utilization. Existing winner traces can seed new measurements;
old memory-system cycle labels are not FireSim measurements on the selected image.
The detailed earlier proposals are retained in
[Git history](https://github.com/ucb-bar/merlin/blob/82b321a89a8a646877484f4bd09637040ce0fa26/examples/gemmini/comparisons/tvm/OPTIMIZATION_KNOBS.md).
