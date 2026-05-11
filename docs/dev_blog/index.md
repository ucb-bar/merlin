# Workstream Logs

This section tracks active engineering work across the Gemmini dialect, NPU
dialect, Radiance HAL, OPU/SaturnOPU codegen, dispatch scheduling, and
runtime/control-plane bring-up.

Each entry begins with a **Repro pin** (merlin + `iree_bar` SHAs) and, where the
debugging arc evolved across the file, a **Final outcome** or **Related
entries** block at the top — read those first before diving in. Use the SHAs
to reproduce the exact state. See [`TEMPLATE.md`](TEMPLATE.md) for the
expected entry shape.

## Entry Format

For each entry:

1. Repro pin (merlin + iree_bar SHAs)
2. Context and goal
3. Implementation changes
4. What worked / did not work
5. Debugging notes
6. Test coverage and exact commands
7. Follow-up tasks

Use filenames `YYYY-MM-DD-<topic>.md`. Optional starter: [`TEMPLATE.md`](TEMPLATE.md).

## Current Entries

- [2026-04-14 F32 Reduction Hang Findings](2026-04-14-f32-reduction-hang-findings.md)
- [2026-04-13 Saturn OPU vfredusum Scalarization](2026-04-13-saturn-opu-vfredusum-scalarization.md)
- [2026-04-06 OPU Utilization + E2E Benchmarking](2026-04-06-opu-utilization-e2e-benchmarking.md)
- [2026-03-25 TargetGen Generation + Mutation Staging](2026-03-25-targetgen-generation-staging.md)
- [2026-03-25 Ray Control Plane Bootstrap](2026-03-25-ray-control-plane-bootstrap.md)
- [2026-03-18 Chipyard Bare-Metal Integration](2026-03-18-chipyard-bare-metal-integration.md)
- [2026-03-17 SpacemiTX60 Dispatch Scheduler + Tracy](2026-03-16-dispatch-level-async.md)
- [2026-03-13 RISC-V MMT4D Ukernel Workstream](2026-03-13-riscv-mmt4d-ukernel-workstream.md)
- [2026-03-12 SmolVLA FP8/INT8 Global-Optimization Workstream](2026-03-12-smolvla-fp8-int8-global-opt-workstream.md)
- [2026-03-11 Gemmini Workstream Log](2026-03-11-gemmini-workstream-log.md)
- [2026-03-11 NPU Dialect E2E Bring-Up](2026-03-11-npu-dialect-e2e.md)
- [2026-03-11 Radiance HAL Workstream Log](2026-03-11-radiance-hal-workstream-log.md)

Caveat: most of these streams are under active development and may not yet be
validated on simulated/programmed/taped-out hardware in this repo.
