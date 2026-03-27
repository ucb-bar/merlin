# Development Blog

This section tracks active engineering work across:

- Gemmini dialect
- NPU dialect
- Radiance HAL

The goal is to keep practical notes in one place:

- What changed
- Why it worked or failed
- How it was debugged
- How it was tested
- What to do next

Current global caveat for these streams: they are under active development and
are not yet validated on simulated/programmed/taped-out hardware in this repo.

## Entry Format

For each entry, use this structure:

1. Context and goal
2. Implementation changes
3. What worked
4. What did not work (and why)
5. Debugging notes
6. Test coverage and exact commands
7. Follow-up tasks

Use filenames:

- `YYYY-MM-DD-<topic>.md`

Optional starter template:

- [`TEMPLATE.md`](TEMPLATE.md)

## Current Entries

- [2026-03-25 TargetGen Generation + Mutation Staging](2026-03-25-targetgen-generation-staging.md)
- [2026-03-18 Chipyard Bare-Metal Integration](2026-03-18-chipyard-bare-metal-integration.md)
- [2026-03-16 SpacemiTX60 Dispatch Scheduler + Tracy](2026-03-16-dispatch-level-async.md)
- [2026-03-13 RISC-V MMT4D Ukernel Workstream](2026-03-13-riscv-mmt4d-ukernel-workstream.md)
- [2026-03-12 SmolVLA FP8/INT8 Global-Optimization Workstream](2026-03-12-smolvla-fp8-int8-global-opt-workstream.md)
- [2026-03-11 Gemmini Workstream Log](2026-03-11-gemmini-workstream-log.md)
- [2026-03-11 NPU Dialect E2E Bring-Up](2026-03-11-npu-dialect-e2e.md)
- [2026-03-11 Radiance HAL Workstream Log](2026-03-11-radiance-hal-workstream-log.md)

## Stream Status

| Stream | Latest entry | Status |
|---|---|---|
| SpacemiTX60 dispatch scheduling | [2026-03-16 Dispatch Scheduler + Tracy](2026-03-16-dispatch-level-async.md) | Active |
| RISC-V mmt4d ukernels | [2026-03-13 RISC-V MMT4D Ukernel Workstream](2026-03-13-riscv-mmt4d-ukernel-workstream.md) | Active |
| SmolVLA FP8/INT8 quantization | [2026-03-12 SmolVLA FP8/INT8 Global-Optimization Workstream](2026-03-12-smolvla-fp8-int8-global-opt-workstream.md) | Active |
| Gemmini dialect | [2026-03-11 Gemmini Workstream Log](2026-03-11-gemmini-workstream-log.md) | Active |
| NPU dialect | [2026-03-11 NPU Dialect E2E Bring-Up](2026-03-11-npu-dialect-e2e.md) | Active |
| Radiance HAL | [2026-03-11 Radiance HAL Workstream Log](2026-03-11-radiance-hal-workstream-log.md) | Active |
| Chipyard bare-metal integration | [2026-03-18 Chipyard Bare-Metal Integration](2026-03-18-chipyard-bare-metal-integration.md) | Active |
| TargetGen generation and staging | [2026-03-25 TargetGen Generation + Mutation Staging](2026-03-25-targetgen-generation-staging.md) | Active |
