# capsule-bench target: `gemmini_universal`

Authored inputs now live in [the separate Universal example](../../../../../examples/gemmini_universal/README.md).
This directory retains historical generated bundles, public harness resources and
compatibility links. Do not treat retained bundles as newly reviewed execution inputs.

Jack's "Universal" ResNet-50 Gemmini (Alveo U250 / FireSim) as a **separate** capsule-bench target,
run as a parallel track beside `gemmini`. **The two configurations are not mixed**: this directory
shares only files that carry no hardware claim, enumerated in `target_experiment.yaml` under
*WHAT IS SHARED*.

The device differs from `gemmini` in four parameters, each of which REMOVES programs `gemmini`
executes: weight-stationary-only dataflow, no full-width accumulator readout, no execute-stage
accumulator read / scratchpad write, and a D operand hardcoded to GARBAGE_ADDR. The readout is the one
that reshapes the experiment — the store DMA is 128 bits (DIM x int8) against 512 (DIM x int32) on the
pinned `gemmini` revision, so **no int32 tensor can leave this accelerator**.

| | `gemmini` | `gemmini_universal` |
|---|---|---|
| source pool | 114 | 114 (same shared corpus) |
| capability-excluded | 11 (bf16 operands) | **81** (11 bf16 + 70 int32-output) |
| resource-excluded | 5 | 5 |
| **graded** | **98** | **28** (16 accelerator-positive) |
| host | chipyard_kodiak (vector) | chipyard_riscv64 (scalar; `soc.rvv: false`) |
| ABI header | `3758ae96…` (defines `ACC_READ_FULL_WIDTH`) | `d6db18f8…` (does not) |

Read `target_experiment.yaml` before launching: its `gaps` block names what a launch would be
launching without, and the exclusion note names the one readiness gate that will disagree.

- Target contract / RTL facts / ABI: `merlin/targets/gemmini_universal/contracts/`
- Provenance pins: `gemmini_universal_rtl`, `gemmini_universal_chipyard` (root env
  `MERLIN_UNIVERSAL_CHIPYARD` — deliberately **not** `MERLIN_CHIPYARD`; pointing that at this tree
  would re-attribute every existing gemmini claim to this device)
