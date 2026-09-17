# Merlin provenance — `merlin_assisted` pilot run

## 1. Merlin tools used

| Tool (path) | Used? | What it contributed |
|---|---|---|
| `targetgen/synthesize/` | yes | Tiling/lever structure and placement of the capacity-fit policy. |
| `targetgen/generate/` scaffold | yes | Initial out-of-tree package organization and target-artifact emitter seam. |
| `xdsl_dialects/` | yes | Typed operation patterns used to define the interface and Gemmini dialects. |
| `targetgen/contract/interface_emit.py` | yes | Frozen v0.1 interface and command-buffer operand conventions. |

The CCA bijection, all seven target escalation ladders, derived levers, RTL facts, and scaffold generator were invoked before the first submission edit. No Merlin runtime/reference implementation was copied into the package.

## 2. Generated and authored files

| submission file | origin | notes |
|---|---|---|
| `manifest.yaml` | hand | Four Python/xDSL CLI routes. |
| `mlir_oot/xdsl_dialects/*` | mixed | Scaffold shape, then hand-authored typed ops and verification. |
| `mlir_oot/targetgen/synthesize/*` | mixed | Generated seams with hand-authored static tiling. |
| `mlir_oot/targetgen/generate/*` | mixed | Generated seam with hand-authored pointer-based LLVM/RoCC emission. |
| `mlir_oot/lowering/isa.py` | hand | Header/RTL-fact-derived instruction packing and schedules. |
| vendored `xdsl` and dependencies | vendored | Required for isolated `python3 -S` execution. |

## 3. Failures and diagnosis

| round | capsule/family | failure plane | response | Merlin help |
|---|---|---|---|---|
| 1 | A2 | Spike, one populated row | Added live CONFIG_EX A/C stride defaults. | ISA disassembly exposed the missing config fields. |
| 1 | A3/GM | Spike K reduction | Corrected full-width/accumulate address flags and WS schedule. | RTL facts and header-grounded emitter seam. |
| 1 | conv | command buffer then Spike | Added modeled im2col recipe and ordinary matmul lowering. | Interface contract identified the admitted derived-input route. |
| 1 | M/N tails | Spike trailing zeros | Tested header schedules, address layouts, drains, and full-height broadcasts; no tested variant fixed the hardware cutoff. | Trace checks ruled out illegal functs and missing tile readbacks. |
| 1 | attention/pool | command-buffer semantics | Tried modeled MATMUL/VREDUCE decompositions; the active ABI lacks a value-preserving transpose recipe and windowed VREDUCE route. | Contract errors localized these to L0. |

## 4. Files changed per iteration

One authoring round modified `ir_ingest.py`, `transforms.py`, both xDSL dialect modules, tiling, ISA lowering, LLVM emission, the manifest, report, and docs. The best completed full Spike response passed 20 of 33 capsules; later experiments were reverted when focused checks were unchanged or worse. Detailed changes are in `iteration_notes.md`.

## 5. Final-artifact integrity

- Final artifact imports Merlin runtime code: **no**.
- Final artifact is self-contained and invoked only through CLI entrypoints: **yes**.
- Accidental Merlin runtime/scaffold adapters in `submission/`: **none**.

## 6. One-line summary

Merlin materially helped establish the typed package, ABI seams, lever routing, and RTL-grounded diagnostics; it did not resolve the remaining primitive-tail hardware behavior or missing command-buffer transpose/window semantics.
