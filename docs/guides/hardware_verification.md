---
title: Verify selected hardware properties
kind: guide
status: current
owner: verification
last_verified: 2026-11-09
related: [verification, phase0_specification, simulator_selection]
code_refs:
  - src/merlin_experiments/phase0/hardware_validation.py
  - src/merlin_experiments/phase0/cell_probe.py
  - examples/gemmini/verification/mac_unit_reference.mlir
---

# Verify selected hardware properties

Hardware facts, simulations and formal checks answer different questions. A CIRCT
extraction can establish the selected module, widths and structure. A native RTL
cell probe checks a finite set of input vectors. A combinational equivalence
proof asks whether **any** raw input bit pattern can make a selected cell differ
from an authored reference at its concrete port widths. A bounded model check
asks about only the declared number of cycles. Keep those evidence kinds separate
in Phase 0 and when using Phase 0 tests in Phase 2.

## Prove one combinational cell property

Author a target-specific HW/Comb reference that implements one numerical or
interface property. Review its arithmetic semantics independently of the RTL.
For a selected integer cell, [this reference](../../examples/gemmini/verification/mac_unit_reference.mlir)
specifies signed 8-bit multiply and 20-bit modular accumulation by sign-extending
both operands, multiplying at 20 bits and adding the low 20 accumulator bits. Its proof is
about the cell's input/output function, including every value of the 32-bit
accumulator input. It does not prove a whole matrix multiplication, DMA command,
quantization policy, or compiler lowering.

Select the actual CIRCT HW source and exact tool binaries. The caller names the
cell and reference modules; the shared runner contains no target arithmetic:

```python
from merlin_experiments.phase0.hardware_validation import (
    prove_combinational_property,
    verify_property_receipt,
)

receipt = prove_combinational_property(
    hw_source="/selected/core.hw.mlir",
    module="MacUnit",
    reference="examples/gemmini/verification/mac_unit_reference.mlir",
    reference_module="MacUnitReference",
    circt_opt="/selected/circt/bin/circt-opt",
    z3="/selected/bin/z3",
    output="out/artifacts/verification/selected-cell-property",
    property_statement="signed INT8 product plus low 20 bits of C modulo 2^20",
)
assert receipt["status"] == "proven"
verify_property_receipt(
    "out/artifacts/verification/selected-cell-property/property.json",
    hw_source="/selected/core.hw.mlir",
)
```

The runner extracts the cell's transitive HW module closure from the selected
source, rejects unresolved/opaque/stateful modules and implicit assumptions,
constructs a CIRCT logical-equivalence miter, lowers it to SMT-LIB, and asks Z3
whether the outputs can differ. An `unsat` result is recorded as `proven`; `sat`
is `refuted` with a saved raw-bit counterexample; timeout, missing tools and
unrecognized output are `unknown`. A proof has no cycle bound because this
runner admits only combinational cells. It deliberately refuses a nonempty
`assumptions` list instead of recording assumptions that were never encoded.

The receipt hashes the complete selected source, reference, CIRCT binaries,
Z3 binary, produced query and logs. Verification re-extracts the selected
module, checks artifact hashes, regenerates the CIRCT and SMT queries with the
pinned binaries, and reruns Z3. The receipt is unsigned local evidence; replay
checks reproducibility and source binding, not who authored the reference.
The source, reference, proof implementation, CIRCT encodings, SMT translator
and Z3 are part of the trusted base. CIRCT and Z3 reason about two-state
bitvectors here; no claim is made about X/Z resolution, analog timing, reset,
clocking or a sequential protocol.

## Record simulation and bounded proofs separately

`phase0.cell_probe.characterize` saves selected-source and tool identities,
exact vectors, generated RTL, native harness, logs and per-output results. Its
`passed` status means only those saved vectors agreed with the independent
expected values. For example, the selected integer and floating-point cell
probes cover large but finite domains; operand formats, exceptional values and
accumulator cases outside those domains remain open. A native full-core/SoC
observation has a still different scope: one program, its environment, its
inputs and outputs. Neither a finite cell probe nor a structural fact may be
promoted to `proven` numerical semantics.

CIRCT also documents `circt-bmc` for sequential `verif.assert` properties and
`circt-lec` for logical equivalence. A bounded property receipt must state
the exact selected module, initial-state/reset assumptions, input constraints,
cycle bound, property text, tool and source hashes, solver result and a trace
on failure. Passing bound `N` does not establish behavior at `N+1` or over an
unbounded execution. A sequential property needs an invariant/induction or
another sound unbounded method before it can be called unboundedly proven.
The combinational runner above does not relabel a bounded check as such a proof.

## Connect hardware checks to compiler and Phase 2 claims

Review a Phase 0 numerical declaration against its proof's exact domain:
cell ports and arithmetic, interface sequencing, transfer semantics, command
encoding, and program execution are separate obligations. A cell proof can
support the corresponding arithmetic premise of an independent oracle; it
cannot validate all those other layers. Phase 1 still needs its own
transformation-equivalence receipts and executable differential tests for
every admitted operation signature and required model region. Phase 2 should
reuse the frozen Phase 0 tests as regression witnesses and add tests for each
optimization's changed assumptions, tails, layouts, aliases, transfer paths,
and numerical tolerances. A passing Phase 2 optimization test is scoped to its
frozen inputs and selected target; coverage of unseen PyTorch models or
operators requires separate inventory, explicit unsupported/fallback handling
and model-level qualification.

`phase0.hardware_validation.prove_combinational_property` owns the
combinational query and `verify_property_receipt` replays it. `cell_probe` owns
finite native vectors. The core compiler consumes only their declared evidence
scopes; neither result automatically selects a backend implementation or
discharges complete program execution.
