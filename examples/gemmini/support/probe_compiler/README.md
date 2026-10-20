# gemmini ISA probe compiler (a probe, not a solution)

This package exists to prove Merlin's grading path end to end on the generic, data-only gemmini
support (`examples/gemmini/support`, served by `merlin.runtime.backends.chipyard_rocc`). It is **not**
a compiler, a reference backend, or an answer to any experiment, and it must never be granted to a
candidate or used as a baseline.

- It accepts exactly two interface shapes and refuses everything else: one DIMxDIM i8 movement
  (`isa/A1_mvin_mvout`) and one DIMxDIM i8 x i8 -> i32 single-tile matmul
  (`isa/A2_single_tile_matmul`). No tiling, no epilogue, no other operation.
- Every encoding constant is read from `gemmini_isa.h`, a snapshot of the header Merlin generates
  from the verified RTL facts and the target contract (`merlin.targetgen.isa_header_gen`; RTL facts
  sha256 recorded in its first comment). The protocol choices the header cannot state -- weight-
  stationary dataflow value, unit float scales, the all-ones "no operand" address -- are named
  constants in `probe_compiler.py`, taken from the public Gemmini ISA description.
- It imports nothing from Merlin (`integrity_exempt: false`) and contains no code from any
  other gemmini support package or compiler.
- `manifest.yaml` exposes the four experiment-ABI commands; the tool emits an LLVM-dialect module
  of raw `.insn r` RoCC instructions for `gemmini_kernel` under the logical kernel ABI v2.
