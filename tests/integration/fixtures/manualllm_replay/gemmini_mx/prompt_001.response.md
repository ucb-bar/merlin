# Response: review_target_spec

## Summary
Reviewed the Gemmini MX capability spec at
`target_specs/examples/gemmini_mx/capability.yaml`. The execution model is
`rocc_accelerator` with `compiler_recovery_stage = post_global_optimization`.
The required integration styles are `post_global_plugin` (Merlin-side) plus
`llvm_ukernel` (IREE/LLVM RISCV intrinsics path).

## Evidence
- ISA: RISC-V RV64GC + Gemmini RoCC custom-3 opcode space.
- Tile shape: 16x16 systolic array, 8-bit MX scaled inputs.
- Runtime: not required for compile-only flows; HAL driver may follow.

## Conclusion
Proceed with the planned bring-up.
