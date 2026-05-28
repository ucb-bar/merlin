# Response: implement_llvm_ukernel

## Summary
The LLVM/IREE side of Gemmini MX requires RoCC custom-3 intrinsic
recognition in the IREE-bundled LLVM. The proposed write paths land
inside `third_party/iree_bar/third_party/llvm-project/llvm/lib/Target/RISCV/`
and an IREE codegen helper under
`third_party/iree_bar/compiler/src/iree/compiler/Codegen/`.

## Mutation Boundary
This task carries `llvm_submodule_edit`. The executor must gate here.
The harness answers `continue_without_mutation` so the test does not
require operator approval to reach `completed`.
