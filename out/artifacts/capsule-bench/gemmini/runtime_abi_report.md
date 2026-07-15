# Runtime ABI report (capsule_bench_v0)

Both the Merlin-assisted package and any baseline interact with the runner through the identical external boundary:

**Package owns** (4 CLI entrypoints, subprocess-invoked):
- `parse` — parse/verify capsule.interface.mlir
- `lower_interface_to_target` — interface MLIR → gemmini target dialect (parses+verifies)
- `emit_command_buffer` — gemmini → command_buffer.json (+ im2col recipe for conv)
- `lower_target_to_llvm` — gemmini → LLVM/RoCC MLIR (`.insn r 0x7b`)

**Runner owns:** deterministic inputs, numpy/Tensor golden, command-buffer reference + simulator, final LLVM→object/ELF, C harness, link, spike/verilator/VCS/FireSim invocation, RoCC trace decode + check, numeric compare, run/artifact manifests, iteration history.

**The runner never** implements matmul/conv/relu/scale as the answer, calls bareMetalC, or calls a high-level Gemmini kernel; the device implementation is the package's MLIR-lowered RoCC.

Instruction trace is decoded by the runner from the package's emitted `lowered.llvm.mlir` (`rocc_decode`), keeping it a parity-fair observation applicable identically to a baseline.
