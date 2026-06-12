"""Whole-model lowering: linalg-on-tensors MLIR -> LLVM IR -> RVV objects.

The llvm-project plane (uses upstream MLIR/LLVM via the torch-mlir wheel + clang).
Stages:

1. xDSL preprocessing (`passes_xdsl`): lower model2MLIR's `quant_ext.dequantize_per_channel`
   to `linalg.generic`, request C wrappers, future scf.parallel -> merlin_parallel_for.
2. Upstream lowering + translation (`pipeline`): bufferize -> loops -> LLVM dialect ->
   LLVM IR, run in the model2MLIR venv (torch-mlir wheel ships LLVM 23 passes).
3. Codegen (`codegen`): clang -O2 to x86 (host check) or rv64gcv (spike/Zephyr).
4. Weights (`weights_pack`): safetensors -> flat blob + offset table (no embedding).

Toolchain locations (env-overridable): MERLIN_M2M_VENV, MERLIN_CLANG, MERLIN_M2M_DIR.
"""
