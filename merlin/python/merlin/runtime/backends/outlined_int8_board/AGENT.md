# AGENT.md — merlin/python/merlin/runtime/backends/outlined_int8_board

## Purpose

BOARD RVV backend for validating outlined INT8 GEMM kernels independently of the
whole-model lowering path.

## Invariants

- Preserve signed INT8 inputs and INT32 accumulation semantics.
- Treat this backend as a correctness and attribution probe, not as an implicit
  whole-model route.
- Keep the C kernel and Python binding ABI synchronized.

## Testing expectations

Run `merlin/tests/runtime/test_outlined_int8_board.py` after changing the binding
or kernel source.
