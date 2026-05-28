# Response: derive_compile_view

## Summary
Compile view for Gemmini MX: lower `linalg.matmul` to a Gemmini dialect op
recovered after IREE global optimization, then translate to RoCC intrinsic
calls in the LLVM backend (no new feature bits required).

## Compile View
```
linalg.matmul (i8) -> gemmini.matmul (tile-shaped)
  -> rocc.intrinsic
```

## Conclusion
Compile view recorded. No operator gate required.
