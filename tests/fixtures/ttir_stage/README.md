# Pinned compiler-only TTIR input

This small source was captured with Triton 3.7.1 (`f797708c`) and converted
by triton2MLIR `b828d3c`. The kernel loads one f32 word, adds one, and stores
the result. Its input/output allocations each contain one element.

The source was:

```python
@triton.jit
def staged_increment(x, out):
    value = tl.load(x)
    tl.store(out, value + 1)
```

The fixture retains the actual TTIR, standard MLIR, ABI and compiler-relevant
provenance. Tool location/execution reports and evaluator archives are excluded.
Tests create the empty parameter container and receipt from these exact bytes.
This fixture protects compiler intake, source/ABI custody and public preparation;
it does not certify original-source numerical equivalence or a hardware target.
Real native/public-compiler execution is qualified separately by triton2MLIR's
public kernel staging tests.
