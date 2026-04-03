# cuda_tile Codegen — Developer Guide

## Adding a New Elementwise Op

To add support for a new elementwise operation (e.g. `math.tan`), make changes in **two files**, four lines total.

### 1. Pass: Recognize the MLIR op (`ConvertElementwiseToCudaTile.cpp`)

Add one line to `mapMathToCudaTile()` (or `mapArithToCudaTile()` for arith ops):

```cpp
// In mapMathToCudaTile():
if (isa<math::TanOp>(op)) return "tan";
```

This tells the pass to tag any `linalg.generic` containing this op as `cuda_tile.kernel_class = "elementwise"` with `cuda_tile.op_name = "tan"`.

### 2. Emitter: Create the cuda_tile dialect op (`CudaTileTarget.cpp`)

In the `CudaTileOpEmitter::emitElementwise()` method, add three things:

```cpp
// (a) Add to the EWOp enum:
enum class EWOp : uint8_t {
  ...
  Tan,   // <-- new
  ...
};

// (b) Add to the StringSwitch:
auto kind = llvm::StringSwitch<EWOp>(opName)
    ...
    .Case("tan", EWOp::Tan)   // <-- new
    ...

// (c) Add to the switch body:
switch (kind) {
    ...
    case EWOp::Tan: return b.create<cuda_tile::TanOp>(loc, a).getResult();  // <-- new
    ...
}
```

### Op Signature Reference

Check the cuda_tile op's builder signature in the generated header:
```
build/host-merlin-release/compiler/plugins/merlin/third_party/cuda-tile/
  include/cuda_tile/Dialect/CudaTile/IR/Ops.cpp.inc
```

Common patterns:
| Category | Signature | Example |
|----------|-----------|---------|
| Unary float (simple) | `(loc, operand)` | `NegFOp`, `ExpOp`, `SinOp` |
| Unary float (rounding) | `(loc, operand, RoundingModeAttr)` | `SqrtOp` |
| Binary float (rounding) | `(loc, lhs, rhs, RoundingModeAttr)` | `AddFOp`, `MulFOp` |
| Binary float (no rounding) | `(loc, lhs, rhs)` | `MaxFOp`, `PowOp` |
| Binary int (overflow) | `(loc, lhs, rhs, IntegerOverflowAttr)` | `AddIOp` |
| Binary int (signedness) | `(loc, Type, ValueRange, Signedness)` | `MaxIOp` |
| Ternary | `(loc, a, b, c, ...)` | `SelectOp`, `FmaOp` |

### Testing

Run the full test suite:
```bash
# Single op test
cat > /tmp/test.mlir << 'EOF'
func.func @tan(%x: tensor<16xf32>) -> tensor<16xf32> {
  %r = math.tan %x : tensor<16xf32>
  return %r : tensor<16xf32>
}
EOF

IC=build/host-merlin-release/tools/iree-compile
IR=build/host-merlin-release/tools/iree-run-module

# Compile for both backends
$IC /tmp/test.mlir --iree-hal-target-backends=cuda_tile \
  --iree-cuda-tile-sm-arch=sm_86 --iree-cuda-tile-enable-codegen=true \
  -o /tmp/test_ct.vmfb

$IC /tmp/test.mlir --iree-hal-target-backends=cuda \
  -o /tmp/test_cuda.vmfb

# Compare outputs
$IR --device=cuda_tile --module=/tmp/test_ct.vmfb --function=tan \
  --input="16xf32=[0,0.5,1,1.5,2,2.5,3,3.5,0,0.5,1,1.5,2,2.5,3,3.5]"

$IR --device=cuda --module=/tmp/test_cuda.vmfb --function=tan \
  --input="16xf32=[0,0.5,1,1.5,2,2.5,3,3.5,0,0.5,1,1.5,2,2.5,3,3.5]"
```

## How the Codegen Pipeline Works

```
linalg/arith/math/scf ops (IREE inner module)
  |
  v  [ConvertDataMovement/Elementwise/Contractions passes]
  |  Tag ops with cuda_tile.kernel_class + metadata attributes
  |
  v  [buildCudaTileKernel() in CudaTileTarget.cpp]
  |  Read tags, dispatch to kernel generator
  |  Build cuda_tile dialect ops via OpBuilder (CudaTileOpEmitter)
  |
  v  [BytecodeWriter::serialize()]
  |  cuda_tile ModuleOp -> tilebc bytes (in-process)
  |
  v  [compileWithTileiras()]
  |  tilebc -> cubin (external process)
  |
  v  [CTL1 FlatBuffer serialization]
  |  cubin -> .vmfb
  v
  GPU execution via cuda_tile HAL driver
```

## Adding Other Op Classes

| Phase | Pass file | Kernel generator | Status |
|-------|-----------|-----------------|--------|
| Data movement | `ConvertDataMovementToCudaTile.cpp` | `generateCopyKernel`, `generateTransposeKernel` | Working |
| Elementwise | `ConvertElementwiseToCudaTile.cpp` | `emitElementwise()` switch table | Working |
| Reductions | `ConvertReductionsToCudaTile.cpp` | `generateReduceKernel` (TODO) | Stub |
| Contractions | `ConvertContractionsToCudaTile.cpp` | `generateMatmulKernel` (TODO: for-loop iter_values) | Partial |
| SCF/Loops | `ConvertSCFToCudaTile.cpp` | TODO | Stub |
