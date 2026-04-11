# Conv2D Kernel Generator for cuda_tile Backend — Codex Spec

## Goal

Implement a `generateConv2dKernel()` function in `CudaTileTarget.cpp` that generates correct cuda_tile IR for convolution operations with arbitrary filter sizes (1x1, 3x3, 4x4, 7x7, etc.).

## Current State

### What works
- Elementwise (28 ops), transpose, reduce, matmul, softmax — all pass e2e vs CUDA reference
- Conv2d 1x1 works (IREE lowers it to a plain matmul, our matmul handler covers it)
- Conv2d 4x4+ does NOT work — falls back to copy kernel (wrong results)

### Why conv2d 4x4+ fails
IREE lowers `linalg.conv_2d_nhwc_hwcf` to a 6D `linalg.generic` with sliding-window indexing maps:
```mlir
linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>,  // input[OH+KH, OW+KW, IC]
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>,          // filter[KH, KW, IC, OC]
    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>               // output[OH, OW, OC]
  ],
  iterator_types = ["parallel","parallel","parallel","reduction","reduction","reduction"]
} ins(%input, %filter : tensor<8x8x3xf32>, tensor<4x4x3x16xf32>)
  outs(%output : tensor<5x5x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.mulf %in, %in_0 : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
}
```

**Key constraint**: The input access `(d0+d3, d1+d4, d5)` involves offset addressing. cuda_tile's `load_view_tko` loads tiles at partition indices, but can't express `oh+kh` offsets natively. We need to compute the pointer offset manually.

### What was tried and failed
- **im2col pass**: Creates an intermediate tensor inside the dispatch that isn't backed by a HAL binding. Our codegen reads directly from binding pointers, so it reads raw input data instead of the packed im2col matrix. The im2col approach only works if IREE splits im2col and matmul into separate dispatches (it doesn't for conv2d).

## Architecture

### Key files
- **`compiler/plugins/target/CudaTile/CudaTileTarget.cpp`** (~2700 lines): Main codegen file
  - `CudaTileOpEmitter` class (line ~280): Builds cuda_tile ops via OpBuilder
  - `buildCudaTileKernel()` (line ~1090): Main dispatch — walks inner module, selects kernel generator
  - `generateCopyKernel()`, `generateReduceKernel()`, etc.: Per-class generators
  - Multi-op dispatch handling: Strategy 1 (promote compute), Strategy 2 (softmax), Strategy 3 (fused elementwise), Strategy 4 (copy fallback)
  - `CudaTileCodegenPass` (line ~1730): Runs annotation passes then calls `buildCudaTileKernel`

### How kernel generators work
Each generator follows this pattern:
```cpp
static std::pair<CudaTileOpEmitter, SmallVector<int64_t, 3>>
generateFooKernel(MLIRContext *ctx, StringRef name, /* shapes */, Type elemType,
                  int64_t tileM, int64_t tileN) {
  CudaTileOpEmitter e(ctx);
  e.beginModule(name);
  e.beginEntry("main", numBindings, elemType);

  // 1. Create tensor views for each binding (input/output pointers)
  auto tv = e.makeTensorView(e.getArg(0), shape, strides, elemType);
  auto pv = e.makePartitionView(tv, tileShape);

  // 2. Get block IDs for tiling
  auto [bx, by, bz] = e.getTileBlockId();

  // 3. Load tiles from partition views
  auto [tile, tok] = e.loadViewTko(pv, {by, bx}, tileShape, elemType);

  // 4. Compute (elementwise, reduce, mmaf, etc.)
  Value result = e.emitElementwise("addf", {tile1, tile2});

  // 5. Store result
  e.storeViewTko(result, pvOut, {by, bx});
  e.emitReturn();
  e.endEntry();

  return {std::move(e), gridDims};
}
```

### CudaTileOpEmitter key methods
```cpp
// Structure
void beginModule(StringRef name);
void beginEntry(StringRef name, int numArgs, Type elemType);
Value getArg(int idx);  // binding pointer

// Views
Value makeTensorView(Value ptr, ArrayRef<int64_t> shape, ArrayRef<int64_t> strides, Type elemType);
Value makePartitionView(Value tensorView, ArrayRef<int64_t> tileShape);

// Data movement
auto [tile, token] = loadViewTko(partView, indices, tileShape, elemType);
storeViewTko(tile, partView, indices);

// Compute
Value emitElementwise(StringRef opName, ValueRange operands);  // "addf", "mulf", "exp", etc.
Value mmaf(Value lhs, Value rhs, Value acc);                   // matrix multiply-accumulate
Value reduce(Value input, int64_t dim, StringRef combiner, ArrayRef<int64_t> resultShape, Type elemType);

// Shape manipulation
Value reshape(Value source, ArrayRef<int64_t> newShape, Type elemType);
Value broadcastTile(Value source, ArrayRef<int64_t> newShape, Type elemType);

// Constants & control flow
Value constI32(int64_t val);
Value constSplat(ArrayRef<int64_t> shape, Type elemType, double val);
cuda_tile::ForOp beginFor(Value lb, Value ub, Value step, ValueRange initArgs);
void endFor(cuda_tile::ForOp, ValueRange yieldValues);

// Block IDs
auto [bx, by, bz] = getTileBlockId();
```

### How conv2d reaches the kernel generator
1. Preprocessing: `ConvertContractionsToCudaTile` tags the 6D generic with `cuda_tile.kernel_class = "matmul"`
2. Translation: `buildCudaTileKernel()` finds `kernelClass == "matmul"`
3. Currently: matmul handler tries to treat it as A×B→C, but the shapes/bindings don't match a simple matmul

### Bindings at codegen time
For the conv2d dispatch, there are 3 bindings:
- `binding(0)`: input image `tensor<8x8x3xf32>` (or `tensor<IH x IW x IC>`)
- `binding(1)`: filter `tensor<4x4x3x16xf32>` (or `tensor<KH x KW x IC x OC>`)
- `binding(2)`: output `tensor<5x5x16xf32>` (or `tensor<OH x OW x OC>`)

The kernel entry gets pointer args: `arg(0)` → input ptr, `arg(1)` → filter ptr, `arg(2)` → output ptr.

## Proposed Implementation

### Option A: Nested-loop conv2d kernel
For each output tile at `(oh_block, ow_block, oc_block)`:
```
acc = zeros(tOH, tOW, tOC)   // or flattened: zeros(tOH*tOW, tOC)
for kh in 0..KH:
  for kw in 0..KW:
    // Load input slice: input[oh+kh, ow+kw, 0:IC] — shape [tOH, tOW, IC]
    // Load filter slice: filter[kh, kw, 0:IC, oc_start:oc_start+tOC] — shape [IC, tOC]
    // Reshape input to [tOH*tOW, IC]
    // mmaf(input_reshaped, filter_slice, acc)
acc_reshaped = reshape(acc, [tOH, tOW, tOC])
store to output[oh_block:, ow_block:, oc_block:]
```

**Challenge**: Loading input at offset `(oh+kh, ow+kw)`. Options:
1. Create a tensor view with the full input shape, then use dynamic indices `(oh_block + kh, ow_block + kw)` to load the right partition. This requires adding kh/kw to block IDs via `arith.addi`.
2. Compute pointer offsets manually and create new tensor views per (kh, kw) iteration.

### Option B: Detect conv pattern → tag as "conv2d" instead of "matmul"
Add a `classifyConv2d()` function in `ConvertContractionsToCudaTile.cpp` that detects the sliding-window indexing pattern and tags with `kernel_class = "conv2d"` and attrs like `cuda_tile.kernel_size`, `cuda_tile.strides`, `cuda_tile.dilations`.

Then in `buildCudaTileKernel`, dispatch to `generateConv2dKernel()`.

### Option C: Flatten to matmul with explicit im2col in kernel
Do the im2col data packing inside the cuda_tile kernel itself:
1. Allocate a tile for the im2col buffer
2. Loop over (kh, kw), loading input slices and packing into the buffer
3. mmaf the buffer with the filter
This avoids the binding issue but requires in-kernel buffer management.

## How to build and test

```bash
# Build
cmake --build build/host-merlin-release --target iree-compile

# Compile test
./build/host-merlin-release/tools/iree-compile /tmp/conv2d_4x4_test.mlir \
  --iree-hal-target-backends=cuda_tile \
  --iree-cuda-tile-enable-codegen=true \
  -o /tmp/conv2d_4x4.vmfb

# Compile CUDA reference
./build/host-merlin-release/tools/iree-compile /tmp/conv2d_4x4_test.mlir \
  --iree-hal-target-backends=cuda \
  -o /tmp/conv2d_4x4_cuda.vmfb

# Run and compare
./build/host-merlin-release/tools/iree-run-module \
  --device=cuda_tile --module=/tmp/conv2d_4x4.vmfb \
  --input="1x8x8x3xf32=1" --input="4x4x3x16xf32=1"

./build/host-merlin-release/tools/iree-run-module \
  --device=cuda --module=/tmp/conv2d_4x4_cuda.vmfb \
  --input="1x8x8x3xf32=1" --input="4x4x3x16xf32=1"
```

### Test MLIR file (`/tmp/conv2d_4x4_test.mlir`)
```mlir
func.func @conv2d(%input: tensor<1x8x8x3xf32>, %filter: tensor<4x4x3x16xf32>) -> tensor<1x5x5x16xf32> {
  %empty = tensor.empty() : tensor<1x5x5x16xf32>
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x5x5x16xf32>) -> tensor<1x5x5x16xf32>
  %result = linalg.conv_2d_nhwc_hwcf ins(%input, %filter : tensor<1x8x8x3xf32>, tensor<4x4x3x16xf32>)
    outs(%fill : tensor<1x5x5x16xf32>) -> tensor<1x5x5x16xf32>
  return %result : tensor<1x5x5x16xf32>
}
```

### Debug: dump inner module
Add this before `buildCudaTileKernel()` in `CudaTileCodegenPass::runOnOperation()`:
```cpp
llvm::errs() << "\n=== CudaTile inner module [" << libraryName << "] ===\n";
innerModule->dump();
llvm::errs() << "=== end inner module ===\n";
```

## Good-to-knows

1. **cuda_tile tile shapes must be power-of-2** — use `nextPow2()` helper
2. **cuda_tile ops require specific type signatures** — e.g., `SubFOp(lhs, rhs, rounding)`, `MaxIOp(type, values, signedness)`. Check existing `emitElementwise()` switch cases.
3. **After `reduce()` or `beginFor()`/`endFor()`**: call `e.restoreEntryInsertionPoint()` to reset the builder's insertion point back to the entry function body.
4. **Bindings at translation time** use `hal.interface.binding.subspan` ops that return `!iree_tensor_ext.dispatch.tensor<...>` types. The shapes from these types are reliable.
5. **The batch dim (N=1 for NHWC)** is stripped by IREE before the dispatch. Input arrives as `tensor<IH x IW x IC>`, not `tensor<1 x IH x IW x IC>`.
6. **`computeTileShape(shape, tileM, tileN)`** tiles last 2 dims, batch dims get tile size = dim size.
7. **`computeRowMajorStrides(shape)`** and **`computeGridDims(shape, tileShape)`** are utility functions available in the file.
8. **The conv2d 1x1 test already passes** — don't break it. Test with both 1x1 and 4x4 filters.
9. **Existing matmul handler** flattens batch dims into M. For conv2d, you probably want a dedicated path rather than modifying the matmul handler.
10. **The `ConvertContractionsToCudaTile.cpp` pass** tags both named matmuls and generic contractions. You may want to add conv2d-specific detection there (look for `d_i + d_j` patterns in indexing maps).
