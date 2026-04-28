# 2026-04-28: cuda_tile AlexNet Handoff

## Context and Goal

The cuda_tile compiler backend now compiles AlexNet end-to-end (no-adaptive
variant) but the output diverges numerically from the cuda reference. This
entry hands the debug session off to a fresh agent so it can pick up on the
"garden" dev machine without re-deriving context.

Branch: `HAL/cuda_tileir` on `git@github.com:ucb-bar/merlin.git`. Latest
commit at handoff time: `eeba18e` ("[CudaTile] Generalize fused dispatches:
matmul+bias+relu, multi-op walker, byte-offset bindings").

Test ladder we are working through (see
`/home/eecs/ashvin.verma/.claude/projects/-scratch-ashvin/memory/cuda_tile_test_targets.md`):

1. linear
2. linear + relu + bias
3. softmax
4. conv + MLP (linear + bias + relu)
5. alexnet
6. Transformer Encoder Layer

Steps 1-3 verified. Step 5 compiles but is wrong. Step 4 not yet exercised
in isolation; that is the next thing to break out.

## Implementation Changes (committed in `eeba18e`)

CudaTileKernelPlan
- Add `CudaTileFusedOpRole {Unknown, Primary, Prologue, Epilogue}` and
  `classifyFusedOpRoles` to mark each fused linalg op as input-side or
  output-side relative to the primary op.
- Track per-subspan `byteOffset` / `hasStaticByteOffset` on
  `CudaTileBindingPlan`. Real models reuse buffer bindings at non-zero
  offsets (e.g. weights and biases sharing binding 1 at different offsets);
  codegen now applies these via `getSubspanArg`/`getBindingArg`.

CudaTileTarget
- Generalized post-matmul fusion epilogue walker: any number of
  parallel-only generics following a matmul are emitted in dataflow
  order (bias + relu, scale, etc.).
- Unified multi-op fused dispatch path that walks tagged ops dynamically
  rather than hardcoding softmax (Strategy 2) and elementwise fusion
  (Strategy 3) shapes.
- Unsupported-dispatch fallback: dispatches containing `linalg.index` or
  `tensor.extract` emit a no-op kernel + warning rather than crashing
  (covers AlexNet's `adaptive_avg_pool2d` lowering until proper handling
  lands).
- Boilerplate, broadcast load path, and post-matmul bias loading all
  thread `bindingShapes` through `getBindingArg`/`getSubspanArg` so
  subspan offsets are honored end-to-end.
- Integer arith ops (and/or/xor) added to `mapArithToCudaTileLocal`.

Tests: new `plan_dump.mlir` cases for elementwise + reduce_sum exercise the
fused-op role classifier and operand plan dump.

tools / docs: `--debug-dumps` convenience flag combining `--dump-phases`
and `--dump-artifacts`; minor index / how-to updates.

## What Worked

| Test | Result |
|------|--------|
| Standalone `linalg.matmul` (e.g., 1x4096x4096) | max_err < 2e-5 vs cuda |
| Fused **matmul + bias + relu** (`/tmp/test_matmul_bias_relu.mlir`, 1x512 @ 512x256) | max_err **1.3e-5** vs cuda |
| Softmax (4x8) | 0 err vs cuda |
| AlexNet compiles end-to-end through cuda_tile | yes (12 dispatches) |

## What Did Not Work

AlexNet (no-adaptive variant) accuracy:
- Test driver: `/tmp/export_alexnet_noadaptive.py` →
  `/tmp/alexnet_noadaptive.mlir`. The script replaces
  `nn.AdaptiveAvgPool2d((6,6))` with `nn.Identity()` because the adaptive
  pool dispatch lowers to `linalg.index` + `tensor.extract`, which we
  currently emit as a no-op kernel. AlexNet's input-by-construction is
  already 6x6 at that stage so the identity substitution is semantically
  fine.
- Symptom: cuda_tile output range ~13x wider than cuda reference
  (cuda_tile `std ≈ 0.13`, cuda `std ≈ 0.01`). `max_err ≈ 0.40` over 1000
  logits. Top-5 predictions disagree completely.
- Output magnitude is *amplified*, not just shifted, so one or more
  dispatches are computing wrong magnitudes — probably a conv (each conv
  = im2col → matmul) or one of the 3 linear layers (4096→4096→1000).
- The fused `matmul+bias+relu` itself is verified correct in isolation,
  so the bug is likely in *another* dispatch shape that AlexNet hits.

False alarm previously diagnosed:
- An earlier session reported the fused matmul+bias+relu producing all-1s
  on cuda_tile vs 5.62 on cuda. After rebuilding `iree-compile` and
  re-running, both backends agreed. The earlier failure was a stale
  `.vmfb`. Lesson: always rebuild `iree-compile` before trusting a
  comparison between backends.

`iree-run-module` `.npy` input/output is broken on this build — both
backends produce identical garbage when fed `.npy`. Always use raw
`.bin` files (`numpy.tofile` and `numpy.fromfile`).

## Debugging Notes

`/scratch` on a19 is regularly 100% full (1.8T/1.9T from other users),
which can break the `uv run tools/merlin.py build` flow. Workarounds:
- Set `UV_CACHE_DIR=/tmp/uv-cache` (we did this).
- Delete unused build dirs first: e.g. `host-merlin-debug` (~3.4 GB) is
  not needed if you build `host-merlin-release`.
- Do not delete `cuda-tile-tools/` (~6 GB build of tileiras/tilebc — the
  cuda_tile dispatch dialect needs them at link/run time).

Useful environment variables / flags:
- `IREE_CUDA_TILE_DUMP_TILE_IR=1` — print the cuda_tile MLIR per dispatch
  to stderr.
- `iree-compile --compile-to=executable-sources -o file.mlir` — see the
  dispatch IR before our codegen runs.
- `iree-compile --mlir-print-ir-after-all -o /dev/null 2>&1 | grep -B1 -A40 'kernel_class\|hal.executable.variant'`
  — trace through pipeline phases.

## Test Coverage and Commands

Reproduce the AlexNet failure:

```bash
IC=build/host-merlin-release/tools/iree-compile
IR=build/host-merlin-release/tools/iree-run-module

# Export the no-adaptive variant once.
/scratch/ashvin/merlin/.venv/bin/python /tmp/export_alexnet_noadaptive.py
# (or use conda env merlin-dev if it exists on this machine)

# Compile both backends.
$IC /tmp/alexnet_noadaptive.mlir \
    --iree-hal-target-backends=cuda_tile \
    --iree-cuda-tile-enable-codegen=true \
    -o /tmp/alexnet_ct.vmfb
$IC /tmp/alexnet_noadaptive.mlir \
    --iree-hal-target-backends=cuda \
    -o /tmp/alexnet_cuda.vmfb

# Generate a deterministic input.
python3 -c "
import numpy as np
np.random.seed(0)
np.random.randn(1,3,224,224).astype(np.float32).tofile('/tmp/alexnet_in.bin')"

# Run both, compare.
$IR --module=/tmp/alexnet_ct.vmfb --device=cuda_tile \
    --input=1x3x224x224xf32=@/tmp/alexnet_in.bin \
    --output=@/tmp/alexnet_ct.bin
$IR --module=/tmp/alexnet_cuda.vmfb --device=cuda \
    --input=1x3x224x224xf32=@/tmp/alexnet_in.bin \
    --output=@/tmp/alexnet_cuda.bin

python3 -c "
import numpy as np
ct = np.fromfile('/tmp/alexnet_ct.bin', dtype=np.float32)
cuda = np.fromfile('/tmp/alexnet_cuda.bin', dtype=np.float32)
print('max_err:', np.abs(ct-cuda).max())
print('ct top5:', np.argsort(-ct)[:5])
print('cuda top5:', np.argsort(-cuda)[:5])"
```

Reproduce the fused matmul+bias+relu pass (sanity check that the build is
healthy before touching anything else):

```bash
cat > /tmp/test_matmul_bias_relu.mlir <<'EOF'
func.func @matmul_bias_relu(%a: tensor<1x512xf32>, %b: tensor<512x256xf32>, %bias: tensor<1x256xf32>) -> tensor<1x256xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x256xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x256xf32>) -> tensor<1x256xf32>
  %mm = linalg.matmul ins(%a, %b : tensor<1x512xf32>, tensor<512x256xf32>)
    outs(%fill : tensor<1x256xf32>) -> tensor<1x256xf32>
  %out = tensor.empty() : tensor<1x256xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm, %bias : tensor<1x256xf32>, tensor<1x256xf32>) outs(%out : tensor<1x256xf32>) {
  ^bb0(%in: f32, %b_val: f32, %o: f32):
    %add = arith.addf %in, %b_val : f32
    %cmp = arith.cmpf ugt, %add, %zero : f32
    %relu = arith.select %cmp, %add, %zero : f32
    linalg.yield %relu : f32
  } -> tensor<1x256xf32>
  return %result : tensor<1x256xf32>
}
EOF

$IC /tmp/test_matmul_bias_relu.mlir \
    --iree-hal-target-backends=cuda_tile \
    --iree-cuda-tile-enable-codegen=true \
    -o /tmp/mbr_ct.vmfb
$IR --module=/tmp/mbr_ct.vmfb --device=cuda_tile \
    --input="1x512xf32=1" --input="512x256xf32=1" --input="1x256xf32=1"
# Expect: 1x256xf32=[513 513 ... ]
```

## Follow-Up Tasks

Recommended next step: walk the test ladder and break AlexNet down into
hand-written MLIR per layer. Compare cuda_tile vs cuda for each:

1. Linear 9216 → 4096 + relu + bias (classifier first layer; M=1, large K).
2. Linear 4096 → 4096 + relu + bias (classifier hidden).
3. Linear 4096 → 1000 + bias only, no relu (classifier output).
4. Conv2d 11x11 stride=4 padding=2 (first conv — most exotic shapes).
5. Conv2d 5x5 padding=2 and 3x3 padding=1.
6. MaxPool2d k=3 stride=2.

For each broken op, dump the tile-IR with
`IREE_CUDA_TILE_DUMP_TILE_IR=1` and compare with the dispatch IR
(`--compile-to=executable-sources`).

Also outstanding:
- Properly handle `adaptive_avg_pool2d` (currently no-op kernel).
- Clean up debug `[reduce-debug]` prints and at least one stale `abort()`
  in `makePartitionView` paths in `CudaTileTarget.cpp`.
- Address remaining `CODEX_REQ.md` items (multi-output dispatch, identity
  pool elimination).

## Submodule State

The handoff commit only includes top-level repo changes. These submodules
have uncommitted local edits that were *not* pushed and look like
unrelated WIP from other workstreams; leave them alone or pin to whatever
the new machine's submodule HEADs already are:

- `third_party/iree_bar` — local edits to
  `compiler/src/iree/compiler/Codegen/LLVMCPU/KernelDispatch.cpp`,
  `compiler/src/iree/compiler/ExternalInterfaces/LinalgExtExternalModels.cpp`,
  `runtime/src/iree/schemas/CMakeLists.txt`, plus an untracked
  `runtime/src/iree/schemas/cuda_tile_executable_def.fbs`.
- `projects/mlirAgent` — unrelated agent harness work.

## Pull Command (for the new agent)

```bash
cd /path/to/merlin
git fetch origin
git checkout HAL/cuda_tileir
git pull --ff-only origin HAL/cuda_tileir
# Optional: refresh submodules to whatever the commit pins
git submodule update --init --recursive third_party/iree_bar
```

Then read this file (`docs/dev_blog/2026-04-28-cuda-tile-alexnet-handoff.md`)
and the test-targets memory note before touching code.
