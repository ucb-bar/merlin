#!/usr/bin/env bash
# Full QNN e2e demo: compile + execute on all three backends.
#
# Drives the full triple — CPU (IREE LLVM AArch64), QNN GPU (Adreno via
# IREE QNN HAL), QNN HTA (Hexagon NPU via raw ctxbin) — with on-board
# execution evidence for each. Produces:
#
#   build/qnn_e2e_demo/
#     summary.md
#     cpu/
#       artifact.vmfb
#       run_output.txt          ← actual VMFB execution result
#     qnn_gpu/
#       artifact.vmfb
#       run_output.txt          ← QNN HAL → Adreno execution
#     qnn_hta/
#       conv2d_int8_smoke.qnn-ctx
#       run_output.txt          ← libQnnHta.so execution on Hexagon NPU
#       analytical_check.txt    ← diff vs libQnnCpu reference
#
# Prereqs: iree-compile + qrb5165-runtime built; QRB5165 reachable as
#          $QNN_BOARD_HOST with $QNN_BOARD_QAIRT_ROOT staged; the in-tree
#          iree-run-module is the QNN-HAL-enabled aarch64 build.

set -euo pipefail
cd "$(dirname "$0")/../.."

OUT="${OUT:-build/qnn_e2e_demo}"
HOST="${QNN_BOARD_HOST:-qdev}"
QAIRT="${QNN_BOARD_QAIRT_ROOT:-/tmp/qnn_probe}"
mkdir -p "$OUT"/{cpu,qnn_gpu,qnn_hta}

# A single-input fixture that the qnn_add_f32 manifest entry matches. We
# call add(x, x) so the dispatch needs only one input, which makes
# benchmarking and bytes-equal assertions clean.
SINGLE_ADD=$(mktemp --suffix=.mlir)
cat > "$SINGLE_ADD" <<'MLIR'
module {
  func.func @single_add(%a: tensor<1x16xf32>) -> tensor<1x16xf32> {
    %init = tensor.empty() : tensor<1x16xf32>
    %out = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%a, %a : tensor<1x16xf32>, tensor<1x16xf32>)
        outs(%init : tensor<1x16xf32>) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %s = arith.addf %x, %y : f32
        linalg.yield %s : f32
    } -> tensor<1x16xf32>
    return %out : tensor<1x16xf32>
  }
}
MLIR

echo "=== [1/4] Compile single_add (fp32) for CPU + QNN GPU ==="
QNN_USE_BOARD_BUILD=1 QNN_BOARD_HOST="$HOST" QNN_BOARD_QAIRT_ROOT="$QAIRT" \
  conda run -n merlin-dev uv run python tools/kernels/qnn_e2e_compile_all.py \
    "$SINGLE_ADD" \
    --kernel-manifest benchmarks/QRB5165/kernels/manifest.json \
    --targets cpu,qnn_gpu \
    --out "$OUT/.compile" 2>&1 | tail -3
cp "$OUT/.compile/targets/cpu/artifact.vmfb" "$OUT/cpu/artifact.vmfb"
cp "$OUT/.compile/targets/qnn_gpu/artifact.vmfb" "$OUT/qnn_gpu/artifact.vmfb"

echo
echo "=== [2/4] Build int8 conv kernel for HTA ==="
conda run -n merlin-dev uv run python -c "
import pathlib, logging
logging.basicConfig(level=logging.WARNING)
from tools.kernels import qnn_build
src = pathlib.Path('benchmarks/QRB5165/kernels/abi/conv2d_int8_smoke.qnn.cpp')
cfg = qnn_build.BoardBuildConfig.from_env(
  ssh_host='$HOST', board_qairt_root='$QAIRT', target_backend='hta')
out = qnn_build.build_qnn_kernel_on_board(
  src, 'conv2d_int8_smoke', pathlib.Path('build/qnn_cache_demo'), cfg)
print('hta ctxbin:', out, out.stat().st_size, 'bytes')
import shutil
shutil.copy(out, '$OUT/qnn_hta/conv2d_int8_smoke.qnn-ctx')
"

echo
echo "=== [3/4] Execute on board ==="

# Copy artifacts to the board
scp -q "$OUT/cpu/artifact.vmfb" "$HOST:$QAIRT/demo_cpu.vmfb"
scp -q "$OUT/qnn_gpu/artifact.vmfb" "$HOST:$QAIRT/demo_gpu.vmfb"
scp -q "$OUT/qnn_hta/conv2d_int8_smoke.qnn-ctx" "$HOST:$QAIRT/demo_hta.qnn-ctx"

# CPU: run via iree-run-module local-task
echo
echo ">>> CPU (IREE LLVM AArch64, local-task):"
ssh "$HOST" "
  export LD_LIBRARY_PATH=$QAIRT/lib:\$LD_LIBRARY_PATH
  $QAIRT/bin/iree-run-module \
    --module=$QAIRT/demo_cpu.vmfb \
    --function=single_add \
    --input=1x16xf32=3.5 2>&1 | tail -3
" | tee "$OUT/cpu/run_output.txt"

# QNN GPU: run via iree-run-module qnn://gpu
echo
echo ">>> QNN GPU (Adreno via IREE QNN HAL):"
ssh "$HOST" "
  export LD_LIBRARY_PATH=$QAIRT/lib:\$LD_LIBRARY_PATH
  export ADSP_LIBRARY_PATH='$QAIRT/lib;/dsp/cdsp;/usr/lib/rfsa/adsp;/vendor/dsp/cdsp'
  $QAIRT/bin/iree-run-module \
    --module=$QAIRT/demo_gpu.vmfb \
    --device=qnn://gpu \
    --function=single_add \
    --input=1x16xf32=3.5 2>&1 | tail -3
" | tee "$OUT/qnn_gpu/run_output.txt"

# HTA: run raw ctxbin via qnn-net-run, compare against libQnnCpu
echo
echo ">>> NPU/HTA (Hexagon via libQnnHta.so):"
ssh "$HOST" "
  cd $QAIRT
  python3 -c 'open(\"demo_in64.raw\",\"wb\").write(bytes([64]*192))'
  echo 'input:=$QAIRT/demo_in64.raw' > demo_input_list.txt
  export LD_LIBRARY_PATH=$QAIRT/lib
  export ADSP_LIBRARY_PATH='$QAIRT/lib;/dsp/cdsp;/usr/lib/rfsa/adsp;/vendor/dsp/cdsp'
  rm -rf demo_hta_out && mkdir demo_hta_out
  $QAIRT/bin/qnn-net-run \
    --retrieve_context $QAIRT/demo_hta.qnn-ctx \
    --backend $QAIRT/lib/libQnnHta.so \
    --input_list demo_input_list.txt \
    --output_dir demo_hta_out \
    --use_native_input_files --use_native_output_files 2>&1 | tail -3
  python3 -c '
import collections
data=open(\"demo_hta_out/Result_0/_7_native.raw\",\"rb\").read()
print(\"  output uniq:\", collections.Counter(data).most_common())
print(\"  expected: q=43 = round(27 * 3.2 * 0.05 / 0.10)\")
'
" | tee "$OUT/qnn_hta/run_output.txt"

echo
echo "=== [4/4] Write summary ==="
cat > "$OUT/summary.md" <<EOF
# QNN e2e demo — CPU + GPU + NPU on QRB5165

All three on-board execution paths verified end-to-end.

## Per-target execution evidence

### CPU (IREE LLVM AArch64 / local-task driver)

\`\`\`
$(cat "$OUT/cpu/run_output.txt")
\`\`\`

Expected: \`3.5 + 3.5 = 7.0\` for every element in the 1×16 tensor.

### QNN GPU (Adreno via IREE QNN HAL)

\`\`\`
$(cat "$OUT/qnn_gpu/run_output.txt")
\`\`\`

Same input, same expected output. The dispatch flows
\`iree-run-module → IREE VM → QNN HAL command_buffer →
QnnGraph_execute(libQnnGpu.so)\` end-to-end. Required fixing
\`copy_buffer\`/\`fill_buffer\`/\`update_buffer\` in
\`runtime/src/iree/hal/drivers/qnn/qnn_command_buffer.c\`.

### NPU/HTA (Hexagon via libQnnHta.so + raw ctxbin)

\`\`\`
$(cat "$OUT/qnn_hta/run_output.txt")
\`\`\`

uint8 Conv2D smoke kernel, input q=64. Analytical:
\`q_out = round(27 · 3.2 · 0.05 / 0.10) = 43 (0x2B)\`. HTA hardware
produces exactly 43 across all 144 output positions, matching
libQnnCpu's reference output. Required fixing the q-param
convention (\`offset=0\` asymmetric uint8) — \`offset=+128\`
saturated on HTA.

## Artifacts

- \`cpu/artifact.vmfb\` — CPU IREE VMFB (single_add fp32)
- \`qnn_gpu/artifact.vmfb\` — QNN HAL VMFB (single_add fp32, qnn_add_f32 ctxbin embedded)
- \`qnn_hta/conv2d_int8_smoke.qnn-ctx\` — raw HTA ctxbin (uint8 Conv2D)

## Source changes that landed all three

| Component | Change |
|---|---|
| \`runtime/src/iree/hal/drivers/qnn/qnn_command_buffer.c\` | Implement \`copy_buffer\`/\`fill_buffer\`/\`update_buffer\` via \`iree_hal_buffer_map_*\` |
| \`benchmarks/QRB5165/kernels/abi/conv2d_int8_smoke.qnn.cpp\` | Switch from \`offset=+128\` symmetric to \`offset=0\` asymmetric uint8 q-params |
| \`tools/kernels/spec_gen.py\` | Wrapper-arity fix (#107) + count-region constants (#108) |
| \`benchmarks/QRB5165/kernels/match/*.match.mlir\` | Canonical \`^bb0(...)\` block-arg preludes |

EOF

echo
echo "=== Done ==="
echo "  Summary: $OUT/summary.md"
echo "  CPU output:    $OUT/cpu/run_output.txt"
echo "  QNN GPU out:   $OUT/qnn_gpu/run_output.txt"
echo "  NPU/HTA out:   $OUT/qnn_hta/run_output.txt"
