#!/usr/bin/env bash
# Workspace convenience wrapper. All costly children inherit the service budget.
set -euo pipefail
byoc_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
merlin_root=$(cd -- "$byoc_dir/../../../../.." && pwd)
build_root="$merlin_root/out/build/baselines/tvm-gemmini"
chipyard_root="${CHIPYARD_ROOT:-$(dirname "$merlin_root")/chipyard}"
tvm_source="${TVM_ROOT:-$build_root/tvm-source}"
tvm_build="${TVM_BUILD:-$build_root/tvm-host-build}"
byoc_python="${BYOC_PYTHON:-$build_root/smolvla-venv/bin/python}"
driver="$byoc_dir/verify_byoc.py"
operation=${1:?Use prepare CASE OUTPUT, spike BUNDLE REPORT, rtl BUNDLE REPORT, or host OUTPUT}
shift
common=(--user --quiet --wait --pipe --working-directory="$merlin_root"
  -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64
  -p RuntimeMaxSec=60 -p TimeoutStopSec=3 -p KillMode=control-group)
environment=(env "TVM_LIBRARY_PATH=$tvm_build"
  "PYTHONPATH=$byoc_dir:$build_root/smol-tvm-host-dependencies-py312:$tvm_source/python"
  "TVM_FFI=ctypes" "OMP_NUM_THREADS=1" "OPENBLAS_NUM_THREADS=1"
  "LD_LIBRARY_PATH=$build_root/llvm18-sdk/lib:$chipyard_root/.conda-env/lib"
  "PATH=$chipyard_root/.conda-env/bin:/usr/bin:/bin")
case "$operation" in
  prepare)
    case_name=${1:?case}; output=${2:?fresh output directory}; shift 2
    command=("$byoc_python" "$driver" prepare --tvm-source "$tvm_source"
      --riscv-gcc "$chipyard_root/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
      --headers "$build_root/pinned-rtl-adapter-20261019T095639Z/headers"
      --adapter "$build_root/pinned-rtl-adapter-20261019T095639Z/matmul.o"
      --case "$case_name" --output-dir "$output" "$@") ;;
  spike|rtl)
    bundle=${1:?bundle}; output=${2:?JSON report}; shift 2
    command=(/usr/bin/python3 "$driver" run --bundle "$bundle" --output "$output" "$@")
    if [[ "$operation" == spike ]]; then
      command+=(--spike "$chipyard_root/.conda-env/riscv-tools/bin/spike"
        --plugin "$build_root/wsl-simulator-20261018/libgemmini.so")
    else
      command+=(--simulator "$build_root/pinned-rtl/chipyard/sims/verilator/simulator-chipyard.harness-GemminiRocketConfig")
    fi ;;
  host) command=("$byoc_python" "$byoc_dir/test_byoc.py" --output-dir "${1:?fresh output directory}") ;;
  *) echo "Unknown operation: $operation" >&2; exit 2 ;;
esac
exec systemd-run "${common[@]}" "${environment[@]}" "${command[@]}"
