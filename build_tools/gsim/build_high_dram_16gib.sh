#!/usr/bin/env bash
# Build the exact bare-metal ELF used to check the selected 16 GiB DRAM span.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CHIPYARD_ROOT OUTPUT_ELF" >&2
  exit 2
fi

chipyard_root=$1
out_elf=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test_src="$chipyard_root/toolchains/riscv-tools/riscv-tests"
riscv_gcc="$chipyard_root/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
object_file="$(dirname -- "$out_elf")/high_dram_16gib.o"

"$riscv_gcc" -c -mcmodel=medany -march=rv64gc -mabi=lp64d \
  -I"$test_src/env/p" -I"$test_src/isa/macros/scalar" \
  "$script_dir/high_dram_16gib.S" -o "$object_file"
"$riscv_gcc" -static -mcmodel=medany -nostdlib -nostartfiles \
  -march=rv64gc -mabi=lp64d -T"$test_src/env/p/link.ld" \
  "$object_file" -o "$out_elf"
