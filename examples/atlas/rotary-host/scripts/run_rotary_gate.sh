#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RISCV:?set RISCV to the pinned RV64GC/lp64d toolchain prefix}"
: "${TORCH_PYTHON:=python3}"
mkdir -p out
nice -n 10 "$TORCH_PYTHON" -c 'from pathlib import Path; Path("out/inv_freq.bin").write_bytes(bytes.fromhex(Path("scripts/inv_freq.hex").read_text()))'
nice -n 10 "$TORCH_PYTHON" scripts/build_core_math.py > out/core-math-build.log
ARCHIVE="$PWD/out/core-math/libcore-math-rotary.a"
sha256sum "$ARCHIVE" > out/libm.sha256
test "$(cut -d' ' -f1 out/libm.sha256)" = 0626df7b1ce6426945820fd03dbd6752931c7594676c94f2043d5673588d178b
nice -n 10 "$RISCV/bin/riscv64-unknown-elf-gcc" -march=rv64gc -mabi=lp64d -O2 -fno-fast-math -ffp-contract=off -fno-builtin scripts/rotary_probe.c "$ARCHIVE" -lm -Wl,-Map,out/rotary_probe.map -o out/rotary_probe
PATH="$RISCV/../bin:$PATH" LD_LIBRARY_PATH="$RISCV/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" nice -n 10 "$RISCV/bin/spike" --isa=rv64gc "$RISCV/riscv64-unknown-elf/bin/pk" out/rotary_probe out/inv_freq.bin > out/rv64-rotary.bin 2> out/spike.log
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$TORCH_PYTHON" scripts/compare_rotary.py > out/compare.log
# Fail closed for callers that use the gate as an admission prerequisite.
nice -n 10 "$TORCH_PYTHON" -c 'import json; raise SystemExit(json.load(open("out/rotary-libm-receipt.json"))["gate"] != "PASS")'
