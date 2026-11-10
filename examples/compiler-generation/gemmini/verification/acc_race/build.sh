#!/bin/bash
# build.sh <outdir> [extra CFLAGS...]
#
# Compile $REPRO_SRC (default acc_race_repro.c) into <outdir>/<name>.elf against a bare-metal Gemmini harness (the
# same riscv-tests crt/syscalls/linker script and flags the whole-model group programs use), and write
# <outdir>/build.json naming every input by sha256 so the ELF can be tied back to what built it.
#
# Env: GEMMINI_HARNESS  a gemmini-rocc-tests-shaped tree (include/, riscv-tests/benchmarks/common/)
#      RISCV_CC         the riscv64-unknown-elf-gcc to use
set -euo pipefail
OUT=${1:?usage: build.sh <outdir> [extra CFLAGS...]}; shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
V=${GEMMINI_HARNESS:?set GEMMINI_HARNESS to the bare-metal gemmini harness tree}
CC=${RISCV_CC:?set RISCV_CC to riscv64-unknown-elf-gcc}
C=$V/riscv-tests/benchmarks/common
SRC=${REPRO_SRC:-$HERE/acc_race_repro.c}; NAME=$(basename "${SRC%.*}")
FLAGS="-DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common \
 -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc -lm -lgcc \
 -DID_STRING= -Wno-incompatible-pointer-types -nostdlib -nostartfiles -static -DBAREMETAL=1"
INC="-I$V/riscv-tests -I$V/riscv-tests/env -I$V -I$C -I$OUT"
mkdir -p "$OUT"
for s in "$SRC" "$C/syscalls.c" "$C/crt.S"; do
  b=$(basename "$s"); $CC $FLAGS "$@" $INC -c "$s" -o "$OUT/${b%.*}.o"
done
$CC $FLAGS "$@" -T "$C/test.ld" "$OUT/$NAME.o" "$OUT/syscalls.o" "$OUT/crt.o" -o "$OUT/$NAME.elf" 2> "$OUT/link.log"
rm -f "$OUT"/*.o
python3 - "$OUT" "$SRC" "$NAME" "$V/include/gemmini.h" "$V/include/gemmini_params.h" "$CC" "$*" <<'PY'
import hashlib, json, sys
out, src, name, hdr, params, cc, extra = sys.argv[1:]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rec = {"elf": {"path": f"{out}/{name}.elf", "sha256": h(f"{out}/{name}.elf")},
       "inputs": {p: h(p) for p in (src, hdr, params, cc)}, "extra_cflags": extra}
json.dump(rec, open(f"{out}/build.json", "w"), indent=1)
print(rec["elf"]["sha256"])
PY
