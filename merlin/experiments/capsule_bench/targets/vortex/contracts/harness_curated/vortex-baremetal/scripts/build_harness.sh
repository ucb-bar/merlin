#!/usr/bin/env bash
# Build the curated Vortex bare-metal harness the capsule-bench arms link against.
#
# Run by the EXPERIMENT OWNER, not by an agent: it reaches into the Vortex tree (a denied surface for
# the arms) and emits a self-contained staging dir that carries no pointer back to it.
#
# What the staged harness contains, and why each piece is board support rather than an answer:
#   libvortex_curated.a  the Vortex kernel runtime MINUS the work-distribution runtime. Vortex's
#                        libvortex2.a is seven objects (start / syscalls / print x2 / tinyprintf /
#                        serial / spawn); exactly one of them, vx_spawn, is the software
#                        thread-distribution loop, and it is deleted here. Nothing else in the
#                        archive computes anything about the workload.
#                        NOTE this costs the agent nothing: under the KMU (-DKMU_ENABLE) the HARDWARE
#                        launches every (block, thread) coordinate and identity comes from CTA CSRs,
#                        so vx_spawn is the legacy non-KMU path. Verified: a kernel with zero Vortex
#                        headers runs correctly on simx.
#   vx_entry.o           the KMU entry stub. The ONLY device-side piece needing the Vortex toolchain
#                        (the `vortex.kernel` annotation -> .vx_entry/VXSYMTAB footer). Prebuilt here
#                        so the agent's compiler can be stock LLVM. See ../src/vx_entry.cpp.
#   link*.ld             the linker script.
#   include/             the harness<->kernel ABI header only.
#
# Usage:  build_harness.sh [--vortex <VORTEX_HOME>] [--out <dir>] [--xlen 32|64]
set -euo pipefail

VORTEX_HOME="${VORTEX_HOME:-}"
OUT=""
XLEN=64
while [ $# -gt 0 ]; do
  case "$1" in
    --vortex) VORTEX_HOME="$2"; shift 2 ;;
    --out)    OUT="$2"; shift 2 ;;
    --xlen)   XLEN="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$VORTEX_HOME" ] || { echo "error: set VORTEX_HOME or pass --vortex" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$HERE/_staged}"

TOOLDIR="${TOOLDIR:-$VORTEX_HOME/tools}"
VX_CLANG="$TOOLDIR/llvm-vortex/bin/clang++"
VX_AR="$TOOLDIR/llvm-vortex/bin/llvm-ar"
KERNEL_LIB="${KERNEL_LIB:-libvortex2.a}"          # the KMU-enabled variant
SPAWN_OBJ="${SPAWN_OBJ:-vx_spawn.c.2.o}"          # the object to curate OUT
PREBUILT_LIB="$VORTEX_HOME/build/sw/kernel/$KERNEL_LIB"

[ -x "$VX_CLANG" ]      || { echo "error: no Vortex clang at $VX_CLANG" >&2; exit 1; }
[ -f "$PREBUILT_LIB" ]  || { echo "error: no $PREBUILT_LIB (build the Vortex kernel lib first)" >&2; exit 1; }

mkdir -p "$OUT/lib" "$OUT/include" "$OUT/link"

echo "== curating $KERNEL_LIB -> libvortex_curated.a (dropping $SPAWN_OBJ)"
cp "$PREBUILT_LIB" "$OUT/lib/libvortex_curated.a"
"$VX_AR" d "$OUT/lib/libvortex_curated.a" "$SPAWN_OBJ"
if "$VX_AR" t "$OUT/lib/libvortex_curated.a" | grep -qx "$SPAWN_OBJ"; then
  echo "error: $SPAWN_OBJ survived curation" >&2; exit 1
fi

echo "== building a MINIMAL startup object (no NEED_GP / NEED_TLS / NEED_INITFINI)"
# Why this exists, and why the archive's copy must NOT be used:
#
# libvortex*.a is built once for all apps, so its vx_start.S is compiled with EVERY startup feature
# enabled — it cannot know what a given kernel needs. That makes each launched hart run __init_tls and
# __libc_init_array at CTA entry. With 4 cores x 8 warps x 8 threads that is 256 harts doing libc
# init concurrently across non-coherent L1 dcaches, which corrupts state and leaves output buffers
# unwritten. It is invisible on simx and on a 1-core RTL build, and fails 64/64 on the 4-core one.
#
# Compiler-emitted kernels need none of those features (verified per link with kernel_startup.sh), so
# the harness stages a startup with KMU dispatch only, and the link puts it AHEAD of the archive.
XC=$(python3 "$VORTEX_HOME/build/ci/gen_config.py" --config="$VORTEX_HOME/VX_config.toml" \
      --cflags="$(grep -oE '\-DVX_CFG_[A-Z0-9_]+(=[^ ]*)?' "$HERE/../../../target_experiment.yaml" \
                  | tr '\n' ' ') -DVX_CFG_XLEN=$XLEN" 2>/dev/null || true)
"$VX_CLANG" \
  --target="riscv$XLEN-unknown-elf" \
  --sysroot="$TOOLDIR/riscv$XLEN-gnu-toolchain/riscv$XLEN-unknown-elf" \
  --gcc-toolchain="$TOOLDIR/riscv$XLEN-gnu-toolchain" \
  $( [ "$XLEN" = 64 ] && echo "-march=rv64imafd -mabi=lp64d" || echo "-march=rv32imaf -mabi=ilp32f" ) \
  -O3 -mcmodel=medany -nostdlib \
  -I"$VORTEX_HOME/sw/kernel/include" -I"$VORTEX_HOME/build/sw" -I"$VORTEX_HOME/build/hw" \
  -I"$VORTEX_HOME/sw/common" -DNDEBUG -D__VORTEX__ $XC -DKMU_ENABLE \
  -c "$VORTEX_HOME/sw/kernel/src/vx_start.S" -o "$OUT/lib/vx_start_min.o"
cp "$VORTEX_HOME/sw/kernel/scripts/kernel_startup.sh" "$OUT/"   # link-time feature check

echo "== building the KMU entry stub with the Vortex toolchain"
"$VX_CLANG" \
  --target="riscv$XLEN-unknown-elf" \
  --sysroot="$TOOLDIR/riscv$XLEN-gnu-toolchain/riscv$XLEN-unknown-elf" \
  --gcc-toolchain="$TOOLDIR/riscv$XLEN-gnu-toolchain" \
  $( [ "$XLEN" = 64 ] && echo "-march=rv64imafd -mabi=lp64d" || echo "-march=rv32imaf -mabi=ilp32f" ) \
  -O2 -mcmodel=medany -fno-rtti -fno-exceptions -nostdlib -fdata-sections -ffunction-sections \
  -I"$HERE/include" -c "$HERE/src/vx_entry.cpp" -o "$OUT/lib/vx_entry.o"

cp "$HERE/include/merlin_vortex_abi.h" "$OUT/include/"
cp "$VORTEX_HOME/sw/kernel/scripts/link$XLEN.ld" "$OUT/link/"
cp "$VORTEX_HOME/sw/kernel/scripts/vxbin.py"     "$OUT/"

echo "== building the generic capsule host driver"
mkdir -p "$OUT/host"
"${HOST_CXX:-g++}" -std=c++17 -O2 -Wall \
  -I"$VORTEX_HOME/sw/runtime/include" -I"$VORTEX_HOME/build/sw" -I"$VORTEX_HOME/build/hw" \
  -I"$VORTEX_HOME/sw/common" -I"$HERE/include" \
  "$HERE/host/merlin_vx_host.cpp" \
  -L"$VORTEX_HOME/build/sw/runtime" -lvortex -o "$OUT/host/merlin_vx_host"

cat > "$OUT/MANIFEST.txt" <<EOF
merlin vortex-baremetal curated harness (xlen=$XLEN)
built from: $KERNEL_LIB with $SPAWN_OBJ removed
excluded deliberately: vx_spawn (work distribution), all compute kernels, all Vortex headers
                       except the harness ABI header
device symbol the compiler must provide: merlin_kernel_body
EOF

echo "== staged -> $OUT"
"$VX_AR" t "$OUT/lib/libvortex_curated.a" | sed 's/^/   obj: /'
