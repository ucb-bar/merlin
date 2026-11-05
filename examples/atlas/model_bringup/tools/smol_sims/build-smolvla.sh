#!/usr/bin/env bash
# Usage: build-smolvla.sh PACKAGE PLATFORM_ROOT WORK_DIR [RISCV_TOOLCHAIN_BIN]
# Cross-compile only. WORK_DIR must be a fresh directory, outside both inputs.
# Optional: SMOLVLA_PHASE=42 for a captured phase, SMOLVLA_PLATFORMS=SIMS
# for one console, SMOLVLA_BUILD_JOBS for build concurrency, and
# SMOLVLA_FAST_ASSETS=DIR for fast mode from generate_fast.py output bound to this package's default ELF.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  printf '%s\n' 'Usage: build-smolvla.sh PACKAGE PLATFORM_ROOT WORK_DIR [RISCV_TOOLCHAIN_BIN]'
  exit 0
fi
P=${1:?provide package directory}
source_tree=${2:?provide board-support tree for the target platform}
W=${3:?provide fresh work directory}
toolbin=${4:-${RISCV_TOOLCHAIN_BIN:-}}
phase=${SMOLVLA_PHASE:-}
platforms=${SMOLVLA_PLATFORMS:-CHIP SIMS}
fast=${SMOLVLA_FAST_ASSETS:-}
if [[ -n $phase && ! $phase =~ ^(0|[1-9][0-9]?|1[0-9][0-9])$ ]]; then
  printf '%s\n' 'SMOLVLA_PHASE must be empty or an integer from 0 through 199' >&2
  exit 2
fi
if [[ -n $fast && $fast != /* ]]; then
  printf '%s\n' 'SMOLVLA_FAST_ASSETS must be an absolute directory' >&2
  exit 2
fi
for platform in $platforms; do
  if [[ $platform != CHIP && $platform != SIMS ]]; then
    printf '%s\n' 'SMOLVLA_PLATFORMS must contain CHIP, SIMS, or both' >&2
    exit 2
  fi
done
if [[ -z $platforms ]]; then
  printf '%s\n' 'SMOLVLA_PLATFORMS must not be empty' >&2
  exit 2
fi
if [[ -n $toolbin ]]; then export PATH="$toolbin:$PATH"; fi
export PYTHONDONTWRITEBYTECODE=1
# Fail before copying the tree or extracting assets when tools are unavailable.
for tool in python3 cmake riscv64-unknown-elf-gcc riscv64-unknown-elf-g++ \
            riscv64-unknown-elf-readelf riscv64-unknown-elf-nm riscv64-unknown-elf-objdump; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf 'Required build tool is unavailable: %s\n' "$tool" >&2
    exit 2
  fi
done
# Validate source paths before copying.
python3 -I - "$P" "$source_tree" "$W" <<'PY'
import json, os, pathlib, shutil, sys
def safe(path):
    path=pathlib.Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise SystemExit('Use absolute paths without parent components')
    cur=pathlib.Path(path.anchor)
    for part in path.parts[1:]:
        cur/=part
        if cur.is_symlink(): raise SystemExit('Symlink path components are unsupported')
    return path
pkg, src, work=map(safe,sys.argv[1:])
if work.exists(): raise SystemExit('WORK_DIR must not already exist')
if any(root==work or root in work.parents or work in root.parents for root in (pkg,src)):
    raise SystemExit('WORK_DIR must be outside package and source inputs')
def copy_tree(src,dst):
    dst.mkdir()
    with os.scandir(src) as entries:
        for e in entries:
            if e.name=='.git' or e.name.startswith('build'):
                continue
            if e.is_symlink(): raise SystemExit('Source symlinks are unsupported')
            target=dst/e.name
            if e.is_dir(follow_symlinks=False): copy_tree(pathlib.Path(e.path),target)
            elif e.is_file(follow_symlinks=False): shutil.copy2(e.path,target)
            else: raise SystemExit('Unsupported special file')
# Audit package paths that the build consumes, without opening excluded entries.
def audit(root):
    with os.scandir(root) as entries:
        for e in entries:
            if e.is_symlink(): raise SystemExit('Package symlinks are unsupported')
            if e.is_dir(follow_symlinks=False): audit(e.path)
for name in ('source','platform_root','tools'): audit(safe(pkg/name))
safe(pkg/'smolvla-full-f2.elf')
work.mkdir(parents=True)
copy_tree(src,work/'platform_root')
unused=work/'platform_root/penguin26-tests/atlas_delivery/CMakeLists.txt'
if unused.exists():unused.write_text('# Unused smoke target omitted for this SmolVLA-only build.\n')
for name,dest in [('CMakeLists.txt','penguin26-tests/atlas_delivery_smolvla/CMakeLists.txt'),
                  ('smolvla_main.c','penguin26-tests/atlas_delivery_smolvla/smolvla_main.c'),
                  ('penguin26-smolvla.ld','platform/penguin26/penguin26-smolvla.ld')]:
    target=work/'platform_root'/dest
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(pkg/'platform_root'/name,target)
f=work/'platform_root/penguin26-tests/CMakeLists.txt'
text=f.read_text()
line='add_subdirectory(atlas_delivery_smolvla)'
if line not in text.splitlines():
    f.write_text(text.rstrip()+'\n'+'set(PENGUIN26_SMOLVLA_PACKAGE "" CACHE PATH "SmolVLA package containing source/")\nadd_subdirectory(atlas_delivery_smolvla)\n')
PY
mkdir -p "$W/out" "$W/tmp"
export TMPDIR="$W/tmp"
run_logged() {
  local log=$1
  shift
  { printf '%q ' "$@"; printf '\n'; } >> "$W/out/commands.log"
  "$@" 2>&1 | tee "$log"
}
run_logged "$W/out/extract-blobs.log" python3 -I "$P/tools/extract_blobs.py" \
  "$P/smolvla-full-f2.elf" "$P/source/assets.S" "$W/blobs"
for platform in $platforms; do
  lower=${platform,,}
  build="$W/platform_root/build-smolvla-$lower"
  extra=()
  if [[ $platform == CHIP ]]; then
    extra+=(-DPENGUIN26_UART_BAUDRATE=3686400 -DPENGUIN26_EXIT_TOHOST=ON)
  fi
  if [[ -n $fast ]]; then
    extra+=(-DSMOLVLA_FAST=ON -DSMOLVLA_FAST_ASSETS="$fast")
  fi
  run_logged "$W/out/configure-$lower.log" cmake -S "$W/platform_root" -B "$build" \
    -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$W/platform_root/riscv-gcc.cmake" \
    -DCHIP=penguin26 -DPLATFORM="$platform" -DRVV_TYPE=1 -DLINKER=smolvla \
    -DPENGUIN26_SMOLVLA_PACKAGE="$P" -DPENGUIN26_SMOLVLA_BLOBS="$W/blobs" \
    -DPENGUIN26_SMOLVLA_PHASE="$phase" \
    -DPENGUIN26_TESTS_ATLAS_DELIVERY_DIR= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${extra[@]}"
  run_logged "$W/out/build-$lower.log" cmake --build "$build" \
    --target atlas-delivery-smolvla --parallel "${SMOLVLA_BUILD_JOBS:-4}" --verbose
  elf="$build/penguin26-tests/atlas_delivery_smolvla/atlas-delivery-smolvla.elf"
  riscv64-unknown-elf-readelf -W -lS "$elf" > "$W/out/readelf-$lower.txt"
  riscv64-unknown-elf-nm -S --defined-only "$elf" > "$W/out/nm-$lower.txt"
  for symbol in atlas_result_uart_putchar atlas_workload_main app_init main uart_init; do
    riscv64-unknown-elf-objdump -d --disassemble="$symbol" "$elf" > "$W/out/disasm-$lower-$symbol.txt"
  done
  printf '%s build succeeded; ELF: %s\n' "$platform" "$elf" >> "$W/out/status.txt"
done
