"""Run bridged Voyager conv schedules (and merlin's own lowering of the same layer) on Gemmini, bare metal.

A :class:`merlin.baselines.voyager_schedule.Schedule` is abstract block ops. This script packs them with
the certified reference package's own encoders -- exactly as ``package_template/voyager_replay.py``
packs a GEMM schedule (a CONFIG_LD before every MVIN, the reference's leading FENCE/FLUSH, no trailing
fence of its own) -- and emits the kernel as LLVM-dialect MLIR in the package's ``llvm_artifact`` form
(pointer-derived operands, one ``.insn`` per command, one closing fence).

Every program is packaged in the SAME format as the phase-2 Gemmini measurement bundles, so it can be
queued on FireSim unchanged next to merlin's: a ``payload/single_run_harness.c`` with the ``MERLIN_*``
marker protocol (one untimed warm-up invocation, a warm reset outside the counters, one measured
invocation bracketed by ``rdcycle``/``rdinstret``, the same eight Gemmini counters, a short console),
``payload/const_blob.S`` embedding the operands (and the golden output when small) by ``.incbin``, the
phase-2 bundle's ``runtime/`` headers and ``crt.S``, and a ``build_elf.sh`` that is the phase-2 one:
chipyard's toolchain, ``test.ld`` and ``syscalls.c``. The same ELF then runs on Spike and on the
cycle-accurate Verilator binary the capsule runner uses; ``summarize_job.sh`` reads its console as is.

Translations beyond the GEMM template, each derived from the RTL or its generated header:

* a halo (``Mvin`` role ``zero``) is an MVIN from DRAM address 0: Gemmini's LoadController marks a read
  at vaddr 0 ``all_zeros`` (``LoadController.scala`` ``val all_zeros = vaddr === 0.U``) and the
  scratchpad routes it to its ``ZeroWriter`` (``Scratchpad.scala``), so no DRAM is read;
* an ``AccMvin`` (bias) is an MVIN to an accumulator address with a stride-0 broadcast, which the
  LoadController serves as one DRAM row repeated (``actual_rows_read = 1`` when ``stride === 0``);
* a scaled accumulator MVIN is refused unless the target's generated header scales accumulator loads:
  ``gemmini_params.h`` defines ``MVIN_SCALE_ACC(x, scale)`` and on an identity definition a scale
  would be silently dropped.

Exactness: the harness checks every output element against the embedded golden (``execute()`` for the
Voyager arm, a direct convolution for merlin's arm) and prints the count of mismatches; an output too
large to embed is checked by its 64-bit FNV-1a digest instead. Either way the console stays small.

The merlin arm is the certified reference package's own ``build_trace`` of a ``merlin_iface.conv2d``
program for the same layer, emitted through the same emitter and harness, so the arms differ only in
the command stream.

Usage (merlin venv):
    python merlin/experiments/voyager_h2h/scripts/emit_gemmini_c.py --model-dir <voyager export dir> \
        [--model-dir ...] --simulators spike verilator
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from merlin.baselines import voyager_schedule as vs
from merlin.baselines.voyager import geometry_for
from merlin.baselines.voyager_ir import UnsupportedConstruct, load_model, replay
from merlin.common import provenance
from merlin.common.artifacts import new_product
from merlin.common.paths import artifacts_dir, merlin_dir, runs_dir
from merlin.common.paths import env as dotenv

TARGET = "gemmini"
DEFAULT_PACKAGE = artifacts_dir() / "targets" / TARGET / "gemmini_xdsl_rtl_v0"
#: The phase-2 bundle whose runtime (headers, crt.S) and build recipe every ELF here reuses.
DEFAULT_PHASE2 = artifacts_dir() / "perf-bench" / TARGET / \
    "resnet50_merlin_phase2_bf62_w8a8_warm_measured_firesim_ready_20260907"
_FNV_OFFSET, _FNV_PRIME = 0xCBF29CE484222325, 0x100000001B3
#: Schedule dtype names -> the package's tensor dtype names.
_DTYPES = {"int32": "i32", "i32": "i32", "int8": "i8", "i8": "i8"}
_NP = {"i8": np.dtype("<i1"), "i32": np.dtype("<i4")}
_C_ELEM = {"i8": "int8_t", "i32": "int32_t"}


# ---------------------------------------------------------------------------------------------------
# the package's encoders
# ---------------------------------------------------------------------------------------------------
def load_package(package: Path):
    """(isa, ir_ingest) modules of an OOT package, imported the way its own tool imports them."""
    root = str(Path(package) / "mlir_oot")
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("lowering.isa"), importlib.import_module("ir_ingest")


def acc_mvin_scales(params_h: Path) -> bool:
    """Whether the generated header's accumulator-load scale does anything (parsed, not assumed)."""
    for line in params_h.read_text().splitlines():
        head, _, body = line.partition(")")
        if head.replace(" ", "").startswith("#defineMVIN_SCALE_ACC(x,scale"):
            return body.strip() not in ("(x)", "x")
    return False


def schedule_instructions(schedule: vs.Schedule, isa, *, names: dict, strides: dict, elems: dict,
                          out_dtype: str, acc_scale_ok: bool) -> list:
    """Pack ``schedule`` with the package encoders (the replay template's packing, extended to conv)."""
    I, A, word = isa.Instruction, isa.Address, isa._tile_word
    readout = isa.ACC_BASE | (isa.ACC_FULL if out_dtype == "i32" else 0)

    def store_config(op) -> tuple:
        scale = getattr(op, "scale", None)
        relu = bool(getattr(op, "relu", False))
        dtype = _DTYPES.get(getattr(op, "out_dtype", None) or out_dtype)
        if dtype != out_dtype:
            raise UnsupportedConstruct(f"a store of {dtype} into the {out_dtype} output buffer")
        return (1.0 if scale is None else float(scale), relu)

    def config_st(key: tuple):
        scale, relu = key
        return isa._config_st(strides["out"], {"epilogue": ["relu"] if relu else [],
                                               "acc_scale": scale}, out_dtype)

    stores = [op for op in schedule.ops if isinstance(op, vs.Mvout)]
    current = store_config(stores[0]) if stores else (1.0, False)
    trace = [I("FENCE"), I("FLUSH", 0, 0), isa._config_ex(weight_stationary=True), config_st(current)]
    for op in schedule.ops:
        kind = type(op).__name__
        if kind == "Mvin":
            if op.role == "zero":
                # DRAM address 0 is the RTL's zero-writer path; the stride is unused there.
                trace += [isa._config_ld(0, channel=0),
                          I("MVIN", 0, word(op.spad_row, op.cols, op.rows))]
                continue
            trace += [isa._config_ld(strides[op.role] * op.row_step, channel=0),
                      I("MVIN", A(names[op.role], op.dram_row * strides[op.role]
                                  + op.dram_col * elems[op.role]),
                        word(op.spad_row, op.cols, op.rows))]
        elif kind == "AccMvin":
            scale = getattr(op, "scale", None)
            if scale not in (None, 1, 1.0) and not acc_scale_ok:
                raise UnsupportedConstruct(f"a {op.role} accumulator load scaled by {scale}: this "
                                           "target's MVIN_SCALE_ACC is the identity")
            if scale not in (None, 1, 1.0):
                raise UnsupportedConstruct("scaled accumulator loads are not packed yet")
            addr = isa.ACC_BASE | op.acc_row | (isa.ACC_ACCUMULATE if getattr(op, "accumulate", False)
                                                else 0)
            trace += [isa._config_ld(strides[op.role] * op.row_step, shrunk=elems[op.role] == 1,
                                     channel=0),
                      I("MVIN", A(names[op.role], op.dram_row * strides[op.role]
                                  + op.dram_col * elems[op.role]),
                        word(addr, op.cols, op.rows))]
        elif kind == "Preload":
            b = word(isa.GARBAGE_ADDR if op.weight_row is None else op.weight_row, isa.DIM, isa.DIM)
            c = readout | op.acc_row | (isa.ACC_ACCUMULATE if op.accumulate else 0)
            trace.append(I("PRELOAD", b, word(c, op.cols, op.rows)))
        elif kind == "Compute":
            trace.append(I("COMPUTE_PRELOADED" if op.fresh_weights else "COMPUTE_ACCUMULATE",
                           word(op.input_row, isa.DIM, op.rows),
                           word(isa.GARBAGE_ADDR, isa.DIM, isa.DIM)))
        elif kind == "Mvout":
            key = store_config(op)
            if key != current:
                trace.append(config_st(key))
                current = key
            trace.append(I("MVOUT", A(names["out"], op.dram_row * strides["out"]
                                      + op.dram_col * elems["out"]),
                           word(readout | isa.ACC_ACCUMULATE | op.acc_row, op.cols, op.rows)))
        else:
            raise UnsupportedConstruct(f"{kind} has no device packing (a host op needs a host lane)")
    return trace


def emit_llvm(tensor_order: list[str], trace: list, isa) -> str:
    """LLVM-dialect MLIR in the package's ``llvm_artifact`` form (text-identical for its own trace)."""
    args = ", ".join(f"%arg{i}: !llvm.ptr" for i in range(len(tensor_order)))
    lines = ["module {", f"  llvm.func @gemmini_kernel({args}) {{"]
    bases: dict[str, str] = {}
    counter = [0]

    def new(prefix: str = "c") -> str:        # the package's numbering, so the texts match
        name = f"%{prefix}{counter[0]}"
        counter[0] += 1
        return name

    for index, name in enumerate(tensor_order):
        value = new("p")
        lines.append(f"    {value} = llvm.ptrtoint %arg{index} : !llvm.ptr to i64")
        bases[name] = value

    def operand(value) -> str:
        if isinstance(value, isa.Address):
            if value.offset == 0:
                return bases[value.tensor]
            const, result = new(), new("a")
            lines.append(f"    {const} = llvm.mlir.constant({value.offset} : i64) : i64")
            lines.append(f"    {result} = llvm.add {bases[value.tensor]}, {const} : i64")
            return result
        const = new()
        lines.append(f"    {const} = llvm.mlir.constant({int(value or 0)} : i64) : i64")
        return const

    for ins in trace:
        if ins.name == "FENCE":
            lines.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')
            continue
        rs1, rs2 = operand(ins.rs1), operand(ins.rs2)
        lines.append(f'    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 0x{ins.funct:x}, x0, '
                     f'$0, $1", "r,r" {rs1}, {rs2} : (i64, i64) -> ()')
    lines += ['    llvm.inline_asm has_side_effects "fence", "" : () -> ()', "    llvm.return", "  }",
              "}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------
# the phase-2 bundle: harness, blob, runtime, build
# ---------------------------------------------------------------------------------------------------
def fnv1a64(data: bytes) -> int:
    h = _FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def _align(n: int, a: int = 64) -> int:
    return -(-n // a) * a


def render_bundle_payload(tensors: list, expected: np.ndarray, *, out_dtype: str, model: str,
                          compiler: str, golden_max_bytes: int) -> dict:
    """The harness text, const-blob assembly and blobs for one program, in phase-2 form.

    ``tensors``: (name, dtype, values or None for the output, shape), in kernel-argument order.
    """
    blob, offsets = bytearray(), {}
    for name, dtype, values, _ in tensors:
        if values is None:
            continue
        blob += b"\0" * (_align(len(blob)) - len(blob))
        offsets[name] = len(blob)
        blob += np.ascontiguousarray(values, dtype=_NP[dtype]).tobytes()
    out_bytes = np.ascontiguousarray(expected, dtype=_NP[out_dtype]).tobytes()
    digest = fnv1a64(out_bytes)
    golden = len(out_bytes) <= golden_max_bytes
    input_sha = hashlib.sha256(bytes(blob)).hexdigest()[:16]
    elements = int(expected.size)
    args = ",\n".join(
        "      (void *)merlin_output" if values is None
        else f"      (void *)(merlin_const_blob_start + {offsets[name]})"
        for name, _, values, _ in tensors)
    params = ", ".join(f"void *p{i}" for i in range(len(tensors)))
    ctype = _C_ELEM[out_dtype]
    check = ["  int bad = 0;"]
    if golden:
        check += [
            f"  const {ctype} *got = (const {ctype} *)merlin_output;",
            f"  const {ctype} *expected = (const {ctype} *)merlin_golden_start;",
            f"  for (int i = 0; i < {elements}; ++i) {{",
            "    if (got[i] != expected[i]) {",
            "      if (bad < 8)",
            '        printf("MERLIN_MISMATCH index=%d got=%d expected=%d\\n", i, (int)got[i], '
            "(int)expected[i]);",
            "      ++bad;", "    }", "  }"]
    else:
        check += [f"  if (checksum != {digest}ULL) bad = 1;"]
    harness = "\n".join([
        "#include <stdint.h>", "#include <stdio.h>", "#include <stdlib.h>", "",
        '#include "include/gemmini_testutils.h"', "",
        "extern const unsigned char merlin_const_blob_start[];",
        *(["extern const unsigned char merlin_golden_start[];"] if golden else []),
        f"extern void gemmini_kernel({params});", "",
        f"#define MERLIN_OUTPUT_BYTES ((size_t){len(out_bytes)})",
        "static unsigned char merlin_output[MERLIN_OUTPUT_BYTES] __attribute__((aligned(64)));", "",
        "static uint64_t read_instret_local(void) {", "  uint64_t value;",
        '  __asm__ volatile("rdinstret %0" : "=r"(value));', "  return value;", "}", "",
        "static uint64_t fnv1a64(const unsigned char *p, size_t n) {",
        f"  uint64_t h = {_FNV_OFFSET}ULL;",
        f"  for (size_t i = 0; i < n; ++i) {{ h ^= p[i]; h *= {_FNV_PRIME}ULL; }}",
        "  return h;", "}", "",
        "/* The output is invocation-local: zero it after the untimed warm-up, outside the counters,",
        "   so the measured invocation must write every element again. */",
        "static void restore_invocation_state_outside_timed_region(void) {",
        "  for (size_t i = 0; i < MERLIN_OUTPUT_BYTES; ++i)", "    merlin_output[i] = 0;", "}", "",
        "static void run_model(void) {", "  gemmini_kernel(", args + ");", "}", "",
        "int main(void) {",
        f'  printf("MERLIN_MODEL {model}\\n");',
        f'  printf("MERLIN_COMPILER {compiler}\\n");',
        f'  printf("MERLIN_INPUT seed0_int8_sha256_{input_sha}\\n");',
        '  printf("MERLIN_INVOCATIONS warmup=1 measured=1 batch=1\\n");',
        '  printf("MERLIN_INPUT_PROLOGUE timed=0 owner=harness kind=preloaded_int8\\n");', "",
        '  printf("MERLIN_PROFILE warmup begin\\n");', "  run_model();", "  gemmini_fence();",
        '  printf("MERLIN_PROFILE warmup end rc=0\\n");', "",
        "  restore_invocation_state_outside_timed_region();",
        '  printf("MERLIN_WARM_RESET timed=0 kind=zero_output bytes=%llu\\n",',
        "         (unsigned long long)MERLIN_OUTPUT_BYTES);", "",
        "  counter_configure(0, MAIN_LD_CYCLES);", "  counter_configure(1, MAIN_ST_CYCLES);",
        "  counter_configure(2, MAIN_EX_CYCLES);", "  counter_configure(3, EXE_ACTIVE_CYCLE);",
        "  counter_configure(4, LOOP_MATMUL_ACTIVE_CYCLES);", "  counter_configure(5, RDMA_BYTES_REC);",
        "  counter_configure(6, WDMA_BYTES_SENT);",
        "  counter_configure(7, RESERVATION_STATION_ACTIVE_CYCLES);", "  counter_reset();",
        "  gemmini_fence();", "",
        '  printf("MERLIN_PROFILE measured begin\\n");',
        "  const uint64_t instret_start = read_instret_local();",
        "  const uint64_t cycle_start = read_cycles();", "  run_model();", "  gemmini_fence();",
        "  const uint64_t cycle_end = read_cycles();",
        "  const uint64_t instret_end = read_instret_local();", "  uint32_t counters[8];",
        "  for (int i = 0; i < 8; ++i)", "    counters[i] = counter_read(i);", "",
        "  const uint64_t checksum = fnv1a64(merlin_output, MERLIN_OUTPUT_BYTES);", *check, "",
        '  printf("MERLIN_METRIC cycles=%llu\\n", (unsigned long long)(cycle_end - cycle_start));',
        '  printf("MERLIN_METRIC instret=%llu\\n", (unsigned long long)(instret_end - instret_start));',
        '  printf("MERLIN_METRIC main_ld_cycles=%u\\n", counters[0]);',
        '  printf("MERLIN_METRIC main_st_cycles=%u\\n", counters[1]);',
        '  printf("MERLIN_METRIC main_ex_cycles=%u\\n", counters[2]);',
        '  printf("MERLIN_METRIC exe_active_cycle=%u\\n", counters[3]);',
        '  printf("MERLIN_METRIC loop_matmul_active_cycles=%u\\n", counters[4]);',
        '  printf("MERLIN_METRIC rdma_bytes_rec=%u\\n", counters[5]);',
        '  printf("MERLIN_METRIC wdma_bytes_sent=%u\\n", counters[6]);',
        '  printf("MERLIN_METRIC reservation_station_active_cycles=%u\\n", counters[7]);',
        f'  printf("MERLIN_RESULT checksum_fnv1a64=%016llx expected_fnv1a64={digest:016x}\\n",',
        "         (unsigned long long)checksum);",
        f'  printf("MERLIN_RESULT logits_checked={elements} bad=%d check='
        f'{"elementwise_" + out_dtype if golden else "fnv1a64"}\\n", bad);',
        "  if (bad != 0) {", '    printf("MERLIN_PROFILE measured end rc=1\\n");',
        '    printf("FAIL: output differs from the golden\\n");', "    exit(1);", "  }",
        '  printf("MERLIN_PROFILE measured end rc=0\\n");',
        f'  printf("PASS: warm-then-measured {model} and all-output check\\n");', "  exit(0);", "}",
        ""])
    blob_s = ["    .section .rodata", "    .balign 64", "    .global merlin_const_blob_start",
              "merlin_const_blob_start:", '    .incbin "payload/const_blob.bin"',
              "    .global merlin_const_blob_end", "merlin_const_blob_end:"]
    if golden:
        blob_s += ["    .balign 64", "    .global merlin_golden_start", "merlin_golden_start:",
                   '    .incbin "payload/golden.bin"', "    .global merlin_golden_end",
                   "merlin_golden_end:"]
    return {"harness": harness, "const_blob_s": "\n".join(blob_s) + "\n", "blob": bytes(blob),
            "golden": out_bytes if golden else None, "digest": digest, "elements": elements,
            "check": "elementwise" if golden else "fnv1a64"}


_BUILD_ELF = """#!/bin/bash
# The phase-2 bundle build (resnet50_merlin_phase2_*/build_elf.sh): chipyard's toolchain, test.ld and
# syscalls.c; this bundle's runtime/ headers and crt.S.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/chipyard" >&2
  exit 2
fi

bundle_dir=$(cd "$(dirname "$0")" && pwd -P)
chipyard=$(cd "$1" && pwd -P)
tests="$chipyard/generators/gemmini/software/gemmini-rocc-tests"
common="$tests/riscv-tests/benchmarks/common"
cc="$chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"

for required in "$cc" "$common/test.ld" "$common/syscalls.c" \\
  "$bundle_dir/runtime/include/gemmini.h" \\
  "$bundle_dir/runtime/include/gemmini_params.h" \\
  "$bundle_dir/runtime/include/gemmini_testutils.h" \\
  "$bundle_dir/runtime/include/gemmini_counter.h" \\
  "$bundle_dir/runtime/crt.S" \\
  "$bundle_dir/compiler/kernel.o" "$bundle_dir/payload/const_blob.bin"{golden_req}; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 2
  fi
done

mkdir -p "$bundle_dir/.build"
cd "$bundle_dir"
cflags=(
  -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2
  -ffast-math -fno-common -fno-builtin-printf
  -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc
  -I"$bundle_dir/runtime"
  -DBAREMETAL=1 -I"$tests/riscv-tests" -I"$tests/riscv-tests/env"
  -I"$tests" -I"$common"
)
"$cc" "${{cflags[@]}}" -c payload/single_run_harness.c -o .build/single_run_harness.o
"$cc" -march=rv64gc -mabi=lp64d -mcmodel=medany \\
  -c payload/const_blob.S -o .build/const_blob.o
"$cc" "${{cflags[@]}}" -c "$common/syscalls.c" -o .build/syscalls.o
"$cc" "${{cflags[@]}}" -c "$bundle_dir/runtime/crt.S" -o .build/crt.o
"$cc" "${{cflags[@]}}" -nostdlib -nostartfiles -static -T "$common/test.ld" \\
  .build/single_run_harness.o compiler/kernel.o .build/const_blob.o \\
  .build/syscalls.o .build/crt.o -lm -lgcc \\
  -o {elf}
sha256sum {elf}
"""


def chipyard_root() -> Path:
    """The chipyard the phase-2 build names, from the repo's environment (else the recipe's compiler)."""
    configured = dotenv("MERLIN_CHIPYARD")
    if configured:
        return Path(configured)
    from merlin.runtime.backends import base as backends
    return Path(backends.harness_build_recipe(TARGET).compiler).parents[3]


def build_bundle(bundle: Path, mlir_text: str, payload: dict, *, phase2: Path, elf_name: str) -> Path:
    """Lay out a phase-2-format bundle in ``bundle`` and build its ELF with the bundle's own script."""
    from merlin.targetgen.contract.compile import llvm_mlir_to_object
    for sub in ("payload", "compiler", "runtime/include"):
        (bundle / sub).mkdir(parents=True, exist_ok=True)
    (bundle / "compiler" / "kernel.llvm.mlir").write_text(mlir_text)
    obj = llvm_mlir_to_object(mlir_text, bundle / "compiler", target=TARGET)
    if obj != bundle / "compiler" / "kernel.o":
        shutil.copy2(obj, bundle / "compiler" / "kernel.o")
    for header in sorted((phase2 / "runtime" / "include").glob("*.h")):
        shutil.copy2(header, bundle / "runtime" / "include" / header.name)
    shutil.copy2(phase2 / "runtime" / "crt.S", bundle / "runtime" / "crt.S")
    (bundle / "payload" / "single_run_harness.c").write_text(payload["harness"])
    (bundle / "payload" / "const_blob.S").write_text(payload["const_blob_s"])
    (bundle / "payload" / "const_blob.bin").write_bytes(payload["blob"])
    if payload["golden"] is not None:
        (bundle / "payload" / "golden.bin").write_bytes(payload["golden"])
    script = bundle / "build_elf.sh"
    script.write_text(_BUILD_ELF.format(
        elf=elf_name, golden_req=' \\\n  "$bundle_dir/payload/golden.bin"' if payload["golden"]
        is not None else ""))
    script.chmod(0o755)
    step = subprocess.run([str(script), str(chipyard_root())], capture_output=True, text=True)
    if step.returncode:
        raise RuntimeError(f"build_elf.sh: {step.stderr[-1500:]}")
    return bundle / elf_name


def run_elf(elf: Path, simulator: str, timeout: int) -> tuple[str, float, int]:
    """(console, wall seconds, exit status). A harness that fails its check exits non-zero; its
    console still carries the markers, so it is kept rather than discarded."""
    from merlin.runtime.backends import base as backends
    backend = backends.get_backend(TARGET)
    t0 = time.monotonic()
    try:
        console, status = backend.run_elf(elf, simulator=simulator, timeout=timeout), 0
    except Exception as exc:  # noqa: BLE001 -- the backend raises on any non-zero exit
        console, status = str(exc), 1
    return console, time.monotonic() - t0, status


def parse_console(console: str) -> dict:
    """The phase-2 markers: ``MERLIN_METRIC k=v``, ``MERLIN_RESULT ...``, ``MERLIN_PROFILE ...``."""
    got: dict = {"metrics": {}}
    for line in console.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "MERLIN_METRIC" and len(parts) == 2:
            key, _, value = parts[1].partition("=")
            got["metrics"][key] = int(value)
        elif parts[0] == "MERLIN_RESULT":
            fields = dict(p.partition("=")[::2] for p in parts[1:])
            if "checksum_fnv1a64" in fields:
                got["checksum"] = int(fields["checksum_fnv1a64"], 16)
            if "logits_checked" in fields:
                got["checked"], got["bad"] = int(fields["logits_checked"]), int(fields["bad"])
                got["check"] = fields.get("check")
        elif parts[:3] == ["MERLIN_PROFILE", "measured", "end"] and len(parts) == 4:
            got["rc"] = int(parts[3].partition("=")[2])
    got["cycles"] = got["metrics"].get("cycles")
    got["console_bytes"] = sum(len(line) + 1 for line in console.splitlines()
                               if line.startswith(("MERLIN_", "PASS", "FAIL")))
    return got


# ---------------------------------------------------------------------------------------------------
# the two arms of one conv layer
# ---------------------------------------------------------------------------------------------------
def conv_case(workload: dict, seed: int = 0):
    """Operands (NHWC input, HWIO weight, bias or None) and the exact int32 convolution plus bias."""
    k, cin, cout = workload["k"], workload["Cin"], workload["Cout"]
    stride, pad = workload.get("stride", 1), workload.get("padding", k // 2)
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, size=(1, workload["H"], workload["W"], cin), dtype=np.int64)
    w = rng.integers(-128, 128, size=(k, k, cin, cout), dtype=np.int64)
    bias = (rng.integers(-2**20, 2**20, size=cout, dtype=np.int64)
            if workload.get("bias", True) else None)
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    oh = (padded.shape[1] - k) // stride + 1
    ow = (padded.shape[2] - k) // stride + 1
    out = np.zeros((1, oh, ow, cout), dtype=np.int64)
    for fy in range(k):
        for fx in range(k):
            window = padded[:, fy:fy + stride * oh:stride, fx:fx + stride * ow:stride]
            out += np.einsum("nhwc,co->nhwo", window, w[fy, fx])
    if bias is not None:
        out = out + bias
    return x, w, bias, out.reshape(-1, cout).astype(np.int32), (oh, ow, stride, pad)


def voyager_arm(model_dir: Path, workload: dict, isa, acc_scale_ok: bool):
    schedule = vs.lower_conv(replay(load_model(model_dir / "model.json")), geometry_for(TARGET))
    x, w, bias, reference, _ = conv_case(workload)
    cin, cout = workload["Cin"], workload["Cout"]
    lhs, weight = x.reshape(-1, cin), w.reshape(-1, cout)
    operands = {"bias": bias[None, :]} if bias is not None else {}
    expected = vs.execute(schedule, lhs, weight, **operands)
    if not np.array_equal(expected, reference):
        raise AssertionError("execute() disagrees with the direct convolution")
    names = {"lhs": "lhs", "weight": "weight", "bias": "bias", "out": "out"}
    strides = {"lhs": cin, "weight": cout, "bias": cout * 4, "out": cout * 4}
    elems = {"lhs": 1, "weight": 1, "bias": 4, "out": 4}
    trace = schedule_instructions(schedule, isa, names=names, strides=strides, elems=elems,
                                  out_dtype="i32", acc_scale_ok=acc_scale_ok)
    tensors = [("lhs", "i8", lhs, lhs.shape), ("weight", "i8", weight, weight.shape)]
    if bias is not None:
        tensors.append(("bias", "i32", bias, bias.shape))
    tensors.append(("out", "i32", None, expected.shape))
    counts = {k.__name__: schedule.count(k) for k in (vs.Mvin, vs.AccMvin, vs.Preload, vs.Compute,
                                                      vs.Mvout)}
    return trace, tensors, expected, counts


def merlin_iface(workload: dict, oh: int, ow: int, stride: int, pad: int) -> str:
    h, w, k, cin, cout = workload["H"], workload["W"], workload["k"], workload["Cin"], workload["Cout"]
    ifm, wt, out = f"tensor<1x{h}x{w}x{cin}xi8>", f"tensor<{k * k * cin}x{cout}xi8>", \
        f"tensor<{oh * ow}x{cout}xi32>"
    return "\n".join([
        'module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", '
        'merlin_iface.abi_version = "0.1"} {',
        f'  %IFM = merlin_iface.tensor {{name = "IFM", role = "input"}} : {ifm}',
        f'  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : {wt}',
        f'  %W_res = merlin_iface.resident_pack %W {{layout = "packed_conv_rhs"}} : ({wt}) -> '
        "!merlin_iface.resident",
        f'  %Y0 = merlin_iface.conv2d %IFM, %W_res {{kernel = [{k}, {k}, {cin}, {cout}], stride = '
        f'[{stride}, {stride}], padding = [{pad}, {pad}, {pad}, {pad}], dilation = [1, 1], name = '
        f'"Y0", epilogue = [], output_dtype = "i32", layout = "nhwc"}} : ({ifm}, '
        f"!merlin_iface.resident) -> {out}",
        "  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()", "}", ""])


def merlin_arm(workload: dict, isa, ingest, workdir: Path):
    if workload.get("bias", True):
        raise UnsupportedConstruct("merlin_iface.conv2d has no bias operand")
    x, w, _, reference, (oh, ow, stride, pad) = conv_case(workload)
    workdir.mkdir(parents=True, exist_ok=True)
    iface = workdir / "conv.interface.mlir"
    iface.write_text(merlin_iface(workload, oh, ow, stride, pad))
    _, module = ingest.parse_verified(iface)
    program = ingest.extract_program(module)
    trace = isa.build_trace(program)          # raises the package's own refusal on capacity
    order = [n for n, spec in program.tensors.items() if spec.role in ("input", "weight", "bias",
                                                                          "output")]
    values = {"IFM": x.reshape(1, -1), "W": w.reshape(-1, workload["Cout"])}
    tensors = [(n, "i32" if n == "Y0" else "i8", None if n == "Y0" else values[n],
                reference.shape if n == "Y0" else values[n].shape) for n in order]
    counts = {name: sum(1 for i in trace if i.name == name)
              for name in ("MVIN", "PRELOAD", "COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE", "MVOUT")}
    return trace, tensors, reference, counts


# ---------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", action="append", required=True, type=Path,
                        help="a voyager_export.py output dir (model.json + manifest.json)")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE,
                        help="the reference package whose encoders pack both arms")
    parser.add_argument("--phase2-bundle", type=Path, default=DEFAULT_PHASE2,
                        help="the phase-2 bundle whose runtime and build recipe the ELFs reuse")
    parser.add_argument("--arms", nargs="+", default=["voyager", "merlin"])
    parser.add_argument("--simulators", nargs="+", default=["spike", "verilator"])
    parser.add_argument("--golden-max-bytes", type=int, default=65536,
                        help="embed the golden output (element-wise check) up to this size; "
                             "check a larger output by its digest")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    isa, ingest = load_package(args.package)
    acc_scale_ok = acc_mvin_scales(args.phase2_bundle / "runtime" / "include" / "gemmini_params.h")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = runs_dir() / TARGET / "voyager-h2h" / "emit_c" / (stamp + (f"_{args.tag}" if args.tag
                                                                     else ""))
    package_sha = hashlib.sha256((args.package / "manifest.yaml").read_bytes()).hexdigest()[:8]
    builds = []
    for model_dir in args.model_dir:
        manifest = json.loads((model_dir / "manifest.json").read_text())
        workload = manifest["workload"]
        probe = model_dir.name
        for arm in args.arms:
            work = root / probe / arm
            record = {"probe": probe, "arm": arm, "workload": workload,
                      "voyager_commit": manifest.get("voyager_commit"), "bundle": str(work)}
            try:
                if arm == "voyager":
                    trace, tensors, expected, counts = voyager_arm(model_dir, workload, isa,
                                                                   acc_scale_ok)
                    compiler = f"voyager_{str(manifest.get('voyager_commit'))[:8]}_bridge_emit_c"
                else:
                    trace, tensors, expected, counts = merlin_arm(workload, isa, ingest, work)
                    compiler = f"merlin_{args.package.name}_{package_sha}"
                record["op_counts"] = counts
                record["rocc_commands"] = sum(1 for i in trace if i.name != "FENCE")
                payload = render_bundle_payload(tensors, expected, out_dtype="i32",
                                                model=f"voyager_conv_probe_{probe}",
                                                compiler=compiler,
                                                golden_max_bytes=args.golden_max_bytes)
                elf_name = f"{probe}_{arm}_warm_measured.elf"
                elf = build_bundle(work, emit_llvm([t[0] for t in tensors], trace, isa), payload,
                                   phase2=args.phase2_bundle, elf_name=elf_name)
                record.update(elf=str(elf), elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
                              elf_bytes=elf.stat().st_size, expected_fnv1a64=f"{payload['digest']:016x}",
                              outputs=payload["elements"], check=payload["check"])
                (work / "firesim_ready.json").write_text(json.dumps({
                    "elf": elf_name, "elf_sha256": record["elf_sha256"],
                    "build": "./build_elf.sh <chipyard>", "harness": "phase-2 single-run",
                    "expect": {"MERLIN_RESULT": f"logits_checked={payload['elements']} bad=0",
                               "expected_fnv1a64": record["expected_fnv1a64"]},
                    "probe": probe, "arm": arm, "compiler": compiler}, indent=1))
            except (UnsupportedConstruct, ValueError) as exc:
                record["status"] = "refused"
                record["reason"] = f"{type(exc).__name__}: {exc}"
            except AssertionError as exc:
                # The abstract schedule itself is wrong: no program is built from it.
                record["status"] = "inexact_lowering"
                record["reason"] = str(exc)
            builds.append(record)
            print(json.dumps({k: record.get(k) for k in ("probe", "arm", "status", "reason",
                                                          "rocc_commands", "elf_bytes")}), flush=True)

    jobs = [(b, sim) for b in builds if "elf" in b for sim in args.simulators]

    def run(job):
        build, sim = job
        console, wall, status = run_elf(Path(build["elf"]), sim, args.timeout)
        (Path(build["bundle"]) / f"console.{sim}.log").write_text(console)
        got = parse_console(console)
        exact = (status == 0 and got.get("bad") == 0 and got.get("rc") == 0
                 and got.get("checksum") == int(build["expected_fnv1a64"], 16))
        return {"probe": build["probe"], "arm": build["arm"], "simulator": sim,
                "status": "pass" if exact else "fail", "exact": exact, "cycles": got.get("cycles"),
                "metrics": got["metrics"], "bad": got.get("bad"), "check": got.get("check"),
                "console_bytes": got["console_bytes"], "exit_status": status,
                "wall_s": round(wall, 1), "cycle_accurate": sim != "spike"}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        runs = []
        for result in pool.map(run, jobs):
            print(json.dumps({k: v for k, v in result.items() if k != "metrics"}), flush=True)
            runs.append(result)

    os.environ.setdefault("MERLIN_EXT_VOYAGER_COMPILER",
                          str(merlin_dir().parent / "out" / "build" / "external" / "voyager-compiler"))
    pins = {}
    for name in ("gemmini_rtl", "voyager_compiler"):
        try:
            pins[name] = provenance.verify(name)
        except Exception as exc:  # noqa: BLE001 -- drift is recorded, not fatal
            print(f"pin {name}: {exc}")
    from merlin.runtime.backends import base as _b
    backend = _b.get_backend(TARGET)
    sims = {}
    for sim in args.simulators:
        locate = getattr(backend, f"{sim}_path", None)
        if callable(locate):
            sims[sim] = Path(locate())
    product = new_product("compare", version=1, target=TARGET,
                          notes="voyager_h2h conv probes as phase-2-format Gemmini programs"
                                + (f" [{args.tag}]" if args.tag else ""))
    doc = {"target": TARGET, "package": str(args.package), "phase2_bundle": str(args.phase2_bundle),
           "acc_mvin_scale_supported": acc_scale_ok, "builds": builds, "runs": runs,
           "provenance": provenance.record(pins=pins, artifacts=sims)}
    (product.path / "results.json").write_text(json.dumps(doc, indent=1, default=str))
    lines = ["# Conv probes as phase-2-format Gemmini programs", "",
             "| probe | arm | simulator | status | cycles | check | RoCC commands | ELF bytes |",
             "|---|---|---|---|---|---|---|---|"]
    info = {(b["probe"], b["arm"]): b for b in builds}
    for r in runs:
        b = info[(r["probe"], r["arm"])]
        lines.append(f"| {r['probe']} | {r['arm']} | {r['simulator']} | {r['status']} | "
                     f"{r.get('cycles')} | {r.get('check')} bad={r.get('bad')} | "
                     f"{b.get('rocc_commands')} | {b.get('elf_bytes')} |")
    for b in builds:
        if b.get("status"):
            lines.append(f"| {b['probe']} | {b['arm']} | - | {b['status']} | - | - | - | "
                         f"{b['reason']} |")
    (product.path / "table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"product": str(product.path)}))
    return 0 if all(r["status"] == "pass" for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
