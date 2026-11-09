#!/usr/bin/env python3
"""Our schedule against the vendor call for ONE compute group, on the same buffers, on the device.

``group_model_sched_kernels`` renders a whole model both ways and reports which groups went which
way; this is the same idea narrowed to a single group so that a NEW readout mode can be accepted or
rejected on its own. It builds a bare-metal program that runs both renderings over byte-identical
inputs into two separate output buffers, times each, then compares them element for element and
prints the first index that differs. Under gsim that is the acceptance criterion: bit-exactness
against the library, with the two cycle counts beside each other.

The cases are the two readouts ResNet-50 needs and the unpooled convolution beside them, so a change
to one readout is measured against a path that already measured rather than only against itself.

    sched_vendor_ab.py --target <target> --case <case> --emulator <gsim emulator> --out <dir>
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from merlin.runtime.backends import base
from merlin.sched.codegen import emit_c_function
from merlin.sched.ir import TensorArg

CFLAGS = (
    "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math",
    "-fno-common", "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns", "-march=rv64gc",
    "-Wa,-march=rv64gc", "-lm", "-lgcc", "-DID_STRING=", "-Wno-incompatible-pointer-types",
    "-nostdlib", "-nostartfiles", "-static", "-DBAREMETAL=1",
)  # fmt: skip


def _rng(shape, seed, dtype=np.int8, lo=-127, hi=128):
    g = np.random.default_rng(seed)
    if dtype == np.int8:
        return g.integers(lo, hi, size=shape, dtype=np.int16).astype(np.int8)
    return g.integers(-(1 << 14), 1 << 14, size=shape, dtype=np.int32)


def conv_case(
    sched,
    facts,
    iset,
    *,
    name,
    in_dim,
    ci,
    co,
    kernel,
    stride,
    padding,
    pool,
    scale,
    relu,
    seed,
    lo=-127,
    hi=128,
    bias_lo=-(1 << 14),
    bias_hi=1 << 14,
):
    out_dim = (in_dim + 2 * padding - kernel) // stride + 1
    ps, pst, ppad = pool
    final = (out_dim + 2 * ppad - ps) // pst + 1 if pst else out_dim
    g = np.random.default_rng(seed + 2)
    arrays = {
        "IN": _rng((1, in_dim, in_dim, ci), seed, lo=lo, hi=hi),
        "WT": _rng((kernel, kernel, ci, co), seed + 1, lo=lo, hi=hi),
        "BS": g.integers(bias_lo, bias_hi, size=(co,), dtype=np.int32),
    }
    ops = {
        "input": TensorArg("p_in", (1, in_dim, in_dim, ci), "i8", "read"),
        "weights": TensorArg("p_w", (kernel, kernel, ci, co), "i8", "read"),
        "bias": TensorArg("p_bias", (co,), "i32", "read"),
        "output": TensorArg("p_out", (1, final, final, co), "i8", "write"),
    }
    kern = sched.conv_reference(
        name=name,
        batch=1,
        in_dim=in_dim,
        in_channels=ci,
        out_channels=co,
        kernel=kernel,
        stride=stride,
        padding=padding,
        operands=ops,
        relu=relu,
        scale=scale,
        facts=facts,
        pool_size=ps,
        pool_stride=pst,
        pool_padding=ppad,
    )
    act = "RELU" if relu else "NO_ACTIVATION"
    vendor = (
        f"tiled_conv_auto(1, {in_dim}, {in_dim}, {ci}, {co}, {out_dim}, {out_dim}, {stride}, 1, 1, "
        f"{padding}, {kernel}, false, false, false, false, false, IN, WT, BS, OUT_V, {act}, "
        f"{scale!r}f, {ps}, {pst}, {ppad}, WS);"
    )
    return {
        "name": name,
        "kernel": kern,
        "arrays": arrays,
        "elements": final * final * co,
        "ctype": "elem_t",
        "ours": f"{name}_ours((elem_t *)IN, (elem_t *)WT, (acc_t *)BS, OUT_O);",
        "vendor": vendor,
        "attrs": dict(kern.attrs),
        "definition": emit_c_function(kern, iset, symbol=f"{name}_ours"),
    }


def matmul_case(sched, facts, iset, *, name, m, k, n, scale, relu, seed, bias=True):
    arrays = {"IN": _rng((m, k), seed), "WT": _rng((k, n), seed + 1)}
    dtype = "i8" if scale is not None else "i32"
    ops = {
        "a": TensorArg("p_in", (m, k), "i8", "read"),
        "b": TensorArg("p_w", (k, n), "i8", "read"),
        "c": TensorArg("p_out", (m, n), dtype, "write"),
    }
    if bias:
        arrays["BS"] = _rng((n,), seed + 2, np.int32)
        ops["d"] = TensorArg("p_bias", (n,), "i32", "read")
    kern = sched.matmul_reference(name=name, m=m, n=n, k=k, operands=ops, relu=relu, scale=scale, facts=facts)
    act = "RELU" if relu else "NO_ACTIVATION"
    acc_scale = "ACC_SCALE_IDENTITY" if scale is None else f"{scale!r}f"
    full = "true" if scale is None else "false"
    vendor = (
        f"tiled_matmul_auto({m}, {n}, {k}, IN, WT, {'BS' if bias else 'NULL'}, OUT_V, {k}, {n}, {n}, "
        f"{n}, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, {act}, {acc_scale}, 0, "
        f"true, false, false, {full}, false, 0, WS);"
    )
    actuals = "(elem_t *)IN, (elem_t *)WT" + (", (acc_t *)BS" if bias else "") + ", OUT_O"
    return {
        "name": name,
        "kernel": kern,
        "arrays": arrays,
        "elements": m * n,
        "ctype": "acc_t" if scale is None else "elem_t",
        "ours": f"{name}_ours({actuals});",
        "vendor": vendor,
        "attrs": dict(kern.attrs),
        "definition": emit_c_function(kern, iset, symbol=f"{name}_ours"),
    }


PROGRAM = """/* GENERATED by sched_vendor_ab.py -- one compute group, our schedule beside the vendor call. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#define BLOB(sym, file, type) \\
    __asm__(".section .rodata\\n.balign 64\\n.global " #sym "\\n" #sym ":\\n.incbin \\"" file "\\"\\n.previous\\n"); \\
    extern const type sym[];
{blobs}

static {ctype} OUT_O[{elements}] row_align(1);
static {ctype} OUT_V[{elements}] row_align(1);

{definition}

int main(void) {{
    gemmini_flush(0);
    uint64_t t0, ours, theirs;
    printf("MERLIN_PROFILE warmup begin\\n");
    {ours}
    {vendor}
    printf("MERLIN_PROFILE warmup end rc=0\\n");
    printf("MERLIN_PROFILE measured begin\\n");
    t0 = read_cycles(); {ours} ours = read_cycles() - t0;
    t0 = read_cycles(); {vendor} theirs = read_cycles() - t0;
    long long diffs = 0; long long first = -1;
    long long sum_o = 0, sum_v = 0;
    for (size_t i = 0; i < {elements}; i++) {{
        sum_o += (long long)OUT_O[i];
        sum_v += (long long)OUT_V[i];
        if (OUT_O[i] != OUT_V[i]) {{ if (first < 0) first = (long long)i; diffs++; }}
    }}
    printf("AB_GROUP {name} elements={elements}\\n");
    printf("AB_CYCLES ours=%llu vendor=%llu\\n", (unsigned long long)ours, (unsigned long long)theirs);
    printf("AB_DIFFS %lld first=%lld sum_ours=%lld sum_vendor=%lld\\n", diffs, first, sum_o, sum_v);
    printf("AB_VERDICT %s\\n", diffs == 0 ? "BIT_EXACT" : "DIFFERS");
    printf("MERLIN_PROFILE measured end rc=0\\n");
    return 0;
}}
"""


def build(case, out: Path, harness: Path, compiler: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for symbol, array in case["arrays"].items():
        (out / f"{symbol}.bin").write_bytes(np.ascontiguousarray(array).tobytes())
    blobs = "\n".join(f'BLOB({s}, "{s}.bin", {"acc_t" if s == "BS" else "elem_t"})' for s in case["arrays"])
    source = out / "ab.c"
    source.write_text(
        PROGRAM.format(
            blobs=blobs,
            ctype=case["ctype"],
            elements=case["elements"],
            definition=case["definition"],
            ours=case["ours"],
            vendor=case["vendor"],
            name=case["name"],
        ),
        encoding="utf-8",
    )
    common = harness / "riscv-tests" / "benchmarks" / "common"
    includes = [f"-I{harness / 'riscv-tests'}", f"-I{harness / 'riscv-tests' / 'env'}", f"-I{harness}", f"-I{common}"]
    objects, log = [], []
    for src in (source, common / "syscalls.c", common / "crt.S"):
        unit = out / f"{src.stem}.o"
        done = subprocess.run(
            [str(compiler), *CFLAGS, *includes, f"-Wa,-I{out}", "-c", str(src), "-o", str(unit)],
            capture_output=True, text=True, cwd=out,
        )  # fmt: skip
        log.append(done.stdout + done.stderr)
        if done.returncode:
            (out / "build.log").write_text("".join(log), encoding="utf-8")
            raise SystemExit(f"compile of {src.name} failed:\n{done.stderr[-3000:]}")
        objects.append(str(unit))
    elf = out / "ab.elf"
    done = subprocess.run(
        [str(compiler), *CFLAGS, "-T", str(common / "test.ld"), *objects, "-o", str(elf)],
        capture_output=True, text=True, cwd=out,
    )  # fmt: skip
    (out / "build.log").write_text("".join(log) + done.stdout + done.stderr, encoding="utf-8")
    if done.returncode:
        raise SystemExit(f"link failed:\n{done.stderr[-3000:]}")
    return elf


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=["stem_pooled", "classifier_full_acc", "stem_unpooled", "stem_pooled_graded"],
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--emulator", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--max-cycles", default="400000000")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)

    backend = base.get_backend(args.target)
    iset = backend.sched_instruction_set()
    sched = importlib.import_module(f"merlin._oot_backends.{args.target}.{args.target}_sched")
    facts = iset.facts
    harness = backend.rocc_tests_dir()
    compiler = backend.gcc_path()

    if args.case == "stem_pooled":
        case = conv_case(
            sched, facts, iset, name="g1", in_dim=224, ci=3, co=64, kernel=7, stride=2, padding=3,
            pool=(3, 2, 1), scale=0.0234375, relu=True, seed=11,
        )  # fmt: skip
    elif args.case == "stem_pooled_graded":
        # Operands small enough that the readout does NOT saturate: an all-127 output would make a
        # max pool trivially right whatever window it read.
        case = conv_case(
            sched, facts, iset, name="g1g", in_dim=224, ci=3, co=64, kernel=7, stride=2, padding=3,
            pool=(3, 2, 1), scale=0.015625, relu=False, seed=23, lo=-8, hi=9, bias_lo=-64, bias_hi=65,
        )  # fmt: skip
    elif args.case == "stem_unpooled":
        case = conv_case(
            sched, facts, iset, name="g1u", in_dim=224, ci=3, co=64, kernel=7, stride=2, padding=3,
            pool=(0, 0, 0), scale=0.0234375, relu=True, seed=11,
        )  # fmt: skip
    else:
        case = matmul_case(sched, facts, iset, name="g71", m=1, k=2048, n=1000, scale=None, relu=False, seed=71)

    elf = build(case, args.out, harness, Path(compiler))
    (args.out / "case.json").write_text(
        json.dumps({"case": args.case, "target": args.target, "attrs": case["attrs"], "elf": str(elf)}, indent=2),
        encoding="utf-8",
    )
    print(f"built {elf}: {case['attrs']}")
    if args.build_only:
        return 0
    done = subprocess.run(
        [str(args.emulator), str(elf), f"+max-cycles={args.max_cycles}", f"+loadmem={elf}"],
        capture_output=True, text=True, cwd=args.out,
    )  # fmt: skip
    (args.out / "run.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    for line in (done.stdout + done.stderr).splitlines():
        if line.startswith("AB_"):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
