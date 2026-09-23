#!/usr/bin/env python3
"""What closing a compute group is worth on the unit, measured with the vendor's own library.

The compute-group pass closes a fake-quantized layer into one device program: the contraction with
its bias, requantization and activation applied by the unit's readout. When a group does not close,
a route emits the contraction alone, reads the bare accumulator back, and runs the rest as host
loops over every element. This builds one bare-metal program that runs BOTH, on the layers a
captured model actually forms, so the difference is a measurement on the FPGA and not a model:

* ``grouped``     one library call: bias in, scale and activation in the readout, int8 out;
* ``split``       the same contraction with a full-width accumulator readout, then the library's
                  own ``scale_and_sat`` over every element on the host. That is the host epilogue
                  the vendor itself would write, in one fused loop, which is the BEST case for the
                  split: a compiled route runs it as several passes.

A model's integer SUMS (residual connections a unit's scaled load adds) are measured the same way:
the library's own residual add on the unit, against the single fused host loop that computes the
single-rounding reference. Those two are different functions by construction, so that row is not
held to byte equality: it is held to the group's DECLARED bound, counted on the FPGA, which is the
first time the bound is checked against RTL and not against a model of it. A multiplier above one
is divided out of the loads and given to the readout's scale, as the group's program states.

Both contraction arms must produce identical bytes or the row is void. Layers come from a
``group_capsules.json`` (:mod:`merlin.targetgen.group_capsules`), so the extents are the model's
and nothing here is typed by hand. Only contraction-form groups are run: the library states a
convolution's bias and scale too, and those rows can be added the same way.

The program speaks the warm-profile window (one unmeasured pass, one measured pass) and prints one
``GROUPED_TOTAL cycles:`` line for the checkpoint harness to observe, beside per-layer rows.

    group_readout_ab.py --groups <group_capsules.json> --vendor-source <snapshot>/source \\
        --compiler <riscv64-unknown-elf-gcc> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCHEMA = "group_readout_ab_v1"
#: The flags the library baseline is built with, so the two programs differ only in their source.
CFLAGS = (
    "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math",
    "-fno-common", "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns", "-march=rv64gc",
    "-Wa,-march=rv64gc", "-lm", "-lgcc", "-DID_STRING=", "-Wno-incompatible-pointer-types",
    "-nostdlib", "-nostartfiles", "-static", "-DBAREMETAL=1",
)  # fmt: skip


def layers(report: dict) -> list[dict]:
    """The closed contraction-form groups of a group-capsule report, with how often each occurs."""
    found = []
    for row in report.get("entries") or ():
        entry = row.get("entry") or {}
        stages = list(entry.get("epilogue") or ())
        if entry.get("op") != "matmul" or "acc_scale" not in stages or row.get("raw_of"):
            continue
        found.append(
            {
                "name": str(row["name"]),
                "count": int(row.get("count") or 1),
                "M": int(entry["M"]),
                "K": int(entry["K"]),
                "N": int(entry["N"]),
                "scale": float(entry["acc_scale"]),
                "bias": any(stage in ("bias_add", "bias") for stage in stages),
                "relu": "relu" in stages,
            }
        )
    return found


def sums(report: dict) -> list[dict]:
    """The integer sums of a group-capsule report, with their multipliers and declared bound."""
    found = []
    for row in report.get("entries") or ():
        entry = row.get("entry") or {}
        if entry.get("op") != "residual_add" or row.get("raw_of"):
            continue
        found.append(
            {
                "name": str(row["name"]),
                "count": int(row.get("count") or 1),
                "rows": int(entry["M"]),
                "cols": int(entry["N"]),
                "elements": int(entry["M"]) * int(entry["N"]),
                "lhs_scale": float(entry["lhs_scale"]),
                "rhs_scale": float(entry["rhs_scale"]),
                "bound": int(entry["bound_lsb"]),
                "relu": "relu" in (entry.get("epilogue") or ()),
            }
        )
    return found


def _render_sums(rows: list[dict]) -> tuple[str, str, str]:
    """``(declarations, function, call)`` for the integer sums; empty strings when there are none."""
    if not rows:
        return "", "", ""
    biggest = max(row["elements"] for row in rows)
    table = ",\n".join(
        f'    {{"{row["name"]}", {row["rows"]}, {row["cols"]}, {row["lhs_scale"]!r}f, {row["rhs_scale"]!r}f, '
        f"{int(row['relu'])}, {row['bound']}, {row['count']}}}"
        for row in rows
    )
    declarations = f"""
typedef struct {{
    const char *name; size_t rows, cols; float lhs, rhs; int relu, bound, count;
}} sum_t;

static const sum_t SUMS[] = {{
{table}
}};
#define N_SUMS (sizeof(SUMS) / sizeof(SUMS[0]))

static elem_t SUM_A[{biggest}] row_align(1);
static elem_t SUM_B[{biggest}] row_align(1);
static elem_t SUM_DEVICE[{biggest}] row_align(1);
static elem_t SUM_HOST[{biggest}] row_align(1);
"""
    function = """
static void run_sums(int measured) {
    uint64_t device_total = 0, host_total = 0, outside = 0;
    for (size_t s = 0; s < N_SUMS; s++) {
        const sum_t *S = &SUMS[s];
        const size_t elements = S->rows * S->cols;
        for (size_t i = 0; i < elements; i++) { SUM_A[i] = next_value(127); SUM_B[i] = next_value(127); }
        /* A load saturates its operand to the element type before the add, so a multiplier above
           one goes to the readout: both loads are divided by the larger and the readout scales by it. */
        float factor = S->lhs > S->rhs ? S->lhs : S->rhs;
        if (factor < 1.0f) factor = 1.0f;

        uint64_t t0 = read_cycles();
        tiled_resadd_auto(S->rows, S->cols, S->lhs / factor, S->rhs / factor, factor,
                          SUM_A, SUM_B, SUM_DEVICE, S->relu, WS);
        uint64_t device = read_cycles() - t0;

        /* The host region this replaces, at its best: one fused loop, rounding ONCE. */
        t0 = read_cycles();
        const int low = S->relu ? 0 : -128;
        for (size_t i = 0; i < elements; i++) {
            float y = ROUND_NEAR_EVEN(SUM_A[i] * S->lhs + SUM_B[i] * S->rhs);
            SUM_HOST[i] = y > 127 ? 127 : (y < low ? low : (elem_t)y);
        }
        uint64_t host = read_cycles() - t0;

        uint64_t differing = 0, beyond = 0; int worst = 0;
        for (size_t i = 0; i < elements; i++) {
            int d = (int)SUM_DEVICE[i] - (int)SUM_HOST[i];
            if (d < 0) d = -d;
            differing += d != 0;
            beyond += d > S->bound;
            if (d > worst) worst = d;
        }
        outside += beyond;
        device_total += device * S->count;
        host_total += host * S->count;
        if (measured)
            printf("AB_SUM %s count=%d elements=%llu device=%llu host=%llu differing=%llu worst=%d bound=%d beyond_bound=%llu\\n",
                   S->name, S->count, (unsigned long long)elements, (unsigned long long)device,
                   (unsigned long long)host, (unsigned long long)differing, worst, S->bound,
                   (unsigned long long)beyond);
    }
    if (measured) {
        printf("AB_SUM_WEIGHTED device=%llu host=%llu\\n",
               (unsigned long long)device_total, (unsigned long long)host_total);
        printf("AB_SUM_BEYOND_BOUND %llu\\n", (unsigned long long)outside);
        printf("SUM_TOTAL cycles: %llu\\n", (unsigned long long)device_total);
    }
}
"""
    return declarations, function, "    run_sums(measured);\n"


def render(rows: list[dict], sum_rows: list[dict] | None = None) -> str:
    """The C program. One table row per layer; buffers are sized for the largest."""
    sum_declarations, sum_function, sum_call = _render_sums(list(sum_rows or ()))
    biggest = {key: max(row[key] for row in rows) for key in ("M", "K", "N")}
    size = {
        "a": max(row["M"] * row["K"] for row in rows),
        "w": max(row["K"] * row["N"] for row in rows),
        "y": max(row["M"] * row["N"] for row in rows),
    }
    table = ",\n".join(
        f'    {{"{row["name"]}", {row["M"]}, {row["K"]}, {row["N"]}, {row["scale"]!r}f, '
        f"{int(row['bias'])}, {int(row['relu'])}, {row['count']}}}"
        for row in rows
    )
    return f"""/* GENERATED by group_readout_ab.py -- the layers are a captured model's compute groups. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

typedef struct {{
    const char *name; size_t m, k, n; float scale; int bias, relu, count;
}} layer_t;

static const layer_t LAYERS[] = {{
{table}
}};
#define N_LAYERS (sizeof(LAYERS) / sizeof(LAYERS[0]))

static elem_t A[{size["a"]}] row_align(1);
static elem_t W[{size["w"]}] row_align(1);
static acc_t BIAS[{biggest["N"]}] row_align_acc(1);
static elem_t OUT_GROUPED[{size["y"]}] row_align(1);
static acc_t ACC[{size["y"]}] row_align_acc(1);
static elem_t OUT_SPLIT[{size["y"]}] row_align(1);

static uint32_t lcg_state = 12345u;
static int next_value(int span) {{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (int)((lcg_state >> 16) % (2 * span + 1)) - span;   /* signed: a readout has to clamp */
}}

{sum_declarations}
{sum_function}
static void run(int measured) {{
    uint64_t grouped_total = 0, device_total = 0, host_total = 0, mismatches = 0;
    for (size_t l = 0; l < N_LAYERS; l++) {{
        const layer_t *L = &LAYERS[l];
        for (size_t i = 0; i < L->m * L->k; i++) A[i] = next_value(8);
        for (size_t i = 0; i < L->k * L->n; i++) W[i] = next_value(8);
        for (size_t j = 0; j < L->n; j++) BIAS[j] = L->bias ? next_value(2000) : 0;
        const int act = L->relu ? RELU : NO_ACTIVATION;

        uint64_t t0 = read_cycles();
        tiled_matmul_auto(L->m, L->n, L->k, A, W, L->bias ? BIAS : NULL, OUT_GROUPED,
                          L->k, L->n, L->n, L->n,
                          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                          act, L->scale, 0, true, false, false, false, false, 0, WS);
        uint64_t grouped = read_cycles() - t0;

        t0 = read_cycles();
        tiled_matmul_auto(L->m, L->n, L->k, A, W, NULL, ACC,
                          L->k, L->n, L->n, L->n,
                          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                          NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, false, false, true, false, 0, WS);
        uint64_t device = read_cycles() - t0;

        t0 = read_cycles();
        for (size_t i = 0; i < L->m; i++)
            for (size_t j = 0; j < L->n; j++)
                OUT_SPLIT[i * L->n + j] = scale_and_sat(ACC[i * L->n + j] + BIAS[j], act, L->scale, 0);
        uint64_t host = read_cycles() - t0;

        uint64_t bad = 0;
        for (size_t i = 0; i < L->m * L->n; i++) bad += OUT_GROUPED[i] != OUT_SPLIT[i];
        mismatches += bad;
        grouped_total += grouped * L->count;
        device_total += device * L->count;
        host_total += host * L->count;
        if (measured)
            printf("AB_LAYER %s count=%d grouped=%llu split_device=%llu split_host=%llu mismatches=%llu\\n",
                   L->name, L->count, (unsigned long long)grouped, (unsigned long long)device,
                   (unsigned long long)host, (unsigned long long)bad);
    }}
    if (measured) {{
        printf("AB_WEIGHTED split_device=%llu split_host=%llu\\n",
               (unsigned long long)device_total, (unsigned long long)host_total);
        printf("AB_MISMATCHES %llu\\n", (unsigned long long)mismatches);
        printf("GROUPED_TOTAL cycles: %llu\\n", (unsigned long long)grouped_total);
    }}
{sum_call}}}

int main(void) {{
    gemmini_flush(0);
    printf("MERLIN_PROFILE warmup begin\\n");
    run(0);
    printf("MERLIN_PROFILE warmup end rc=0\\n");
    lcg_state = 12345u;
    printf("MERLIN_PROFILE measured begin\\n");
    run(1);
    printf("MERLIN_PROFILE measured end rc=0\\n");
    return 0;
}}
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: str, vendor: Path, compiler: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    program, elf = out / "group_readout_ab.c", out / "group_readout_ab.elf"
    program.write_text(source, encoding="utf-8")
    common = vendor / "riscv-tests" / "benchmarks" / "common"
    command = [
        str(compiler), *CFLAGS,
        f"-I{vendor / 'riscv-tests'}", f"-I{vendor / 'riscv-tests' / 'env'}", f"-I{vendor}", f"-I{common}",
        "-T", str(common / "test.ld"), str(program), "-o", str(elf),
        str(common / "syscalls.c"), str(common / "crt.S"),
    ]  # fmt: skip
    done = subprocess.run(command, capture_output=True, text=True)
    (out / "build.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    if done.returncode != 0:
        raise SystemExit(f"build failed (see {out / 'build.log'}):\n{done.stderr[-2000:]}")
    return {
        "elf": str(elf),
        "elf_sha256": _sha256(elf),
        "program_sha256": _sha256(program),
        "parameter_header_sha256": _sha256(vendor / "include" / "gemmini_params.h"),
        "library_header_sha256": _sha256(vendor / "include" / "gemmini.h"),
        "compiler": str(compiler),
        "compiler_sha256": _sha256(compiler),
        "flags": list(CFLAGS),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", required=True, type=Path, help="a group_capsules.json")
    parser.add_argument("--vendor-source", required=True, type=Path, help="a snapshot's source/ directory")
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.groups.read_text(encoding="utf-8"))
    rows, sum_rows = layers(report), sums(report)
    if not rows:
        print("no closed contraction-form group in the report: nothing to measure", file=sys.stderr)
        return 2
    receipt = build(render(rows, sum_rows), args.vendor_source, args.compiler, args.out)
    manifest = {
        "schema": SCHEMA,
        "model": report.get("model"),
        "groups_report_sha256": _sha256(args.groups),
        "layers": rows,
        "sums": sum_rows,
        "groups_covered": sum(row["count"] for row in rows) + sum(row["count"] for row in sum_rows),
        **receipt,
    }
    (args.out / "group_readout_ab.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"built {receipt['elf']} over {len(rows)} layer(s) standing for {manifest['groups_covered']} group(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
