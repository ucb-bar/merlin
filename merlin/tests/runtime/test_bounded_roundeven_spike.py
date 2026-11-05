"""Actual scalar CPU code preserves guarded quantization under explicit RNE."""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower.late_quant_rne import rewrite
from merlin.llvmlower.toolchain import clang
from merlin.tests.ir.test_late_quant_rne_intrinsic import source


def kernel(bits, reverse=False):
    text = source(bits)
    if reverse:
        lower, upper = float(-(1 << (bits - 1))), float((1 << (bits - 1)) - 1)
        text = text.replace(f"maximum.f32(float %x, float {lower})", f"minimum.f32(float %x, float {upper})")
        text = text.replace(f"minimum.f32(float %lo, float {upper})", f"maximum.f32(float %lo, float {lower})")
    body, declarations = text.split("  ret ")
    body = body.split("{\n", 1)[1]
    declarations = declarations.split("}\n", 1)[1]
    return f"""define void @kernel(ptr %inputs, ptr %output, i64 %count) {{
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%next, %loop]
  %input = getelementptr float, ptr %inputs, i64 %i
  %x = load float, ptr %input
{body}
  %destination = getelementptr i{bits}, ptr %output, i64 %i
  store i{bits} %result, ptr %destination
  %next = add i64 %i, 1
  %more = icmp ult i64 %next, %count
  br i1 %more, label %loop, label %done
done:
  ret void
}}
{declarations}"""


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("reverse", [False, True])
def test_cpu_words_guards_repetition_and_instruction_counts(tmp_path, bits, reverse):
    gcc = os.environ.get("MERLIN_TEST_GCC")
    spike = os.environ.get("MERLIN_TEST_SPIKE")
    if not gcc or not spike:
        pytest.skip("explicit standalone RV64GC compiler and Spike required")
    runtime = Path(__file__).resolve().parents[2] / "runtime/baremetal/spike"
    assert runtime.is_dir()
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    half = np.arange(lower - 2, upper + 3, dtype=np.float32) + np.float32(0.5)
    rng = np.random.default_rng(181)
    random = rng.integers(0, 2**32, 12000, dtype=np.uint32).view(np.float32)
    values = np.concatenate(
        [
            half,
            np.nextafter(half, np.float32(np.inf)),
            np.nextafter(half, np.float32(-np.inf)),
            random[~np.isnan(random)],
            np.array([-np.inf, np.inf, 0.0, -0.0, 1e-40, -1e-40], np.float32),
        ]
    )
    expected = np.rint(np.clip(values, lower, upper)).astype(np.int64)
    driver = tmp_path / f"driver{bits}.c"
    driver.write_text(
        """#include <stdint.h>
#include "htif.h"
extern void kernel(const float *, void *, uint64_t);
"""
        + f"const uint32_t words[] = {{{','.join(map(str, values.view(np.uint32)))}}};\n"
        + f"const int{bits}_t expected[] = {{{','.join(map(str, expected))}}};\n"
        + f"int{bits}_t guarded[{len(values) + 16}];\n"
        + f"""int main(void) {{
  console_init();
  for (int mode=0; mode<5; ++mode) {{
    __asm__ volatile("csrw frm, %0" :: "r"(mode) : "memory");
    for (int repeat=0; repeat<2; ++repeat) {{
      for (unsigned i=0; i<{len(values) + 16}; ++i) guarded[i]=73;
      uint64_t cycle0, retired0, cycle1, retired1;
      __asm__ volatile("csrr %0, mcycle\\ncsrr %1, minstret" : "=r"(cycle0), "=r"(retired0) :: "memory");
      kernel((const float *)words, guarded+8, {len(values)});
      __asm__ volatile("csrr %0, mcycle\\ncsrr %1, minstret" : "=r"(cycle1), "=r"(retired1) :: "memory");
      for (unsigned i=0; i<{len(values)}; ++i) if (guarded[i+8]!=expected[i]) htif_exit(11);
      for (int i=0; i<8; ++i) if (guarded[i]!=73 || guarded[{len(values) + 8}+i]!=73) htif_exit(12);
      htif_puts("COUNTERS "); htif_putd(mode); htif_putc(' '); htif_putd(repeat);
      htif_putc(' '); htif_putd(cycle1-cycle0); htif_putc(' '); htif_putd(retired1-retired0); htif_putc('\\n');
    }}
  }}
  htif_puts("ALL_WORDS_AND_GUARDS_PASS\\n"); htif_exit(0);
}}"""
    )
    original = kernel(bits, reverse=reverse)
    rows = {}
    for label, combine in [("control", None), ("conversion", False), ("combined_clamp", True)]:
        work = tmp_path / f"{label}{bits}"
        work.mkdir()
        program, report = (
            (original, {}) if combine is None else rewrite(original, host_isa="rv64gc", combine_clamp=combine)
        )
        llvm, obj, elf = (work / name for name in ("kernel.ll", "kernel.o", "test.elf"))
        llvm.write_text(program)
        subprocess.run(
            [
                clang(),
                "--target=riscv64-unknown-elf",
                "-march=rv64gc",
                "-mabi=lp64d",
                "-mcmodel=medany",
                "-O3",
                "-ffp-contract=off",
                "-fno-vectorize",
                "-fno-slp-vectorize",
                "-c",
                str(llvm),
                "-o",
                str(obj),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                gcc,
                "-march=rv64gc",
                "-mabi=lp64d",
                "-mcmodel=medany",
                "-O2",
                "-ffreestanding",
                "-fno-builtin",
                "-nostdlib",
                "-nostartfiles",
                "-I" + str(runtime),
                "-T" + str(runtime / "link.ld"),
                str(runtime / "crt.S"),
                str(runtime / "htif.c"),
                str(driver),
                str(obj),
                "-lm",
                "-lgcc",
                "-o",
                str(elf),
            ],
            check=True,
            capture_output=True,
        )
        run = subprocess.run([spike, "--isa=rv64gc", "-m128", str(elf)], text=True, capture_output=True, timeout=60)
        (work / "stdout.txt").write_text(run.stdout)
        (work / "stderr.txt").write_text(run.stderr)
        assert run.returncode == 0 and "ALL_WORDS_AND_GUARDS_PASS" in run.stdout, run.stdout + run.stderr
        counters = [[int(x) for x in match] for match in re.findall(r"COUNTERS (\d+) (\d+) (\d+) (\d+)", run.stdout)]
        assert len(counters) == 10
        rows[label] = dict(
            routes=report.get("routes", []),
            counters=counters,
            llvm_sha256=hashlib.sha256(llvm.read_bytes()).hexdigest(),
            elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
        )
    receipt = dict(
        bits=bits,
        reversed_clamps=reverse,
        elements=len(values),
        expected_source="independent_numpy_rint_clip",
        rounding_modes=5,
        repeats_per_mode=2,
        rows=rows,
        counter_scope="Spike CPI=1 instruction proxy; not physical Rocket timing",
    )
    (tmp_path / f"result{bits}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert rows["combined_clamp"]["counters"][0][3] < rows["control"]["counters"][0][3]
