"""Scalar CPU rounding helpers compile without any accelerator implementation."""

import os
import subprocess

import numpy as np
import pytest

from merlin.common.paths import merlin_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.host_rounding import rewrite_roundeven
from merlin.llvmlower.toolchain import clang, objdump
from merlin.runtime.host_rounding import roundeven_f32_assembly


def test_scalar_rewrite_preserves_other_formats_and_custom_abi():
    module = parse_mlir_text("""module {
      func.func @forward(%a: f32, %b: f64) -> (f32, f64) {
        %x = math.roundeven %a : f32
        %y = math.roundeven %b : f64
        return %x, %y : f32, f64
      }
    }""")
    assert rewrite_roundeven(module, symbol="custom_roundeven") == 1
    assert "func.call @custom_roundeven" in str(module)
    assert str(module).count("math.roundeven") == 1
    assert rewrite_roundeven(module) == 0


@pytest.mark.parametrize(
    "isa,abi,symbol",
    [
        ("unknown", "lp64d", "rounder"),
        ("rv64gc", "lp64", "rounder"),
        ("rv64gc", "lp64d", "x;bad"),
        ("rv64gc", "lp64d", "1bad"),
    ],
)
def test_missing_capability_and_invalid_symbols_refuse(isa, abi, symbol):
    with pytest.raises(ValueError):
        roundeven_f32_assembly(isa=isa, abi=abi, symbol=symbol)


def test_actual_cpu_object_uses_fixed_rne_and_preserves_flags(tmp_path):
    source, obj = tmp_path / "rounding.S", tmp_path / "rounding.o"
    source.write_text(roundeven_f32_assembly(isa="rv64gc", abi="lp64d", symbol="custom_roundeven"))
    subprocess.run(
        [
            str(clang()),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    listing = subprocess.check_output([str(objdump()), "-d", str(obj)], text=True, timeout=30)
    assert "<custom_roundeven>" in listing
    assert "fcvt.w.s" in listing and "fcvt.s.w" in listing and "rne" in listing
    assert "frflags" in listing and "fsflags" in listing


@pytest.mark.parametrize("selected_runtime", [False, True])
def test_cpu_all_modes_values_flags_and_signed_zero_on_spike(tmp_path, selected_runtime):
    gcc, spike = os.environ.get("MERLIN_TEST_GCC"), os.environ.get("MERLIN_TEST_SPIKE")
    if not gcc or not spike:
        pytest.skip("explicit standalone RV64GC compiler and Spike required")
    words = np.concatenate(
        [
            np.array(
                [
                    0,
                    0x80000000,
                    1,
                    0x80000001,
                    0x3F000000,
                    0xBF000000,
                    0x3FC00000,
                    0xBFC00000,
                    0x4AFFFFFF,
                    0x4B000000,
                    0x7F800000,
                    0xFF800000,
                    0x7FC00001,
                    0x7F800001,
                    0xFFC00001,
                ],
                np.uint32,
            ),
            np.random.default_rng(98).integers(0, 2**32, 4096, dtype=np.uint32),
        ]
    )
    with np.errstate(invalid="ignore"):
        expected = np.rint(words.view(np.float32)).view(np.uint32)
    nan = (words & 0x7FFFFFFF) > 0x7F800000
    expected[nan] = 0x7FC00000
    invalid = nan & ((words & 0x00400000) == 0)
    asm, driver, elf = (tmp_path / name for name in ("round.S", "driver.c", "round.elf"))
    symbol = "roundevenf" if selected_runtime else "merlin_host_roundevenf"
    asm.write_text(roundeven_f32_assembly(isa="rv64gc", abi="lp64d", symbol=symbol))
    driver.write_text(
        """#include <stdint.h>
#include "htif.h"
extern float merlin_host_roundevenf(float);
"""
        + f"const uint32_t words[] = {{{','.join(map(str, words))}}};\n"
        + f"const uint32_t expected[] = {{{','.join(map(str, expected))}}};\n"
        + f"const uint8_t invalid[] = {{{','.join(str(int(x)) for x in invalid)}}};\n"
        + f"""int main(void) {{
  console_init();
  for (unsigned mode=0; mode<5; ++mode) {{
    __asm__ volatile("csrw frm, %0" :: "r"(mode) : "memory");
    for (unsigned repeat=0; repeat<2; ++repeat)
      for (unsigned i=0; i<{len(words)}; ++i) {{
        unsigned flags=(i+repeat)&31, actual_flags, actual_mode;
        union {{ uint32_t u; float f; }} input={{.u=words[i]}}, output;
        __asm__ volatile("csrw fflags, %0" :: "r"(flags) : "memory");
        output.f=merlin_host_roundevenf(input.f);
        __asm__ volatile("csrr %0, fflags\\ncsrr %1, frm"
                         : "=r"(actual_flags), "=r"(actual_mode) :: "memory");
        if (output.u!=expected[i]) htif_exit(11);
        if (actual_flags!=(flags|(invalid[i]?16:0))) htif_exit(12);
        if (actual_mode!=mode) htif_exit(13);
      }}
  }}
  htif_puts("ROUND_WORDS_FLAGS_AND_MODES_PASS\\n"); htif_exit(0);
}}"""
    )
    if selected_runtime:
        driver.write_text(driver.read_text().replace("merlin_host_roundevenf", symbol))
    runtime = merlin_dir() / "runtime/baremetal/spike"
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
            str(asm),
            *(
                [
                    "-DMERLIN_EXTERNAL_ROUNDEVEN_F32",
                    "-ffunction-sections",
                    "-fdata-sections",
                    "-Wl,--gc-sections",
                    str(merlin_dir() / "runtime/abi/mlir_runtime.c"),
                ]
                if selected_runtime
                else []
            ),
            "-lgcc",
            "-o",
            str(elf),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    run = subprocess.run([spike, "--isa=rv64gc", "-m128", str(elf)], text=True, capture_output=True, timeout=60)
    assert run.returncode == 0 and "ROUND_WORDS_FLAGS_AND_MODES_PASS" in run.stdout, run.stdout + run.stderr
