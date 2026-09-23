"""The runtime's soft-BF16 helpers must use the register class LLVM callers use."""

from __future__ import annotations

import re
import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower import toolchain
from merlin.mining import k1


def _symbol_disassembly(objdump, obj, symbol: str) -> str:
    result = subprocess.run(
        [str(objdump), "-d", f"--disassemble-symbols={symbol}", str(obj)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"could not disassemble {symbol}:\n{result.stderr[-2000:]}")
    return result.stdout


def test_k1_clang_bf16_helpers_use_floating_register_class(tmp_path):
    """Catch the a0/fa0 mismatch that zeroed smolVLA's first BF16 projection."""
    cc = k1.toolchain_cc()
    objdump = toolchain.objdump()
    if cc is None or not objdump.is_file():
        pytest.skip("K1 compiler or LLVM objdump is not installed")

    source = runtime_dir() / "abi" / "mlir_runtime.c"
    obj = tmp_path / "mlir_runtime.o"
    result = subprocess.run(
        [
            str(cc),
            "--target=riscv64-unknown-linux-gnu",
            f"-march={k1.K1_MARCH}",
            f"-mabi={k1.K1_MABI}",
            "-O2",
            "-ffreestanding",
            "-fno-builtin",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"K1 runtime compile failed:\n{result.stderr[-2000:]}")

    trunc = _symbol_disassembly(objdump, obj, "__truncsfbf2")
    extend = _symbol_disassembly(objdump, obj, "__extendbfsf2")
    assert re.search(r"fmv\.w\.x\s+fa0,", trunc), trunc
    assert re.search(r"fmv\.x\.w\s+\w+,\s*fa0", extend), extend
