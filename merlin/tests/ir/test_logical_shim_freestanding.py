"""Descriptor initialization must not add a freestanding libc dependency."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower.device_shim import LogicalBuffer, emit_logical_translation_unit


@pytest.mark.parametrize("cross", [False, True])
def test_small_logical_descriptor_compiles_without_libc_memory_calls(tmp_path, cross):
    if cross:
        cc = os.environ.get("MERLIN_RISCV_GCC")
        if not cc:
            pytest.skip("select MERLIN_RISCV_GCC for the freestanding cross-compiler regression")
        nm = str(Path(cc).with_name(Path(cc).name.removesuffix("gcc") + "nm"))
    else:
        cc, nm = shutil.which("cc"), shutil.which("nm")
    if cc is None or nm is None:
        pytest.skip("host compiler and nm required")
    buffers = (
        LogicalBuffer("source", "input", "input_0", (17, 33), "f32"),
        LogicalBuffer("destination", "output", "out_0", (700,), "i8"),
    )
    unit = emit_logical_translation_unit(
        "synthetic",
        {"prepare": (buffers, [{"role": b.role, "shape": b.shape, "dtype": b.dtype} for b in buffers])},
        kernel_symbol_for=lambda _: "prepare_kernel",
    )
    assert not unit.skipped
    source, obj = tmp_path / "shim.c", tmp_path / "shim.o"
    source.write_text(unit.text)
    subprocess.run(
        [cc, "-O2", "-ffreestanding", "-fno-builtin", "-c", str(source), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    symbols = subprocess.run(
        [nm, "--undefined-only", "--format=posix", str(obj)], check=True, capture_output=True, text=True
    )
    undefined = {line.split()[0] for line in symbols.stdout.splitlines() if line.split()}
    assert "prepare_kernel" in undefined
    assert not undefined & {"memset", "memcpy", "memmove"}, symbols.stdout
