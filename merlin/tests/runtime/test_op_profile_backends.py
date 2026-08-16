"""The per-op profiler compiles for three substrates, and each needs BOTH a console and a counter.

Selecting one without the other is not a theoretical mistake: the bare-metal backend was first added
with an HTIF console but no counter selection, so it fell through to `rdtime` -- CSR 0xc01, which needs
a delegated timer that a bare M-mode image does not have. The first mark took an illegal-instruction
trap (mcause 2, mtval 0xc0102773 = `csrrs a4, time, x0`) a few instructions into the model, and the
whole run was lost to it. These compile the real C for RISC-V and read the emitted instruction back.
"""
from __future__ import annotations

import subprocess

import pytest

from merlin.common.paths import runtime_dir


def _compile_asm(tmp_path, defines):
    """Compile the profiler to assembly for rv64, or skip if the bare-metal toolchain is absent."""
    from merlin.runtime.backends import spike as spike_backend
    try:
        gcc = spike_backend.gcc_path()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no riscv toolchain: {e}")
    if not gcc or not gcc.is_file():
        pytest.skip("no riscv toolchain")
    src = runtime_dir() / "c" / "merlin_op_prof.c"
    out = tmp_path / "prof.s"
    got = subprocess.run([str(gcc), "-march=rv64gcv", "-mabi=lp64d", "-ffreestanding", "-nostdlib",
                          "-O2", "-S", *defines, "-I", str(runtime_dir() / "baremetal" / "spike"),
                          str(src), "-o", str(out)], capture_output=True, text=True)
    if got.returncode != 0:
        pytest.fail(f"the profiler does not compile with {defines}:\n{got.stderr[-2000:]}")
    return out.read_text()


def test_the_baremetal_backend_reads_mcycle_not_rdtime(tmp_path):
    asm = _compile_asm(tmp_path, ["-DMERLIN_PROF_BAREMETAL"])
    assert "mcycle" in asm, "the bare-metal profiler must read mcycle"
    assert "rdtime" not in asm, (
        "rdtime needs a delegated timer this substrate does not have; it traps on the first mark")


def test_the_baremetal_backend_prints_through_the_harness_console(tmp_path):
    """-nostdlib means there is no printf to fall back on, so the console must be the harness's own."""
    asm = _compile_asm(tmp_path, ["-DMERLIN_PROF_BAREMETAL"])
    assert "htif_puts" in asm and "printf" not in asm


def test_the_default_backend_still_uses_rdtime(tmp_path):
    """The K1 path is the reason rdtime is there at all -- its kernel traps userspace rdcycle."""
    asm = _compile_asm(tmp_path, [])
    assert "rdtime" in asm
