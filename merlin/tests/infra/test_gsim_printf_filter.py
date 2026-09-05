"""The GSIM printf shim drops only its receipt-declared high-volume stream."""
from __future__ import annotations

import shutil
import subprocess
import importlib.util
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir


_SOURCE = (
    merlin_dir()
    / "experiments/gemmini_perf_bench/scripts/gsim_printf_filter.cpp"
)
_DERIVE_SOURCE = _SOURCE.with_name("derive_gsim_printf_filter.py")
_SPEC = importlib.util.spec_from_file_location("derive_gsim_printf_filter", _DERIVE_SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
DERIVE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = DERIVE
_SPEC.loader.exec_module(DERIVE)


def test_filter_prefix_is_derived_from_unique_firrtl_disassembly_stream(
        tmp_path: Path) -> None:
    firrtl = tmp_path / "model.fir"
    firrtl.write_text(
        "FIRRTL version 3.3.0\n"
        "circuit Top :\n"
        "  module Top :\n"
        "    input clock : Clock\n"
        "    printf(clock, UInt<1>(0h1), \"Assertion failed\\n\") : p0 "
        "@[Monitor.scala 1:1]\n"
        "    printf(clock, valid, \"C%d: pc=[%x] DASM(%x)\\n\", hart, pc, inst) : p1 "
        "@[Core.scala 2:1]\n",
        encoding="utf-8",
    )
    record = DERIVE.derive(firrtl)
    assert record["selection"] == {
        "rule": "unique_printf_with_disassembly_marker",
        "marker": "DASM(",
        "line": 6,
        "source_locator": "Core.scala 2:1",
        "format": "C%d: pc=[%x] DASM(%x)\n",
        "suppress_prefix": "C%d: pc=[",
    }
    assert record["inventory"] == {"printf_count": 2, "candidate_count": 1}


def test_filter_derivation_refuses_ambiguous_disassembly_streams(tmp_path: Path) -> None:
    firrtl = tmp_path / "ambiguous.fir"
    firrtl.write_text(
        "FIRRTL version 3.3.0\n"
        "circuit Top :\n"
        "  module Top :\n"
        "    input clock : Clock\n"
        "    printf(clock, en, \"A%d x=%x DASM(%x)\\n\", a, x, i)\n"
        "    printf(clock, en, \"B%d x=%x DASM(%x)\\n\", b, x, i)\n",
        encoding="utf-8",
    )
    with pytest.raises(DERIVE.DerivationError, match="exactly one"):
        DERIVE.derive(firrtl)


def test_filter_suppresses_declared_prefix_and_preserves_diagnostics(
        tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required to exercise the GSIM printf shim")

    driver = tmp_path / "driver.cpp"
    driver.write_text(
        "#include <cstdint>\n"
        "void gprintf(const char *, ...);\n"
        "int main() {\n"
        "  gprintf(\"TRACE %d\\n\", 32, std::uint32_t{9});\n"
        "  gprintf(\"Assertion failed: code=%x\\n\", 64, std::uint64_t{42});\n"
        "}\n",
        encoding="utf-8",
    )
    executable = tmp_path / "filter_test"
    subprocess.run(
        [compiler, "-std=c++17", '-DGSIM_SUPPRESS_PRINTF_PREFIX="TRACE "',
         str(_SOURCE), str(driver), "-o", str(executable)],
        check=True,
    )
    completed = subprocess.run(
        [str(executable)], check=True, text=True, capture_output=True,
    )
    assert completed.stdout == ""
    assert completed.stderr == "Assertion failed: code=2a\n"


def test_filter_requires_an_explicit_build_time_prefix(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required to exercise the GSIM printf shim")

    completed = subprocess.run(
        [compiler, "-std=c++17", "-c", str(_SOURCE), "-o", str(tmp_path / "filter.o")],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "GSIM_SUPPRESS_PRINTF_PREFIX must be provided" in completed.stderr
