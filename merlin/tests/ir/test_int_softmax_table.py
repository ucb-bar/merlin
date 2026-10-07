"""``int-softmax-table``: the integer softmax restructured over its captured IR, and still the same bits.

The rewrite (``merlin.llvmlower.int_softmax_table``; its runner half is ``_int_softmax_table_rt.py``)
turns the integer exp and floor division per element into a table read, drops the never-binding upper
clamp, makes the lower one a compare-and-select, sums each row in i32, quantizes the probabilities once
per row on their candidate values, and moves the score scale before a reshape. Every one of those is
claimed EXACT. What this pins:

* the pass is listed, default off, and every lowering runner variant carries it behind its own gate;
* the runner source is self-contained (the compiler's Python runs it) and its integer evaluator has the
  IR's semantics, refusing a domain point where an op would be poison;
* the comparison over every float32 ``x - max <= 0``: the select reads the clamp's grid index;
* in the compiler's Python (skipped when absent): the original and the rewritten module, executed on the
  same inputs, agree bit for bit -- the table at every grid index against the capture's own definition,
  the softmax alone, and two whole int8 attentions including flat rows -- a mismatch is caught, and a
  module that differs from the structure is left alone;
* through the real lowering, selecting the pass by name reaches the runner and its report.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.llvmlower import int_softmax_table as IST
from merlin.llvmlower import optional_passes as OP
from merlin.llvmlower import toolchain

_DATA = merlin_dir() / "tests" / "data" / "int_softmax_table"
_EXACTNESS = merlin_dir() / "tests" / "fixtures" / "integer_softmax_ir_exactness.py"
_TARGETGEN = repo_root() / "src" / "merlin" / "targetgen"

m2m = pytest.mark.skipif(not toolchain.m2m_python().is_file(), reason="the compiler's Python (torch-mlir) is absent")


def _rt():
    import importlib

    return importlib.import_module("merlin.llvmlower._int_softmax_table_rt")


def test_it_is_listed_default_off_and_registered():
    entry = OP.get("int-softmax-table")
    assert entry.exactness == OP.EXACT and entry.default == "off" and entry.feature == IST.FEATURE
    from merlin.llvmlower import impr_features

    assert impr_features.get(IST.FEATURE).action_class == "PASS"


def test_every_runner_variant_runs_it_behind_its_gate():
    from merlin.llvmlower import accum_microkernel, pipeline

    gate = f"_INT_SOFTMAX_TABLE = len(sys.argv) > {IST.ARGV_INDEX} and sys.argv[{IST.ARGV_INDEX}] == '1'"
    for source in (pipeline._RUNNER_SRC, pipeline._activation_poly_runner(), accum_microkernel.run_source()):
        assert gate in source
        assert "if _INT_SOFTMAX_TABLE:\n    _ist_run_and_report(ctx, module)\n" in source


def test_the_runner_half_is_self_contained():
    tree = ast.parse(IST.runtime_source())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"torch_mlir", "json"}, imported
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert node.name.startswith(("_ist", "_Ist", "_int_softmax_table")), node.name
        elif isinstance(node, ast.Assign):
            assert all(t.id.startswith("_IST") for t in node.targets if isinstance(t, ast.Name))


def _evaluate(op: str, a: int, b: int, width: int, extra=None) -> int:
    rt = _rt()
    steps = [("const", 1, (), 0, b), (op, 2, (0, 1), (width, width, width), extra)]
    return rt._ist_run(steps, 3, a)[2]


def test_the_evaluator_has_the_integer_semantics_of_the_ir():
    rt = _rt()
    rng = np.random.default_rng(0)
    for width in (8, 32, 64):
        lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
        values = [lo, lo + 1, -1, 0, 1, hi - 1, hi, *map(int, rng.integers(lo, hi, 40, endpoint=True))]
        for a in values:
            for b in values:
                assert _evaluate("arith.addi", a, b, width) == rt._ist_signed(a + b, width)
                assert _evaluate("arith.muli", a, b, width) == rt._ist_signed(a * b, width)
                assert _evaluate("arith.minui", a, b, width) == (a if a % (1 << width) <= b % (1 << width) else b)
                if b != 0 and not (a == lo and b == -1):
                    assert _evaluate("arith.floordivsi", a, b, width) == a // b
                    toward_zero = abs(a) // abs(b) * (1 if (a >= 0) == (b >= 0) else -1)
                    assert _evaluate("arith.divsi", a, b, width) == rt._ist_signed(toward_zero, width)
                if 0 <= b < width:
                    assert _evaluate("arith.shrsi", a, b, width) == a >> b
    for op, a, b in (("arith.floordivsi", 7, 0), ("arith.divsi", -(1 << 31), -1), ("arith.shrsi", 5, 32)):
        with pytest.raises(rt._IstRefusal, match="poison"):
            _evaluate(op, a, b, 32)
    with pytest.raises(rt._IstRefusal, match="no-wrap"):
        _evaluate("arith.addi", (1 << 31) - 1, 1, 32, "checked")


def _grid_agrees(start: int, stop: int) -> None:
    """The clamp and the compare-and-select give the same grid index for every non-NaN pattern."""
    from merlin.llvmlower import passes_quant_int as PQI

    step = np.float32(PQI._IEXP_S)
    lo = np.float32(-PQI._IEXP_QMAX)
    patterns = np.arange(start, stop, dtype=np.uint64).astype(np.uint32)
    d = patterns.view(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        q = np.rint(d / step)  # f32 division, round half to even: arith.divf then math.roundeven
    clamped = np.minimum(np.maximum(q, lo), np.float32(0.0))
    selected = np.where(q >= lo, q, lo)
    defined = ~np.isnan(d)
    assert np.array_equal(clamped[defined].astype(np.int32), selected[defined].astype(np.int32)), hex(start)
    assert bool(((selected >= lo) & (selected <= 0)).all())  # NaN included: always inside the table


def test_select_and_clamp_agree_on_a_sample_of_non_positive_float32():
    # +0 and the negative patterns 0x80000000..0xFF800000 (-0 through -inf), and NaN, in strides.
    _grid_agrees(0, 1)
    for start in range(0x80000000, 0xFF800001, 0x00800000):
        _grid_agrees(start, min(start + 4096, 0xFF800001))
    _grid_agrees(0xFF800000, 0xFF800010)


@pytest.mark.slow
def test_select_and_clamp_agree_on_every_non_positive_float32():
    for start in range(0x80000000, 0xFF800001, 1 << 24):
        _grid_agrees(start, min(start + (1 << 24), 0xFF800001))


def test_a_requested_rewrite_must_report_once(tmp_path):
    with pytest.raises(ValueError, match="0 times"):
        IST.require_report("OK something else\n", tmp_path)
    line = IST.REPORT_PREFIX + json.dumps({"softmax": 0, "refused": []})
    with pytest.raises(ValueError, match="2 times"):
        IST.require_report(f"{line}\n{line}\n", tmp_path)
    assert IST.require_report(f"noise\n{line}\n", tmp_path) == {"softmax": 0, "refused": []}
    assert json.loads((tmp_path / "int_softmax_table_report.json").read_text())["softmax"] == 0


@pytest.fixture(scope="module")
def runtime_library(tmp_path_factory) -> Path:
    """The runtime's own bf16 helpers, built for the host, so bf16 programs execute with the bits the
    target's runtime produces."""
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        pytest.skip("no host C compiler to build the runtime's bf16 helpers")
    out = tmp_path_factory.mktemp("rt") / "libmerlin_runtime.so"
    source = merlin_dir() / "runtime" / "abi" / "mlir_runtime.c"
    subprocess.run([compiler, "-O1", "-shared", "-fPIC", "-o", str(out), str(source)], check=True)
    return out


@m2m
@pytest.mark.parametrize("section", ["table", "softmax", "attention", "refusal", "mutant"])
def test_the_rewritten_module_computes_the_same_bits(section, runtime_library):
    run = subprocess.run(
        [
            str(toolchain.m2m_python()),
            str(_EXACTNESS),
            str(repo_root() / "src" / "merlin" / "llvmlower" / "_int_softmax_table_rt.py"),
            str(_DATA),
            str(_TARGETGEN),
            str(runtime_library),
            section,
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, "TMPDIR": os.environ.get("TMPDIR", tempfile.gettempdir())},
    )
    assert run.returncode == 0 and f"ok {section}" in run.stdout, (run.stdout[-2000:], run.stderr[-3000:])


@m2m
def test_selecting_it_by_name_reaches_the_lowering(tmp_path, monkeypatch):
    if not toolchain.clang().is_file():
        pytest.skip("clang is absent")
    from merlin.llvmlower.lower import lower_model

    source = (_DATA / "softmax_wide.mlir").read_text(encoding="utf-8")
    monkeypatch.delenv(OP.ENV, raising=False)
    plain = lower_model(source, tmp_path / "plain", targets=(), textual=True)
    assert not (tmp_path / "plain" / "int_softmax_table_report.json").exists()
    with OP.applied(OP.Selection.of(["int-softmax-table"])):
        chosen = lower_model(source, tmp_path / "chosen", targets=(), textual=True)
    report = json.loads((tmp_path / "chosen" / "int_softmax_table_report.json").read_text())
    assert (report["softmax"], report["int32_sums"]) == (1, 1), report
    assert Path(plain.ll_path).read_bytes() != Path(chosen.ll_path).read_bytes()
