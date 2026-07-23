"""B3: the RTL-derived FileCheck suite — checks that expose VERILATOR-level truth WITHOUT running
verilator, so a bad backend is rejected instantly (fast convergence + concrete feedback) instead of
after a multi-minute RTL run.

Each test below is an example of information spike (the functional ISA sim) CANNOT give but the RTL
decoder/structure CAN — compiled into a FileCheck assertion over the decoded RoCC trace. The verdict
these produce is exactly what `rtl_check_runner` uses to skip the spike/verilator oracle on a hard reject.

The flagship is `test_phantom_funct_25_rejected_without_verilator`: a trace using funct 25 is accepted
by spike functionally, but the RTL decoder (module ReservationStation) never matches funct 25, so on
real hardware/verilator it is a no-op. Our decoder-derived legal set catches it statically.
"""
from __future__ import annotations

import json
import math

import pytest
import yaml

from merlin.targetgen import rtl_check_compiler as CC, rtl_check_runner as RR, rtl_checks as RC
from merlin.targetgen.rtl.facts import rtl_facts_path

_FC = RR.find_filecheck()
pytestmark = pytest.mark.skipif(_FC is None, reason="FileCheck binary not available")
_FACTS = json.loads(rtl_facts_path("gemmini").read_text())


def _matmul_capsule(min_tiles: int = 1):
    """A real matmul capsule from the corpus with exactly/at-least the requested tile count."""
    mr, mc = RC._mesh(CC._facts_to_rc(_FACTS))
    for name, p in sorted(RR._capsule_index().items()):
        cap = yaml.safe_load(p.read_text())
        shp = RC._declared_output_shape(cap)
        if RC._declared_op(cap) in ("matmul", "matmul_resident") and shp:
            if math.ceil(shp[0] / mr) * math.ceil(shp[1] / mc) >= min_tiles:
                return cap
    return None


def _trace(seq, abi=None):
    return {"abi": abi or {"custom_opcode": "0x7b", "funct3": "0x3"},
            "instructions": [{"index": i, "class": c, "funct": f} for i, (c, f) in enumerate(seq)]}


def _good_matmul_trace(cap):
    """A legal single/multi-tile matmul RoCC sequence matching the capsule's declared tile count."""
    mr, mc = RC._mesh(CC._facts_to_rc(_FACTS))
    M, N = RC._declared_output_shape(cap)
    tiles = math.ceil(M / mr) * math.ceil(N / mc)
    seq = [("CONFIG_EX", 0), ("CONFIG_LD", 0)]
    for _ in range(tiles):
        seq += [("MVIN", 2), ("PRELOAD", 6), ("COMPUTE_PRELOADED", 4), ("MVOUT", 3)]
    return _trace(seq)


def _fc(cap, trace):
    return RR.run_filecheck(_FC, CC.compile_trace_checks(_FACTS, cap), RR.render_trace(trace, _FACTS), "TRACE")


def test_good_matmul_trace_passes():
    cap = _matmul_capsule()
    assert cap is not None
    ok, diag = _fc(cap, _good_matmul_trace(cap))
    assert ok, diag


def test_phantom_funct_25_rejected_without_verilator():
    """FLAGSHIP: funct 25 (LOOP_WS_CONFIG_SPAD_C) is in the ISA HEADER but the RTL decoder never matches
    it. Spike accepts it functionally; the RTL-derived check rejects it — no verilator run needed."""
    cap = _matmul_capsule()
    tr = _good_matmul_trace(cap)
    tr["instructions"].append({"index": 999, "class": "UNKNOWN", "funct": 25})
    rendered = RR.render_trace(tr, _FACTS)
    assert "ILLEGAL_FUNCT_COUNT 1" in rendered      # the decoder-derived set flags 25
    ok, _ = _fc(cap, tr)
    assert not ok                                    # rejected statically


def test_real_funct_126_is_legal():
    """funct 126 (COUNTER_OP) is decoded by the RTL but OMITTED from the header. A backend using it must
    NOT be flagged — the decoder-derived set includes it (the header-based check would wrongly reject)."""
    cap = _matmul_capsule()
    tr = _good_matmul_trace(cap)
    tr["instructions"].append({"index": 998, "class": "COUNTER_OP", "funct": 126})
    assert "ILLEGAL_FUNCT_COUNT 0" in RR.render_trace(tr, _FACTS)
    ok, diag = _fc(cap, tr)
    assert ok, diag


def test_wrong_tile_count_rejected_without_verilator():
    """MVOUT_COUNT must equal ceil(M/DIM)*ceil(N/DIM) with the RTL's real mesh DIM. A wrong tile count is
    wrong hardware coverage; spike may still emit a plausible output, but the RTL-derived count rejects
    it statically."""
    cap = _matmul_capsule()
    tr = _good_matmul_trace(cap)
    tr["instructions"].append({"index": 997, "class": "MVOUT", "funct": 3})   # one extra tile store
    ok, _ = _fc(cap, tr)
    assert not ok


def _op_form_mlir(cap):
    """A minimal op-form gemmini-dialect MLIR carrying the tokens the compiled DIALECT checks look for
    (res_pack -> matmul -> commit + the declared output_dtype)."""
    dt = ((cap.get("operation") or {}).get("attributes") or {}).get("output_dtype")
    tail = f' {{output_dtype = "{dt}"}}' if dt else ""
    return ("%0 = gemmini.res_pack %w : tensor\n"
            "%1 = gemmini.matmul %a, %0 : tensor\n"
            f"%2 = gemmini.commit %1 : tensor{tail}\n")


@pytest.mark.parametrize("good", [True, False])
def test_combined_single_pass_matches_separate_runs(good):
    """The fast path (one FileCheck pass over concatenated dialect+trace input with both prefixes) must
    yield the SAME pass/fail as running the two checks separately — the disjoint check vocabularies mean
    a combined --check-prefixes run cannot cross-match. Covered for a passing and a failing trace."""
    cap = _matmul_capsule()
    assert cap is not None and RR._is_op_form(_op_form_mlir(cap))
    trace = _good_matmul_trace(cap)
    if not good:
        trace["instructions"].append({"index": 997, "class": "MVOUT", "funct": 3})  # wrong tile count
    mlir = _op_form_mlir(cap)
    ttxt = RR.render_trace(trace, _FACTS)
    dchecks, tchecks = CC.compile_dialect_checks(cap), CC.compile_trace_checks(_FACTS, cap)

    combined, _ = RR.run_filecheck(
        _FC, dchecks + "\n" + tchecks, f"{mlir}\n; ---rtlcheck-trace-region---\n{ttxt}",
        ["DIALECT", "TRACE"])
    okt, _ = RR.run_filecheck(_FC, tchecks, ttxt, "TRACE")
    okd, _ = RR.run_filecheck(_FC, dchecks, mlir, "DIALECT")
    assert combined == (okt and okd)
    assert okt is good                                   # trace bears the verdict; wrong tiles => fail


def test_compiled_checks_are_memoized():
    """compiled_checks is a pure function of (capsule name, facts sha) — repeated calls hit the cache."""
    cap = _matmul_capsule()
    RR._COMPILED_CACHE.clear()
    a = RR.compiled_checks(_FACTS, cap)
    b = RR.compiled_checks(_FACTS, cap)
    assert a is b                                        # same object -> served from cache
    assert (cap.get("name"), RR._facts_sha(_FACTS)) in RR._COMPILED_CACHE


def test_dialect_dataflow_order_mlir_lit_fixture():
    """A literal `// RUN: FileCheck` .mlir test: the RTL-enforced res_pack->matmul->commit dataflow
    structural check over gemmini-dialect MLIR (spike is lenient about this ordering)."""
    from pathlib import Path
    import subprocess
    from merlin.common.paths import repo_root
    mlir = repo_root() / "merlin/tests/targetgen/data/rtl_filecheck/matmul_dialect_good.mlir"
    p = subprocess.run([_FC, "--check-prefix=DIALECT", str(mlir)],
                       stdin=open(mlir), capture_output=True, text=True)
    assert p.returncode == 0, (p.stderr or p.stdout)
