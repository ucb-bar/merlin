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
from merlin.targetgen.rtl import mlc_bridge as MB
from merlin.targetgen.rtl.facts import load_facts

_FC = RR.find_filecheck()
pytestmark = pytest.mark.skipif(_FC is None, reason="FileCheck binary not available")
_FACTS = load_facts("gemmini")
# The behavioural role probe needs a live arc model (+ the mlc venv). Where it is present, the derived
# roles are an ALWAYS-ON cross-check of the hand ABI semantic_class; where it is absent (CI without the
# arc), we skip those assertions rather than fail — the honest-empty path is exercised by the SIMT test.
_ARC = MB.arc_available("gemmini")


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


def test_compiled_checks_are_memoized():
    """compiled_checks is a pure function of (capsule name, facts sha) — repeated calls hit the cache."""
    cap = _matmul_capsule()
    RR._COMPILED_CACHE.clear()
    a = RR.compiled_checks(_FACTS, cap)
    b = RR.compiled_checks(_FACTS, cap)
    assert a is b                                        # same object -> served from cache
    assert (cap.get("name"), RR._facts_sha(_FACTS), "gemmini") in RR._COMPILED_CACHE


@pytest.mark.skipif(not _ARC, reason="gemmini arc/mlc unavailable — cannot derive behavioural roles")
def test_provenance_flags_derived_vs_handpicked():
    """Every check family is tagged with its source + whether it is genuinely DERIVED. For gemmini the
    legality/ABI/coverage families are grounded in mlc facts, and — now that ``semantic_roles`` regenerates
    the behavioural effect-probe cache on demand from the live arc — the opcode->role family is DERIVED too,
    and those derived roles VERIFY the hand ABI semantic_class (no disagreements). This is how we audit 'did
    we hand-pick this?' — every family names its source and the role family is cross-checked, not declared."""
    cap = _matmul_capsule()
    prov = CC.compile_checks(_FACTS, cap, target="gemmini")["provenance"]
    assert prov["isa_legality"]["derived"] is True
    assert prov["abi_encoding"]["derived"] is True
    assert prov["tile_coverage"]["derived"] is True
    assert prov["semantic_roles"]["derived"] is True         # regenerated behavioural probe -> derived
    assert prov["semantic_roles"]["n_roles"] >= 9
    assert MB.crosscheck_semantic_class("gemmini") == []     # derived roles verify the hand ABI labels


@pytest.mark.skipif(not _ARC, reason="gemmini arc/mlc unavailable — cannot derive behavioural roles")
def test_derived_roles_crosscheck_hand_semantic_class():
    """ALWAYS-ON gate: the behaviourally-derived coarse roles VERIFY gemmini's hand ABI semantic_class 9/9.
    An empty disagreement list means every hand-declared funct label (MVIN/MVOUT/COMPUTE*/PRELOAD/CONFIG*/
    LOOP_*/FLUSH) matches the role the RTL behaviour actually exhibits — a mislabel or RTL drift would
    surface here instead of being silently trusted."""
    assert MB.crosscheck_semantic_class("gemmini") == []
    roles = MB.semantic_roles("gemmini")
    assert roles["derived"] is True and len(roles["roles"]) >= 9


def test_non_rocc_target_drops_rocc_shaped_checks():
    """A non-RoCC target drops the RoCC-shaped TRACE check rather than emitting meaningless ones — emit
    only what is groundable for the endpoint, never guess. The RoCC ``trace`` family is absent; there is
    no dialect-MLIR check family at all (op mnemonics are un-derivable per generated OOT dialect)."""
    simt = {"facts": {"interfaces": [{"name": "warp_dispatch"}], "arrays": [], "memories": []}}
    cc = CC.compile_checks(simt, _matmul_capsule(), target="radiance")
    assert "dialect" not in cc and cc["trace"] is None
    assert cc["provenance"]["isa_legality"]["derived"] is False
    assert "no discovered_roles cache" in cc["provenance"]["semantic_roles"]["reason"]
