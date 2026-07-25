"""B1: decoder-derived ISA extraction via the mlc bridge, and its reconciliation with the header parse.

The core value: the legal RoCC funct set must come from the DECODER (the silicon), not the ISA header
(provably wrong). These tests cover the pure reconciliation logic (no mlc needed), the honest fallback
when the HW dialect cannot be parsed, and — when mlc is available — the bridge resolve/guard.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.rtl import mlc_bridge as B
from merlin.targetgen.rtl import circt_introspect as C
from merlin.common.paths import repo_root

_MLC_OK = B.mlc_available()[0]


# --------------------------------------------------------------------------- pure reconciliation
def test_reconcile_prefers_decoder_and_records_discrepancy():
    header = {"name": "funct_decode_table", "legal_funct": list(range(26)),  # [0..25], header-parsed
              "names": {"25": "LOOP_WS_CONFIG_SPAD_C", "3": "COMPUTE"}, "method": "scala_header_parse"}
    decoder = {"name": "funct_decode_table", "legal_funct": [0, 1, 2, 3, 126],
               "method": "decoder_icmp_fanout(mlc)", "evidence": "decoder set"}
    out = C._reconcile_funct(decoder, header)
    assert out["method"].startswith("decoder_icmp_fanout")          # decoder wins (it is the silicon)
    assert out["legal_funct"] == [0, 1, 2, 3, 126]
    assert 25 in out["header_only_functs"]                          # phantom: header claims, silicon doesn't
    assert 126 in out["decoder_only_functs"]                        # missing: silicon decodes, header omits
    assert out["names"]["3"] == "COMPUTE"                           # names borrowed from the header
    assert out["names"]["126"] == "?"                               # decoded-but-unnamed surfaced honestly


def test_reconcile_falls_back_to_header_when_no_decoder():
    header = {"name": "funct_decode_table", "legal_funct": [0, 1, 2]}
    out = C._reconcile_funct(None, header)
    assert out is header and out["method"] == "scala_header_parse"
    assert C._reconcile_funct(None, None) is None


# --------------------------------------------------------------------------- honest fallback
def test_decoder_extraction_returns_none_when_no_core_hw(monkeypatch):
    """With NO version-matched core HW dialect available for the target, the decoder path returns None —
    a clean fallback to the header parse, never a crash or a fake pass."""
    monkeypatch.setattr(B, "core_hw_mlir", lambda target: None)  # simulate: no prebuilt core dialect
    assert C.extract_funct_table_via_decoder("gemmini") is None


# --------------------------------------------------------------------------- B2 end-to-end (mlc-gated)
@pytest.mark.skipif(not _MLC_OK or B.core_hw_mlir("gemmini") is None,
                    reason="mlc / prebuilt core HW dialect not available for the example target")
def test_decoder_derived_opcode_set_fixes_the_header_set():
    """The decisive B1/B2 result on the gemmini example target: the decoder-derived legal opcode set
    (from mlc's version-matched core HW dialect) drops the phantom header code 25 and adds the real
    decoded code 126 the header omits. Same agnostic path works for any target."""
    funct = C.extract_funct_table_via_decoder("gemmini")
    assert funct is not None and funct["method"].startswith("decoder_icmp_fanout")
    legal = set(funct["legal_funct"])
    assert 25 not in legal and 126 in legal          # phantom dropped, real decoded code added
    header = C.extract_funct_table(C.isa_scala_path("gemmini").read_text(errors="replace"))
    reconciled = C._reconcile_funct(funct, header)
    assert reconciled["header_only_functs"] == [25] and reconciled["decoder_only_functs"] == [126]


# --------------------------------------------------------------------------- bridge guard (mlc-gated)
@pytest.mark.skipif(not _MLC_OK, reason="mlc not available (MERLIN_MLC_DIR / circt-opt)")
def test_bridge_resolves_mlc_and_circt_opt():
    assert B.mlc_dir() is not None and (B.mlc_dir() / "mlc").is_dir()
    assert B.circt_opt_bin() is not None and B.circt_opt_bin().exists()


def test_require_mlc_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(B, "mlc_dir", lambda: None)
    with pytest.raises(RuntimeError, match="mlc unavailable"):
        B.require_mlc()


# --------------------------------------------------------------------------- arc oracle (B2, mlc-gated)
@pytest.mark.skipif(not _MLC_OK or not B.arc_available("gemmini"),
                    reason="mlc / prebuilt arc model not available for the example target")
def test_arc_core_loads_rtl_model_for_target():
    """The compile-from-RTL oracle primitive, TARGET-AGNOSTIC: arc_core(target) loads mlc's arcilator
    model for any target (gemmini here as the example argument) and exposes its state by NAME — the
    basis for internal-state probes spike cannot give, without verilator."""
    core = B.arc_core("gemmini")
    assert core.num_state_bytes > 0                 # a real RTL model loaded from mlc's arc .so
    assert core.manifest_port_names()               # named input/output ports discovered from the RTL


# --------------------------------------------------------- FINE behavioural role classification (pure)
# The feature-vector -> fine ISA name mapping is a pure function of MEASURED behaviour (no arc, no mlc):
# an RTL change that altered a funct's behaviour would change the derived name and trip the cross-check.

def test_classify_fine_role_compute_cluster_split_by_accumulator_and_weight_flip():
    # writes the accumulator AND flips the weight double-buffer => uses the freshly preloaded weights
    assert B._classify_fine_role("compute", {"writes_accumulator": True, "weight_flip": True}) \
        == ("COMPUTE_PRELOADED", True)
    # writes the accumulator, NO flip => reuses the stationary weights
    assert B._classify_fine_role("compute", {"writes_accumulator": True, "weight_flip": False}) \
        == ("COMPUTE_ACCUMULATE", True)
    # loads weights, commits no MAC => PRELOAD
    assert B._classify_fine_role("compute", {"writes_accumulator": False, "loads_weights": True}) \
        == ("PRELOAD", True)


def test_classify_fine_role_load_cluster_split_by_config_id():
    assert B._classify_fine_role("load", {"config_id": 0}) == ("MVIN", True)
    assert B._classify_fine_role("load", {"config_id": 1}) == ("MVIN2", True)
    assert B._classify_fine_role("load", {"config_id": 2}) == ("MVIN3", True)
    # no config-id resolved => honestly unnamed, not guessed
    assert B._classify_fine_role("load", {"config_id": None}) == (None, False)


def test_classify_fine_role_loop_macros_split_by_fsm_and_take_precedence():
    # loop FSM tagging positively names the loop macros, over the coarse compute/config bucket
    assert B._classify_fine_role("config", {"drives_matmul_fsm": True}) == ("LOOP_WS", True)
    assert B._classify_fine_role("compute", {"drives_conv_fsm": True}) == ("LOOP_CONV", True)


def test_classify_fine_role_coarse_grounded_fallbacks_are_flagged_not_fine_derived():
    assert B._classify_fine_role("store", {}) == ("MVOUT", False)
    assert B._classify_fine_role("barrier", {}) == ("FLUSH", False)
    assert B._classify_fine_role("config", {}) == ("CONFIG", False)


@pytest.mark.skipif(not _MLC_OK or not B.arc_available("gemmini"),
                    reason="mlc / prebuilt arc model not available for the example target")
def test_fine_roles_behaviourally_reproduce_the_hand_semantic_class():
    """END-TO-END on the arc: derive the FINE roles, then assert every hand-declared semantic_class NAME is
    reproduced by the RTL behaviour (fine cross-check empty). This is what makes the manifest names provably
    RTL-behaviour-grounded — COMPUTE_PRELOADED vs _ACCUMULATE, MVIN vs MVIN2/3, LOOP_WS vs LOOP_CONV are all
    distinguished by measured effect, not by the header alias."""
    rec = B.derive_fine_roles("gemmini")
    fine = rec["fine_roles"]
    # the fine cross-check must find NO disagreement (each declared name == its derived fine behaviour)
    assert B.crosscheck_semantic_class_fine("gemmini") == []
    # the four collapsed distinctions are behaviourally fine-derived (not coarse fallbacks)
    for code in ("2", "4", "5", "6", "8", "14", "15"):
        assert rec["fine_derived"][code] is True
    assert fine["4"] == "COMPUTE_PRELOADED" and fine["5"] == "COMPUTE_ACCUMULATE" and fine["6"] == "PRELOAD"
    assert fine["2"] == "MVIN" and fine["1"] == "MVIN2" and fine["14"] == "MVIN3"
    assert fine["8"] == "LOOP_WS" and fine["15"] == "LOOP_CONV"
    # and the fine_roles() cache reader round-trips the derived map
    assert B.fine_roles("gemmini")["roles"][4] == "COMPUTE_PRELOADED"
