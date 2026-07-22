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
def test_decoder_extraction_returns_none_on_unparseable_hw(tmp_path):
    """A HW dialect with a firtool/circt-opt version skew (or missing mlc) yields None — a clean
    fallback to the header parse, never a crash or a fake pass."""
    bad = tmp_path / "bad.hw.mlir"
    bad.write_text("this is not valid mlir\n")
    assert C.extract_funct_table_via_decoder(bad) is None


# --------------------------------------------------------------------------- bridge guard (mlc-gated)
@pytest.mark.skipif(not _MLC_OK, reason="mlc not available (MERLIN_MLC_DIR / circt-opt)")
def test_bridge_resolves_mlc_and_circt_opt():
    assert B.mlc_dir() is not None and (B.mlc_dir() / "mlc").is_dir()
    assert B.circt_opt_bin() is not None and B.circt_opt_bin().exists()


def test_require_mlc_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(B, "mlc_dir", lambda: None)
    with pytest.raises(RuntimeError, match="mlc unavailable"):
        B.require_mlc()
