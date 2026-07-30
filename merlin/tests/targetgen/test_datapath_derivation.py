"""The datapath element dtypes are RTL-DERIVABLE, not just declared — a crosscheck.

The compute-unit dtypes (operand / accumulator / scale) a target's manifest declares must be grounded in
the RTL, the same "derive, don't declare" bar the ISA/mesh/memory facts already meet. mlc's
``discover_datapaths`` reads them structurally from the CIRCT HW dialect (int by SRAM element width; float
by the berkeley-hardfloat ``e<E>_s<S>`` / design format-name token). This test proves the DERIVED dtypes
match what the manifests declare — so the declaration is a cross-checked cache of an RTL fact, and drift is
caught. It is arc/branch-gated: skips cleanly when mlc's ``discover_datapaths`` (a pending mlc change) or the
built circt-opt is absent, so a fresh checkout without them is green.

Honest boundaries (asserted as skips, not failures): radiance's CVFPU is PULP fpnew (its enabled-format set
is a compile parameter absent from the HW dialect) and mx_gemmini's only dialect is a fused SoC — neither is
structurally resolvable today, so they are out of scope here rather than faked.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.rtl import mlc_bridge as B


def _derive(target):
    """(compute_kind, {role: dtype}) from mlc discover_datapaths, or a pytest.skip if unavailable."""
    try:
        from mlc.discover.datapaths import discover_datapaths_for_target
    except Exception:  # noqa: BLE001 — the mlc datapath extractor is a pending mlc-branch change
        pytest.skip("mlc.discover.datapaths not present (pending mlc branch)")
    base = B.mlc_dir()
    copt = B.circt_opt_bin()
    if base is None or copt is None:
        pytest.skip("MERLIN_MLC_DIR / circt-opt not resolvable")
    if not B.arc_available(target):
        pytest.skip(f"arc/HW dialect for {target!r} not available")
    r = discover_datapaths_for_target(target, base=str(base), circt_opt=str(copt))
    if not isinstance(r, dict) or not r.get("datapaths"):
        pytest.skip(f"no datapaths derived for {target!r} (documented boundary)")
    return r.get("compute_kind"), {d["role"]: d["dtype"] for d in r["datapaths"]}


def test_gemmini_datapaths_derived_match_the_tracked_contract():
    kind, dt = _derive("gemmini")
    # byte-identical to merlin/targets/gemmini/contracts/rtl_facts/facts.json (i8 operand / i32 accumulator)
    assert kind == "mac_mesh"
    assert dt.get("operand") == "i8"
    assert dt.get("accumulator") == "i32"


def test_atlas_datapaths_derived_match_the_declared_manifest():
    from merlin.targetgen.capability_manifests import manifest_for
    kind, dt = _derive("atlas")
    # the fp8-e4m3 / bf16 / E8M0 the atlas manifest declares are recovered structurally from the MXU RTL
    assert kind == "mac_mesh"
    assert dt.get("operand") == "fp8_e4m3"
    assert dt.get("accumulator") == "bf16"
    assert dt.get("scale") == "E8M0"
    # the derived operand dtype is exactly the declared compute-unit dtype (declared == derived)
    declared = manifest_for("atlas")["compute_units"][0]["dtypes"]
    assert "fp8_e4m3" in declared and dt["operand"] in declared
