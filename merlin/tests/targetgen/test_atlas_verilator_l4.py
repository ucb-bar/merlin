"""The atlas external_backend target gains an ADDITIVE L4 tier: a program-driven Verilator sim of the
RTL top (bare AtlasCore over TileLink), the first truly RTL-CERTIFIED atlas oracle. arc cosim (L3)
stays the RTL-derived functional gold; L4 runs the elaborated Verilog and must agree bit-exact.

Two checks:
  * routing (hermetic, no venv): L4 is present iff the target registers a vsim (MERLIN_EXT_<TARGET>_VSIM),
    and a target with no vsim gets no L4 — the tier is target-agnostic (no atlas literal in the wiring).
  * cross-tier equivalence (gated on the sim + arc + model venv being present): the SAME self-contained
    program run through the merlin L3 (arc) and L4 (verilator) adapters returns bit-exact identical outputs.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import ext_path
from merlin.targetgen import program_oracle as PO
from merlin.targetgen.capsule_runner import oracle_adapters


def _vsim_dir() -> Path | None:
    try:
        d = ext_path("atlas_vsim")
    except KeyError:
        return None
    return d if (d / "verilator_run.py").is_file() and (d / "vobj_dir" / "VAtlasCore").is_file() else None


def test_verilator_adapter_is_target_agnostic():
    # gemmini registers no vsim -> no L4 (the gate is MERLIN_EXT_<TARGET>_VSIM, not a target-name branch).
    assert PO.program_verilator_adapter("gemmini", model_ext="npu_model") is None


def test_atlas_exposes_l4_when_a_vsim_is_registered():
    ad = oracle_adapters("atlas")
    assert {"L2", "L3"} <= set(ad)                       # the functional + arc-cosim tiers always route
    if _vsim_dir() is not None:
        assert "L4" in ad and callable(ad["L4"])         # additive RTL tier, present when the sim is built
        assert ad["L4"].__module__ == "merlin.targetgen.program_oracle"
    else:
        assert "L4" not in ad                            # honestly absent when no sim is registered


@pytest.mark.skipif(_vsim_dir() is None, reason="atlas vsim (MERLIN_EXT_ATLAS_VSIM + built VAtlasCore) absent")
def test_verilator_l4_matches_arc_l3_bit_exact():
    """A self-contained program through both merlin adapters — arc cosim (L3) and Verilator (L4) — must
    produce bit-exact identical outputs (the RTL sim runs the elaborated Verilog the arc model was
    derived from). Skips cleanly if the arc cosim / model venv is not available on this machine."""
    vsim = _vsim_dir()
    prog = "MatmulProgram"
    try:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            l3 = PO.run_program_oracle("atlas", model_ext="npu_model", program=prog,
                                       workdir=Path(d1), timeout=500)
            l4 = PO.run_program_verilator_oracle("atlas", model_ext="npu_model", vsim_dir=vsim,
                                                 program=prog, workdir=Path(d2), timeout=500)
    except PO.OracleUnavailable as e:
        pytest.skip(f"program oracle infra unavailable: {e}")

    a3 = np.array(next(iter(l3["outputs"].values())))
    a4 = np.array(next(iter(l4["outputs"].values())))
    assert a3.shape == a4.shape, f"shape mismatch L3 {a3.shape} vs L4 {a4.shape}"
    assert np.array_equal(a3, a4), "verilator L4 diverged from arc L3 (should be bit-exact on the same RTL)"
    assert l4["oracle"] == "atlas-verilator-rtl"
