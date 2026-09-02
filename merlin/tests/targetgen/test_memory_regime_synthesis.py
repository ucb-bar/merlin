"""The memory-regime axis, from requirement to synthesized capsule.

The corpus and reality did not overlap on the one axis that decides tiling: 90.1% of 1829 real
contraction regions spill the operand store, while 100% of the public capsules fit it twice over. A
corpus like that cannot detect a memory-mapping failure of any kind, and on a hardware-interlocked
target nothing else reports one either -- the schedule is correct whatever it chooses.

These pin the two properties that make the fix trustworthy: the synthesizer finds extents with the SAME
sizing the coverage gate measures with (so a capsule cannot be emitted for one regime and then be
classified into another), and a regime it cannot reach is REPORTED rather than dropped.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import corpus_synth as CS
from merlin.targetgen import memory_regime as MR

TARGET = "gemmini"


def _store_or_skip():
    store, capacity = MR.operand_store(TARGET)
    if store is None or not capacity:
        pytest.skip(f"{TARGET} declares no derivable operand store in this checkout")
    return store, capacity


def test_derived_extents_land_in_the_regime_they_were_derived_for():
    """The round trip that matters: size the extents, then classify them the way the gate does."""
    store, capacity = _store_or_skip()
    for regime in (MR.FITS_DOUBLE, MR.FITS_SINGLE, MR.SPILLS):
        ext = MR.extents_for_regime(TARGET, regime, tile_dim=16, dtype="i8")
        assert ext is not None, f"no capsule shape reaches {regime} on {TARGET}"
        rows = ext["rows"]
        assert MR.classify(rows, rows, capacity) == regime, (
            f"{regime}: derived {ext['M']}x{ext['K']}x{ext['N']} = {rows} rows, which the gate "
            f"classifies as {MR.classify(rows, rows, capacity)}")


def test_spills_actually_exceeds_capacity_and_the_fits_do_not():
    """A `spills` capsule that fits is the failure this axis exists to prevent."""
    _, capacity = _store_or_skip()
    spills = MR.extents_for_regime(TARGET, MR.SPILLS, tile_dim=16, dtype="i8")
    double = MR.extents_for_regime(TARGET, MR.FITS_DOUBLE, tile_dim=16, dtype="i8")
    assert spills["rows"] > capacity
    assert double["rows"] * 2 <= capacity


def test_fits_on_reuse_is_unreachable_from_a_capsules_inputs_and_says_so():
    """Not a gap in the search: a capsule's declared inputs are ALL live at once, so peak-live and
    total coincide and the regime that separates them cannot arise. Answering `None` is correct; the
    danger would be answering with a capsule that is really something else."""
    _store_or_skip()
    assert MR.extents_for_regime(TARGET, MR.FITS_ON_REUSE, tile_dim=16, dtype="i8") is None


def test_an_unsizeable_target_returns_none_rather_than_a_default():
    assert MR.extents_for_regime("no_such_target_anywhere", MR.SPILLS, tile_dim=16) is None
    assert MR.extents_for_regime(TARGET, MR.SPILLS, tile_dim=0, dtype="i8") is None


def _spec_with_regimes():
    store, capacity = _store_or_skip()
    regimes = [MR.FITS_DOUBLE, MR.FITS_SINGLE, MR.SPILLS]
    return {
        "target": TARGET,
        "cells": [{"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8",
                   "alignment": "aligned", "basis": "observed"}],
        "boundaries": {"extent_probes": [{"boundary": "tile_edge", "edge": 16,
                                          "points": [15, 16, 17, 32]}]},
        "memory_mapping": {
            "required": {r: ["some_capture"] for r in regimes},
            "regime_dtype": "i8",
            "regime_extents": MR.required_regime_extents(TARGET, regimes, tile_dim=16, dtype="i8"),
        },
    }


def test_synthesis_emits_one_capsule_per_required_regime():
    out = CS.synthesize(_spec_with_regimes())
    names = {e["name"] for e in out["capsules"]}
    for regime in (MR.FITS_DOUBLE, MR.FITS_SINGLE, MR.SPILLS):
        assert f"{CS.SYNTH_PREFIX}_regime_{regime}" in names, f"no capsule synthesized for {regime}"
    assert out["provenance"]["memory_regimes_unreachable"] == []


def test_a_regime_the_target_cannot_reach_is_reported_not_dropped():
    """A silently absent regime reads downstream as a covered one -- the failure mode the whole
    fail-closed discipline in this module exists to prevent."""
    spec = _spec_with_regimes()
    spec["memory_mapping"]["required"]["fits_on_reuse"] = ["some_capture"]
    spec["memory_mapping"]["regime_extents"]["fits_on_reuse"] = None
    out = CS.synthesize(spec)
    assert "fits_on_reuse" in out["provenance"]["memory_regimes_unreachable"]
    assert f"{CS.SYNTH_PREFIX}_regime_fits_on_reuse" not in {e["name"] for e in out["capsules"]}


def test_a_synthesized_regime_capsule_carries_its_residency_as_provenance():
    """The number is the claim. A reviewer must be able to see why this shape is that regime."""
    out = CS.synthesize(_spec_with_regimes())
    spills = next(e for e in out["capsules"] if e["name"].endswith(f"_regime_{MR.SPILLS}"))
    assert "operand-store rows" in spills["source_reference"]
    assert spills["source_role"] == CS.SOURCE_ROLE
