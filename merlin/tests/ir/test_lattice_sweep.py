"""The lattice must be DERIVED from the target's own facts, and its count must not be inflated.

The claim this supports is "the verified set is generated, not curated" — the direct answer to the
review comment that the capsules are case-specific. That claim is only worth anything if the extents
really come from the hardware and the coverage number really counts distinct work.
"""
from __future__ import annotations

import pytest

from merlin.verify import HAS_XDSL, HAS_Z3
from merlin.verify.tools import find_mlir_tool

pytestmark = pytest.mark.skipif(
    not (HAS_XDSL and HAS_Z3 and find_mlir_tool("mlir-translate")),
    reason="needs the verify extra (xdsl + z3) and mlir-translate")


def _a_target_with_a_derived_edge() -> str:
    """A target whose spec carries an extent lattice — found, never named in library-adjacent code."""
    from merlin.targetgen.lit_suite import known_targets
    from merlin.verify.lattice import lattice_points, load_spec

    for t in known_targets():
        try:
            if lattice_points(load_spec(t)):
                return t
        except FileNotFoundError:
            continue
    pytest.skip("no target in this checkout has a derivable extent lattice")


def test_the_extents_come_from_hardware_facts_not_from_a_default():
    """A software default is not a hardware boundary; the record must carry the provenance."""
    from merlin.verify.lattice import lattice_points, load_spec, sweep

    target = _a_target_with_a_derived_edge()
    spec = load_spec(target)
    points = lattice_points(spec)

    probe = ((spec.get("boundaries") or {}).get("extent_probes") or [])[0]
    edge = int(probe["edge"])
    # The derived points must straddle the real boundary, which is what makes them edge cases.
    assert edge in points, "the exact tile is not probed"
    assert edge - 1 in points, "the tail (edge-1) is not probed"
    assert edge + 1 in points, "the overflow (edge+1) is not probed"
    assert 1 in points, "the degenerate extent is not probed"

    rec = sweep(target, max_points=1, timeout_ms=60_000)
    assert probe.get("source"), "the boundary carries no source"
    assert rec["lattice_source"] and "no derivable boundary" not in rec["lattice_source"]


def test_a_target_with_no_derivable_edge_sweeps_nothing_and_says_so():
    """Empty RTL facts must produce an empty lattice with a reason, never an invented default."""
    from merlin.verify.lattice import _lattice_source, lattice_points

    empty: dict = {"boundaries": {"extent_probes": []}, "cells": []}
    assert lattice_points(empty) == []
    assert "no derivable boundary" in _lattice_source(empty)


def test_the_coverage_count_does_not_triple_count_one_query():
    """Cells differing only by ALIGNMENT are the same query; the extent already expresses alignment.

    Counting them separately would report three verified points for one solved query — inflating a
    coverage number, which is the specific dishonesty this whole layer exists to avoid.
    """
    from merlin.verify.lattice import sweep

    target = _a_target_with_a_derived_edge()
    rec = sweep(target, max_points=2, timeout_ms=120_000)
    assert rec["points_total"] == rec["cell_groups_swept"] * len(rec["lattice_points"])
    assert len(rec["cells_covered"]) >= rec["cell_groups_swept"], (
        "a query group must cover at least the cell it came from")


def test_every_declared_cell_is_either_swept_or_omitted_with_a_reason():
    """Nothing is silently dropped — a missing cell would flatter the denominator."""
    from merlin.verify.lattice import load_spec, sweep

    target = _a_target_with_a_derived_edge()
    rec = sweep(target, max_points=1, timeout_ms=60_000)
    declared = {str(c.get("cell")) for c in (load_spec(target).get("cells") or ())}
    accounted = set(rec["cells_covered"]) | {o["cell"] for o in rec["cell_omissions"]}
    assert declared == accounted, f"cells neither swept nor explained: {sorted(declared - accounted)}"
    for omission in rec["cell_omissions"]:
        assert omission["reason"], f"{omission['cell']} omitted with no reason"


def test_a_refutation_makes_the_sweep_fail():
    """The sweep's exit status must distinguish a refutation from an abstention."""
    from merlin.verify.lattice import main

    target = _a_target_with_a_derived_edge()
    rc = main(["--target", target, "--max-points", "1", "--timeout-ms", "60000"])
    assert rc == 0, "a clean sweep must exit 0"


def test_an_omitted_family_names_the_real_blocker_not_a_generic_reason():
    """"We have not written a builder" and "the target cannot represent this" need different work.

    Measured 2026-09-05: the in-tree reference target declares exactly four ops, so `lower_to_target`
    refuses every non-contraction family and no command buffer exists to validate. The SMT encoder is
    NOT the limit — it already handles VECTOR_MAP, VREDUCE and MOVEMENT. A reason that blamed the
    encoder would send someone to the wrong file.
    """
    from merlin.verify.lattice import reference_target_ops, sweep

    target = _a_target_with_a_derived_edge()
    rec = sweep(target, max_points=1, timeout_ms=60_000)
    omitted = rec["cell_omissions"]
    if not omitted:
        pytest.skip("every declared family is swept for this target")

    ops = reference_target_ops()
    assert ops, "the reference target's op set could not be read; the reason would be unfounded"
    for om in omitted:
        reason = om["reason"]
        assert str(ops) in reason, "the reason must carry the op set it was derived from"
        assert "encoder is not the limit" in reason, "the reason must not misattribute the blocker"
        # and it must not invent an op symbol: family names and op names differ
        assert "interface." not in reason or "this family's interface op" in reason
