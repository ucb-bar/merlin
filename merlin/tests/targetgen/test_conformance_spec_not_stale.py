"""The tracked conformance spec must still agree with the facts it was derived from.

A derived artifact that is committed and then never re-checked is a hand-authored artifact with extra
steps: it keeps asserting a requirement the target no longer has, and nothing says so. These tests do not
demand byte-equality with a fresh derivation — the requirement legitimately WIDENS when a new model
capture lands, and failing on that would train people to regenerate without reading. They demand that
every cell the spec still claims is one the CURRENT capability manifest still admits, and that the spec's
recorded boundaries still match the target's own facts.

When one of these fails the fix is to regenerate, not to edit:

    build_tools/scripts/check_conformance_coverage.py --target <t> \\
        --write merlin/contract/capsules/conformance/<t>.yaml
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import conformance as CF

SPEC_DIR = repo_root() / "merlin" / "contract" / "capsules" / "conformance"


def _specs():
    return sorted(SPEC_DIR.glob("*.yaml")) if SPEC_DIR.is_dir() else []


def _load(p):
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


@pytest.mark.parametrize("spec_path", _specs(), ids=lambda p: p.stem)
def test_every_required_cell_is_still_admitted(spec_path):
    """A cell whose family/dtype the hardware no longer declares is a stale requirement."""
    doc = _load(spec_path)
    target = doc.get("target") or spec_path.stem
    adm = CF.admitted(target)
    if not adm:
        pytest.skip(f"no capability manifest resolvable for {target!r} in this environment")
    admitted_pairs = {(fam, CF.capsule_dtype(dt)) for fam, dts in adm.items() for dt in dts}
    stale = [c["cell"] for c in (doc.get("cells") or [])
             if (c.get("family"), c.get("dtype")) not in admitted_pairs]
    assert not stale, (
        f"{spec_path.name} requires {len(stale)} cell(s) the manifest no longer admits: {stale[:8]} — "
        f"regenerate the spec")


@pytest.mark.parametrize("spec_path", _specs(), ids=lambda p: p.stem)
def test_recorded_boundaries_still_match_the_target(spec_path):
    """The tile edge and block-scale group are hardware/derivation facts; a drift means the spec's
    alignment axis and extent probes describe a machine that is no longer this one."""
    doc = _load(spec_path)
    target = doc.get("target") or spec_path.stem
    rec = doc.get("boundaries") or {}
    now = CF.boundaries(target)
    if now.tile_edge is None and rec.get("tile_edge") is None:
        pytest.skip("no tile edge either then or now")
    assert rec.get("tile_edge") == now.tile_edge, (
        f"tile edge drifted: spec says {rec.get('tile_edge')}, target derives {now.tile_edge}")
    assert rec.get("block_scale_group") == now.block_scale_group, (
        f"block-scale group drifted: spec says {rec.get('block_scale_group')}, target derives "
        f"{now.block_scale_group}")


@pytest.mark.parametrize("spec_path", _specs(), ids=lambda p: p.stem)
def test_a_declared_cell_carries_its_citation(spec_path):
    """`declared` is the escape hatch for a target-model with no capture. It is only honest while it
    says WHO asserted it — otherwise it is indistinguishable from an observation."""
    doc = _load(spec_path)
    naked = [c["cell"] for c in (doc.get("cells") or [])
             if c.get("basis") == CF.DECLARED and not c.get("citation")]
    assert not naked, f"declared cell(s) with no citation: {naked}"


@pytest.mark.parametrize("spec_path", _specs(), ids=lambda p: p.stem)
def test_the_spec_states_the_basis_of_each_axis(spec_path):
    """The dtype axis is admitted-only, not observed. A reader who takes cell count as measured demand
    will over-read the requirement, so the spec has to say so itself."""
    doc = _load(spec_path)
    basis = (doc.get("diagnostics") or {}).get("axis_basis") or {}
    assert basis, f"{spec_path.name} records no axis_basis — regenerate"
    assert "ADMITTED ONLY" in (basis.get("dtype") or "")


@pytest.mark.parametrize("spec_path", _specs(), ids=lambda p: p.stem)
def test_composite_cells_name_the_primitives_that_evidence_them(spec_path):
    """`observed_via_primitives` is an INFERENCE (the importer decomposes attention before we see it).
    It must carry the primitives it rests on, or it reads as a direct observation."""
    doc = _load(spec_path)
    bad = [c["cell"] for c in (doc.get("cells") or [])
           if c.get("basis") == CF.OBSERVED_VIA_PRIMITIVES and not c.get("via_primitives")]
    assert not bad, f"composite cell(s) with no recorded primitives: {bad}"
