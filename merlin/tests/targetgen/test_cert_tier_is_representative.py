"""Full coverage at the functional tier, representative coverage at the cert tier.

Two changes are pinned here, both aimed at the same measurement: on this repo's SIMT target the agent's
self-check was 80% of a round's wall clock, and 34% of that round returned nothing at all, because the
cycle-accurate tier ran on every capsule every sweep -- for a verdict the score never read.

  1. The tier ladder short-circuits once a MANDATORY tier has failed. A capsule whose numerics are already
     known wrong cannot be rescued by an RTL verdict, and RTL is the expensive one. A sweep scoring 18/35
     was paying cert cost for all 35.
  2. `cert_capsule_cover` picks a representative subset for the cert tier, derived from what capsules
     declare rather than hand-listed, so a new target's cover falls out of its own corpus.

The invariant that must not be lost in either: an OPTIONAL cert tier failing is the "passes the functional
oracle, fails RTL" signal the benchmark exists to surface, so it must never suppress anything.
"""
from __future__ import annotations

import yaml

from merlin.targetgen.contract.materialize import cert_capsule_cover


def _corpus(tmp_path, specs):
    """specs: [(name, family, [dtypes])] -> a corpus root of public capsules."""
    root = tmp_path / "corpus"
    for name, fam, dts in specs:
        d = root / name
        d.mkdir(parents=True)
        (d / "capsule.yaml").write_text(yaml.safe_dump({
            "name": name, "kind": "op", "label": "public",
            "semantic": {"semantic_family": fam},
            "inputs": [{"name": f"A{i}", "role": "input", "shape": [4, 4], "dtype": dt}
                       for i, dt in enumerate(dts)],
        }), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------------------------
# the cover
# ---------------------------------------------------------------------------------------------
def test_a_redundant_corpus_collapses_to_one_per_cell(tmp_path):
    """Three capsules in the same (family, dtype) cell are interchangeable to the HARDWARE, so the cert
    tier should run one, not three. This is where the saving comes from."""
    root = _corpus(tmp_path, [("A", "contraction", ["f32"]),
                              ("B", "contraction", ["f32"]),
                              ("C", "contraction", ["f32"])])
    r = cert_capsule_cover([root])
    assert len(r["capsules"]) == 1
    assert r["uncovered"] == []


def test_every_cell_is_covered_and_none_is_dropped_silently(tmp_path):
    root = _corpus(tmp_path, [("A", "contraction", ["f32"]),
                              ("B", "contraction", ["mxfp4"]),
                              ("C", "normalization", ["f32"]),
                              ("D", "attention", ["bf16"])])
    r = cert_capsule_cover([root])
    assert len(r["capsules"]) == 4          # four distinct cells -> nothing is redundant
    assert r["uncovered"] == []
    assert set(r["cells"]) == {"contraction/f32", "contraction/mxfp4",
                               "normalization/f32", "attention/bf16"}


def test_both_compute_units_survive_the_cover(tmp_path):
    """dtype is the proxy for WHICH unit runs the kernel -- block-scaled formats go to the MX PE. A cover
    that drops every MX capsule would leave the MX datapath with no RTL evidence at all."""
    root = _corpus(tmp_path, [("F1", "contraction", ["f32"]), ("F2", "contraction", ["fp16"]),
                              ("M1", "contraction", ["mxfp8"]), ("M2", "contraction", ["mxfp4"])])
    got = set(cert_capsule_cover([root])["capsules"])
    assert {"M1", "M2"} <= got, "the MX datapath lost its cert coverage"
    assert {"F1", "F2"} <= got


def test_the_declared_axes_are_reported_not_assumed(tmp_path):
    """`instruction_classes` would be the most faithful axis, but every capsule here declares it empty.
    Selecting on an empty axis silently returns a cover of one, so the basis must say what actually
    carried the choice."""
    root = _corpus(tmp_path, [("A", "contraction", ["f32"]), ("B", "attention", ["fp16"])])
    r = cert_capsule_cover([root])
    assert r["basis"]["axes"] == ["semantic_family", "dtype"]
    assert r["basis"]["instruction_classes_available"] == 0
    assert "instruction_classes" in r["basis"]["note"]


def test_a_capsule_with_no_declared_dtype_is_skipped_not_guessed(tmp_path):
    root = tmp_path / "c"
    (root / "X").mkdir(parents=True)
    (root / "X" / "capsule.yaml").write_text(yaml.safe_dump({
        "name": "X", "label": "public", "semantic": {"semantic_family": "contraction"}, "inputs": []}),
        encoding="utf-8")
    assert cert_capsule_cover([root])["capsules"] == []


def test_non_public_capsules_are_not_in_the_public_cover(tmp_path):
    root = _corpus(tmp_path, [("A", "contraction", ["f32"])])
    (root / "H").mkdir()
    (root / "H" / "capsule.yaml").write_text(yaml.safe_dump({
        "name": "H", "label": "hidden", "semantic": {"semantic_family": "reduction"},
        "inputs": [{"name": "A0", "shape": [4], "dtype": "f32"}]}), encoding="utf-8")
    assert cert_capsule_cover([root])["capsules"] == ["A"]


# ---------------------------------------------------------------------------------------------
# the real corpus
# ---------------------------------------------------------------------------------------------
def test_the_shipped_corpus_covers_every_cell():
    """Whatever this target's corpus is, the cover must leave nothing uncovered -- an uncovered cell is a
    corpus gap, and the function reports it rather than hiding it."""
    from merlin.common.paths import repo_root
    roots = [repo_root() / "merlin/contract/capsules/radiance/isa",
             repo_root() / "merlin/contract/capsules/radiance/model_slices"]
    if not all(r.is_dir() for r in roots):
        import pytest
        pytest.skip("radiance corpus absent in this checkout")
    r = cert_capsule_cover(roots)
    assert r["uncovered"] == [], f"cells no capsule covers: {r['uncovered']}"
    assert 0 < len(r["capsules"]) < r["basis"]["n_candidates"], "a cover that saves nothing is not a cover"


# ---------------------------------------------------------------------------------------------
# the short-circuit
# ---------------------------------------------------------------------------------------------
def _ladder(tiers_recorded):
    """The predicate the ladder uses, lifted so it can be exercised without a live oracle: 'has a
    MANDATORY tier already failed', derived from what was RECORDED rather than a flag each failure path
    must remember to set."""
    return next((t for t, r in tiers_recorded.items()
                 if r.get("mandatory") and r.get("status") == "fail"), None)


def test_a_failed_mandatory_tier_suppresses_deeper_tiers():
    """The saving: a capsule whose numerics are already wrong cannot be rescued by an RTL verdict, and
    RTL is the tier that costs minutes."""
    assert _ladder({"L2": {"mandatory": True, "status": "fail"}}) == "L2"


def test_a_failed_OPTIONAL_tier_suppresses_nothing():
    """The invariant that must survive the optimisation. 'Passes the functional oracle, fails RTL' is the
    signal this benchmark exists to surface -- measured live as `L2:pass 6, L3:fail 6`. An optional cert
    tier going red must never switch off anything below or after it."""
    assert _ladder({"L3": {"mandatory": False, "status": "fail"}}) is None


def test_a_passing_mandatory_tier_suppresses_nothing():
    assert _ladder({"L2": {"mandatory": True, "status": "pass"}}) is None


def test_unavailable_is_not_failure():
    """An oracle that could not run is not a capsule that failed; it must not suppress a deeper tier that
    might be reachable."""
    assert _ladder({"L2": {"mandatory": True, "status": "unavailable"}}) is None


def test_suppressed_tiers_are_skipped_never_failed():
    """`not_run_is_not_pass` reads a fabricated `fail` as evidence the capsule was CERTIFIED at that tier
    and found wrong. A tier that never executed has no such evidence, so it must say `skipped` and name
    why -- otherwise the optimisation invents a cycle-accurate verdict for a capsule no RTL ever saw."""
    from merlin.targetgen.capsule_runner import suppressed_tier_result
    r = suppressed_tier_result("L3", mandatory=False, failed_tier="L2", from_rtl=True)
    assert r.status == "skipped"
    assert r.status != "fail"
    assert "L2" in r.reason and "already failed" in r.reason
    assert r.derived_from_rtl is True


def test_a_suppressed_tier_keeps_its_own_mandatory_flag():
    """Suppression must not quietly demote a mandatory tier to optional -- the fail-open guard reads that
    flag to decide whether an absent verdict is acceptable."""
    from merlin.targetgen.capsule_runner import suppressed_tier_result
    assert suppressed_tier_result("L3", mandatory=True, failed_tier="L2").mandatory is True


# ---------------------------------------------------------------------------------------------
# tile alignment: the axis a functional model is least able to stand in for
# ---------------------------------------------------------------------------------------------
def _corpus_shaped(tmp_path, specs):
    """specs: [(name, family, dtype, shape)]."""
    root = tmp_path / "shaped"
    for name, fam, dt, shape in specs:
        d = root / name
        d.mkdir(parents=True)
        (d / "capsule.yaml").write_text(yaml.safe_dump({
            "name": name, "kind": "op", "label": "public",
            "semantic": {"semantic_family": fam},
            "inputs": [{"name": "A0", "role": "input", "shape": list(shape), "dtype": dt}],
        }), encoding="utf-8")
    return root


def test_a_partial_tile_is_its_own_cell(tmp_path):
    """A taped-out unit in this repo computed partial N tiles (n %% 64 != 0) WRONGLY while every functional
    check passed. Family and dtype cannot see that: both capsules here are contraction/f32, so a 2-axis
    cover certifies one of them and might never run a ragged extent at all."""
    root = _corpus_shaped(tmp_path, [("Aligned", "contraction", "f32", [32, 32]),
                                     ("Ragged", "contraction", "f32", [32, 33])])
    two_axis = cert_capsule_cover([root])
    assert len(two_axis["capsules"]) == 1, "precondition: without alignment these look interchangeable"

    three_axis = cert_capsule_cover([root], tile_dim=16)
    assert set(three_axis["capsules"]) == {"Aligned", "Ragged"}
    assert any(c.endswith("/partial") for c in three_axis["cells"])
    assert any(c.endswith("/aligned") for c in three_axis["cells"])


def test_one_ragged_axis_is_enough_to_be_partial(tmp_path):
    """Exercising the tile-edge path is the point; a capsule that is ragged on any axis does that."""
    root = _corpus_shaped(tmp_path, [("X", "contraction", "f32", [32, 32, 17])])
    assert cert_capsule_cover([root], tile_dim=16)["cells"] == ["contraction/f32/partial"]


def test_omitting_tile_dim_leaves_the_blind_spot_and_says_so(tmp_path):
    """The axis is optional because the tile edge is the caller's fact, not the corpus's. What must never
    happen is a cover that silently claims coverage it does not have."""
    root = _corpus_shaped(tmp_path, [("A", "contraction", "f32", [32, 33])])
    r = cert_capsule_cover([root])
    assert "tile_alignment" not in r["basis"]["axes"]
    assert r["basis"]["tile_dim"] is None


def test_the_alignment_axis_only_ever_widens_the_cover(tmp_path):
    """Adding an axis must never DROP a capsule that was covering something -- a narrower cover after
    adding information would mean the extra axis lost a cell."""
    root = _corpus_shaped(tmp_path, [("A", "contraction", "f32", [32, 32]),
                                     ("B", "attention", "fp16", [32, 33]),
                                     ("C", "attention", "fp16", [16, 16])])
    two, three = cert_capsule_cover([root]), cert_capsule_cover([root], tile_dim=16)
    assert len(three["capsules"]) >= len(two["capsules"])
    assert three["uncovered"] == [] and two["uncovered"] == []
