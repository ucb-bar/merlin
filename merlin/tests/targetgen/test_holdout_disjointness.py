"""The holdout/public disjointness gate — including the case where it FAILS.

A gate that has only ever been shown to pass is indistinguishable from one that cannot fail, so the
load-bearing test here is :func:`test_overlapping_pair_is_caught`: it constructs a holdout that is a
public capsule under another name and watches the check catch it. The rest pin the normalizations that
make that possible (a rename is not a new point; tiles and absolute extents are one spelling) and the
non-leakage property that lets the gate run in CI at all (it reports counts, never points).
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import repo_root

TILE = 16  # a stand-in array edge for the synthetic profiles below; nothing here reads a real target


def _checker():
    """Import the gate by path (it is a build_tools script, not an installed module)."""
    path = repo_root() / "build_tools" / "scripts" / "check_holdout_disjointness.py"
    spec = importlib.util.spec_from_file_location("merlin_check_holdout_disjointness", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _checker()


# --------------------------------------------------------------------------------------------------
# The failing case. This is the whole point of the gate.
# --------------------------------------------------------------------------------------------------
def test_overlapping_pair_is_caught():
    """A "holdout" that is a public capsule under another name must be reported as an overlap.

    The two entries below differ in every field a reader looks at first — the capsule name, the operand
    tensor labels, the category, the prose rationale — and in none of the fields that decide what
    program gets built. That is exactly the shape of the renames a prior audit of this repo found being
    scored as generalisation.
    """
    public = [{"name": "A2_matmul", "cat": "isa", "kind": "isa", "op": "matmul",
               "M_tiles": 1, "K_tiles": 1, "N_tiles": 1,
               "lhs": "A0", "weight": "W", "out": "Y0", "label": "public",
               "source_reference": "single-tile int8 matmul"}]
    holdout = [{"name": "H9_matmul_hidden", "cat": "hidden", "kind": "isa", "op": "matmul",
                "M_tiles": 1, "K_tiles": 1, "N_tiles": 1,
                "lhs": "Ah9", "weight": "Wh9", "out": "Y0", "label": "hidden",
                "source_reference": "hidden int8 matmul (allegedly unseen)"}]

    r = C.compare(public, holdout, TILE)
    assert r["disjoint"] is False
    assert r["n_holdout_coinciding_with_public"] == 1


def test_genuinely_unseen_holdout_is_disjoint():
    """The same obligation at a tile count the public set never reaches is NOT an overlap."""
    public = [{"name": "A2_matmul", "cat": "isa", "kind": "isa", "op": "matmul",
               "M_tiles": 1, "K_tiles": 1, "N_tiles": 1, "lhs": "A0", "weight": "W", "out": "Y0"}]
    holdout = [{"name": "H5_matmul_m2_hidden", "cat": "hidden", "kind": "isa", "op": "matmul",
                "M_tiles": 2, "K_tiles": 1, "N_tiles": 1, "lhs": "A0", "weight": "W", "out": "Y0"}]

    r = C.compare(public, holdout, TILE)
    assert r["disjoint"] is True
    assert r["n_holdout_coinciding_with_public"] == 0


def test_tile_relative_and_absolute_spellings_are_one_point():
    """``M_tiles: 2`` on a 16-wide array IS ``M: 32``.

    Without this a holdout could hide behind a spelling: author the public capsule in tiles, the
    "holdout" in absolute extents, and a naive field-by-field comparison calls them different.
    """
    public = [{"name": "P", "op": "matmul", "M": 32, "K": 16, "N": 16}]
    holdout = [{"name": "H", "op": "matmul", "M_tiles": 2, "K_tiles": 1, "N_tiles": 1}]
    assert C.compare(public, holdout, TILE)["n_holdout_coinciding_with_public"] == 1


def test_a_real_parameter_difference_is_not_an_overlap():
    """Only the identity/label fields are ignored — a differing epilogue or dtype is a different point."""
    base = {"name": "P", "op": "matmul", "M_tiles": 1, "K_tiles": 1, "N_tiles": 1}
    public = [dict(base)]
    for differing in ({"epilogue": ["relu"]}, {"operand_dtype": "bf16"}, {"acc_scale": 0.25},
                      {"modes": {"k_accumulate": True}}, {"op": "movement"}):
        holdout = [dict(base, name="H", **differing)]
        r = C.compare(public, holdout, TILE)
        assert r["disjoint"] is True, f"{differing} was wrongly collapsed onto the public point"


def test_null_valued_keys_do_not_distinguish_points():
    """An unquoted prose value inside a YAML flow mapping splits on its own commas and leaves the tail
    as a bare null-valued KEY. Five sidecars in this repo carried one. If a null counted, two entries
    with differently-worded rationales would compare as different programs and every rename would
    escape."""
    public = [{"name": "P", "op": "matmul", "M": 16, "K": 16, "N": 16}]
    holdout = [{"name": "H", "op": "matmul", "M": 16, "K": 16, "N": 16, "2 M-tiles)": None}]
    assert C.compare(public, holdout, TILE)["n_holdout_coinciding_with_public"] == 1


def test_holdout_set_that_duplicates_itself_is_reported():
    """Two identical holdouts are one holdout; the set is smaller than its count says."""
    e = {"op": "matmul", "M": 16, "K": 16, "N": 16}
    r = C.compare([], [dict(e, name="H1", lhs="A"), dict(e, name="H2", lhs="B")], TILE)
    assert r["n_holdout_points"] == 2
    assert r["n_holdout_internal_duplicates"] == 1


# --------------------------------------------------------------------------------------------------
# Non-leakage: the gate must be able to FAIL without saying what collided.
# --------------------------------------------------------------------------------------------------
def test_report_carries_counts_and_booleans_only():
    """No value from either entry list may appear in the report.

    The specification of a holdout (op + dtype + shape) is itself an answer, and this repo has had an
    answer-key incident. A gate that printed the colliding shape in order to fail a build would leak
    precisely what the untracked sidecar exists to hide.
    """
    public = [{"name": "P", "op": "matmul", "operand_dtype": "int8", "M": 16, "K": 16, "N": 16}]
    holdout = [{"name": "SECRET_HOLDOUT", "op": "matmul", "operand_dtype": "int8",
                "M": 999, "K": 777, "N": 555}]
    r = C.compare(public, holdout, TILE)
    assert set(r) == {"n_public_distinct_points", "n_holdout_points",
                      "n_holdout_coinciding_with_public", "n_holdout_internal_duplicates", "disjoint"}
    for value in r.values():
        assert isinstance(value, (int, bool))
    blob = repr(r)
    for leaked in ("SECRET_HOLDOUT", "999", "777", "555", "matmul", "int8"):
        assert leaked not in blob


# --------------------------------------------------------------------------------------------------
# Ratchet: debt is per-target, and it may only shrink.
# --------------------------------------------------------------------------------------------------
def test_ratchet_is_scoped_per_target(tmp_path):
    """One target's tolerated overlap must not excuse another's."""
    p = tmp_path / "ratchet.txt"
    p.write_text("# comment\ntarget_a overlap:2\n\ntarget_b overlap:0\n")
    r = C._load_ratchet(p)
    assert r == {"target_a overlap": 2, "target_b overlap": 0}
    assert C._debt_key("target_a") in r
    assert C._debt_key("target_c") not in r          # unlisted target allows nothing


def test_ratchet_rejects_a_malformed_line(tmp_path):
    p = tmp_path / "ratchet.txt"
    p.write_text("target_a 2\n")
    with pytest.raises(ValueError):
        C._load_ratchet(p)


def test_shipped_ratchet_parses_and_every_target_is_within_it():
    """The gate as it is actually configured: run it over the real profiles and hold every target to
    its ratcheted allowance. Counts only — nothing here reads or prints a holdout's values."""
    ratchet = C._load_ratchet(repo_root() / "build_tools" / "scripts"
                              / "holdout_disjointness_ratchet.txt")
    reports = [C.audit(t) for t in C.targets_with_holdouts()]
    assert reports, "no target declares a holdout sidecar; the gate would be vacuous"
    for r in reports:
        # Deliberately strict: every shipped target must be genuinely CHECKED. Accepting
        # "could not run" here is how a gate quietly stops gating (`no_holdout_sidecar` would be the
        # honest answer in a public clone, but not in this tree, where every sidecar is present).
        assert r["status"] == "ok", f"{r['target']}: {r['status']} — {r.get('detail')}"
        allowed = ratchet.get(C._debt_key(r["target"]), 0)
        assert r["n_holdout_coinciding_with_public"] <= allowed, (
            f"{r['target']}: {r['n_holdout_coinciding_with_public']} holdout point(s) coincide with a "
            f"public point, allowance {allowed}. The values are deliberately not shown; read the "
            f"untracked sidecar.")
        assert not r["unclassified_keys"], (
            f"{r['target']}: unclassified profile field(s) {r['unclassified_keys']} — classify them in "
            f"check_holdout_disjointness.py rather than leaving the comparison UNKNOWN about them")


def test_every_target_with_a_sidecar_generates_some_holdouts():
    """A sidecar whose holdouts are all hand-authored is the state this work exists to move off.

    Not every point must be generated — the hand-authored data-independence checks are deliberate — but
    a target with no generated holdout at all has no family that transfers when its array edge changes.
    """
    for r in (C.audit(t) for t in C.targets_with_holdouts()):
        if r["status"] != "ok":
            continue
        assert r["n_holdout_generated"] > 0, (
            f"{r['target']}: every holdout is hand-authored; declare a `sweeps:` block in its hidden "
            f"profile so the points are computed from the target's own tile edge")
