from __future__ import annotations

import importlib.util

from merlin.common.paths import repo_root

SCRIPTS = repo_root() / "build_tools" / "checks"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CC = _load("check_conformance_coverage")


def _report(**axes) -> dict:
    """A clean `status: ok` report, with only the named axes carrying a gap."""
    r = {"target": "T", "status": "ok", "uncovered": []}
    for key, _tag in CC.AXES:
        r[key] = {"uncovered": []}
    r.update(axes)
    return r


def test_shape_geometry_gap_reaches_the_verdict():
    """A gap on ONE axis, everything else clean, must produce debt.

    Before the fix `shape_geometry` produced none, so --fail-on-uncovered returned 0 on it.
    """
    key = "shape_geometry"
    tag = "geometry"
    debt = CC.uncovered_debt([_report(**{key: {"uncovered": ["probe_gap"]}})], set())
    assert debt == [f"T {tag}:probe_gap"], (
        f"an uncovered {tag} gap did not reach the gate's verdict; "
        f"--fail-on-uncovered cannot fail on this axis. got: {debt}"
    )


CS = _load("check_claim_set_disjointness")


def test_uncaptured_claim_reaches_the_unmeasured_verdict():
    status = "claim_model_uncaptured"
    want = "unmeasured"
    assert CS.verdict_bucket(status) == want, (
        f"status {status!r} does not map to {want!r}; a status that maps to nothing cannot change the exit code, "
        "which is how check #2 (claim_model_uncaptured) came to be unfailable"
    )
