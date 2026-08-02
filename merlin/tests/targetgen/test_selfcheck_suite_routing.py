"""The self-check (and grade's trace-coverage) must READ capsule_result.json from the SAME suite dir
run_capsule WRITES them to — derived per target (cfg.suite), never the gemmini `SUITE` literal.

Regression for the atlas 0/11 blind-loop: agent_selfcheck re-globbed `<runs_root>/runs/<CR.SUITE>`
('gemmini-capsule-bench') while run_capsule wrote atlas results under 'atlas-capsule-bench'. The glob
found nothing, so every self-check returned n_capsules:0 / per_capsule:[] and the agent iterated blind
(flat 0/11, 12 rounds) even though the driver's in-memory grade was correct. The fix is CR.suite_for(target).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.capsule_common import make_run_paths


def _write_parent(target: str) -> Path:
    """Where run_capsule lays a capsule's run dir for this target (parent of capsule_result.json)."""
    cfg = CR._config_for_target(target, None, "i8xi8_i32")
    rr = Path(tempfile.mkdtemp())
    paths = make_run_paths(rr, "AT2", suite=cfg.suite, target=cfg.target, dtype=cfg.dtype, benchmark="c")
    return paths.run_path.parent, rr


@pytest.mark.parametrize("target", ["gemmini", "atlas", "radiance"])
def test_read_suite_matches_write_suite(target):
    # the reader's path (runs_root/runs/suite_for(target)) MUST equal run_capsule's write parent.
    write_parent, rr = _write_parent(target)
    read_root = rr / "runs" / CR.suite_for(target)
    assert read_root == write_parent, (
        f"{target}: self-check would glob {read_root} but results are written under {write_parent}")


def test_suite_for_is_target_derived_not_the_gemmini_literal():
    # atlas must NOT resolve to the gemmini suite literal — that identity was the whole bug.
    assert CR.suite_for("atlas") != CR.SUITE
    assert CR.suite_for("gemmini") == CR.SUITE          # gemmini legitimately owns the literal
    assert "atlas" in CR.suite_for("atlas")


def test_reader_finds_a_result_written_under_the_target_suite():
    # end-to-end path proof (no build/oracle): drop a capsule_result.json where run_capsule would write
    # it for atlas, and confirm the suite_for-based glob (what agent_selfcheck now uses) sees it while the
    # old gemmini-literal glob does not.
    _wp, rr = _write_parent("atlas")
    suite = CR.suite_for("atlas")
    cap_dir = rr / "runs" / suite / "AT2_single_tile_matmul"
    cap_dir.mkdir(parents=True)
    (cap_dir / "capsule_result.json").write_text(json.dumps({"capsule": "AT2_single_tile_matmul",
                                                             "status": "fail", "tiers": {}}))
    fixed = list((rr / "runs" / CR.suite_for("atlas")).glob("*/capsule_result.json"))
    old = list((rr / "runs" / CR.SUITE).glob("*/capsule_result.json"))
    assert len(fixed) == 1                              # the fix finds the atlas result
    assert len(old) == 0                                # the old gemmini-literal glob was blind
