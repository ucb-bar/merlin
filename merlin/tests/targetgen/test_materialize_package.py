"""Installing a run's submission as a target's compiler is gated on the score being real evidence.

`_default_oot_package` resolves out/artifacts/targets/<t>/{agent_spec_v1_mlir_oot,reference_v0}; a target
with neither cannot emit a mesh kernel at all, so every whole-model layer falls back to the host. Filling
that in from a certified run is what turns "a run that scored well" into "an installed compiler" — which
is exactly why the score behind it has to be checked rather than trusted.
"""
from __future__ import annotations

import json

import pytest
import yaml

from merlin.targetgen import publish as PB


def _score(**over):
    s = {"integrity_status": "clean", "gradeable": True, "n_passed": 2, "n_capsules": 2,
         "per_capsule": [{"capsule": "R0", "status": "pass", "tiers": {"L2": "pass"}},
                         {"capsule": "R1", "status": "pass", "tiers": {"L2": "pass"}}]}
    s.update(over)
    return s


def test_a_score_with_tier_evidence_is_accepted():
    ok, detail = PB._score_is_honest(_score())
    assert ok and "2/2" in detail


def test_a_pass_with_an_empty_tier_map_is_refused():
    """The exact shape that let four whole-model capsules report `pass` without executing. Installing a
    compiler on that evidence would trust the same hollow row a second time."""
    s = _score(per_capsule=[{"capsule": "R0", "status": "pass", "tiers": {"L2": "pass"}},
                            {"capsule": "M0", "status": "pass", "tiers": {}}])
    ok, detail = PB._score_is_honest(s)
    assert not ok and "M0" in detail and "no tier evidence" in detail


def test_an_empty_run_is_refused_not_vacuously_accepted():
    """`all([])` is True — an empty suite must never read as a clean sweep."""
    ok, detail = PB._score_is_honest(_score(per_capsule=[]))
    assert not ok and "nothing was graded" in detail


def test_a_dirty_integrity_status_is_refused():
    ok, detail = PB._score_is_honest(_score(integrity_status="tampered"))
    assert not ok and "integrity_status" in detail


def test_an_ungradeable_run_is_refused():
    ok, _ = PB._score_is_honest(_score(gradeable=False))
    assert not ok


def test_materialize_installs_the_package_and_records_where_it_came_from(tmp_path):
    src = tmp_path / "submission"
    (src / "mlir_oot").mkdir(parents=True)
    (src / "mlir_oot" / "tool").write_text("#!/bin/sh\n", encoding="utf-8")
    (src / "manifest.yaml").write_text(yaml.safe_dump({"package_id": "radiance-xdsl-oot",
                                                       "family": "mlir_oot"}), encoding="utf-8")
    score = tmp_path / "score.json"
    score.write_text(json.dumps(_score()), encoding="utf-8")

    dst = PB.materialize_package("radiance", src, certified_by_run="run_v4", score_path=score,
                                 artifacts_root=tmp_path / "artifacts")
    assert (dst / "mlir_oot" / "tool").is_file()
    man = yaml.safe_load((dst / "manifest.yaml").read_text(encoding="utf-8"))
    assert man["package_id"] == "agent_spec_v1_mlir_oot"
    assert man["promotion"]["source_package_id"] == "radiance-xdsl-oot", "the original id is preserved"
    assert man["promotion"]["evidence"]["n_passed"] == 2
    assert man["provenance"], "a compiler install records which tree it came from"


def test_materialize_does_not_claim_a_certification_rank_it_lacks(tmp_path):
    """`_STATUS_RANK` knows rtl_certified / k1_verified / spike_verified. A functional-tier verdict is
    none of those, and writing one would both overstate the evidence and silently win select_champion,
    whose ties break on directory name."""
    src = tmp_path / "submission"
    src.mkdir()
    (src / "manifest.yaml").write_text(yaml.safe_dump({"package_id": "x"}), encoding="utf-8")
    dst = PB.materialize_package("radiance", src, artifacts_root=tmp_path / "artifacts")
    man = yaml.safe_load((dst / "manifest.yaml").read_text(encoding="utf-8"))
    assert man.get("status") not in PB.CERTIFIED_STATUSES
    assert man["publication"]["champion"] is False


def test_materialize_refuses_a_dishonest_score(tmp_path):
    src = tmp_path / "submission"
    src.mkdir()
    (src / "manifest.yaml").write_text(yaml.safe_dump({"package_id": "x"}), encoding="utf-8")
    score = tmp_path / "score.json"
    score.write_text(json.dumps(_score(per_capsule=[{"capsule": "M0", "status": "pass", "tiers": {}}])),
                     encoding="utf-8")
    with pytest.raises(PB.MaterializeRefused):
        PB.materialize_package("radiance", src, score_path=score, artifacts_root=tmp_path / "artifacts")


def test_materialize_refuses_a_source_that_is_not_a_package(tmp_path):
    src = tmp_path / "not_a_package"
    src.mkdir()
    with pytest.raises(PB.MaterializeRefused):
        PB.materialize_package("radiance", src, artifacts_root=tmp_path / "artifacts")


def test_materialize_will_not_silently_replace_an_installed_compiler(tmp_path):
    src = tmp_path / "submission"
    src.mkdir()
    (src / "manifest.yaml").write_text(yaml.safe_dump({"package_id": "x"}), encoding="utf-8")
    PB.materialize_package("radiance", src, artifacts_root=tmp_path / "artifacts")
    with pytest.raises(PB.MaterializeRefused):
        PB.materialize_package("radiance", src, artifacts_root=tmp_path / "artifacts")
    PB.materialize_package("radiance", src, artifacts_root=tmp_path / "artifacts", force=True)

def test_installing_a_package_does_not_take_the_champion_slot(tmp_path):
    """`select_champion`'s last tie-break is the directory NAME, so installing `agent_spec_v1_mlir_oot`
    beside `hand_v0` — neither carrying a ranked certification status — handed the champion slot to the
    new package purely because "a" sorts before "h". Every consumer that asks for the champion would have
    been silently redirected to a different compiler. An explicit `champion: false` now demotes."""
    art = tmp_path / "artifacts"
    (art / "targets" / "radiance" / "hand_v0").mkdir(parents=True)
    (art / "targets" / "radiance" / "hand_v0" / "manifest.yaml").write_text(
        yaml.safe_dump({"package_id": "hand_v0"}), encoding="utf-8")

    src = tmp_path / "submission"
    src.mkdir()
    (src / "manifest.yaml").write_text(yaml.safe_dump({"package_id": "x"}), encoding="utf-8")
    PB.materialize_package("radiance", src, artifacts_root=art)

    assert PB.select_champion("radiance", artifacts_root=art).package_id == "hand_v0"


def test_a_declined_package_still_wins_when_it_is_the_only_one(tmp_path):
    """Demotion must not mean 'never selectable' — a target whose sole package is an install still has
    to resolve to it."""
    art = tmp_path / "artifacts"
    src = tmp_path / "submission"
    src.mkdir()
    (src / "manifest.yaml").write_text(yaml.safe_dump({"package_id": "x"}), encoding="utf-8")
    PB.materialize_package("radiance", src, artifacts_root=art)
    assert PB.select_champion("radiance", artifacts_root=art).package_id == "agent_spec_v1_mlir_oot"
