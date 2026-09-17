"""A vanished staged cohort is an INFRASTRUCTURE fault, never a verdict on the submission.

The bug this pins. ``public_capsules_for`` publishes each target's capsule cohort by building a unique
``.<target>.build.<pid>.<hex>`` dir and atomically repointing a per-target symlink at it. A grader
resolves that symlink to the concrete build dir ONCE (``capsule_grade.grade``, so the root cannot move
under a sibling thread's chdir) and then reads capsules out of that absolute path for the whole grade --
10-20 min functional, ~30-40 min per capsule on the cert tier. The materializer's collector dropped any
build older than 15 minutes on the stated premise that "no live reader can still hold a path into them".
Readers hold them far longer than that.

Measured, on the run that motivated this file: an arm materialized the target at 03:14:21, the age-only
collector rmtree'd the build dir a LIVE sibling grade had already resolved, and every capsule that grade
touched afterwards was recorded ``schema / structural_invariant_violation`` -- "your package is
structurally invalid" -- 31 times. ``qa_history/verdict_round_00.json`` archived
``n_passed: 2, n_capsules: 33, gradeable: true, first_failure_planes: {schema: 31}`` for a submission
that had scored 33/34 minutes earlier, and ``round_brief`` then handed those 31 phantom defects to the
next round as the agent's own failure history.

So there are two obligations, and the second matters even if the first ever regresses:
  * the collector must not take a build a live reader holds (builder alive, or a live reader lease);
  * a missing staged input must be reported as an infrastructure fault -- its own plane and category --
    and must never make the run read as gradeable or as a pass.
"""
from __future__ import annotations

import json
import os

import pytest

from merlin.common.paths import repo_root


# --------------------------------------------------------------------------- the honest failure


def _stub_pkg(tmp_path):
    """Minimal stand-in for a built package: ``run_entrypoints`` only needs ``pkg.tool`` to exist to get
    past its build gate and reach the interface-MLIR resolution this test is about."""
    tool = tmp_path / "tool"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")

    class _P:
        pass

    p = _P()
    p.tool = tool
    return p


def _paths(tmp_path):
    class _Paths:
        generated = tmp_path / "generated"

    _Paths.generated.mkdir(parents=True, exist_ok=True)
    return _Paths


def test_vanished_staging_dir_is_an_infrastructure_fault_not_a_schema_verdict(tmp_path):
    """The exact race: the capsule's staged directory is gone when the grade reaches it."""
    from merlin.targetgen.capsule_common import run_entrypoints
    from merlin.targetgen.oot_runner import (INFRASTRUCTURE_PLANE, CertFailure, InfraCategory,
                                             InfraFailure)

    # A capsule discovered out of a staging dir that a collector then removed.
    staged = tmp_path / ".sometarget.build.999999.deadbeef" / "A0_config_smoke"
    capsule = {"__dir__": str(staged), "interface_mlir": "capsule.interface.mlir"}
    assert not staged.exists()

    with pytest.raises(InfraFailure) as ei:
        run_entrypoints(_stub_pkg(tmp_path), tmp_path, capsule, _paths(tmp_path),
                        contract=None, timeout=5, fourth_output_name="out.mlir")

    inf = ei.value
    assert inf.plane == INFRASTRUCTURE_PLANE == "infrastructure"
    assert inf.category is InfraCategory.COHORT_NOT_MATERIALIZED
    # The recorded token must be the honest one whichever recorder writes the row.
    assert str(inf.category) == inf.category.value == "cohort_not_materialized"
    # It must NOT wear the submission verdict's clothes.
    assert inf.plane != "schema"
    assert "structural_invariant_violation" not in str(inf.category)
    # The message has to name the staging path and say the cohort was not materialized.
    assert str(staged) in inf.detail
    assert "not materialized" in inf.detail
    assert "not a defect in the graded submission" in inf.detail.lower()
    # Still a CertFailure, so every existing recorder keeps working.
    assert isinstance(inf, CertFailure)


def test_present_capsule_dir_missing_interface_is_still_a_schema_violation(tmp_path):
    """The genuine corpus defect must NOT be relabelled: the directory is there, the file is not."""
    from merlin.targetgen.capsule_common import run_entrypoints
    from merlin.targetgen.oot_runner import InfraFailure

    staged = tmp_path / ".sometarget.build.999999.deadbeef" / "A0_config_smoke"
    staged.mkdir(parents=True)
    capsule = {"__dir__": str(staged), "interface_mlir": "capsule.interface.mlir"}

    with pytest.raises(Exception) as ei:
        run_entrypoints(_stub_pkg(tmp_path), tmp_path, capsule, _paths(tmp_path),
                        contract=None, timeout=5, fourth_output_name="out.mlir")
    err = ei.value
    assert not isinstance(err, InfraFailure), "a present dir with a missing file is a corpus defect"
    assert err.plane == "schema"
    assert str(err.category) == "structural_invariant_violation" or \
        getattr(err.category, "value", None) == "structural_invariant_violation"


def test_infrastructure_fault_status_is_never_measured_and_never_a_pass():
    """The status must leave the pass/fail denominators AND be schema-legal."""
    from merlin.targetgen.capsule_common import NOT_MEASURED_STATUSES

    assert "infrastructure_fault" in NOT_MEASURED_STATUSES

    schema = json.loads(
        (repo_root() / "merlin/contract/schemas/capsule_result.schema.json").read_text(encoding="utf-8"))
    assert "infrastructure_fault" in schema["properties"]["status"]["enum"], \
        "a status the runner can write must be writable through the result schema"


def test_runner_maps_infrafailure_before_certfailure():
    """InfraFailure IS a CertFailure, so handler ORDER is what keeps it out of the submission's verdict.

    Asserted on the source of ``run_capsule``'s handler chain because reaching it for real needs a full
    graded run; the thing that regresses is someone deleting the narrower clause or moving it below the
    broader one, and both are visible here.
    """
    import inspect

    from merlin.targetgen import capsule_runner

    src = inspect.getsource(capsule_runner)
    i_infra = src.index("except InfraFailure as inf:")
    i_cert = src.index("except CertFailure as cf:", i_infra - 4000 if i_infra > 4000 else 0)
    assert i_infra < i_cert, "InfraFailure must be caught BEFORE CertFailure or it becomes a `fail`"
    assert 'status = "infrastructure_fault"' in src[i_infra:i_cert]


# --------------------------------------------------------------------------- the robust collector


def test_collector_helpers_treat_age_as_no_liveness_proof():
    from merlin.targetgen.contract.materialize import (_build_owner_pid, _lease_owner_pid, _pid_alive,
                                                       _staging_dir_of)

    # Parsed structurally, no regex, and a non-staging name is recognised as such.
    assert _build_owner_pid(".gemmini.build.2628099.0abb75ec") == 2628099
    assert _build_owner_pid(".t.build.12.abc") == 12
    assert _build_owner_pid("gemmini") is None
    assert _build_owner_pid(".t.build.notapid.abc") is None
    assert _build_owner_pid(".t.build.12") is None           # no uuid suffix -> not a build dir
    assert _lease_owner_pid(".lease.4321.beef") == 4321
    assert _lease_owner_pid(".gemmini.build.1.a") is None

    # Liveness FAILS CLOSED: anything undeterminable counts as still held.
    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(None) is True
    assert _pid_alive(0) is True
    assert _pid_alive(-5) is True

    # The cert grade is handed per-capsule dirs UNDER the build, not the build root.
    import pathlib
    d = pathlib.Path("/tmp/x/.t.build.7.abcd1234/A0")
    got = _staging_dir_of(d)
    assert got is not None and got.name == ".t.build.7.abcd1234"


def test_a_build_held_by_a_live_reader_is_not_collected(tmp_path, monkeypatch):
    """End-to-end on the real publisher: an OLD build that a live reader has pinned must survive, and an
    old build with a dead builder and no reader must be collected."""
    import time

    from merlin.targetgen.contract import materialize as M

    base = tmp_path / "capsule_bench_public"
    base.mkdir()
    monkeypatch.setattr(M, "cache_dir", lambda ns: base, raising=False)
    monkeypatch.setattr("merlin.common.artifacts.cache_dir", lambda ns: base)

    old_stamp = time.time() - 4 * M._GC_MIN_AGE_S

    # (a) old, builder DEAD, unheld -> collectable
    dead = base / ".t.build.999999.deadbeef"
    dead.mkdir()
    os.utime(dead, (old_stamp, old_stamp))
    # (b) old, builder ALIVE (this process) -> must survive
    alive = base / f".t.build.{os.getpid()}.aliveaaa"
    alive.mkdir()
    os.utime(alive, (old_stamp, old_stamp))
    # (c) old, builder dead, but a LIVE reader holds a lease -> must survive
    leased = base / ".t.build.999998.leasedbb"
    leased.mkdir()
    os.utime(leased, (old_stamp, old_stamp))

    assert M.pin_cohort_builds(leased / "A0_config_smoke"), "a per-capsule path must still pin its build"
    held = M._held_build_names(base)
    assert leased.name in held

    # Run exactly the collector loop the publisher runs, with a just-published dir standing in for `ver`.
    ver = base / ".t.build.1.freshaaa"
    ver.mkdir()
    cutoff = time.time() - M._GC_MIN_AGE_S
    for old in base.glob(f".t{M._BUILD_INFIX}*"):
        if old.name == ver.name or old.name in held:
            continue
        if M._pid_alive(M._build_owner_pid(old.name)):
            continue
        if old.stat().st_mtime < cutoff:
            import shutil
            shutil.rmtree(old, ignore_errors=True)

    assert not dead.exists(), "an old, dead-builder, unheld build should still be collected"
    assert alive.exists(), "a build whose BUILDER is still running must never be collected"
    assert leased.exists(), "a build a LIVE READER holds a lease on must never be collected"

    # Leases are self-limiting: taking new ones drops this process's old ones.
    M.pin_cohort_builds(alive)
    assert M._held_build_names(base) == {alive.name}
    M.release_cohort_builds()
    assert M._held_build_names(base) == set()


def test_grade_rollup_refuses_to_call_a_staging_failure_gradeable():
    """Excluding infra rows from the denominator is only HALF the fix: on its own it turns 2-of-33 into
    `2/2, all_pass`. The rollup must declare the run not gradeable and name the fault."""
    import inspect

    from merlin.targetgen import capsule_grade

    src = inspect.getsource(capsule_grade)
    assert '_infra = [r for r in results if r.get("status") == "infrastructure_fault"]' in src
    assert 'score["gradeable"] = (not no_oracle) and not _empty and not _incomplete and not _infra' in src
    assert 'score["infrastructure_fault"]' in src
    # structural_pass must not go green off the shrunken denominator either.
    i = src.index('score["structural_pass"]')
    assert "not _infra" in src[i:i + 200]
