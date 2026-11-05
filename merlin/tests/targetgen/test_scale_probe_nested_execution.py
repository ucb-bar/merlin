"""Nested public-scale dispatch preserves the original formal compiler owner."""

import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase1.feedback import scale_probe_child as S
from merlin_experiments.phase2.contracts import StageGateError
from test_scale_probe_batch import batch_setup as batch_setup_fixture
from test_scale_probe_batch import emit
from test_scale_probe_process import emit as emit_single
from test_scale_probe_process import original  # noqa: F401 -- shared exact-owner fixture
from test_scale_probe_process import setup as setup_fixture

from merlin.targetgen import package_runtime as P

batch_setup = batch_setup_fixture
setup = setup_fixture


def _scope(case):
    return R.original_compiler_execution(
        case.owner, target_experiment=case.target, candidate=case.candidate,
        work_root=case.private / "formal-executor", sandbox_inputs=case.inputs,
    )


def _nested(case, **changes):
    arguments = dict(
        owner=case.owner, candidate=case.candidate, target_experiment=case.target,
        work_root=case.private / "unused-nested-owner", sandbox_inputs=case.inputs,
    )
    arguments.update(changes)
    return S._execution_lifetime(**arguments)


def test_formal_scope_reuses_equivalent_owner_and_dispatches_all_original_batch_cases(batch_setup, monkeypatch):
    case = batch_setup
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    equivalent = CommonTransformService(
        workspace=case.owner.workspace, bundle_manifest=case.owner.bundle_manifest, repo=case.owner.repo,
    )
    with _scope(case):
        executor = P.active_package_executor()
        case.owner = equivalent
        rows = emit(case)
        assert P.active_package_executor() is executor
        assert executor._snapshot_entry is not None
        assert not (case.output / "child-owner").exists()
        assert len(rows) == len(case.cases) == 4
        assert all(row["lowering_status"] == "emitted" for row in rows)
        assert all(row["numeric"] == row["rtl"] == "not_attempted" for row in rows)
    assert P.active_package_executor() is None
    assert executor._snapshot_entry is None


def test_nonbatch_dispatch_also_preserves_the_outer_original_executor(setup, monkeypatch):
    monkeypatch.delenv("MERLIN_PUBLIC_SCALE_BATCH_WORKERS", raising=False)
    with _scope(setup):
        executor = P.active_package_executor()
        rows = emit_single(setup)
        assert P.active_package_executor() is executor
        assert not (setup.output / "child-owner").exists()
        assert len(rows) == len(setup.cases) == 2
        assert all(row["lowering_status"] == "emitted" for row in rows)
    assert P.active_package_executor() is None


@pytest.mark.parametrize("changed", ["owner", "target", "candidate", "source", "inputs"])
def test_nested_scope_rejects_changed_selection_before_case_creation(batch_setup, changed):
    case = batch_setup
    with _scope(case):
        executor = P.active_package_executor()
        arguments = {}
        if changed == "owner":
            archive = case.private / "other_manifest.yaml"
            archive.write_bytes(case.owner.bundle_manifest.read_bytes())
            arguments["owner"] = CommonTransformService(
                workspace=case.owner.workspace, bundle_manifest=archive, repo=case.owner.repo,
            )
        elif changed == "target":
            arguments["target_experiment"] = replace(case.target, target="other_target")
        elif changed == "candidate":
            copy = case.private / "same-bytes-other-owner"
            shutil.copytree(case.candidate, copy)
            arguments["candidate"] = copy
        elif changed == "source":
            (case.candidate / "tool.py").write_text("# another compiler revision\n")
        else:
            arguments["sandbox_inputs"] = replace(case.inputs, harness="another harness")
        with pytest.raises((ValueError, StageGateError), match="selection|source revision"):
            with _nested(case, **arguments):
                pytest.fail("changed original selection was accepted")
        assert P.active_package_executor() is executor
        assert not (case.private / "unused-nested-owner").exists()
        assert not list(case.output.glob("case-*"))


def test_foreign_executor_is_never_dropped_or_replaced(batch_setup):
    foreign = SimpleNamespace(build_package=lambda *_: None, run_entrypoint=lambda *_: None)
    with P.scoped_package_executor(foreign):
        with pytest.raises(StageGateError, match="foreign or incomplete"):
            with _nested(batch_setup):
                pytest.fail("foreign executor was accepted")
        assert P.active_package_executor() is foreign
    assert not (batch_setup.private / "unused-nested-owner").exists()


def test_no_executor_keeps_fresh_lifetime_even_with_an_existing_resource_boundary(batch_setup, monkeypatch):
    from merlin_experiments.phase1.feedback import compile_only as C
    from merlin_experiments.phase2 import external_resources as resources

    boundary = object()
    monkeypatch.setattr(resources, "active_resource_boundary", lambda: boundary)
    with C._snapshot_operation(batch_setup.owner):
        with _nested(batch_setup):
            executor = P.active_package_executor()
            assert type(executor) is R._Executor
            assert executor.candidate == batch_setup.candidate
            assert resources.active_resource_boundary() is boundary
    assert P.active_package_executor() is None
    assert list((batch_setup.private / "unused-nested-owner").iterdir()) == []


def test_nested_scope_rechecks_source_before_returning_to_formal(batch_setup):
    with _scope(batch_setup):
        executor = P.active_package_executor()
        with pytest.raises(ValueError, match="source revision"):
            with _nested(batch_setup):
                (batch_setup.candidate / "tool.py").write_text("# changed while dispatching\n")
        assert P.active_package_executor() is executor


def test_nested_scope_preserves_default_sandbox_selection(batch_setup):
    case = batch_setup
    with R.original_compiler_execution(
        case.owner, target_experiment=case.target, candidate=case.candidate,
        work_root=case.private / "formal-default",
    ):
        executor = P.active_package_executor()
        with _nested(case, sandbox_inputs=None):
            assert P.active_package_executor() is executor
        with pytest.raises(StageGateError, match="selection"):
            with _nested(case):
                pytest.fail("explicit inputs replaced the selected default")
