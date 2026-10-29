"""Trusted epoch stops defer new commands; entered custody still completes."""
import pytest

from test_compiler_profile_concurrency import original
from merlin.common import bundle_snapshot as B
from merlin.targetgen import package_runtime as P
from merlin.targetgen.capsule_scheduling import observe_admission
from merlin_experiments.phase1 import joint_control
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R


def test_checkpoint_defers_before_any_new_snapshot_verification(original, monkeypatch):
    owner, candidate, work = original
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        @R.snapshot_scoped_entrypoint
        def command(self):
            pytest.fail('checkpoint admitted a new command')
        monkeypatch.setattr(joint_control, 'stop_reason', lambda: 'library epoch checkpoint')
        with observe_admission(lambda reason: None), monkeypatch.context() as patch:
            patch.setattr(B, 'verified_snapshot_scope', lambda *a, **k: pytest.fail('new verification started'))
            with pytest.raises(P.NotMeasuredFailure) as caught:
                command(executor)
        assert caught.value.status == 'gated'
        assert caught.value.facts['correctness'] == 'unknown'
        assert caught.value.facts['execution_status'] == 'deferred_before_command'


def test_ordinary_edits_and_final_checker_are_not_cancelled(monkeypatch):
    monkeypatch.setattr(joint_control, 'stop_reason', lambda: None)
    with observe_admission(lambda reason: None, stop_requested=lambda: False):
        R._checkpoint_command_admission()
    # Explicit checkpoint exists, but final checker has no background owner.
    monkeypatch.setattr(joint_control, 'stop_reason', lambda: 'final drain')
    R._checkpoint_command_admission()


def test_checkpoint_during_command_preserves_its_final_payload_check(original, monkeypatch):
    owner, candidate, work = original
    stopped, reads = [False], []
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        native = B._payload_content
        def read(*args, **kwargs):
            reads.append(stopped[0])
            return native(*args, **kwargs)
        monkeypatch.setattr(B, '_payload_content', read)
        monkeypatch.setattr(joint_control, 'stop_reason', lambda: 'library epoch' if stopped[0] else None)
        @R.snapshot_scoped_entrypoint
        def command(self):
            stopped[0] = True
            return 'retained output'
        with observe_admission(lambda reason: None):
            assert command(executor) == 'retained output'
        assert reads == [True]  # entered shared custody's independent final read


def test_checkpoint_defers_outer_guard_refresh_before_verification(original, monkeypatch):
    owner, candidate, work = original
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        monkeypatch.setattr(joint_control, 'stop_reason', lambda: 'library epoch checkpoint')
        monkeypatch.setattr(B, 'shared_entry_verification', lambda *a, **k: pytest.fail('refresh started'))
        with observe_admission(lambda reason: None), pytest.raises(P.NotMeasuredFailure):
            with executor.refreshed_operation_snapshot_v1(owner):
                pytest.fail('checkpoint admitted outer operation')
