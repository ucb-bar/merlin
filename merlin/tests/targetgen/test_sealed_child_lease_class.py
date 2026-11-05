"""A context-only lease class reaches real sealed scale-probe children without changing their environment.

The in-turn grader selects its class with ``cpu_priority`` alone. Each child's complete environment is
sealed before launch, so the class must already be part of it: a launcher ``env`` prefix would hand the
child another mapping and it would refuse its provenance.
"""

import contextlib
import subprocess

import pytest
import test_scale_probe_batch as SB
import test_scale_probe_process as SP
from merlin_experiments.phase1.feedback import scale_probe_child as S
from test_compiler_profile_child import original as original_fixture

from merlin.common import cpu_pool
from merlin.common import invocation_record as I

original = original_fixture
setup = SP.setup
batch_setup = SB.batch_setup


@pytest.fixture
def observed(monkeypatch):
    for key in (cpu_pool.PRIORITY_ENV, "MERLIN_PREPARATION_CPUS", "MERLIN_GRADE_CPUS"):
        monkeypatch.delenv(key, raising=False)
    sealed, launched = [], []
    prepare = S.CHILD.prepare_model_children

    @contextlib.contextmanager
    def recorded_seal(**kwargs):
        with prepare(**kwargs) as transports:
            sealed.extend(transport.document["environment"] for transport in transports)
            yield transports

    monkeypatch.setattr(S.CHILD, "prepare_model_children", recorded_seal)

    def install():
        launch = subprocess.Popen  # the fixture's control launcher

        def recorded_launch(command, **kwargs):
            if "--spec" in command:
                launched.append((list(command), dict(kwargs["env"])))
            return launch(command, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", recorded_launch)

    return sealed, launched, install


def _verify(sealed, launched, count):
    assert len(launched) == len(sealed) == count
    for command, environment in launched:
        assert command[0] == "taskset" and "env" not in command
        assert environment[cpu_pool.PRIORITY_ENV] == str(cpu_pool.GRADE)
    assert sealed == [I.environment_identity(environment) for _, environment in launched]


def test_scale_children_restore_their_seal_under_a_context_only_grade_class(setup, observed):
    sealed, launched, install = observed
    install()
    with cpu_pool.cpu_priority(cpu_pool.GRADE):
        rows = SP.emit(setup)
    assert [row["lowering_status"] for row in rows] == ["emitted"] * 2
    _verify(sealed, launched, 2)


def test_scale_batch_workers_restore_their_seal_under_a_context_only_grade_class(batch_setup, observed):
    sealed, launched, install = observed
    install()
    with cpu_pool.cpu_priority(cpu_pool.GRADE):
        rows = SB.emit(batch_setup)
    assert [row["lowering_status"] for row in rows] == ["emitted"] * 4
    _verify(sealed, launched, 2)
