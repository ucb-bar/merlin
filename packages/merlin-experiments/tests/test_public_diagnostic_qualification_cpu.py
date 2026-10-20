"""Real CPU handoff with original synthetic qualification owners; no target work."""

import json
import os
import sys
from pathlib import Path

import pytest
import test_public_source_diagnostic as PUBLIC
from merlin_experiments.phase2 import capsule_ordinary_scope as C
from merlin_experiments.phase2 import public_source_diagnostic as D
from merlin_experiments.phase2.contracts import StageGateError
from test_capsule_certification import selected as selected
from test_capsule_ordinary_scope import scoped as scoped
from test_capsule_qualification_handoff import qualified as qualified

from merlin.common import owned_process
from merlin.common.cpu_pool import current_thread_owns_cpu_lease, shared_compute_pool


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "public-policy"
    root.mkdir()
    return PUBLIC.inputs.__wrapped__(root, monkeypatch)


def test_original_scope_replays_sequentially_after_single_cpu_metadata_leaf(inputs, scoped, monkeypatch):
    # This child is an unpaid CPU-custody probe, followed by the unchanged real
    # synthetic qualification verifier. It supplies no qualification verdict.
    with shared_compute_pool(timeout_s=5):
        cpu = next(iter(os.sched_getaffinity(0)))
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_PREPARATION_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_GRADE_CPUS", str(cpu))
    code = (
        "import json,os;from merlin.common.cpu_pool import shared_compute_pool;"
        "\nwith shared_compute_pool(timeout_s=.25):"
        "\n print(json.dumps(sorted(os.sched_getaffinity(0))))"
    )
    # Use the actual imported core, independent of ambient namespace ordering.
    import merlin.common.cpu_pool as CPU

    source_root = str(Path(CPU.__file__).resolve().parents[2])
    command = [sys.executable, "-I", "-S", "-c", "import sys;sys.path.insert(0," + repr(source_root) + ");" + code]
    original = C._verify_original
    replay_cpus = []

    def replay(*args):
        result = owned_process.run(command, timeout=3, text=True)
        if result.returncode:
            raise RuntimeError("qualification CPU probe could not acquire its assigned slot: " + result.stderr)
        replay_cpus.append(json.loads(result.stdout))
        return original(*args)

    monkeypatch.setattr(C, "_verify_original", replay)
    # Reproduce the original scheduling inversion with a short real slot wait.
    with shared_compute_pool(timeout_s=5):
        with pytest.raises(RuntimeError, match="could not acquire"):
            C.verify(scoped.scope)
    assert not replay_cpus

    args, *_ = inputs
    policy = D.prepare_public_source_diagnostic(**args, capsule_development_scope=scoped.scope)
    initial = len(replay_cpus)
    expected = policy.binding()
    assert len(replay_cpus) == initial + 1
    assert policy.binding() == expected
    assert len(replay_cpus) == initial + 2
    assert all(cpus == [cpu] for cpus in replay_cpus)
    assert not current_thread_owns_cpu_lease()
    assert expected["capsule_development"] == C.verify(scoped.scope)
    assert expected["formal_admission"] is False and expected["promotion_authority"] == "NONE"

    with shared_compute_pool(timeout_s=5):
        with pytest.raises(StageGateError, match="parent metadata CPU lease"):
            policy.validate_search_inputs()
    assert len(replay_cpus) == initial + 3
    with pytest.raises(StageGateError, match="local metadata"):
        policy._binding_metadata()


@pytest.mark.parametrize("fault", ["scope", "qualification", "source"])
def test_scoped_binding_keeps_original_owner_and_mutation_refusals(inputs, scoped, monkeypatch, fault):
    args, *_ = inputs
    policy = D.prepare_public_source_diagnostic(**args, capsule_development_scope=scoped.scope)
    if fault == "scope":
        policy._capsule_development_scope = object()
    elif fault == "qualification":
        path = scoped.owner.qualification_path
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b" ")
    else:
        original = D._source
        monkeypatch.setattr(
            D,
            "_source",
            lambda path: {**original(path), "sha256": "0" * 64}
            if Path(path) == Path(C.__file__)
            else original(path),
        )
    with pytest.raises((StageGateError, ValueError)):
        policy.validate_search_inputs()
