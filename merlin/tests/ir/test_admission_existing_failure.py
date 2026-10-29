"""Prior worker failures keep their verdict and evidence; no proof is simulated."""

from types import SimpleNamespace

import pytest

from merlin.verify import pass_validation
from merlin.verify.speculative_gate import GateFailure


@pytest.mark.parametrize("verdict", ["not_established", "unsupported", "unavailable"])
def test_source_admission_preserves_existing_worker_failure(tmp_path, verdict):
    failure = GateFailure(
        dict(
            check="compiler_loop_refinement",
            label="previous actual worker",
            verdict=verdict,
            reason="existing worker failure",
            directory=str(tmp_path / "original-evidence"),
        )
    )

    def prior_failure(raw):
        raise failure

    validation = SimpleNamespace(gate=SimpleNamespace(directory=tmp_path / "current"))
    token = pass_validation._ACTIVE.set(validation)
    try:
        with pytest.raises(GateFailure) as caught:
            pass_validation.source_admission("ordered_loop_unrolling", "source", predicate=prior_failure)
    finally:
        pass_validation._ACTIVE.reset(token)
    assert caught.value is failure
    assert caught.value.report["verdict"] == verdict
    assert caught.value.report["directory"] == str(tmp_path / "original-evidence")
