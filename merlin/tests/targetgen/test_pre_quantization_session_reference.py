"""A prequantized paper-ready stage gets its quality trajectory from the untransformed program."""

import random
import sys
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "src/merlin/targetgen"))
from _capture_session_reference import pre_quantization_session_reference  # noqa: E402


@pytest.fixture
def trajectory(monkeypatch):
    calls = []

    def capture_session_trajectory(module, inputs, session):
        calls.append((module, inputs, session))
        torch.rand(3)  # a forward that draws random numbers must not perturb the later capture
        np.random.rand()
        random.random()
        return np.arange(4, dtype=np.float32)

    bundle = ModuleType("m2m.capture.bundle")
    bundle.capture_session_trajectory = capture_session_trajectory
    for name in ("m2m", "m2m.capture"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or ModuleType(name))
    monkeypatch.setitem(sys.modules, "m2m.capture.bundle", bundle)
    return calls


def test_paper_ready_session_receives_an_independent_reference(trajectory):
    module, inputs = torch.nn.Identity(), (torch.zeros(2),)
    session = {"paper_ready": True, "quality": {"output_index": 0}, "steps": 2}
    states = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state())
    result = pre_quantization_session_reference(module, inputs, session)
    assert trajectory[0][0] is module and trajectory[0][1] == inputs
    np.testing.assert_array_equal(result["quality"]["reference_values"], np.arange(4, dtype=np.float32))
    assert result["quality"]["reference"] == "eager_fp32" and result["quality"]["output_index"] == 0
    assert "reference_values" not in session["quality"]  # the loader's session is not mutated
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1])
    assert torch.equal(torch.get_rng_state(), states[2])


@pytest.mark.parametrize(
    "session",
    [
        None,
        {"paper_ready": False, "quality": {}},
        {"paper_ready": True, "quality": {"reference_values": np.ones(2, dtype=np.float32)}},
    ],
)
def test_other_sessions_are_returned_unchanged(trajectory, session):
    assert pre_quantization_session_reference(torch.nn.Identity(), (), session) is session
    assert trajectory == []
