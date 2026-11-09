"""Dynamic multi-program quantization owns one actual shared TorchAO transform."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from merlin.targetgen import _m2m_capture_worker as worker

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
_external = pytest.importorskip("m2m.capture.external_runtime")
verify_bundle_integrity = pytest.importorskip("m2m.capture.bundle_integrity").verify_bundle_integrity
ExternalRuntimeProgram = _external.ExternalRuntimeProgram
ExternalRuntimeSharedModel = _external.ExternalRuntimeSharedModel
make_external_runtime_session = _external.make_external_runtime_session


class Shared(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)

    def forward(self, value):
        return self.linear(value)


class Program(torch.nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def forward(self, value, state):
        result = self.shared(value) + state
        return result, result


class Capture:
    def __init__(self):
        self.model = Shared().eval()
        self.selected = []

    def external_runtime_shared_model(self):
        def factory(selected):
            self.selected.append(selected)
            first = Program(selected).eval()
            value = torch.arange(32, dtype=torch.float32).reshape(1, 32) / 16
            state = torch.zeros(1, 32)
            child = {
                "kind": "synthetic_recurrent",
                "paper_ready": False,
                "steps": 2,
                "stages": ["step"],
                "stage_schedule": [{"name": "step", "steps": 2, "execution": "compiled_recurrent", "timed": True}],
                "states": [{"name": "state", "input_index": 1, "output_index": 1}],
                "streams": [],
                "quality": {"output_index": 0},
            }
            root = {
                "kind": "synthetic_recurrent",
                "paper_ready": False,
                "stages": ["first", "step"],
                "quality_program": "step",
                "states": ["state"],
                "stage_schedule": [
                    {"name": "first", "steps": 1, "execution": "compiled", "timed": True},
                    {"name": "step", "steps": 2, "execution": "compiled_recurrent", "timed": True},
                ],
                "bindings": [
                    {
                        "name": "state",
                        "from": {"program": "first", "output_index": 1},
                        "to": {"program": "step", "input_index": 1},
                    }
                ],
                "provenance": {"source_kind": "authored_test"},
            }
            with torch.no_grad():
                _, carried = first(value, state)
            return make_external_runtime_session(
                version=2,
                programs=(
                    ExternalRuntimeProgram("first", first, (value, state), 1),
                    ExternalRuntimeProgram("step", Program(selected).eval(), (value / 2, carried), 2, child),
                ),
                metadata=root,
            )

        return ExternalRuntimeSharedModel(self.model, factory, ("first", "step"))


def arguments(tmp_path):
    return argparse.Namespace(
        dtype="int8",
        scheme="int8_dyn_act_int8_weight",
        materialize_bundle=True,
        recipe="",
        already_quantized=False,
        stage_fp32=False,
        quantize_activation_contractions=False,
        integer_nonlinear=False,
        seed=0,
        loader=str(Path(__file__)),
    )


@pytest.mark.parametrize("detached_globals", [False, True])
def test_actual_shared_dynamic_writer_quantizes_once_but_refuses_missing_source_lineage(
    tmp_path, monkeypatch, detached_globals
):
    import m2m.capture.torchao_pipeline as pipeline

    torch.manual_seed(0)
    capture = Capture()
    calls = []
    original = pipeline.apply_quantization

    def observed(model, quant):
        calls.append(model)
        return original(model, quant)

    monkeypatch.setattr(pipeline, "apply_quantization", observed)
    out = tmp_path / "capture"
    out.mkdir()
    entry = worker._materialize_shared_dynamic_session
    if detached_globals:
        # runpy's globals need not be registered in sys.modules. Services must
        # still refer to this selected worker, not a process __main__ module.
        entry = runpy.run_path(worker.__file__, run_name="_detached_capture_worker")[
            "_materialize_shared_dynamic_session"
        ]
    rc = entry(capture, arguments(tmp_path), out, determinism={"seed": 0}, dependencies=[], torch=torch)
    # The public frontend does not yet prove eager TorchAO original-to-quantized
    # lineage. Typed integer math and goldens cannot substitute for that proof.
    assert rc == 3
    assert calls == [capture.model]
    assert len(capture.selected) == 2 and all(model is capture.model for model in capture.selected)
    verify_bundle_integrity(out)
    import yaml

    contract = yaml.safe_load((out / "session_contract.yaml").read_text())
    binding = json.loads((out / "quantized-session-binding.json").read_text())
    assert (
        contract["quantized_session_binding"]["sha256"]
        == hashlib.sha256((out / "quantized-session-binding.json").read_bytes()).hexdigest()
    )
    assert binding["quantization"]["shared_transform_calls"] == 1
    assert any(row.get("dtype") == "torch.int8" for row in binding["shared_tensors"])
    for name in ["first", "step"]:
        stage = out / "stages" / name
        meta = json.loads((stage / "meta.json").read_text())
        assert meta["scheme"] == "int8_dyn_act_int8_weight"
        assert meta["captured_integer_mm_roots"] == 1
        trace = json.loads((stage / "frontend-trace.json").read_text())
        assert not meta["ok"] and trace["status"] == "diagnostic"
        assert "original -> quantized correspondence incomplete" in trace["blockers"]
        assert len(meta["output_abi"]) == 2
        assert set(np.load(stage / "goldens.npz").files) == {"Y0", "Y1"}
        assert set(np.load(stage / "original-source-outputs.npz").files) == {"out0", "out1"}
        assert (
            binding["stage_receipts"][name] == hashlib.sha256((stage / "capture_receipt.json").read_bytes()).hexdigest()
        )


@pytest.mark.parametrize(
    "flag", ["already_quantized", "stage_fp32", "quantize_activation_contractions", "integer_nonlinear"]
)
def test_extra_numerical_transforms_are_refused_before_model_mutation(tmp_path, flag):
    args = arguments(tmp_path)
    setattr(args, flag, True)
    capture = Capture()
    with pytest.raises(ValueError, match="only the explicit dynamic W8A8"):
        worker._materialize_shared_dynamic_session(
            capture, args, tmp_path, determinism={}, dependencies=[], torch=torch
        )
    assert capture.selected == []


def test_a_missing_explicit_shared_capability_is_not_inferred(tmp_path):
    with pytest.raises(TypeError, match="requires external_runtime_shared_model"):
        worker._materialize_shared_dynamic_session(
            torch.nn.Linear(32, 32), arguments(tmp_path), tmp_path, determinism={}, dependencies=[], torch=torch
        )
