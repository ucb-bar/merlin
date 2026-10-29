"""Source reconstruction host controls; substituted frontends grant no native verdict."""

import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from merlin.verify.compiler_qdq_source import reconstruct


@pytest.mark.parametrize("mutation", [None, "argument", "return", "result_type", "producer_identity"])
def test_original_program_must_match_complete_recorded_graph(tmp_path, monkeypatch, mutation):
    from merlin.targetgen import qdq_capture, qdq_capture_identity

    def install(name):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, leaf = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], leaf, module)
        return module

    torch = install("torch")
    for name in (
        "torch.ao",
        "torch.ao.quantization",
        "torch.ao.quantization.fx",
        "torch.ao.quantization.fx._decomposed",
    ):
        install(name)
    for name in ("m2m", "m2m.capture"):
        install(name)
    trace_module = install("m2m.capture.trace")
    graph = {
        "nodes": [
            {
                "id": "x",
                "op": "placeholder",
                "target": "x",
                "args": [],
                "kwargs": {},
                "results": [{"dtype": "float32"}],
            },
            {
                "id": "q",
                "op": "call_function",
                "target": "quantize",
                "args": ["x", 0.125],
                "kwargs": {},
                "results": [{"dtype": "int8"}],
            },
            {"id": "out", "op": "output", "target": "output", "args": [["q"]], "kwargs": {}, "results": []},
        ]
    }
    original = copy.deepcopy(graph)
    if mutation == "argument":
        original["nodes"][1]["args"][1] = 0.25
    elif mutation == "return":
        original["nodes"][2]["args"] = [["x"]]
    elif mutation == "result_type":
        original["nodes"][1]["results"] = [{"dtype": "float32"}]
    torch.export = SimpleNamespace(load=lambda path: original)
    trace_module.snapshot_exported_program = lambda program, stage: program
    identity = {"producer": {"commit": "unit-source"}, "torch": {"wheel_sha256": "unit-wheel"}}
    (tmp_path / "qdq-producer.json").write_text(json.dumps(identity))
    (tmp_path / "qdq-frontend.json").write_text(json.dumps({"graphs": {"original": graph}}))
    monkeypatch.setattr(
        qdq_capture_identity, "begin", lambda *args: {} if mutation == "producer_identity" else identity
    )
    rechecks = []
    monkeypatch.setattr(qdq_capture_identity, "recheck", lambda record: rechecks.append(record))
    replays = []

    def replay(directory):
        replays.append(directory)
        return {"status": "matched", "sites": [{"unit_control": True}]}

    monkeypatch.setattr(qdq_capture, "replay", replay)
    selection = {
        "directory": str(tmp_path),
        "producer": str(tmp_path / "producer"),
        "torch_wheel": str(tmp_path / "wheel"),
    }
    if mutation is not None:
        with pytest.raises(ValueError, match="differs|differ"):
            reconstruct(selection)
        assert replays == rechecks == []
    else:
        result = reconstruct(selection)
        assert replays == [tmp_path] and rechecks == [identity]
        assert result["original_program_bound"] is True
        assert result["native_execution_replayed"] is False and result["source_closure_verified"] is False
