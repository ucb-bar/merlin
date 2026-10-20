"""A parameter-free program keeps its real empty payload and host-only route."""

from copy import deepcopy

import pytest

from merlin.model_check.chain_image import _compiler_binding
from merlin.model_check.chain_image_layout import layout, linker_script


def test_empty_weights_are_pinned_but_have_no_live_interval_or_linker_reservation(tmp_path):
    root = tmp_path / "program"
    (root / "cgen").mkdir(parents=True)
    (root / "cgen/weights.bin").write_bytes(b"")
    selected = {"layout": {"weights": [0], "inputs": 64, "references": 128, "arena": {"begin": 1024, "bytes": 1024**2}}}
    plan = {
        "programs": [{"production": {"root": root, "receipt": {"inputs": {"arena_mb": 1}}}}],
        "input_blob": {"bytes": 8},
    }
    result = layout(selected, plan, {"blob": {"bytes": 8}})
    assert result["weights"] == [{"begin": 0, "bytes": 0}]
    assert result["source_pins"][0]["bytes"] == 0
    original = tmp_path / "original.ld"
    original.write_text(
        "SECTIONS {\n  .text : { *(.text*) }\n  . = MERLIN_WEIGHTS_BASE;\n .weights : { *(.data*) }\n}\n"
    )
    output = linker_script(original, tmp_path / "chain.ld", result).read_text()
    assert ".chain_weights_0" not in output
    assert ".chain_inputs" in output and ".chain_references" in output
    selected["layout"]["weights"] = [-64]
    with pytest.raises(ValueError):
        layout(selected, plan, {"blob": {"bytes": 8}})


def test_host_only_compiler_binding_is_explicit_and_does_not_claim_a_candidate_invocation():
    receipt = {
        "execution_route": "host_baseline",
        "inputs": {"device": None},
        "output": {"device_artifacts": None, "matrix_routing": None},
    }
    program = {"production": {"receipt": receipt}}
    assert _compiler_binding(program, "f" * 64) == {"execution_route": "host_baseline", "device_compiler_sha256": None}
    for mutation in [
        {"execution_route": "unknown"},
        {"output": {"device_artifacts": {"present": True}}},
        {"output": {"matrix_routing": {"present": True}}},
    ]:
        changed = {**deepcopy(receipt), **mutation}
        with pytest.raises(ValueError):
            _compiler_binding({"production": {"receipt": changed}}, "f" * 64)
    receipt["inputs"]["device"] = {"package_tree": {"sha256": "b" * 64}}
    with pytest.raises(ValueError):
        _compiler_binding(program, "f" * 64)
