"""Original empty input projections stay distinct from missing declared reads.

These source-first binding controls use tiny standard tensor constants and the
ordinary provenance reader. They grant no compiler, stage, or runtime authority.
The two repeated source returns remain two complete, named output slots.
"""

import copy

import pytest

from merlin.targetgen import capsule_golden as golden
from merlin.targetgen import native_component_execution as native
from merlin.targetgen.golden_store import write_golden


def _original(tmp_path, *, dtype="f32", read=False, provenance=None):
    values = [1.25, -2.0] if dtype == "f32" else [-7, 13]
    tensor = f"tensor<2x{dtype}>"
    source = tmp_path / "original.mlir"
    argument = f"%input: {tensor}" if read else ""
    source.write_text(
        f"module {{ func.func @original({argument}) -> ({tensor}, {tensor}) {{\n"
        f"  %value = arith.constant dense<{values}> : {tensor}\n"
        f"  func.return %value, %value : {tensor}, {tensor}\n"
        "} }\n"
    )
    outputs = [dict(name=name, role="output", shape=[2], dtype=dtype) for name in ("z", "a")]
    inputs = [dict(name="input", role="input", shape=[2], dtype=dtype)] if read else []
    capsule = dict(
        kind="model",
        inputs=inputs + outputs,
        operation={"op": "model"},
        numeric_policy={"compare": "tolerance_float", "atol": 0.0, "rtol": 0.0}
        if dtype == "f32"
        else {"compare": "exact_int"},
        __dir__=str(tmp_path),
    )
    write_golden(
        tmp_path,
        {
            "golden_source": "private_original_constant",
            "outputs": {row["name"]: values for row in outputs},
            "oracle_provenance": {"inputs": provenance or {}},
        },
    )
    leaves = ["ptr_input"] if read else []
    tensors = {
        **({"ptr_input": dict(shape=[2], dtype=dtype, role="input")} if read else {}),
        **{name: dict(shape=[2], dtype=dtype, role="output") for name in ("ptr_z", "ptr_a")},
    }
    cb = dict(
        abi_version=1,
        target="private_input_projection",
        commands=[],
        operand_naming="positional",
        tensors=tensors,
        params={"global_program_plan": {"entry_bindings": leaves}},
        kernel_abi={
            "kind": "whole_program",
            "outputs": ["ptr_z", "ptr_a"],
            "args": [dict(tensor=name, access="read") for name in leaves]
            + [dict(tensor=name, access="write") for name in ("ptr_z", "ptr_a")],
        },
    )
    return capsule, cb, source


@pytest.mark.parametrize("dtype", ["f32", "i16"])
def test_zero_argument_source_keeps_empty_original_projection_and_all_returns(tmp_path, monkeypatch, dtype):
    capsule, cb, source = _original(tmp_path, dtype=dtype)
    assert golden.canonical_input_values(capsule, tmp_path) == {}
    assert golden.canonical_input_raws(capsule, tmp_path) == {}

    def unselected(_capsule):
        pytest.fail("an explicitly empty source input roster reached legacy stimulus materialization")

    monkeypatch.setattr(golden, "materialized_input_values", unselected)
    before = copy.deepcopy(cb)
    bound, projected, bindings = native._bind(capsule, cb, source)
    assert cb == before and bound == before and bound is not cb
    assert projected == {}
    assert bindings == {"inputs": [], "outputs": {"z": "ptr_z", "a": "ptr_a"}}
    assert tuple(bindings["outputs"]) == ("z", "a")
    assert len(bound["kernel_abi"]["outputs"]) == 2


def test_declared_read_still_requires_complete_independent_projection(tmp_path):
    capsule, cb, source = _original(tmp_path, read=True)
    with pytest.raises(native.NativeComponentExecutionError, match="no complete selected input projection"):
        native._bind(capsule, cb, source)


def test_actual_declared_read_projection_is_bound_without_a_fallback(tmp_path):
    projection = {"input": {"shape": [2], "dtype": "f32", "decoded": [3.0, -4.0]}}
    capsule, cb, source = _original(tmp_path, read=True, provenance=projection)
    bound, projected, bindings = native._bind(capsule, cb, source)
    assert projected == {"ptr_input": {"shape": [2], "values": [3.0, -4.0]}}
    assert bindings["inputs"][0]["source"] == "input"
    assert bindings["inputs"][0]["harness"] == "ptr_input"
    assert "preload_b64" in bound["tensors"]["ptr_input"]


def test_zero_argument_source_refuses_an_unexpected_recorded_operand(tmp_path):
    projection = {"extra": {"shape": [2], "dtype": "f32", "decoded": [3.0, -4.0]}}
    capsule, cb, source = _original(tmp_path, provenance=projection)
    with pytest.raises(native.NativeComponentExecutionError, match="canonical input roster is incomplete"):
        native._bind(capsule, cb, source)


def test_nonempty_integer_source_retains_its_declared_stimulus(tmp_path):
    capsule, cb, source = _original(tmp_path, dtype="i16", read=True)
    bound, projected, bindings = native._bind(capsule, cb, source)
    assert projected == {
        "ptr_input": golden.materialized_input_values(capsule)["input"],
    }
    assert bindings["inputs"][0]["raw_sha256"] is None
    assert "preload_b64" not in bound["tensors"]["ptr_input"]


def test_partial_declared_read_projection_still_refuses(tmp_path):
    projection = {"input": {"shape": [2], "dtype": "f32", "decoded": [3.0]}}
    capsule, cb, source = _original(tmp_path, read=True, provenance=projection)
    with pytest.raises(native.NativeComponentExecutionError, match="wrong shape"):
        native._bind(capsule, cb, source)


def test_zero_argument_source_still_requires_nonempty_original_outputs(tmp_path):
    capsule, cb, source = _original(tmp_path)
    capsule["inputs"] = []
    with pytest.raises(native.NativeComponentExecutionError, match="complete typed source output roster"):
        native._bind(capsule, cb, source)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "shape", "dtype", "return_type"])
def test_empty_inputs_do_not_relax_complete_original_outputs(tmp_path, defect):
    capsule, cb, source = _original(tmp_path)
    if defect == "missing":
        cb["kernel_abi"]["outputs"].pop()
    elif defect == "duplicate":
        cb["kernel_abi"]["outputs"][1] = cb["kernel_abi"]["outputs"][0]
    elif defect in ("shape", "dtype"):
        cb["tensors"]["ptr_a"][defect] = [1] if defect == "shape" else "i32"
    else:
        source.write_text(
            source.read_text().replace("tensor<2xf32>", "tensor<1xf32>").replace("[1.25, -2.0]", "[1.25]")
        )
    with pytest.raises(native.NativeComponentExecutionError):
        native._bind(capsule, cb, source)
