"""Only original typed source-return correspondence may rename native outputs."""

import copy
import hashlib

import numpy as np
import pytest
import yaml

from merlin.targetgen.native_model_execution import NativeModelExecutionError, _digest
from merlin.targetgen.native_model_output_binding import bind_source_outputs, checked_source_output_values


def fixture(tmp_path):
    source = """module { func.func @forward(%x:tensor<2xi32>) -> (tensor<2xi32>,tensor<2xi8>) {
      %a="test.copy"(%x):(tensor<2xi32>)->tensor<2xi32>
      %b="test.cast"(%a):(tensor<2xi32>)->tensor<2xi8>
      func.return %a,%b:tensor<2xi32>,tensor<2xi8>
    }}"""
    root, runtime = tmp_path / "capsule", tmp_path / "runtime"
    root.mkdir()
    runtime.mkdir()
    (root / "capsule.interface.mlir").write_text(source)
    interface = runtime / "model.mlir"
    interface.write_text(source)
    names = ["wide", "narrow"]
    order = ["I0", *names]
    declaration = dict(
        interface_mlir="capsule.interface.mlir",
        linalg_mlir="capsule.interface.mlir",
        inputs=[dict(name="I0", shape=[2], dtype="i32")],
        operation=dict(attributes=dict(outs=names, out=names[0], arg_order=order)),
        numeric_policy=dict(
            compare="per_output",
            outputs={"wide": dict(compare="exact_int", dtype="i32"), "narrow": dict(compare="exact_int", dtype="i8")},
        ),
    )
    (root / "capsule.yaml").write_text(yaml.safe_dump(declaration))
    golden = dict(
        outputs={"wide": np.asarray([131, -132]), "narrow": np.asarray([-125, 124])},
        oracle_provenance=dict(interface="linalg_positional", arg_order=order, linalg_mlir="capsule.interface.mlir"),
    )
    plan = dict(
        schema="mixed_program_plan_v1",
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source_op_count=2,
        entry_bindings=["arg0"],
        source_values=[dict(op_index=0, result_index=0, tensor="Y0"), dict(op_index=1, result_index=0, tensor="Y1")],
        output_bindings=["Y0", "Y1"],
        compiler_temporaries=[],
        schedule_instruction_count=2,
        prologue_instruction_range=[0, 0],
        epilogue_instruction_range=[2, 2],
        tasks=[
            dict(
                task_index=0,
                kind="host",
                source_op_indices=[0],
                instruction_start=0,
                instruction_end=1,
                reads=["arg0"],
                writes=["Y0"],
            ),
            dict(
                task_index=1,
                kind="host",
                source_op_indices=[1],
                instruction_start=1,
                instruction_end=2,
                reads=["Y0"],
                writes=["Y1"],
            ),
        ],
    )
    cb = dict(
        kernel_abi=dict(
            kind="whole_program",
            outputs=["Y0", "Y1"],
            args=[
                dict(tensor="arg0", access="read"),
                dict(tensor="Y0", access="write"),
                dict(tensor="Y1", access="write"),
            ],
        ),
        tensors={
            "arg0": dict(shape=[2], dtype="i32", role="input"),
            "Y0": dict(shape=[2], dtype="i32", role="output"),
            "Y1": dict(shape=[2], dtype="i8", role="output"),
        },
        params=dict(global_program_plan=plan),
    )
    return root, interface, declaration, golden, cb


def test_checked_source_order_and_type_bind_names_without_changing_original_outputs(tmp_path):
    root, interface, declaration, golden, cb = fixture(tmp_path)
    source_bytes, original_values = interface.read_bytes(), copy.deepcopy(golden["outputs"])
    binding = bind_source_outputs(cb, root, interface, golden)
    assert [(row["source_name"], row["compiled_name"], row["dtype"]) for row in binding["returns"]] == [
        ("wide", "Y0", "i32"),
        ("narrow", "Y1", "i8"),
    ]
    assert binding["returns"][1]["source_value"] == dict(op_index=1, result_index=0)
    native = dict(
        source_output_binding=binding,
        source=dict(capsule_declaration=_digest(root / "capsule.yaml"), interface=_digest(interface)),
    )
    observed = {"Y0": [131, -132], "Y1": [-125, 124]}
    joined = checked_source_output_values(cb, native, golden, observed)
    for name, values in joined.items():
        np.testing.assert_array_equal(values, original_values[name])
        np.testing.assert_array_equal(golden["outputs"][name], original_values[name])
    assert set(observed) == {"Y0", "Y1"} and interface.read_bytes() == source_bytes
    assert yaml.safe_load((root / "capsule.yaml").read_text()) == declaration
    native["source_output_binding"] = copy.deepcopy(binding)
    native["source_output_binding"]["returns"][0]["position"] = False
    with pytest.raises(NativeModelExecutionError, match="freshly checked"):
        checked_source_output_values(cb, native, golden, observed)
    native["source_output_binding"] = copy.deepcopy(binding)
    native["source_output_binding"]["returns"][0]["source_name"] = "narrow"
    with pytest.raises(NativeModelExecutionError, match="freshly checked"):
        checked_source_output_values(cb, native, golden, observed)


@pytest.mark.parametrize(
    "change",
    [
        "order",
        "count",
        "dtype",
        "shape",
        "source",
        "digest",
        "result",
        "declaration",
        "golden_order",
        "reference_shape",
        "duplicate",
        "policy_dtype",
        "plan",
    ],
)
def test_mutated_order_count_type_source_or_arbitrary_alias_cannot_bind(tmp_path, change):
    root, interface, declaration, golden, cb = fixture(tmp_path)
    plan = cb["params"]["global_program_plan"]
    if change == "order":
        cb["kernel_abi"]["outputs"].reverse()
        plan["output_bindings"].reverse()
    elif change == "count":
        cb["kernel_abi"]["outputs"].pop()
        plan["output_bindings"].pop()
    elif change in ("dtype", "shape"):
        cb["tensors"]["Y0"][change] = "f32" if change == "dtype" else [1, 2]
    elif change == "source":
        interface.write_text(interface.read_text().replace("func.return %a,%b", "func.return %b,%a"))
    elif change == "digest":
        plan["source_sha256"] = "0" * 64
    elif change == "result":
        plan["source_values"][0]["op_index"] = 1
    elif change == "declaration":
        declaration["operation"]["attributes"]["outs"].reverse()
        (root / "capsule.yaml").write_text(yaml.safe_dump(declaration))
    elif change == "golden_order":
        golden["oracle_provenance"]["arg_order"] = ["I0", "narrow", "wide"]
    elif change == "reference_shape":
        golden["outputs"]["wide"] = [[131, -132]]
    elif change == "duplicate":
        cb["kernel_abi"]["outputs"] = ["Y0", "Y0"]
    elif change == "policy_dtype":
        declaration["numeric_policy"]["outputs"]["wide"]["dtype"] = "i8"
        (root / "capsule.yaml").write_text(yaml.safe_dump(declaration))
    else:
        cb["params"].pop("global_program_plan")
    with pytest.raises(NativeModelExecutionError):
        bind_source_outputs(cb, root, interface, golden)


def test_missing_extra_or_reordered_actual_result_values_do_not_change_binding(tmp_path):
    root, interface, _declaration, golden, cb = fixture(tmp_path)
    binding = bind_source_outputs(cb, root, interface, golden)
    native = dict(
        source_output_binding=binding,
        source=dict(capsule_declaration=_digest(root / "capsule.yaml"), interface=_digest(interface)),
    )
    for observed in ({"Y0": [1, 2]}, {"Y0": [1, 2], "Y1": [3, 4], "extra": [5]}):
        with pytest.raises(NativeModelExecutionError, match="actual compiled output roster"):
            checked_source_output_values(cb, native, golden, observed)
    native["source_output_binding"] = None
    with pytest.raises(NativeModelExecutionError, match="freshly checked"):
        checked_source_output_values(cb, native, golden, {"Y0": [1, 2], "Y1": [3, 4]})


def test_historical_same_name_path_requires_no_new_binding_or_source_evidence():
    cb = dict(kernel_abi=dict(outputs=["out"]))
    golden = dict(outputs=dict(out=[1, 2]))
    assert bind_source_outputs(cb, None, None, golden) is None
    assert checked_source_output_values(cb, {}, golden, dict(out=[1, 2])) == dict(out=[1, 2])
