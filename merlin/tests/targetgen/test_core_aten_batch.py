"""The merged Core ATen bundle has stable positional input and output mappings."""

from __future__ import annotations

import json

import numpy as np
import pytest
from targetgen_fixtures import capture

from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.targetgen.core_aten_batch import build_core_aten_batch
from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch


def unary(op, declarations=""):
    """A capsule applying scalar ``op`` (``%5 = op ...``) elementwise to one tensor<2xf32>."""
    return (
        "builtin.module {\n" + declarations + "  func.func @forward(%0: tensor<2xf32>) -> tensor<2xf32> {\n"
        "    %1 = tensor.empty() : tensor<2xf32>\n"
        "    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, "
        'affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} '
        "ins(%0 : tensor<2xf32>) outs(%1 : tensor<2xf32>) {\n"
        "      ^bb0(%3: f32, %4: f32):\n"
        f"        %5 = {op}\n"
        "        linalg.yield %5 : f32\n"
        "    } -> tensor<2xf32>\n"
        "    func.return %2 : tensor<2xf32>\n"
        "  }\n}\n"
    )


F32X2 = [{"dtype": "f32", "shape": [2]}]


def test_nested_nonfinite_documents_preserve_ieee_values():
    from merlin.targetgen.core_aten_batch_grade import _complex_values, _grade_output

    values = [{"kind": "float", "value": value} for value in ("nan", "inf", "-inf", "-0.0")]
    decoded = np.asarray(_complex_values([values]), np.float64)
    assert np.isnan(decoded[0, 0])
    assert np.signbit(decoded[0, 3])
    expected = {"kind": "tensor", "dtype": "float64", "shape": [1, 4], "values": [values]}
    assert (
        _grade_output(
            decoded.tobytes(),
            {"dtype": "f64", "shape": [1, 4]},
            expected,
            {"comparison_parameters": {"equal_nan": True}},
        )["status"]
        == "pass"
    )


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("real", [2.5, "nan", "-inf", "-0.0"])
def test_complex_scalar_is_one_output_and_uses_declared_comparison(dtype, real):
    from merlin.targetgen.core_aten_batch_grade import _grade_output, _leaves

    expected = dict(kind="complex", real=dict(kind="float", value=real) if isinstance(real, str) else real, imag=0.0)
    data = np.array(complex(float(real), 0.0), dtype=dtype)
    abi = dict(dtype="complex<f32>" if dtype == "complex64" else "complex<f64>", shape=[])
    policy = dict(comparison_parameters=dict(equal_nan=True, rtol=1e-5, atol=1e-6))
    assert _leaves(dict(kind="tuple", items=[expected, 7])) == [expected, 7]
    assert _grade_output(data.tobytes(), abi, expected, policy)["status"] == "pass"
    assert _grade_output(data.tobytes(), {**abi, "shape": [1]}, expected, policy)["status"] == "mismatch"
    wrong = np.array(9 + 8j, dtype=dtype)
    assert _grade_output(wrong.tobytes(), abi, expected, policy)["status"] == "mismatch"
    if real == "-0.0":
        exact = dict(comparison_parameters=dict(storage_bits=True))
        assert _grade_output(data.tobytes(), abi, expected, exact)["status"] == "pass"
        positive_zero = np.array(complex(0.0, 0.0), dtype=dtype)
        assert _grade_output(positive_zero.tobytes(), abi, expected, exact)["status"] == "mismatch"


def test_malformed_complex_scalar_is_ungradable():
    from merlin.targetgen.core_aten_batch_grade import _grade_output

    verdict = _grade_output(
        np.array(1 + 2j, np.complex64).tobytes(),
        dict(dtype="complex<f32>", shape=[]),
        dict(kind="complex", real=1.0),
        {},
    )
    assert verdict["status"] == "ungradable"


def test_module_symbols_are_namespaced_and_dynamic_return_is_preserved():
    from merlin.targetgen.core_aten_batch import _function_parts, _rename_ssa

    source = """builtin.module attributes {prov.example = "source"} {
      ml_program.global private mutable @state(dense<0> : tensor<i64>) : tensor<i64>
      func.func @forward(%x: tensor<?xi64>) -> tensor<?xi64> {
        %s = "ml_program.global_load"() <{global = @state}> : () -> tensor<i64>
        func.return %x : tensor<?xi64>
      }
    }"""
    names, body, returns, types, declarations = _function_parts(source)
    assert (names, returns, types) == (["%x"], ["%x"], ["tensor<?xi64>"])
    assert "@left_state" in _rename_ssa(declarations, "left_", {})
    assert "@right_state" in _rename_ssa(body, "right_", {})
    assert '"@state"' == _rename_ssa('"@state"', "left_", {})


@pytest.mark.parametrize("extent", [0, 1, 5])
def test_dynamic_grading_uses_runtime_extents(extent):
    data = np.arange(extent * 2, dtype=np.int64).reshape(extent, 2)
    record = {
        "overload": "fixture",
        "status": "bundled",
        "output_indices": [0],
        "output_abi": [{"dtype": "i64", "shape": [4, 2], "declared_shape": [-1, 2]}],
        "case": {"expected": {"kind": "tensor", "dtype": "int64", "shape": [extent, 2], "values": data.tolist()}},
    }
    assert (
        grade_core_aten_batch({"cases": [record]}, [data.tobytes()], output_shapes=[[extent, 2]])["passed_count"] == 1
    )
    assert grade_core_aten_batch({"cases": [record]}, [data.tobytes()])["passed_count"] == 0


def test_batch_inlines_clean_cases_and_keeps_unavailable_case(tmp_path):
    captures = tmp_path / "captures"
    cases = [
        {"overload": "aten.add.Tensor", "expected": {"kind": "tensor"}},
        {"overload": "aten.sub.Tensor", "expected": {"kind": "tensor"}},
        {"overload": "aten.missing.default", "expected": {"kind": "tensor"}},
    ]
    for name, op, values in (
        ("aten__add__Tensor", "arith.addf", [[1.0, 2.0], [3.0, 4.0]]),
        ("aten__sub__Tensor", "arith.subf", [[4.0, 3.0], [2.0, 1.0]]),
    ):
        source = unary(f"{op} %3, %3 : f32")
        capture(captures / name, source, [values[0]], values[1], F32X2, F32X2, status="captured_exact")
    corpus = {"cases": cases, "denominator_sha256": "fixture"}
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = build_core_aten_batch(corpus, captures, first)
    build_core_aten_batch(corpus, captures, second)
    counts = (report["case_count"], report["bundled_count"], report["input_count"], report["output_count"])
    assert counts == (3, 2, 2, 2)
    assert [row["output_indices"] for row in report["cases"][:2]] == [[0], [1]]
    assert report["cases"][2]["status"] == "capture_unavailable"
    assert "%c0_2" in (first / "model.mlir").read_text()
    assert "%c1_2" in (first / "model.mlir").read_text()
    parse_mlir_file(first / "model.mlir")
    with np.load(first / "inputs.npz") as data:
        assert data.files == ["in0", "in1"]
        np.testing.assert_array_equal(data["in0"], [1, 2])
        np.testing.assert_array_equal(data["in1"], [4, 3])
    for filename in (
        "model.mlir",
        "inputs.npz",
        "core_aten_batch_map.json",
        "weights.safetensors.manifest.json",
        "input_order.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_batch_grader_uses_float_tolerance_and_keeps_all_case_failures():
    mapping = {
        "cases": [
            {
                "overload": "aten.close.default",
                "status": "bundled",
                "output_indices": [0],
                "output_abi": [{"dtype": "f32", "shape": [2]}],
                "case": {
                    "comparison_parameters": {"rtol": 1e-5, "atol": 1e-6, "equal_nan": True},
                    "expected": {"kind": "tensor", "dtype": "float32", "shape": [2], "values": [1.0, 2.0]},
                },
            },
            {
                "overload": "aten.bad.default",
                "status": "bundled",
                "output_indices": [1],
                "output_abi": [{"dtype": "i64", "shape": [1]}],
                "case": {
                    "comparison_parameters": {},
                    "expected": {"kind": "tensor", "dtype": "int64", "shape": [1], "values": [7]},
                },
            },
            {"overload": "aten.unavailable.default", "status": "capture_unavailable", "reason": "opaque"},
        ]
    }
    outputs = [np.array([1.000001, 2.0], np.float32).tobytes(), np.array([8], np.int64).tobytes()]
    verdict = grade_core_aten_batch(mapping, outputs)
    assert verdict["status_counts"] == {"capture_unavailable": 1, "mismatch": 1, "pass": 1}
    assert verdict["cases"]["aten.bad.default"]["results"][0]["actual"] == 8
    failed = grade_core_aten_batch(mapping, execution_error="compiler crash")
    assert failed["status_counts"] == {"capture_unavailable": 1, "execution_failed": 2}


def test_metadata_only_empty_result_requires_descriptor_observations():
    record = {
        "overload": "aten.empty.memory_format",
        "status": "bundled",
        "output_indices": [0],
        "output_abi": [{"dtype": "f32", "shape": [2, 3]}],
        "case": {
            "comparison": "metadata",
            "comparison_parameters": {},
            "expected": {
                "kind": "tensor",
                "dtype": "float32",
                "shape": [2, 3],
                "stride": [3, 1],
                "requires_grad": False,
            },
        },
    }
    raw = np.zeros((2, 3), np.float32).tobytes()
    verdict = grade_core_aten_batch({"cases": [record]}, [raw])
    assert verdict["status_counts"] == {"ungradable": 1}
    assert "descriptor observations" in verdict["cases"]["aten.empty.memory_format"]["results"][0]["reason"]


def test_identical_functional_results_have_unambiguous_single_user_value(tmp_path):
    captures = tmp_path / "captures"
    source = (
        "builtin.module { func.func @forward(%x: tensor<2xi64>) -> (tensor<2xi64>, tensor<2xi64>) {\n"
        "func.return %x, %x : tensor<2xi64>, tensor<2xi64>\n} }"
    )
    abi = [dict(dtype="i64", shape=[2])]
    capture(captures / "fixture", source, [[5, 9]], [5, 9], abi, abi)
    report = build_core_aten_batch(
        {"cases": [{"overload": "fixture", "expected": {"kind": "tensor"}}]}, captures, tmp_path / "bundle"
    )
    assert report["bundled_count"] == 1
    assert report["output_count"] == 1
    assert report["cases"][0]["result_projection"]["proof"] == "all_results_identical_ssa"


def test_bundle_carries_independent_mutable_globals(tmp_path):
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import m2m_python

    if not m2m_python().is_file():
        pytest.skip("native MLIR interpreter unavailable")
    cases = [{"overload": name, "expected": {"kind": "tensor"}} for name in ("left", "right")]
    captures = tmp_path / "captures"
    source = """builtin.module {
          ml_program.global private mutable @state(dense<0> : tensor<i64>) : tensor<i64>
          func.func @forward() -> tensor<i64> {
            %state = "ml_program.global_load"() <{global = @state}> : () -> tensor<i64>
            "ml_program.global_store"(%state) <{global = @state}> : (tensor<i64>) -> ()
            func.return %state : tensor<i64>
          }
        }"""
    for case in cases:
        capture(captures / case["overload"], source, [], 0, output_abi=[{"dtype": "i64", "shape": []}])
    bundle = tmp_path / "bundle"
    report = build_core_aten_batch({"cases": cases}, captures, bundle)
    assert report["bundled_count"] == 2
    llvm = lower_to_llvm_ir((bundle / "model.mlir").read_text(), workdir=tmp_path / "lower")
    assert "@c0_state" in llvm
    assert "@c1_state" in llvm


def test_generic_reprinted_bundle_still_restores_caller_layouts(tmp_path, monkeypatch):
    """A module with private declarations is re-printed generically; layouts use the custom form."""
    from merlin.llvmlower import generic_form

    def generic(path, *args, **kwargs):
        out = tmp_path / "generic.mlir"
        out.write_text(
            '"builtin.module"() ({\n  "func.func"() <{sym_name = "forward"}> ({\n  }) : () -> ()\n}) : () -> ()\n'
        )
        return out

    monkeypatch.setattr(generic_form, "to_generic_form", generic)
    layout = {"dtype": "float32", "shape": [2], "stride": [1], "storage_offset": 0, "requires_grad": False}
    contract = {"authority": "fixture", "results": [{**layout, "role": "user_output"}], "inputs": [layout]}
    source = unary("arith.negf %3 : f32", "  func.func private @helper(f32) -> f32\n")
    directory = tmp_path / "captures" / "aten__neg__default"
    capture(directory, source, [[1.0, 2.0]], [-1.0, -2.0], F32X2, F32X2, "captured_exact", result_contract=contract)
    tensor = {"kind": "tensor", **layout, "values": [1.0, 2.0]}
    case = {
        "overload": "aten.neg.default",
        "arguments": {"args": {"kind": "tuple", "items": [tensor]}, "kwargs": {}},
        "expected": {**tensor, "values": [-1.0, -2.0]},
    }
    report = build_core_aten_batch(
        {"cases": [case], "denominator_sha256": "fixture"}, tmp_path / "captures", tmp_path / "out"
    )
    assert report["bundled_count"] == 1
    assert '"func.func"()' in (tmp_path / "out" / "model.mlir").read_text()
    boundary = json.loads((tmp_path / "out" / "semantic_io.json").read_text())
    assert boundary["post_indices"] == [1]


_CUSTOM_STATEFUL = """builtin.module {
  ml_program.global private mutable @state(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%x: tensor<2xf32>, %y: tensor<2xf32>) -> (tensor<2xf32>, tensor<i64>) {
    %s = ml_program.global_load @state : tensor<i64>
    func.return %x, %s : tensor<2xf32>, tensor<i64>
  }
}"""

_GENERIC_STATEFUL = """"builtin.module"() ({
  "ml_program.global"() {sym_name = "state", type = tensor<i64>, is_mutable, value = dense<0> : tensor<i64>, \
sym_visibility = "private"} : () -> ()
  "func.func"() <{sym_name = "forward", \
function_type = (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<i64>)}> ({
  ^bb0(%x: tensor<2xf32>, %y: tensor<2xf32>):
    %s = "ml_program.global_load"() <{global = @state}> {prov.aten = "aten.rand.default"} : () -> tensor<i64>
    "func.return"(%x, %s) : (tensor<2xf32>, tensor<i64>) -> ()
  }) : () -> ()
}) : () -> ()
"""


def test_generic_form_capture_matches_custom_form_parts():
    from merlin.targetgen.core_aten_batch import _function_parts

    custom = _function_parts(_CUSTOM_STATEFUL)
    generic = _function_parts(_GENERIC_STATEFUL)
    assert (generic[0], generic[2], generic[3]) == (custom[0], custom[2], custom[3])
    assert '"ml_program.global_load"() <{global = @state}>' in generic[1]
    assert "func.return" not in generic[1] and "^bb0" not in generic[1]
    assert 'sym_name = "state"' in generic[4] and "@forward" not in generic[4] and '"forward"' not in generic[4]


def test_generic_symbol_definitions_rename_with_their_references():
    from merlin.targetgen.core_aten_batch import _function_parts, _rename_ssa

    _, body, _, _, declarations = _function_parts(_GENERIC_STATEFUL)
    assert 'sym_name = "c0_state"' in _rename_ssa(declarations, "c0_", {})
    assert "@c0_state" in _rename_ssa(body, "c0_", {})
    assert 'sym_visibility = "private"' in _rename_ssa(declarations, "c0_", {})
    assert '"aten.rand.default"' in _rename_ssa(body, "c0_", {})


@pytest.mark.parametrize("argument_types", ["()", "(tensor<2xf32>, tensor<2xf32>)"])
def test_generic_function_type_before_symbol_keeps_arrow_and_nested_commas(argument_types):
    from merlin.targetgen.core_aten_batch import _function_parts

    old = 'sym_name = "forward", function_type = (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<i64>)'
    reordered = f'function_type = {argument_types} -> (tensor<2xf32>, tensor<i64>), sym_name = "forward"'
    source = _GENERIC_STATEFUL.replace(old, reordered)
    assert source != _GENERIC_STATEFUL
    if argument_types == "()":
        source = source.replace("  ^bb0(%x: tensor<2xf32>, %y: tensor<2xf32>):\n", "")
    args, body, values, types, declarations = _function_parts(source)
    assert args == ([] if argument_types == "()" else ["%x", "%y"])
    assert values == ["%x", "%s"]
    assert types == ["tensor<2xf32>", "tensor<i64>"]
    assert '"ml_program.global_load"() <{global = @state}>' in body
    assert 'sym_name = "state"' in declarations


def test_function_type_first_still_refuses_multiple_forward_definitions():
    from merlin.targetgen.core_aten_batch import _function_parts

    function_start = _GENERIC_STATEFUL.index('  "func.func"()')
    function_end = _GENERIC_STATEFUL.rindex("}) : () -> ()")
    function = _GENERIC_STATEFUL[function_start:function_end]
    source = _GENERIC_STATEFUL[:function_end] + function + _GENERIC_STATEFUL[function_end:]
    with pytest.raises(ValueError, match="exactly one @forward"):
        _function_parts(source)


@pytest.mark.parametrize("generic", [False, True])
def test_caller_layouts_accept_alias_preamble_and_preserve_computed_outputs(generic):
    from merlin.targetgen.core_aten_batch import caller_boundary_layouts

    alias = "#map = affine_map<(d0) -> (d0)>\n"
    if generic:
        source = (
            alias
            + """"builtin.module"() ({
  "func.func"() <{function_type = (tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>), sym_name = "forward"}> ({
  ^bb0(%x: tensor<2xf32>):
    %y = "arith.addf"(%x, %x) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "func.return"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> ()
  }) : () -> ()
}) : () -> ()
"""
        )
    else:
        source = (
            alias
            + """builtin.module {
  func.func @forward(%x: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
    %y = arith.addf %x, %x : tensor<2xf32>
    func.return %x, %y : tensor<2xf32>, tensor<2xf32>
  }
}
"""
        )
    layout = dict(dtype="f32", shape=[2], stride=[1], storage_offset=0, requires_grad=False)
    document = dict(kind="tensor", **{**layout, "stride": [3], "storage_offset": 4})
    boundary = dict(outputs=[dict(layout), dict(layout)])
    records = [dict(case=dict(arguments=[document]), input_indices=[0], semantic_boundary=dict(post_indices=[0]))]
    observed = caller_boundary_layouts(source, boundary, records)
    assert observed["outputs"][0] == {**layout, "stride": [3], "storage_offset": 4}
    assert observed["outputs"][1] == layout
    assert boundary["outputs"] == [layout, layout]


@pytest.mark.parametrize(
    "source",
    [
        _GENERIC_STATEFUL.replace('sym_name = "forward"', 'sym_name = "other"'),
        _GENERIC_STATEFUL.replace('"func.return"(%x, %s) : (tensor<2xf32>, tensor<i64>) -> ()', ""),
    ],
)
def test_generic_form_capture_refuses_missing_forward_or_return(source):
    from merlin.targetgen.core_aten_batch import _function_parts

    with pytest.raises(ValueError):
        _function_parts(source)


@pytest.mark.parametrize("declared", [False, True])
def test_complex_grading_interprets_observed_component_axis(declared):
    data = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    record = dict(
        overload="fixture",
        status="bundled",
        output_indices=[0],
        output_abi=[dict(dtype="complex<f32>", shape=[2], declared_shape=[2, 2])],
        case=dict(
            expected=dict(
                kind="tensor",
                dtype="complex64",
                shape=[2],
                values=[dict(kind="complex", real=1, imag=2), dict(kind="complex", real=3, imag=4)],
            )
        ),
    )
    if not declared:
        del record["output_abi"][0]["declared_shape"]
    assert grade_core_aten_batch({"cases": [record]}, [data.tobytes()], output_shapes=[[2, 2]])["passed_count"] == 1


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_complex_metadata_only_grading_requires_descriptor_observations(dtype):
    from merlin.targetgen.core_aten_batch_grade import _grade_output

    data = np.zeros((3, 2), dtype=dtype)
    abi = dict(dtype="complex<f32>" if dtype == "complex64" else "complex<f64>", shape=[3, 2])
    expected = dict(kind="tensor", dtype=dtype, shape=[3, 2])
    verdict = _grade_output(data.tobytes(), abi, expected, dict(comparison="metadata"))
    assert verdict == dict(status="ungradable", reason="metadata-only result requires compiled descriptor observations")
    expected["values"] = {"kind": "complex", "real": 0.0, "imag": 0.0}
    verdict = _grade_output(data.tobytes(), abi, expected, dict(comparison="torch_close"))
    assert verdict["status"] == "ungradable"
    assert "cannot decode canonical expected value" in verdict["reason"]


def test_frontend_layout_contract_is_rebased_with_batch_arguments(tmp_path):
    from merlin.llvmlower.result_layout import module_attribute, read_layout

    captures, bundle = tmp_path / "captures", tmp_path / "bundle"
    cases = [{"overload": name, "expected": {"kind": "tensor"}} for name in ("left", "right")]
    contract = dict(
        inputs=[dict(shape=[3], stride=[1], storage_offset=0)],
        results=[dict(shape=[3], stride=[1], storage_offset=0, alias_inputs=[0], alias_results=[])],
    )
    source = (
        "builtin.module"
        + module_attribute(contract)
        + """ {
          func.func @forward(%x: tensor<3xf32>) -> tensor<3xf32> {
            func.return %x : tensor<3xf32>
          }
        }"""
    )
    abi = [dict(shape=[3], dtype="f32")]
    for case in cases:
        capture(captures / case["overload"], source, [[1, 2, 3]], [1, 2, 3], abi, abi)
    report = build_core_aten_batch(dict(cases=cases), captures, bundle)
    assert report["output_count"] == 2
    layout = read_layout((bundle / "model.mlir").read_text())
    assert [r["alias_inputs"] for r in layout["results"]] == [[0], [1]]


def test_overlapping_input_observation_borrows_readonly_storage(tmp_path):
    from merlin.llvmlower.input_layout import refuse_overlapping_input_aliases
    from merlin.llvmlower.result_layout import module_attribute, read_layout

    input_layout = dict(shape=[2, 3], stride=[0, 1], storage_offset=0)
    output_layout = dict(shape=[2, 3], stride=[3, 1], storage_offset=0, alias_inputs=[])
    layout = dict(inputs=[input_layout], results=[output_layout])
    source = (
        "builtin.module"
        + module_attribute(layout)
        + """ {
      func.func @forward(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %zero = arith.constant 0.0 : f32
        %out = tensor.splat %zero : tensor<2x3xf32>
        func.return %out : tensor<2x3xf32>
      }
    }"""
    )
    abi = [dict(shape=[2, 3], dtype="f32")]
    contract = dict(
        authority="same_conversion_exported_program",
        inputs=[dict(input_layout, dtype="float32", requires_grad=False)],
        results=[dict(output_layout, dtype="float32", requires_grad=False, role="user_output")],
    )
    directory = tmp_path / "captures/fixture"
    capture(directory, source, [[[1, 2, 3], [1, 2, 3]]], [[0, 0, 0], [0, 0, 0]], abi, abi, result_contract=contract)
    bundle = tmp_path / "bundle"
    build_core_aten_batch(
        dict(
            cases=[
                dict(
                    overload="fixture",
                    expected=dict(kind="tensor"),
                    arguments=dict(args=[dict(input_layout, kind="tensor", dtype="float32", requires_grad=False)]),
                )
            ]
        ),
        directory.parent,
        bundle,
    )
    actual = read_layout((bundle / "model.mlir").read_text())
    assert actual["inputs"] == [input_layout]
    assert actual["results"][1]["stride"] == [0, 1]
    assert actual["results"][1]["alias_inputs"] == [0]
    refuse_overlapping_input_aliases(actual)
    actual["results"][0] = dict(output_layout, alias_inputs=[0])
    refuse_overlapping_input_aliases(actual)
    actual["results"][0].update(role="user_input_mutation", input_index=0)
    # This dense mutation outgrows the expanded caller and owns a new allocation.
    refuse_overlapping_input_aliases(actual)
    actual["results"][0].update(shape=[1, 3], stride=[3, 1])
    with pytest.raises(ValueError, match="overlapping logical elements"):
        refuse_overlapping_input_aliases(actual)
