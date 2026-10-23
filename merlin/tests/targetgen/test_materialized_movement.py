"""Borrowed movement outputs need a static source-bound copy-out contract."""

import json

import pytest
from mlir_fixtures import assert_unchanged
from targetgen_fixtures import selector_catalog

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.result_layout import ATTRIBUTE, module_attribute
from merlin.targetgen.materialized_movement import alias_materialization_valid, expose_alias_results


def source_and_report():
    layout = dict(
        inputs=[dict(shape=[2, 3], stride=[4, 1], storage_offset=1)],
        results=[dict(shape=[3, 2], stride=[1, 4], storage_offset=1, alias_inputs=[0], role="user_output")],
    )
    module = parse_mlir_text(
        "builtin.module"
        + module_attribute(layout)
        + """ {
          func.func @forward(%x: tensor<2x3xi64>) -> tensor<3x2xi64> {
            %empty = tensor.empty() : tensor<3x2xi64>
            %y = linalg.transpose ins(%x : tensor<2x3xi64>)
              outs(%empty : tensor<3x2xi64>) permutation = [1, 0]
            func.return %y : tensor<3x2xi64>
          }
        }"""
    )
    report = dict(cases=[dict(status="bundled", output_indices=[0], case=dict(mutated_arguments=[]))])
    return module, report


def test_static_alias_selection_marks_only_returned_observation():
    module, report = source_and_report()
    assert expose_alias_results(module, report) == [0]
    layout = json.loads(module.attributes[ATTRIBUTE].data)
    assert layout["results"][0]["materialized_byte_selection"] is True
    assert layout["results"][0]["alias_inputs"] == [0]
    assert expose_alias_results(module, report) == [0]


@pytest.mark.parametrize("bad", ["shape", "stride", "offset", "base", "mutation", "dtype"])
def test_unknown_or_invalid_alias_layout_never_grants_copy_out(bad):
    from xdsl.dialects.builtin import StringAttr

    module, report = source_and_report()
    layout = json.loads(module.attributes[ATTRIBUTE].data)
    if bad == "mutation":
        report["cases"][0]["case"]["mutated_arguments"] = ["args[0]"]
    elif bad == "base":
        layout["results"][0]["alias_inputs"] = [2]
    elif bad == "dtype":
        # The compiled return dtype, not metadata alone, is authoritative.
        module = parse_mlir_text(str(module).replace("tensor<3x2xi64>", "tensor<3x2xf64>"))
    else:
        layout["results"][0][bad if bad != "offset" else "storage_offset"] = {
            "shape": [3, 3],
            "stride": [-1, 4],
            "offset": 99,
        }[bad]
    module.attributes[ATTRIBUTE] = StringAttr(json.dumps(layout))
    assert expose_alias_results(module, report) == []


def test_alias_marker_alone_is_no_device_proof(tmp_path):
    module, report = source_and_report()
    expose_alias_results(module, report)
    (tmp_path / "model.mlir").write_text(str(module) + "\n")
    assert not alias_materialization_valid(tmp_path, report, dict(exact_dtype_sites=1, numerical_family="F1"))
    assert not alias_materialization_valid(
        tmp_path,
        report,
        dict(
            exact_dtype_sites=1,
            numerical_family="F1",
            materialized_output_aliases=dict(source_sha256="0" * 64, output_indices=[0]),
        ),
    )


def test_complete_selector_proof_reopens_the_exact_alias_source(tmp_path):
    module, report = source_and_report()
    indices = expose_alias_results(module, report)
    operations = list(module.walk())
    root = next(op for op in operations if op.name == "linalg.transpose")
    manifest = selector_catalog(module, "select_bytes", (operations.index(root), dict(x=dict(func_arg=0))))
    plan = dict(
        numerical_family="F1",
        exact_dtype_sites=1,
        provider_catalog=manifest,
        materialized_output_aliases=dict(source_sha256=manifest["source_sha256"], output_indices=indices),
    )
    path = tmp_path / "model.mlir"
    path.write_text(str(module) + "\n")
    assert alias_materialization_valid(tmp_path, report, plan)
    plan["numerical_family"] = "F2"
    assert not alias_materialization_valid(tmp_path, report, plan)
    plan["numerical_family"] = "F1"
    manifest["bindings"][0]["numeric"]["exact_arithmetic"] = False
    assert not alias_materialization_valid(tmp_path, report, plan)
    manifest["bindings"][0]["numeric"]["exact_arithmetic"] = True
    path.write_text(str(module).replace("[1, 0]", "[0, 1]") + "\n")
    assert not alias_materialization_valid(tmp_path, report, plan)


@pytest.mark.parametrize("dtype", ["i1", "bf16", "f64", "i64"])
@pytest.mark.parametrize("shape", [[], [2, 3]])
def test_folded_argument_result_gets_real_boundary_movement(dtype, shape):
    prefix = "x".join(str(n) for n in shape)
    typ = f"tensor<{prefix + 'x' if prefix else ''}{dtype}>"
    strides = [] if not shape else [3, 1]
    layout = dict(
        inputs=[dict(shape=shape, stride=strides, storage_offset=0)],
        results=[dict(shape=shape, stride=strides, storage_offset=0, alias_inputs=[0])] * 2,
    )
    module = parse_mlir_text(
        "builtin.module"
        + module_attribute(layout)
        + f" {{ func.func @forward(%x: {typ}) -> ({typ}, {typ}) {{ func.return %x, %x : {typ}, {typ} }} }}"
    )
    report = dict(cases=[dict(status="bundled", output_indices=[0], case=dict(mutated_arguments=[]))])
    assert expose_alias_results(module, report) == [0]
    module.verify()
    function = next(op for op in module.walk() if op.name == "func.func")
    returned = function.body.block.last_op
    assert returned.operands[0].owner.name == "linalg.generic"
    assert returned.operands[1] is function.body.block.args[0]
    assert len([op for op in module.walk() if op.name == "linalg.generic"]) == 1
    assert_unchanged(module, lambda m: expose_alias_results(m, report), [0])
