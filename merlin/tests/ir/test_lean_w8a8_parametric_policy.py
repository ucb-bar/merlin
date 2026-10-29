"""Qualify full-module policy matching under the provisional source interpretation.

SSA renaming is checked structurally, including nested regions and internal calls. Matching
still supplies neither source definedness nor the whole-function numerical domain.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_module import calibrated_source, source

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8

pytestmark = pytest.mark.usefixtures("lean_oracle")


def check(before, after, work, tools, scope="all", mutate=None):
    paths = []
    for name, program in [("before", before), ("after", after)]:
        directory = work / name
        directory.mkdir()
        paths.append(exported(program, directory, tools, generic=True, allow_unregistered=True))
    if mutate:
        document = json.loads(paths[1].read_text())
        mutate(document)
        paths[1].write_text(json.dumps(document))
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_parametric_module.lean"),
            str(paths[0]),
            str(paths[1]),
            scope,
        ],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "w8a8_provisional_source_semantics_match"
    assert report["whole_pass_verified"] is False and report["policy_reviewed"] is False
    assert report["semantic_laws"] == "MerlinVerify.Rules.W8A8SourceAlpha.laws"
    assert report["theorem"] == "MerlinVerify.Check.PolicyParametric.check_sound"
    assert (proc.returncode == 0) == report["match"]
    (work / "parametric-report.json").write_text(json.dumps(report, indent=2))
    return report


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 0, 2), (0, 3, 2)])
def test_actual_dynamic_pass_matches_with_renamed_ssa(tmp_path, lean_tools, shape):
    before = source(*shape)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    assert check(before, after, tmp_path, lean_tools)["match"]


@pytest.mark.parametrize("reshape,transpose", [(False, False), (True, False), (False, True), (True, True)])
def test_actual_captured_layout_paths_match(tmp_path, lean_tools, reshape, transpose):
    before = calibrated_source(reshape_left=reshape, transpose_right=transpose)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    assert check(before, after, tmp_path, lean_tools, scope="captured")["match"]


def renumber(document):
    def operations(ops):
        for op in ops:
            for result in op["results"]:
                result["id"] = 7 * result["id"] + 1000
            op["operands"] = [7 * value + 1000 for value in op["operands"]]
            for region in op["regions"]:
                for block in region["blocks"]:
                    for argument in block["args"]:
                        argument["id"] = 7 * argument["id"] + 1000
                    operations(block["ops"])

    operations(document["module"]["ops"])


def test_consistent_renaming_includes_captures_and_nested_regions(tmp_path, lean_tools):
    before = source()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    assert check(before, after, tmp_path, lean_tools, mutate=renumber)["match"]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("wrapper", [False, True])
def test_initialized_source_contract_paths_support_renaming(tmp_path, lean_tools, width, wrapper):
    from test_lean_w8a8_source_program import source as quantized

    # These are the same recognized Q/DQ contracts executed by the source-program/native
    # suite, including actual constant/splat parameters. No W8A8 contraction is selected here.
    before = quantized(width, (2, 8), wrapper=wrapper, dequantize=True)
    assert check(before, before.clone(), tmp_path, lean_tools, scope="captured", mutate=renumber)["match"]


def test_matching_unsupported_source_is_not_qualification(tmp_path, lean_tools):
    before = parse_mlir_text("""func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {
      %r = "frontend.unsupported"(%x) : (tensor<2xf32>) -> tensor<2xf32>
      return %r : tensor<2xf32>
    }""")
    report = check(before, before.clone(), tmp_path, lean_tools, scope="captured", mutate=renumber)
    assert report["match"] and report["whole_pass_verified"] is False


@pytest.mark.parametrize("mutation", ["return_value", "scale_floor", "contraction_inputs"])
def test_compiler_valid_changed_values_are_refused(tmp_path, lean_tools, mutation):
    from test_lean_w8a8_scalar import parts
    from xdsl.dialects.builtin import FloatAttr, f32

    before = source(2, 2, 2)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    function = next(op for op in after.walk() if op.name == "func.func")
    if mutation == "return_value":
        returned = next(op for op in function.walk() if op.name == "func.return")
        returned.operands = [function.body.block.args[0]]
    elif mutation == "scale_floor":
        scale = next(op for kind, _, op in parts(after) if kind == "scale")
        list(scale.body.block.ops)[2].properties["value"] = FloatAttr(1e-4, f32)
    else:
        contraction = next(
            op
            for op in after.walk()
            if op.name == "linalg.generic" and [str(arg.type) for arg in op.body.block.args] == ["i8", "i8", "i32"]
        )
        contraction.operands = [contraction.inputs[1], contraction.inputs[0], *contraction.outputs]
    assert not check(before, after, tmp_path, lean_tools)["match"]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_result",
        "changed_capture",
        "module_attr",
        "function_attr",
        "extra_op",
        "extra_region",
        "missing_function",
    ],
)
def test_complete_structure_and_ssa_mutations_refuse(tmp_path, lean_tools, mutation):
    before = source(2, 2, 2)
    after = before.clone()
    assert lower_contraction_int8(after) == 1

    def mutate(document):
        module = document["module"]
        function = module["ops"][0]
        block = function["regions"][0]["blocks"][0]
        if mutation == "duplicate_result":
            # Direct malformed SSA witness: freshness must be recomputed.
            producer = next(op for op in block["ops"] if op["results"])
            producer["results"][0]["id"] = block["args"][0]["id"]
        elif mutation == "changed_capture":
            # Same-typed capture substitution inside an actual emitted region.
            nested = next(op for op in block["ops"] if op["name"] == "linalg.generic")
            body = nested["regions"][0]["blocks"][0]
            body["ops"][-1]["operands"] = [body["args"][-1]["id"]]
        elif mutation == "module_attr":
            module["attrs"]["test.changed_semantics"] = {"kind": "string", "value": "changed"}
        elif mutation == "function_attr":
            function["attrs"]["test.changed_semantics"] = {"kind": "string", "value": "changed"}
        elif mutation == "extra_op":
            module["ops"].append(
                {
                    "name": "frontend.unknown",
                    "results": [],
                    "operands": [],
                    "attrs": {},
                    "props": {},
                    "regions": [],
                    "successors": [],
                }
            )
        elif mutation == "extra_region":
            # The function view reads only its first body. An extra malformed region must
            # refuse instead of being hidden by the outer-skeleton comparison.
            function["regions"].append({"blocks": []})
        else:
            module["ops"] = []

    assert not check(before, after, tmp_path, lean_tools, mutate=mutate)["match"]


def with_internal_call():
    from xdsl.dialects.builtin import StringAttr

    before = source()
    kernel = next(op for op in before.walk() if op.name == "func.func")
    kernel.properties["sym_name"] = StringAttr("kernel")
    wrapper = parse_mlir_text("""func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
      %r = func.call @kernel(%a, %b) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
      return %r : tensor<2x4xf32>
    }""")
    function = next(op for op in wrapper.walk() if op.name == "func.func")
    function.detach()
    before.body.block.add_op(function)
    return before


def test_whole_module_matching_covers_internal_calls(tmp_path, lean_tools):
    before = with_internal_call()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    assert check(before, after, tmp_path, lean_tools)["match"]


def test_changed_internal_callee_refuses(tmp_path, lean_tools):
    from xdsl.dialects.builtin import SymbolRefAttr

    before = with_internal_call()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    called = next(op for op in after.walk() if op.name == "func.call")
    called.properties["callee"] = SymbolRefAttr("forward")
    assert not check(before, after, tmp_path, lean_tools)["match"]


def test_unexpanded_source_cannot_use_candidate_as_reference(tmp_path, lean_tools):
    before = calibrated_source(zero_point=1)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report = check(before, after, tmp_path, lean_tools, scope="captured")
    assert not report["expanded"] and not report["match"]
