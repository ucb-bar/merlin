"""Complete operational typing certificates and malformed SSA/region/call rejection."""

import copy
import json
from pathlib import Path

import pytest

from merlin.common.paths import artifacts_dir, repo_root
from merlin.verify.lean.export import export_file
from merlin.verify.lean.scalar_square import _run
from merlin.verify.lean.scalar_square_typing import prepare, run

PROGRAM = """module {
  func.func private @leaf(%x: f32) -> f32 {
    %two = arith.constant 2.0 : f32
    %r = math.powf %x, %two : f32
    return %r : f32
  }
  func.func @subject(%input: tensor<2xf32>) -> tensor<2xf32> {
    %empty = tensor.empty() : tensor<2xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%input : tensor<2xf32>) outs(%empty : tensor<2xf32>) {
      ^bb0(%x: f32, %old: f32):
        %r = func.call @leaf(%x) : (f32) -> f32
        linalg.yield %r : f32
    } -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}"""


@pytest.fixture(scope="module", autouse=True)
def typing_driver():
    prepare()


@pytest.fixture
def exported(tmp_path, lean_tools):
    mlir = tmp_path / "source.mlir"
    mlir.write_text(PROGRAM)
    path = tmp_path / "source.json"
    export_file(mlir, path, backend="mlir", m2m_python=lean_tools.m2m_python)
    return json.loads(path.read_text())


def check_doc(tmp_path, doc):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(doc))
    return run(path)


def functions(doc):
    return doc["module"]["ops"]


def body(function):
    return function["regions"][0]["blocks"][0]


def generic(doc):
    return next(o for o in body(functions(doc)[1])["ops"] if o["name"] == "linalg.generic")


def call(doc):
    return body({"regions": generic(doc)["regions"]})["ops"][0]


def int32():
    return {"kind": "int", "width": 32}


def test_nested_regions_and_internal_calls_accept(tmp_path, exported):
    result = check_doc(tmp_path, exported)
    assert result["program_typing_checked"] and result["closed_capture_checked"]
    assert result["source_typing_discharged"]
    assert result["external_declarations"] == 0
    for key in (
        "source_definedness_proved",
        "successful_execution_proved",
        "poison_freedom_proved",
        "numerical_law_discharged",
        "policy_approved",
        "original_source_equivalence_claimed",
    ):
        assert result[key] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "operand_type",
        "result_type",
        "constant_type",
        "unbound_operand",
        "function_arguments",
        "function_returns",
        "region_argument_type",
        "region_argument_count",
        "region_yield_type",
        "call_argument",
        "call_result",
        "unknown_callee",
        "nested_callee",
        "call_result_count",
        "region_extra_block",
        "region_missing_block",
        "terminator_result",
        "after_terminator",
        "function_extra_region",
    ],
)
def test_incorrect_operational_types_rejected(tmp_path, exported, mutation):
    doc = copy.deepcopy(exported)
    leaf = functions(doc)[0]
    entry = functions(doc)[1]
    lb = body(leaf)
    gen = generic(doc)
    region_body = gen["regions"][0]["blocks"][0]
    invoke = call(doc)
    if mutation == "operand_type":
        lb["args"][0]["type"] = int32()
        leaf["attrs"]["function_type"]["value"]["inputs"][0] = int32()
    elif mutation == "result_type":
        lb["ops"][1]["results"][0]["type"] = int32()
    elif mutation == "constant_type":
        lb["ops"][0]["results"][0]["type"] = int32()
    elif mutation == "unbound_operand":
        lb["ops"][1]["operands"][0] = 987654321
    elif mutation == "function_arguments":
        leaf["attrs"]["function_type"]["value"]["inputs"][0] = int32()
    elif mutation == "function_returns":
        leaf["attrs"]["function_type"]["value"]["results"][0] = int32()
    elif mutation == "region_argument_type":
        region_body["args"][0]["type"] = int32()
    elif mutation == "region_argument_count":
        region_body["args"].pop()
    elif mutation == "region_yield_type":
        region_body["ops"][-1]["operands"] = [body(entry)["args"][0]["id"]]
    elif mutation == "call_argument":
        invoke["operands"] = [body(entry)["args"][0]["id"]]
    elif mutation == "call_result":
        invoke["results"][0]["type"] = int32()
    elif mutation == "unknown_callee":
        invoke["attrs"]["callee"]["root"] = "missing"
    elif mutation == "nested_callee":
        invoke["attrs"]["callee"]["nested"] = ["nested"]
    elif mutation == "call_result_count":
        invoke["results"].append({"id": 12345678, "type": {"kind": "float", "format": "f32"}})
    elif mutation == "region_extra_block":
        gen["regions"][0]["blocks"].append(copy.deepcopy(region_body))
    elif mutation == "region_missing_block":
        gen["regions"][0]["blocks"] = []
    elif mutation == "function_extra_region":
        leaf["regions"].append(copy.deepcopy(leaf["regions"][0]))
    elif mutation == "terminator_result":
        lb["ops"][-1]["results"] = [{"id": 55555, "type": int32()}]
    else:
        lb["ops"].append(copy.deepcopy(lb["ops"][0]))
    result = check_doc(tmp_path, doc)
    assert not result["program_typing_checked"], mutation
    assert not result["source_typing_discharged"], mutation


def test_external_contract_stays_explicit(tmp_path, exported):
    leaf = functions(exported)[0]
    leaf["regions"] = []
    result = check_doc(tmp_path, exported)
    assert result["program_typing_checked"]
    assert result["external_declarations"] == 1
    assert not result["closed_capture_checked"] and not result["source_typing_discharged"]


def test_kernel_checked_typing_and_poison_distinction():
    fixture = repo_root() / "merlin/tests/fixtures/lean_scalar_square_typing.lean"
    output = _run(["lake", "env", "lean", str(fixture)])
    assert "sorryAx" not in output


@pytest.mark.slow
def test_tiny_llama_retained_typing_replays(tmp_path, lean_tools):
    from merlin.verify.lean.scalar_square_typing import replay_typing

    path = artifacts_dir() / "verification/merlin-passes/frontend/scalar-square/typing/SY_model_tiny_llama/receipt.json"
    result = replay_typing(path, tmp_path / "replay", compiler_python=lean_tools.m2m_python)
    assert result["fresh"] and result["matches_recorded"]
    assert result["source_typing_discharged"]
    assert not result["source_definedness_proved"] and not result["policy_approved"]


def test_f64_call_and_region_typing(tmp_path, lean_tools):
    mlir = tmp_path / "f64.mlir"
    mlir.write_text(PROGRAM.replace("f32", "f64"))
    export = tmp_path / "f64.json"
    export_file(mlir, export, backend="mlir", m2m_python=lean_tools.m2m_python)
    assert run(export)["source_typing_discharged"]


@pytest.mark.parametrize("mutation", ["none", "late_nested_result", "late_region_argument", "quant_result"])
def test_complete_actual_capture_checked_after_unsupported_prefix(tmp_path, mutation):
    base = artifacts_dir() / "verification/merlin-passes/frontend/scalar-square/SY_model_tiny_llama/receipt.json"
    receipt = json.loads(base.read_text())
    doc = json.loads(Path(receipt["exports"]["before"]["path"]).read_text())
    ops = body(functions(doc)[0])["ops"]
    assert any(o["name"].startswith("quant_ext.") for o in ops)
    if mutation == "late_nested_result":
        late = next(o for o in reversed(ops) if o["name"] == "linalg.generic")
        b = late["regions"][0]["blocks"][0]
        nested = next(o for o in b["ops"] if o["results"])
        nested["results"][0]["type"] = int32()
    elif mutation == "late_region_argument":
        late = next(o for o in reversed(ops) if o["name"] == "linalg.generic")
        late["regions"][0]["blocks"][0]["args"][0]["type"] = int32()
    elif mutation == "quant_result":
        quant = next(o for o in ops if o["name"] == "quant_ext.dequantize_per_tensor")
        quant["results"][0]["type"]["element"] = int32()
    result = check_doc(tmp_path, doc)
    assert result["source_typing_discharged"] is (mutation == "none")
    assert result["operations_including_functions_and_terminators"] == 10133


@pytest.mark.parametrize("field", ["inputs", "results"])
def test_invalid_external_call_signature_rejected(tmp_path, exported, field):
    leaf = functions(exported)[0]
    leaf["regions"] = []
    leaf["attrs"]["function_type"]["value"][field][0] = int32()
    result = check_doc(tmp_path, exported)
    assert not result["program_typing_checked"]
    assert not result["source_typing_discharged"]


def test_duplicate_function_signature_rejected(tmp_path, exported):
    functions(exported)[1]["attrs"]["sym_name"]["value"] = "leaf"
    assert not check_doc(tmp_path, exported)["program_typing_checked"]
