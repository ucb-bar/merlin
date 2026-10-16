"""Real dequantization lowering through source support, C1, and authoritative MLIR export."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_dequant_scalar import corrected_channel_body, dequant_source, legacy_channel_body

from merlin.llvmlower.passes_xdsl import lower_quant_ext
from merlin.verify.lean.export import export_file


def export_pair(tmp_path, tools, kind, shape=(2,), axis=0, correct_channel=False):
    module = dequant_source(kind, shape, axis)
    before = tmp_path / "before.mlir"
    before.write_text(str(module))
    assert lower_quant_ext(module) == 1
    if correct_channel:
        generic = next(op for op in module.walk() if op.name == "linalg.generic")
        corrected_channel_body(generic)
    module.verify()
    after = tmp_path / "after.mlir"
    after.write_text(str(module))
    documents = []
    for path in (before, after):
        document = path.with_suffix(".json")
        export_file(path, document, backend="mlir", m2m_python=tools.m2m_python)
        documents.append(document)
    return documents


def invoke(tools, *args):
    proc = subprocess.run([str(tools.lean_exe), *map(str, args)], capture_output=True, text=True, check=False)
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.parametrize(
    "kind,shape,axis",
    [
        ("per_tensor", (), 0),
        ("per_tensor", (2,), 0),
        ("per_tensor", (2, 3), 0),
        ("per_tensor", (0, 3), 0),
        ("per_channel", (2,), 0),
        ("per_channel", (2, 3), 1),
    ],
)
def test_source_support_and_whole_function_c1(tmp_path, lean_tools, kind, shape, axis):
    before, after = export_pair(tmp_path, lean_tools, kind, shape, axis, correct_channel=False)
    support = invoke(lean_tools, "check-support", before, "forward")
    assert support["verdict"] == "verified", support
    result = invoke(lean_tools, "check-c1", before, after, "forward")
    assert result["verdict"] == "verified", result
    assert result["rule_sites"] == 1
    result = invoke(lean_tools, "check-module", before, after)
    assert result["verdict"] == "verified", result


def test_legacy_per_channel_bug_is_refused(tmp_path, lean_tools):
    from merlin.frontends.linalg_mlir import parse_mlir_text

    before, after = export_pair(tmp_path, lean_tools, "per_channel")
    path = tmp_path / "after.mlir"
    module = parse_mlir_text(path.read_text())
    legacy_channel_body(next(op for op in module.walk() if op.name == "linalg.generic"))
    path.write_text(str(module))
    export_file(path, after, backend="mlir", m2m_python=lean_tools.m2m_python)
    assert invoke(lean_tools, "check-support", before, "forward")["verdict"] == "verified"
    assert invoke(lean_tools, "check-c1", before, after, "forward")["verdict"] == "abstracted"


@pytest.mark.parametrize(
    "mutation",
    [
        "input_wiring",
        "destination_wiring",
        "body_wiring",
        "return_temporary",
        "result_collision",
        "temporary_collision",
        "output_map",
        "reduction",
        "fastmath",
        "unknown_attr",
        "drop_empty",
    ],
)
def test_c1_rejects_corrupted_whole_function(tmp_path, lean_tools, mutation):
    before, after = export_pair(tmp_path, lean_tools, "per_tensor")
    document = json.loads(after.read_text())
    block = document["module"]["ops"][0]["regions"][0]["blocks"][0]
    empty, generic, returned = block["ops"]
    body = generic["regions"][0]["blocks"][0]
    if mutation == "input_wiring":
        generic["operands"][1:3] = generic["operands"][2:0:-1]
    elif mutation == "destination_wiring":
        generic["operands"][-1] = block["args"][0]["id"]
    elif mutation == "body_wiring":
        body["ops"][3]["operands"][1] = body["args"][3]["id"]
    elif mutation == "return_temporary":
        returned["operands"] = [empty["results"][0]["id"]]
    elif mutation == "result_collision":
        generic["results"][0]["id"] = block["args"][0]["id"]
    elif mutation == "temporary_collision":
        empty["results"][0]["id"] = block["args"][0]["id"]
        generic["operands"][-1] = block["args"][0]["id"]
    elif mutation == "output_map":
        maps = generic["attrs"]["indexing_maps"]["elements"]
        maps[-1] = maps[1]
    elif mutation == "reduction":
        generic["attrs"]["iterator_types"]["elements"][0]["value"] = "reduction"
    elif mutation == "fastmath":
        body["ops"][2]["attrs"]["fastmath"]["value"] = "reassoc"
    elif mutation == "unknown_attr":
        generic["attrs"]["numeric_contract"] = {"kind": "string", "value": "different"}
    else:
        block["ops"].remove(empty)
    after.write_text(json.dumps(document))
    assert invoke(lean_tools, "check-c1", before, after, "forward")["verdict"] == "abstracted"


def test_source_support_refuses_unsigned_input(tmp_path, lean_tools):
    before, _ = export_pair(tmp_path, lean_tools, "per_tensor")
    data = json.loads(before.read_text())
    block = data["module"]["ops"][0]["regions"][0]["blocks"][0]
    block["args"][0]["type"]["element"]["kind"] = "uint"
    before.write_text(json.dumps(data))
    assert invoke(lean_tools, "check-support", before, "forward")["verdict"] == "unsupported"


@pytest.mark.parametrize("bounds,supported", [((-128, 127), True), ((0, 255), False), ((-128, 126), False)])
def test_source_quantization_bounds_are_checked(tmp_path, lean_tools, bounds, supported):
    before, after = export_pair(tmp_path, lean_tools, "per_tensor")
    document = json.loads(before.read_text())
    source = document["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"][0]
    for key, value in zip(("quant_min", "quant_max"), bounds, strict=True):
        source["props"][key] = {"kind": "int", "value": str(value), "type": {"kind": "int", "width": 64}}
    before.write_text(json.dumps(document))
    assert (invoke(lean_tools, "check-support", before, "forward")["verdict"] == "verified") == supported
    assert (invoke(lean_tools, "check-c1", before, after, "forward")["verdict"] == "verified") == supported


def test_textual_per_channel_lowering_verifies(tmp_path, lean_tools):
    from merlin.llvmlower.passes_xdsl import preprocess_text_textual

    before_text = str(dequant_source("per_channel", (2, 3), 1))
    after_text, stats = preprocess_text_textual(before_text)
    assert stats["dequantize_lowered"] == 1
    documents = []
    for name, text in (("before", before_text), ("after", after_text)):
        source, document = tmp_path / f"{name}.mlir", tmp_path / f"{name}.json"
        source.write_text(text)
        export_file(source, document, backend="mlir", m2m_python=lean_tools.m2m_python)
        documents.append(document)
    assert invoke(lean_tools, "check-support", documents[0], "forward")["verdict"] == "verified"
    assert invoke(lean_tools, "check-c1", *documents, "forward")["verdict"] == "verified"
