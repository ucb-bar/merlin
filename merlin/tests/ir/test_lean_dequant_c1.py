"""Real dequantization lowering through source support, C1, and authoritative MLIR export."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_dequant_scalar import dequant_source, legacy_channel_body

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_xdsl import lower_quant_ext
from merlin.verify.lean.export import export_file


def export_pair(tmp_path, tools, kind, shape=(2,), axis=0):
    module = dequant_source(kind, shape, axis)
    before = tmp_path / "before.mlir"
    before.write_text(str(module))
    assert lower_quant_ext(module) == 1
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
        ("per_channel", (0, 3), 0),
        ("per_channel", (2, 3, 4), 2),
        ("per_channel", (2, 3), 0),
    ],
)
def test_source_support_and_whole_function_c1(tmp_path, lean_tools, kind, shape, axis):
    before, after = export_pair(tmp_path, lean_tools, kind, shape, axis)
    support = invoke(lean_tools, "check-support", before, "forward")
    assert support["verdict"] == "verified", support
    assert invoke(lean_tools, "check-support", after, "forward")["verdict"] == "verified"
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


@pytest.mark.parametrize("bounds", [(-128, 127), (0, 255), (-128, 126)])
def test_per_tensor_dequantization_bounds_are_metadata_only(tmp_path, lean_tools, bounds):
    before, after = export_pair(tmp_path, lean_tools, "per_tensor")
    document = json.loads(before.read_text())
    source = document["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"][0]
    for key, value in zip(("quant_min", "quant_max"), bounds, strict=True):
        source["props"][key] = {"kind": "int", "value": str(value), "type": {"kind": "int", "width": 64}}
    before.write_text(json.dumps(document))
    # PyTorch's per-tensor dequantizer does not read either bound; per-channel
    # validation is exercised separately in test_lean_dequant_contract.py.
    assert invoke(lean_tools, "check-support", before, "forward")["verdict"] == "verified"
    assert invoke(lean_tools, "check-c1", before, after, "forward")["verdict"] == "verified"


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


@pytest.mark.parametrize("dtype,supported", [("i8", True), ("ui8", False), ("i16", False)])
def test_capture_storage_dtype_is_checked(tmp_path, lean_tools, dtype, supported):
    before, after = export_pair(tmp_path, lean_tools, "per_tensor")
    document = json.loads(before.read_text())
    source = document["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"][0]
    source["props"]["input_dtype"] = {"kind": "string", "value": dtype}
    before.write_text(json.dumps(document))
    assert (invoke(lean_tools, "check-support", before, "forward")["verdict"] == "verified") == supported
    assert (invoke(lean_tools, "check-c1", before, after, "forward")["verdict"] == "verified") == supported


@pytest.mark.parametrize("kind", ["per_tensor", "per_channel"])
@pytest.mark.parametrize("with_definedness", [False, True])
def test_real_preprocessing_receipts_compose(tmp_path, monkeypatch, lean_tools, kind, with_definedness):
    import jsonschema
    from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64

    from merlin.llvmlower.passes_xdsl import preprocess_text
    from merlin.verify.lean import stages
    from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain

    schema_root = repo_root() / "merlin" / "contract" / "schemas"
    receipt_schema = json.loads((schema_root / "lean_receipt.schema.json").read_text())
    chain_schema = json.loads((schema_root / "lean_chain.schema.json").read_text())
    module = dequant_source(kind, (2, 3), 1)
    source = next(op for op in module.walk() if getattr(op, "op_name", None) is not None)
    source.properties.update(
        {"input_dtype": StringAttr("i8"), "quant_min": IntegerAttr(-128, i64), "quant_max": IntegerAttr(127, i64)}
    )
    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "stages"))
    lowered_text, _ = preprocess_text(str(module))
    recorded = stages.completed(tmp_path / "stages")
    quant_stage = next(stage for stage in recorded if stage.name.endswith("xdsl-quant-lowered"))
    assert json.loads((quant_stage / "meta.json").read_text())["changed"]
    for stage in recorded:
        receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
        jsonschema.validate(receipt, receipt_schema)
        assert receipt["verdict"] == "verified", receipt.get("reason")
    capture = recorded[0] / "before.mlir"
    support = check_support(capture, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert support["verdict"] == "verified", support.get("reason")
    definedness = None
    if with_definedness:
        receipt = check_definedness(capture, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
        jsonschema.validate(receipt, receipt_schema)
        assert receipt["verdict"] == "verified", receipt.get("reason")
        definedness = tmp_path / "defined" / "receipt.json"
    handoff = tmp_path / "handoff.mlir"
    handoff.write_text(lowered_text)
    stages.record_handoff(handoff)
    chain = compose_chain(
        recorded,
        tmp_path / "support" / "receipt.json",
        "forward",
        tmp_path / "chain.json",
        handoff=tmp_path / "stages" / "handoff.json",
        m2m_python=lean_tools.m2m_python,
        definedness=definedness,
    )
    jsonschema.validate(chain, chain_schema)
    assert chain["refinement_verdict"] == "verified", chain.get("reason")
    assert chain["refinement_theorem"] == "MerlinVerify.Check.chain_refines"
    assert chain["verdict"] == ("verified" if with_definedness else "abstracted"), chain.get("reason")
    assert chain["source_definedness"]["verdict"] == ("verified" if with_definedness else "unmeasured")
    assert chain["replayed"] is True
    assert chain["chained"] is True
    # A previously verified lowering is not reusable after its emitted arithmetic changes.
    output = quant_stage / "after.mlir"
    text = output.read_text()
    needle = '"arith.subi"' if kind == "per_channel" else '"arith.subf"'
    replacement = '"arith.addi"' if kind == "per_channel" else '"arith.addf"'
    assert needle in text
    output.write_text(text.replace(needle, replacement, 1))
    broken = compose_chain(
        recorded,
        tmp_path / "support" / "receipt.json",
        "forward",
        tmp_path / "broken.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert broken["verdict"] == "abstracted"


def test_multiple_rewrite_sites_preserve_result_correspondence(tmp_path, lean_tools):
    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text("""func.func @forward(%q: tensor<2x3xi8>, %st: tensor<f32>, %zt: tensor<i32>,
        %sc: tensor<3xf32>, %zc: tensor<3xi32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
      %a = "quant_ext.dequantize_per_tensor"(%q,%st,%zt) :
        (tensor<2x3xi8>,tensor<f32>,tensor<i32>) -> tensor<2x3xf32>
      %b = "quant_ext.dequantize_per_channel"(%q,%sc,%zc) <{axis = 1 : i64}> :
        (tensor<2x3xi8>,tensor<3xf32>,tensor<3xi32>) -> tensor<2x3xf32>
      return %b, %a : tensor<2x3xf32>, tensor<2x3xf32>
    }""")
    paths = []
    for name in ("before", "after"):
        if name == "after":
            assert lower_quant_ext(module) == 2
        source, document = tmp_path / f"{name}.mlir", tmp_path / f"{name}.json"
        source.write_text(str(module))
        export_file(source, document, backend="mlir", m2m_python=lean_tools.m2m_python)
        paths.append(document)
    assert invoke(lean_tools, "check-support", paths[0], "forward")["verdict"] == "verified"
    result = invoke(lean_tools, "check-c1", *paths, "forward")
    assert result["verdict"] == "verified", result
    assert result["rule_sites"] == 2
