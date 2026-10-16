"""Typed map rewriting uses checked SSA facts and preserves the receipt/chain input domain."""

import json
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.verify.lean.check import check_definedness, check_stage, check_support, check_typed_stage, compose_chain


def _source(shape="2x3xi32", scalar="i32", inputs=2, *, dirty=False, read=True, shorthand=False,
            constant_first=True):
    ty = f"tensor<{shape}>"
    args = [f"%a{i}: {ty}" for i in range(inputs)]
    if not dirty:
        args.append(f"%init: {ty}")
    ins = ("ins(" + ", ".join(f"%a{i}" for i in range(inputs)) + " : "
           + ", ".join([ty] * inputs) + ")") if inputs else ""
    block_args = ", ".join([f"%x{i}: {scalar}" for i in range(inputs)] + [f"%old: {scalar}"])
    prefix = f"%init = tensor.empty() : {ty}" if dirty else ""
    if not inputs:
        constant = "%captured = arith.constant 7 : i32"
        prefix = f"{constant}\n{prefix}" if constant_first else f"{prefix}\n{constant}"
    value = "%old" if read else "%x0" if inputs else "%captured"
    payload = "{ arith.addi }" if shorthand else ""
    body = "" if shorthand else f"({block_args}) {{ linalg.yield {value} : {scalar} }}"
    return f"""func.func @forward({', '.join(args)}) -> {ty} {{
      {prefix}
      %r = linalg.map {payload} {ins} outs(%init : {ty}) {body}
      return %r : {ty}
    }}"""


def _stage(tmp_path, lean_tools, source):
    stage = tmp_path / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text(source)
    program = """
import sys
from pathlib import Path
from torch_mlir import ir, passmanager
with ir.Context():
    m = ir.Module.parse(Path(sys.argv[1]).read_text())
    passmanager.PassManager.parse('builtin.module(func.func(linalg-generalize-named-ops))').run(m.operation)
    Path(sys.argv[2]).write_text(str(m))
"""
    result = subprocess.run([str(lean_tools.m2m_python), "-c", program, str(before), str(after)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "linalg.generic" in after.read_text()
    (stage / "meta.json").write_text(json.dumps({"boundary": "typed-map-generalization"}))
    return stage


@pytest.mark.parametrize("shape,scalar,inputs,dirty,read,shorthand", [
    ("2x3xi32", "i32", 0, False, True, False),
    ("2x3xi32", "i32", 1, False, True, False),
    ("2x3xi32", "i32", 2, False, False, False),
    ("i32", "i32", 1, False, True, False),
    ("0x3xi32", "i32", 2, True, False, False),
    ("2xf32", "f32", 1, False, True, False),
    ("2xi32", "i32", 0, True, False, False),
    ("2xi32", "i32", 2, True, False, True),
])
def test_actual_typed_map_generalization(tmp_path, lean_tools, shape, scalar, inputs, dirty, read, shorthand):
    import jsonschema

    stage = _stage(tmp_path, lean_tools, _source(shape, scalar, inputs, dirty=dirty, read=read,
                                               shorthand=shorthand))
    receipt = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    schema = repo_root() / "merlin/contract/schemas/lean_receipt.schema.json"
    jsonschema.validate(receipt, json.loads(schema.read_text()))
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert receipt["theorem"] == "MerlinVerify.Check.TypedModule.check_sound"
    assert receipt["index_bits"] == 64
    assert receipt["domain"] == "inputs of the declared argument types; contexts with 64-bit index"
    assert receipt["axioms"] == ["propext", "Classical.choice", "Quot.sound"]


@pytest.mark.parametrize("position", ["before", "after"])
def test_mixed_typed_and_universal_chain(tmp_path, lean_tools, position):
    import jsonschema

    stage = _stage(tmp_path, lean_tools, _source(dirty=True, read=False))
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    identity = tmp_path / "identity"
    identity.mkdir()
    for name in ["before.mlir", "after.mlir"]:
        (identity / name).write_bytes((stage / f"{position}.mlir").read_bytes())
    (identity / "meta.json").write_text(json.dumps({"boundary": "universal-identity"}))
    assert check_stage(identity, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    source = stage / "before.mlir"
    assert check_support(source, "forward", tmp_path / "support",
                         m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    assert check_definedness(source, "forward", tmp_path / "defined",
                             m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    stages = [identity, stage] if position == "before" else [stage, identity]
    chain = compose_chain(stages, tmp_path / "support/receipt.json", "forward", tmp_path / "chain.json",
                          m2m_python=lean_tools.m2m_python, definedness=tmp_path / "defined/receipt.json")
    jsonschema.validate(chain, json.loads((repo_root() / "merlin/contract/schemas/lean_chain.schema.json").read_text()))
    assert chain["verdict"] == chain["refinement_verdict"] == "verified", chain.get("reason")
    assert chain["theorem"] == "MerlinVerify.Check.chain_typed_defined_refines"
    assert chain["refinement_theorem"] == "MerlinVerify.Check.chain_typed_refines"
    assert chain["index_bits"] == 64


@pytest.mark.parametrize("fault", ["body", "input_map", "output_map"])
def test_typed_map_rejects_corrupted_generalization(tmp_path, lean_tools, fault):
    stage = _stage(tmp_path, lean_tools, _source("2x2xi32", shorthand=True))
    after = stage / "after.mlir"
    text = after.read_text()
    if fault == "body":
        corrupt = text.replace("arith.addi", "arith.subi")
    else:
        # Replace only one identity-map use, retaining legal square tensor types.
        line = next(line for line in text.splitlines() if "indexing_maps" in line)
        key = line.split("indexing_maps = [", 1)[1].split(",", 1)[0]
        maps = f"{key}, {key}, {key}"
        swapped = "affine_map<(d0, d1) -> (d1, d0)>"
        replacement = f"{swapped}, {key}, {key}" if fault == "input_map" else f"{key}, {key}, {swapped}"
        corrupt = text.replace(maps, replacement)
    assert corrupt != text
    after.write_text(corrupt)
    receipt = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_constant_motion_is_a_separate_rewrite_obligation(tmp_path, lean_tools):
    # MLIR's rewrite driver hoists this constant across tensor.empty. Map expansion preserves
    # source operation order; the independent motion must have its own proof before acceptance.
    stage = _stage(tmp_path, lean_tools, _source("2xi32", inputs=0, dirty=True, read=False,
                                               constant_first=False))
    before, after = ((stage / name).read_text() for name in ("before.mlir", "after.mlir"))
    assert before.index("tensor.empty") < before.index("arith.constant")
    assert after.index("arith.constant") < after.index("tensor.empty")
    receipt = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def _source_receipts(tmp_path, lean_tools, stage, *, width=64):
    source = stage / "before.mlir"
    assert check_support(source, "forward", tmp_path / "support", index_bits=width,
                         m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    return check_definedness(source, "forward", tmp_path / "defined", index_bits=width,
                             m2m_python=lean_tools.m2m_python)


def _compose(tmp_path, lean_tools, stages):
    return compose_chain(stages, tmp_path / "support/receipt.json", "forward", tmp_path / "chain.json",
                         m2m_python=lean_tools.m2m_python, definedness=tmp_path / "defined/receipt.json")


@pytest.mark.parametrize("width", [32, 64])
def test_intermediate_typed_values_and_escaping_results(tmp_path, lean_tools, width):
    # Two maps of different ranks; the second destination is produced by a checked tensor.splat.
    # The first result both escapes and feeds the second map through a checked reshape.
    source = """func.func @forward(%a: tensor<2x3xi32>, %c: i32)
        -> (tensor<2x3xi32>, tensor<6xi32>) {
      %empty = tensor.empty() : tensor<2x3xi32>
      %r = linalg.map ins(%a : tensor<2x3xi32>) outs(%empty : tensor<2x3xi32>)
        (%x: i32, %old: i32) { linalg.yield %x : i32 }
      %flat = tensor.collapse_shape %r [[0, 1]] : tensor<2x3xi32> into tensor<6xi32>
      %init = tensor.splat %c : tensor<6xi32>
      %s = linalg.map ins(%flat : tensor<6xi32>) outs(%init : tensor<6xi32>)
        (%x: i32, %old: i32) { %sum = arith.addi %x, %old : i32 linalg.yield %sum : i32 }
      return %r, %s : tensor<2x3xi32>, tensor<6xi32>
    }"""
    stage = _stage(tmp_path, lean_tools, source)
    receipt = check_typed_stage(stage, index_bits=width, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert _source_receipts(tmp_path, lean_tools, stage, width=width)["verdict"] == "verified"
    chain = _compose(tmp_path, lean_tools, [stage])
    assert chain["verdict"] == "verified", chain.get("reason")
    assert chain["index_bits"] == width


@pytest.mark.parametrize("fault", ["domain", "width", "theorem", "universal", "source_bytes", "export_bytes"])
def test_typed_chain_rejects_wrong_domains_or_stale_evidence(tmp_path, lean_tools, fault):
    stage = _stage(tmp_path, lean_tools, _source())
    receipt = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert _source_receipts(tmp_path, lean_tools, stage)["verdict"] == "verified"
    if fault == "domain":
        receipt["domain"] = "all inputs, including untyped runtime values"
    elif fault == "width":
        receipt["index_bits"] = 32
        receipt["domain"] = "inputs of the declared argument types; contexts with 32-bit index"
    elif fault == "theorem":
        receipt["theorem"] = "MerlinVerify.Check.moduleCheck_sound"
    elif fault == "universal":
        receipt["checker"] = "MerlinVerify.Check.moduleCheck"
        receipt["theorem"] = "MerlinVerify.Check.moduleCheck_sound"
    elif fault == "source_bytes":
        before = stage / "before.mlir"
        before.write_text(before.read_text() + "\n")
    else:
        from pathlib import Path

        exported = Path(receipt["exports"]["before"]["path"])
        exported.write_text(exported.read_text() + "\n")
    (stage / "receipt.json").write_text(json.dumps(receipt))
    chain = _compose(tmp_path, lean_tools, [stage])
    assert chain["verdict"] == "abstracted", chain.get("reason")
    assert chain["refinement_verdict"] == "abstracted"


def test_typed_rewrite_does_not_establish_source_definedness(tmp_path, lean_tools):
    stage = _stage(tmp_path, lean_tools, _source(dirty=True, read=True))
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    assert _source_receipts(tmp_path, lean_tools, stage)["verdict"] == "abstracted"
    chain = _compose(tmp_path, lean_tools, [stage])
    assert chain["refinement_verdict"] == "verified", chain.get("reason")
    assert chain["source_definedness"]["verdict"] == chain["verdict"] == "abstracted"


def test_distinct_tensor_shapes_cannot_share_a_typed_chain_domain(tmp_path, lean_tools):
    stage = _stage(tmp_path, lean_tools, _source())
    other = tmp_path / "other"
    other.mkdir()
    second = _stage(other, lean_tools, _source("3x2xi32"))
    for boundary in (stage, second):
        assert check_typed_stage(boundary, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    assert _source_receipts(tmp_path, lean_tools, stage)["verdict"] == "verified"
    chain = _compose(tmp_path, lean_tools, [stage, second])
    assert chain["verdict"] == "abstracted"
    assert "argument types or tensor shapes" in chain["reason"]


def test_function_annotations_do_not_establish_block_argument_types(tmp_path, lean_tools):
    from pathlib import Path

    stage = _stage(tmp_path, lean_tools, _source())
    receipt = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified"
    exports = []
    for key in ("before", "after"):
        doc = json.loads(Path(receipt["exports"][key]["path"]).read_text())
        function = next(op for op in doc["module"]["ops"] if op["name"] == "func.func")
        # Change both signatures consistently, while actual block arguments remain tensor<2x3>.
        function["attrs"]["function_type"]["value"]["inputs"][0]["shape"][0] = 7
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(doc))
        exports.append(str(path))
    result = subprocess.run([str(lean_tools.lean_exe), "check-module-typed", *exports, "64"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stderr
    answer = json.loads(result.stdout)
    assert answer["verdict"] == "abstracted"
    assert not answer["functions_verified"]
    assert not answer["unverified_changes"]


@pytest.mark.parametrize("width", [0, -1, True, "64"])
def test_typed_boundary_rejects_invalid_index_width(tmp_path, lean_tools, width):
    with pytest.raises(ValueError, match="positive integer"):
        check_typed_stage(tmp_path, m2m_python=lean_tools.m2m_python, index_bits=width)
