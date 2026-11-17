"""Public source words supply runtime sizes only through independent body proof."""

import json

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.frontends.model_shape_facts import WORD_PROFILE, ShapeFacts
from merlin.frontends.shape_expressions import Program
from merlin.llvmlower.runtime_invocation import resolve

SOURCE = """module {func.func @forward(%count:tensor<i32>)->tensor<?xi32> {
  %raw="tensor.extract"(%count):(tensor<i32>)->i32
  %n=arith.extsi %raw:i32 to i64
  %zero=arith.constant 0:i64
  %high=arith.constant 31:i64
  %lo=arith.cmpi sge,%n,%zero:i64
  %hi=arith.cmpi sle,%n,%high:i64
  cf.assert %lo,"source lower bound"
  cf.assert %hi,"source upper bound"
  %size=arith.index_cast %n:i64 to index
  %out=tensor.empty(%size):tensor<?xi32>
  return %out:tensor<?xi32>
}}"""


def example(root, n=7, native=False):
    root.mkdir()
    source = (
        SOURCE
        if not native
        else SOURCE.replace("%count:tensor<i32>", "%raw:i32").replace(
            '%raw="tensor.extract"(%count):(tensor<i32>)->i32', ""
        )
    )
    (root / "model.mlir").write_text(source)
    function = next(iter(parse_mlir_text(source).ops))
    facts = ShapeFacts.entry(function, expressions=True, data_words=True)
    contract = facts.contract(function)
    assert contract["schema"] == "t2m.runtime-shapes.v4"
    signature = {
        **contract,
        "schema": "merlin.compile-signature.v5",
        "entry": "forward",
        "model_shape_profile": WORD_PROFILE,
        "shape_scope": "source-proved-input-words-and-dimensions",
    }
    (root / "signature.json").write_text(json.dumps(signature))
    np.savez(root / "inputs.npz", in0=np.asarray(n, np.int32))
    return root, signature


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("n", (0, 1, 7, 31))
def test_runtime_storage_words_size_descriptors_without_specializing_source(tmp_path, n, native):
    root, _signature = example(tmp_path / "source", n, native)
    original = (root / "model.mlir").read_bytes()
    plan = resolve(root / "model.mlir", root / "signature.json", root / "inputs.npz")
    assert plan["outputs"] == [{"shape": [n], "dtype": "i32"}]
    assert (root / "model.mlir").read_bytes() == original
    assert plan["mlir_argument_specialization"] == "not-performed"


@pytest.mark.parametrize(
    "damage", ("binding", "program", "profile", "old-schema", "result", "guard", "late", "truncation")
)
def test_rehashed_word_contract_and_incomplete_guards_never_supply_shape_authority(tmp_path, damage):
    root, signature = example(tmp_path / "source")
    if damage == "binding":
        signature["word_bindings"]["w0"]["indices"] = [0]
    elif damage == "program":
        program = Program(signature["shape_program"]["nodes"])
        signature["outputs"][0]["shape"] = [program.dimension(7)]
        signature["shape_program"] = program.export(signature["outputs"][0]["shape"], word_profile=True)
    elif damage == "profile":
        signature["model_shape_profile"] = "t2m.symbolic-model-shapes.v2"
    elif damage == "old-schema":
        signature["schema"] = "merlin.compile-signature.v4"
    elif damage == "result":
        signature["outputs"][0]["shape"] = [1]
    elif damage == "guard":
        (root / "model.mlir").write_text(SOURCE.replace('cf.assert %lo,"source lower bound"', ""))
    elif damage == "late":
        source = SOURCE.replace('cf.assert %lo,"source lower bound"', "")
        source = source.replace("return %out:", 'cf.assert %lo,"late bound"\n return %out:')
        (root / "model.mlir").write_text(source)
    else:
        source = SOURCE.replace(
            "%n=arith.extsi %raw:i32 to i64", "%short=arith.trunci %raw:i32 to i8\n %n=arith.extsi %short:i8 to i64"
        )
        (root / "model.mlir").write_text(source)
    (root / "signature.json").write_text(json.dumps(signature))
    with pytest.raises(ValueError):
        resolve(root / "model.mlir", root / "signature.json", root / "inputs.npz")
