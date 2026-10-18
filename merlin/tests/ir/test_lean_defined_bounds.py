"""Actual source-definedness from producer-proved static tensor index bounds."""

import subprocess

import pytest
from test_lean_reshape_compose import _canonicalize, _verify_mlir
from test_lean_reshape_typed import CAPTURE_MUTATE, _capture_source
from test_lean_splat_nested import _source as _splat_source
from test_lean_splat_nested import _stage as _splat_stage

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain


def _type(shape, element):
    return "tensor<" + "".join(f"{size}x" for size in shape) + element + ">"


def _source(shape=(2, 3), indices=(1, 2), element="i32", *, dynamic=False, dirty=False):
    tensor = _type(shape, element)
    arguments = f"%data: {tensor}" + (", %index: index" if dynamic else "")
    constants = "\n".join(f"%index{i} = arith.constant {value} : index" for i, value in enumerate(indices))
    coordinates = ", ".join(["%index"] * len(shape) if dynamic else [f"%index{i}" for i in range(len(indices))])
    return f"""func.func @subject({arguments}) -> {element} {{
      {"" if dynamic else constants}
      {f"%empty = tensor.empty() : {tensor}" if dirty else ""}
      %value = tensor.extract {"%empty" if dirty else "%data"}[{coordinates}] : {tensor}
      return %value : {element}
    }}"""


def _program(tmp_path, source, compiler):
    path = tmp_path / "program.mlir"
    path.write_text(source)
    _verify_mlir(path, compiler)
    return path


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize(
    "shape,indices,element",
    [
        ((2, 3), (1, 2), "i32"),
        ((1,), (0,), "f32"),
        ((2, 1, 3), (1, 0, 2), "i13"),
        ((), (), "f64"),
        ((2,), (1,), "index"),
    ],
)
def test_static_bounds_prove_actual_extraction(tmp_path, lean_tools, index_bits, shape, indices, element):
    path = _program(tmp_path, _source(shape, indices, element), lean_tools.m2m_python)
    result = check_support(
        path, "subject", tmp_path / "support", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize(
    "shape,indices,options",
    [
        ((2, 3), (-1, 0), {}),
        ((2, 3), (2, 0), {}),
        ((2, 3), (0, 3), {}),
        ((0,), (0,), {}),
        ((2,), (0,), {"dynamic": True}),
        ((2,), (0,), {"dirty": True}),
    ],
)
def test_unsupported_bounds_do_not_claim_definedness(tmp_path, lean_tools, index_bits, shape, indices, options):
    path = _program(tmp_path, _source(shape, indices, **options), lean_tools.m2m_python)
    # Support permits modeled UB/poison, so this also prevents vacuous parser/type refusal.
    result = check_support(
        path, "subject", tmp_path / "support", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("n,shape", [(4294967297, (2,)), (4294967296, (1,)), (-4294967295, (2,))])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_index_constants_use_actual_wrapped_signed_width(tmp_path, lean_tools, n, shape, index_bits):
    path = _program(tmp_path, _source(shape, (n,)), lean_tools.m2m_python)
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == ("verified" if index_bits == 32 else "abstracted"), result.get("reason")


def _chain(tmp_path, stage, function, tools, index_bits):
    for checker, label in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(
            stage / "before.mlir", function, tmp_path / label, index_bits=index_bits, m2m_python=tools.m2m_python
        )
        assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        function,
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=tools.m2m_python,
    )
    assert result["verdict"] == "verified", result.get("reason")
    assert result["theorem"] == "MerlinVerify.Check.chain_typed_defined_refines"


def _captured_source(shape=(2, 3), coordinates=(1, 2), *, dirty=False, multiple=False):
    source = _splat_source(inner_shape=shape, tensor_capture=True, multiple=multiple, dirty=dirty)
    constants = "\n".join(f"%coordinate{i} = arith.constant {value} : index" for i, value in enumerate(coordinates))
    source = source.replace("%index = arith.constant 0 : index", constants)
    old = ", ".join(["%index"] * len(shape))
    new = ", ".join(f"%coordinate{i}" for i in range(len(shape)))
    return source.replace(old, new)


CAPTURE_CASES = [((2, 3), (0, 0)), ((2, 3), (1, 2)), ((1, 1), (0, 0)), ((2, 1, 3), (1, 0, 2))]


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("shape,coordinates", CAPTURE_CASES)
def test_actual_nested_splat_capture_gets_strict_defined_chain(
    tmp_path, lean_tools, index_bits, dirty, shape, coordinates
):
    source = _captured_source(shape, coordinates, multiple=True, dirty=dirty)
    stage, count = _splat_stage(tmp_path, source)
    assert count == 2
    _verify_mlir(stage / "before.mlir", lean_tools.m2m_python)
    _verify_mlir(stage / "after.mlir", lean_tools.m2m_python)
    _chain(tmp_path, stage, "subject", lean_tools, index_bits)


def _reshape_source():
    return _capture_source().replace("arith.constant 0 : index", "arith.constant 1 : index")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_reshape_capture_gets_strict_defined_chain(tmp_path, lean_tools, index_bits):
    stage, _, _ = _canonicalize(tmp_path, lean_tools.m2m_python, _reshape_source())
    _chain(tmp_path, stage, "forward", lean_tools, index_bits)


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    function = module.body.operations[0]
    block = function.regions[0].blocks[0]
    constant = next(op for op in block.operations if op.operation.name == 'arith.constant')
    constant.attributes['value'] = ir.IntegerAttr.get(ir.IndexType.get(), int(sys.argv[2]))
    module.operation.verify()
    Path(sys.argv[1]).write_text(str(module))
"""


@pytest.mark.parametrize("n", [-1, 1, 2])
def test_verified_nested_bounds_mutants_refused(tmp_path, lean_tools, n):
    stage, count = _splat_stage(tmp_path, _splat_source(tensor_capture=True))
    assert count == 1
    _chain(tmp_path, stage, "subject", lean_tools, 64)
    subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(stage / "after.mlir"), str(n)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    result = check_definedness(
        stage / "after.mlir", "subject", tmp_path / "mutant-defined", m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == ("verified" if n == 1 else "abstracted"), result.get("reason")
    result = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "subject",
        tmp_path / "stale.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert result["verdict"] == "abstracted", result.get("reason")


def test_changed_reshape_capture_retains_bounds_but_refuses_refinement(tmp_path, lean_tools):
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, _reshape_source())
    _chain(tmp_path, stage, "forward", lean_tools, 64)
    subprocess.run(
        [str(lean_tools.m2m_python), "-c", CAPTURE_MUTATE, str(after)], check=True, capture_output=True, text=True
    )
    result = check_definedness(after, "forward", tmp_path / "mutant-defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
