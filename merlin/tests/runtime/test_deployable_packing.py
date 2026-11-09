"""Only literal-preserving producers and proved byte permutations move to glue."""

import ctypes
import shutil
import subprocess

import pytest
from xdsl.dialects.builtin import StringAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_ir import checked_generic_text
from merlin.llvmlower.deployable_outline import plan_deployable_chain
from merlin.llvmlower.deployable_packing import canonicalize_constant_splats
from merlin.runtime.outlined_chain import _views

SOURCE = """module {
 func.func @forward(%a:tensor<2x3xi8>,%b:tensor<5x3xi8>)->tensor<2x5xi32> {
  %z=arith.constant 0:i32
  %init=tensor.splat %z:tensor<2x5xi32>
  %e=tensor.empty():tensor<3x5xi8>
  %bt=linalg.transpose ins(%b:tensor<5x3xi8>) outs(%e:tensor<3x5xi8>) permutation=[1,0]
  %c=linalg.generic {indexing_maps=[affine_map<(m,n,k)->(m,k)>,
    affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
    iterator_types=["parallel","parallel","reduction"]}
   ins(%a,%bt:tensor<2x3xi8>,tensor<3x5xi8>) outs(%init:tensor<2x5xi32>) {
  ^bb0(%x:i8,%y:i8,%acc:i32):
   %xx=arith.extsi %x:i8 to i32
   %yy=arith.extsi %y:i8 to i32
   %p=arith.muli %xx,%yy:i32
   %s=arith.addi %acc,%p:i32
   linalg.yield %s:i32
  }->tensor<2x5xi32>
  func.return %c:tensor<2x5xi32>
 }
}"""


def plan(module, slots=(0,), *, enabled=True):
    return plan_deployable_chain(
        module,
        worker_slots=slots,
        worker_count=2,
        workspace_budget_bytes=65536,
        alignment_bytes=64,
        data_preparation=enabled,
    )


def test_opt_in_preserves_source_and_binds_only_transpose_as_controller_copy():
    module = parse_mlir_text(SOURCE)
    before = checked_generic_text(module)
    with pytest.raises(ValueError):
        plan(module, (0, 1), enabled=False)
    result = plan(module)
    assert checked_generic_text(module) == before
    assert len(result.dispatches) == 1
    assert result.data_preparation["constant_splats_canonicalized"] == 1
    perm = result.data_preparation["controller_permutations"]
    assert len(perm) == 1 and perm[0]["source_strides"] == (1, 3)
    assert perm[0]["output_shape"] == (3, 5)
    assert result.dispatches[0]["views_before"] == perm
    kernel = next(
        op
        for op in result.outlined.module.body.block.ops
        if op.name == "func.func" and op.sym_name.data == result.dispatches[0]["symbol"]
    )
    assert len(kernel.body.block.args) == 2
    assert any(op.name == "linalg.fill" for op in kernel.walk())
    assert result.workspace_bytes <= 65536


@pytest.mark.parametrize("literal", ["-0.000000e+00", "0x7f800000", "0xff800000", "0x7fc12345", "1.1920929e-07"])
def test_constant_attribute_is_reused_without_numeric_reencoding(literal):
    module = parse_mlir_text(
        "module { func.func @f()->tensor<2xf32> {"
        f"%c=arith.constant {literal}:f32 "
        "%s=tensor.splat %c:tensor<2xf32> "
        "func.return %s:tensor<2xf32> }}"
    )
    constant = next(op for op in module.walk() if op.name == "arith.constant")
    value = constant.properties["value"]
    assert canonicalize_constant_splats(module) == 1
    fill = next(op for op in module.walk() if op.name == "linalg.fill")
    assert fill.inputs[0] is constant.results[0]
    assert constant.properties["value"] is value
    module.verify()


def test_unknown_contract_does_not_become_a_constant_or_controller_permutation():
    module = parse_mlir_text(SOURCE)
    splat = next(op for op in module.walk() if op.name == "tensor.splat")
    splat.attributes["unknown.contract"] = StringAttr("retained")
    assert canonicalize_constant_splats(module) == 0
    with pytest.raises(ValueError):
        plan(module)
    module = parse_mlir_text(SOURCE)
    transpose = next(op for op in module.walk() if op.name == "linalg.transpose")
    transpose.attributes["unknown.contract"] = StringAttr("retained")
    result = plan(module, (0, 1))
    assert result.data_preparation["controller_permutations"] == []
    assert len(result.dispatches) == 2


def test_quantization_inner_contract_grants_no_controller_packing():
    module = parse_mlir_text(SOURCE)
    transpose = next(op for op in module.walk() if op.name == "linalg.transpose")
    transpose.attributes["prov.quant_inner"] = StringAttr("owned quantization")
    result = plan(module, (0, 1))
    assert result.data_preparation["controller_permutations"] == []
    assert len(result.dispatches) == 2


def test_nonidentity_transpose_body_stays_arithmetic():
    module = parse_mlir_text(SOURCE)
    trans = next(op for op in module.walk() if op.name == "linalg.transpose")
    trans.body.block.last_op.operands = (trans.body.block.args[1],)
    result = plan(module, (0, 1))
    assert not result.data_preparation["controller_permutations"]
    assert len(result.dispatches) == 2


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_emitted_controller_byte_gather_preserves_every_bit_and_guards(tmp_path, width):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    view = {
        "kind": "pack_permutation",
        "input": "a",
        "output": "b",
        "output_shape": (3, 2),
        "source_strides": (1, 3),
        "element_bytes": width,
    }
    code = "#include <stddef.h>\n#include <string.h>\nstruct C{void*buffers[2];};\n"
    code += "void run(void*a,void*b){struct C value={{a,b}},*c=&value;\n"
    code += "\n".join(_views([view], {"a": 0, "b": 1})) + "\n}\n"
    src, lib = tmp_path / f"copy_{width}.c", tmp_path / f"copy_{width}.so"
    src.write_text(code)
    subprocess.run([compiler, "-shared", "-fPIC", "-O2", str(src), "-o", str(lib)], check=True, capture_output=True)
    payload = bytes((i * 73 + 19) % 256 for i in range(6 * width))
    source = ctypes.create_string_buffer(b"\x5a" * 64 + payload + b"\x5a" * 64)
    dest = ctypes.create_string_buffer(b"\x5a" * (6 * width + 128))
    function = ctypes.CDLL(str(lib)).run
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    function(ctypes.byref(source, 64), ctypes.byref(dest, 64))
    expected = b"".join(payload[(i * 3 + j) * width : (i * 3 + j + 1) * width] for j in range(3) for i in range(2))
    assert source.raw[: 128 + len(payload)] == b"\x5a" * 64 + payload + b"\x5a" * 64
    assert dest.raw[: 128 + len(payload)] == b"\x5a" * 64 + expected + b"\x5a" * 64
