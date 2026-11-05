"""Prepared source lifetimes require actual effects, not purity labels."""

import pytest
from xdsl.dialects import arith, builtin, func, memref
from xdsl.dialects.linalg.ops import TransposeOp
from xdsl.ir import Block, Region

from merlin.llvmlower.prepared_operand_effects import validate_prepared_source_lifetime, validate_pure_tensor_function
from merlin.llvmlower.tensor_preparation_identity import TensorPreparationRequest, find_tensor_preparation_opportunities


def fixture():
    typ = builtin.TensorType(builtin.f32, [3, 17])
    body = Block(arg_types=[typ])
    body.add_op(func.ReturnOp(body.args[0]))
    reader = func.FuncOp("read", ([typ], [typ]), Region(body), visibility="private")
    block = Block(arg_types=[typ])
    calls = [func.CallOp("read", [block.args[0]], [typ]) for _ in range(4)]
    block.add_ops([*calls, func.ReturnOp(calls[-1].results[0])])
    owner = func.FuncOp("owner", ([typ], [typ]), Region(block))
    module = builtin.ModuleOp([reader, owner])
    (opportunity,) = find_tensor_preparation_opportunities(
        [TensorPreparationRequest(block.args[0], call, "1" * 64) for call in calls]
    )
    module.verify()
    return module, reader, owner, calls, opportunity


def test_complete_source_bodies_are_checked():
    module, _, _, _, opportunity = fixture()
    validate_prepared_source_lifetime(opportunity, module)


@pytest.mark.parametrize("mode", ["external", "recursive", "allocation"])
def test_opaque_recursive_or_buffer_body_refuses(mode):
    module, reader, _, _, opportunity = fixture()
    body = reader.body.block
    terminator = body.last_op
    if mode == "external":
        body.insert_op_before(func.CallOp("opaque", [body.args[0]], []), terminator)
    elif mode == "recursive":
        body.insert_op_before(func.CallOp("read", [body.args[0]], [body.args[0].type]), terminator)
    else:
        allocation = memref.AllocOp.get(builtin.f32, shape=[17])
        body.insert_op_before(allocation, terminator)
    # Re-discover after mutation: refusal is from live effect inspection,
    # independent of the source snapshot mismatch check.
    with pytest.raises(ValueError):
        validate_pure_tensor_function(reader, {"read": reader})


def test_unknown_effect_between_borrows_refuses():
    module, _, owner, calls, opportunity = fixture()
    owner.body.block.insert_op_before(func.CallOp("unknown_writer", [], []), calls[2])
    with pytest.raises(ValueError, match="unknown external effects"):
        validate_prepared_source_lifetime(opportunity, module)


def test_complete_source_declaration_is_not_a_purity_certificate():
    declaration = func.FuncOp("read", ([], []), Region(), visibility="private")
    with pytest.raises(ValueError, match="complete"):
        validate_pure_tensor_function(declaration, {"read": declaration})


def test_scoped_callee_never_borrows_a_flat_function_effect_proof():
    _, reader, owner, calls, _ = fixture()
    calls[0].properties["callee"] = builtin.SymbolRefAttr("read", ["nested"])
    with pytest.raises(ValueError, match="nested symbol"):
        validate_pure_tensor_function(owner, {"read": reader, "owner": owner})


def test_named_tensor_region_permission_does_not_grant_external_or_buffer_effects():
    typ = builtin.TensorType(builtin.f32, [3, 3])
    block = Block(arg_types=[typ, typ])
    transpose = TransposeOp(block.args[0], block.args[1], builtin.DenseArrayBase.from_list(builtin.i64, [1, 0]))
    block.add_ops([transpose, func.ReturnOp(transpose.results[0])])
    owner = func.FuncOp("owner", ([typ, typ], [typ]), Region(block))
    builtin.ModuleOp([owner]).verify()
    with pytest.raises(ValueError):
        validate_pure_tensor_function(owner, {"owner": owner})
    validate_pure_tensor_function(owner, {"owner": owner}, allow_named_linalg=True)
    with pytest.raises(ValueError, match="explicit boolean"):
        validate_pure_tensor_function(owner, {"owner": owner}, allow_named_linalg=1)
    transpose.attributes["library_call"] = builtin.StringAttr("opaque")
    with pytest.raises(ValueError, match="unknown external"):
        validate_pure_tensor_function(owner, {"owner": owner}, allow_named_linalg=True)
    del transpose.attributes["library_call"]
    allocation = memref.AllocOp.get(builtin.f32, shape=[3, 3])
    block.insert_op_before(allocation, transpose)
    with pytest.raises(ValueError):
        validate_pure_tensor_function(owner, {"owner": owner}, allow_named_linalg=True)


def test_standard_typed_integer_widening_needs_no_unknown_effect_assumption():
    block = Block(arg_types=[builtin.i8])
    widening = arith.ExtSIOp(block.args[0], builtin.i32)
    block.add_ops([widening, func.ReturnOp(widening.result)])
    function = func.FuncOp("widen", ([builtin.i8], [builtin.i32]), Region(block))
    builtin.ModuleOp([function]).verify()
    validate_pure_tensor_function(function, {"widen": function})
    widening.attributes["execution.override"] = builtin.StringAttr("unknown")
    with pytest.raises(ValueError, match="unknown"):
        validate_pure_tensor_function(function, {"widen": function})


@pytest.mark.parametrize("axis", [0, 1])
def test_tensor_iteration_coordinate_has_no_memory_effects(axis):
    from xdsl.dialects.linalg.ops import IndexOp

    from merlin.common import mlir_query as mq

    module = mq.parse(f"""func.func @coordinate(%x:tensor<3x7xi64>)->tensor<3x7xi64>{{
      %out=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>],
        iterator_types=["parallel","parallel"]}} outs(%x:tensor<3x7xi64>) {{
        ^bb0(%unused:i64):
        %index=linalg.index {axis}:index
        %value=arith.index_cast %index:index to i64
        linalg.yield %value:i64
      }}->tensor<3x7xi64>
      return %out:tensor<3x7xi64>}}""")
    module.verify()
    function = module.body.block.first_op
    validate_pure_tensor_function(function, {"coordinate": function})
    index = next(op for op in module.walk() if type(op) is IndexOp)
    index.attributes["execution.override"] = builtin.StringAttr("unknown")
    with pytest.raises(ValueError, match="ordinary tensor"):
        validate_pure_tensor_function(function, {"coordinate": function})
    del index.attributes["execution.override"]
    index.properties["dim"] = builtin.IntegerAttr(2, builtin.i64)
    with pytest.raises(ValueError, match="valid axis"):
        validate_pure_tensor_function(function, {"coordinate": function})


def test_iteration_coordinate_outside_tensor_generic_refuses():
    from xdsl.dialects.linalg.ops import IndexOp

    block = Block()
    index = IndexOp(0)
    block.add_ops([index, func.ReturnOp(index.result)])
    function = func.FuncOp("outside", ([], [builtin.IndexType()]), Region(block))
    with pytest.raises(ValueError, match="tensor generic"):
        validate_pure_tensor_function(function, {"outside": function})


@pytest.mark.parametrize("control", ["for", "if"])
def test_serial_tensor_control_checks_each_nested_effect(control):
    from xdsl.dialects import scf

    from merlin.common import mlir_query as mq

    loop = """%lo=arith.constant 0:index
      %hi=arith.constant 3:index
      %step=arith.constant 1:index
      %out=scf.for %i=%lo to %hi step %step iter_args(%v=%x)->(tensor<3xi8>) {
        scf.yield %v:tensor<3xi8>
      }"""
    conditional = """%flag=arith.constant true
      %out=scf.if %flag -> (tensor<3xi8>) {
        scf.yield %x:tensor<3xi8>
      } else {scf.yield %x:tensor<3xi8>}"""
    module = mq.parse(
        "func.func @owner(%x:tensor<3xi8>)->tensor<3xi8>{"
        + (loop if control == "for" else conditional)
        + " return %out:tensor<3xi8>}"
    )
    module.verify()
    owner = module.body.block.first_op
    functions = {"owner": owner}
    validate_pure_tensor_function(owner, functions)
    structured = next(op for op in owner.walk() if isinstance(op, (scf.ForOp, scf.IfOp)))
    structured.attributes["execution.override"] = builtin.UnitAttr()
    with pytest.raises(ValueError, match="unknown execution"):
        validate_pure_tensor_function(owner, functions)
    del structured.attributes["execution.override"]
    body = structured.regions[0].block
    module.body.block.add_op(func.FuncOp.external("unknown_writer", [], []))
    body.insert_op_before(func.CallOp("unknown_writer", [], []), body.last_op)
    with pytest.raises(ValueError, match="unknown external"):
        validate_pure_tensor_function(owner, functions)
