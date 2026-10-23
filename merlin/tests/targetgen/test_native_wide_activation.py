"""The generated storage codecs preserve ReLU for every signed word."""

import pytest
from mlir_fixtures import generic

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen import native_byte_activation
from merlin.targetgen.native_wide_activation import prepare


def source(width, *, threshold=0, effect=False, scalar=False):
    typ = f"tensor<{'4x' if not scalar else ''}i{width}>"
    body = f"""%zero=arith.constant {threshold} : i{width}
      %p=arith.cmpi slt,%a,%zero : i{width}
      %v=arith.select %p,%zero,%a : i{width}"""
    effect_op = '"fixture.effect"() : () -> ()' if effect else ""
    return parse_mlir_text(f"""module {{func.func @forward(%x:{typ}) -> ({typ},{typ}) {{
      {generic([("%x", typ)], typ, body)}
      {effect_op}
      return %r,%x : {typ},{typ}
    }} }}""")


def evaluate(block, arguments):
    import z3

    values = dict(zip(block.args, arguments, strict=True))
    for op in block.ops:
        if op.name == "linalg.yield":
            return values[op.operands[0]]
        width = op.results[0].type.width.data
        if op.name == "arith.constant":
            value = z3.BitVecVal(op.value.value.data, width)
        else:
            args = [values[v] for v in op.operands]
            if op.name == "arith.trunci":
                value = z3.Extract(width - 1, 0, args[0])
            elif op.name == "arith.extui":
                value = z3.ZeroExt(width - args[0].size(), args[0])
            elif op.name == "arith.shli":
                value = args[0] << args[1]
            elif op.name == "arith.shrui":
                value = z3.LShR(*args)
            elif op.name == "arith.andi":
                value = args[0] & args[1]
            elif op.name == "arith.ori":
                value = args[0] | args[1]
            else:
                pytest.fail(f"unproved host codec instruction: {op.name}")
        values[op.results[0]] = value
    pytest.fail("missing codec yield")


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("scalar", [False, True])
def test_generated_host_codecs_and_native_byte_relu_prove_full_word_domain(tmp_path, monkeypatch, width, scalar):
    import z3

    from merlin.targetgen.native_unit_policy import construction_evidence

    captured = []

    def compile(*args, **kwargs):
        module, root, data, binding = args[5:]
        module.verify()
        assert construction_evidence(module)["status"] == "eligible"
        assert data.owner.name == "tensor.concat"
        assert kwargs == dict(storage_movements=(data.owner,))
        assert root.inputs[0] is data
        assert binding == dict(op_ordinal=list(module.walk()).index(data.owner), result=0)
        captured.append(module)
        return dict()

    monkeypatch.setattr(native_byte_activation, "compile_activation", compile)
    plan = prepare(tmp_path, {}, {}, None, None, source(width, scalar=scalar))
    assert plan["native_semantic_proof"]["input_domain"] == "all_storage_bit_patterns"
    roots = [op for op in captured[0].walk() if op.name == "linalg.generic"]
    x = z3.BitVec("source_word", width)
    bytes_ = [evaluate(op.body.block, [x, z3.BitVecVal(0, 8)]) for op in roots[:-2]]
    activated = [z3.If(b < 0, z3.BitVecVal(0, 8), b) for b in bytes_]
    result = evaluate(roots[-1].body.block, [*activated, z3.BitVecVal(0, width)])
    solver = z3.Solver()
    solver.add(result != z3.If(x < 0, z3.BitVecVal(0, width), x))
    assert solver.check() == z3.unsat
    # Without the original sign replicated into every digit, negative inputs
    # survive byte activation. The proof must detect that plausible mistake.
    wrong = [b & 127 for b in bytes_]
    wrong_result = evaluate(roots[-1].body.block, [*wrong, z3.BitVecVal(0, width)])
    solver = z3.Solver()
    solver.add(wrong_result != z3.If(x < 0, z3.BitVecVal(0, width), x))
    assert solver.check() == z3.sat


@pytest.mark.parametrize("options", [dict(threshold=1), dict(effect=True)])
def test_unproved_threshold_or_additional_effect_refuses(tmp_path, monkeypatch, options):
    monkeypatch.setattr(native_byte_activation, "compile_activation", lambda *a: pytest.fail("unproved candidate"))
    assert prepare(tmp_path, {}, {}, None, None, source(64, **options)) is None
