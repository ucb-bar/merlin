"""Counted host loops must issue the exact original primitive stream."""

from dataclasses import replace

import pytest
from test_llvm_index_program import ExecutingBuilder

from merlin.compile.scheduling.command_compaction import compact_command_stream, verify_command_compaction
from merlin.compile.scheduling.llvm_program import emit_index_program
from merlin.sched.ir import Call, Const, Kernel, Loop, Var


def stream(rows, columns, stride):
    body = []
    for i in range(rows):
        body.append(Call("load", (("source", Const(i * stride)), ("row", Const(i * columns)))))
        body.append(Call("compute", (("input", Const(i * columns)), ("output", Const(i * columns)))))
        body.append(Call("store", (("output", Const(i * columns)), ("count", Const(columns)))))
    # A partial tail must survive with its original extent and coordinates.
    body.append(Call("store", (("output", Const(rows * columns)), ("count", Const(1)))))
    return Kernel("stream", (), tuple(body))


@pytest.mark.parametrize("rows,columns,stride", [(3, 2, 5), (17, 16, 65), (33, 32, 97)])
def test_compacted_loop_reaches_actual_builder_in_original_order(rows, columns, stride):
    original = stream(rows, columns, stride)
    result = compact_command_stream(original, index_bits=64)
    observed = []
    emit_index_program(result.kernel, ExecutingBuilder(), lambda name, args: observed.append((name, args)))
    assert observed == [(call.instr, {key: value.value for key, value in call.args}) for call in original.body]
    assert result.static_calls == 4 and result.original_calls == rows * 3 + 1
    assert result.loops == 1


@pytest.mark.parametrize("mutation", ["trip", "operand", "order", "drop", "extra"])
def test_independent_stream_verifier_rejects_changed_emission(mutation):
    original = stream(7, 4, 17)
    good = compact_command_stream(original, index_bits=32).kernel
    loop = good.body[0]
    if mutation == "trip":
        bad = replace(good, body=(replace(loop, extent=6), good.body[-1]))
    elif mutation == "operand":
        call = replace(loop.body[0], args=(("source", Var(loop.var) * 18), ("row", Var(loop.var) * 4)))
        bad = replace(good, body=(replace(loop, body=(call, *loop.body[1:])), good.body[-1]))
    elif mutation == "order":
        bad = replace(good, body=(replace(loop, body=tuple(reversed(loop.body))), good.body[-1]))
    elif mutation == "drop":
        bad = replace(good, body=(loop,))
    else:
        bad = replace(good, body=(*good.body, good.body[-1]))
    with pytest.raises(ValueError):
        verify_command_compaction(original, bad, index_bits=32)


def test_execution_tokens_are_preserved_as_literal_calls():
    source = Kernel(
        "effects", (), tuple(replace(call, produces=f"ready{i}") for i, call in enumerate(stream(3, 4, 7).body))
    )
    result = compact_command_stream(source, index_bits=64)
    assert result.kernel == source and result.loops == 0


def test_signed_intermediate_overflow_declines_unsafe_affine_loop():
    source = Kernel("overflow", (), tuple(Call("value", (("x", Const(x)),)) for x in (2**31 - 1, -1, -(2**31) - 1)))
    with pytest.raises(ValueError, match="overflow"):
        compact_command_stream(source, index_bits=32)
    source = Kernel("intermediate", (), tuple(Call("value", (("x", Const(x)),)) for x in (2**31 - 1, 0, -(2**31) + 1)))
    result = compact_command_stream(source, index_bits=32)
    assert result.loops == 0, "all final constants fit, but a repeated multiplication would overflow"


def test_huge_or_empty_actual_loops_cannot_bypass_finite_verification_budget():
    source = Kernel("tiny", (), (Call("value", (("x", Const(0)),)),))
    bad = replace(source, body=(Loop("i", 2**31 - 1, source.body),))
    with pytest.raises(ValueError, match="count"):
        verify_command_compaction(source, bad, index_bits=64)
    empty = Kernel("empty", (), ())
    verify_command_compaction(empty, replace(empty, body=(Loop("i", 2**31 - 1, ()),)), index_bits=64)
    with pytest.raises(ValueError, match="budget"):
        compact_command_stream(stream(4, 4, 9), index_bits=64, max_comparisons=1)


def test_generated_native_host_loop_preserves_effect_order_and_output_guards(tmp_path):
    import ctypes
    import shutil
    import subprocess

    import numpy as np

    from merlin.sched.codegen import emit_c_function
    from merlin.sched.ir import TensorArg
    from merlin.sched.isa import InstrDef, InstructionSet, Operand

    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    original = Kernel(
        "native_commands",
        (TensorArg("output", (97,), "i32", "write"),),
        tuple(Call("put", (("position", Const(i)), ("value", Const(3 * i + 7)))) for i in range(97)),
    )
    result = compact_command_stream(original, index_bits=64)
    isa = InstructionSet(
        "neutral-test-store",
        {
            "put": InstrDef(
                "put",
                (Operand("position", "int"), Operand("value", "int")),
                render_c=lambda args: f"output[{args[0]}] = (state = state * 33u + (uint32_t)({args[1]}));",
            )
        },
        {"i32": "int32_t"},
        "",
    )
    baseline = emit_c_function(original, isa, symbol="baseline", index_bits=64)
    compacted = emit_c_function(result.kernel, isa, symbol="compacted", index_bits=64)
    assert len(compacted) < len(baseline)
    source, library = tmp_path / "commands.c", tmp_path / "commands.so"
    source.write_text(
        "#include <stdint.h>\nstatic uint32_t state;\nvoid reset(void){state=0;}\n"
        + baseline
        + compacted
        + "\nvoid run_baseline(int32_t *p){baseline(p);}\nvoid run_compacted(int32_t *p){compacted(p);}"
    )
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    native = ctypes.CDLL(str(library))
    expected, state = [], 0
    for i in range(97):
        state = (state * 33 + 3 * i + 7) % 2**32
        expected.append(state)
    for name in ("baseline", "compacted"):
        destination = np.full(97 + 16, 0xABCD1234, np.uint32)
        output = destination[8:-8]
        native.reset()
        function = getattr(native, "run_" + name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = None
        function(output.ctypes.data)
        np.testing.assert_array_equal(output, np.asarray(expected, np.uint32))
        assert np.all(destination[:8] == 0xABCD1234) and np.all(destination[-8:] == 0xABCD1234)


@pytest.mark.parametrize("value", [-0.0, float("inf"), float("nan")])
def test_floating_operands_preserve_exact_representation(value):
    import struct

    original = Kernel("floating", (), (Call("value", (("x", value),)),))
    result = compact_command_stream(original, index_bits=64)
    assert result.loops == 0
    actual = result.kernel.body[0].args[0][1]
    assert struct.pack(">d", actual) == struct.pack(">d", value)
    if value == 0.0:
        changed = replace(original, body=(Call("value", (("x", 0.0),)),))
    else:
        changed = replace(original, body=(Call("value", (("x", -value),)),))
    with pytest.raises(ValueError, match="operands"):
        verify_command_compaction(original, changed, index_bits=64)


def test_nonliteral_constant_and_pointer_expressions_pass_through():
    from merlin.sched.ir import Ptr, TensorArg
    from merlin.sched.ir.expr import Sum

    original = Kernel(
        "symbolic",
        (TensorArg("input", (32,), "i8", "read"),),
        (
            Call("value", (("x", Sum((Const(3), Const(4)))),)),
            Call("load", (("p", Ptr("input", Sum((Const(4), Const(5))))),)),
        ),
    )
    result = compact_command_stream(original, index_bits=64)
    assert result.kernel == original and result.loops == 0


@pytest.mark.parametrize("value", [True, 1.5, 2**80])
def test_bad_integer_literals_refuse_before_emission(value):
    original = Kernel("bad", (), (Call("value", (("x", Const(value)),)),))
    with pytest.raises(ValueError, match="literal"):
        compact_command_stream(original, index_bits=64)


@pytest.mark.parametrize("form", ["scale", "unreachable", "selector"])
def test_unreachable_or_zero_expression_does_not_hide_unrepresentable_literals(form):
    from merlin.sched.ir.expr import Scaled, Select

    original = Kernel("literal", (), (Call("value", (("x", Const(0)),)),))
    expression = {
        "scale": Scaled(Var("i"), 2**80),
        "unreachable": Select("i", 1, Const(2**80), Const(0)),
        "selector": Select("i", 2**80, Const(0), Const(0)),
    }[form]
    actual = replace(original, body=(Loop("i", 1, (Call("value", (("x", expression),)),)),))
    with pytest.raises(ValueError, match="literal"):
        verify_command_compaction(original, actual, index_bits=64)


def test_deep_actual_nesting_refuses_before_recursive_ir_checks():
    original = Kernel("deep", (), (Call("value", (("x", Const(0)),)),))
    body = original.body
    for i in range(1200):
        body = (Loop(f"i{i}", 1, body),)
    with pytest.raises(ValueError, match="budget"):
        verify_command_compaction(original, replace(original, body=body), index_bits=64)


def test_loop_variable_avoids_tensor_argument_name():
    from merlin.sched.ir import TensorArg

    original = replace(stream(7, 4, 17), args=(TensorArg("compact_0", (4,), "i32", "read"),))
    result = compact_command_stream(original, index_bits=64)
    assert result.loops == 1
    assert result.kernel.body[0].var != "compact_0"
    verify_command_compaction(original, result.kernel, index_bits=64)


def test_oversized_expression_and_operand_budgets_refuse():
    from merlin.sched.ir.expr import Scaled, Sum

    expressions = [Sum((Const(0),) * 65)]
    deep = Const(0)
    for _ in range(66):
        deep = Scaled(deep, 1)
    expressions.append(deep)
    for expression in expressions:
        original = Kernel("expression", (), (Call("value", (("x", expression),)),))
        with pytest.raises(ValueError, match="budget"):
            compact_command_stream(original, index_bits=64, max_calls=1)
    original = Kernel("operands", (), (Call("value", tuple((f"x{i}", Const(0)) for i in range(65))),))
    with pytest.raises(ValueError, match="operands"):
        compact_command_stream(original, index_bits=64)


@pytest.mark.parametrize("annotation", ["sets", "assumes"])
def test_configuration_annotations_are_bit_exact_and_bounded(annotation):
    from merlin.sched.ir.expr import Scaled

    original = Kernel("annotation", (), (Call("cfg", (), **{annotation: (("scale", -0.0),)}),))
    assert compact_command_stream(original, index_bits=64).kernel == original
    changed = replace(original.body[0], **{annotation: (("scale", 0.0),)})
    with pytest.raises(ValueError, match="operands"):
        verify_command_compaction(original, replace(original, body=(changed,)), index_bits=64)
    deep = Const(0)
    for _ in range(1200):
        deep = Scaled(deep, 1)
    changed = replace(original.body[0], **{annotation: (("scale", deep),)})
    with pytest.raises(ValueError, match="budget"):
        verify_command_compaction(original, replace(original, body=(changed,)), index_bits=64)


def test_nested_annotation_values_cannot_bypass_comparison_budget():
    value = -0.0
    for _ in range(1200):
        value = (value,)
    original = Kernel("annotation", (), (Call("cfg", (), sets=(("scale", value),)),))
    with pytest.raises(ValueError, match="budget"):
        compact_command_stream(original, index_bits=64)
