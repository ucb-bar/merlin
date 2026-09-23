"""The target-neutral kernel IR, its static checker and its C emitter, on a toy instruction set."""

import pytest

from merlin.sched.check.static import check_kernel
from merlin.sched.codegen import emit_c_function
from merlin.sched.ir import (
    NULL,
    ExprError,
    Kernel,
    Ptr,
    TensorArg,
    Var,
    add,
    call,
    check_structure,
    evaluate,
    instances,
    loop,
    mul,
    render,
    select,
)
from merlin.sched.isa import InstrDef, InstructionSet, Operand


def test_expressions_simplify_evaluate_and_render():
    i = Var("i")
    e = add(mul(i, 4), 3, 5)
    assert render(e) == "((i * 4) + 8)" and evaluate(e, {"i": 2}) == 16
    assert mul(mul(i, 2), 3) == mul(i, 6) and mul(i, 0) == add(0) and add(i, 0) == i
    s = select(i, 3, 7, i * 2)
    assert evaluate(s, {"i": 3}) == 7 and evaluate(s, {"i": 1}) == 2
    assert render(s, c=True) == "((i == 3) ? 7 : (i * 2))"
    assert select(i, 0, 5, 5) == add(5)  # equal branches collapse
    with pytest.raises(ExprError):
        mul(i, i)
    with pytest.raises(ExprError):
        evaluate(i, {})


def _copy_def():
    # copy n bytes from src to dst (a toy instruction; nothing target-specific)
    def footprint(v):
        return [("src", v["n"], "read"), ("dst", v["n"], "write")]

    def check(v, state):
        state["copies"] = state.get("copies", 0) + 1
        return [] if v["n"] > 0 else ["empty copy"]

    return InstrDef(
        "copy",
        (Operand("src", "ptr"), Operand("dst", "ptr"), Operand("n", "int"), Operand("scale", "float")),
        render_c=lambda a: f"toy_copy({', '.join(a)});",
        check=check,
        footprint=footprint,
    )


ISET = InstructionSet(
    target="toy",
    instrs={"copy": _copy_def()},
    c_types={"i8": "int8_t", "i32": "int32_t"},
    drain_c="toy_drain();",
    finish=lambda s: [] if s.get("copies") else ["nothing copied"],
)


def _kernel(extent=4, chunk=16, src_access="read"):
    i = Var("i")
    return Kernel(
        "k",
        (TensorArg("x", (64,), "i8", src_access), TensorArg("y", (64,), "i8", "write")),
        (loop("i", extent, call("copy", src=Ptr("x", i * chunk), dst=Ptr("y", i * chunk), n=chunk, scale=0.1)),),
    )


def test_text_is_the_identity():
    a, b = _kernel(), _kernel()
    assert a.text() == b.text() and a.digest() == b.digest()
    assert _kernel(extent=3).digest() != a.digest()
    assert "copy(src=&x[(i * 16)], dst=&y[(i * 16)], n=16, scale=" + (0.1).hex() + ")" in a.text()


def test_instances_enumerate_in_program_order():
    envs = [env for _, env in instances(_kernel())]
    assert envs == [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}]


def test_structure_errors():
    bad = Kernel(
        "k",
        (TensorArg("x", (8,), "i8", "read"),),
        (loop("i", 2, loop("i", 2, call("copy", src=Ptr("x", Var("j")), dst=Ptr("z", 0), n=1, scale=1.0))),),
    )
    msgs = " | ".join(check_structure(bad))
    assert "shadows" in msgs and "unbound" in msgs and "unknown tensor 'z'" in msgs


def test_static_check_accepts_a_legal_kernel_and_runs_finish():
    assert check_kernel(_kernel(), ISET) == []
    empty = Kernel("k", (TensorArg("x", (8,), "i8", "read"),), ())
    assert check_kernel(empty, ISET) == ["end of kernel: nothing copied"]


def test_static_check_refuses_out_of_bounds_and_readonly_writes():
    oob = check_kernel(_kernel(chunk=20), ISET)  # the 4th instance reads past 64 bytes
    assert any("touches bytes [60, 80)" in m for m in oob)
    ro = Kernel(
        "k", (TensorArg("x", (64,), "i8", "read"),), (call("copy", src=Ptr("x", 0), dst=Ptr("x", 0), n=4, scale=1.0),)
    )
    assert any("writes read-only x" in m for m in check_kernel(ro, ISET))


def test_static_check_refuses_unknown_instruction_and_operand_order():
    k = Kernel(
        "k",
        (TensorArg("x", (8,), "i8", "readwrite"),),
        (call("copy", dst=Ptr("x", 0), src=Ptr("x", 0), n=1, scale=1.0), call("nope")),
    )
    msgs = " | ".join(check_kernel(k, ISET))
    assert "unknown instruction 'nope'" in msgs and "operands ('dst', 'src'" in msgs
    kinds = Kernel(
        "k",
        (TensorArg("x", (8,), "i8", "readwrite"),),
        (call("copy", src=NULL, dst=Ptr("x", 0), n=Ptr("x", 0), scale=1.0),),
    )
    assert any("expected a int operand" in m for m in check_kernel(kinds, ISET))


def test_emitter_renders_loops_pointers_and_exact_floats():
    src = emit_c_function(_kernel(), ISET, symbol="mk_k")
    assert src.startswith("static void mk_k(int8_t *x, int8_t *y) {")
    assert "for (int i = 0; i < 4; i++) {" in src
    assert "toy_copy(((uint8_t *)x + (i * 16)), ((uint8_t *)y + (i * 16)), 16, 0x1.99999a0000000p-4f);" in src
    assert src.rstrip().endswith("toy_drain();\n}")
