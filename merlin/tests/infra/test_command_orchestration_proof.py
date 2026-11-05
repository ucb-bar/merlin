"""The command-only exemption: which kernels are only target-command orchestration, and why not.

Each kernel below spends far more integer instructions per output element than the element budget
allows, so its verdict rests on the exemption alone. The ISA facts are fixture data, as a target's
RTL facts would supply them.
"""

from __future__ import annotations

import pytest
from xdsl.context import Context
from xdsl.dialects import builtin, llvm
from xdsl.parser import Parser

from merlin.targetgen import offload_census as OC
from merlin.targetgen.rocc import decode
from merlin.verify import host_compute_audit as HA

_ISA = {
    "CUSTOM_OPCODE": 123,
    "FUNCT_CLASS": {0: "CONFIG", 2: "MVIN", 7: "FLUSH", 14: "MVIN3"},
    "CONFIG_CLASS": "CONFIG",
}
_COMMAND = (
    '"llvm.inline_asm"(OPS) <{asm_string = "TEMPLATE", constraints = "CONSTRAINTS", has_side_effects}>'
    " : (TYPES) -> ()"
)


def _asm(template: str, *operands: str, constraints: str | None = None) -> str:
    spelled = constraints if constraints is not None else ",".join(["r"] * len(operands) + ["~{memory}"])
    return (
        _COMMAND.replace("OPS", ", ".join(operands))
        .replace("TEMPLATE", template)
        .replace("CONSTRAINTS", spelled)
        .replace("TYPES", ", ".join("i64" for _ in operands))
    )


_CONFIG = ".insn r 0x7b, 0x3, 0x0, x0, $0, $1"
_MVIN = ".insn r 0x7b, 0x3, 0x2, x0, $0, $1"


def _kernel(prologue: str = "", body: str = "", epilogue: str = "", *, tiles: int = 16) -> str:
    """Address math and one transfer per tile; ``body`` runs per tile and may use %t and %c1."""
    return f"""module {{
      llvm.func @entry(%scale: !llvm.ptr, %input: !llvm.ptr, %output: !llvm.ptr) {{
        %c0 = llvm.mlir.constant(0 : i64) : i64
        %c1 = llvm.mlir.constant(1 : i64) : i64
        %width = llvm.mlir.constant(56 : i64) : i64
        %tiles = llvm.mlir.constant({tiles} : i64) : i64
        {prologue}
        llvm.br ^loop(%c0 : i64)
      ^loop(%t: i64):
        %more = llvm.icmp "ult" %t, %tiles : i64
        llvm.cond_br %more, ^tile, ^done
      ^tile:
        %x = llvm.urem %t, %width : i64
        %y = llvm.udiv %t, %width : i64
        %row = llvm.mul %y, %width : i64
        %offset = llvm.add %row, %x : i64
        %src = llvm.getelementptr %input[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %addr = llvm.ptrtoint %src : !llvm.ptr to i64
        {_asm(_MVIN, "%addr", "%t")}
        {body}
        %next = llvm.add %t, %c1 : i64
        llvm.br ^loop(%next : i64)
      ^done:
        {epilogue}
        llvm.return
      }}
    }}"""


@pytest.fixture
def judge(monkeypatch):
    monkeypatch.setattr(decode, "isa_constants", lambda target: dict(_ISA))

    def run(source: str, elements: int = 4) -> dict:
        context = Context(allow_unregistered=True)
        for dialect in (builtin.Builtin, llvm.LLVM):
            context.load_dialect(dialect)
        module = Parser(context, source).parse_module()
        module.verify()
        site = HA.GroupSite(group=0, placement="unit0", symbol="entry", elements=elements, element_bytes=1)
        return HA.audit(module, [site], target="fixture")["groups"][0]

    return run


def test_plain_tile_walk_is_orchestration(judge) -> None:
    row = judge(_kernel())
    assert row["verdict"] == HA.CLEAN, row
    assert row["arithmetic_basis"] == "pointer_only_target_commands_v1"
    assert row["host_command_orchestration_arithmetic"] > 4 and "command_only_refusal" not in row


def test_a_flush_with_literal_x0_sources_takes_no_operands_and_is_clean(judge) -> None:
    flush = _asm(".insn r 0x7b, 0x3, 0x7, x0, x0, x0", constraints="~{memory}")
    assert judge(_kernel(epilogue=flush))["verdict"] == HA.CLEAN
    # One x0 source beside an operand reference is the same form.
    half = _asm(".insn r 0x7b, 0x3, 0x7, x0, $0, x0", "%c1")
    assert judge(_kernel(epilogue=half))["verdict"] == HA.CLEAN


@pytest.mark.parametrize(
    "command,refusal",
    [
        (_asm(".insn r 0x7b, 0x3, 0x7, x0, x5, x0", constraints="~{memory}"), "literal register 'x5'"),
        (_asm(".insn r 0x7b, 0x3, 0x7, x0, a0, $0", "%c1"), "literal register 'a0'"),
        (_asm(".insn r 0x7b, 0x3, 0x2, $0, $1, x0", "%c1", "%c1"), "writes register '$0'"),
        (_asm(".insn r 0x7b, 0x3, 0x2, x0, $0, x0", "%c1", "%c1"), "unconsumed"),
        (_asm(".insn r 0x7b, 0x3, 0x2, x0, $0, $2", "%c1", "%c1"), "does not have"),
        (_asm(".insn r 0x7b, 0x3, 0x9, x0, x0, x0", constraints="~{memory}"), "funct 9"),
        (_asm(".insn r 0x0b, 0x3, 0x2, x0, x0, x0", constraints="~{memory}"), "custom opcode"),
        (_asm(_MVIN, "%c1", "%c1", constraints="r,m,~{memory}"), "plain register inputs"),
    ],
)
def test_unparsed_or_smuggling_command_forms_keep_the_veto(judge, command, refusal) -> None:
    row = judge(_kernel(epilogue=command))
    assert row["verdict"] == HA.HOST_COMPUTE, row
    assert refusal in row["command_only_refusal"]
    # No tensor value was computed or moved: the finding says what defeated the exemption.
    assert "command-only exemption fails on" in row["why"] and refusal in row["why"]
    assert "element scale" not in row["why"]


@pytest.mark.parametrize("fence,clean", [("fence", True), ("fence iorw, iorw", True), ("fence rw,w", True),
                                         ("fence iorw", False), ("fence x5, rw", False), ("fence.i", False)])
def test_host_fences_in_any_defined_spelling_are_orchestration(judge, fence, clean) -> None:
    row = judge(_kernel(epilogue=_asm(fence, constraints="~{memory}")))
    assert row["verdict"] == (HA.CLEAN if clean else HA.HOST_COMPUTE), row


def test_target_named_third_load_state_and_unused_stack_scratch_are_clean(judge) -> None:
    mvin3 = _asm(".insn r 0x7b, 0x3, 14, x0, $0, $1", "%c1", "%c1")
    assert judge(_kernel(epilogue=mvin3))["verdict"] == HA.CLEAN
    slot = "%n = llvm.mlir.constant(4 : i32) : i32\n %slot = llvm.alloca %n x i32 : (i32) -> !llvm.ptr"
    assert judge(_kernel(prologue=slot))["verdict"] == HA.CLEAN
    reread = slot + "\n %v = llvm.load %slot : !llvm.ptr -> i64\n" + _asm(_CONFIG, "%v", "%c1")
    row = judge(_kernel(prologue=reread))
    assert row["verdict"] == HA.HOST_COMPUTE and "kernel argument" in row["command_only_refusal"]


# Exo's config_st: a scalar fp32 scale read once from memory, reinterpreted as bits and packed
# into the configuration word beside a constant stride and subtype.
_SCALE = """%s = llvm.load %scale : !llvm.ptr -> f32
        %bits = llvm.bitcast %s : f32 to i32
        %wide = llvm.zext %bits : i32 to i64
        %c32 = llvm.mlir.constant(32 : i64) : i64
        %hi = llvm.shl %wide, %c32 : i64
        %stride = llvm.mlir.constant(131074 : i64) : i64
        %word = llvm.or %hi, %stride : i64
        """


def test_a_scale_loaded_only_to_configure_the_unit_is_orchestration(judge) -> None:
    row = judge(_kernel(prologue=_SCALE + _asm(_CONFIG, "%word", "%c1")))
    assert row["verdict"] == HA.CLEAN, row
    assert (row["host_command_parameter_loads"], row["host_command_parameter_payload_bytes"]) == (1, 4)
    assert row["host_payload_bytes"] == 0 and row["host_raw_payload_bytes"] == 4
    # A float-to-integer conversion of the loaded value is a parameter cast as well.
    converted = "%s = llvm.load %scale : !llvm.ptr -> f32\n %q = \"llvm.fptosi\"(%s) : (f32) -> i64\n"
    assert judge(_kernel(prologue=converted + _asm(_CONFIG, "%q", "%c1")))["verdict"] == HA.CLEAN


@pytest.mark.parametrize(
    "use,refusal",
    [
        # The loaded value scales a stored product: host computation, and a store.
        (
            "%f = llvm.fmul %s, %s : f32\n %q = \"llvm.fptosi\"(%f) : (f32) -> i8\n"
            " llvm.store %q, %output : i8, !llvm.ptr\n" + _asm(_CONFIG, "%word", "%c1"),
            "host memory store",
        ),
        # Arithmetic on the loaded value is not packing, even if only a command consumes it.
        (
            "%m = llvm.mul %wide, %c32 : i64\n" + _asm(_CONFIG, "%m", "%c1"),
            "unproven memory load feeding llvm.mul",
        ),
        # A transfer address read from memory is not a configuration parameter.
        (_asm(_MVIN, "%word", "%c1"), "only configuration commands"),
        # A control decision on a loaded value is host work on that value.
        (
            "%big = llvm.icmp \"ugt\" %wide, %c32 : i64\n" + _asm(_CONFIG, "%word", "%c1"),
            "unproven memory load feeding llvm.icmp",
        ),
    ],
)
def test_a_loaded_value_with_any_other_use_keeps_the_veto(judge, use, refusal) -> None:
    row = judge(_kernel(prologue=_SCALE + use), elements=8)
    assert row["verdict"] == HA.HOST_COMPUTE, row
    assert refusal in row["command_only_refusal"]
    # Float work names itself; without it the finding names the construct that refused.
    assert (refusal in row["why"]) == (row["value_arithmetic_per_element"] <= 0.25)


def test_a_parameter_load_inside_an_element_loop_is_payload(judge) -> None:
    per_element = (
        "%p = llvm.getelementptr %scale[%t] : (!llvm.ptr, i64) -> !llvm.ptr, f32\n"
        " %e = llvm.load %p : !llvm.ptr -> f32\n %eb = llvm.bitcast %e : f32 to i32\n"
        " %ew = llvm.zext %eb : i32 to i64\n" + _asm(_CONFIG, "%ew", "%c1")
    )
    row = judge(_kernel(body=per_element, tiles=1024), elements=1024)
    assert row["verdict"] == HA.HOST_COMPUTE, row
    assert "1024 parameter loads per call exceed" in row["command_only_refusal"]
    # The same load per tile pass, a handful of times, stays configuration.
    assert judge(_kernel(body=per_element, tiles=8), elements=1)["verdict"] == HA.CLEAN


def test_a_host_zero_fill_beside_commands_is_vetoed(judge) -> None:
    # Three accumulators cleared on the host with stores of constant 0, one element per trip.
    fill = "\n".join(
        f"%z{i} = llvm.mlir.constant(0 : i32) : i32\n"
        f" %d{i} = llvm.getelementptr %output[%t] : (!llvm.ptr, i64) -> !llvm.ptr, i32\n"
        f" llvm.store %z{i}, %d{i} : i32, !llvm.ptr"
        for i in range(3)
    )
    row = judge(_kernel(body=fill, tiles=1024), elements=1024)
    assert row["verdict"] == HA.HOST_COMPUTE, row
    assert "host memory store" in row["command_only_refusal"]
    assert "element scale" in row["why"] and row["payload_ratio"] > 1


def test_the_grade_names_unproven_orchestration_without_claiming_host_work(tmp_path, judge, monkeypatch) -> None:
    del judge  # installs the fixture ISA facts, which only this process sees
    import json

    from merlin.targetgen import host_compute_veto_process

    monkeypatch.setenv(host_compute_veto_process.ENV, "0")

    stray = _kernel(epilogue=_asm(".insn r 0x7b, 0x3, 0x7, x0, x5, x0", constraints="~{memory}"))
    generated = tmp_path / "stray" / "generated"
    generated.mkdir(parents=True)
    buffer = {
        "tensors": {"y": {"shape": [4], "dtype": "i8"}},
        "kernel_abi": {"kind": "whole_program", "args": [], "outputs": ["y"]},
        "commands": [{"opcode": "COMMIT", "operands": {"dst": "y"}}],
    }
    (generated / "command_buffer.json").write_text(json.dumps(buffer), encoding="utf-8")
    (generated / "lowered.llvm.mlir").write_text(stray, encoding="utf-8")
    (tmp_path / "stray" / "capsule_result.json").write_text(json.dumps({"capsule": "stray", "status": "pass"}))
    results = [{"capsule": "stray", "kind": "layer", "status": "pass"}]
    assert OC.apply_host_compute_veto(results, tmp_path, target="fixture")
    detail = results[0]["failure"]["detail"]
    assert detail.startswith("the program passes numerics; its host integer work is not proven")
    assert "literal register 'x5'" in detail and "spends its work on the host" not in detail
