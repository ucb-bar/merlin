"""Actual declared instruction IR cleanup, independent output decoding and mutations.

Synthetic contracts qualify this mechanism only; no hardware target support or
instruction implementation qualification follows from these tests.
"""

import json
from dataclasses import replace

import pytest
from test_selected_schedule_contracts import inputs
from xdsl.dialects import arith, builtin, func
from xdsl.ir import Block, Region

from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection, verify_configuration_ir
from merlin.targetgen.isa_dialect_gen import decode_instruction, encode_instruction
from merlin.transforms.certificates import ir_text

QUALIFICATION = "independent-neutral-persistent-state-premise-v1"


def configured(tmp_path, *, width=4, ordered=True, capacity=2, **controls):
    selected, *_ = inputs(tmp_path, width=width, ordered=ordered, capacity=capacity)
    return ConfigurationIRSelection(selected, QUALIFICATION, **controls)


def instructions(width=4, *, name="run", body=None):
    bits = width * 8
    body = (
        body
        or """
      "neutral_isa.set"(%s0,%a3,%zero) : (i8,IT,IT) -> ()
      "neutral_isa.set"(%s1,%a2,%zero) : (i8,IT,IT) -> ()
      "neutral_isa.work"(%s0,%zero,%zero) : (i8,IT,IT) -> ()
      "neutral_isa.set"(%s0,%a3,%zero) : (i8,IT,IT) -> ()
      "neutral_isa.set"(%s1,%a2,%zero) : (i8,IT,IT) -> ()
      "neutral_isa.work"(%s0,%zero,%zero) : (i8,IT,IT) -> ()
    """
    )
    return f"""func.func @{name}() {{
      %s0 = arith.constant 0 : i8
      %s1 = arith.constant 1 : i8
      %a3 = arith.constant 3 : i{bits}
      %a2 = arith.constant 2 : i{bits}
      %zero = arith.constant 0 : i{bits}
      {body.replace("IT", f"i{bits}")}
      func.return
    }}"""


def source(selection, *, width=4, body=None):
    return selection.parse("module {" + instructions(width, body=body) + "}")


def encoded(module, selection):
    """Independently read actual instruction operands, then decode actual encoded words."""
    words, observations, state, outputs = [], [], {}, []
    isa = selection.selected_contracts.isa
    for op in module.walk():
        if not op.name.startswith(isa.namespace + "."):
            continue
        instruction = isa.instruction(op.name.split(".")[1])
        bindings = {
            argument.name: operand.owner.value.value.data & ((1 << argument.type.bits) - 1)
            for argument, operand in zip(instruction.operands, op.operands, strict=True)
        }
        for argument in instruction.immediates:
            bindings[argument.name] = op.attributes[argument.name].value.data
        word = encode_instruction(isa, instruction.name, bindings)
        words.append(word.to_bytes(isa.word_bits // 8, selection.selected_contracts.instruction_byteorder))
        name, actual = decode_instruction(isa, word)
        observations.append((name, actual))
        if name == "set":
            state[actual["slot"]] = actual["a"]
        elif name == "reset":
            state.pop(actual["slot"], None)
        elif name == "work":
            outputs.append((state.get(0), state.get(1)))
    return b"".join(words), observations, outputs, state


@pytest.mark.parametrize("width,ordered", [(4, True), (8, True), (8, False)])
def test_actual_target_ops_and_decoded_words_preserve_outputs_and_state(tmp_path, width, ordered):
    selection = configured(tmp_path, width=width, ordered=ordered)
    module = source(selection, width=width)
    before = ir_text(module)
    result = selection.apply(module)
    result.revalidate()
    assert ir_text(module) == before
    original, original_ops, expected, expected_state = encoded(module, selection)
    realized, actual_ops, actual, state = encoded(result.module, selection)
    assert actual == expected == [(3, 2), (3, 2)] and state == expected_state == {0: 3, 1: 2}
    assert len(actual_ops) == (4 if ordered else 6)
    assert len(realized) == len(original) - (2 if ordered else 0) * selection.selected_contracts.isa.word_bits // 8
    assert len(original_ops) == 6
    proof = result.report["verification"]
    assert proof["removed_operations"] == (2 if ordered else 0)
    assert proof["execution_qualification"] == "not_established" and proof["timing"] == "UNKNOWN"
    stages = result.report["passes"]
    assert [stage["name"] for stage in stages] == [
        "configuration-ir-source",
        "configuration-ir-cleanup",
        "configuration-ir-conformance",
    ]
    assert tuple(stages[1]["invalidated"]) == ("configuration",)


@pytest.mark.parametrize("scope", ["function", "loop", "conditional", "opaque"])
def test_scopes_start_unknown_and_opaque_regions_are_never_optimized(tmp_path, scope):
    selection = configured(tmp_path)
    pair = '"neutral_isa.set"(%s0,%a3,%zero) : (i8,i32,i32) -> ()\n' * 2
    if scope == "function":
        text = "module {" + instructions(name="first", body=pair) + instructions(name="second", body=pair) + "}"
    elif scope == "loop":
        body = pair + "%lb=arith.constant 0:index\n%ub=arith.constant 3:index\n%step=arith.constant 1:index\n"
        body += "scf.for %i=%lb to %ub step %step {" + pair + "}\n" + pair
        text = "module {" + instructions(body=body) + "}"
    elif scope == "conditional":
        text = (
            "module {"
            + instructions(body=pair + "%true=arith.constant true\nscf.if %true {" + pair + "}\n" + pair)
            + "}"
        )
    else:
        text = "module {" + instructions(body=pair + '"closed.unknown"() ({' + pair + "}) : () -> ()\n" + pair) + "}"
    module = selection.parse(text)
    result = selection.apply(module)
    result.revalidate()
    expected = {"function": 2, "loop": 3, "conditional": 3, "opaque": 2}[scope]
    assert result.report["verification"]["removed_operations"] == expected
    if scope == "opaque":
        region = next(
            op for op in result.module.walk() if op.attributes.get("op_name__") == builtin.StringAttr("closed.unknown")
        )
        assert len(tuple(region.regions[0].block.ops)) == 2


@pytest.mark.parametrize("barrier", ["call", "unknown", "metadata", "dynamic", "pulse", "reset"])
def test_unknown_and_nonpure_effects_keep_required_setters(tmp_path, barrier):
    selection = configured(tmp_path)
    setter = '"neutral_isa.set"(%s0,%a3,%zero) : (i8,i32,i32) -> ()\n'
    middle = {
        "call": "func.call @external() : () -> ()",
        "unknown": '"opaque.effect"() : () -> ()',
        "metadata": '"neutral_isa.set"(%s0,%a3,%zero) {unknown} : (i8,i32,i32) -> ()',
        "dynamic": '"neutral_isa.set"(%s0,%dynamic,%zero) : (i8,i32,i32) -> ()',
        "pulse": '"neutral_isa.pulse"(%s0,%a3,%zero) : (i8,i32,i32) -> ()',
        "reset": '"neutral_isa.reset"(%s0,%zero,%zero) : (i8,i32,i32) -> ()',
    }[barrier]
    text = instructions(body=setter + middle + "\n" + setter)
    text = text.replace("@run()", "@run(%dynamic:i32)")
    module = selection.parse("module {func.func private @external()\n" + text + "}")
    result = selection.apply(module)
    result.revalidate()
    # A pulse is retained even though its persistent value remains unchanged.
    assert result.report["verification"]["removed_operations"] == (1 if barrier == "pulse" else 0)
    assert (
        any(op.name == "neutral_isa." + barrier for op in result.module.walk())
        if barrier in ("pulse", "reset")
        else True
    )


@pytest.mark.parametrize(
    "controls,reason",
    [
        (dict(enabled=False), "disabled"),
        (dict(qualification_identity=None), "qualification unavailable"),
        (dict(max_operations=1), "operation analysis budget"),
        (dict(max_fields=1), "field analysis budget"),
    ],
)
def test_unavailable_routes_preserve_exact_ir(tmp_path, controls, reason):
    selection = configured(tmp_path)
    module = source(selection)
    selection = replace(selection, **controls)
    result = selection.apply(module)
    assert ir_text(result.module) == ir_text(module)
    assert reason in result.report["verification"]["fallback"]
    result.revalidate()


@pytest.mark.parametrize("mutation", ["facts", "effects", "policy", "qualification", "budget"])
def test_selected_original_files_effects_and_controls_are_fresh(tmp_path, mutation):
    selection = configured(tmp_path)
    if mutation == "facts":
        path = selection.selected_contracts.facts_path
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation == "effects":
        effects = json.loads(selection.selected_contracts.effects_json)
        effects["set"]["pure_write"] = False
        object.__setattr__(selection.selected_contracts, "effects_json", json.dumps(effects))
    elif mutation == "policy":
        object.__setattr__(selection.selected_contracts, "numerical_policy", "different-policy")
    elif mutation == "qualification":
        object.__setattr__(selection, "qualification_identity", "new-self-declared-ID")
    else:
        object.__setattr__(selection, "max_operations", 8000)
    with pytest.raises(ValueError, match="changed|reselect|original declarations"):
        selection.revalidate()


@pytest.mark.parametrize("mutation", ["first", "consumer", "value", "metadata", "extra", "ssa", "opaque_type"])
def test_actual_ir_mutations_fail_independent_conformance(tmp_path, mutation):
    selection = configured(tmp_path)
    module = source(selection)
    result = selection.apply(module)
    actual = result.module.clone()
    function = next(op for op in actual.walk() if type(op) is func.FuncOp)
    device = [op for op in function.body.block.ops if op.name.startswith("neutral_isa.")]
    if mutation in ("first", "consumer"):
        function.body.block.erase_op(device[0 if mutation == "first" else 2])
    elif mutation == "value":
        device[0].operands = (device[0].operands[0], device[1].operands[1], device[0].operands[2])
    elif mutation == "metadata":
        device[0].attributes["unknown"] = builtin.UnitAttr()
    elif mutation == "extra":
        function.body.block.insert_op_before(device[0].clone(), function.body.block.last_op)
    elif mutation == "ssa":
        # Two same-typed values must not become interchangeable merely because they are constants.
        constants = [op for op in function.body.block.ops if type(op) is arith.ConstantOp]
        constants[2].properties["value"] = builtin.IntegerAttr(2, builtin.i32)
    else:
        opaque = builtin.UnregisteredOp.with_name("opaque.region").create(regions=[Region(Block())])
        nested = builtin.UnregisteredOp.with_name("opaque.result").create(result_types=[builtin.i32])
        opaque.regions[0].block.add_op(nested)
        function.body.block.insert_op_before(opaque, function.body.block.last_op)
    actual.verify()
    with pytest.raises(ValueError, match="configuration"):
        verify_configuration_ir(module, actual, selection=selection)


def test_result_revalidation_rejects_changed_accepted_ir_and_mutated_report(tmp_path):
    selection = configured(tmp_path)
    result = selection.apply(source(selection))
    result.report["verification"]["removed_operations"] = 100
    with pytest.raises(ValueError, match="changed"):
        result.revalidate()


def test_unknown_nested_result_types_and_captured_wiring_are_preserved(tmp_path):
    selection = configured(tmp_path)
    text = instructions(body='"opaque.region"(%a3) ({%v="opaque.result"(%a3):(i32)->i32}) : (i32)->()')
    module = selection.parse("module {" + text + "}")
    result = selection.apply(module)
    result.revalidate()
    actual = result.module.clone()
    nested = next(op for op in actual.walk() if op.attributes.get("op_name__") == builtin.StringAttr("opaque.result"))
    nested.results[0]._type = builtin.i16
    actual.verify()
    with pytest.raises(ValueError, match="opaque region"):
        verify_configuration_ir(module, actual, selection=selection)


def test_explicit_byte_budget_precedes_parsing_and_module_cloning(tmp_path):
    selection = configured(tmp_path)
    module = source(selection)
    bounded = replace(selection, max_ir_bytes=32)
    with pytest.raises(ValueError, match="input byte budget"):
        bounded.parse(ir_text(module))
    with pytest.raises(ValueError, match="input byte budget"):
        bounded.apply(module)
    with pytest.raises(ValueError, match="input byte budget"):
        bounded.parse("é" * 17)
