"""Declared pointer transport; these tests issue no implementation qualification."""

from dataclasses import replace

import pytest
from test_selected_dense_contracts import inputs

from merlin.compile.scheduling.block_schedule import Compute, Load, Store
from merlin.compile.scheduling.dense_contracts import DenseSchedulingUnavailable
from merlin.sched.ir import Const, Ptr


def pointer_inputs(tmp_path, *, mutate=None, **kwargs):
    def v2(facts, software, dense):
        dense["schema"] = "merlin.dense_schedule.v2"
        dense["logical_pointer_abi"] = dict(
            schema="merlin.logical_pointer_abi.v1",
            layout="dense_row_major",
            alignment_bytes=[1, 1, 1],
            access=["read_only", "read_only", "fully_written"],
            bounds="logical_elements",
            completion="synchronous",
        )
        dense["external_pointer_arguments"] = {
            kind: dict(argument="buffer", space=space)
            for kind, space in [("lhs_load", "input_lhs"), ("rhs_load", "input_rhs"), ("store", "output")]
        }
        facts["word_bits"] = 584
        facts["offsets"] = [8 + 64 * i for i in range(9)]
        facts["padding_offsets"] = [8 + 64 * i for i in range(10)]
        facts["padding_widths"] = [64 * (9 - i) for i in range(10)]

        def ref(*parts):
            return {"fact": ["facts", *parts]}

        for instruction, kind in zip(
            software["isa"]["instructions"], ("lhs_load", "rhs_load", "preload", "compute", "store"), strict=True
        ):
            if kind in dense["external_pointer_arguments"]:
                instruction["operands"].append(
                    dict(
                        name="buffer",
                        unit="bytes",
                        type=dict(
                            kind="address",
                            signed=False,
                            bits=ref("arg_bits"),
                            space=dense["external_pointer_arguments"][kind]["space"],
                        ),
                    )
                )
            args = instruction["operands"]
            instruction["encoding"] = [instruction["encoding"][0]] + [
                dict(offset=ref("offsets", j), width=ref("arg_bits"), argument=arg["name"])
                for j, arg in enumerate(args)
            ]
            if len(args) < 9:
                instruction["encoding"].append(
                    dict(
                        offset=ref("padding_offsets", len(args)),
                        width=ref("padding_widths", len(args)),
                        fixed=ref("origin"),
                    )
                )
        if mutate:
            mutate(facts, software, dense)

    return inputs(tmp_path, mutate=v2, **kwargs)


def bound(model):
    selected, problem, storage, adapter = model
    _, effective = selected.dense.bind(
        problem,
        storage,
        adapter,
        dtypes=selected.dense.dtypes,
        hardware=selected.hardware_identity,
        policy=selected.numerical_policy,
    )
    return effective


def test_pointer_binding_uses_current_problem_not_shape_heuristics(tmp_path):
    model = pointer_inputs(tmp_path)
    adapter = bound(model)
    operations = [Load("lhs", (0, 0), 1, 2, 2, 3, 0, 1), Load("weight", (0, 0), 1, 2, 3, 4, 0, 1), Store(1, 2, 2, 3, 0)]
    expected = [Ptr("lhs", Const(16)), Ptr("rhs", Const(15)), Ptr("output", Const(60))]
    for operation, pointer in zip(operations, expected, strict=True):
        actual = adapter.encode(operation)
        assert dict(actual.args)["buffer"] == pointer
        assert adapter.project(actual) == model[3].project(model[3].encode(operation))
    other = bound((model[0], replace(model[1], n=17), model[2], model[3]))
    assert dict(other.encode(operations[1]).args)["buffer"] == Ptr("rhs", Const(19))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda f, s, d: d.pop("external_pointer_arguments"),
        lambda f, s, d: d["external_pointer_arguments"]["rhs_load"].update(space="input_lhs"),
        lambda f, s, d: s["isa"]["instructions"][1]["operands"][-1]["type"].update(space="input_lhs"),
        lambda f, s, d: s["isa"]["instructions"][1]["operands"][-1]["type"].update(
            kind="integer", signed=True, space=None
        ),
        lambda f, s, d: d.update(schema="merlin.dense_schedule.v1"),
        lambda f, s, d: d["external_pointer_arguments"].update(compute=dict(argument="buffer", space="input_lhs")),
    ],
)
def test_malformed_or_v1_pointer_declarations_refuse(tmp_path, mutation):
    with pytest.raises(ValueError):
        pointer_inputs(tmp_path, mutate=mutation)


@pytest.mark.parametrize(
    "field,value",
    [("layout", "unknown"), ("completion", "asynchronous"), ("bounds", "unchecked"), ("access", ["read_only"] * 3)],
)
def test_unknown_logical_transport_is_not_qualified(tmp_path, field, value):
    with pytest.raises(DenseSchedulingUnavailable, match="logical pointer ABI"):
        pointer_inputs(tmp_path, mutate=lambda f, s, d: d["logical_pointer_abi"].update({field: value}))


def test_actual_wrong_hidden_or_duplicate_pointer_refuses(tmp_path):
    adapter = bound(pointer_inputs(tmp_path))
    operation = Load("weight", (0, 0), 1, 2, 3, 4, 0, 1)
    actual = adapter.encode(operation)
    for pointer in (Ptr("lhs", Const(15)), Ptr("rhs", Const(-1)), Ptr("rhs", Const(16))):
        bad = replace(actual, args=tuple((k, pointer if k == "buffer" else v) for k, v in actual.args))
        with pytest.raises(ValueError, match="actual external pointer"):
            adapter.project(bad)
    bad = replace(actual, args=tuple((k, Ptr("rhs", Const(0)) if k == "a" else v) for k, v in actual.args))
    with pytest.raises(ValueError, match="undeclared external pointer"):
        adapter.project(bad)
    with pytest.raises(ValueError, match="operand"):
        adapter.project(replace(actual, args=(*actual.args, ("buffer", dict(actual.args)["buffer"]))))
    original = adapter.encode
    poisoned = replace(
        pointer_inputs(tmp_path)[3],
        encode=lambda op: replace(
            original(op), args=tuple((k, Ptr("output", Const(0)) if k == "buffer" else v) for k, v in original(op).args)
        ),
    )
    model = pointer_inputs(tmp_path)
    with pytest.raises(ValueError, match="encoded external pointer"):
        bound((*model[:3], poisoned)).encode(operation)


def test_v2_readout_is_explicitly_unavailable(tmp_path):
    with pytest.raises(DenseSchedulingUnavailable, match="readout pointer"):
        pointer_inputs(tmp_path, mutate=lambda f, s, d: d.update(output_readout={}))


def test_v1_contract_and_encode_remain_integer_only(tmp_path):
    model = inputs(tmp_path)
    assert model[0].dense.external_pointer_arguments == ()
    adapter = bound(model)
    operation = Compute(0, (1, 2), 2, 3, True)
    assert not any(type(v) is Ptr for _, v in adapter.encode(operation).args)


def test_v2_pointer_binding_is_consumed_by_existing_single_schedule(tmp_path):
    from test_selected_dense_contracts import schedule

    from merlin.compile.scheduling.compact_dense import _static_calls

    result = schedule(pointer_inputs(tmp_path))
    assert result is not None
    calls = tuple(_static_calls(result.kernel.body))
    assert all(
        any(type(v) is Ptr for _, v in call.args)
        for call in calls
        if call.instr in ("lhs_load", "weight_load", "store")
    )


def test_synthesized_pointer_keeps_original_isa_operand_position(tmp_path):
    def middle(facts, software, dense):
        for ins in software["isa"]["instructions"]:
            if ins["name"] in ("lhs_load", "weight_load", "store"):
                ins["operands"].insert(2, ins["operands"].pop())

    model = pointer_inputs(tmp_path, mutate=middle)
    adapter = bound(model)
    for operation in (
        Load("lhs", (0, 0), 1, 2, 2, 3, 0, 1),
        Load("weight", (0, 0), 1, 2, 3, 4, 0, 1),
        Store(1, 2, 2, 3, 0),
    ):
        actual = adapter.encode(operation)
        declared = model[0].isa.instruction(actual.instr).arguments()
        assert tuple(key for key, _ in actual.args) == tuple(arg.name for arg in declared)
        assert type(actual.args[2][1]) is Ptr
        assert adapter.project(actual) == model[3].project(model[3].encode(operation))
