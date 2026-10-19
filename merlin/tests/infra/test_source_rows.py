"""Typed source row admission, immutable inputs and independent complete output."""

from dataclasses import replace
from math import prod

import numpy as np
import pytest
from test_row_softmax import (
    execute_native,
    exp_object,  # noqa: F401
    native_program,
)
from test_selected_row_contracts import independent_reference, inputs
from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.ir import Block, Region

from merlin.compile.scheduling.row_softmax import RowSoftmaxUnavailable, compile_source_rows, read_source_rows
from merlin.llvmlower.cpu_softmax import CPU_SOFTMAX_F32_AVX2_POLICY, build_cpu_softmax_lastdim


def module(shape=(2, 19), *, two_inputs=False):
    typ = builtin.TensorType(builtin.f32, shape)
    block = Block(arg_types=[typ] * (2 if two_inputs else 1))
    operations, result = build_cpu_softmax_lastdim(
        block.args[0], backend_policy=CPU_SOFTMAX_F32_AVX2_POLICY, assume_finite_inputs=True
    )
    block.add_ops([*operations, func.ReturnOp(result)])
    original = builtin.ModuleOp([func.FuncOp("forward", ([arg.type for arg in block.args], [typ]), Region(block))])
    original.verify()
    return original, operations[-1]


def case(path, shape=(2, 19), *, separate=True, index_bits=64):
    def capacity(facts, software, row):
        size = max(512, 4 * (shape[-1] + 16))
        facts["sizes"][:] = [size, size]
        facts["capacities"][:2] = [size, size]

    selected, options = inputs(
        path,
        dtype="f32",
        separate=separate,
        columns=shape[-1],
        rows=prod(shape[:-1]),
        index_bits=index_bits,
        mutate=capacity,
    )
    original, final = module(shape)
    kwargs = dict(
        selected_contracts=selected,
        adapter=options["adapter"],
        numerical_policy=CPU_SOFTMAX_F32_AVX2_POLICY,
        assume_finite_inputs=True,
        private_ownership=True,
        disjoint_arguments=True,
    )
    return selected, options, original, final, kwargs


@pytest.mark.parametrize(
    "shape,separate,index_bits",
    [((19,), False, 64), ((2, 3, 33), True, 32), ((2, 113), False, 64), ((2, 257), True, 64)],
)
def test_source_rows_match_independent_policy_and_realized_native_effects(
    tmp_path,
    exp_object,  # noqa: F811
    shape,
    separate,
    index_bits,
):
    _, options, original, final, kwargs = case(tmp_path, shape, separate=separate, index_bits=index_bits)
    values = np.random.default_rng(711).normal(0, 4, shape).astype("f4")
    rows = values.reshape(-1, shape[-1])
    rows[0] = np.linspace(-120, 0, shape[-1], dtype="f4")
    rows[-1] = np.finfo("f4").min
    rows[-1, 0] = np.finfo("f4").max
    wanted = independent_reference(tmp_path, values, None)
    observations = []
    for name, chunk in (("one", 1), ("default", None)):
        result = compile_source_rows(original, final, chunk_elements=chunk, **kwargs)
        result.revalidate()
        assert result.source.source is next(op for op in original.walk() if op.name == "func.func").body.block.args[0]
        source = native_program(result, options)
        actual, counts = execute_native(
            tmp_path, source, values, exp_object, name, tuple(dict.fromkeys(bank.backing for bank in options["banks"]))
        )
        np.testing.assert_array_equal(actual.view("u4"), wanted.view("u4"))
        for key, value in counts.items():
            assert value == result.compiled.report[key]
        observations.append(counts)
    assert observations[1]["executed_commands"] < observations[0]["executed_commands"]
    assert observations[1]["requested_read_bytes"] < observations[0]["requested_read_bytes"]


def read(final, **kwargs):
    return read_source_rows(final, numerical_policy=CPU_SOFTMAX_F32_AVX2_POLICY, assume_finite_inputs=True, **kwargs)


def test_source_row_admission_survives_ordinary_serialization(tmp_path):
    from merlin.common.mlir_query import parse
    from merlin.transforms.certificates import ir_text

    original, _ = module()
    source = tmp_path / "model.mlir"
    source.write_text(ir_text(original))
    parsed = parse(source)
    body = next(op for op in parsed.walk() if type(op) is func.FuncOp).body.block
    admitted = read(body.last_op.arguments[0].owner)
    seed = admitted.operations[0].value
    assert seed.data.data == bytes.fromhex("000080ff") * 2
    assert admitted.shape == (2, 19)


@pytest.mark.parametrize(
    "fault",
    [
        "tree",
        "loop_step",
        "loop_end",
        "lane_initialization",
        "seed",
        "polynomial",
        "source_binding",
        "metadata",
        "intermediate_escape",
        "nested_escape",
        "block_argument_escape",
        "final_region_escape",
        "result_type",
        "underflow_sign",
    ],
)
def test_changed_source_algorithm_cannot_reuse_selected_row_primitives(fault):
    original, final = module(two_inputs=True)
    block = next(op for op in original.walk() if op.name == "func.func").body.block
    source = read(final)
    if fault == "underflow_sign":
        constant = next(
            op for op in original.walk() if type(op) is arith.ConstantOp
            and isinstance(op.value, builtin.FloatAttr) and op.value.value.data == 0.0
        )
        constant.properties["value"] = builtin.FloatAttr(-0.0, builtin.f32)
    elif fault == "tree":
        denominator = source.operations[-3]
        add = next(op for op in denominator.walk() if type(op) is arith.AddfOp)
        add.operands = tuple(reversed(add.operands))
    elif fault in ("loop_step", "loop_end", "lane_initialization"):
        loop = next(op for op in original.walk() if type(op) is scf.ForOp)
        if fault != "lane_initialization":
            value = loop.step if fault == "loop_step" else loop.ub
            value.owner.properties["value"] = builtin.IntegerAttr(
                7 if fault == "loop_step" else 18, builtin.IndexType()
            )
        else:
            zero = arith.ConstantOp(builtin.FloatAttr(0.0, builtin.f32))
            loop.parent.insert_op_before(zero, loop)
            loop.operands = (*loop.operands[:3], zero.result)
    elif fault == "seed":
        init = source.operations[0]
        init.properties["value"] = builtin.DenseIntOrFPElementsAttr.from_list(init.results[0].type, [0.0])
    elif fault == "polynomial":
        constant = next(
            op
            for op in original.walk()
            if type(op) is arith.ConstantOp
            and isinstance(op.value, builtin.FloatAttr)
            and op.value.value.data == -104.0
        )
        constant.properties["value"] = builtin.FloatAttr(-103.0, builtin.f32)
    elif fault == "source_binding":
        exponential = source.operations[3]
        exponential.operands = (block.args[1], *exponential.operands[1:])
    elif fault == "metadata":
        next(op for op in final.walk() if type(op) is arith.MulfOp).attributes["unknown"] = builtin.StringAttr("effect")
    elif fault.endswith("escape"):
        if fault == "intermediate_escape":
            value = source.operations[1].results[0]
        elif fault == "nested_escape":
            value = next(op for op in source.operations[3].walk() if type(op) is arith.SubfOp).result
        elif fault == "block_argument_escape":
            value = next(op for op in original.walk() if type(op) is scf.ForOp).body.block.args[1]
        else:
            value = next(op for op in final.walk() if type(op) is arith.MulfOp).result
        declaration = func.FuncOp.external("observe", [value.type], [])
        original.body.block.add_op(declaration)
        block.insert_op_before(func.CallOp("observe", [value], []), block.last_op)
        original.verify()  # xDSL currently does not diagnose these region escapes.
    else:
        # xDSL structural equivalence alone misses this result-type mutation.
        # Deliberately call the standalone reader, independently of module.verify.
        empty = next(op for op in source.operations if type(op) is tensor.EmptyOp)
        empty.results[0]._type = builtin.TensorType(builtin.f64, empty.results[0].type.get_shape())
    with pytest.raises(RowSoftmaxUnavailable, match="differs|observation|bindings"):
        read(final)


def test_pattern_matching_never_adds_detached_uses_or_changes_source():
    original, final = module()
    before = str(original)
    uses = {value: tuple(value.uses) for op in original.walk() for value in (*op.operands, *op.results)}
    for _ in range(3):
        assert read(final).operations[-1] is final
    assert str(original) == before
    assert all(tuple(value.uses) == prior for value, prior in uses.items())


@pytest.mark.parametrize(
    "field,value", [("assume_finite_inputs", False), ("numerical_policy", "unknown"), ("max_source_elements", 3)]
)
def test_missing_numerical_or_source_budget_admission_keeps_original_route(tmp_path, field, value):
    _, _, original, final, kwargs = case(tmp_path)
    calls = []
    kwargs["adapter"] = replace(kwargs["adapter"], encode=lambda step: calls.append(step))
    kwargs[field] = value
    before = str(original)
    with pytest.raises(RowSoftmaxUnavailable):
        compile_source_rows(original, final, **kwargs)
    assert not calls and str(original) == before


def test_invalid_source_pass_order_rejects_before_any_encoding(tmp_path):
    _, _, original, final, kwargs = case(tmp_path)
    calls = []
    kwargs["adapter"] = replace(kwargs["adapter"], encode=lambda step: calls.append(step))
    with pytest.raises(ValueError, match="prerequisites|ordering"):
        compile_source_rows(original, final, pass_order=("row_source_lowering", "row_source_admission"), **kwargs)
    assert not calls


@pytest.mark.parametrize("fault", [
    "source", "facts", "software", "schedule", "index", "index_stamp", "source_binding", "policy",
])
def test_source_row_results_expire_with_current_inputs(tmp_path, fault):
    selected, _, original, final, kwargs = case(tmp_path)
    result = compile_source_rows(original, final, **kwargs)
    if fault in ("facts", "software"):
        path = selected.facts_path if fault == "facts" else selected.software_path
        path.write_bytes(path.read_bytes() + b"\n")
    elif fault == "source":
        final.attributes["changed"] = builtin.UnitAttr()
    elif fault == "schedule":
        result = replace(result, compiled=replace(result.compiled, kernel=replace(result.compiled.kernel, body=())))
    elif fault in ("index", "index_stamp"):
        changed = dict(index_bits=32)
        if fault == "index_stamp":
            changed["checked_index_bits"] = 32
        result = replace(result, compiled=replace(result.compiled, **changed))
    elif fault == "source_binding":
        result = replace(result, source=replace(result.source, source=final.results[0]))
    else:
        result = replace(
            result, source=replace(result.source, numerics=replace(result.source.numerics, lanes=1, tree=0))
        )
    with pytest.raises(ValueError, match="changed|replan|reselect"):
        result.revalidate()


@pytest.mark.parametrize("replacement", ["software", "missing"])
def test_replacing_a_still_valid_selection_cannot_reuse_checked_rows(tmp_path, replacement):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    selected, _, original, final, kwargs = case(first)
    result = compile_source_rows(original, final, **kwargs)
    other, _ = inputs(second, dtype="f32", separate=True, columns=19, rows=2,
                      mutate=lambda facts, software, row: row["support"].update(partial_chunks=False))
    assert other.hardware_identity == selected.hardware_identity and other.software_bytes != selected.software_bytes
    changed = replace(result.compiled, selected_contracts=other if replacement == "software" else None)
    with pytest.raises(ValueError, match="RTL/software contracts changed"):
        changed.revalidate()
    with pytest.raises(ValueError, match="RTL/software contracts changed"):
        replace(result, compiled=changed).revalidate()
