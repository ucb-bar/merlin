"""Actual compact PE residency, ordered arithmetic and corrupted retained owners."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, fixture
from test_compact_dense import encode as plain_encode
from test_compact_dense import project as plain_project

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import (
    DenseLoopAdapter,
    emit_compact_dense,
    schedule_compact_dense,
)
from merlin.compile.scheduling.compact_verification import verify_compact_dense
from merlin.sched.ir import Call, Loop, as_expr, evaluate, instances
from merlin.transforms.dense_planning import dense_blocks


def encode(operation):
    if isinstance(operation, Preload) and operation.weight_row is None:
        names = ("k", "n", "acc", "continue", "rows", "cols", "depth")
        values = (
            *operation.weight_block,
            operation.accumulator_row,
            operation.accumulate,
            operation.rows,
            operation.cols,
            operation.weight_rows,
        )
        return Call("preload_keep", tuple(zip(names, map(as_expr, values))))
    if isinstance(operation, Compute) and not operation.fresh_weights:
        return replace(plain_encode(operation), instr="compute_keep")
    return plain_encode(operation)


def project(call):
    if call.instr == "preload_keep":
        v = dict(call.args)
        return Preload(None, (v["k"], v["n"]), v["acc"], v["continue"], v["rows"], v["cols"], v["depth"])
    if call.instr == "compute_keep":
        return replace(plain_project(replace(call, instr="compute")), fresh_weights=False)
    return plain_project(call)


ADAPTER_PE = DenseLoopAdapter(
    encode,
    project,
    "neutral-config",
    "source-ordered",
    True,
    True,
    64,
    qualification_identity="neutral-pe-retention-v1",
    retained_weights=True,
)


def schedule(p, r, s, *, adapter=ADAPTER_PE, dtypes=("i8", "i8", "i32"), **kw):
    return schedule_compact_dense(
        p,
        r,
        s,
        adapter,
        name="dense",
        dtypes=dtypes,
        static_budget=100,
        external_buffers_disjoint=True,
        verification_timeout_ms=20_000,
        reuse_pe_weights=True,
        **kw,
    )


def check(kernel, p, r, s, adapter=ADAPTER_PE):
    return verify_compact_dense(kernel, p, r, s, adapter, static_budget=100, timeout_ms=20_000)


def rewrite(kernel, transform):
    def body(statements):
        return tuple(transform(replace(s, body=body(s.body)) if isinstance(s, Loop) else s) for s in statements)

    return replace(kernel, body=body(kernel.body))


@pytest.mark.parametrize("shape", [(5, 14, 13), (7, 13, 25), (1, 14, 13), (8, 15, 16)])
@pytest.mark.parametrize("order", [("m", "n"), ("n", "m")])
def test_rectangular_tails_and_real_weight_major_reuse(shape, order):
    import numpy as np

    p, r, s = fixture(shape=shape, padded=True)
    result = schedule(p, r, s, outer_order=order)
    A = np.arange(p.m * p.k).reshape(p.m, p.k) % 7 - 3
    B = np.arange(p.k * p.n).reshape(p.k, p.n) % 5 - 2
    output = np.zeros((p.m, p.n), dtype=np.int64)
    memory = {v.name: {} for v in (s.lhs, s.rhs, s.accumulator)}
    weights = descriptor = None
    fresh = retained = 0
    for call, env in instances(result.kernel):
        op = project(replace(call, args=tuple((k, evaluate(v, env)) for k, v in call.args)))
        if isinstance(op, Load):
            view, source = (s.lhs, A) if op.role == "lhs" else (s.rhs, B)
            for i in range(op.rows):
                for j in range(op.cols):
                    memory[view.name][op.row + i, j] = int(source[op.dram_row + i, op.dram_col + j])
        elif isinstance(op, Preload):
            assert descriptor is None
            descriptor = op
            if op.weight_row is not None:
                weights = tuple(
                    tuple(memory[s.rhs.name][op.weight_row + i, j] for j in range(op.cols))
                    for i in range(op.weight_rows)
                )
                fresh += 1
        elif isinstance(op, Compute):
            assert descriptor is not None and weights is not None
            retained += not op.fresh_weights
            for i in range(op.rows):
                for j in range(descriptor.cols):
                    value = (
                        memory[s.accumulator.name][descriptor.accumulator_row + i, j] if descriptor.accumulate else 0
                    )
                    for k in range(op.cols):
                        value += memory[s.lhs.name][op.input_row + i, k] * weights[k][j]
                    memory[s.accumulator.name][descriptor.accumulator_row + i, j] = value
            descriptor = None
        else:
            assert isinstance(op, Store) and descriptor is None
            for i in range(op.rows):
                for j in range(op.cols):
                    output[op.dram_row + i, op.dram_col + j] = memory[s.accumulator.name][op.accumulator_row + i, j]
    np.testing.assert_array_equal(output, A @ B)
    assert fresh + retained == p.tiles[0] * p.tiles[1] * p.tiles[2]
    if p.tiles[0] > 1:
        assert retained > 0
        assert result.realization["pe_weight_loads"] == fresh
        assert result.realization["retained_weight_computes"] == retained
    result.revalidate()


def test_default_emission_remains_byte_identical():
    p, r, s = fixture()
    block = dense_blocks(p, r)[-1]
    before = emit_compact_dense(p, block, s, ADAPTER, name="dense", dtypes=("i8", "i8", "i32"))
    after = emit_compact_dense(p, block, s, ADAPTER, name="dense", dtypes=("i8", "i8", "i32"), reuse_pe_weights=False)
    assert before.text() == after.text()
    # Pinned from the original 1f7470f1 emitter for these exact neutral inputs.
    assert after.digest() == "404fb27bbfaf87f11c8bdcecc6f4c84a16e9c15d7a2986983b5a490c68a0fd54"
    assert "retained_weight_computes" not in check(before, p, r, s, ADAPTER)


@pytest.mark.parametrize(
    "adapter", [replace(ADAPTER_PE, retained_weights=False), replace(ADAPTER_PE, qualification_identity=None)]
)
def test_missing_original_permission_refuses_before_encoding(adapter):
    p, r, s = fixture()

    def forbidden(_op):
        raise AssertionError("encoder must not run")

    with pytest.raises(ValueError, match="qualified retained"):
        schedule(p, r, s, adapter=replace(adapter, encode=forbidden))


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_permission_requires_a_boolean(value):
    with pytest.raises(ValueError, match="Boolean"):
        replace(ADAPTER_PE, retained_weights=value)


def test_retained_weights_need_a_prior_matching_pe_writer():
    p, r, s = fixture()
    result = schedule(p, r, s)

    def without_fresh(statement):
        if isinstance(statement, Call) and statement.instr == "preload":
            return encode(replace(project(statement), weight_row=None))
        if isinstance(statement, Call) and statement.instr == "compute":
            return replace(statement, instr="compute_keep")
        return statement

    with pytest.raises(ValueError, match="latest fresh weight owner"):
        check(rewrite(result.kernel, without_fresh), p, r, s)


def test_fresh_weight_clobber_between_reused_consumers_is_rejected():
    p, r, s = fixture(shape=(6, 12, 16))
    result = schedule(p, r, s)
    changed = False

    def split(statements):
        nonlocal changed
        answer = []
        for statement in statements:
            if (
                not changed
                and isinstance(statement, Loop)
                and len(statement.body) == 3
                and isinstance(statement.body[0], Call)
                and statement.body[0].instr == "preload"
                and isinstance(statement.body[2], Loop)
                and statement.extent > 1
            ):
                answer.extend(
                    (replace(statement, body=statement.body[:2]), replace(statement, body=statement.body[2:]))
                )
                changed = True
            else:
                answer.append(
                    replace(statement, body=split(statement.body)) if isinstance(statement, Loop) else statement
                )
        return tuple(answer)

    bad = replace(result.kernel, body=split(result.kernel.body))
    assert changed
    with pytest.raises(ValueError, match="latest fresh weight owner"):
        check(bad, p, r, s)


@pytest.mark.parametrize("operand", ["k", "n", "acc", "continue", "rows", "depth"])
def test_changed_retained_operands_refuse(operand):
    p, r, s = fixture()
    result = schedule(p, r, s)

    def change(statement):
        if isinstance(statement, Call) and statement.instr == "preload_keep":
            return replace(statement, args=tuple((k, as_expr(100) if k == operand else v) for k, v in statement.args))
        return statement

    with pytest.raises(ValueError):
        check(rewrite(result.kernel, change), p, r, s)


def test_512_cube_checks_symbolically_without_execution_expansion(monkeypatch):
    p, r, s = fixture(shape=(512, 512, 512), tile=(16, 16, 16))
    r = replace(r, operand_bytes=1 << 20, accumulator_bytes=1 << 20, max_m_blocks=4, max_n_blocks=4, max_k_blocks=2)
    s = replace(
        s,
        lhs=replace(s.lhs, capacity=256),
        rhs=replace(s.rhs, capacity=256),
        accumulator=replace(s.accumulator, capacity=256),
    )
    import merlin.sched.ir as ir

    monkeypatch.setattr(
        ir, "instances", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no dynamic expansion"))
    )
    result = schedule(p, r, s)
    assert result is not None and result.realization["output_tiles"] == 1024
    assert result.realization["retained_weight_computes"] > 0
    assert result.realization["static_instruction_calls"] < 40
    assert len(result.kernel.text()) < 15000
