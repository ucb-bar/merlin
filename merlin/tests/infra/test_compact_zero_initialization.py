"""Explicit actual accumulator zeros, ordered continuation and reset barriers."""

from dataclasses import asdict, replace

import pytest
from test_compact_dense import ADAPTER, fixture
from test_compact_dense import encode as plain_encode
from test_compact_dense import project as plain_project

from merlin.compile.scheduling.block_schedule import AccumulatorZero, Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import emit_compact_dense
from merlin.compile.scheduling.compact_verification import verify_compact_dense
from merlin.sched.ir import Call, Kernel, Loop, TensorArg, Var, as_expr, evaluate, instances, select
from merlin.transforms.dense_planning import dense_blocks


def encode(operation):
    if isinstance(operation, AccumulatorZero):
        return Call(
            "zero",
            tuple(zip(("row", "rows", "cols"), map(as_expr, (operation.local_row, operation.rows, operation.cols)))),
        )
    return plain_encode(operation)


def project(call):
    if call.instr == "zero":
        values = dict(call.args)
        return AccumulatorZero(values["row"], values["rows"], values["cols"])
    return plain_project(call)


ADAPTER_ZERO = replace(
    ADAPTER,
    encode=encode,
    project=project,
    zero_initialization=True,
    qualification_identity="neutral-exact-synchronous-zero-v1",
)


def explicit_kernel(p, s, dtypes=("i8", "i8", "i32")):
    """A distinct actual program: clear each output tile, then accumulate all K."""
    m, k, n = map(Var, ("m", "k", "n"))
    mt, kt, nt = p.tiles
    rows = select(m, mt - 1, (p.m - 1) % p.tile_m + 1, p.tile_m)
    depth = select(k, kt - 1, (p.k - 1) % p.tile_k + 1, p.tile_k)
    cols = select(n, nt - 1, (p.n - 1) % p.tile_n + 1, p.tile_n)
    zero = encode(AccumulatorZero(s.accumulator.base, rows, cols))
    reduction = (
        encode(Load("lhs", (m, k), m * p.tile_m, k * p.tile_k, rows, depth, s.lhs.base, 1)),
        encode(Load("weight", (k, n), k * p.tile_k, n * p.tile_n, depth, cols, s.rhs.base, 1)),
        encode(Preload(s.rhs.base, (k, n), s.accumulator.base, 1, rows, cols, depth)),
        encode(Compute(s.lhs.base, (m, k), rows, depth, True)),
    )
    store = encode(Store(m * p.tile_m, n * p.tile_n, rows, cols, s.accumulator.base))
    return Kernel(
        "explicit_zero",
        (
            TensorArg("lhs", (p.m, p.k), dtypes[0], "read"),
            TensorArg("rhs", (p.k, p.n), dtypes[1], "read"),
            TensorArg("output", (p.m, p.n), dtypes[2], "write"),
        ),
        (Loop("m", mt, (Loop("n", nt, (zero, Loop("k", kt, reduction), store)),)),),
    )


def check(kernel, p, r, s, adapter=ADAPTER_ZERO):
    return verify_compact_dense(kernel, p, r, s, adapter, static_budget=100, timeout_ms=20_000)


def tile_body(kernel, change):
    outer = kernel.body[0]
    inner = outer.body[0]
    return replace(kernel, body=(replace(outer, body=(replace(inner, body=change(inner.body)),)),))


@pytest.mark.parametrize("shape,widths", [((5, 14, 13), (1, 1, 4)), ((7, 9, 16), (2, 1, 4)), ((1, 3, 2), (2, 2, 4))])
def test_explicit_full_zero_and_every_compute_accumulates(shape, widths):
    import numpy as np

    p, r, s = fixture(shape=shape, widths=widths, padded=True)
    kernel = explicit_kernel(p, s, tuple("i" + str(w * 8) for w in widths))
    report = check(kernel, p, r, s)
    assert report["accumulator_zero_writes"] == p.tiles[0] * p.tiles[2]
    a = (np.arange(p.m * p.k) % 7 - 3).reshape(p.m, p.k)
    b = (np.arange(p.k * p.n) % 5 - 2).reshape(p.k, p.n)
    output = np.full((p.m, p.n), 999, dtype=np.int64)
    memory = {view.name: {} for view in (s.lhs, s.rhs, s.accumulator)}
    preload = None
    for call, env in instances(kernel):
        op = project(replace(call, args=tuple((key, evaluate(value, env)) for key, value in call.args)))
        if isinstance(op, AccumulatorZero):
            for i in range(op.rows):
                for j in range(op.cols):
                    memory[s.accumulator.name][op.local_row + i, j] = 0
        elif isinstance(op, Load):
            view, source = (s.lhs, a) if op.role == "lhs" else (s.rhs, b)
            for i in range(op.rows):
                for j in range(op.cols):
                    memory[view.name][op.row + i, j] = int(source[op.dram_row + i, op.dram_col + j])
        elif isinstance(op, Preload):
            assert op.accumulate == 1
            preload = op
        elif isinstance(op, Compute):
            assert preload is not None
            for i in range(op.rows):
                for j in range(preload.cols):
                    value = memory[s.accumulator.name][preload.accumulator_row + i, j]
                    for k in range(op.cols):
                        value += memory[s.lhs.name][op.input_row + i, k] * memory[s.rhs.name][preload.weight_row + k, j]
                    memory[s.accumulator.name][preload.accumulator_row + i, j] = value
        else:
            assert isinstance(op, Store)
            for i in range(op.rows):
                for j in range(op.cols):
                    output[op.dram_row + i, op.dram_col + j] = memory[s.accumulator.name][op.accumulator_row + i, j]
    np.testing.assert_array_equal(output, a @ b)


@pytest.mark.parametrize(
    "adapter", [replace(ADAPTER_ZERO, zero_initialization=False), replace(ADAPTER_ZERO, qualification_identity=None)]
)
def test_original_zero_law_is_required(adapter):
    p, r, s = fixture()
    with pytest.raises(ValueError, match="qualified zero"):
        check(explicit_kernel(p, s), p, r, s, adapter)


@pytest.mark.parametrize("bad", [0, 1, None, "true"])
def test_zero_permission_is_exact_boolean(bad):
    with pytest.raises(ValueError, match="Boolean"):
        replace(ADAPTER_ZERO, zero_initialization=bad)


@pytest.mark.parametrize(
    "mode", ["missing", "late", "partial_rows", "partial_cols", "wrong_row", "interior_reset", "before_store_reset"]
)
def test_missing_partial_or_late_zero_and_live_reduction_clobbers_refuse(mode):
    p, r, s = fixture()
    kernel = explicit_kernel(p, s)

    def change(body):
        zero, loop, store = body
        if mode == "missing":
            return loop, store
        if mode == "late":
            return loop, zero, store
        if mode == "interior_reset":
            return zero, replace(loop, body=(*loop.body[:2], zero, *loop.body[2:])), store
        if mode == "before_store_reset":
            return zero, loop, zero, store
        field = {"partial_rows": "rows", "partial_cols": "cols", "wrong_row": "row"}[mode]
        value = s.accumulator.base + 1 if field == "row" else 1
        changed = replace(zero, args=tuple((key, as_expr(value) if key == field else old) for key, old in zero.args))
        return changed, loop, store

    with pytest.raises(ValueError, match="zero seed|accumulator continuation|accumulator overwrite"):
        check(tile_body(kernel, change), p, r, s)


def test_zero_reset_is_also_a_clobber_of_overwrite_seeded_reduction():
    p, r, s = fixture()
    kernel = explicit_kernel(p, s)

    def change(body):
        zero, loop, store = body
        # K0 overwrites, later K continues; a zero before store still destroys it.
        preload = loop.body[2]
        changed = replace(
            preload,
            args=tuple((key, select(Var("k"), 0, 0, 1) if key == "continue" else old) for key, old in preload.args),
        )
        return replace(loop, body=(*loop.body[:2], changed, loop.body[3])), zero, store

    with pytest.raises(ValueError, match="accumulator overwrite"):
        check(tile_body(kernel, change), p, r, s)


def test_implicit_overwrite_seed_remains_separate_and_unchanged():
    p, r, s = fixture()
    kernel = emit_compact_dense(p, dense_blocks(p, r)[-1], s, ADAPTER, name="dense", dtypes=("i8", "i8", "i32"))
    assert kernel.digest() == "404fb27bbfaf87f11c8bdcecc6f4c84a16e9c15d7a2986983b5a490c68a0fd54"
    assert "accumulator_zero_writes" not in check(kernel, p, r, s)


def test_512_cube_zero_seed_is_proved_without_expanding_dynamic_products(monkeypatch):
    p, r, s = fixture(shape=(512, 512, 512), tile=(16, 16, 16))
    s = replace(
        s,
        lhs=replace(s.lhs, capacity=32),
        rhs=replace(s.rhs, capacity=32),
        accumulator=replace(s.accumulator, capacity=32),
    )
    r = replace(r, operand_bytes=4096, accumulator_bytes=4096)
    import merlin.sched.ir as ir

    monkeypatch.setattr(
        ir, "instances", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no dynamic expansion"))
    )
    kernel = explicit_kernel(p, s)
    report = check(kernel, p, r, s)
    assert report["output_tiles"] == 1024 and report["accumulator_zero_writes"] == 1024
    assert report["static_instruction_calls"] == 6
    assert len(kernel.text()) < 5000


def test_staged_zero_constructor_remains_the_same_shared_effect():
    from merlin.compile.scheduling.staged_dense import AccumulatorZero as StagedZero

    assert StagedZero is AccumulatorZero
    assert asdict(StagedZero(1, 2, 3)) == {"local_row": 1, "rows": 2, "cols": 3}


def test_zero_permission_changed_by_projector_refuses_immediately():
    p, r, s = fixture()
    calls = []
    adapter = None

    def changed(call):
        calls.append(1)
        object.__setattr__(adapter, "zero_initialization", False)
        return project(call)

    adapter = replace(ADAPTER_ZERO, project=changed)
    with pytest.raises(ValueError, match="permissions or callbacks changed"):
        check(explicit_kernel(p, s), p, r, s, adapter)
    assert len(calls) == 1


def test_zero_between_fresh_and_retained_compute_is_a_pe_clobber():
    from test_compact_pe_weights import ADAPTER_PE, schedule
    from test_compact_pe_weights import project as pe_project

    p, r, s = fixture(shape=(6, 3, 8))
    kernel = schedule(p, r, s).kernel

    def with_zero(body):
        result = []
        for statement in body:
            if isinstance(statement, Loop):
                statement = replace(statement, body=with_zero(statement.body))
            elif statement.instr == "preload_keep":
                descriptor = pe_project(statement)
                result.append(encode(AccumulatorZero(descriptor.accumulator_row, descriptor.rows, descriptor.cols)))
            result.append(statement)
        return tuple(result)

    adapter = replace(
        ADAPTER_PE,
        project=lambda call: project(call) if call.instr == "zero" else pe_project(call),
        zero_initialization=True,
    )
    with pytest.raises(ValueError, match="latest fresh weight owner"):
        check(replace(kernel, body=with_zero(kernel.body)), p, r, s, adapter)
