"""Exact Presburger writer witnesses; no skipped lifetime obligations."""

import pytest
import z3

from merlin.compile.scheduling.compact_verification import _defined_exists


def equivalent(left, right):
    solver = z3.Solver()
    solver.set(timeout=2000)
    solver.add(z3.Xor(left, right))
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("coefficient", [-8, -3, -1, 1, 3, 8])
def test_defined_integer_witness_retains_divisibility_bounds_and_nested_overwrites(coefficient):
    writer, reader, other = z3.Ints("writer reader other")
    claim = z3.And(
        writer >= -5,
        writer <= 5,
        coefficient * writer + 2 * reader + 1 == 0,
        z3.Not(z3.Exists(other, z3.And(other > writer, other < writer + 2, other == reader))),
    )
    complete, preferred = _defined_exists([writer], claim)
    reference = z3.Or(*(z3.substitute(claim, (writer, z3.IntVal(i))) for i in range(-5, 6)))
    equivalent(complete, reference)
    equivalent(preferred, complete)
    assert not z3.is_quantifier(complete)


def test_nonlinear_or_disjunctive_equations_keep_the_original_quantifier():
    writer, reader = z3.Ints("writer reader")
    for claim in (writer * writer == reader, z3.Or(writer == reader, writer == reader + 1)):
        complete, _ = _defined_exists([writer], claim)
        assert z3.is_quantifier(complete)
        equivalent(complete, z3.Exists(writer, claim))


def test_preferred_common_scope_witness_is_sufficient_but_not_assumed():
    writer, reader, unrelated = z3.Ints("writer reader unrelated")
    claim = z3.And(writer > reader, writer < reader + 2, unrelated >= 0)
    complete, preferred = _defined_exists([writer], claim, instances=((writer, reader),))
    equivalent(complete, unrelated >= 0)
    equivalent(preferred, z3.BoolVal(False))
    # A failing same-iteration proposal cannot remove a valid preceding/next
    # iteration witness from the complete obligation used by the caller.
    equivalent(z3.Or(preferred, complete), z3.Exists(writer, claim))


def test_preferred_witness_retains_all_other_quantifiers_and_cannot_admit_bad_write():
    writer, reader, other = z3.Ints("writer reader other")
    claim = z3.And(writer <= reader, z3.Not(z3.Exists(other, z3.And(other >= writer, other <= reader))))
    complete, preferred = _defined_exists([writer], claim, instances=((writer, reader),))
    equivalent(complete, z3.BoolVal(False))
    equivalent(preferred, z3.BoolVal(False))


@pytest.mark.parametrize("shape", [(512, 512, 512), (32, 2048, 2048)])
def test_actual_four_band_panel_projection_fits_original_five_second_budget(shape):
    from dataclasses import replace

    from test_compact_dense import ADAPTER, fixture

    from merlin.compile.scheduling.block_schedule import Load
    from merlin.compile.scheduling.dense_panel_pipeline import (
        DensePanelOverlapAdmission,
        build_dense_panel_pipeline,
        verify_dense_panel_pipeline,
    )
    from merlin.sched.ir import Loop
    from merlin.sched.primitives import Cursor, divide_loop, unroll
    from merlin.transforms.dense_planning import dense_blocks

    p, r, storage = fixture(shape=shape, tile=(16, 16, 16))
    storage = replace(
        storage,
        lhs=replace(storage.lhs, capacity=64),
        rhs=replace(storage.rhs, capacity=1024),
        accumulator=replace(storage.accumulator, capacity=1024),
    )
    r = replace(
        r, operand_bytes=1088 * 16, accumulator_bytes=1024 * 64, max_m_blocks=2, max_k_blocks=1, max_n_blocks=32
    )
    block = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, 1, 32))
    admission = DensePanelOverlapAdmission(
        ADAPTER.configuration_identity, ADAPTER.numerical_policy_identity, "neutral-four-band-effects", 64, 34, 2, 2
    )
    kernel = build_dense_panel_pipeline(
        p, block, r, storage, ADAPTER, name="expanded_bands", dtypes=("i8", "i8", "i32")
    )

    def paths(body, prefix=()):
        for statement in body:
            if type(statement) is not Loop:
                continue
            path = (*prefix, statement.var)
            if len(statement.body) == 1 and not isinstance(statement.body[0], Loop):
                operation = (
                    ADAPTER.project(statement.body[0])
                    if statement.body[0].instr != "merlin_dense_panel_drain"
                    else None
                )
                if type(operation) is Load and operation.role == "weight" and statement.extent == 32:
                    yield path
            yield from paths(statement.body, path)

    # This is the exact primitive-band representation of wide commands: four
    # static adjacent RHS loads, each counted group advancing by four columns.
    # No dynamic loop domain is visited to prepare or verify this regression.
    for index, path in enumerate(tuple(paths(kernel.body))):
        outer, inner = f"band_group_{index}", f"band_lane_{index}"
        kernel = divide_loop(kernel, Cursor(loops=path), 4, names=(outer, inner))
        kernel = unroll(kernel, Cursor(loops=(*path[:-1], outer, inner)))
    report = verify_dense_panel_pipeline(
        kernel, p, block, r, storage, ADAPTER, admission, static_budget=128, timeout_ms=5000
    )
    assert report["source"]["output_tiles"] == p.tiles[0] * p.tiles[2]
    assert report["source"]["rhs_loads"] == p.tiles[0] // 2 * p.tiles[1] * p.tiles[2]
