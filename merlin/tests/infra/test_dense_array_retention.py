"""Counted fresh/STAY transport, independent products and conservative proofs."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER as FRESH_ADAPTER
from test_compact_dense import encode as fresh_encode
from test_compact_dense import project as fresh_project
from test_dense_retained_reduction import chosen, inputs

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.dense_array_retention import (
    DenseArrayRetentionAdmission,
    propose_dense_array_retention,
    verify_dense_array_retention,
    verify_dense_array_tile_loads,
)
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
from merlin.sched.ir import Call, Const, Kernel, Loop, Var, instances
from merlin.sched.ir.kernel import concretize


def encode(op):
    if type(op) is Preload and op.weight_row is None:
        assert op.weight_block is None
        return Call("keep_descriptor", tuple(zip(("acc", "continue", "rows", "cols", "depth"),
                    (op.accumulator_row, op.accumulate, op.rows, op.cols, op.weight_rows))))
    if type(op) is Compute and not op.fresh_weights:
        return replace(fresh_encode(op), instr="stay")
    return fresh_encode(op)


def project(call):
    a = dict(call.args)
    if call.instr == "keep_descriptor":
        return Preload(None, None, a["acc"], a["continue"], a["rows"], a["cols"], a["depth"])
    if call.instr == "stay":
        return replace(fresh_project(replace(call, instr="compute")), fresh_weights=False)
    return fresh_project(call)


ADAPTER = replace(FRESH_ADAPTER, encode=encode, project=project)
LAW = DenseArrayRetentionAdmission(ADAPTER.configuration_identity, ADAPTER.numerical_policy_identity,
                                   "synthetic-array-capture-stay-output-only-v1")


def setup(shape=(9, 14, 25), tile=(2, 3, 4)):
    p, r, s, b, a = inputs(shape, tile)
    original = chosen(p, r, s, b, a, compute_order=("k", "n", "m"))
    return p, r, s, b, a, original


def execute(kernel, problem):
    # Independent scalar semantics over actual projected calls. In particular,
    # STAY reads the captured array, not a reconstructed RHS scratchpad read.
    lhs = [[(m * problem.k + k) % 7 - 3 for k in range(problem.k)] for m in range(problem.m)]
    rhs = [[(k * problem.n + n) % 5 - 2 for n in range(problem.n)] for k in range(problem.k)]
    memory, accumulators, output = {}, {}, {}
    pending = array = None
    for call, env in instances(kernel):
        if call.instr == DRAIN:
            assert pending is None
            continue
        op = project(replace(call, args=concretize(call, env)))
        if type(op) is Load:
            source = lhs if op.role == "lhs" else rhs
            columns = problem.tile_k if op.role == "lhs" else problem.tile_n
            stride = problem.tile_m if op.role == "lhs" else problem.tile_k
            for r in range(op.rows):
                for c in range(op.cols):
                    memory[op.role, op.row + r + c // columns * stride, c % columns] = (
                        source[op.dram_row + r][op.dram_col + c])
        elif type(op) is Preload:
            assert pending is None
            pending = op
        elif type(op) is Compute:
            assert pending is not None
            if op.fresh_weights:
                array = [[memory["weight", pending.weight_row + k, n]
                          for n in range(pending.cols)] for k in range(pending.weight_rows)]
            assert array is not None and len(array) == op.cols
            for r in range(op.rows):
                for n in range(pending.cols):
                    key = (pending.accumulator_row + r, n)
                    value = accumulators[key] if pending.accumulate else 0
                    for k in range(op.cols):
                        value += memory["lhs", op.input_row + r, k] * array[k][n]
                    accumulators[key] = value
            pending = None
        elif type(op) is Store:
            for r in range(op.rows):
                for n in range(op.cols):
                    key = (op.dram_row + r, op.dram_col + n)
                    assert key not in output
                    output[key] = accumulators[op.accumulator_row + r, n]
    expected = {(m, n): sum(lhs[m][k] * rhs[k][n] for k in range(problem.k))
                for m in range(problem.m) for n in range(problem.n)}
    assert output == expected
    return output


def test_actual_counted_first_and_remaining_preserve_tails_products_and_completion():
    p, r, s, b, a, original = setup()
    proposal = propose_dense_array_retention(original.kernel, ADAPTER, LAW)
    report = verify_dense_array_retention(proposal, p, b, r, s, ADAPTER, a, static_budget=256)
    assert proposal.checking_kernel == original.kernel != proposal.kernel
    assert report["retained_pairs"] > 0 and report["scratchpad_rhs_retained"]
    assert execute(proposal.kernel, p) == execute(original.kernel, p)
    assert sum(call.instr == DRAIN for call, _ in instances(proposal.kernel)) == sum(
        call.instr == DRAIN for call, _ in instances(original.kernel))
    from test_scf_commands import record

    from merlin.compile.scheduling.scf_commands import materialize_scf_commands

    actual = materialize_scf_commands(proposal.kernel)
    actual.verify()
    calls = record(actual)
    assert any(name == "stay" for name, _ in calls)
    assert sum(name == "stay" for name, _ in calls) == report["retained_pairs"]
    assert not any(name == "stay" for name, _ in record(materialize_scf_commands(original.kernel)))


def test_finite_block_scheduler_is_an_independent_resident_weight_oracle():
    from merlin.compile.scheduling.block_schedule import Contraction, Geometry, schedule_contraction

    p, _, _, _, _, original = setup((4, 6, 4), (2, 2, 2))
    proposal = propose_dense_array_retention(original.kernel, ADAPTER, LAW)
    finite = schedule_contraction(Contraction(p.m, p.k, p.n), Geometry(2, 128, None, 128, True))
    states = []
    resident = None
    for op in finite.ops:
        if type(op) is Preload and op.weight_row is not None:
            resident = op.weight_block
        elif type(op) is Compute:
            states.append((op.input_block, resident, op.fresh_weights))
    actual, resident = [], None
    for call, env in instances(proposal.kernel):
        if call.instr == DRAIN:
            continue
        op = project(replace(call, args=concretize(call, env)))
        if type(op) is Preload and op.weight_row is not None:
            resident = op.weight_block
        elif type(op) is Compute:
            actual.append((op.input_block, resident, op.fresh_weights))
    # This source is one original output panel: compare exact K,N,M stream.
    assert actual == states


def test_law_adapter_stream_and_unknown_effect_mutations_refuse():
    _, _, _, _, _, original = setup()
    with pytest.raises(ValueError, match="separate typed"):
        propose_dense_array_retention(original.kernel, ADAPTER, None)
    with pytest.raises(ValueError, match="original selected"):
        propose_dense_array_retention(original.kernel, ADAPTER, replace(LAW, hardware_identity="other"))
    with pytest.raises((ValueError, TypeError)):
        propose_dense_array_retention(original.kernel, FRESH_ADAPTER, LAW)
    p, r, s, b, a = inputs()
    wrong_order = chosen(p, r, s, b, a, compute_order=("m", "k", "n"))
    with pytest.raises(ValueError, match="K,N,M"):
        propose_dense_array_retention(wrong_order.kernel, ADAPTER, LAW)
    proposal = propose_dense_array_retention(original.kernel, ADAPTER, LAW)
    with pytest.raises(ValueError, match="changed"):
        replace(proposal, kernel=replace(proposal.kernel, name="mutated")).revalidate(ADAPTER, LAW)

    def add_effect(body):
        return tuple(replace(statement, body=add_effect(statement.body)) if type(statement) is Loop else
                     replace(statement, unit="unknown") if statement.instr == "compute" else statement
                     for statement in body)
    with pytest.raises(ValueError, match="additional call"):
        propose_dense_array_retention(replace(original.kernel, body=add_effect(original.kernel.body)), ADAPTER, LAW)


def test_actual_operand_tampering_and_rhs_alias_still_fail_full_proof():
    p, r, s, b, a, original = setup()
    proposal = propose_dense_array_retention(original.kernel, ADAPTER, LAW)
    # A forged projection/kernel join fails independently of its supplied hash.
    forged = replace(proposal, kernel=replace(proposal.kernel, attrs=(("forged", "yes"),)))
    with pytest.raises(ValueError, match="changed"):
        verify_dense_array_retention(forged, p, b, r, s, ADAPTER, a, static_budget=256)
    aliased = replace(s, rhs=replace(s.rhs, backing=s.lhs.backing, base=s.lhs.base))
    with pytest.raises(ValueError, match="alias|overlapping|capacity"):
        verify_dense_array_retention(proposal, p, b, r, aliased, ADAPTER, a, static_budget=256)


def test_a_clobber_inside_the_resident_run_is_not_an_array_reuse_proposal():
    _, _, _, _, _, original = setup()

    def corrupt(body):
        if len(body) == 2 and all(type(s) is Call for s in body):
            return (body[0], Call("unknown_array_clobber", ()), body[1])
        return tuple(replace(s, body=corrupt(s.body)) if type(s) is Loop else s for s in body)
    with pytest.raises(ValueError):
        propose_dense_array_retention(replace(original.kernel, body=corrupt(original.kernel.body)), ADAPTER, LAW)


def test_conservative_projection_still_proves_actual_load_addresses_and_budget():
    p, r, s, b, a, original = setup()

    def corrupt(body):
        return tuple(replace(c, body=corrupt(c.body)) if type(c) is Loop else
                     replace(c, args=tuple((key, value + 10000 if key == "dst" else value)
                                           for key, value in c.args)) if c.instr == "weight_load" else c for c in body)
    proposal = propose_dense_array_retention(replace(original.kernel, body=corrupt(original.kernel.body)), ADAPTER, LAW)
    with pytest.raises(ValueError, match="storage bounds"):
        verify_dense_array_retention(proposal, p, b, r, s, ADAPTER, a, static_budget=256)
    proposal = propose_dense_array_retention(original.kernel, ADAPTER, LAW)
    with pytest.raises(ValueError, match="static command budget"):
        verify_dense_array_retention(proposal, p, b, r, s, ADAPTER, a, static_budget=1)


def test_actual_scf_to_llvm_consumes_stay_without_emitting_the_fresh_projection():
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp, i64

    from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
    from merlin.compile.scheduling.scf_commands import emit_scf_commands, materialize_scf_commands
    from merlin.llvmlower.host_builder import PTR, FnBuilder

    p, r, s, b, a, original = setup()
    proposal = propose_dense_array_retention(original.kernel, ADAPTER, LAW)
    verify_dense_array_retention(proposal, p, b, r, s, ADAPTER, a, static_budget=256)
    builder, declarations = FnBuilder([PTR] * 3), {}
    bridge = FnBuilderIndexAdapter(builder)

    def lower(instruction, operands, metadata):
        declarations[instruction] = len(operands)
        builder.add(llvm.CallOp(instruction, *(bridge.value(v) for v in operands.values())))

    emit_scf_commands(materialize_scf_commands(proposal.kernel), bridge, lower)
    function = llvm.FuncOp("compiled", llvm.LLVMFunctionType([PTR] * 3, llvm.LLVMVoidType()), body=builder.finish())
    output = ModuleOp([*(llvm.FuncOp(name, llvm.LLVMFunctionType([i64] * count, llvm.LLVMVoidType()))
                         for name, count in declarations.items()), function])
    output.verify()
    assert "stay" in declarations and "keep_descriptor" in declarations
    assert proposal.checking_kernel.digest() == original.kernel.digest()


def test_large_counted_proposal_does_not_expand_the_dynamic_compute_domain():
    # Proposal counts are exact structural counts, not physical admission.
    k, n, m = (Var(axis) for axis in ("k", "n", "m"))
    pair = (encode(Preload(k * 512 + n * 16, (k, n), (m * 32 + n) * 16,
                           Const(1), Const(16), Const(16), Const(16))),
            encode(Compute(m * 2048 + k * 16, (m, k), Const(16), Const(16), True)))
    original = Kernel("counted", (), (Loop("k", 128, (Loop("n", 32, (Loop("m", 32, pair),)),)),))
    proposal = propose_dense_array_retention(original, ADAPTER, LAW)
    assert len(proposal.changes) == 1
    assert proposal.changes[0]["fresh_pairs"] == 128 * 32
    assert proposal.changes[0]["retained_pairs"] == 128 * 32 * 31
    assert len(proposal.kernel.body[0].body[0].body) == 2


def test_m_varying_weight_address_and_ignored_scalar_operands_cannot_be_erased():
    _, _, _, _, _, original = setup()

    def corrupt(body, mode, m=None):
        result = []
        for statement in body:
            if type(statement) is Loop:
                result.append(replace(statement, body=corrupt(statement.body, mode, statement.var)))
            elif statement.instr == "preload":
                # The innermost loop is M in this original K,N,M source.
                if mode == "vary":
                    statement = replace(statement, args=tuple((key, value + Var(m) if key == "weight" else value)
                                                              for key, value in statement.args))
                else:
                    statement = replace(statement, args=(*statement.args, ("unmodeled", Const(1))))
                result.append(statement)
            else:
                result.append(statement)
        return tuple(result)
    with pytest.raises(ValueError, match="invariant across M"):
        bad = replace(original.kernel, body=corrupt(original.kernel.body, "vary"))
        propose_dense_array_retention(bad, ADAPTER, LAW)
    with pytest.raises(ValueError, match="round trips"):
        bad = replace(original.kernel, body=corrupt(original.kernel.body, "extra"))
        propose_dense_array_retention(bad, ADAPTER, LAW)


def wide_setup():
    from merlin.compile.scheduling.tile_load_coalescing import (
        TileLoadAddressLaw,
        TileLoadMapping,
        _linear,
        propose_tile_load_coalescing,
    )

    p, r, s, b, a, original = setup((8, 8, 8), (2, 2, 2))
    laws = {role: TileLoadAddressLaw(2, 2, 8, 1, pitch, 2, "independent-wide-array-test")
            for role, pitch in (("lhs", p.k), ("weight", p.n))}

    def mapping(call):
        if call.instr not in ("lhs_load", "weight_load"):
            return None
        op = project(call)
        view = s.lhs if op.role == "lhs" else s.rhs
        law = laws[op.role]
        return TileLoadMapping(op.role, view.backing, op.dram_row * law.source_pitch_bytes + op.dram_col,
                               op.row, op.rows, op.cols, law)

    def wider(call, columns):
        return replace(call, args=tuple((key, Const(columns) if key == "cols" else value)
                                       for key, value in call.args))

    def band(call, decoded):
        offset, terms = _linear(decoded.source_byte)
        base, base_terms = _linear(mapping(call).source_byte)
        assert terms == base_terms
        columns = offset - base
        return replace(call, args=tuple((key, decoded.local_row if key == "dst" else decoded.columns if key == "cols"
                                        else value + columns if key == "src_col"
                                        else value + columns // decoded.law.tile_columns if key == "b" else value)
                                       for key, value in call.args))

    wide, changes = propose_tile_load_coalescing(original.kernel, project=mapping, encode=wider, max_tiles=4)
    assert changes["changes"]
    proposal = propose_dense_array_retention(wide, ADAPTER, LAW)
    return p, r, s, b, a, original, proposal, mapping, band


def test_final_wide_bands_and_array_retention_compose_without_emitting_either_checking_projection():
    p, r, s, b, a, original, proposal, mapping, band = wide_setup()
    report = verify_dense_array_tile_loads(proposal, p, b, r, s, ADAPTER, a,
                                          project=mapping, encode_band=band, static_budget=256)
    projection = report["tile_load_projection"]
    assert projection["physical_load_commands"] < projection["expanded_logical_tile_bands"]
    assert report["actual_kernel_sha256"] == proposal.kernel.digest()
    assert report["expanded_actual_kernel_sha256"] != proposal.kernel.digest()
    assert report["array_retention"]["retained_pairs"] > 0
    assert execute(proposal.kernel, p) == execute(original.kernel, p)
    from merlin.compile.scheduling.scf_commands import materialize_scf_commands

    materialize_scf_commands(proposal.kernel).verify()
    with pytest.raises(ValueError):
        # A wide physical command cannot enter the primitive-only checker.
        verify_dense_array_retention(proposal, p, b, r, s, ADAPTER, a, static_budget=256)


def test_wide_array_checks_every_second_band_pitch_metadata_and_missing_mapping():
    p, r, s, b, a, _, proposal, mapping, band = wide_setup()

    def bad_band(call, decoded):
        result = band(call, decoded)
        if decoded.local_row != mapping(call).local_row:
            result = replace(result, args=tuple((key, value + 1 if key == "dst" else value)
                                               for key, value in result.args))
        return result

    def bad_pitch(call):
        decoded = mapping(call)
        return None if decoded is None else replace(decoded, law=replace(decoded.law, source_pitch_bytes=p.k + 1))

    for decode, encode_band, message in ((mapping, bad_band, "changes the final wide command mapping"),
                                         (bad_pitch, band, "source layout"),
                                         (lambda call: None, band, "every actual panel load")):
        with pytest.raises(ValueError, match=message):
            verify_dense_array_tile_loads(proposal, p, b, r, s, ADAPTER, a,
                                          project=decode, encode_band=encode_band, static_budget=256)

    def effects(body):
        return tuple(replace(call, body=effects(call.body)) if type(call) is Loop else
                     replace(call, unit="unmodeled") if call.instr == "weight_load" else call for call in body)

    changed = propose_dense_array_retention(replace(proposal.checking_kernel, body=effects(
        proposal.checking_kernel.body)), ADAPTER, LAW)
    with pytest.raises(ValueError, match="plain exact"):
        verify_dense_array_tile_loads(changed, p, b, r, s, ADAPTER, a,
                                      project=mapping, encode_band=band, static_budget=256)
