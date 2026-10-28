"""Library reconstruction must change actual checked schedules and replay them."""
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import DenseLoopAdapter, DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.loop_schedule import LoopTransform
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.kernels.reconstruction import (
    MechanismObservation,
    goals_from_cca,
    make_dense_reconstruction_adapter,
    reconstruct,
    replay,
)
from merlin.sched.ir import Call, as_expr, evaluate, instances
from merlin.transforms.dense_planning import DenseProblem, DenseResources
from merlin.transforms.pass_exploration import EmittedProgram


def encode(op):
    if isinstance(op, Load):
        names = ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step")
        values = (*op.block, op.dram_row, op.dram_col, op.rows, op.cols, op.row, op.step)
        name = op.role + "_load"
    elif isinstance(op, Preload):
        names = ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth")
        values = (op.weight_row, *op.weight_block, op.accumulator_row, op.accumulate, op.rows, op.cols, op.weight_rows)
        name = "preload"
    elif isinstance(op, Compute):
        names = ("lhs", "m", "k", "rows", "depth")
        values = (op.input_row, *op.input_block, op.rows, op.cols)
        name = "compute"
    elif isinstance(op, Store):
        names = ("row", "col", "rows", "cols", "acc")
        values = (op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row)
        name = "store"
    else:
        raise ValueError("unsupported neutral instruction")
    return Call(name, tuple(zip(names, map(as_expr, values))))


def project(call):
    a = dict(call.args)
    if call.instr in ("lhs_load", "weight_load"):
        return Load(call.instr.removesuffix("_load"), (a["a"], a["b"]),
                    a["src_row"], a["src_col"], a["rows"], a["cols"], a["dst"], a["step"])
    if call.instr == "preload":
        return Preload(a["weight"], (a["k"], a["n"]), a["acc"], a["continue"],
                       a["rows"], a["cols"], a["depth"])
    if call.instr == "compute":
        return Compute(a["lhs"], (a["m"], a["k"]), a["rows"], a["depth"], True)
    if call.instr == "store":
        return Store(a["row"], a["col"], a["rows"], a["cols"], a["acc"])
    raise ValueError("unsupported neutral instruction")


def source(shape):
    m, k, n = shape
    return parse_mlir_text(f'''module {{
      func.func @dense(%a: tensor<{m}x{k}xi8>, %b: tensor<{k}x{n}xi8>) -> tensor<{m}x{n}xi32> {{
        %zero = arith.constant 0 : i32
        %empty = tensor.empty() : tensor<{m}x{n}xi32>
        %fill = linalg.fill ins(%zero : i32) outs(%empty : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
        %result = linalg.generic {{indexing_maps = [affine_map<(m,n,k)->(m,k)>,
          affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
          iterator_types = ["parallel", "parallel", "reduction"]}}
          ins(%a, %b : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>) outs(%fill : tensor<{m}x{n}xi32>) {{
        ^bb0(%x: i8, %y: i8, %sum: i32):
          %u = arith.extsi %x : i8 to i32
          %v = arith.extsi %y : i8 to i32
          %product = arith.muli %u, %v : i32
          %next = arith.addi %sum, %product : i32
          linalg.yield %next : i32
        }} -> tensor<{m}x{n}xi32>
        func.return %result : tensor<{m}x{n}xi32>
      }}
    }}''')


def adapter(shape=(4, 6, 8), **kwargs):
    p = DenseProblem(*shape, 2, 3, 4, 1, 1, 4)
    r = DenseResources(512, 512, 2, 2, 2, partial_output_tiles=True, partial_output_blocks=True)
    s = DenseLoopStorage(DenseMemoryView("left", "left_store", 3, 0, 32),
                         DenseMemoryView("right", "right_store", 4, 0, 32),
                         DenseMemoryView("sum", "sum_store", 16, 0, 32))
    target = DenseLoopAdapter(encode, project, "neutral-test-config", "ordered-modular-test", True, True, 64)
    return make_dense_reconstruction_adapter(source(shape), problem=p, resources=r, storage=s,
        target_adapter=target, block_tiles=kwargs.pop("block_tiles", ((1, 1, 1), (2, 2, 2))),
        external_buffers_disjoint=True, verification_timeout_ms=20000, **kwargs)


def execute(kernel, shape):
    m, k, n = shape
    lhs = (np.arange(m * k) % 7 - 3).reshape(m, k)
    rhs = (np.arange(k * n) % 5 - 2).reshape(k, n)
    memories = {"lhs": {}, "weight": {}, "acc": {}}
    output = np.full((m, n), -10000)
    preload = None
    for call, env in instances(kernel):
        concrete = replace(call, args=tuple((key, evaluate(value, env)) for key, value in call.args))
        operation = project(concrete)
        if isinstance(operation, Load):
            src = lhs if operation.role == "lhs" else rhs
            for i in range(operation.rows):
                for j in range(operation.cols):
                    memories[operation.role][operation.row + i, j] = int(
                        src[operation.dram_row + i, operation.dram_col + j])
        elif isinstance(operation, Preload):
            preload = operation
        elif isinstance(operation, Compute):
            for i in range(operation.rows):
                for j in range(preload.cols):
                    slot = (preload.accumulator_row + i, j)
                    value = memories["acc"][slot] if preload.accumulate else 0
                    for depth in range(operation.cols):
                        value += (memories["lhs"][operation.input_row + i, depth]
                                  * memories["weight"][preload.weight_row + depth, j])
                    memories["acc"][slot] = value
        else:
            for i in range(operation.rows):
                for j in range(operation.cols):
                    output[operation.dram_row + i, operation.dram_col + j] = memories["acc"][
                        operation.accumulator_row + i, j]
    np.testing.assert_array_equal(output, lhs @ rhs)


@pytest.mark.parametrize("shape", [(4, 6, 8), (5, 7, 9), (1, 7, 9)])
def test_dialect_plans_reconstruct_actual_reuse_and_exact_tailed_arithmetic(shape):
    result = reconstruct(adapter(shape), ("reuse.lhs", "reuse.rhs", "schedule.consumed"))
    assert len(result.paths) == 4
    emitted = [p for p in result.paths if p.candidate.emission is not None]
    assert emitted
    for path in emitted:
        execute(path.candidate.evaluation.result.artifact.kernel, shape)
        assert any('"schedule.plan"' in row["text"] for row in path.intermediates)
        assert all(hashlib.sha256(row["text"].encode()).hexdigest() == row["sha256"] for row in path.intermediates)
    if shape[0] > 1:
        assert any(p.satisfied for p in result.paths)
    if shape[0] > 1:
        assert len({p.candidate.emission.data for p in emitted}) > 1
    else:
        assert any(p.candidate.evaluation.status == "failed" for p in result.paths)
    summary = json.loads(json.dumps(result.summary()))
    assert summary["backward_requirements"][0]["providers"]
    assert "numerical execution" in summary["qualification"]


def test_composed_real_loop_transforms_survive_dialect_and_scf_consumer():
    from merlin.compile.scheduling.scf_commands import materialize_scf_commands
    selected = adapter((8, 6, 8), loop_transform_sets=((), (LoopTransform("unroll", ("outer_m",), factor=2),)))
    result = reconstruct(selected, ("schedule.loop_transformed",), max_candidates=8)
    successful = [p for p in result.paths if p.satisfied]
    assert successful
    for path in successful:
        kernel = path.candidate.evaluation.result.artifact.kernel
        materialize_scf_commands(kernel).verify()
        assert any("schedule.loop_transform" in row["text"] for row in path.intermediates)
        execute(kernel, (8, 6, 8))


def test_exact_path_replays_and_detects_actual_emission_drift():
    selected = adapter()
    path = reconstruct(selected, ("reuse.lhs",)).paths[-1]
    record = json.loads(json.dumps(path.summary()["replay"]))
    actual = replay(selected, record, goals=("reuse.lhs",))
    assert actual.paths[0].candidate.emission.data == path.candidate.emission.data
    repeated = replay(selected, actual.paths[0].summary()["replay"])
    assert repeated.paths[0].candidate.emission.data == path.candidate.emission.data
    ordinary = selected.emission
    def changed(result):
        emitted = ordinary(result)
        return replace(emitted, data=emitted.data + b"\n")
    with pytest.raises(ValueError, match="program_sha256"):
        replay(replace(selected, emission=changed), record)


def test_no_path_or_unknown_goal_does_not_assert_a_missing_optimization():
    result = reconstruct(adapter(), ("unknown.mechanism",), max_candidates=1)
    assert not result.paths[0].satisfied
    kinds = {gap.kind for gap in result.gaps}
    assert kinds == {"bounded_search_exhaustion", "unbound_library_requirement"}
    assert result.search["truncated"]
    assert "absence from the library is not proved" in result.gaps[-1].reason
    # A declared provider alone never establishes a realized optimization.
    empty_observer = replace(adapter(), observe=lambda _: MechanismObservation())
    assert not any(p.satisfied for p in reconstruct(empty_observer, ("reuse.lhs",)).paths)


def test_additional_live_source_computation_is_a_machine_actionable_refusal():
    selected = adapter()
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import i32
    function = selected.source.body.block.first_op
    function.body.block.insert_op_before(arith.ConstantOp.from_int_and_width(17, i32), function.body.block.last_op)
    result = reconstruct(selected, ("schedule.consumed",), max_candidates=1)
    assert result.paths[0].candidate.emission is None
    assert result.gaps[0].kind == "forward_execution_failure"
    assert "other live source operations" in result.gaps[0].reason


def test_binding_unknown_source_semantics_and_witnessless_claims_refuse():
    selected = adapter()
    from xdsl.dialects import arith
    operation = next(op for op in selected.source.walk() if type(op) is arith.ConstantOp)
    from xdsl.dialects.builtin import IntegerAttr, i32
    operation.properties["value"] = IntegerAttr(1, i32)
    result = reconstruct(selected, ("schedule.consumed",), max_candidates=1)
    assert result.paths[0].candidate.emission is None
    with pytest.raises(ValueError, match="actual witnesses"):
        MechanismObservation(("reuse.lhs",))


def test_wrong_schedule_emission_identity_propagates_as_infrastructure_error():
    selected = adapter()
    ordinary = selected.emission
    def bad_emission(result):
        emitted = ordinary(result)
        return EmittedProgram(emitted.data, emitted.format, "0" * 64)
    with pytest.raises(ValueError, match="actual checked schedule"):
        reconstruct(replace(selected, emission=bad_emission), max_candidates=1)


def test_existing_cca_axes_drive_library_goals_and_compare_actual_output():
    from merlin.kernels.cca import CCA, ComputeFacet, DispatchFacet, MemoryFacet
    reference = CCA(op="matmul", backend=[],
        compute=ComputeFacet(register_block=(4, 8), accumulator_resident=True),
        memory=MemoryFacet(onchip_resident="both", panel_reuse=True),
        provenance={"axis_evidence": {"memory.onchip_resident": {
            "status": "observed", "evidence": ("pinned-reference-command-stream",)}}})
    plan = goals_from_cca(reference)
    assert {"reuse.lhs", "reuse.rhs", "compute.register_block", "compute.accumulator_resident"} <= set(plan["goals"])
    assert any(row["status"] == "observed" for row in plan["relations"])
    result = reconstruct(adapter(), reference_cca=reference)
    matched = [path for path in result.paths if path.satisfied]
    assert matched
    for path in matched:
        actual = path.observation.cca
        assert actual.memory.onchip_resident == "both"
        assert actual.compute.accumulator_resident is True
        assert path.summary()["cca_comparison"]["divergences"] == ()
        assert actual.provenance["machine_emission"] == "not_established"
    assert result.summary()["goal_relations"]
    # An unsupported existing lever remains visible to automatic experiments.
    reference.dispatch = DispatchFacet(loop_offloaded=True)
    expanded = reconstruct(adapter(), reference_cca=reference, max_candidates=1)
    assert any(gap.goals == ("dispatch.loop_offloaded",) for gap in expanded.gaps)
    assert not any(path.satisfied for path in expanded.paths)


def test_unknown_cca_evidence_and_requested_metadata_never_establish_mechanisms():
    from merlin.kernels.cca import CCA, MemoryFacet
    reference = CCA(op="matmul", backend=[], memory=MemoryFacet(onchip_resident="both"),
                    provenance={"axis_evidence": {"memory.onchip_resident": {"status": "unknown"}}})
    assert goals_from_cca(reference)["goals"] == ()
    # A callback can edit diagnostics, but observation reruns the actual source
    # and lifetime checker rather than trusting that mutable report dictionary.
    selected = adapter()
    ordinary = selected.emission
    def corrupt_report(result):
        emitted = ordinary(result)
        result.artifact.realization["lhs_loads"] = 0
        result.artifact.realization["rhs_loads"] = 0
        return emitted
    result = reconstruct(replace(selected, emission=corrupt_report), ("reuse.lhs",), max_candidates=1)
    assert not result.paths[0].satisfied
    assert result.paths[0].observation.cca.memory.onchip_resident == "none"


def test_library_split_and_reorder_composition_roundtrips_and_emits_same_arithmetic():
    from merlin.compile.scheduling.compact_dense import emit_compact_dense
    from merlin.compile.scheduling.loop_schedule import tile_loop_nest_transforms
    from merlin.compile.scheduling.scf_commands import materialize_scf_commands
    from merlin.transforms.dense_planning import dense_blocks
    p = DenseProblem(8, 6, 16, 2, 3, 4, 1, 1, 4)
    r = DenseResources(512, 512, 1, 1, 1, partial_output_tiles=True, partial_output_blocks=True)
    storage = DenseLoopStorage(DenseMemoryView("left", "left_store", 3, 0, 32),
                               DenseMemoryView("right", "right_store", 4, 0, 32),
                               DenseMemoryView("sum", "sum_store", 16, 0, 32))
    target = DenseLoopAdapter(encode, project, "neutral-test-config", "ordered-modular-test", True, True, 64)
    raw = emit_compact_dense(p, dense_blocks(p, r)[0], storage, target,
                             name="reconstructed", dtypes=("i8", "i8", "i32"))
    requests = tile_loop_nest_transforms(raw, ("outer_m",), factors=(2, 2), outer_order=(1, 0))
    assert {request.action for request in requests} == {"split", "reorder"}
    selected = make_dense_reconstruction_adapter(source((8, 6, 16)), problem=p, resources=r,
        storage=storage, target_adapter=target, external_buffers_disjoint=True,
        loop_transform_sets=((), requests), verification_timeout_ms=20000)
    result = reconstruct(selected, ("schedule.loop_transformed",))
    successful = [path for path in result.paths if path.satisfied]
    assert successful
    for path in successful:
        actual = path.candidate.evaluation.result.artifact.kernel
        materialize_scf_commands(actual).verify()
        assert actual.digest() != raw.digest()
        execute(actual, (8, 6, 16))


def test_operand_byte_accounting_is_recovered_from_actual_tailed_instructions():
    from merlin.compile.scheduling.block_schedule import Load
    selected = adapter((5, 7, 9))
    result = reconstruct(selected)
    for path in result.paths:
        if path.candidate.emission is None:
            continue
        compiled = path.candidate.evaluation.result.artifact
        requested = {"lhs": 0, "weight": 0}
        for call, env in instances(compiled.kernel):
            concrete = replace(call, args=tuple((key, evaluate(value, env)) for key, value in call.args))
            operation = project(concrete)
            if type(operation) is Load:
                requested[operation.role] += operation.rows * operation.cols
        report = compiled.realization
        assert report["requested_lhs_bytes"] == requested["lhs"]
        assert report["requested_rhs_bytes"] == requested["weight"]
        assert report["requested_load_bytes"] == sum(requested.values())


def test_full_operand_retention_uses_checked_library_motion_and_actual_bytes():
    from merlin.compile.scheduling.compact_dense import DenseLoopAdapter, DenseLoopStorage, DenseMemoryView
    problem = DenseProblem(4, 6, 32, 2, 3, 4, 1, 1, 4)
    resources = DenseResources(512, 512, 2, 2, 2, partial_output_tiles=True, partial_output_blocks=True)
    storage = DenseLoopStorage(DenseMemoryView("left", "left_store", 3, 0, 32),
                               DenseMemoryView("right", "right_store", 4, 0, 32),
                               DenseMemoryView("sum", "sum_store", 16, 0, 32))
    target = DenseLoopAdapter(encode, project, "neutral-test-config", "ordered-modular-test", True, True, 64)
    selected = make_dense_reconstruction_adapter(source((4, 6, 32)), problem=problem, resources=resources,
        storage=storage, target_adapter=target, external_buffers_disjoint=True,
        block_tiles=((2, 2, 1),), retained_operands=("none", "lhs"), verification_timeout_ms=20000)
    result = reconstruct(selected, ("memory.lhs_retained_full_reduction",))
    matched = [path for path in result.paths if path.satisfied]
    assert matched
    assert any(not path.satisfied for path in result.paths)
    for path in matched:
        compiled = path.candidate.evaluation.result.artifact
        assert compiled.realization["requested_lhs_bytes"] == 4 * 6
        assert '"hoist"' in compiled.realization["schedule_ir"]["text"]
        execute(compiled.kernel, (4, 6, 32))
