"""Actual grouped memory effects, source arithmetic, schedule consumption and refusals."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, encode, project

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_contracts import DenseIntegerReadout
from merlin.compile.scheduling.dense_panel_epochs import EPOCH_FLAGS, DensePanelEpochLaw
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN, DenseRetainedReductionAdmission
from merlin.compile.scheduling.dense_shared_rhs import DenseRHSReuseLaw, schedule_dense_shared_rhs
from merlin.compile.scheduling.dense_shared_rhs_check import verify_dense_shared_rhs
from merlin.compile.scheduling.schedule_ir import parse_dense_schedule_plan
from merlin.llvmlower.integer_readout import emit_exact_scalar_ir
from merlin.sched.ir import Const, Loop, Ptr, as_expr, evaluate, instances
from merlin.transforms.dense_planning import DenseProblem, DenseResources


def fixture(tile=2, *, quantized=False):
    problems = tuple(
        DenseProblem(m, 2 * tile + 1, 3 * tile + 1, tile, tile, tile, 1, 1, 4) for m in (tile + 1, 2 * tile + 1)
    )
    kt, nt = problems[0].tiles[1:]
    rhs = DenseMemoryView("rhs", "operand", tile, 16 * tile, kt * nt * tile)
    lhs = DenseMemoryView("lhs", "operand", tile, 0, 2 * kt * tile)
    acc = DenseMemoryView("acc", "accumulator", 4 * tile, 0, 4 * tile)
    storages = tuple(DenseLoopStorage(lhs, rhs, acc) for _ in problems)
    resources = DenseResources(
        (lhs.capacity + rhs.capacity) * tile,
        acc.capacity * acc.unit_bytes,
        2,
        2,
        kt,
        partial_output_tiles=True,
        partial_output_blocks=True,
    )
    adapter = replace(ADAPTER, qualification_identity="independent-neutral-group-byte-machine-v1")
    law = DenseRHSReuseLaw(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        adapter.qualification_identity,
        "neutral-omitted-load-observation-v1",
        rhs_unchanged_by_member_operations=True,
        independent_member_interleaving=True,
        suppressed_loads_no_faults=True,
        suppressed_loads_no_extra_observations=True,
    )
    readouts = tuple(
        DenseIntegerReadout(emit_exact_scalar_ir([1 / 128])[0], "i8", "store_i8", ()) if quantized else None
        for _ in problems
    )

    def encoder(member, op):
        call = encode(op)
        p = problems[0 if member is None else member]
        if type(op) is Load:
            tensor, stride, width = ("rhs", p.n, p.rhs_bytes) if member is None else (f"lhs{member}", p.k, p.lhs_bytes)
            pointer = Ptr(tensor, (as_expr(op.dram_row) * stride + op.dram_col) * width)
        elif type(op) is Store:
            pointer = Ptr(f"output{member}", (as_expr(op.dram_row) * p.n + op.dram_col) * (1 if quantized else 4))
            if op.readout is not None:
                call = replace(call, instr="store_i8")
        else:
            pointer = None
        args = call.args if pointer is None else (*call.args, ("buffer", pointer))
        return replace(call, args=(*args, ("member", Const(-1 if member is None else member))))

    def projector(call):
        a = dict(call.args)
        selector = a["member"]
        if type(selector) is not Const:
            raise ValueError("member selector must be an actual constant")
        member = None if selector.value == -1 else selector.value
        op = project(replace(call, instr="store") if call.instr == "store_i8" else call)
        if call.instr == "store_i8":
            op = replace(op, readout=readouts[member].identity)
        return member, op

    admission = DenseRetainedReductionAdmission(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        "independent-neutral-ordered-epoch-v1",
        10000,
        10000,
        4,
        4,
    )
    epochs = DensePanelEpochLaw(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        "independent-neutral-ordering-v1",
        **dict.fromkeys(EPOCH_FLAGS, True),
    )
    return problems, resources, storages, adapter, law, encoder, projector, admission, epochs, readouts


def schedule(model, *, asynchronous=True, **kwargs):
    p, r, s, a, law, encoder, projector, admission, epochs, readouts = model
    return schedule_dense_shared_rhs(
        p,
        r,
        s,
        a,
        encode_member=encoder,
        project_member=projector,
        retention_law=law,
        source_identity="a" * 64,
        name="group",
        dtypes=("i8", "i8", "i32"),
        static_budget=256,
        external_buffers_disjoint=True,
        epoch_admission=admission,
        epoch_law=epochs if asynchronous else replace(epochs, local_write_after_read=False),
        output_readouts=readouts,
        readout_qualification_identities=tuple("neutral-readout-v1" if x else None for x in readouts),
        verification_timeout_ms=30000,
        **kwargs,
    )


def run(result, model):
    problems, _, storages, _, _, _, projector, _, _, readouts = model
    tensors = {"rhs": [[(k * 11 + n * 7) % 127 - 63 for n in range(problems[0].n)] for k in range(problems[0].k)]}
    tensors.update(
        {
            f"lhs{i}": [[(m * 3 + k * 5 + i * 17) % 127 - 63 for k in range(p.k)] for m in range(p.m)]
            for i, p in enumerate(problems)
        }
    )
    output = [[[None] * p.n for _ in range(p.m)] for p in problems]
    memory, pending = {}, []
    descriptor = None

    def execute(member, op, pre=None):
        p, s = problems[0 if member is None else member], storages[0 if member is None else member]

        def key(view, row, col, width):
            return view.backing, row * view.unit_bytes + col * width

        if type(op) is Load:
            tensor, view, width = (
                ("rhs", s.rhs, p.rhs_bytes) if member is None else (f"lhs{member}", s.lhs, p.lhs_bytes)
            )
            for i in range(op.rows):
                for j in range(op.cols):
                    memory[key(view, op.row + i, j, width)] = tensors[tensor][op.dram_row + i][op.dram_col + j]
        elif type(op) is Compute:
            for i in range(op.rows):
                for j in range(pre.cols):
                    dest = key(s.accumulator, pre.accumulator_row + i, j, 4)
                    total = memory[dest] if pre.accumulate else 0
                    for k in range(op.cols):
                        total += (
                            memory[key(s.lhs, op.input_row + i, k, p.lhs_bytes)]
                            * memory[key(s.rhs, pre.weight_row + k, j, p.rhs_bytes)]
                        )
                        total = (total + 2**31) % 2**32 - 2**31
                    memory[dest] = total
        else:
            for i in range(op.rows):
                for j in range(op.cols):
                    value = memory[key(s.accumulator, op.accumulator_row + i, j, 4)]
                    output[member][op.dram_row + i][op.dram_col + j] = (
                        max(-128, min(127, round(value / 128))) if readouts[member] else value
                    )

    for call, env in instances(result.kernel):
        if call.instr == DRAIN:
            # Disjoint loads may complete out of issue order; reductions retain
            # the separately qualified accumulator order. No overlap is assumed.
            jobs = list(reversed(pending)) if pending and type(pending[0][1]) is Load else pending
            for args in jobs:
                execute(*args)
            pending = []
            continue
        actual = replace(
            call,
            args=tuple(
                (name, Ptr(v.tensor, Const(evaluate(v.offset, env))) if type(v) is Ptr else Const(evaluate(v, env)))
                for name, v in call.args
            ),
        )
        member, op = projector(actual)

        def scalar(v):
            if type(v) in (int, bool) or v is None or type(v) is str:
                return v
            if type(v) is tuple:
                return tuple(scalar(x) for x in v)
            return evaluate(v, {})

        op = replace(op, **{key: scalar(value) for key, value in vars(op).items()})
        if type(op) is Preload:
            descriptor = member, op
        elif type(op) is Compute:
            assert descriptor[0] == member
            pending.append((member, op, descriptor[1]))
        else:
            pending.append((member, op))
    assert not pending
    for member, p in enumerate(problems):
        for i in range(p.m):
            for j in range(p.n):
                value = sum(tensors[f"lhs{member}"][i][k] * tensors["rhs"][k][j] for k in range(p.k))
                value = (value + 2**31) % 2**32 - 2**31
                if readouts[member]:
                    value = max(-128, min(127, round(value / 128)))
                assert output[member][i][j] == value


@pytest.mark.parametrize("tile,quantized", [(2, False), (3, False), (2, True)])
def test_actual_rhs_lhs_reuse_tails_epoch_completion_and_readouts(tile, quantized):
    model = fixture(tile, quantized=quantized)
    result = schedule(model)
    assert result is not None
    run(result, model)
    p = model[0]
    assert result.realization["requested_rhs_bytes"] == p[0].k * p[0].n
    assert result.realization["requested_lhs_bytes"] == sum(x.m * x.k for x in p)
    assert result.realization["epoch_mode"] == "pure_phase_batches"
    assert result.realization["target_execution"] == "not_established"
    assert "schedule.group_dispatch" in result.plan_text and "schedule.panel_pipeline" in result.plan_text
    replay = schedule(model, schedule_plan=parse_dense_schedule_plan(result.plan_text))
    assert replay.kernel.digest() == result.kernel.digest()


def test_unknown_epoch_premise_keeps_safe_drains_without_losing_reuse():
    model = fixture()
    fallback = schedule(model, asynchronous=False)
    fast = schedule(model)
    run(fallback, model)
    assert fallback.realization["epoch_mode"] == "per_effect_drains"
    assert fallback.realization["dynamic_epoch_drains"] > fast.realization["dynamic_epoch_drains"]
    assert fallback.realization["requested_lhs_bytes"] == fast.realization["requested_lhs_bytes"]


@pytest.mark.parametrize("mutation", ["pointer", "overwrite", "member", "drain", "readout"])
def test_actual_effect_mutations_refused(mutation):
    model = fixture(quantized=True)
    result = schedule(model)

    def change(body):
        items = []
        for item in body:
            if type(item) is Loop:
                item = replace(item, body=change(item.body))
            elif item.instr == DRAIN and mutation == "drain":
                continue
            elif item.instr == "lhs_load" and mutation in ("pointer", "overwrite", "member"):
                a = dict(item.args)
                if mutation == "pointer":
                    a["buffer"] = Ptr("lhs1", a["buffer"].offset)
                elif mutation == "overwrite":
                    a["dst"] = Const(model[2][0].rhs.base)
                else:
                    a["member"] = Const(1)
                item = replace(item, args=tuple(a.items()))
            elif item.instr == "store_i8" and mutation == "readout":
                item = replace(item, instr="store")
            items.append(item)
        return tuple(items)

    p, r, s, a, law, _, projector, admission, epochs, readouts = model
    with pytest.raises(ValueError):
        verify_dense_shared_rhs(
            replace(result.kernel, body=change(result.kernel.body)),
            p,
            result.blocks,
            r,
            s,
            a,
            project_member=projector,
            retention_law=law,
            dtypes=("i8", "i8", "i32"),
            static_budget=256,
            external_buffers_disjoint=True,
            epoch_admission=admission,
            epoch_law=epochs,
            output_readouts=readouts,
            readout_qualification_identities=("neutral-readout-v1",) * 2,
            timeout_ms=30000,
        )


def test_plan_context_and_output_roster_cannot_be_dropped():
    model = fixture()
    result = schedule(model)
    text = result.plan_text.replace("a" * 64, "b" * 64)
    with pytest.raises(ValueError, match="binding"):
        schedule(model, schedule_plan=parse_dense_schedule_plan(text))
    with pytest.raises(ValueError, match="recheck"):
        replace(result, kernel=replace(result.kernel, args=result.kernel.args[:-1])).revalidate()


def test_explicit_library_block_decisions_change_the_emitted_program():
    model = fixture()
    default = schedule(model)
    alternate = schedule(model, block_tiles=tuple((1, p.tiles[1], 1) for p in model[0]))
    run(alternate, model)
    assert alternate.kernel.digest() != default.kernel.digest()
    assert alternate.realization["schedule_sha256"] != default.realization["schedule_sha256"]
    assert alternate.realization["requested_rhs_bytes"] == default.realization["requested_rhs_bytes"]
    assert alternate.realization["requested_lhs_bytes"] == default.realization["requested_lhs_bytes"]


def test_capacity_missing_retention_and_unqualified_readout_decline():
    model = fixture()
    p, r, s, a, law, encoder, projector, admission, epochs, readouts = model
    unknown = (*model[:4], replace(law, rhs_unchanged_by_member_operations=False), *model[5:])
    assert schedule(unknown) is None
    short = tuple(replace(x, rhs=replace(x.rhs, capacity=x.rhs.capacity - 1)) for x in s)
    assert schedule((p, r, short, a, law, encoder, projector, admission, epochs, readouts)) is None
    quantized = fixture(quantized=True)
    with pytest.raises(ValueError, match="qualification"):
        schedule_dense_shared_rhs(
            *quantized[:4],
            encode_member=quantized[5],
            project_member=quantized[6],
            retention_law=quantized[4],
            source_identity="a" * 64,
            name="group",
            dtypes=("i8", "i8", "i32"),
            static_budget=256,
            external_buffers_disjoint=True,
            output_readouts=quantized[-1],
        )


def test_lowering_consumes_actual_counted_schedule_and_explicit_completions():
    model = fixture()
    result = schedule(model)
    seen = []

    class Control:
        def loop(self, start, stop, step, callback):
            for i in range(start, stop, step):
                callback(i)

        def op(self, opcode, a, b):
            return a + b if opcode == "add" else a * b

        def cmp(self, predicate, a, b):
            return a == b

        def select(self, condition, a, b):
            return a if condition else b

    receipt = result.emit_index_program(
        Control(), lambda tag, args: seen.append((tag, args)), complete_epoch=lambda: seen.append((DRAIN, {}))
    )
    assert sum(tag == DRAIN for tag, _ in seen) == result.realization["dynamic_epoch_drains"]
    assert receipt["checked_kernel_sha256"] == result.kernel.digest()
    assert receipt["pointer_schemas"] and all(not isinstance(v, Ptr) for _, a in seen for v in a.values())
