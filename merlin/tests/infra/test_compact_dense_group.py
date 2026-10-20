"""Actual multi-buffer loop effects and independent ordered integer arithmetic."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, encode, project

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.compact_dense_group import DenseLHSReuseLaw, schedule_compact_dense_group
from merlin.compile.scheduling.compact_dense_group_check import verify_compact_dense_group
from merlin.sched.ir import Const, Loop, Ptr, as_expr, evaluate, instances
from merlin.transforms.dense_planning import DenseProblem, DenseResources


def fixture(width=1, *, same_shape=False):
    if width == 1:
        geometry, tile = (5, 5, 5, 9), (2, 3, 4)
        lhs = DenseMemoryView("lhs", "left", 7, 2, 12)
        rhs = [DenseMemoryView(f"rhs{i}", "right", 10, 3 + 12 * i, 12) for i in range(2)]
        acc = [DenseMemoryView(f"acc{i}", "sums", 24, 1 + 12 * i, 12) for i in range(2)]
    else:
        geometry, tile = (4, 3, 4, 7), (3, 2, 5)
        lhs = DenseMemoryView("lhs", "unified", 8, 2, 12)
        rhs = [DenseMemoryView(f"rhs{i}", "unified", 16, 16 + 12 * i, 12) for i in range(2)]
        acc = [DenseMemoryView(f"acc{i}", "unified", 28, 32 + 12 * i, 12) for i in range(2)]
    m, k, n0, n1 = geometry
    problems = tuple(DenseProblem(m, k, n, *tile, width, width, 4) for n in (n0, n0 if same_shape else n1))
    storages = tuple(DenseLoopStorage(lhs, r, a) for r, a in zip(rhs, acc, strict=True))
    resources = DenseResources(
        lhs.capacity * lhs.unit_bytes + sum(v.capacity * v.unit_bytes for v in rhs),
        sum(v.capacity * v.unit_bytes for v in acc),
        3,
        3,
        2,
        partial_output_tiles=True,
        partial_output_blocks=True,
    )
    adapter = replace(ADAPTER, qualification_identity="independent-neutral-group-byte-machine-v1")
    law = DenseLHSReuseLaw(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        adapter.qualification_identity,
        "independent-neutral-retention-and-interleaving-v1",
        lhs_unchanged_by_member_operations=True,
        independent_member_interleaving=True,
        suppressed_loads_no_faults=True,
        suppressed_loads_no_extra_observations=True,
    )

    def encoder(member, op):
        call = encode(op)
        if type(op) is Load:
            p = problems[0 if member is None else member]
            name, stride, bits = ("lhs", p.k, p.lhs_bytes) if member is None else (f"rhs{member}", p.n, p.rhs_bytes)
            pointer = Ptr(name, (as_expr(op.dram_row) * stride + op.dram_col) * bits)
        elif type(op) is Store:
            p = problems[member]
            pointer = Ptr(f"output{member}", (as_expr(op.dram_row) * p.n + op.dram_col) * p.accumulator_bytes)
        else:
            pointer = None
        args = call.args if pointer is None else (*call.args, ("buffer", pointer))
        # Actual encoded member participates in compute descriptor selection.
        return replace(call, args=(*args, ("member", Const(-1 if member is None else member))))

    def projector(call):
        operands = dict(call.args)
        member = operands["member"]
        if type(member) is not Const:
            raise ValueError("unknown actual member selector")
        actual = None if member.value == -1 else member.value
        op = project(call)
        if type(op) in (Load, Store):
            pointer = operands["buffer"]
            expected = "lhs" if actual is None else (f"rhs{actual}" if type(op) is Load else f"output{actual}")
            if type(pointer) is not Ptr or pointer.tensor != expected:
                raise ValueError("actual pointer/member mismatch")
        return actual, op

    return problems, resources, storages, adapter, law, encoder, projector


def schedule(model, **kwargs):
    p, r, s, a, law, encoder, projector = model
    return schedule_compact_dense_group(
        p,
        r,
        s,
        a,
        encode_member=encoder,
        project_member=projector,
        retention_law=law,
        name="group",
        dtypes=("i" + str(8 * p[0].lhs_bytes), "i" + str(8 * p[0].rhs_bytes), "i32"),
        static_budget=160,
        external_buffers_disjoint=True,
        verification_timeout_ms=30000,
        **kwargs,
    )


def run(kernel, model):
    p, _, storages, _, _, _, projector = model
    inputs = {"lhs": [[(i * 7 + j * 13) % 31 - 15 for j in range(p[0].k)] for i in range(p[0].m)]}
    for member, problem in enumerate(p):
        inputs[f"rhs{member}"] = [
            [(k * 11 + n * 3 + member * 5) % 29 - 14 for n in range(problem.n)] for k in range(problem.k)
        ]
    outputs = [[[None] * problem.n for _ in range(problem.m)] for problem in p]
    memory = {}
    pending = None
    lhs_bytes = 0
    for actual, env in instances(kernel):
        call = replace(
            actual,
            args=tuple(
                (
                    key,
                    Ptr(value.tensor, Const(evaluate(value.offset, env)))
                    if type(value) is Ptr
                    else Const(evaluate(value, env)),
                )
                for key, value in actual.args
            ),
        )
        member, op = projector(call)

        def value(x):
            return x if type(x) in (int, bool) else evaluate(x, {})

        storage = storages[0 if member is None else member]
        problem = p[0 if member is None else member]

        def key(view, row, col, width):
            return view.backing, value(row) * view.unit_bytes + col * width

        if type(op) is Load:
            view = storage.lhs if member is None else storage.rhs
            width = problem.lhs_bytes if member is None else problem.rhs_bytes
            tensor = dict(call.args)["buffer"].tensor
            for i in range(value(op.rows)):
                for j in range(value(op.cols)):
                    memory[key(view, value(op.row) + i, j, width)] = inputs[tensor][value(op.dram_row) + i][
                        value(op.dram_col) + j
                    ]
            if member is None:
                lhs_bytes += value(op.rows) * value(op.cols) * width
        elif type(op) is Preload:
            pending = member, op
        elif type(op) is Compute:
            assert pending[0] == member
            pre = pending[1]
            for i in range(value(op.rows)):
                for j in range(value(pre.cols)):
                    dest = key(storage.accumulator, value(pre.accumulator_row) + i, j, 4)
                    total = memory[dest] if value(pre.accumulate) else 0
                    for k in range(value(op.cols)):
                        total += (
                            memory[key(storage.lhs, value(op.input_row) + i, k, problem.lhs_bytes)]
                            * memory[key(storage.rhs, value(pre.weight_row) + k, j, problem.rhs_bytes)]
                        )
                        total = (total + (1 << 31)) % (1 << 32) - (1 << 31)
                    memory[dest] = total
        elif type(op) is Store:
            for i in range(value(op.rows)):
                for j in range(value(op.cols)):
                    outputs[member][value(op.dram_row) + i][value(op.dram_col) + j] = memory[
                        key(storage.accumulator, value(op.accumulator_row) + i, j, 4)
                    ]
    for member, problem in enumerate(p):
        expected = []
        for i in range(problem.m):
            row = []
            for j in range(problem.n):
                total = 0
                for k in range(problem.k):
                    total = (total + inputs["lhs"][i][k] * inputs[f"rhs{member}"][k][j] + (1 << 31)) % (1 << 32) - (
                        1 << 31
                    )
                row.append(total)
            expected.append(row)
        assert outputs[member] == expected
    return lhs_bytes


@pytest.mark.parametrize("width", [1, 2])
def test_group_retains_actual_lhs_all_outputs_and_one_physical_map(width):
    model = fixture(width)
    result = schedule(model)
    assert result is not None
    assert run(result.kernel, model) == model[0][0].m * model[0][0].k * width
    assert result.realization["requested_lhs_bytes"] == model[0][0].m * model[0][0].k * width
    assert result.output_shapes == tuple((p.m, p.n) for p in model[0])
    assert tuple(p.name for p in result.kernel.args) == ("lhs", "rhs0", "rhs1", "output0", "output1")
    result.revalidate()
    assert result.implementation_paths()


def change(body, predicate, mutation):
    return tuple(
        replace(s, body=change(s.body, predicate, mutation)) if type(s) is Loop else mutation(s) if predicate(s) else s
        for s in body
    )


@pytest.mark.parametrize(
    "kind", ["wrong_rhs", "wrong_output", "negative", "prefix", "overwrite", "async", "partial", "member"]
)
def test_actual_group_mutations_refuse(kind):
    model = fixture(same_shape=True)
    result = schedule(model)
    p, r, s, a, law, _, projector = model

    def mutate(call):
        args = dict(call.args)
        if kind in ("wrong_rhs", "wrong_output"):
            key = "buffer"
            args[key] = Ptr("rhs1" if kind == "wrong_rhs" else "output1", args[key].offset)
        elif kind == "negative":
            args["buffer"] = Ptr(args["buffer"].tensor, Const(-1))
        elif kind == "prefix":
            args["buffer"] = Ptr("prefix_rhs0", args["buffer"].offset)
        elif kind == "overwrite":
            args["dst"] = Const(s[0].lhs.base)
        elif kind == "async":
            return replace(call, produces="pending")
        elif kind == "partial":
            args["cols"] = Const(1)
        elif kind == "member":
            args["member"] = Const(1)
        return replace(call, args=tuple(args.items()))

    instruction = "store" if kind == "wrong_output" else "weight_load"
    body = change(
        result.kernel.body, lambda call: call.instr == instruction and dict(call.args)["member"] == Const(0), mutate
    )
    with pytest.raises(ValueError):
        verify_compact_dense_group(
            replace(result.kernel, body=body),
            p,
            r,
            s,
            a,
            project_member=projector,
            retention_law=law,
            dtypes=("i8", "i8", "i32"),
            static_budget=160,
            external_buffers_disjoint=True,
            timeout_ms=30000,
        )


def test_missing_law_full_k_capacity_and_unknown_completion_fall_back():
    model = fixture()
    assert schedule((*model[:4], None, *model[5:])) is None
    assert schedule((*model[:4], replace(model[4], suppressed_loads_no_faults=False), *model[5:])) is None
    assert schedule((model[0], replace(model[1], max_k_blocks=1), *model[2:])) is None
    assert schedule((*model[:3], replace(model[3], synchronous_completion=False), *model[4:])) is None


def test_original_inputs_and_code_stale_after_success_refuse():
    model = fixture()
    result = schedule(model)
    object.__setattr__(model[0][0], "n", 4)
    with pytest.raises(ValueError, match="inputs"):
        result.revalidate()


def selected_fixture(tmp_path):
    from test_dense_pointer_contracts import pointer_inputs

    def union(facts, software, dense):
        for name in ("right", "sums"):
            facts["capacities"][name] *= 2
        facts["word_bits"] = 648
        facts["offsets"] = [8 + 64 * i for i in range(10)]
        facts["padding_offsets"] = [8 + 64 * i for i in range(11)]
        facts["padding_widths"] = [64 * (10 - i) for i in range(11)]

        def ref(*path):
            return {"fact": ["facts", *path]}

        for ins in software["isa"]["instructions"]:
            ins["operands"].append(
                dict(name="member", unit="integer", type=dict(kind="integer", signed=True, bits=ref("arg_bits")))
            )
            args = ins["operands"]
            ins["encoding"] = [ins["encoding"][0]] + [
                dict(offset=ref("offsets", i), width=ref("arg_bits"), argument=a["name"]) for i, a in enumerate(args)
            ]
            if len(args) < 10:
                ins["encoding"].append(
                    dict(
                        offset=ref("padding_offsets", len(args)),
                        width=ref("padding_widths", len(args)),
                        fixed=ref("origin"),
                    )
                )

    selected, p, s, adapter = pointer_inputs(tmp_path, shape=(5, 5, 5), mutate=union)
    model = fixture()
    p2 = replace(p, n=9)
    storages = tuple(
        DenseLoopStorage(
            s.lhs,
            replace(s.rhs, name=f"rhs{i}", base=s.rhs.base + 12 * i),
            replace(s.accumulator, name=f"acc{i}", base=s.accumulator.base + 12 * i),
        )
        for i in range(2)
    )
    adapter = replace(adapter, encode=lambda op: replace(encode(op), args=(*encode(op).args, ("member", Const(0)))))
    law = replace(
        model[4],
        configuration_identity=adapter.configuration_identity,
        numerical_policy_identity=adapter.numerical_policy_identity,
        adapter_qualification_identity=adapter.qualification_identity,
    )
    actual = ((p, p2), None, storages, adapter, law, model[5], model[6])
    return selected, actual


def test_selected_v2_multi_buffer_boundary_is_consumed(tmp_path):
    selected, actual = selected_fixture(tmp_path)
    result = schedule(actual, selected_contracts=selected)
    assert result is not None
    assert run(result.kernel, actual) == 25
    assert result.realization["pipeline"]["selected_contracts"]["resolved_isa_identity"] == selected.isa.identity()


@pytest.mark.parametrize("target", ["source", "adapter", "law", "code"])
def test_encoder_cannot_rebase_original_inputs(target):
    model = fixture()
    original = model[5]

    if target == "code":

        def replacement(member, op):
            return model

        def corrupt_project(call):
            original.__code__ = replacement.__code__
            return model[6](call)

        with pytest.raises(ValueError, match="implementation"):
            schedule((*model[:6], corrupt_project))
        return

    def corrupt(member, op):
        if target == "source":
            object.__setattr__(model[0][0], "n", 99)
        elif target == "adapter":
            object.__setattr__(model[3], "qualification_identity", "changed")
        elif target == "law":
            object.__setattr__(model[4], "suppressed_loads_no_faults", False)
        return encode(op)

    with pytest.raises(ValueError, match="inputs|implementation"):
        schedule((*model[:5], corrupt, model[6]))


def test_search_budget_and_mismatched_source_do_not_invoke_encoder():
    model = fixture()
    count = []

    def unexpected(*args):
        count.append(args)
        raise AssertionError("must refuse before encoding")

    with pytest.raises(ValueError, match="budget"):
        schedule((model[0], replace(model[1], candidate_budget=1), *model[2:5], unexpected, model[6]))
    assert count == []
    problems = (model[0][0], replace(model[0][1], k=4))
    with pytest.raises(ValueError, match="common"):
        schedule((problems, *model[1:5], unexpected, model[6]))
    assert count == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("configuration_identity", "changed"),
        ("numerical_policy_identity", "changed"),
        ("qualification_identity", "changed"),
        ("synchronous_completion", False),
        ("exact_accumulator_continuation", False),
        ("index_bits", 32),
    ],
)
def test_direct_checker_rejects_projection_rebasing_original_adapter(field, value):
    model = fixture()
    result = schedule(model)
    p, r, s, a, law, _, projector = model

    def corrupt(call):
        object.__setattr__(a, field, value)
        return projector(call)

    with pytest.raises(ValueError, match="adapter or projection changed"):
        verify_compact_dense_group(
            result.kernel,
            p,
            r,
            s,
            a,
            project_member=corrupt,
            retention_law=law,
            dtypes=("i8", "i8", "i32"),
            static_budget=160,
            external_buffers_disjoint=True,
            timeout_ms=30000,
        )


@pytest.mark.parametrize("field", ["kernel", "context", "index", "choice", "paths"])
def test_checked_group_result_cannot_retarget_emission(field):
    model = fixture()
    result = schedule(model)
    if field == "kernel":
        result = replace(result, kernel=replace(result.kernel, name="changed"))
    elif field == "context":
        object.__setattr__(result.checked_emission.context, "hardware", "changed")
    elif field == "index":
        result = replace(result, index_bits=32)
    elif field == "choice":
        result = replace(result, choice=replace(result.choice, static_commands=1))
    else:
        result = replace(result, _paths=())
    with pytest.raises(ValueError, match="checked shared LHS"):
        result.revalidate()


def test_overlapping_member_windows_decline_before_encoding_but_actual_checker_refuses():
    model = fixture()
    result = schedule(model)
    p, r, storage, a, law, _, projector = model
    # Each separate contraction remains legal; only cross-member reuse is unavailable.
    overlap = (storage[0], replace(storage[1], rhs=replace(storage[1].rhs, base=storage[0].rhs.base)))
    count = []

    def unexpected(*args):
        count.append(args)
        raise AssertionError("overlap must decline before encoding")

    assert schedule((p, r, overlap, a, law, unexpected, projector)) is None
    assert count == []
    with pytest.raises(ValueError, match="overlap"):
        verify_compact_dense_group(
            result.kernel,
            p,
            r,
            overlap,
            a,
            project_member=projector,
            retention_law=law,
            dtypes=("i8", "i8", "i32"),
            static_budget=160,
            external_buffers_disjoint=True,
            timeout_ms=30000,
        )


@pytest.mark.parametrize("boundary", ["revalidate", "record"])
@pytest.mark.parametrize("direct", [False, True])
def test_selected_owner_mutation_is_rejected_before_next_provider(tmp_path, monkeypatch, boundary, direct):
    selected, model = selected_fixture(tmp_path)
    result = schedule(model, selected_contracts=selected) if direct else None
    adapter = model[3]
    expected = adapter.qualification_identity
    visits, delivered = {}, []
    owner_type = type(selected)
    original = getattr(owner_type, boundary)

    def owner(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        if self is selected and delivered:
            count = len(delivered)
            visits[count] = visits.get(count, 0) + 1
            # One original owner post-check has already seen this delivered
            # operation. Mutate at the next boundary before another provider
            # receives the source; detecting it afterward is too late.
            if visits[count] == 2:
                object.__setattr__(adapter, "qualification_identity", "changed-by-original-owner")
        return value

    monkeypatch.setattr(owner_type, boundary, owner)
    if direct:

        def observing(call):
            assert adapter.qualification_identity == expected, "projector received changed original context"
            delivered.append(call)
            return model[6](call)

        with pytest.raises(ValueError, match="adapter or projection changed"):
            resources = DenseResources(
                model[2][0].lhs.capacity * model[2][0].lhs.unit_bytes
                + sum(s.rhs.capacity * s.rhs.unit_bytes for s in model[2]),
                sum(s.accumulator.capacity * s.accumulator.unit_bytes for s in model[2]),
                3,
                3,
                2,
                partial_output_tiles=True,
                partial_output_blocks=True,
            )
            verify_compact_dense_group(
                result.kernel,
                model[0],
                resources,
                model[2],
                adapter,
                project_member=observing,
                retention_law=model[4],
                dtypes=("i8", "i8", "i32"),
                static_budget=160,
                external_buffers_disjoint=True,
                selected_contracts=selected,
                timeout_ms=30000,
            )
    else:

        def observing(member, op):
            assert adapter.qualification_identity == expected, "encoder received changed original context"
            delivered.append(op)
            return model[5](member, op)

        with pytest.raises(ValueError, match="inputs or callback implementation changed"):
            schedule((*model[:5], observing, model[6]), selected_contracts=selected)
    assert len(delivered) == 1


def test_combined_search_exhaustion_retains_legal_single_routes():
    from merlin.transforms.dense_planning import dense_blocks

    model = fixture()
    resources = replace(model[1], candidate_budget=9)
    for p, s in zip(model[0], model[2], strict=True):
        assert dense_blocks(p, resources, k_limit=lambda m, n, p=p, s=s: s.k_limit(p, resources, m, n))
    delivered = []

    def unexpected(*args):
        delivered.append(args)
        raise AssertionError("combined search must decline before encoding")

    assert schedule((model[0], resources, *model[2:5], unexpected, model[6])) is None
    assert delivered == []


@pytest.mark.parametrize("repeat", [False, True])
def test_repeated_single_assignments_decline_without_selected_owner_callbacks(tmp_path, monkeypatch, repeat):
    selected, model = selected_fixture(tmp_path)
    result = schedule(model, selected_contracts=selected)
    first, second = model[2]
    duplicate = first if repeat else replace(second, rhs=replace(second.rhs, name=first.rhs.name))
    storages = (first, duplicate)
    calls = []

    def unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unsupported group must decline before original selected callbacks")

    monkeypatch.setattr(type(selected), "revalidate", unexpected)
    monkeypatch.setattr(type(selected), "record", unexpected)
    assert schedule((*model[:2], storages, *model[3:]), selected_contracts=selected) is None
    assert calls == []
    resources = DenseResources(10000, 10000, 3, 3, 2, partial_output_tiles=True, partial_output_blocks=True)
    with pytest.raises(ValueError, match="unique assigned windows"):
        verify_compact_dense_group(
            result.kernel,
            model[0],
            resources,
            storages,
            model[3],
            project_member=model[6],
            retention_law=model[4],
            dtypes=("i8", "i8", "i32"),
            static_budget=160,
            external_buffers_disjoint=True,
            selected_contracts=selected,
        )
    assert calls == []
