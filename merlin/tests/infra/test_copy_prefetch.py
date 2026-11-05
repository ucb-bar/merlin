"""Qualified async issue ordering, real decoded effects and legal fallback.

The byte interpreter is neutral functional evidence, not target timing or native
execution qualification. Ordinary source/native coverage is tested separately.
"""

from dataclasses import replace

import pytest
from test_private_copy_elimination import command
from test_selected_program_contracts import inputs as program_inputs

from merlin.compile.scheduling import schedule_pipeline
from merlin.compile.scheduling.fusion_semantics import ElementwiseSemantics
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, compile_schedule
from merlin.compile.scheduling.temporary_storage import (
    TemporaryObject,
    verify_copy_prefetch,
)
from merlin.sched.ir import evaluate

QUALIFICATION = "independent-neutral-copy-integer-overlap-v1"
INCREMENT = b"""module {func.func @source(%x:i8)->i8 {
%one=arith.constant 1:i8
%r=arith.addi %x,%one:i8
func.return %r:i8
}}"""


def inputs(tmp_path, *, width=7, separate=True, word_format="<BBBQQQ", mutate=None):
    def declare(facts, software, program):
        facts.update(prefetch_inflight=1, prefetch_queue=1)
        storage = software["scheduling"]["storage"]
        storage["instructions"]["async_copy"]["exact_copy"] = True
        storage["prefetch"] = {
            "schema": "merlin.async_prefetch.v1",
            "max_in_flight": {"fact": ["facts", "prefetch_inflight"]},
            "queue_capacity": {"fact": ["facts", "prefetch_queue"]},
            "independence": "memory_disjoint",
            "pairs": [
                {"launch_instruction": "async_copy", "crossed_instruction": name} for name in ("copy", "increment")
            ],
        }
        if mutate:
            mutate(facts, software, storage)

    model = list(
        program_inputs(tmp_path, width=width, separate=separate, word_format=word_format, count=128, mutate=declare)
    )
    selected, kernel, _, adapter, *_ = model
    project = adapter.project

    def qualified(call):
        effect = project(call)
        if call.instr == "async_copy":
            return replace(effect, exact_copy=True)
        if call.instr == "increment":
            return replace(
                effect,
                elementwise=ElementwiseSemantics(
                    INCREMENT, evaluate(call.arg("size"), {}), ("increment",), ("little", "little")
                ),
            )
        return effect

    model[1] = replace(
        kernel,
        body=(
            command("copy", 0, 16, width, 2, 0),
            command("increment", 16, 16, width, 0, 0),
            command("copy", 16, 0, width, 0, 3),
            command("async_copy", width, 48, width, 2, 1),
            command("wait", 0, 0, width),
            command("increment", 48, 48, width, 1, 1),
            command("copy", 48, width, width, 1, 3),
        ),
    )
    model[2] = (TemporaryObject("current", "bytes", 16, width), TemporaryObject("next", "typed_view", 48, width))
    model[3] = replace(adapter, project=qualified)
    return model


def compile_case(model, *, distance=3, qualification=QUALIFICATION, **kwargs):
    selected, kernel, objects, adapter, encoding, limits, *_ = model
    return compile_schedule(
        kernel,
        objects,
        None,
        adapter,
        encoding,
        limits,
        selected_contracts=selected,
        private_ownership=True,
        prefetch_qualification_identity=qualification,
        budgets=ScheduleBudgets(prefetch_distance=distance, max_fusions=0, eliminate_copies=False),
        **kwargs,
    )


def execute(image, word, width, raw):
    """Decode real words independently; async bytes become visible at wait."""
    first, second = bytearray([99] * 192), bytearray([99] * 192)
    output = bytearray([91] * len(raw))
    memory = [first, second, bytearray(raw), output]
    pending, trace = None, []
    for offset in range(0, len(image), word.size):
        opcode, read, write, source, destination, size = word.unpack_from(image, offset)
        trace.append((opcode, read, write, source, destination, size))
        if opcode == 3:
            assert pending is not None and pending[-1] == size
            r, w, s, d, n = pending
            memory[w][d : d + n] = memory[r][s : s + n]
            pending = None
            continue
        assert source + size <= len(memory[read]) and destination + size <= len(memory[write])
        if opcode == 2:
            assert pending is None
            pending = read, write, source, destination, size
        elif opcode == 0:
            memory[write][destination : destination + size] = memory[read][source : source + size]
        else:
            assert opcode == 1
            memory[write][destination : destination + size] = bytes(
                (x + 1) % 256 for x in memory[read][source : source + size]
            )
    assert pending is None and memory[2] == raw
    assert first[:8] == bytes([99] * 8) and second[:16] == bytes([99] * 16)
    return bytes(output), trace


@pytest.mark.parametrize("width,word_format", [(7, "<BBBQQQ"), (13, ">BBBIII")])
def test_ordinary_selected_prefetch_changes_actual_order_and_preserves_complete_state(tmp_path, width, word_format):
    model = inputs(tmp_path, width=width, word_format=word_format)
    baseline, optimized = compile_case(model, distance=0), compile_case(model)
    assert baseline.report["context"]["library"] == optimized.report["context"]["library"]
    report = optimized.report["storage"]["prefetch"]
    assert report["status"] == "applied" and report["moves"][0]["distance"] == 3
    assert report["original_positions"] == [3, 0, 1, 2, 4, 5, 6]
    assert optimized.report["final_conformance"]["prefetch"]["actual_launches"][0]["actual_instruction_index"] == 0
    assert [p["name"] for p in optimized.report["passes"]] == [
        "semantic_admission",
        "prefetch",
        "storage",
        "completion",
        "emission",
        "final_program",
    ]
    states = []
    for result in (baseline, optimized):
        state = bytes([253] * width)
        for invocation in range(3):
            raw = bytes((i * 31 + invocation + 249) % 256 for i in range(width)) + state + bytes([71] * 8)
            output, trace = execute(result.program.segments[0], model[6], width, raw)
            expected = bytes((x + 1) % 256 for x in raw[: 2 * width]) + bytes([91] * 8)
            assert output == expected
            state = output[width : 2 * width]
            assert len(trace) == result.report["final_conformance"]["actual_commands"] == 7
            assert sum(row[-1] for row in trace if row[0] != 3) == 6 * width
        states.append(state)
    assert states[0] == states[1]
    assert len(baseline.program.segments[0]) == len(optimized.program.segments[0]) == 7 * model[6].size


@pytest.mark.parametrize("missing", ["qualification", "queue", "inflight", "pair", "alias", "scalar"])
def test_missing_overlap_or_physical_independence_retains_original_order(tmp_path, missing):
    def mutate(facts, software, storage):
        if missing == "queue":
            facts["prefetch_queue"] = "UNKNOWN"
        elif missing == "inflight":
            storage["prefetch"]["max_in_flight"] = {"fact": ["absent"]}
        elif missing == "pair":
            storage["prefetch"]["pairs"] = []

    model = inputs(tmp_path, separate=missing != "alias", mutate=mutate)
    if missing == "scalar":
        project = model[3].project
        model[3] = replace(
            model[3],
            project=lambda call: (
                replace(project(call), elementwise=None) if call.instr == "increment" else project(call)
            ),
        )
    result = compile_case(model, qualification=None if missing == "qualification" else QUALIFICATION)
    calls = model[4].decode_calls(result.program.segments[0])
    position = next(i for i, call in enumerate(calls) if call.instr == "async_copy")
    assert position == (2 if missing == "scalar" else 3)


def test_prefetch_allocation_failure_restores_original_order_and_replans(tmp_path, monkeypatch):
    model = inputs(tmp_path)
    allocate = schedule_pipeline.allocate_temporary_storage
    attempted = []

    def controlled(source, *args, **kwargs):
        attempted.append(source.body[0].instr)
        return None if source.body[0].instr == "async_copy" else allocate(source, *args, **kwargs)

    monkeypatch.setattr(schedule_pipeline, "allocate_temporary_storage", controlled)
    result = compile_case(model)
    assert attempted == ["async_copy", "copy"]
    assert result.report["storage"]["prefetch"]["status"] == "allocation_fallback"
    assert result.report["repair_rounds"] == 1
    assert model[4].decode_calls(result.program.segments[0])[0].instr == "copy"


@pytest.mark.parametrize("mutation", ["duplicate", "wait", "dependency", "pair"])
def test_actual_prefetch_candidate_must_preserve_membership_waits_and_qualified_pairs(tmp_path, mutation):
    model = inputs(tmp_path)
    selected, source, objects, adapter, *_ = model
    banks, _, external, checked = selected.storage.bind(
        selected.isa, adapter, banks=None, reservations=(), external_windows=None, private_ownership=True
    )
    admission = selected.storage.bind_prefetch(QUALIFICATION)
    order = (3, 0, 1, 2, 4, 5, 6)
    if mutation == "duplicate":
        order = (3, 0, 1, 2, 4, 5, 5)
    elif mutation == "wait":
        order = (4, 3, 0, 1, 2, 5, 6)
    elif mutation == "dependency":
        order = (3, 5, 0, 1, 2, 4, 6)
    else:
        admission = replace(admission, pairs=())
    candidate = replace(source, body=tuple(source.body[i] for i in order))
    with pytest.raises(ValueError, match="membership|unqualified|completion"):
        verify_copy_prefetch(
            source,
            candidate,
            order,
            objects,
            banks,
            checked,
            admission,
            distance=3,
            max_moves=8,
            external_windows=external,
            private_ownership=True,
        )


@pytest.mark.parametrize("mutation", ["literal", "unknown_field", "sync_launch", "wrong_effect"])
def test_prefetch_declaration_is_exact_and_fact_bound(tmp_path, mutation):
    def mutate(facts, software, storage):
        if mutation == "literal":
            storage["prefetch"]["queue_capacity"] = 16
        elif mutation == "unknown_field":
            storage["prefetch"]["estimated_cycles"] = 3
        elif mutation == "sync_launch":
            storage["prefetch"]["pairs"][0]["launch_instruction"] = "copy"
        else:
            storage["instructions"]["async_copy"]["exact_copy"] = False

    with pytest.raises(ValueError, match="storage|prefetch"):
        inputs(tmp_path, mutate=mutate)


def test_selected_prefetch_object_cannot_replace_original_overlap_declaration(tmp_path):
    model = inputs(tmp_path)
    object.__setattr__(model[0].storage.prefetch, "pairs", (("async_copy", "wait"),))
    with pytest.raises(ValueError, match="original software/RTL declaration"):
        compile_case(model)


def test_overlap_mutation_during_provider_encoding_refuses_result(tmp_path):
    model = inputs(tmp_path)
    encode = model[4].encode

    def corrupt(kernel):
        image = encode(kernel)
        object.__setattr__(model[0].storage.prefetch, "queue_capacity", 999)
        return image

    model[4] = replace(model[4], encode=corrupt)
    with pytest.raises(ValueError, match="prefetch overlap declaration changed"):
        compile_case(model)


def test_final_bytes_must_realize_the_selected_issue_order(tmp_path):
    model = inputs(tmp_path)
    encode = model[4].encode

    def old_order(kernel):
        assert kernel.body[0].instr == "async_copy"
        body = list(kernel.body)
        body.insert(3, body.pop(0))
        return encode(replace(kernel, body=tuple(body)))

    model[4] = replace(model[4], encode=old_order)
    with pytest.raises(ValueError, match="actual emitted.*effects differ"):
        compile_case(model)


def test_qualified_default_automatically_advances_one_operation(tmp_path):
    selected, kernel, objects, adapter, encoding, limits, *_ = inputs(tmp_path)
    result = compile_schedule(
        kernel,
        objects,
        None,
        adapter,
        encoding,
        limits,
        selected_contracts=selected,
        private_ownership=True,
        prefetch_qualification_identity=QUALIFICATION,
    )
    assert ScheduleBudgets().prefetch_distance == 1
    assert result.report["storage"]["prefetch"]["moves"][0]["distance"] == 1
    assert result.report["final_conformance"]["prefetch"]["actual_launches"][0]["actual_instruction_index"] == 2


def test_zero_move_budget_keeps_original_order(tmp_path):
    selected, kernel, objects, adapter, encoding, limits, *_ = inputs(tmp_path)
    result = compile_schedule(
        kernel,
        objects,
        None,
        adapter,
        encoding,
        limits,
        selected_contracts=selected,
        private_ownership=True,
        prefetch_qualification_identity=QUALIFICATION,
        budgets=ScheduleBudgets(max_prefetch_moves=0),
    )
    assert encoding.decode_calls(result.program.segments[0])[3].instr == "async_copy"


def test_prefetch_requires_its_explicit_pipeline_prerequisites(tmp_path):
    model = inputs(tmp_path)
    with pytest.raises(ValueError, match="pass|analysis|prerequisite|order"):
        compile_case(
            model, pass_order=("semantic_admission", "storage", "prefetch", "completion", "emission", "final_program")
        )
