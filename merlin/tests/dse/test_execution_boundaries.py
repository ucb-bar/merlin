"""Independent event traces close provider-neutral call/stack summaries."""

from collections import Counter
from dataclasses import replace

import pytest

from merlin.common.jsonio import canonical_sha256 as sha
from merlin.perf.execution_boundaries import (
    BoundaryDomain,
    BoundaryInstruction,
    FunctionExtent,
    boundary_features,
    derive_boundary_domain,
    summarize_boundaries,
)


def records(width=2, frame=16, repeats=2):
    # A fully expanded trace is constructed independently from the static
    # records below. It includes a repeated row load, read/write instruction,
    # nested direct calls and a callee with its own frame.
    trace = []
    for _ in range(repeats):
        trace.append((0, 0, 8, "none"))
        for _ in range(3):
            trace.append((1, 8, 0, "none"))
            trace.append((2, 0, 0, "direct"))
            trace.extend([(5, 0, 4, "none"), (6, 4, 4, "none"), (7, 4, 0, "none")])
            trace.append((3, 8, 8, "none"))
        trace.append((4, 8, 0, "none"))
    counts = Counter(pc for pc, *_ in trace)
    positions = [100 + i * width for i in range(8)]
    functions = [
        FunctionExtent("entry", positions[0], positions[5], positions[0], frame),
        FunctionExtent("child", positions[5], positions[7] + width, positions[5], frame * 2),
    ]
    decoded = [
        (0, 8, "none", None),
        (8, 0, "none", None),
        (0, 0, "direct", "child"),
        (8, 8, "none", None),
        (8, 0, "none", None),
        (0, 4, "none", None),
        (4, 4, "none", None),
        (4, 0, "none", None),
    ]
    instructions = [
        BoundaryInstruction(positions[i], width, counts[i], call, callee, read, write)
        for i, (read, write, call, callee) in enumerate(decoded)
    ]
    oracle = dict(
        executed_instructions=len(trace),
        direct_calls=sum(call == "direct" for *_, call in trace),
        stack_reads=sum(read > 0 for _, read, _, _ in trace),
        stack_writes=sum(write > 0 for _, _, write, _ in trace),
        stack_read_bytes=sum(read for _, read, _, _ in trace),
        stack_write_bytes=sum(write for _, _, write, _ in trace),
    )
    return functions, instructions, oracle


def summarize(functions, instructions):
    return summarize_boundaries(
        functions,
        instructions,
        program_sha256=sha("program"),
        census_sha256=sha("census"),
        provider_sha256=sha("provider"),
    )


@pytest.mark.parametrize("width,frame", [(2, 16), (7, 81)])
def test_expanded_event_oracle_and_two_distinct_provider_geometries(width, frame):
    functions, instructions, oracle = records(width, frame)
    result = summarize(functions, instructions)
    for field, expected in oracle.items():
        assert sum(row[field] for row in result["functions"].values()) == expected
    features = boundary_features(result, entry_function="entry")
    assert features["/boundaries/direct_calls_per_entry"] == 3
    assert features["/boundaries/stack_reads_per_entry"] == oracle["stack_reads"] / 2
    assert features["/boundaries/max_executed_frame_bytes"] == 2 * frame
    # This is an entry-weighted byte sum, not peak stack or traffic. There are
    # three child calls per complete entry invocation.
    assert features["/boundaries/frame_entry_bytes_per_entry"] == frame + 3 * (2 * frame)
    assert result["functions"]["entry"]["known_direct_callees"] == {"child": 6}
    # Both the loop's load and read/write instruction execute more than the
    # entry count. The child's instructions are once per *child* invocation.
    assert features["/boundaries/repeated_stack_sites"] == 2
    assert result["unknown"] and all("cycle" not in key for key in features)


def test_unresolved_direct_call_and_indirect_call_remain_explicit():
    functions, instructions, _ = records()
    instructions[2] = replace(instructions[2], callee=None)
    instructions[3] = replace(instructions[3], call="indirect")
    result = summarize(functions, instructions)
    features = boundary_features(result, entry_function="entry")
    assert features["/boundaries/unresolved_direct_calls_per_entry"] == 3
    assert features["/boundaries/indirect_calls_per_entry"] == 3
    assert result["functions"]["entry"]["known_direct_callees"] == {}


def test_unknown_executed_classification_is_not_zero_and_dead_unknown_is_zero():
    functions, instructions, _ = records()
    instructions[2] = replace(instructions[2], call="unknown", callee=None, stack_read_bytes=None)
    result = summarize(functions, instructions)
    features = boundary_features(result, entry_function="entry")
    for pointer in ("direct_calls", "indirect_calls", "unresolved_direct_calls", "stack_reads", "stack_read_bytes"):
        assert features["/boundaries/" + pointer + "_per_entry"] is None
    instructions[2] = replace(instructions[2], executions=0)
    features = boundary_features(summarize(functions, instructions), entry_function="entry")
    assert features["/boundaries/direct_calls_per_entry"] == 0
    assert features["/boundaries/stack_reads_per_entry"] is not None


def test_missing_execution_count_and_frame_are_unknown():
    functions, instructions, _ = records()
    instructions[1] = replace(instructions[1], executions=None)
    result = summarize(functions, instructions)
    features = boundary_features(result, entry_function="entry")
    assert features["/boundaries/executed_instructions_per_entry"] is None
    assert features["/boundaries/repeated_stack_sites"] is None
    assert features["/boundaries/max_executed_frame_bytes"] is None
    functions[1] = replace(functions[1], frame_bytes=None)
    assert (
        boundary_features(summarize(functions, instructions), entry_function="entry")[
            "/boundaries/max_executed_frame_bytes"
        ]
        is None
    )


def test_empty_execution_cannot_normalize_by_zero_entries():
    functions, instructions, _ = records()
    instructions = [replace(item, executions=0) for item in instructions]
    result = summarize(functions, instructions)
    assert result["functions"]["entry"]["executed_instructions"] == 0
    features = boundary_features(result, entry_function="entry")
    assert features["/boundaries/direct_calls_per_entry"] is None
    assert features["/boundaries/max_executed_frame_bytes"] == 0


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "split",
        "outside",
        "overlap",
        "entry",
        "identity",
        "callee",
        "width",
        "boolean",
        "negative",
        "call",
    ],
)
def test_malformed_or_partial_provider_reports_refuse(change):
    functions, instructions, _ = records()
    if change == "missing":
        instructions.pop(1)
    elif change == "duplicate":
        instructions.append(instructions[1])
    elif change == "split":
        instructions[1] = replace(instructions[1], width_bytes=3)
    elif change == "outside":
        instructions.append(replace(instructions[0], address=500))
    elif change == "overlap":
        functions[1] = replace(functions[1], start=functions[0].start)
    elif change == "entry":
        functions[0] = replace(functions[0], entry=101)
    elif change == "identity":
        functions[1] = replace(functions[1], identity="entry")
    elif change == "callee":
        instructions[2] = replace(instructions[2], callee="not-selected")
    elif change == "width":
        instructions[1] = replace(instructions[1], width_bytes=0)
    elif change == "boolean":
        instructions[1] = replace(instructions[1], executions=True)
    elif change == "negative":
        instructions[1] = replace(instructions[1], stack_read_bytes=-1)
    else:
        instructions[1] = replace(instructions[1], call="maybe")
    with pytest.raises(ValueError):
        summarize(functions, instructions)


def test_binding_mutation_is_detected_and_record_order_is_canonical():
    functions, instructions, _ = records()
    result = summarize(functions, instructions)
    reordered = summarize(list(reversed(functions)), list(reversed(instructions)))
    assert result == reordered
    changed = dict(result, program_sha256=sha("different"))
    with pytest.raises(ValueError, match="identity changed"):
        boundary_features(changed, entry_function="entry")
    rebound = summarize_boundaries(
        functions,
        instructions,
        program_sha256=sha("different"),
        census_sha256=sha("census"),
        provider_sha256=sha("provider"),
    )
    assert rebound["summary_sha256"] != result["summary_sha256"]


def test_training_boundary_domain_refuses_new_call_stack_and_frame_schedule():
    functions, instructions, _ = records()
    # Original schedules have no source call and no loop stack load.
    instructions[2] = replace(instructions[2], call="none", callee=None)
    instructions[1] = replace(instructions[1], stack_read_bytes=0)
    training = summarize(functions, instructions)
    pointers = [
        "/boundaries/direct_calls_per_entry",
        "/boundaries/repeated_stack_sites",
        "/boundaries/max_executed_frame_bytes",
    ]
    domain = derive_boundary_domain(
        [training], pointers=pointers, entry_function="entry", domain_sha256=sha("scope-regime")
    )
    known = domain.admit(boundary_features(training, entry_function="entry"), domain_sha256=sha("scope-regime"))
    assert known["status"] == "supported_boundary_subdomain"
    assert not known["ranking_approved"] and known["cycles"] == "UNKNOWN"
    candidate_functions, candidate_instructions, _ = records(frame=200)
    candidate = boundary_features(summarize(candidate_functions, candidate_instructions), entry_function="entry")
    refusal = domain.admit(candidate, domain_sha256=sha("scope-regime"))
    assert refusal["status"] == "unknown" and len(refusal["reasons"]) == 3
    assert domain.admit(candidate, domain_sha256=sha("different-regime"))["status"] == "unknown"


def test_unknown_training_and_unavailable_admission_features_refuse():
    functions, instructions, _ = records()
    functions[0] = replace(functions[0], frame_bytes=None)
    training = summarize(functions, instructions)
    with pytest.raises(ValueError, match="UNKNOWN"):
        derive_boundary_domain(
            [training],
            pointers=["/boundaries/max_executed_frame_bytes"],
            entry_function="entry",
            domain_sha256=sha("d"),
        )
    domain = BoundaryDomain(sha("d"), (("/boundaries/direct_calls_per_entry", 0, 2),), sha("e"))
    assert domain.admit({}, domain_sha256=sha("d"))["reasons"] == ["/boundaries/direct_calls_per_entry is UNKNOWN"]
    assert (
        domain.admit({"/boundaries/direct_calls_per_entry": float("nan")}, domain_sha256=sha("d"))["status"]
        == "unknown"
    )


def test_domain_bounds_need_explicit_valid_evidence():
    for bounds in (
        (),
        (("not-pointer", 0, 1),),
        (("/boundaries/a", 0, float("inf")),),
        (("/boundaries/a", 2, 1),),
        (("/boundaries/a", 0, 1), ("/boundaries/a", 0, 2)),
    ):
        with pytest.raises(ValueError):
            BoundaryDomain(sha("d"), bounds, sha("e"))
