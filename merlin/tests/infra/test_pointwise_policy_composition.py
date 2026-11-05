"""Candidate pass order changes executable programs under fixed independent checks.

This neutral functional fixture demonstrates policy expressivity, not learned
breakpoints, accelerator qualification or a latency improvement.
"""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_configured_pointwise import EXECUTOR, HIGH, LOW, case
from test_pointwise_source import execute as native_executor_fixture
from test_pointwise_source import source_program

from merlin.common.cpu_pool import shared_compute_pool
from merlin.compile.scheduling.pointwise_source import (
    PointwiseScheduleCandidate,
    PointwiseScheduleRewrite,
    compile_source_pointwise,
)
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets


def deduplicate_settings(candidate, context):
    """Candidate implementation; the fixed checker separately proves omissions."""
    known, retained = {}, []
    assert len(context.configuration_effects) == len(candidate.kernel.body)
    for call, effect in zip(candidate.kernel.body, context.configuration_effects, strict=True):
        redundant = (
            effect.ordered and effect.pure_write and all(known.get(field) == value for field, value in effect.writes)
        )
        if not redundant:
            retained.append(call)
        if not effect.ordered:
            known.clear()
        else:
            for field in effect.clobbers:
                known.pop(field, None)
            known.update(effect.writes)
    return replace(candidate, kernel=replace(candidate.kernel, body=tuple(retained)))


def compose_adjacent(candidate, context):
    """Use an actual selected emitter only when the current arithmetic is adjacent."""
    calls = candidate.kernel.body
    assert len(context.storage_effects) == len(context.configuration_effects) == len(calls)
    arithmetic = [i for i, effect in enumerate(context.storage_effects) if effect.elementwise is not None]
    if len(arithmetic) != 2 or arithmetic[1] != arithmetic[0] + 1:
        return candidate
    first, second = arithmetic
    for capability in context.fusion_capabilities:
        fused = capability.emit(calls[first], calls[second])
        if fused is not None:
            return PointwiseScheduleCandidate(
                replace(candidate.kernel, body=(*calls[:first], fused, *calls[second + 1 :])), ()
            )
    return candidate


CLEANUP = PointwiseScheduleRewrite("deduplicate-settings", deduplicate_settings)
COMPOSE = PointwiseScheduleRewrite("compose-adjacent", compose_adjacent)


def choose_order(shape, facts):
    """An authored test hypothesis over shape and selected physical resources."""
    elements = math.prod(shape)
    if elements >= 20 and elements <= facts["max_elements"] and facts["state_fields"] >= 2:
        return CLEANUP, COMPOSE
    return COMPOSE, CLEANUP


@pytest.fixture
def execute(tmp_path):
    # This native fixture child does not acquire its own lease. Release the
    # construction lease before returning, as ordinary queued children must.
    with shared_compute_pool(timeout_s=120):
        return native_executor_fixture.__wrapped__(tmp_path, SimpleNamespace(param=dict(source=EXECUTOR, counters=5)))


@pytest.mark.parametrize(
    "m,n,separate,physical_limit,selected_order",
    [
        (3, 7, False, 64, (CLEANUP, COMPOSE)),
        (2, 11, True, 64, (CLEANUP, COMPOSE)),
        (1, 19, False, 64, (COMPOSE, CLEANUP)),
        (3, 7, True, 14, (COMPOSE, CLEANUP)),
    ],
    ids=["odd-shape-shared-bank", "narrow-separate-banks", "below-policy-breakpoint", "physical-fusion-refusal"],
)
def test_candidate_composition_order_changes_actual_program_not_complete_results(
    tmp_path, execute, m, n, separate, physical_limit, selected_order
):
    # Only actual bounded compiler/native work owns a slot; there is no queued
    # worker or RPC beneath this scope that also waits for one.
    with shared_compute_pool(timeout_s=120):
        module, operation = source_program(m=m, n=n)
        source_before = str(module)
        options = case(
            tmp_path,
            m=m,
            n=n,
            separate=separate,
            mutate=lambda facts, software, fusion: facts.update(max_elements=physical_limit),
        )
        facts = json.loads(options["selected_contracts"].facts_path.read_text())["facts"]
        original_digest = options["kernel"].digest()
        # Disable the corresponding built-in passes: ordering of these two
        # candidate implementations is the only composition mechanism here.
        budgets = ScheduleBudgets(max_fusions=0, eliminate_configuration=False, eliminate_copies=False)
        baseline = compile_source_pointwise(module, operation, budgets=budgets, **options)
        selected = choose_order((m, n), facts)
        assert selected == selected_order
        orders = (selected, tuple(reversed(selected)))
        arms = {
            tuple(rewrite.name for rewrite in order): compile_source_pointwise(
                module, operation, budgets=budgets, rewrites=order, **options
            )
            for order in orders
        }
        cleanup_first = arms[(CLEANUP.name, COMPOSE.name)]
        compose_first = arms[(COMPOSE.name, CLEANUP.name)]
        composed = m * n <= physical_limit
        expected_calls = ["setup", "setup", "fused"] if composed else ["setup", "setup", "readout", "finish"]
        decoder = options["encoding"].decode_calls
        assert [call.instr for call in decoder(b"".join(cleanup_first.images()))] == expected_calls
        assert [call.instr for call in decoder(b"".join(compose_first.images()))] == [
            "setup",
            "setup",
            "readout",
            "finish",
        ]
        assert (cleanup_first.images() != compose_first.images()) == composed

        # Order is effective before emission, with refreshed current effects at
        # every pass. The reverse arm's first pass is a genuine no-op.
        for order, result in arms.items():
            checks = result.rewrite_checks
            assert tuple(check["name"] for check in checks) == order
            assert checks[0]["rewritten_schedule_sha256"] == checks[1]["source_schedule_sha256"]
            assert all(check["timing"] == "UNKNOWN" for check in checks)
        first = compose_first.rewrite_checks[0]
        assert first["source_schedule_sha256"] == first["rewritten_schedule_sha256"]
        first = cleanup_first.rewrite_checks[0]
        assert first["source_schedule_sha256"] != first["rewritten_schedule_sha256"]
        second = cleanup_first.rewrite_checks[1]
        assert (second["source_schedule_sha256"] != second["rewritten_schedule_sha256"]) == composed

        # Two independent signed input domains exercise unsaturated elements,
        # both clamp ends, and i32 wraparound before the selected i8 readout.
        elements = m * n
        inputs = [
            (np.arange(elements, dtype="<i4") * 37 - 350, np.full(elements, 17, "<i4")),
            (np.arange(elements, dtype="<i4") * 73 - 1000, np.full(elements, (1 << 31) - 2, "<i4")),
        ]
        for x, bias in inputs:
            total = x.astype(np.int64) + bias.astype(np.int64)
            wrapped = (total + (1 << 31)) % (1 << 32) - (1 << 31)
            expected = np.clip(wrapped, -128, 127).astype("i1").tobytes() + bytes([193] * (1032 - elements))
            for result, commands, fused in (
                (baseline, 18, False),
                (cleanup_first, 3 if composed else 4, composed),
                (compose_first, 4, False),
            ):
                output, counters = execute(result.images(), {"x": x.tobytes(), "b0": bias.tobytes()}, separate=separate)
                # The executor additionally checks every bank's end guards and
                # all immutable input storage, including the aliased layout.
                assert output == expected
                assert counters == (
                    commands,
                    (8 if fused else 12) * elements,
                    (1 if fused else 5) * elements,
                    LOW,
                    HIGH,
                )
                assert counters[:3] == tuple(
                    result.conformance[key]
                    for key in ("actual_commands", "requested_read_bytes", "requested_write_bytes")
                )
                assert result.conformance["configuration"]["known_final_state"] == {
                    "settings:0": LOW,
                    "settings:1": HIGH,
                }
        if composed:
            # On the overflow input, a physical fused readout which skips i32
            # wrapping must fail the independent oracle. The ordinary input
            # intentionally would not distinguish that numerical defect.
            incorrect, _ = execute(
                cleanup_first.images(), {"x": x.tobytes(), "b0": bias.tobytes()}, separate=separate, wrong=True
            )
            assert incorrect != expected
        assert str(module) == source_before
        assert options["kernel"].digest() == original_digest
