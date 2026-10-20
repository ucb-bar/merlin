"""New pass implementations change emitted bytes under fixed source/effect checks."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_configured_pointwise import EXECUTOR, HIGH, LOW, case
from test_pointwise_source import execute as execute
from test_pointwise_source import source_program
from test_readout_model_fusion import command, params
from xdsl.dialects import builtin

from merlin.compile.scheduling.fusion_semantics import UnsupportedElementwise
from merlin.compile.scheduling.pointwise_source import (
    PointwiseScheduleCandidate,
    PointwiseScheduleRewrite,
    compile_source_pointwise,
)
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets


def compose(candidate, context):
    """A new pass body, independent of the built-in adjacency fusion planner.

    Pure setup between producer and consumer prevents that planner's adjacency
    match. Preserve it while selecting a declared physical composition.
    """
    assert context.shape and context.scalar_ir and context.banks
    assert context.numerical_policy and context.hardware_identity
    calls = candidate.kernel.body
    assert len(context.storage_effects) == len(context.configuration_effects) == len(calls)
    arithmetic = [
        call for call, effect in zip(calls, context.configuration_effects, strict=True) if not effect.pure_write
    ]
    if len(arithmetic) != 2:
        return candidate
    result = next((cap.emit(*arithmetic) for cap in context.fusion_capabilities), None)
    if result is None:
        return candidate
    return PointwiseScheduleCandidate(
        replace(
            candidate.kernel,
            body=(
                *[call for call, effect in zip(calls, context.configuration_effects, strict=True) if effect.pure_write],
                result,
            ),
        ),
        (),
    )


@pytest.mark.parametrize("execute", [dict(source=EXECUTOR, counters=5)], indirect=True, ids=["native-stateful-machine"])
@pytest.mark.parametrize("m,n,separate", [(3, 5, False), (2, 11, True), (1, 64, False)])
def test_new_source_transform_changes_real_bytes_with_builtin_fusion_disabled(tmp_path, execute, m, n, separate):
    module, operation = source_program(m=m, n=n)
    options = case(tmp_path, m=m, n=n, separate=separate)
    budgets = ScheduleBudgets(max_fusions=0, eliminate_configuration=False, eliminate_copies=False)
    original = compile_source_pointwise(module, operation, budgets=budgets, **options)
    optimized = compile_source_pointwise(
        module,
        operation,
        budgets=budgets,
        rewrites=(PointwiseScheduleRewrite("compose-readout", compose),),
        **options,
    )
    x = np.arange(m * n, dtype=np.int32) * 73 - 1000
    bias = np.full(m * n, (1 << 31) - 2, dtype=np.int32)
    wrapped = (x.astype(np.int64) + bias.astype(np.int64) + (1 << 31)) % (1 << 32) - (1 << 31)
    expected = np.clip(wrapped, -128, 127).astype("i1").tobytes() + bytes([193] * (1032 - m * n))
    before, initial = execute(original.images(), {"x": x.tobytes(), "b0": bias.tobytes()}, separate=separate)
    after, actual = execute(optimized.images(), {"x": x.tobytes(), "b0": bias.tobytes()}, separate=separate)
    assert before == after == expected
    assert original.images() != optimized.images()
    assert initial[0] == actual[0] + 1 == 18
    assert initial[1] - actual[1] == initial[2] - actual[2] == 4 * m * n
    assert initial[3:] == actual[3:] == (LOW, HIGH)
    assert optimized.rewrite_checks[0]["name"] == "compose-readout"
    assert optimized.rewrite_checks[0]["timing"] == "UNKNOWN"
    assert optimized.conformance["configuration"]["known_final_state"] == {"settings:0": LOW, "settings:1": HIGH}
    assert [record.name for record in optimized.source_passes] == [
        "source-admission",
        "candidate-rewrites",
        "schedule-lowering",
        "source-conformance",
    ]
    assert options["kernel"].digest() == original.compiled.report["source_sha256"]
    # A falsely implemented primitive still fails independent native checking.
    wrong, _ = execute(optimized.images(), {"x": x.tobytes(), "b0": bias.tobytes()}, separate=separate, wrong=True)
    assert wrong != expected


@pytest.mark.parametrize("damage", ["arithmetic", "output", "interface", "objects", "state", "unknown", "async"])
def test_new_pass_cannot_relax_source_effects_or_invocation(tmp_path, damage):
    module, operation = source_program()
    options = case(tmp_path)
    emitted = []
    options["encoding"] = replace(options["encoding"], encode=lambda kernel: emitted.append(kernel))

    def wrong(candidate, context):
        result = compose(candidate, context)
        kernel = result.kernel
        calls = list(kernel.body)
        if damage in {"arithmetic", "output"}:
            values = params(calls[-1])
            values["bias" if damage == "arithmetic" else "dst"] = 3
            calls[-1] = command("fused", **values)
        elif damage == "interface":
            return replace(result, kernel=replace(kernel, name="different-entry"))
        elif damage == "objects":
            return replace(result, objects=candidate.objects)
        elif damage == "state":
            calls.append(command("setup", src=0, source=3))
        elif damage == "unknown":
            calls[-1] = replace(calls[-1], instr="opaque")
        else:
            calls[-1] = replace(calls[-1], produces="pending")
        return replace(result, kernel=replace(kernel, body=tuple(calls)))

    with pytest.raises(ValueError):
        compile_source_pointwise(
            module, operation, rewrites=(PointwiseScheduleRewrite("invalid-new-pass", wrong),), **options
        )
    assert emitted == []


def test_direct_new_instruction_cannot_bypass_rtl_capacity(tmp_path):
    module, operation = source_program()
    options = case(tmp_path, mutate=lambda facts, software, fusion: facts.update(max_elements=14))

    def bypass(candidate, context):
        return PointwiseScheduleCandidate(
            replace(
                candidate.kernel,
                body=(
                    *[c for c in candidate.kernel.body if c.instr == "setup"],
                    command("fused", m=3, n=5, src=3, dst=9, bias=6),
                ),
            ),
            (),
        )

    with pytest.raises(UnsupportedElementwise, match="selected RTL resource domain"):
        compile_source_pointwise(module, operation, rewrites=(PointwiseScheduleRewrite("bypass", bypass),), **options)
    # The qualified emitter refuses that same out-of-domain candidate; a pass
    # can retain the legal unfused schedule without changing the cohort.
    fallback = compile_source_pointwise(
        module, operation, rewrites=(PointwiseScheduleRewrite("bounded-compose", compose),), **options
    )
    assert [call.instr for call in fallback.compiled.kernel.body] == ["setup", "setup", "readout", "finish"]


def test_missing_configuration_effects_or_qualification_refuses_new_rewrite(tmp_path):
    module, operation = source_program()
    options = case(tmp_path)
    options["configuration_qualification_identity"] = None
    with pytest.raises(UnsupportedElementwise, match="qualified configuration"):
        compile_source_pointwise(module, operation, rewrites=(PointwiseScheduleRewrite("new", compose),), **options)


@pytest.mark.parametrize("mutation", ["facts", "source", "input"])
def test_rewrite_cannot_mutate_current_source_or_selected_contracts(tmp_path, mutation):
    module, operation = source_program()
    options = case(tmp_path)

    def mutate(candidate, context):
        if mutation == "facts":
            path = options["selected_contracts"].facts_path
            path.write_text(json.dumps({"changed": True}))
        elif mutation == "source":
            constant = next(op for op in operation.walk() if op.name == "arith.constant")
            constant.properties["value"] = builtin.IntegerAttr(-127, builtin.i32)
        else:
            object.__setattr__(candidate.kernel, "name", "mutated")
        return candidate

    with pytest.raises(ValueError, match="specification changed|changed IR|mutated its original"):
        compile_source_pointwise(module, operation, rewrites=(PointwiseScheduleRewrite("mutate", mutate),), **options)


@pytest.mark.parametrize("kind", ["duplicate", "many", "untyped"])
def test_rewrite_roster_is_explicit_and_bounded(tmp_path, kind):
    module, operation = source_program()
    options = case(tmp_path)
    rewrite = PointwiseScheduleRewrite("same", compose)
    rewrites = (
        (rewrite, rewrite)
        if kind == "duplicate"
        else tuple(PointwiseScheduleRewrite(str(i), compose) for i in range(9))
        if kind == "many"
        else (compose,)
    )
    with pytest.raises(ValueError, match="eight unique typed"):
        compile_source_pointwise(module, operation, rewrites=rewrites, **options)
