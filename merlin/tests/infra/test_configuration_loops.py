"""Persistent configuration loop motion checked by independent state execution."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.configuration import ConfigurationAdapter, ConfigurationEffects, ConfigurationUnavailable
from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration, verify_loop_configuration_rewrite
from merlin.sched.ir import Call, Const, Kernel, Loop, Var, evaluate


def setup(value=7):
    return Call("setup", (("value", Const(value)),))


def project(call):
    if call.instr == "setup":
        value = call.arg("value")
        # A dynamic value is not an independently decoded constant setup.
        return (
            ConfigurationEffects(writes=(("mode", value.value),), ordered=True, pure_write=True)
            if type(value) is Const
            else None
        )
    if call.instr == "compute":
        return ConfigurationEffects(reads=(("mode", 7),), ordered=True)
    if call.instr == "reset":
        return ConfigurationEffects(clobbers=("mode",), ordered=True)
    if call.instr == "pulse":
        return ConfigurationEffects(ordered=True)  # Has no configuration effect, but is NOT a pure write.
    return None


ADAPTER = ConfigurationAdapter(project, "neutral-persistent-registers", "independent-test-state-machine-only")


def run(kernel):
    state, observations, counts = {"mode": -191}, [], {"setup": 0, "compute": 0, "reset": 0, "pulse": 0}

    def execute(body, indices):
        for op in body:
            if type(op) is Loop:
                for i in range(op.extent):
                    execute(op.body, {**indices, op.var: i})
            else:
                values = {key: evaluate(expr, indices) for key, expr in op.args}
                if op.instr == "setup":
                    state["mode"] = values["value"]
                elif op.instr == "compute":
                    observations.append((state["mode"], values.get("address", 0)))
                elif op.instr in ("reset", "unknown"):
                    state["mode"] = -312
                elif op.instr == "pulse":
                    observations.append(("pulse", state["mode"]))
                counts[op.instr] = counts.get(op.instr, 0) + 1

    execute(kernel.body, {})
    return state, observations, counts


@pytest.mark.parametrize("extent,stride", [(3, 4), (17, 16), (33, 64)])
def test_real_loop_execution_preserves_state_and_observations_with_fewer_setups(extent, stride):
    original = Kernel(
        "commands",
        (),
        (Loop("i", extent, (setup(), Call("compute", (("address", Var("i") * stride),)))), Call("compute", ())),
    )
    result = hoist_loop_configuration(original, ADAPTER)
    before, after = run(original), run(result.kernel)
    assert before[:2] == after[:2]
    assert before[2]["setup"] == extent and after[2]["setup"] == 1
    assert result.report["removed_dynamic_configuration_calls"] == extent - 1
    assert result.kernel.body[0] == setup()
    cloned = replace(result.kernel, body=(replace(result.kernel.body[0]), *result.kernel.body[1:]))
    verify_loop_configuration_rewrite(original, cloned, ADAPTER)


def test_nested_loops_never_assume_unknown_initial_state():
    original = Kernel(
        "nested",
        (),
        (
            Loop(
                "outer",
                7,
                (Loop("inner", 11, (setup(), Call("compute", (("address", Var("outer") * 32 + Var("inner")),)))),),
            ),
        ),
    )
    result = hoist_loop_configuration(original, ADAPTER)
    assert run(original)[:2] == run(result.kernel)[:2]
    assert run(result.kernel)[2]["setup"] == 7
    assert result.report["removed_dynamic_configuration_calls"] == 70
    disabled = hoist_loop_configuration(original, ADAPTER, enabled=False)
    assert disabled.kernel == original


@pytest.mark.parametrize("barrier", [Call("unknown", ()), Call("reset", ()), setup(9)])
def test_unknown_or_clobbering_body_preserves_setup_on_every_iteration(barrier):
    original = Kernel("hazards", (), (Loop("i", 8, (setup(), Call("compute", ()), barrier)),))
    result = hoist_loop_configuration(original, ADAPTER)
    assert result.kernel == original
    assert result.report["hoisted_static_calls"] == 0
    illegal = replace(original, body=(setup(), replace(original.body[0], body=original.body[0].body[1:])))
    with pytest.raises(ValueError, match="invariant|unknown"):
        verify_loop_configuration_rewrite(original, illegal, ADAPTER)


def test_pulse_with_known_configuration_effects_is_never_hoisted():
    original = Kernel("pulse", (), (Loop("i", 7, (Call("pulse", ()), setup(), Call("compute", ()))),))
    result = hoist_loop_configuration(original, ADAPTER)
    assert result.kernel == original and run(result.kernel)[2]["pulse"] == 7


@pytest.mark.parametrize("mutation", ["value", "domain", "drop", "extra", "compute"])
def test_independent_verifier_rejects_changed_actual_call_or_loop(mutation):
    source = Kernel("checked", (), (Loop("i", 9, (setup(), Call("compute", ()))),))
    actual = hoist_loop_configuration(source, ADAPTER).kernel
    if mutation == "value":
        actual = replace(actual, body=(setup(8), actual.body[1]))
    elif mutation == "domain":
        actual = replace(actual, body=(actual.body[0], replace(actual.body[1], extent=8)))
    elif mutation == "drop":
        actual = replace(actual, body=(replace(actual.body[1], body=()),))
    elif mutation == "extra":
        actual = replace(actual, body=(*actual.body, setup()))
    else:
        actual = replace(actual, body=(actual.body[0], replace(actual.body[1], body=(Call("pulse", ()),))))
    with pytest.raises(ValueError):
        verify_loop_configuration_rewrite(source, actual, ADAPTER)


def test_annotation_and_static_budget_cannot_be_erased():
    annotated = Kernel("dependency", (), (Loop("i", 4, (replace(setup(), produces="ready"), Call("compute", ()))),))
    with pytest.raises(ValueError):
        hoist_loop_configuration(annotated, ADAPTER)
    ordinary = Kernel("bounded", (), (Loop("i", 4, (setup(), Call("compute", ()))),))
    with pytest.raises(ConfigurationUnavailable, match="budget"):
        hoist_loop_configuration(ordinary, ADAPTER, max_calls=1)


def test_zero_trip_loop_is_not_speculated():
    original = Kernel("empty", (), (Loop("i", 0, (setup(), Call("compute", ()))),))
    from merlin.sched.ir.kernel import check_structure

    if check_structure(original):
        with pytest.raises(ValueError):
            hoist_loop_configuration(original, ADAPTER)
    else:
        result = hoist_loop_configuration(original, ADAPTER)
        assert result.kernel == original and run(result.kernel) == run(original)


def test_changed_actual_effect_projection_cannot_inherit_original_legality():
    source_call = setup()
    source = Kernel("effect_custody", (), (Loop("i", 9, (source_call, Call("compute", ()))),))
    actual = hoist_loop_configuration(source, ADAPTER).kernel
    clone = replace(actual.body[0])
    actual = replace(actual, body=(clone, actual.body[1]))
    foreign = replace(ADAPTER, project=lambda call: None if call is clone else project(call))
    with pytest.raises(ValueError, match="effects"):
        verify_loop_configuration_rewrite(source, actual, foreign)


def test_pointer_valued_setup_retains_its_original_scope():
    from merlin.sched.ir import Ptr, TensorArg

    pointer_call = Call("setup", (("value", Ptr("input", Var("i"))),))
    source = Kernel(
        "pointer", (TensorArg("input", (64,), "i8", "read"),), (Loop("i", 7, (pointer_call, Call("compute", ()))),)
    )
    adapter = replace(
        ADAPTER,
        project=lambda call: (
            ConfigurationEffects(writes=(("mode", 7),), ordered=True, pure_write=True)
            if call is pointer_call
            else project(call)
        ),
    )
    result = hoist_loop_configuration(source, adapter)
    assert result.kernel == source
