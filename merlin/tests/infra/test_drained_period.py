"""Counted emitted calls execute under an independent delayed-completion runtime."""
from dataclasses import replace
import importlib.util
import shutil
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.compile.scheduling.periodic_effects import check_drained_period
from merlin.sched.ir import Const, Loop

SPEC = importlib.util.spec_from_file_location("drained_example", repo_root() / "examples/compiler/drained_period.py")
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def native(source, tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    path = tmp_path / "example.c"
    binary = tmp_path / "example"
    path.write_text(source)
    subprocess.run([compiler, "-std=c11", "-O2", str(path), "-o", str(binary)], check=True, capture_output=True)
    return subprocess.run([str(binary)], capture_output=True, text=True)


def check(model, *, kernel=None, admission=None, **overrides):
    original, objects, banks, adapter, selected, options, _ = model
    return check_drained_period(kernel or original, objects, banks, adapter, admission or selected,
                                **(options | overrides))


@pytest.mark.parametrize("iterations", [1, 3, 7])
def test_counted_emission_executes_real_delayed_work_with_complete_outputs(tmp_path, iterations):
    source, report = EXAMPLE.program(iterations)
    assert source.count("for (int64_t period") == 1
    emitted = source.split("static void generated(")[1].split("int main(")[0]
    assert emitted.count("op_fetch(") == 2 and emitted.count("op_publish(") == 2
    result = native(source, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "peak outstanding=2" in result.stdout
    assert report["dynamic_calls_checked"] == iterations * 10 and report["static_calls"] == 10
    assert not report["affine_recurrence_proven"] and not report["numerical_equivalence_proven"]
    assert not report["target_execution_qualified"]


@pytest.mark.parametrize("mutation", ["overwrite", "missing_drain", "missing_input_wait"])
def test_actual_effect_hazards_refuse(mutation):
    model = EXAMPLE.fixture()
    kernel = model[0]
    body = list(kernel.body[0].body)
    if mutation == "overwrite":
        body[1] = replace(body[1], args=tuple((name, Const(0) if name == "slot" else value)
                                           for name, value in body[1].args))
    elif mutation == "missing_drain":
        body.pop()
    else:
        body.pop(2)
    changed = replace(kernel, body=(replace(kernel.body[0], body=tuple(body)),))
    with pytest.raises(ValueError, match="actual completion|returns before actual completion"):
        check(model, kernel=changed)


def test_budget_refuses_before_projecting_and_reports_needed_work():
    model = list(EXAMPLE.fixture(7))
    model[3] = replace(model[3], project=lambda _: pytest.fail("over-budget projection started"))
    with pytest.raises(ValueError, match="needs 70 dynamic call checks; budget is 69"):
        check(tuple(model), max_dynamic_calls=69)


def test_concurrency_assumptions_and_pending_limit_are_not_inferred():
    model = EXAMPLE.fixture()
    with pytest.raises(ValueError, match="concurrency assumption"):
        check(model, admission=replace(model[4], concurrent_pairs=()))
    with pytest.raises(ValueError, match="pending-work bound"):
        check(model, admission=replace(model[4], max_pending=1))


def test_callback_mutation_refuses_emission():
    model = EXAMPLE.fixture()
    checked = check(model)
    iset = model[-1]
    instruction = iset.instrs["fetch"]
    def mutate(args):
        object.__setattr__(model[0], "attrs", (("changed", "true"),))
        return instruction.render_c(args)
    changed = replace(iset, instrs=dict(iset.instrs, fetch=replace(instruction, render_c=mutate)))
    with pytest.raises(ValueError, match="changed"):
        checked.emit_c(changed, symbol="generated")


def test_checker_does_not_claim_external_output_coverage_or_renderer_semantics(tmp_path):
    model = EXAMPLE.fixture()
    checked = check(model)
    source, _ = EXAMPLE.program()
    # This renderer corruption leaves the checked scheduling input untouched.
    # Independent execution still detects the omitted output; a receipt cannot.
    source = source.replace("else output[h] = local[s];", "else (void)local[s];")
    assert checked.report()["numerical_equivalence_proven"] is False
    result = native(source, tmp_path)
    assert result.returncode != 0


def test_cross_period_pending_and_nested_loops_are_not_admitted():
    model = EXAMPLE.fixture()
    kernel = model[0]
    changed = replace(kernel, body=(replace(kernel.body[0], body=(Loop("nested", 2, kernel.body[0].body),)),))
    with pytest.raises(ValueError, match="exactly one counted loop"):
        check(model, kernel=changed)


def test_every_iteration_is_checked_not_only_a_good_prefix():
    model = EXAMPLE.fixture(3)
    kernel = model[0]
    body = tuple(replace(call, args=tuple(
        (name, value * 2 if name == "host" else value) for name, value in call.args))
        if call.instr in ("fetch", "publish") else call for call in kernel.body[0].body)
    changed = replace(kernel, body=(replace(kernel.body[0], body=body),))
    with pytest.raises(ValueError, match="external access lacks"):
        check(model, kernel=changed)
