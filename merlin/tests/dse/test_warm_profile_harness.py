"""The deployment wrapper warms once and times only completed target work."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.perf.execution_policy import WarmProfileContract
from merlin.perf.warm_profile_harness import (
    TargetInvocationHooks,
    WarmProfileHarnessError,
    render_target_warm_then_measure_main,
    render_warm_then_measure_main,
)
from merlin.targetgen.contract.harness_abi import HarnessAbi, HarnessAbiError


def _source(*, warmup_runs: int = 1) -> str:
    abi = HarnessAbi(entry_symbol="synthetic_launch", fence_symbol="synthetic_wait")
    return render_warm_then_measure_main(
        prepare_input="prepare_input_outside_timing();",
        invocation=abi.warm_profile_invocation("model_context"),
        validate_outputs="validate_outputs_after_timing()",
        contract=WarmProfileContract(warmup_runs=warmup_runs),
        cycle_reader="synthetic_cycles",
    )


def test_one_warm_then_one_completed_measured_invocation() -> None:
    source = _source()
    assert source.count("synthetic_launch(model_context);") == 2
    assert source.count("synthetic_wait();") == 2
    assert source.count("synthetic_cycles();") == 2
    assert source.count("METRIC cycles ") == 1
    assert "MERLIN_INVOCATIONS warmup=1 measured=1" in source

    warm_begin = source.index("MERLIN_PROFILE warmup begin")
    warm_launch = source.index("synthetic_launch(model_context);", warm_begin)
    warm_wait = source.index("synthetic_wait();", warm_launch)
    warm_end = source.index("MERLIN_PROFILE warmup end rc=0", warm_wait)
    measured_begin = source.index("MERLIN_PROFILE measured begin", warm_end)
    cycle_start = source.index("merlin_profile_cycle_start", measured_begin)
    measured_launch = source.index("synthetic_launch(model_context);", cycle_start)
    measured_wait = source.index("synthetic_wait();", measured_launch)
    cycle_end = source.index("merlin_profile_cycle_end", measured_wait)
    validation = source.index("validate_outputs_after_timing()", cycle_end)
    metric = source.index("METRIC cycles ", validation)
    measured_end = source.index("MERLIN_PROFILE measured end rc=0", metric)
    assert (warm_begin < warm_launch < warm_wait < warm_end < measured_begin
            < cycle_start < measured_launch < measured_wait < cycle_end
            < validation < metric < measured_end)
    assert source.index("prepare_input_outside_timing();") < warm_begin


def test_configured_warm_invocations_are_all_unmeasured_and_completed() -> None:
    source = _source(warmup_runs=3)
    assert "MERLIN_INVOCATIONS warmup=3 measured=1" in source
    assert source.count("synthetic_launch(model_context);") == 4
    assert source.count("synthetic_wait();") == 4
    cycle_start = source.index("merlin_profile_cycle_start")
    assert source[:cycle_start].count("synthetic_launch(model_context);") == 3
    assert source[cycle_start:].count("synthetic_launch(model_context);") == 1


def test_mutable_reset_and_result_readback_stay_outside_the_cycle_window() -> None:
    abi = HarnessAbi(entry_symbol="launch", fence_symbol="complete")
    source = render_warm_then_measure_main(
        prepare_input="prepare();",
        invocation=abi.warm_profile_invocation("context"),
        reset_after_warm="restore_mutable_state();",
        validate_outputs="validate()",
        success_body='readback();\nprintf("DONE\\n");',
    )
    warm_complete = source.index("complete();")
    reset = source.index("restore_mutable_state();", warm_complete)
    start = source.index("merlin_profile_cycle_start", reset)
    end = source.index("merlin_profile_cycle_end", start)
    metric = source.index("METRIC cycles", end)
    readback = source.index("readback();", metric)
    assert warm_complete < reset < start < end < metric < readback


def test_failed_validation_has_no_metric_before_the_failure_branch() -> None:
    source = _source()
    validation = source.index("const int merlin_profile_validation_rc")
    failure = source.index("if (merlin_profile_validation_rc != 0)", validation)
    metric = source.index("METRIC cycles ", failure)
    assert "return merlin_profile_validation_rc;" in source[failure:metric]


def test_profile_requires_target_declared_completion() -> None:
    abi = HarnessAbi(entry_symbol="synchronous_looking_but_unproved", fence_symbol=None)
    with pytest.raises(HarnessAbiError, match="explicit completion/fence"):
        abi.warm_profile_invocation("context")


@pytest.mark.parametrize("warmup", [0, -1, True])
def test_profile_refuses_invalid_warm_counts(warmup) -> None:
    with pytest.raises((ValueError, WarmProfileHarnessError), match="warm"):
        render_warm_then_measure_main(
            prepare_input="prepare();",
            invocation=TargetInvocationHooks("launch();", "wait();"),
            validate_outputs="validate()",
            contract=WarmProfileContract(warmup_runs=warmup),
        )


def test_profile_refuses_more_than_one_measured_invocation() -> None:
    with pytest.raises(ValueError, match="exactly one measured"):
        WarmProfileContract(measured_runs=2)


def test_profile_refuses_extra_metrics_and_protocol_injection() -> None:
    with pytest.raises(WarmProfileHarnessError, match="only total_compute_cycles"):
        render_warm_then_measure_main(
            prepare_input="prepare();",
            invocation=TargetInvocationHooks("launch();", "wait();"),
            validate_outputs="validate()",
            contract=WarmProfileContract(captured_metrics=frozenset({
                "total_compute_cycles", "idle_cycles",
            })),
        )
    with pytest.raises(WarmProfileHarnessError, match="reserved profile tokens"):
        TargetInvocationHooks(
            'printf("METRIC cycles 0\\n");', "wait();")


def test_profile_uses_no_target_specific_vocabulary() -> None:
    source = _source().lower()
    assert "gemmini" not in source
    assert "npu_" not in source


def test_bundle_entrypoint_resolves_hooks_from_the_requested_target(monkeypatch) -> None:
    from merlin.targetgen.contract import harness_abi

    requested = []

    def resolve(target: str) -> HarnessAbi:
        requested.append(target)
        return HarnessAbi(entry_symbol="selected_launch", fence_symbol="selected_wait")

    monkeypatch.setattr(harness_abi, "for_target", resolve)
    source = render_target_warm_then_measure_main(
        target="synthetic_target",
        arguments="context",
        prepare_input="prepare();",
        validate_outputs="validate()",
    )
    assert requested == ["synthetic_target"]
    assert source.count("selected_launch(context);") == 2
    assert source.count("selected_wait();") == 2


def test_rendered_profile_is_valid_c_and_publishes_one_cycle_value(tmp_path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    main = _source()
    source = tmp_path / "profile.c"
    executable = tmp_path / "profile"
    source.write_text(
        """#include <stdint.h>
#include <stdio.h>
static int prepared;
static int launches;
static int completions;
static int cycle_reads;
static void prepare_input_outside_timing(void) { prepared += 1; }
static void synthetic_launch(void *context) { (void)context; launches += 1; }
static void synthetic_wait(void) { completions += 1; }
static uint64_t synthetic_cycles(void) { return cycle_reads++ ? 137 : 100; }
static int validate_outputs_after_timing(void) {
  return prepared == 1 && launches == 2 && completions == 2 ? 0 : 7;
}
static void *model_context;
""" + main,
        encoding="utf-8",
    )
    built = subprocess.run(
        [compiler, "-std=c11", "-Wall", "-Werror", str(source), "-o", str(executable)],
        capture_output=True, text=True, timeout=10,
    )
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.count("METRIC cycles ") == 1
    assert "METRIC cycles 37\n" in ran.stdout


class TestCounterBracketPlacement:
    """WHERE the counter bracket goes is the reason it belongs to the generator.

    Every bundle that pasted its own bracket also chose its own eight slots, and one of them quietly
    stopped emitting a complete occupancy partition -- so a whole-model run reported accelerator-busy
    as a lower bound and nothing downstream could price a bucket from it.
    """

    BRACKET = {"prologue": "counter_configure(0, 1);\ncounter_reset();",
               "epilogue": 'printf("HWC %u\\n", counter_read(0));'}

    def _rendered(self, bracket=None):
        return render_warm_then_measure_main(
            prepare_input="prepare();",
            invocation=TargetInvocationHooks("launch();", "wait();"),
            validate_outputs="validate()",
            reset_after_warm="reset_mutable();",
            counter_bracket=self.BRACKET if bracket is None else bracket)

    def test_configuration_follows_the_warm_run_and_its_reset(self):
        """Warm-up traffic inside the counted window would be attributed to the measured run."""
        source = self._rendered()
        assert (source.index("warmup end")
                < source.index("counter_configure(0, 1);")
                < source.index("MERLIN_PROFILE measured begin"))
        assert source.index("reset_mutable();") < source.index("counter_configure(0, 1);")

    def test_read_back_follows_completion_and_precedes_validation(self):
        """Validation is host work whose cost belongs to neither the counters nor the cycle window."""
        source = self._rendered()
        assert (source.index("merlin_profile_cycle_end")
                < source.index("counter_read(0)")
                < source.index("merlin_profile_validation_rc"))

    def test_omitting_the_bracket_leaves_the_harness_byte_identical(self):
        with_none = render_warm_then_measure_main(
            prepare_input="prepare();",
            invocation=TargetInvocationHooks("launch();", "wait();"),
            validate_outputs="validate()", reset_after_warm="reset_mutable();")
        assert "counter_" not in with_none

    def test_a_configure_without_a_read_back_is_refused(self):
        """It measures nothing; a read-back without a configure reports whatever the slots held."""
        for partial in ({"prologue": "counter_reset();"},
                        {"epilogue": "counter_read(0);"},
                        {"prologue": "counter_reset();", "epilogue": ""}):
            with pytest.raises(WarmProfileHarnessError, match="missing"):
                self._rendered(partial)

    def test_a_hand_written_slot_assignment_cannot_be_passed_as_text(self):
        with pytest.raises(WarmProfileHarnessError, match="mapping a counter selector returned"):
            self._rendered("counter_configure(0, 1);")

    def test_a_bracket_may_not_shadow_the_reserved_profile_tokens(self):
        with pytest.raises(WarmProfileHarnessError, match="reserved profile tokens"):
            self._rendered({"prologue": 'printf("METRIC cycles 0\\n");',
                            "epilogue": "counter_read(0);"})
