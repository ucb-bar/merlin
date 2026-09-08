"""The declared correctness gate, and a MUTATION test proving it can fail.

A check that cannot fail reports success — this repo has recorded that shape fifteen times. So the
last class here compiles the generated harness, links it against a stub kernel, and runs it four
ways: correct, argmax moved, one element outside tolerance, and non-finite. Three must be caught and
``METRIC cycles`` must appear only in the clean run, because a metric published beside a failed gate
is a performance number for a program that computed the wrong answer.

The refusals encode measured history. In an ``*_int8_*`` recapture ``golden.npy`` is a
weight-only-int8 reference and merlin's path is W8A8; grading one against the other measures
activation-quantization error, and a correct K1 run once scored **cos 0.484** that way and was chased
as a codegen defect. ``tiny_llama_int8_w8a8_consistent`` ships *only* ``golden.npy`` today, so that
refusal fires on the real tree.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from merlin.targetgen import bundle_gate as BG
from merlin.targetgen import bundle_harness as BH
from merlin.targetgen import bundle_pack as BP
from merlin.targetgen.bundle_gate import GateError


def _resnet_gate(**kw):
    base = dict(model="resnet50", datapath="w8a8", reference_kind="pt2e_integer",
                comparison="exact_elementwise", atol=2e-5, rtol=2e-5,
                output_elements=1000, expected_argmax=258)
    base.update(kw)
    return BG.gate_for(**base)


class TestTheReferenceMustMeanWhatTheGateClaims:
    def test_a_w8a8_datapath_cannot_be_graded_against_the_weight_only_reference(self):
        """THE cos-0.484 REFUSAL, and it fires on tiny_llama's real bundle contents."""
        with pytest.raises(GateError) as exc:
            BG.gate_for(model="tiny_llama", datapath="w8a8", reference_kind="weight_only_int8",
                        comparison="tolerance_and_topk", atol=0.03125, rtol=0.02,
                        output_elements=256000)
        message = str(exc.value)
        assert "cos 0.484" in message, "the refusal must carry the measured evidence"
        assert "w8a8_independent" in message and "make_w8a8_independent_golden.py" in message

    def test_the_fp32_reference_is_refused_for_a_w8a8_datapath_too(self):
        with pytest.raises(GateError, match="cannot be graded against"):
            BG.gate_for(model="m", datapath="w8a8", reference_kind="fp32",
                        comparison="tolerance_and_topk", atol=1e-3, rtol=1e-3,
                        output_elements=10)

    def test_an_fp32_datapath_graded_against_a_quantized_reference_is_refused(self):
        with pytest.raises(GateError, match="measuring the reference's quantization"):
            BG.gate_for(model="m", datapath="fp32", reference_kind="w8a8_independent",
                        comparison="tolerance_and_topk", atol=1e-3, rtol=1e-3,
                        output_elements=10)

    def test_the_execution_reference_requires_its_scope_declared(self):
        """It scores cos 1.0 / rel 0.0 by construction; using it silently would overclaim."""
        with pytest.raises(GateError, match="must declare that scope"):
            BG.gate_for(model="m", datapath="w8a8", reference_kind="w8a8_execution",
                        comparison="tolerance_and_topk", atol=1e-4, rtol=1e-4,
                        output_elements=10)
        got = BG.gate_for(model="m", datapath="w8a8", reference_kind="w8a8_execution",
                          comparison="tolerance_and_topk", atol=1e-4, rtol=1e-4,
                          output_elements=10, scope_note="device reproduces the host compiler only")
        assert got.reference_file == "golden_w8a8.npy"
        assert "not evidence about the arithmetic" in got.to_dict()["reference_scope"]

    def test_an_exact_comparison_needs_an_exact_reference(self):
        with pytest.raises(GateError, match="two different roundings agree"):
            BG.gate_for(model="m", datapath="w8a8", reference_kind="w8a8_independent",
                        comparison="exact_elementwise", atol=0.0, rtol=0.0, output_elements=10)

    def test_a_missing_reference_names_its_generator_instead_of_falling_back(self):
        """tiny_llama's actual state: only golden.npy is present."""
        with pytest.raises(GateError, match="make_w8a8_independent_golden.py"):
            BG.gate_for(model="tiny_llama", datapath="w8a8", reference_kind="w8a8_independent",
                        comparison="tolerance_and_topk", atol=0.03125, rtol=0.02,
                        output_elements=256000,
                        available_references={"weight_only_int8": True,
                                              "w8a8_independent": False})

    def test_an_unknown_reference_or_comparison_is_refused(self):
        with pytest.raises(GateError, match="is not one of"):
            BG.gate_for(model="m", datapath="w8a8", reference_kind="vibes",
                        comparison="tolerance_and_topk", atol=1.0, rtol=1.0, output_elements=1)
        with pytest.raises(GateError, match="is not one of"):
            BG.gate_for(model="m", datapath="w8a8", reference_kind="w8a8_independent",
                        comparison="eyeball", atol=1.0, rtol=1.0, output_elements=1)


class TestScopeClaims:
    def test_a_single_step_may_not_be_declared_a_trajectory(self):
        with pytest.raises(GateError, match="must not be reported as the trajectory"):
            BG.gate_for(model="smolvla", datapath="w8a8", reference_kind="eager_same_precision",
                        comparison="trajectory", atol=1e-4, rtol=1e-4, output_elements=1600,
                        steps=1, session_key="actions")

    def test_a_trajectory_must_name_the_output_it_grades(self):
        with pytest.raises(GateError, match="must name the session output"):
            BG.gate_for(model="smolvla", datapath="w8a8", reference_kind="eager_same_precision",
                        comparison="trajectory", atol=1e-4, rtol=1e-4, output_elements=1600,
                        steps=10)

    def test_a_single_step_comparison_may_not_claim_many_steps(self):
        with pytest.raises(GateError, match="a scope claim the comparison does not support"):
            BG.gate_for(model="m", datapath="w8a8", reference_kind="w8a8_independent",
                        comparison="tolerance_and_topk", atol=1e-4, rtol=1e-4,
                        output_elements=10, steps=10)

    def test_the_smolvla_trajectory_gate_is_reachable(self):
        got = BG.gate_for(model="smolvla", datapath="w8a8", reference_kind="eager_same_precision",
                          comparison="trajectory", atol=1e-4, rtol=1e-4, output_elements=16000,
                          steps=10, session_key="actions")
        assert got.steps == 10 and got.session_key == "actions"
        assert got.reference_file == "session_goldens.npz"


class TestWhatTheHarnessMayPrint:
    def test_a_small_output_prints_its_values(self):
        assert _resnet_gate().prints_values is True

    def test_a_large_output_falls_back_to_a_digest_and_argmax(self):
        """256,000 logits over a console that is the binding constraint on gradeability."""
        got = BG.gate_for(model="tiny_llama", datapath="w8a8", reference_kind="w8a8_independent",
                          comparison="tolerance_and_topk", atol=0.03125, rtol=0.02,
                          output_elements=256000,
                          available_references={"w8a8_independent": True})
        assert got.prints_values is False
        rendered = BH._gate_check(got, output_offset=0, output_ctype="float")  # noqa: SLF001
        assert "digest_and_argmax" in rendered and "MERLIN_OUT" not in rendered

    def test_the_declaration_records_every_field_a_reader_needs(self):
        d = _resnet_gate().to_dict()
        for key in ("reference_file", "reference_scope", "reference_generator", "comparison",
                    "comparison_licence", "atol", "rtol", "prints_values", "declared"):
            assert key in d, key
        assert "not a gate" in d["declared"]


class TestTheHarnessReadsItsOutputFromThePlan:
    def _plan(self):
        buf = {"tensors": {"arg0": {"shape": [4, 4], "dtype": "i8"},
                           "Y0": {"shape": [1, 8], "dtype": "f32"},
                           "Y1": {"shape": [1, 8], "dtype": "f32"}},
               "kernel_abi": {"args": [{"tensor": "arg0", "access": "read"},
                                       {"tensor": "Y0", "access": "write"},
                                       {"tensor": "Y1", "access": "write"}]}}
        return BP.plan(buf, row_pitch_elements=16)

    def _gate(self):
        return BG.gate_for(model="m", datapath="w8a8", reference_kind="pt2e_integer",
                           comparison="exact_elementwise", atol=1e-5, rtol=1e-5,
                           output_elements=8, expected_argmax=3)

    def test_a_second_write_argument_is_read_at_ITS_offset_not_zero(self):
        """The shipped harness reads mutable_blob + 0, which is ResNet-50's layout, not the ABI."""
        got = BH.render_bundle_harness(self._plan(), self._gate(), entry_symbol="k",
                                       output_tensor="Y1")
        assert got["graded_output"]["tensor"] == "Y1"
        assert got["graded_output"]["offset"] > 0
        assert f"merlin_mutable_blob + {got['graded_output']['offset']}" in got["gate"]

    def test_an_output_that_is_not_a_write_argument_is_refused(self):
        with pytest.raises(BH.BundleHarnessError, match="correct arithmetic on the wrong bytes"):
            BH.render_bundle_harness(self._plan(), self._gate(), entry_symbol="k",
                                     output_tensor="arg0")

    def test_a_gate_over_a_different_element_count_than_the_tensor_is_refused(self):
        gate = BG.gate_for(model="m", datapath="w8a8", reference_kind="pt2e_integer",
                           comparison="exact_elementwise", atol=1e-5, rtol=1e-5,
                           output_elements=999, expected_argmax=3)
        with pytest.raises(BH.BundleHarnessError, match="padding or someone else's tensor"):
            BH.render_bundle_harness(self._plan(), gate, entry_symbol="k", output_tensor="Y0")

    def test_one_pointer_per_abi_argument_in_declaration_order(self):
        got = BH.render_bundle_harness(self._plan(), self._gate(), entry_symbol="k",
                                       output_tensor="Y0")
        assert got["n_arguments"] == 3
        assert got["call"].count("(void *)") == 3
        assert got["declarations"].count("void *") == 3

    def test_the_validation_is_one_expression_the_profile_harness_accepts(self):
        """It must be a call, because the gate's variables do not exist at validation time."""
        from merlin.perf.warm_profile_harness import (TargetInvocationHooks,
                                                      render_warm_then_measure_main)
        got = BH.render_bundle_harness(self._plan(), self._gate(), entry_symbol="k",
                                       output_tensor="Y0")
        assert got["validate"] == "merlin_gate_check()"
        source = render_warm_then_measure_main(
            prepare_input="/* x */",
            invocation=TargetInvocationHooks(got["call"], "/* fence */"),
            validate_outputs=got["validate"])
        # The gate is declared before use, and the metric follows the validation.
        assert source.index("merlin_gate_check()") < source.index("METRIC cycles")

    def test_a_bad_entry_symbol_is_refused(self):
        for bad in ("", "not an identifier", "0k"):
            with pytest.raises(BH.BundleHarnessError, match="plain C identifier"):
                BH.render_bundle_harness(self._plan(), self._gate(), entry_symbol=bad,
                                         output_tensor="Y0")


_CC = next((c for c in ("third_party/llvm-install/bin/clang", "clang", "gcc")
            if os.path.exists(c) or shutil.which(c)), None)

_STUB = """
#include <stdint.h>
#include <string.h>
static const float REF[8] = {0.f, 1.f, 2.f, 9.f, 3.f, 4.f, 5.f, 6.f};
const float merlin_reference[8] = {0.f, 1.f, 2.f, 9.f, 3.f, 4.f, 5.f, 6.f};
const unsigned char merlin_const_blob_start[512];
unsigned char merlin_mutable_blob[512];
#ifndef MERLIN_MUTATE
#define MERLIN_MUTATE 0
#endif
void k(void *a, void *b) {
  (void)a;
  float *out = (float *)b;
  memcpy(out, REF, sizeof(REF));
#if MERLIN_MUTATE == 1
  out[3] = 0.5f;
#elif MERLIN_MUTATE == 2
  out[5] = REF[5] + 1.0f;
#elif MERLIN_MUTATE == 3
  out[2] = 0.0f / 0.0f;
#endif
}
"""


@pytest.mark.skipif(_CC is None, reason="no C compiler on this host")
class TestTheGateActuallyFails:
    """MUTATION. A check that cannot fail reports success; this repo has seen that fifteen times."""

    def _build(self, tmp_path, mutate):
        from merlin.perf.warm_profile_harness import (TargetInvocationHooks,
                                                      render_warm_then_measure_main)
        buf = {"tensors": {"arg0": {"shape": [4, 4], "dtype": "i8"},
                           "Y0": {"shape": [1, 8], "dtype": "f32"}},
               "kernel_abi": {"args": [{"tensor": "arg0", "access": "read"},
                                       {"tensor": "Y0", "access": "write"}]}}
        plan = BP.plan(buf, row_pitch_elements=16)
        gate = BG.gate_for(model="m", datapath="w8a8", reference_kind="pt2e_integer",
                           comparison="exact_elementwise", atol=2e-5, rtol=2e-5,
                           output_elements=8, expected_argmax=3)
        h = BH.render_bundle_harness(plan, gate, entry_symbol="k", output_tensor="Y0")
        main = render_warm_then_measure_main(
            prepare_input="/* const blob */",
            invocation=TargetInvocationHooks(h["call"], "/* fence */"),
            validate_outputs=h["validate"], reset_after_warm="/* arena */")
        (tmp_path / "h.c").write_text(
            "#include <stdio.h>\n#include <stdint.h>\n"
            "static uint64_t read_cycles(void){ static uint64_t c; return ++c; }\n"
            + h["declarations"] + "\n" + main, encoding="utf-8")
        (tmp_path / "stub.c").write_text(_STUB, encoding="utf-8")
        binary = tmp_path / f"run{mutate}"
        build = subprocess.run(
            [_CC, "-O1", f"-DMERLIN_MUTATE={mutate}", "-Wall", "-Wextra", "-Werror",
             "-Wno-gcc-install-dir-libstdcxx", "-o", str(binary),
             str(tmp_path / "h.c"), str(tmp_path / "stub.c")],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr[:2000]
        run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
        return run.returncode, run.stdout

    def test_a_correct_output_passes_and_publishes_the_metric(self, tmp_path):
        rc, out = self._build(tmp_path, 0)
        assert rc == 0
        assert "MERLIN_GATE_RESULT bad=0 nonfinite=0 argmax=3" in out
        assert "METRIC cycles" in out

    @pytest.mark.parametrize("mutate,marker", [
        (1, "argmax=7"),          # the ranking breaks
        (2, "bad=1"),             # one element outside tolerance
        (3, "nonfinite=1"),       # a NaN
    ])
    def test_each_mutation_is_caught_and_NO_metric_is_published(self, tmp_path, mutate, marker):
        """A metric beside a failed gate is a performance number for the wrong answer."""
        rc, out = self._build(tmp_path, mutate)
        assert rc != 0, "the mutation must fail the gate"
        assert marker in out
        assert "METRIC cycles" not in out
