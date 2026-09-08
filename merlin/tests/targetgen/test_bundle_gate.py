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


class TestTheGateIsDerivedFromTheCapturesOwnContract:
    """A session contract declares reference, key, output index, steps and a digest. Reading them is
    deriving a gate; choosing them again here would be choosing one.

    ``reference_sha256`` was in the schema and verified by NOTHING. Its subject is not the file --
    an ``.npz`` is a zip whose bytes carry timestamps -- and getting the subject wrong reports every
    capture as corrupt, which is indistinguishable from the check being broken. So the subject is
    established by agreement with every declared digest in the tree, and these tests hold it there.
    """

    def _contract(self, **over):
        base = {"version": 1, "steps": 4,
                "correctness": {"scope": "trajectory", "golden": "session_goldens.npz",
                                "key": "actions", "output_index": 0,
                                "reference": "eager_same_precision", "reference_sha256": ""},
                "quality": {"scope": "trajectory", "golden": "session_quality_fp32.npz",
                            "key": "actions", "output_index": 0,
                            "reference": "eager_fp32", "reference_sha256": ""}}
        base.update(over)
        return base

    #: The fixture writes its quality reference at correctness + 0.5, so 0.5 is the spread the
    #: derivable-tolerance rule measures. Tests that are about a DIFFERENT rule clear it rather
    #: than disabling it, so the tolerance rule stays live in all of them.
    FIXTURE_FLOOR = 0.6

    def _bundle(self, tmp_path, steps=4, per_step=(2, 3)):
        import numpy as np
        from merlin.targetgen import bundle_gate as G
        contract = self._contract()
        for field, name in (("correctness", "session_goldens.npz"),
                            ("quality", "session_quality_fp32.npz")):
            values = np.arange(steps * 6, dtype=np.float32).reshape((steps, *per_step))
            values = values + (0.5 if field == "quality" else 0.0)
            np.savez(tmp_path / name, actions=values)
            contract[field]["reference_sha256"] = G.reference_digest(tmp_path / name, "actions")
        return contract

    def test_it_builds_the_gate_the_contract_declares(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        gate = G.gate_from_session_contract(self._bundle(tmp_path), model="m", datapath="w8a8",
                                            bundle_dir=tmp_path, atol=self.FIXTURE_FLOOR,
                                            rtol=0.0)
        assert gate.reference_kind == "eager_same_precision"
        assert gate.comparison == "trajectory" and gate.steps == 4
        assert gate.session_key == "actions" and gate.output_elements == 6
        assert gate.channel == "correctness"

    def test_the_digest_subject_is_the_keyed_array_not_the_container(self, tmp_path):
        """Rewriting the same values into a fresh .npz must still verify: the zip bytes differ."""
        import numpy as np
        from merlin.targetgen import bundle_gate as G
        contract = self._bundle(tmp_path)
        with np.load(tmp_path / "session_goldens.npz") as data:
            same = np.array(data["actions"])
        (tmp_path / "session_goldens.npz").unlink()
        np.savez(tmp_path / "session_goldens.npz", actions=same, unrelated=np.zeros(3))
        assert G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                            bundle_dir=tmp_path, atol=self.FIXTURE_FLOOR,
                                            rtol=0.0)

    def test_a_golden_that_is_not_the_declared_one_is_REFUSED(self, tmp_path):
        import numpy as np
        from merlin.targetgen import bundle_gate as G
        contract = self._bundle(tmp_path)
        with np.load(tmp_path / "session_goldens.npz") as data:
            perturbed = np.array(data["actions"])
        perturbed[0, 0, 0] += np.float32(1e-3)
        np.savez(tmp_path / "session_goldens.npz", actions=perturbed)
        with pytest.raises(G.GateError) as excinfo:
            G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                         bundle_dir=tmp_path, atol=1e-4, rtol=1e-4)
        assert "is not the one this contract was written against" in str(excinfo.value)

    def test_a_contract_with_no_digest_FAILS_CLOSED(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        contract = self._bundle(tmp_path)
        contract["correctness"]["reference_sha256"] = ""
        with pytest.raises(G.GateError) as excinfo:
            G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                         bundle_dir=tmp_path, atol=1e-4, rtol=1e-4)
        assert "regenerated golden keeps its filename" in str(excinfo.value)

    def test_a_declared_step_count_the_golden_does_not_hold_is_REFUSED(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        contract = self._bundle(tmp_path)
        contract["steps"] = 10
        with pytest.raises(G.GateError) as excinfo:
            G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                         bundle_dir=tmp_path, atol=1e-4, rtol=1e-4)
        assert "partial trajectory gets published as the whole one" in str(excinfo.value)

    def test_a_reference_kind_this_module_cannot_place_is_REFUSED(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        contract = self._bundle(tmp_path)
        contract["correctness"]["reference"] = "eager_bf16_someday"
        with pytest.raises(G.GateError) as excinfo:
            G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                         bundle_dir=tmp_path, atol=1e-4, rtol=1e-4)
        assert "cannot place" in str(excinfo.value)

    def test_a_missing_channel_block_is_REFUSED_rather_than_invented(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        contract = self._bundle(tmp_path)
        contract.pop("correctness")
        with pytest.raises(G.GateError) as excinfo:
            G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                         bundle_dir=tmp_path, atol=1e-4, rtol=1e-4)
        assert "would be a gate chosen by the grader" in str(excinfo.value)

    def test_a_non_trajectory_scope_is_not_silently_regraded(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        contract = self._bundle(tmp_path)
        contract["correctness"]["scope"] = "final_step"
        with pytest.raises(G.GateError) as excinfo:
            G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                         bundle_dir=tmp_path, atol=1e-4, rtol=1e-4)
        assert "must not silently regrade a different scope" in str(excinfo.value)


class TestTheTwoChannelsAreDifferentClaims:
    """``correctness`` asks whether the device matched the compiler's precision; ``quality`` asks
    what the quantization cost against full precision. The 0.484 incident is reporting the second as
    the first, so the fp32 reference is refused on one channel and required on the other.
    """

    def _bundle(self, tmp_path):
        return TestTheGateIsDerivedFromTheCapturesOwnContract()._bundle(tmp_path)

    def test_the_quality_channel_grades_w8a8_against_fp32_and_is_ALLOWED(self, tmp_path):
        """Refusing this would refuse the measurement, not the misattribution."""
        from merlin.targetgen import bundle_gate as G
        gate = G.gate_from_session_contract(self._bundle(tmp_path), model="m", datapath="w8a8",
                                            bundle_dir=tmp_path, atol=0.6, rtol=0.0,
                                            channel="quality")
        assert gate.reference_kind == "eager_fp32" and gate.channel == "quality"
        assert "quantization cost" in gate.to_dict()["channel_asks"]

    def test_the_same_fp32_reference_on_the_CORRECTNESS_channel_is_REFUSED(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        contract = self._bundle(tmp_path)
        contract["correctness"] = contract["quality"]
        with pytest.raises(G.GateError) as excinfo:
            G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                         bundle_dir=tmp_path, atol=1e-4, rtol=1e-4)
        assert "0.484" in str(excinfo.value)

    def test_the_recorded_block_always_says_which_claim_it_is(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        contract = self._bundle(tmp_path)
        blocks = [G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                               bundle_dir=tmp_path, atol=0.6, rtol=0.0,
                                               channel=ch).to_dict()
                  for ch in ("correctness", "quality")]
        assert {b["channel"] for b in blocks} == {"correctness", "quality"}
        assert blocks[0]["channel_asks"] != blocks[1]["channel_asks"]

    def test_an_undeclared_channel_is_refused(self):
        from merlin.targetgen import bundle_gate as G
        with pytest.raises(G.GateError):
            G.gate_for(model="m", datapath="w8a8", reference_kind="w8a8_independent",
                       comparison="tolerance_and_topk", atol=1e-4, rtol=1e-4,
                       output_elements=10, channel="vibes")


class TestPrecisionNeedsTwoAxesNotOne:
    """``weight_only_int8`` is int8 weights with fp32 activations. A single "is it quantized" field
    puts it on the wrong side of one rule whichever value it takes -- and the value that reads as
    "quantized" is the one that switches OFF the 0.484 refusal, so the failure is silent AND
    permissive. These tests hold both axes and both rules.
    """

    def test_every_reference_declares_both_axes(self):
        from merlin.targetgen import bundle_gate as G
        for name, kind in G.REFERENCE_KINDS.items():
            assert kind.get("weights") in G._PRECISIONS, name
            assert kind.get("activations") in G._PRECISIONS, name

    def test_weight_only_int8_is_quantized_weights_and_full_activations(self):
        from merlin.targetgen import bundle_gate as G
        kind = G.REFERENCE_KINDS["weight_only_int8"]
        assert kind["weights"] == "quantized" and kind["activations"] == "full"

    def test_the_0484_refusal_still_fires_on_it(self):
        """The regression this de-overfit could have introduced, named."""
        from merlin.targetgen import bundle_gate as G
        with pytest.raises(G.GateError) as excinfo:
            G.gate_for(model="m", datapath="w8a8", reference_kind="weight_only_int8",
                       comparison="tolerance_and_topk", atol=1e-4, rtol=1e-4, output_elements=10)
        assert "cos 0.484" in str(excinfo.value)

    def test_an_fp32_datapath_still_refuses_it_too(self):
        """Both rules fire on the same reference, via different axes."""
        from merlin.targetgen import bundle_gate as G
        with pytest.raises(G.GateError) as excinfo:
            G.gate_for(model="m", datapath="fp32", reference_kind="weight_only_int8",
                       comparison="tolerance_and_topk", atol=1e-4, rtol=1e-4, output_elements=10)
        assert "measuring the reference's quantization" in str(excinfo.value)

    def test_a_datapath_following_reference_is_admissible_on_either_datapath(self):
        from merlin.targetgen import bundle_gate as G
        for datapath in ("w8a8", "fp32"):
            gate = G.gate_for(model="m", datapath=datapath, reference_kind="eager_same_precision",
                              comparison="trajectory", atol=1e-4, rtol=1e-4, output_elements=10,
                              steps=4, session_key="actions")
            assert gate.datapath == datapath

    def test_a_reference_missing_an_axis_is_UNKNOWN_and_refused(self, monkeypatch):
        """Adding a reference without placing it must refuse, not default to permissive."""
        from merlin.targetgen import bundle_gate as G
        kinds = dict(G.REFERENCE_KINDS)
        kinds["undeclared_someday"] = {"file": "x.npy", "quantization": "?", "scope": "?",
                                       "generator": "?"}
        monkeypatch.setattr(G, "REFERENCE_KINDS", kinds)
        with pytest.raises(G.GateError) as excinfo:
            G.gate_for(model="m", datapath="w8a8", reference_kind="undeclared_someday",
                       comparison="tolerance_and_topk", atol=1e-4, rtol=1e-4, output_elements=10)
        assert "UNKNOWN and is refused rather than assumed" in str(excinfo.value)


class TestItAgreesWithEveryContractOnDisk:
    """The acceptance test: the four captures in the tree, each graded by its own declaration.

    Every one of the eight declared digests verifies against the keyed array's contiguous float32
    bytes. That agreement is what establishes the subject -- it was not chosen.
    """

    def _contracts(self):
        from merlin.common.paths import artifacts_dir
        from merlin.common.yaml import load_yaml
        root = artifacts_dir() / "recaptures"
        if not root.is_dir():
            pytest.skip("no recaptures in this tree")
        found = []
        for path in sorted(root.rglob("session_contract.yaml")):
            contract = load_yaml(path)
            if isinstance(contract, dict) and contract.get("correctness"):
                found.append((path, contract))
        if not found:
            pytest.skip("no leaf session contracts in this tree")
        return found

    def test_every_declared_digest_verifies(self):
        from merlin.targetgen import bundle_gate as G
        for path, contract in self._contracts():
            for field in ("correctness", "quality"):
                spec = contract.get(field) or {}
                golden = path.parent / str(spec["golden"])
                assert G.reference_digest(golden, str(spec["key"])) == spec["reference_sha256"], \
                    f"{path}:{field}"

    def test_every_capture_yields_a_gate_from_its_own_DERIVED_tolerance(self):
        """The tolerance comes from the capture's own reference spread, not from a chosen number.

        A fixed 1e-4 was refused for SmolVLA, correctly: its two references differ by 5.36e-2.
        """
        from merlin.targetgen import bundle_gate as G
        for path, contract in self._contracts():
            datapath = "fp32" if "fp32" in path.parent.name or "fp32" in str(path.parent) else "w8a8"
            spread = G.reference_spread(path.parent / contract["correctness"]["golden"],
                                        path.parent / contract["quality"]["golden"],
                                        str(contract["correctness"]["key"]))
            gate = G.gate_from_session_contract(contract, model=path.parent.name,
                                                datapath=datapath, bundle_dir=path.parent,
                                                atol=spread.max_absolute, rtol=0.0)
            assert gate.steps >= 2 and gate.output_elements >= 1
            assert gate.session_key
            assert f"{spread.max_absolute:.6g}" in gate.scope_note

    def test_a_FIXED_tolerance_is_refused_wherever_the_spread_exceeds_it(self):
        """The defect this rule closes, held against the real captures."""
        from merlin.targetgen import bundle_gate as G
        refused = 0
        for path, contract in self._contracts():
            datapath = "fp32" if "fp32" in path.parent.name or "fp32" in str(path.parent) else "w8a8"
            spread = G.reference_spread(path.parent / contract["correctness"]["golden"],
                                        path.parent / contract["quality"]["golden"],
                                        str(contract["correctness"]["key"]))
            if spread.max_absolute <= 2e-4:
                continue                     # nothing to refuse for this capture
            with pytest.raises(G.GateError, match="tighter than this capture"):
                G.gate_from_session_contract(contract, model=path.parent.name,
                                             datapath=datapath, bundle_dir=path.parent,
                                             atol=1e-4, rtol=1e-4)
            refused += 1
        assert refused >= 1, "no capture on disk exercises the rule; it would be untested"


_SESSION_STUB = """
#include <stdint.h>
#include <string.h>
/* A 4-step trajectory that ADVANCES: each step's output depends on the carried timestep, so a gate
   comparing every step against step 0 fails -- which is the point. reference[s][i] = s + i. */
const float merlin_reference[12] = {0.f, 1.f, 2.f,
                                    1.f, 2.f, 3.f,
                                    2.f, 3.f, 4.f,
                                    3.f, 4.f, 5.f};
const unsigned char merlin_const_blob_start[512];
unsigned char merlin_mutable_blob[1024];
#ifndef MERLIN_MUTATE
#define MERLIN_MUTATE 0
#endif
/* args: arg0 (weight, const), arg1 (carried timestep -- working copy in mutable),
         Y0 (graded output), Y1 (next timestep) */
void k(void *a, void *t, void *y0, void *y1) {
  (void)a;
  const float step = *(const float *)t;
  float *out = (float *)y0;
  for (int i = 0; i < 3; ++i) out[i] = step + (float)i;
#if MERLIN_MUTATE == 1
  if (step >= 2.0f) out[1] += 1.0f;      /* diverges only at step 2: a late carry defect */
#elif MERLIN_MUTATE == 2
  out[2] = 0.0f / 0.0f;                  /* a NaN */
#elif MERLIN_MUTATE == 3
  *(float *)y1 = step; return;           /* the carry never advances: a stalled trajectory */
#endif
  *(float *)y1 = step + 1.0f;
}
"""


@pytest.mark.skipif(_CC is None, reason="no C compiler on this host")
class TestTheTrajectoryGateActuallyFails:
    """A session gate that graded only the last step, or graded every step against step 0, would
    pass a broken carry. Both are compiled and RUN here.

    SmolVLA's flow_denoise is 10 recurrent steps whose carried state lands in the READ-ONLY blob
    from the command buffer alone -- within one invocation the state input genuinely is a read. So
    the placement, the re-seed and the per-step reference offset are all load-bearing, and each is
    mutated below.
    """

    def _buffer(self):
        return {"tensors": {"arg0": {"shape": [4, 4], "dtype": "i8"},
                            "arg1": {"shape": [1], "dtype": "f32"},
                            "Y0": {"shape": [3], "dtype": "f32"},
                            "Y1": {"shape": [1], "dtype": "f32"}},
                "kernel_abi": {"args": [{"tensor": "arg0", "access": "read"},
                                        {"tensor": "arg1", "access": "read"},
                                        {"tensor": "Y0", "access": "write"},
                                        {"tensor": "Y1", "access": "write"}]}}

    def _plan(self):
        states = (BP.SessionState(name="timestep", input_arg=1, output_index=1),)
        return BP.plan(self._buffer(), row_pitch_elements=16, session_states=states)

    def _gate(self):
        return BG.gate_for(model="m", datapath="w8a8", reference_kind="eager_same_precision",
                           comparison="trajectory", atol=1e-6, rtol=1e-6, output_elements=3,
                           steps=4, session_key="actions")

    def _build(self, tmp_path, mutate, *, step_expression=None):
        from merlin.perf.warm_profile_harness import (TargetInvocationHooks,
                                                      render_warm_then_measure_main)
        plan, gate = self._plan(), self._gate()
        h = BH.render_bundle_harness(plan, gate, entry_symbol="k", output_tensor="Y0")
        declarations = h["declarations"]
        if step_expression is not None:
            # THE STEP-INDEXING MUTATION, applied to the REFERENCE slice only. Moving the source
            # slice too would keep the two consistent and the mutation would pass -- which is
            # itself worth knowing: the defect is a reference read at the wrong step, not a
            # step index that is uniformly wrong.
            before = "merlin_reference + (long)merlin_step *"
            assert before in declarations
            declarations = declarations.replace(
                before, f"merlin_reference + (long)({step_expression}) *")
        # ONE body for warm and measured, the re-seed between them, and the gate strictly after the
        # closing cycle read -- the ordering belongs to the profile harness, not to this module.
        main = render_warm_then_measure_main(
            prepare_input=h["reseed"],
            invocation=TargetInvocationHooks(h["call"], "/* fence */"),
            validate_outputs=h["validate"], reset_after_warm=h["reseed"])
        (tmp_path / "h.c").write_text(
            "#include <stdio.h>\n#include <stdint.h>\n#include <string.h>\n"
            "static uint64_t read_cycles(void){ static uint64_t c; return ++c; }\n"
            + declarations + "\n" + main, encoding="utf-8")
        (tmp_path / "stub.c").write_text(_SESSION_STUB, encoding="utf-8")
        binary = tmp_path / f"run{mutate}_{step_expression or 'plain'}"
        build = subprocess.run(
            [_CC, "-O1", f"-DMERLIN_MUTATE={mutate}", "-Wall", "-Wextra", "-Werror",
             "-Wno-gcc-install-dir-libstdcxx", "-o", str(binary),
             str(tmp_path / "h.c"), str(tmp_path / "stub.c")],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr[:2000]
        run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
        return run.returncode, run.stdout

    def test_an_advancing_trajectory_passes_every_step(self, tmp_path):
        rc, out = self._build(tmp_path, 0)
        assert "MERLIN_GATE_TRAJECTORY steps=4 failed=0" in out, out[-2000:]
        assert out.count("MERLIN_GATE reference=") == 4, "every step must be graded, not just one"
        assert rc == 0 and "METRIC cycles" in out

    def test_grading_every_step_against_step_0_is_CAUGHT(self, tmp_path):
        """The mutation the per-step reference offset exists to stop: every step graded against
        step 0's reference. Three of four steps disagree; step 0 rightly does not."""
        rc, out = self._build(tmp_path, 0, step_expression="0")
        assert "MERLIN_GATE_TRAJECTORY steps=4 failed=3" in out, out[-2000:]
        assert rc != 0 and "METRIC cycles" not in out

    def test_a_divergence_that_starts_at_step_2_is_CAUGHT(self, tmp_path):
        """Grading only the first step would pass this; grading only the last would miss step 2."""
        rc, out = self._build(tmp_path, 1)
        assert "MERLIN_GATE_TRAJECTORY steps=4 failed=2" in out, out[-2000:]
        assert rc != 0 and "METRIC cycles" not in out

    def test_a_nan_is_caught_on_every_step(self, tmp_path):
        rc, out = self._build(tmp_path, 2)
        assert "nonfinite=1" in out and rc != 0
        assert "METRIC cycles" not in out

    def test_a_carry_that_never_advances_is_CAUGHT(self, tmp_path):
        """A stalled session recomputes step 0 four times. Without a per-step reference this PASSES."""
        rc, out = self._build(tmp_path, 3)
        assert "MERLIN_GATE_TRAJECTORY steps=4 failed=3" in out, out[-2000:]
        assert rc != 0 and "METRIC cycles" not in out


class TestFreestandingSupportIsDeclaredNotPatchedIn:
    """A missing symbol at link time is repaired from a declared table, or refused by name.

    ResNet-50's kernel references nothing outside ``memcpy``/``memset``. SmolVLA's flow-matching
    time embedding calls ``sin``, ``cos`` and ``pow``, and newlib's ``pow`` reaches an errno write,
    so the link failed on ``__errno`` -- a symbol the curated baremetal environment does not define.
    The valuable half is the refusal: a program that needs real functionality must not link against
    a stub that returns zero.
    """

    def test_it_reads_the_symbols_the_LINKER_named(self):
        """From what the linker said, not from what the object references: `sin`, `pow` and
        `memcpy` are all referenced and all resolve."""
        report = (
            "ld: warning: has a LOAD segment with RWX permissions\n"
            "ld: libm.a(libm_a-w_pow.o): in function `pow':\n"
            "w_pow.c:(.text.pow+0xb0): undefined reference to `__errno'\n"
            "w_pow.c:(.text.pow+0xc2): undefined reference to `__errno'\n"
            "math_err.c:(.text.with_errno+0xe): undefined reference to `__errno'\n")
        assert BH.unresolved_symbols(report) == ("__errno",), "de-duplicated, in order"

    def test_a_clean_link_reports_no_symbols(self):
        assert BH.unresolved_symbols("ld: warning: RWX segment\n") == ()
        assert BH.unresolved_symbols("") == ()

    def test_several_distinct_symbols_are_all_reported(self):
        report = ("a.c:(.text+0x1): undefined reference to `__errno'\n"
                  "b.c:(.text+0x2): undefined reference to `_kill'\n"
                  "c.c:(.text+0x3): undefined reference to `__errno'\n")
        assert BH.unresolved_symbols(report) == ("__errno", "_kill")

    def test_a_declared_symbol_yields_its_definition_AND_its_justification(self):
        out = BH.render_freestanding_support(("__errno",))
        assert "int *__errno(void)" in out
        assert "never reads it" in out, "the argument travels with the definition, in the C"

    def test_an_UNDECLARED_symbol_is_REFUSED_not_stubbed(self):
        with pytest.raises(BH.BundleHarnessError) as excinfo:
            BH.render_freestanding_support(("_write", "__errno"))
        message = str(excinfo.value)
        assert "_write" in message and "no honest freestanding definition" in message
        assert "compute something other than what it declares" in message

    def test_no_symbols_means_an_explicit_no_op_not_an_empty_string(self):
        """An empty fragment spliced into a harness reads as "this step did not run"."""
        out = BH.render_freestanding_support(())
        assert out.strip() and "no shim needed" in out

    def test_every_declared_shim_carries_both_a_definition_and_a_why(self):
        for name, shim in BH.FREESTANDING_SHIMS.items():
            assert shim.get("definition"), name
            assert len(str(shim.get("why", ""))) > 80, f"{name} needs a real argument, not a label"

    @pytest.mark.skipif(_CC is None, reason="no C compiler on this host")
    def test_the_emitted_shim_COMPILES(self, tmp_path):
        source = tmp_path / "shim.c"
        source.write_text(BH.render_freestanding_support(tuple(BH.FREESTANDING_SHIMS)),
                          encoding="utf-8")
        build = subprocess.run(
            [_CC, "-std=c11", "-Wall", "-Wextra", "-Werror",
             "-Wno-gcc-install-dir-libstdcxx", "-c", str(source), "-o", str(tmp_path / "shim.o")],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr[:2000]


class TestAFarConstBlobIsAddressedAbsolutelyNotByRelocation:
    """tiny_llama's plan projects 2.237 GiB, past the PC-relative reach.

    Linking its 1.209 GiB const blob beside the code pushes ordinary symbols out of the window, and
    the failure is silent: the program links and reads the wrong bytes. The repo has solved this
    twice in linker scripts -- the blob goes to a fixed absolute address reached by a compile-time
    LITERAL, which compiles to `li` and emits no relocation at all.
    """

    def _plan(self, *, const_rows, mutable_rows):
        """Y0 is always the small graded output; the caller's mutable rows follow it.

        Keeping the graded tensor a fixed 128 elements lets these tests vary the BLOB sizes without
        also tripping the gate's element-count check, which is a different rule.
        """
        tensors = {"Y0": {"shape": [8, 16], "dtype": "i8"}}
        args = []
        for index, count in enumerate(const_rows):
            tensors[f"arg{index}"] = {"shape": [count, 16], "dtype": "i8"}
            args.append({"tensor": f"arg{index}", "access": "read"})
        args.append({"tensor": "Y0", "access": "write"})
        for index, count in enumerate(mutable_rows):
            tensors[f"Y{index + 1}"] = {"shape": [count, 16], "dtype": "i8"}
            args.append({"tensor": f"Y{index + 1}", "access": "write"})
        return BP.plan({"tensors": tensors, "kernel_abi": {"args": args}, "params": {}},
                       row_pitch_elements=16)

    def _gate(self, elements):
        return BG.gate_for(model="m", datapath="w8a8", reference_kind="w8a8_independent",
                           comparison="tolerance_and_topk", atol=1e-4, rtol=1e-4,
                           output_elements=elements, expected_argmax=0)

    def test_by_default_the_const_blob_is_a_LINKER_SYMBOL(self):
        plan = self._plan(const_rows=[64], mutable_rows=[8])
        h = BH.render_bundle_harness(plan, self._gate(128), entry_symbol="k", output_tensor="Y0")
        assert h["const_addressing"] == "linker_symbol"
        assert f"extern const unsigned char {BH.CONST_SYMBOL}[]" in h["declarations"]
        assert BH.CONST_SYMBOL in h["call"]
        assert BH.CONST_BASE_MACRO not in h["call"]

    def test_a_far_blob_is_reached_from_a_compile_time_literal(self):
        plan = self._plan(const_rows=[64], mutable_rows=[8])
        h = BH.render_bundle_harness(plan, self._gate(128), entry_symbol="k", output_tensor="Y0",
                                     const_blob_base=0x200000000)
        assert h["const_addressing"] == "absolute_literal"
        assert h["const_blob_base"] == 0x200000000
        assert BH.CONST_BASE_MACRO in h["call"]
        assert BH.CONST_SYMBOL not in h["call"], "no symbol means no relocation"
        # And the harness refuses to BUILD without the literal, rather than defaulting to zero.
        assert f"#ifndef {BH.CONST_BASE_MACRO}" in h["declarations"]
        assert "#error" in h["declarations"]

    def test_the_reseed_uses_the_SAME_addressing_as_the_arguments(self):
        """A carried state's seed is read from the const blob; two addressings would read one of
        them from an unreachable symbol."""
        tensors = {"arg0": {"shape": [4, 4], "dtype": "i8"},
                   "arg1": {"shape": [1, 8], "dtype": "f32"},
                   "Y0": {"shape": [1, 8], "dtype": "f32"},
                   "Y1": {"shape": [1, 8], "dtype": "f32"}}
        buffer = {"tensors": tensors, "params": {}, "kernel_abi": {"args": [
            {"tensor": "arg0", "access": "read"}, {"tensor": "arg1", "access": "read"},
            {"tensor": "Y0", "access": "write"}, {"tensor": "Y1", "access": "write"}]}}
        plan = BP.plan(buffer, row_pitch_elements=16,
                       session_states=(BP.SessionState("s", input_arg=1, output_index=0),))
        gate = BG.gate_for(model="m", datapath="w8a8", reference_kind="eager_same_precision",
                           comparison="trajectory", atol=1e-4, rtol=1e-4, output_elements=8,
                           steps=4, session_key="a")
        h = BH.render_bundle_harness(plan, gate, entry_symbol="k", output_tensor="Y0",
                                     const_blob_base=0x200000000)
        assert BH.CONST_BASE_MACRO in h["reseed"]
        assert BH.CONST_SYMBOL not in h["reseed"]

    def test_a_near_only_image_past_the_reach_is_REFUSED_and_names_the_remedy(self):
        # 2.5 GiB of const beside the code: the whole image must be reachable and is not.
        plan = self._plan(const_rows=[(5 * (1 << 30)) // (2 * 16)], mutable_rows=[8])
        with pytest.raises(BH.BundleHarnessError) as excinfo:
            BH.render_bundle_harness(plan, self._gate(128), entry_symbol="k", output_tensor="Y0")
        message = str(excinfo.value)
        assert "PC-relative reach" in message and "reads the wrong bytes" in message
        assert "const_blob_base" in message, "the refusal must name the remedy"

    def test_the_same_plan_with_a_far_blob_is_ACCEPTED(self):
        """Which is the point: the const bytes stop needing to be reachable."""
        plan = self._plan(const_rows=[(5 * (1 << 30)) // (2 * 16)], mutable_rows=[8])
        h = BH.render_bundle_harness(plan, self._gate(128), entry_symbol="k",
                                     output_tensor="Y0", const_blob_base=0x200000000)
        assert h["reachable_bytes"] < BH.PC_RELATIVE_REACH_BYTES
        assert h["reachable_bytes"] == plan.mutable_bytes

    def test_a_NEAR_region_past_the_reach_is_refused_even_with_a_far_blob(self):
        """A far blob does not make the mutable arena reachable, and must not appear to."""
        plan = self._plan(const_rows=[8], mutable_rows=[(5 * (1 << 30)) // (2 * 16)])
        with pytest.raises(BH.BundleHarnessError) as excinfo:
            BH.render_bundle_harness(plan, self._gate(128), entry_symbol="k",
                                     output_tensor="Y0", const_blob_base=0x200000000)
        assert "even with the const blob addressed absolutely" in str(excinfo.value)

    def test_the_compilers_own_static_arena_counts_against_the_near_region(self):
        """It is .bss in the same image, reached by relocation. Passed in, since the plan cannot
        see it -- and it is 95.3 MiB on SmolVLA, which is not a rounding error."""
        plan = self._plan(const_rows=[8], mutable_rows=[(1 << 30) // 16])
        ok = BH.render_bundle_harness(plan, self._gate(128), entry_symbol="k",
                                      output_tensor="Y0", const_blob_base=0x200000000)
        assert ok["reachable_bytes"] == plan.mutable_bytes
        with pytest.raises(BH.BundleHarnessError, match="near region"):
            BH.render_bundle_harness(plan, self._gate(128), entry_symbol="k", output_tensor="Y0",
                                     const_blob_base=0x200000000,
                                     near_additional_bytes=1 << 30)

    def test_a_negative_allowance_and_a_bad_base_are_refused(self):
        plan = self._plan(const_rows=[64], mutable_rows=[8])
        with pytest.raises(BH.BundleHarnessError):
            BH.render_bundle_harness(plan, self._gate(128), entry_symbol="k",
                                     output_tensor="Y0", near_additional_bytes=-1)
        for bad in (0, -1):
            with pytest.raises(BH.BundleHarnessError, match="positive absolute address"):
                BH.render_bundle_harness(plan, self._gate(128), entry_symbol="k",
                                         output_tensor="Y0", const_blob_base=bad)

    def test_the_reach_is_the_same_constant_the_liveness_rule_uses(self):
        """Two spellings of the ISA's window would let one path admit what the other refuses."""
        import inspect

        from merlin.liveness import preconditions
        source = inspect.getsource(preconditions.medany_span)
        assert "1 << 31" in source
        assert BH.PC_RELATIVE_REACH_BYTES == 1 << 31

    @pytest.mark.skipif(_CC is None, reason="no C compiler on this host")
    def test_a_far_harness_COMPILES_only_with_the_literal_defined(self, tmp_path):
        plan = self._plan(const_rows=[64], mutable_rows=[8])
        h = BH.render_bundle_harness(plan, self._gate(128), entry_symbol="k",
                                     output_tensor="Y0", const_blob_base=0x200000000)
        source = tmp_path / "far.c"
        source.write_text("#include <stdio.h>\n#include <string.h>\n" + h["declarations"]
                          + "\nint use(void);\nint use(void) {\n" + h["call"]
                          + "\n  return merlin_gate_check();\n}\n", encoding="utf-8")
        def build(extra):
            return subprocess.run(
                [_CC, "-std=c11", "-Wall", "-Wextra", "-Werror",
                 "-Wno-gcc-install-dir-libstdcxx", *extra, "-c", str(source),
                 "-o", str(tmp_path / "far.o")], capture_output=True, text=True)
        without = build([])
        assert without.returncode != 0, "an undefined base must not silently become zero"
        assert BH.CONST_BASE_MACRO in without.stderr
        withit = build([f"-D{BH.CONST_BASE_MACRO}=0x200000000UL"])
        assert withit.returncode == 0, withit.stderr[:2000]


class TestTheFarBlobsTwoHalvesComeFromOneNumber:
    """A far blob has two halves that must agree: the address the harness compiles into an `li`, and
    the address the linker puts the bytes at. If they disagree the program still links and still
    runs -- it reads whatever happens to be at the literal. So both come from one call.
    """

    def test_the_link_flag_and_the_compile_literal_carry_the_same_address(self):
        base = 0x200000000
        link = BH.far_blob_link_flags(base)
        compile_ = BH.far_blob_compile_flags(base)
        assert link == (f"-Wl,--section-start={BH.FAR_BLOB_SECTION}=0x200000000",)
        assert compile_ == (f"-D{BH.CONST_BASE_MACRO}=0x200000000UL",)
        assert "0x200000000" in link[0] and "0x200000000" in compile_[0]

    def test_the_assembly_places_the_blob_in_the_section_the_flag_names(self):
        asm = BH.render_far_blob_assembly(blob_path="payload/const_blob.bin")
        assert f".section {BH.FAR_BLOB_SECTION}," in asm
        assert BH.FAR_BLOB_SECTION in BH.far_blob_link_flags(0x200000000)[0]
        assert '.incbin "payload/const_blob.bin"' in asm

    def test_the_assembly_says_its_symbol_is_not_the_access_path(self):
        """Taking the symbol's address would be a relocation -- the thing the literal avoids."""
        asm = BH.render_far_blob_assembly(blob_path="b.bin")
        assert "never through this symbol" in asm

    @pytest.mark.parametrize("bad", [0, -1, 1 << 62])
    def test_a_bad_base_is_refused_or_accepted_consistently_by_BOTH(self, bad):
        """Whatever one half refuses, the other must refuse: a half-applied base is the defect."""
        link_error = compile_error = None
        try:
            BH.far_blob_link_flags(bad)
        except BH.BundleHarnessError as exc:
            link_error = str(exc)
        try:
            BH.far_blob_compile_flags(bad)
        except BH.BundleHarnessError as exc:
            compile_error = str(exc)
        assert (link_error is None) == (compile_error is None)

    def test_a_non_integer_base_is_refused_by_both(self):
        for bad in ("0x200000000", True, 1.5, None):
            with pytest.raises(BH.BundleHarnessError):
                BH.far_blob_link_flags(bad)
            with pytest.raises(BH.BundleHarnessError):
                BH.far_blob_compile_flags(bad)

    def test_a_section_name_that_is_not_one_is_refused(self):
        with pytest.raises(BH.BundleHarnessError, match="ELF section name"):
            BH.far_blob_link_flags(0x200000000, section="merlin_const_blob")

    def test_an_empty_blob_path_is_refused(self):
        with pytest.raises(BH.BundleHarnessError):
            BH.render_far_blob_assembly(blob_path="")


class TestAConsoleWithNoFloatSupportMustNotBeAskedForOne:
    """A baremetal printf implements `c s d u x l` and no float conversions.

    MEASURED, on a real 10-step SmolVLA run: the harness asked for `%.9g`, and the target's
    `vprintfmt` printed the SPECIFIER LITERALLY and then mis-consumed the varargs. 16,000 value
    lines each read `MERLIN_OUT 0 %.9g`, and the header claimed `elements=-350469331`. The verdict
    itself was sound -- the comparison is C arithmetic and `bad`/`argmax`/`digest` use integer
    conversions -- but the diagnostic carried no data, so no magnitude could be recovered and the
    run had to be repeated. A value dump that prints no values still looks like a value dump.
    """

    def _plan(self):
        return BP.plan({"tensors": {"arg0": {"shape": [1, 16], "dtype": "i8"},
                                    "Y0": {"shape": [1, 8], "dtype": "f32"}},
                        "params": {},
                        "kernel_abi": {"args": [{"tensor": "arg0", "access": "read"},
                                                {"tensor": "Y0", "access": "write"}]}},
                       row_pitch_elements=16)

    def _gate(self, **kw):
        base = dict(model="m", datapath="w8a8", reference_kind="w8a8_independent",
                    comparison="tolerance_and_topk", atol=1e-4, rtol=1e-4, output_elements=8,
                    expected_argmax=0)
        base.update(kw)
        return BG.gate_for(**base)

    def test_no_rendered_fragment_asks_printf_for_a_float(self):
        h = BH.render_bundle_harness(self._plan(), self._gate(), entry_symbol="k",
                                     output_tensor="Y0")
        for key in ("declarations", "call", "reseed", "gate"):
            BH.assert_console_portable(h[key])          # raises if it does

    def test_the_guard_catches_every_float_conversion(self):
        for bad in ("%f", "%g", "%e", "%.9g", "%.17g", "%lf", "%G", "%E", "%F"):
            with pytest.raises(BH.BundleHarnessError, match="does not implement"):
                BH.assert_console_portable(f'printf("v={bad}\\n", x);')

    def test_integer_conversions_are_allowed(self):
        BH.assert_console_portable('printf("%d %u %s %c %016llx %lu\\n", a, b, c, d, e, f);')

    def test_values_are_dumped_as_BIT_PATTERNS_with_the_step_on_every_line(self):
        h = BH.render_bundle_harness(self._plan(), self._gate(), entry_symbol="k",
                                     output_tensor="Y0")
        assert "MERLIN_GATE_MODE value_bits" in h["gate"]
        assert 'MERLIN_OUT %d %d %016llx' in h["gate"], "step, index, bits"
        assert h["dumps_values"] is True

    def test_the_tolerances_travel_as_bits_since_their_decimals_are_unprintable(self):
        h = BH.render_bundle_harness(self._plan(), self._gate(), entry_symbol="k",
                                     output_tensor="Y0")
        assert "atol_bits=%016llx" in h["gate"] and "rtol_bits=%016llx" in h["gate"]
        # and the decimal values remain recorded where they ARE readable
        assert h["gate_declaration"]["atol"] == 1e-4

    def test_the_dump_needs_no_header_it_cannot_be_sure_of(self):
        """The fragment is spliced into someone else's translation unit."""
        h = BH.render_bundle_harness(self._plan(), self._gate(), entry_symbol="k",
                                     output_tensor="Y0")
        assert "memcpy(&merlin_word" not in h["gate"], "no <string.h> dependency"
        assert "union {" not in h["gate"], "no type punning either"

    @pytest.mark.skipif(_CC is None, reason="no C compiler on this host")
    def test_the_bit_dump_round_trips_through_a_real_run(self, tmp_path):
        """Compile it, run it, and decode the printed bits back to the floats that were stored."""
        import struct
        h = BH.render_bundle_harness(self._plan(), self._gate(), entry_symbol="k",
                                     output_tensor="Y0")
        values = [0.0, 1.5, -2.25, 3.0e-8, 1234.5, -0.0, 7.0, 0.1]
        stub = ("#include <string.h>\n"
                "const float merlin_reference[8] = {"
                + ",".join(f"{v!r}f" for v in values) + "};\n"
                "const unsigned char merlin_const_blob_start[64];\n"
                "unsigned char merlin_mutable_blob[256];\n"
                "static const float SRC[8] = {" + ",".join(f"{v!r}f" for v in values) + "};\n"
                "void k(void *a, void *b) { (void)a; memcpy(b, SRC, sizeof(SRC)); }\n")
        (tmp_path / "stub.c").write_text(stub, encoding="utf-8")
        (tmp_path / "h.c").write_text(
            "#include <stdio.h>\n" + h["declarations"]
            + "\nint main(void) {\n" + h["call"] + "\n  return merlin_gate_check();\n}\n",
            encoding="utf-8")
        binary = tmp_path / "bits"
        build = subprocess.run(
            [_CC, "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-gcc-install-dir-libstdcxx",
             "-o", str(binary), str(tmp_path / "h.c"), str(tmp_path / "stub.c")],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr[:2000]
        run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
        printed = {}
        for line in run.stdout.splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[0] == "MERLIN_OUT":
                word = int(parts[3], 16)
                printed[int(parts[2])] = struct.unpack(
                    "<f", struct.pack("<I", word & 0xFFFFFFFF))[0]
        assert len(printed) == 8, run.stdout[:1000]
        for index, want in enumerate(values):
            # The stored value is a `float`, so the comparison is against its SINGLE-precision
            # round-trip -- not the Python double, which is a different number.
            as_f32 = struct.unpack("<f", struct.pack("<f", want))[0]
            assert printed[index] == as_f32, (
                f"element {index}: decoded {printed[index]!r} from bits, stored {as_f32!r}")
        assert "%.9g" not in run.stdout and "%g" not in run.stdout


class TestTheConsoleBudgetIsATotalNotAPerStepFigure:
    """`prints_values` asks about ONE step's elements, which is right for a one-shot program and
    wrong for a session: SmolVLA's 1,600 elements clear the 4,096 cap, and ten steps is 16,000
    lines. The console is the binding constraint on what is gradeable at all.
    """

    def _plan(self, elements):
        return BP.plan({"tensors": {"arg0": {"shape": [1, 16], "dtype": "i8"},
                                    "arg1": {"shape": [1, 8], "dtype": "f32"},
                                    "Y0": {"shape": [elements], "dtype": "f32"},
                                    "Y1": {"shape": [1, 8], "dtype": "f32"}},
                        "params": {},
                        "kernel_abi": {"args": [{"tensor": "arg0", "access": "read"},
                                                {"tensor": "arg1", "access": "read"},
                                                {"tensor": "Y0", "access": "write"},
                                                {"tensor": "Y1", "access": "write"}]}},
                       row_pitch_elements=16,
                       session_states=(BP.SessionState("s", input_arg=1, output_index=1),))

    def _gate(self, elements, steps):
        return BG.gate_for(model="m", datapath="w8a8", reference_kind="eager_same_precision",
                           comparison="trajectory", atol=1e-4, rtol=1e-4,
                           output_elements=elements, steps=steps, session_key="a")

    def test_a_per_step_count_under_the_cap_still_overflows_across_steps(self):
        """SmolVLA's real shape: 1,600 x 10 = 16,000 lines."""
        gate = self._gate(1600, 10)
        assert gate.prints_values, "per-step, it is under the cap -- which is the trap"
        h = BH.render_bundle_harness(self._plan(1600), gate, entry_symbol="k", output_tensor="Y0")
        assert h["dumps_values"] is False
        assert h["total_value_lines"] == 0
        assert "MERLIN_GATE_MODE digest_and_argmax" in h["gate"]
        assert "steps=%d" in h["gate"], "and it says the step count is why"

    def test_a_session_that_fits_the_total_budget_still_dumps(self):
        gate = self._gate(100, 10)
        h = BH.render_bundle_harness(self._plan(100), gate, entry_symbol="k", output_tensor="Y0")
        assert h["dumps_values"] is True and h["total_value_lines"] == 1000

    def test_the_budget_is_reported_so_a_reader_can_see_what_bound_it(self):
        gate = self._gate(1600, 10)
        h = BH.render_bundle_harness(self._plan(1600), gate, entry_symbol="k", output_tensor="Y0")
        assert h["console_line_budget"] == BH.CONSOLE_LINE_BUDGET


class TestATOLERANCEMustBeDerivableNotMerelyDeclaredEarly:
    """Declaring a tolerance before the run is the right discipline and is not sufficient.

    MEASURED. A gate for SmolVLA was declared up front at atol=rtol=1e-4 -- in good faith, and
    arbitrary. Comparing the capture's OWN two references to each other, 98.4% of elements fail that
    tolerance and the worst disagreement is 5.36e-2. The device run duly reported bad=1600/1600 on
    every step, which says nothing about the device: a gate that tight fails for any conforming
    datapath. That is this repo's cos-0.484 incident in a subtler form -- there the wrong REFERENCE
    was chosen, here the right reference with an impossible threshold.

    So the capture's own reference pair supplies the floor, and a tighter tolerance is refused with
    the numbers rather than accepted because it arrived early.
    """

    def _paths(self, tmp_path, gap):
        import numpy as np
        from merlin.targetgen import bundle_gate as G
        base = np.arange(24, dtype=np.float32).reshape((4, 6))
        np.savez(tmp_path / "session_goldens.npz", actions=base)
        np.savez(tmp_path / "session_quality_fp32.npz", actions=base + np.float32(gap))
        contract = {
            "version": 1, "steps": 4,
            "correctness": {"scope": "trajectory", "golden": "session_goldens.npz",
                            "key": "actions", "output_index": 0,
                            "reference": "eager_same_precision",
                            "reference_sha256": G.reference_digest(
                                tmp_path / "session_goldens.npz", "actions")},
            "quality": {"scope": "trajectory", "golden": "session_quality_fp32.npz",
                        "key": "actions", "output_index": 0, "reference": "eager_fp32",
                        "reference_sha256": G.reference_digest(
                            tmp_path / "session_quality_fp32.npz", "actions")}}
        return contract

    def test_the_spread_is_measured_from_the_two_references(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        self._paths(tmp_path, 0.25)
        spread = G.reference_spread(tmp_path / "session_goldens.npz",
                                    tmp_path / "session_quality_fp32.npz", "actions")
        assert spread.max_absolute == pytest.approx(0.25, rel=1e-6)
        assert spread.mean_absolute == pytest.approx(0.25, rel=1e-6)
        assert spread.elements == 24
        assert 0.0 < spread.cosine <= 1.0

    def test_a_tolerance_INSIDE_the_spread_is_refused_with_the_numbers(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        contract = self._paths(tmp_path, 0.25)
        with pytest.raises(G.GateError) as excinfo:
            G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                         bundle_dir=tmp_path, atol=1e-4, rtol=1e-4)
        message = str(excinfo.value)
        assert "tighter than this capture" in message
        assert "0.25" in message, "the refusal must carry the measured floor"
        assert "Declare a tolerance at or above" in message

    def test_a_tolerance_AT_the_spread_is_accepted(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        contract = self._paths(tmp_path, 0.25)
        gate = G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                            bundle_dir=tmp_path, atol=0.25, rtol=0.0)
        assert gate.atol == 0.25

    def test_atol_and_rtol_are_counted_TOGETHER_against_the_floor(self, tmp_path):
        """Either can carry the budget; a gate splitting it between them is not tighter."""
        from merlin.targetgen import bundle_gate as G
        contract = self._paths(tmp_path, 0.25)
        assert G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                            bundle_dir=tmp_path, atol=0.10, rtol=0.15)

    def test_the_accepted_gate_RECORDS_the_floor_it_was_checked_against(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        contract = self._paths(tmp_path, 0.25)
        gate = G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                            bundle_dir=tmp_path, atol=0.25, rtol=0.0)
        assert "own two references differ by up to 0.25" in gate.scope_note
        assert "which is the floor this tolerance was checked against" in gate.scope_note

    def test_the_rule_can_be_waived_but_only_EXPLICITLY(self, tmp_path):
        from merlin.targetgen import bundle_gate as G
        contract = self._paths(tmp_path, 0.25)
        gate = G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                            bundle_dir=tmp_path, atol=1e-4, rtol=1e-4,
                                            require_derivable_tolerance=False)
        assert gate.atol == 1e-4

    def test_a_capture_with_only_ONE_reference_cannot_be_checked_and_is_not_pretended_to_be(
            self, tmp_path):
        """No pair means no floor. The gate is still built; its scope_note simply says nothing."""
        from merlin.targetgen import bundle_gate as G
        contract = self._paths(tmp_path, 0.25)
        contract.pop("quality")
        gate = G.gate_from_session_contract(contract, model="m", datapath="w8a8",
                                            bundle_dir=tmp_path, atol=1e-9, rtol=0.0)
        assert "floor this tolerance was checked against" not in gate.scope_note

    def test_references_of_different_shapes_yield_no_number(self, tmp_path):
        import numpy as np
        from merlin.targetgen import bundle_gate as G
        np.savez(tmp_path / "a.npz", actions=np.zeros((4, 6), dtype=np.float32))
        np.savez(tmp_path / "b.npz", actions=np.zeros((4, 7), dtype=np.float32))
        with pytest.raises(G.GateError, match="not a number about this model"):
            G.reference_spread(tmp_path / "a.npz", tmp_path / "b.npz", "actions")

    def test_an_absent_key_is_refused(self, tmp_path):
        import numpy as np
        from merlin.targetgen import bundle_gate as G
        np.savez(tmp_path / "a.npz", other=np.zeros(4, dtype=np.float32))
        np.savez(tmp_path / "b.npz", other=np.zeros(4, dtype=np.float32))
        with pytest.raises(G.GateError, match="no spread can be measured"):
            G.reference_spread(tmp_path / "a.npz", tmp_path / "b.npz", "actions")

    def test_the_real_smolvla_floor_is_the_number_that_refuted_my_gate(self):
        from merlin.common.paths import artifacts_dir
        from merlin.targetgen import bundle_gate as G
        capture = (artifacts_dir() / "recaptures" / "smolvla_int8_w8a8_consistent"
                   / "stages" / "flow_denoise")
        if not capture.is_dir():
            pytest.skip("no smolvla recapture in this tree")
        spread = G.reference_spread(capture / "session_goldens.npz",
                                    capture / "session_quality_fp32.npz", "actions")
        assert spread.elements == 16000
        assert spread.max_absolute > 1e-4, "the floor must exceed the tolerance it refuted"
        assert spread.cosine > 0.999, "and the two references are otherwise in close agreement"
