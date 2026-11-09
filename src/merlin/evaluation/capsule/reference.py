"""Capsule evaluation: reference."""

from __future__ import annotations

from merlin.evaluation.capsule.cpu import _grade_cpu_step
from merlin.evaluation.capsule.diagnostics import _absent_outputs as _absent_outputs
from merlin.evaluation.capsule.diagnostics import _unwritten_output_detail as _unwritten_output_detail
from merlin.evaluation.capsule.numerics import _absent_output_detail
from merlin.evaluation.capsule.tier import TierResult as TierResult
from merlin.runtime.reference import reference_outputs
from merlin.runtime.simulator import simulate
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen.capsule_common import _cat
from merlin.targetgen.oot_runner import CertFailure


def check_reference(capsule, paths, policy, required, evidence, inputs):
    if inputs.independent_float:
        # The integer reference/simulate engines cannot execute a float (fp8/bf16) datapath, so the
        # integer L0-reference / L1-sim numeric floor is INAPPLICABLE — skipped honestly (not failed),
        # exactly as a non-RoCC target drops the RoCC trace gate. The real numeric grade rides the RTL
        # program-oracle output vs the independent golden.yaml (tolerance_float), scored in the oracle
        # loop below. These skips are marked not_applicable so not_run_is_not_pass does NOT flip them to
        # incomplete (only an unavailable/absent RTL oracle does).
        inputs.reference = inputs.simulation = None
        evidence.numeric = {
            "status": "skipped",
            "policy": policy.get("compare"),
            "golden_source": inputs.golden_source,
            "note": "integer reference/simulate N/A for a float datapath; correctness is graded "
            "by running your artifact on the RTL oracle and checking it computes the "
            "declared operation within tolerance",
        }
        evidence.tiers["L0"] = TierResult(
            "L0",
            "skipped",
            mandatory="L0" in required,
            not_applicable=True,
            reason="integer reference not applicable to float datapath; graded vs independent "
            f"golden ({inputs.golden_source}) at the RTL oracle",
        )
        evidence.tiers["L1"] = TierResult(
            "L1",
            "skipped",
            mandatory="L1" in required,
            not_applicable=True,
            reason="integer simulate not applicable to float datapath",
        )
    elif evidence.whole_program:
        # The command list remains available for structural checks, accelerator work accounting,
        # and decoded-trace correspondence.  It is not a semantic model of the CPU regions between
        # accelerator calls.  Calling reference_outputs here would materialize those intermediates
        # as unrelated deterministic leaves and compare a different program.  Correctness is gated
        # below by executing the complete submitted ELF at every required device tier.
        inputs.reference = inputs.simulation = None
        evidence.numeric = {
            "status": "skipped",
            "policy": policy.get("compare"),
            "golden_source": inputs.golden_source,
            "note": (
                "command-buffer reference/simulate are inapplicable to an explicitly declared "
                "whole-program kernel: its command list projects accelerator regions only; "
                "correctness is graded by executing the complete submitted ELF and comparing "
                "its declared outputs with the capsule golden"
            ),
        }
        evidence.tiers["L0"] = TierResult(
            "L0",
            "skipped",
            mandatory="L0" in required,
            not_applicable=True,
            reason="command-buffer reference cannot interpret scalar regions of a whole-program ELF",
        )
        evidence.tiers["L1"] = TierResult(
            "L1",
            "skipped",
            mandatory="L1" in required,
            not_applicable=True,
            reason="command-buffer simulate cannot execute scalar regions of a whole-program ELF",
        )
    else:
        # Interpreting the AGENT's command buffer (reference/simulate) can fail if the cb is
        # structurally invalid for the reference interpreter's op/shape model. That is the agent's
        # bug, NOT a harness crash: report it as a gradeable command_buffer failure (so the agent gets
        # feedback and both arms are scored identically) instead of a RUNNER_CRASH. The reason reports
        # the ACTUAL exception rather than asserting one hardcoded cause — the interpreter models a
        # fixed op vocabulary, so an unmodeled op/operand-shape/name is one of several possible causes.
        try:
            inputs.reference = _grade_cpu_step(reference_outputs, evidence.command_buffer)
            inputs.simulation = _grade_cpu_step(simulate, evidence.command_buffer)["outputs"]
        except (ValueError, KeyError, IndexError, TypeError) as ce:
            raise CertFailure(
                "command_buffer",
                _cat("STRUCTURAL_INVARIANT_VIOLATION"),
                f"command buffer could not be interpreted by reference/simulate "
                f"({type(ce).__name__}: {ce}); check operand ranks/shapes and that each command's op "
                f"is one the reference interpreter models (e.g. a windowed op lowered to a 2D matmul)",
            ) from ce
        nrep = _grade_cpu_step(
            CG.compare_declared_outputs,
            inputs.gold,
            inputs.reference,
            capsule["numeric_policy"],
            **inputs.output_contract,
        )
        evidence.numeric = {
            "status": nrep["status"],
            "policy": nrep["policy"],
            "max_abs_diff": nrep["max_abs_error"],
            "max_rel_error": nrep["max_rel_error"],
            "mismatch_count": nrep["mismatch_count"],
            "first_mismatch": nrep["first_mismatch"],
            # same DROPPED-store surfacing as the float path (see _absent_outputs): a missing
            # declared output is reported distinctly, never as a phantom 0-magnitude mismatch.
            "per_output": nrep.get("per_output", {}),
            "missing_outputs": _absent_outputs(nrep),
        }
        CG.write_numeric_report(paths.generated / "numeric_report.yaml", nrep)
        evidence.tiers["L0"] = TierResult(
            "L0",
            "pass" if nrep["status"] == "pass" else "fail",
            mandatory="L0" in required or True,
            reason=(
                None if nrep["status"] == "pass" else "your command buffer does not compute the declared operation"
            ),
            evidence="numeric_report.yaml",
        )
        if nrep["status"] != "pass":
            _absent_cb = _absent_output_detail(
                nrep, "command_buffer", inputs.gold, inputs.reference
            ) or _unwritten_output_detail(nrep, "command_buffer")
            raise CertFailure(
                "numeric_golden",
                _cat("FUNCTIONAL_MISMATCH"),
                _absent_cb
                or (
                    "your command buffer does not compute the declared operation "
                    f"(first divergence at index={(nrep['first_mismatch'] or {}).get('index')})"
                ),
            )

        l1_ok = (
            _grade_cpu_step(
                CG.compare_declared_outputs, inputs.reference, inputs.simulation, policy, **inputs.output_contract
            )["status"]
            == "pass"
        )
        evidence.tiers["L1"] = TierResult(
            "L1",
            "pass" if l1_ok else "fail",
            mandatory=True,
            reason=None if l1_ok else "reference(cb) != simulate(cb)",
        )
        if not l1_ok:
            raise CertFailure("command_buffer_reference", _cat("FUNCTIONAL_MISMATCH"), "reference(cb) != simulate(cb)")
