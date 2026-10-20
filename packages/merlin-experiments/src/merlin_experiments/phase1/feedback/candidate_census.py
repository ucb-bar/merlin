"""Optional bounded census of a fresh, numerically screened own executable."""

from __future__ import annotations

from pathlib import Path

from merlin_experiments.phase1.feedback.candidate_observations import observe_generated_artifacts


def unknown(reason):
    return {"schema": "merlin.dynamic_census.v1", "status": "UNKNOWN", "reason": reason,
            "timing_cycles": None, "physical_traffic_bytes": None, "array_utilization": None}


def observe_passed_census(root, *, target, compiler, remaining_s, numerical_pass,
                         descriptor=None, policy=None):
    """Missing original ELF/sandbox ownership executes nothing; no raw fallback."""
    if not numerical_pass:
        return unknown("original functional numerical screen did not pass")
    if remaining_s < 1:
        return unknown("bounded census collection deadline exhausted")
    try:
        from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME
        from merlin_experiments.phase2 import campaign as PC
        from merlin_experiments.phase2.census_launch import observe_dynamic_census

        images = [path for path in Path(root).rglob(PACKAGE_ELF_NAME) if path.is_file() and not path.is_symlink()]
        if len(images) != 1:
            return unknown("no unique original own executable is available")
        directory = Path(root) / "dynamic_census"
        if policy is None:
            if descriptor is None:
                return unknown("original selected package sandbox is unavailable")
            from merlin.targetgen.target_experiment import load_target_experiment

            selected = load_target_experiment(descriptor)
            if selected.target != target:
                return unknown("original descriptor and target selections disagree")
            policy = PC.package_sandbox_policy(selected, Path(root), Path(compiler))
        return observe_dynamic_census(images[0], target=target, directory=directory,
                                      remaining_s=remaining_s, policy=policy)
    except Exception as error:  # noqa: BLE001 -- optional census cannot replace original numerical gates
        return unknown("original boxed census unavailable: " + type(error).__name__)


def attach_candidate_census(row, root, *, result, context, compiler, remaining_s):
    """Ordinary selfcheck projection, distinct from private runner-owned model artifacts."""
    row["candidate_diagnostics"] = observe_generated_artifacts(root, compiler=compiler)
    passed = (result.get("tiers", {}).get("L2") or {}).get("status") == "pass"
    row["dynamic_census"] = observe_passed_census(
        root, target=context.target, compiler=compiler, descriptor=context.descriptor,
        remaining_s=remaining_s, numerical_pass=passed,
    )
