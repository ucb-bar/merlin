"""Original submitted-model source custody and grader-context assembly.

The canonical runner retains all evaluator callbacks. This helper selects no
backend or new executor and supplies no numerical, tier or timing authority.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path


def grade_selected_model(runner, capsule, *, target, package_dir, paths, runs_root, run_id,
                         cfg, contract, pkg, adapters, budget_s, timeout,
                         model_execution_selection, checked_source_diagnostic):
    from . import capsule_runner as original

    if runner is not original:
        raise ValueError("submitted model context requires its original runner owner")
    suite_pinned = capsule.get("__suite_model_pin") is runner._SUITE_MODEL_PIN
    readback_policy = runner._adapter_readback_policy(adapters)
    candidate_capsule = {key: value for key, value in capsule.items() if key != "__suite_model_pin"}
    if not suite_pinned and candidate_capsule.get("__dir__"):
        private_root = paths.run_path / ".private_model_sources"
        if private_root.resolve().is_relative_to(Path(candidate_capsule["__dir__"]).resolve()):
            raise ValueError("candidate model source snapshot cannot be nested inside its source")
        if private_root.exists() or private_root.is_symlink():
            raise ValueError(f"candidate model source snapshot already exists: {private_root}")
        private_root.mkdir(mode=0o700)
        candidate_capsule = runner._pin_model_capsule(candidate_capsule, private_root / "capsule")
    checked_scope = nullcontext()
    if checked_source_diagnostic is not None:
        from .checked_source_model_diagnostic import scoped_checked_source_diagnostic

        checked_scope = scoped_checked_source_diagnostic(checked_source_diagnostic)
    with checked_scope:
        return runner._grade_model_capsule_unlocked(
            candidate_capsule, target=target, timeout=timeout, package_dir=package_dir,
            budget_s=budget_s if checked_source_diagnostic is None else min(budget_s, timeout),
            candidate_pkg=pkg, candidate_source_pinned=runner._SUITE_MODEL_PIN,
            candidate_context={
                "runs_root": str(Path(runs_root).resolve()), "run_id": run_id,
                "suite": cfg.suite, "dtype": cfg.dtype,
                "contract": str(contract) if contract is not None else None,
                "fourth_output_name": cfg.fourth_output_name,
                "reachable_oracle_tiers": sorted(adapters), "package_prebuilt": pkg is not None,
                "owned_host_build": (json.loads(os.environ["MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION"])
                                     if os.environ.get("MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION") else None),
                **({"public_program_execution": model_execution_selection}
                   if model_execution_selection is not None else {}),
                **({"readback_policy": readback_policy.record()} if readback_policy is not None else {}),
                **({"checked_source_diagnostic": checked_source_diagnostic.record()}
                   if checked_source_diagnostic is not None else {}),
            },
        )
