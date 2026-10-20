"""Explicit Phase 2 iteration on capsule-certified inputs, never final qualification.

This variant admits the same existing portfolio authoring/analysis tools while
external model validation remains pending. It cannot be promoted by changing a
saved status flag or by passing the capsule compiler to FrozenPhase1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.benchharness import hash_tree
from merlin_experiments.phase1.feedback import capsule_certification as C

SCHEMA = "global_capsule_phase1_iteration_binding_v1"


@dataclass(frozen=True)
class FrozenCapsulePhase1:
    run_dir: Path
    descriptor: Path
    submission_sha256: str
    publication_sha256: str

    def _publication(self):
        record = C.verify_phase2_handoff(self.run_dir, descriptor=self.descriptor)
        if (
            record["receipt"]["sha256"] != self.publication_sha256
            or record["submission_sha256"] != self.submission_sha256
        ):
            raise ValueError("capsule iteration differs from the independently selected publication or compiler")
        return record

    @property
    def compiler(self) -> Path:
        return Path(self._publication()["handoff"]["compiler"]["path"])

    def verify(self, baseline: Path) -> dict:
        record = self._publication()
        if hash_tree(baseline)["sha256"] != self.submission_sha256:
            raise ValueError("capsule iteration baseline differs from its frozen compiler")
        technical = C.verify(self.run_dir, descriptor=self.descriptor)
        names = (
            "capsule_publication.json",
            "capsule_certification.json",
            "environment.yaml",
            "qa_loop_summary.yaml",
            "timing_detailed.json",
            "capsule_workflow.json",
            "freeze.json",
            "grading_public/score_capsule.json",
            "grading_hidden/score_capsule.json",
        )
        return {
            "schema": SCHEMA,
            "scope": "capsule_iteration",
            "run_id": Path(self.run_dir).name,
            "run_dir": str(Path(self.run_dir).absolute()),
            "submission_sha256": self.submission_sha256,
            "descriptor": record["descriptor"],
            "capsule_publication": record["receipt"],
            "public_passed": technical["public"]["summary"]["n_passed"],
            "public_total": technical["public"]["summary"]["n_capsules"],
            "known_functional_gap_ids": [],
            "waiver_predicates": [],
            "observed_deviations": [
                {
                    "predicate": "external_full_models_pending",
                    "detail": "Capsule-only Phase 2 iteration; no final model qualification.",
                }
            ],
            "evidence_sha256": {name: C.pin(Path(self.run_dir) / name)["sha256"] for name in names},
            "qualification_action": "reopen_selected_capsule_publication_only",
            "final_results_qualified": False,
            "external_full_models_complete": False,
        }


@dataclass(frozen=True)
class FrozenQualifiedCapsulePhase1:
    """Technical capsule qualification selected only for Phase 2 iteration.

    H.verify owns the original producer paths and all observation revalidation.
    This adapter grants no author-convergence or full-model qualification claim.
    """

    run_dir: Path
    descriptor: Path
    submission_sha256: str
    qualification_path: Path
    qualification_sha256: str

    def _qualification(self):
        if C.pin(self.qualification_path)["sha256"] != self.qualification_sha256:
            raise ValueError("capsule iteration qualification differs from its selected SHA")
        from merlin_experiments.phase1.feedback import capsule_qualification_replay

        pointer = {"path": str(self.qualification_path), "sha256": self.qualification_sha256}
        record = capsule_qualification_replay.verify(pointer, descriptor=self.descriptor)
        if (
            record["original_run_dir"] != str(Path(self.run_dir).absolute())
            or record["submission_sha256"] != self.submission_sha256
        ):
            raise ValueError("capsule iteration qualification differs from its selected run or compiler")
        return record

    @property
    def compiler(self) -> Path:
        return Path(self._qualification()["handoff"]["compiler"]["path"])

    def verify(self, baseline: Path) -> dict:
        record = self._qualification()
        if hash_tree(baseline)["sha256"] != self.submission_sha256:
            raise ValueError("capsule iteration baseline differs from its qualified compiler")
        phases = record["numerical_producer_epoch"]
        return {
            "schema": SCHEMA,
            "scope": "capsule_iteration",
            "run_id": Path(self.run_dir).name,
            "run_dir": str(Path(self.run_dir).absolute()),
            "submission_sha256": self.submission_sha256,
            "descriptor": record["descriptor"],
            "capsule_qualification": record["receipt"],
            "baseline_qualification": "independently_verified_capsules_and_resources",
            "phase1_authoring_converged": False,
            "public_passed": phases["public"]["summary"]["n_passed"],
            "public_total": phases["public"]["summary"]["n_capsules"],
            "known_functional_gap_ids": [],
            "waiver_predicates": [],
            "observed_deviations": [
                {"predicate": "external_full_models_pending", "detail": "Iteration only; final models pending."}
            ],
            "qualification_action": "reopen_selected_capsule_qualification_only",
            "final_results_qualified": False,
            "external_full_models_complete": False,
        }


def _qualified_functional_run(run_dir, config, descriptor):
    from . import campaign as PC
    from . import functional_inputs as FI

    pointer = config["capsule_qualification"]
    if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"} or "capsule_publication_sha256" in config:
        raise ValueError("select exactly one capsule qualification pin or authored publication")
    path = Path(pointer["path"])
    if not path.is_absolute() or C.pin(path) != pointer:
        raise ValueError("capsule iteration requires the exact independent qualification pin")
    original_descriptor = C._read(pointer)["descriptor"]
    if C.pin(descriptor)["sha256"] != original_descriptor["sha256"]:
        raise ValueError("capsule iteration target differs from the selected qualification descriptor")
    selected = FrozenQualifiedCapsulePhase1(
        run_dir,
        Path(original_descriptor["path"]),
        str(config["functional_submission_sha256"]),
        path,
        str(pointer["sha256"]),
    )
    record = selected._qualification()
    phases = record["numerical_producer_epoch"]
    public, hidden = (C._read(phases[name]["score"]) for name in ("public", "hidden"))
    base = PC.FunctionalRun(
        run_dir,
        Path(record["handoff"]["compiler"]["path"]),
        run_dir.name,
        selected.submission_sha256,
        public["n_capsules"],
        hidden["n_capsules"],
        public,
        hidden,
        C._read(record["freeze"])["frozen_at"],
        (PC.Deviation("external_full_models_pending", "Capsule qualification permits iteration only."),),
    )
    return FI._functional_input_snapshot(base), selected


def selected_functional_run(run_root: Path, config: dict, descriptor: Path):
    """Use the existing V4 private/public input owner after strict capsule admission."""
    from . import campaign as PC
    from . import functional_inputs as FI

    if config.get("phase1_validation_scope") != "capsules" or config.get("waive_functional_gate"):
        raise ValueError("capsule iteration requires explicit scope and permits no functional waivers")
    run_id = str(config["functional_run_id"])
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ValueError("capsule iteration run ID must be one name")
    run_dir = Path(run_root).absolute() / run_id
    if "capsule_qualification" in config:
        return _qualified_functional_run(run_dir, config, descriptor)
    publication_path = run_dir / "capsule_publication.json"
    if C.pin(publication_path)["sha256"] != config.get("capsule_publication_sha256"):
        raise ValueError("capsule iteration requires the independently selected publication SHA")
    original_descriptor = json.loads(publication_path.read_bytes())["descriptor"]
    if C.pin(descriptor)["sha256"] != original_descriptor["sha256"]:
        raise ValueError("capsule iteration target differs from the selected capsule descriptor")
    selected = FrozenCapsulePhase1(
        run_dir,
        Path(original_descriptor["path"]),
        str(config["functional_submission_sha256"]),
        str(config["capsule_publication_sha256"]),
    )
    record = selected._publication()
    technical = C.verify(selected.run_dir, descriptor=descriptor)
    public = json.loads(Path(technical["public"]["score"]["path"]).read_bytes())
    hidden = json.loads(Path(technical["hidden"]["score"]["path"]).read_bytes())
    base = PC.FunctionalRun(
        selected.run_dir,
        Path(record["handoff"]["compiler"]["path"]),
        run_id,
        selected.submission_sha256,
        public["n_capsules"],
        hidden["n_capsules"],
        public,
        hidden,
        json.loads((selected.run_dir / "freeze.json").read_bytes())["frozen_at"],
        (PC.Deviation("external_full_models_pending", "Capsule iteration only; final model validation pending."),),
    )
    return FI._functional_input_snapshot(base), selected


def require_paid_iteration(inputs) -> None:
    """Admit the opt-in variant only through its live strict receipt owner."""
    binding = inputs.phase1_binding
    if binding is None:
        raise ValueError("paid macro authoring requires an existing Phase-1 qualification or capsule publication")
    owner = getattr(inputs, "phase1", None)
    owners = (FrozenCapsulePhase1, FrozenQualifiedCapsulePhase1)
    if isinstance(owner, owners) or binding.get("schema") == SCHEMA:
        if type(owner) not in owners or owner.verify(inputs.baseline) != binding:
            raise ValueError("paid capsule iteration lost its exact selected publication owner")


def launch_scope(binding) -> str:
    if binding is None:
        return "development_readiness_only"
    return "capsule_iteration" if binding.get("schema") == SCHEMA else "qualified_macro_experiment"


def checkpoint_scope(binding) -> dict:
    if binding is None or binding.get("schema") != SCHEMA:
        return {}
    result = {
        "launch_scope": "capsule_iteration",
        "final_results_qualified": False,
        "external_full_models_complete": False,
        "qualification_scope": "capsule_iteration_only",
    }
    if "capsule_qualification" in binding:
        result.update(
            baseline_qualification="independently_verified_capsules_and_resources",
            phase1_authoring_converged=False,
        )
    return result


def verify_checkpoint_scope(document) -> None:
    """Iteration checkpoints are reusable; they cannot claim a final release."""
    binding = document.get("phase1_qualification")
    if binding is None or binding.get("schema") != SCHEMA:
        if document.get("launch_scope") == "capsule_iteration" or (
            binding is not None and {"capsule_publication", "capsule_qualification"}.intersection(binding)
        ):
            raise ValueError("capsule iteration checkpoint lost its selected binding")
        if binding is not None and isinstance(binding.get("run_dir"), str):
            environment = Path(binding["run_dir"]) / "environment.yaml"
            if (
                environment.is_file()
                and (yaml.safe_load(environment.read_bytes()) or {}).get("validation_scope") == "capsules"
            ):
                raise ValueError("capsule iteration cannot masquerade as full qualification")
        return
    if any(document.get(key) != value for key, value in checkpoint_scope(binding).items()) or any(
        document.get(key) is True
        for key in ("full_model_numerics_qualified", "qualified_model_results", "final_release")
    ):
        raise ValueError("capsule iteration cannot promote itself to qualified model results or final release")
    if "capsule_qualification" in binding and document.get("authoring_converged") is True:
        raise ValueError("technical capsule qualification cannot claim author convergence")
    arguments = (Path(binding["run_dir"]), Path(binding["descriptor"]["path"]), binding["submission_sha256"])
    if "capsule_qualification" in binding:
        if "capsule_publication" in binding:
            raise ValueError("capsule iteration checkpoint has ambiguous qualification")
        pointer = binding["capsule_qualification"]
        selected = FrozenQualifiedCapsulePhase1(*arguments, Path(pointer["path"]), pointer["sha256"])
    else:
        selected = FrozenCapsulePhase1(*arguments, binding["capsule_publication"]["sha256"])
    if selected.verify(selected.compiler) != binding:
        raise ValueError("capsule iteration checkpoint publication changed")


def require_final_results(binding, *, descriptor: Path | None = None) -> None:
    """Reopen the existing full formal gate before any final model/result promotion."""
    if binding is not None and (
        binding.get("schema") == SCHEMA or {"capsule_publication", "capsule_qualification"}.intersection(binding)
    ):
        raise ValueError("external full-model qualification is required before final results or release")
    if binding is None or binding.get("schema") != "global_frozen_phase1_binding_v1" or descriptor is None:
        raise ValueError("final results require an explicit existing full qualification and target descriptor")
    from merlin.targetgen.target_experiment import load_target_experiment

    from . import functional_inputs as FI
    from .global_inputs import FrozenPhase1

    run_dir = Path(binding["run_dir"])
    selected = FrozenPhase1(
        run_dir.parent,
        run_dir.name,
        binding["submission_sha256"],
        tuple(binding["waiver_predicates"]),
        binding["public_passed"],
        binding["public_total"],
        tuple(binding["known_functional_gap_ids"]),
    )
    functional = FI.inspect_stage_functional_run(
        run_dir.parent,
        run_dir.name,
        selected.submission_sha256,
        waive=selected.waiver_predicates,
    )
    if selected.verify(functional.submission_dir) != binding:
        raise ValueError("final result qualification differs from its selected frozen baseline")
    FI.require_formal_qualification(functional, load_target_experiment(descriptor))
