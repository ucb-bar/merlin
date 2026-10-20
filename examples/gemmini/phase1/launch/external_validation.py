"""Validate a capsule-certified compiler under an independent finite model budget.

No author is started. Captures, weights, references and comparison policies are
reused byte-for-byte. Only run-owned paths, identities and the explicitly selected
external deadline are rebound. The existing full qualifier retains every model
source/offload/instruction, exact-kernel and all-declared-output gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import launch_phase1 as P
import yaml
from merlin_experiments.phase1 import launch_config as L
from merlin_experiments.phase1.feedback import capsule_certification as C

from merlin.common.digest import sha256_file


def pin(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def rebound_spec(template, capsule_spec, *, config, static_root, deadline):
    original = yaml.safe_load(Path(template).read_text())
    selected = yaml.safe_load(Path(capsule_spec).read_text())
    for key in ("schema", "target", "models", "numerics_bundle"):
        if selected.get(key) != original.get(key):
            raise ValueError(
                f"external model template differs from selected capsule host inputs: {key}"
            )
    result = copy.deepcopy(original)
    numerical = result["numerical_qualification"]
    numerical.update(config=str(config), static_root=str(static_root), deadline_unix=deadline)
    return result


def prepare(args, *, started):
    if args.prep.exists():
        raise ValueError("external validation requires a new preparation directory")
    args.prep.mkdir(parents=True, mode=0o700)
    fresh = L.build(
        L.load(args.config),
        task_root=args.task_root,
        prep=args.capsule_prep,
        run_id=args.capsule_run.name,
    )
    descriptor = fresh.path(fresh.document["inputs"]["descriptor"])
    publication = C.verify_phase2_handoff(args.capsule_run, descriptor=descriptor)
    technical = json.loads((args.capsule_run / "capsule_certification.json").read_text())
    if publication.get("phase2_iteration_ready") is not True:
        raise ValueError("external validation requires final capsule publication")
    proof = args.capsule_run / "capsule_certification.json"
    package = Path(technical["handoff"]["compiler"]["path"])
    budget_s = args.max_seconds
    if not 0 < budget_s <= fresh.document["external_validation"]["budget_s"]:
        raise ValueError("external budget exceeds its selected launch policy")
    deadline = started + budget_s
    config = args.prep / "config.yaml"
    private = args.prep / "private-models.yaml"
    doc = copy.deepcopy(fresh.document)
    if doc.get("release_config"):
        doc["release_config"] = str((fresh.config.path.parent / doc["release_config"]).resolve())
    if doc.get("errata"):
        doc["errata"]["source"] = str(
            (fresh.config.path.parent / doc["errata"]["source"]).resolve()
        )
    doc["validation_scope"] = "full"
    doc["variables"]["READINESS"] = str(args.prep)
    doc["inputs"]["private_full_model_spec"] = str(private)
    doc["full_model_verification"].update(
        readiness_receipt=str(args.prep / "readiness.json"),
        readiness_sha256="0" * 64,
        all_output_plan={"path": str(args.prep / "all-output-plan.json"), "sha256": "0" * 64},
    )
    spec = rebound_spec(
        args.private_spec_template,
        fresh.path(fresh.document["inputs"]["private_full_model_spec"]),
        config=config,
        static_root=args.prep / "native-chain",
        deadline=deadline,
    )
    private.write_text(yaml.safe_dump(spec, sort_keys=False))
    doc["runtime_pins"]["private_model_selection"] = {
        "path": str(private),
        "sha256": sha256_file(private),
    }
    config.write_text(yaml.safe_dump(doc, sort_keys=False))

    def launch():
        return L.build(L.load(config), task_root=args.task_root, prep=args.prep, run_id=args.run_id)

    selected = launch()
    F = P.operator_module("full_model_verification")
    identity = F.selected_identity(selected)
    expected = args.prep / "expected-identity.json"
    P.save(expected, identity)
    old = fresh.document["full_model_verification"]["all_output_plan"]
    original_plan = fresh.path(old["path"])
    if sha256_file(original_plan) != old["sha256"]:
        raise ValueError("selected all-output template changed")
    plan = F.read(original_plan)
    plan.pop("run_id", None)
    helper = selected.path("${REPO}/src/merlin/model_check/full_output_audit.py")
    plan.update(
        selected_identity=identity,
        expected_identity=pin(expected),
        helper=pin(helper),
        helper_source_commit=selected.variables["SOURCE_REVISION"],
        rebind_origin={
            "original_plan": pin(original_plan),
            "scope": (
                "Separate external validation of the newly capsule-certified exact compiler; "
                "no reference regeneration or author restart"
            ),
        },
    )
    P.save(args.prep / "all-output-plan.json", plan)
    doc["full_model_verification"]["all_output_plan"]["sha256"] = sha256_file(
        args.prep / "all-output-plan.json"
    )
    config.write_text(yaml.safe_dump(doc, sort_keys=False))
    selected = launch()
    original_readiness = fresh.document["full_model_verification"]
    if (
        args.infrastructure_template != fresh.path(original_readiness["readiness_receipt"])
        or sha256_file(args.infrastructure_template) != original_readiness["readiness_sha256"]
    ):
        raise ValueError("external infrastructure template differs from the selected fresh launch")
    old_ready = F.read(args.infrastructure_template)
    diagnostics = {
        name: {kind: Path(row["path"]) for kind, row in records.items()}
        for name, records in old_ready["retained_diagnostics"].items()
    }
    # Reopen actual source/runtime-bound mechanism evidence. No target rerun is
    # needed when the compiler harness, hardware, executable and bytes agree.
    with P.configured_environment(selected):
        readiness = F.build_infrastructure_receipt(
            selected,
            Path(old_ready["retained_submission"]["path"]),
            diagnostics,
            Path(old_ready["target_witness"]["path"]),
            Path(old_ready["cleanup_witness"]["path"]),
        )
    P.save(args.prep / "readiness.json", readiness)
    doc["full_model_verification"]["readiness_sha256"] = sha256_file(args.prep / "readiness.json")
    config.write_text(yaml.safe_dump(doc, sort_keys=False))
    authorization = args.prep / "authorization.json"
    P.save(
        authorization,
        {
            "authorized_by_user_text": args.authorization_text,
            "new_author_allowed": False,
            "scope": "External validation following separately completed Phase 1 capsules",
        },
    )
    budget = args.prep / "budget.json"
    P.save(
        budget,
        {
            "schema": "merlin.operator.external_validation_budget.v1",
            "scope": "external_full_model_validation_after_capsule_closure",
            "capsule_receipt": pin(proof),
            "submission_sha256": technical["submission_sha256"],
            "authorized_at_unix": started,
            "wall_budget_s": budget_s,
            "absolute_deadline_unix": deadline,
            "new_author_allowed": False,
            "authorization": pin(authorization),
        },
    )
    return launch(), package, proof, budget, deadline


def main():
    started = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "task-root",
        "prep",
        "capsule-prep",
        "capsule-run",
        "private-spec-template",
        "infrastructure-template",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--authorization-text", required=True)
    args = parser.parse_args()
    try:
        launch, package, proof, budget, deadline = prepare(args, started=started)
        qualifier = P.operator_module("qualify_phase1")
        # Keep the binder's preparation beside, not inside, the qualifier's fresh status root.
        return qualifier.run(
            launch,
            args.prep / "qualification",
            package,
            proof,
            sha256_file(proof),
            deadline,
            budget_extension=budget,
            budget_extension_sha256=sha256_file(budget),
        )
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        args.prep.mkdir(parents=True, exist_ok=True)
        P.save(
            args.prep / "external-preparation-failure.json",
            {
                "status": "refused",
                "error": str(exc),
                "external_full_models_complete": False,
                "new_author_started": False,
            },
        )
        print(json.dumps({"status": "external_validation_refused", "error": str(exc)}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
