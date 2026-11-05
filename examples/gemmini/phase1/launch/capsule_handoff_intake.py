"""Admit a certified capsule artifact into a separately budgeted model job."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from merlin.common.digest import sha256_file


def ordinary_pin(record):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("external qualification requires an exact input pin")
    path = Path(record["path"])
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != record["sha256"]
    ):
        raise ValueError("external qualification input changed or is indirect")
    return path


def admit(F, proof_path, proof_sha256, submission, deadline, *, budget_path, budget_sha256):
    """Verify actual capsule proof; this grants no model or numerical pass credit."""
    from merlin_experiments.phase1.feedback import capsule_certification

    proof_path = ordinary_pin({"path": str(proof_path), "sha256": proof_sha256})
    original = json.loads(proof_path.read_text())
    descriptor = ordinary_pin(original["descriptor"])
    from merlin_experiments.phase1.feedback import capsule_qualification_handoff as Q

    qualification = original.get("schema") == Q.SCHEMA
    if qualification:
        verified = Q.verify(proof_path, descriptor=descriptor)
        publication = {"receipt": verified["receipt"]}
        ready = (
            verified.get("qualification_only") is True
            and verified.get("authoring_converged") is False
            and verified.get("external_validation_ready") is True
        )
    else:
        if proof_path.name != "capsule_certification.json":
            raise ValueError("capsule qualification needs the original capsule receipt")
        publication = capsule_certification.verify_phase2_handoff(proof_path.parent, descriptor=descriptor)
        if publication.get("capsule_certification") != {"path": str(proof_path), "sha256": proof_sha256}:
            raise ValueError("final publication selects another capsule certificate")
        verified = json.loads(proof_path.read_text())
        ready = publication.get("phase2_iteration_ready") is True
    if (
        verified.get("capsule_complete") is not True
        or not ready
        or verified.get("formal_grade_complete") is not False
        or verified.get("external_full_models_complete") is not False
    ):
        raise ValueError("capsule handoff is not independently ready for the next phase")
    package = F.package_identity(Path(submission))
    selected = verified["handoff"]["compiler"]
    if (
        Path(selected["path"]) != Path(submission)
        or selected["hash_tree_sha256"] != package["hash_tree_sha256"]
        or verified["submission_sha256"] != package["hash_tree_sha256"]
    ):
        raise ValueError("external compiler differs from the certified capsule artifact")
    budget_path = ordinary_pin({"path": str(budget_path), "sha256": budget_sha256})
    budget = json.loads(budget_path.read_text())
    keys = {
        "schema",
        "scope",
        "capsule_receipt",
        "submission_sha256",
        "authorized_at_unix",
        "wall_budget_s",
        "absolute_deadline_unix",
        "new_author_allowed",
        "authorization",
    }
    deferred = budget.get("schema") == "merlin.operator.external_validation_budget.v2"
    if deferred:
        keys.add("preparation_budget_s")
    if not isinstance(budget, dict) or set(budget) != keys:
        raise ValueError("external budget has an invalid closed schema")
    times = (budget["authorized_at_unix"], budget["absolute_deadline_unix"], deadline)
    if any(type(n) not in (int, float) or not math.isfinite(n) for n in times):
        raise ValueError("external budget requires finite explicit times")
    authorization = ordinary_pin(budget["authorization"])
    authorized = json.loads(authorization.read_text())
    if (
        budget["schema"]
        not in {"merlin.operator.external_validation_budget.v1", "merlin.operator.external_validation_budget.v2"}
        or budget["scope"] != "external_full_model_validation_after_capsule_closure"
        or budget["capsule_receipt"] != {"path": str(proof_path), "sha256": proof_sha256}
        or budget["submission_sha256"] != package["hash_tree_sha256"]
        or budget["new_author_allowed"] is not False
        or type(budget["wall_budget_s"]) is not int
        or budget["wall_budget_s"] <= 0
        or not 0 < budget["authorized_at_unix"] <= time.time() < deadline
        or deadline != budget["absolute_deadline_unix"]
        or (not deferred and deadline > budget["authorized_at_unix"] + budget["wall_budget_s"])
        or authorized.get("new_author_allowed") is not False
        or not authorized.get("authorized_by_user_text")
    ):
        raise ValueError("external validation differs from its artifact, authorization or budget")
    if deferred and (
        type(budget["preparation_budget_s"]) is not int
        or not 0 < budget["preparation_budget_s"] <= 3600
        or budget["wall_budget_s"] > 7200
        or not qualification
        or time.time() >= budget["authorized_at_unix"] + budget["preparation_budget_s"]
        or deadline != budget["authorized_at_unix"] + budget["preparation_budget_s"] + budget["wall_budget_s"]
    ):
        raise ValueError("external execution budget must start after separately bounded preparation")
    for record in (
        budget["capsule_receipt"],
        budget["authorization"],
        {"path": str(budget_path), "sha256": budget_sha256},
    ):
        ordinary_pin(record)
    return package, {
        "path": str(budget_path),
        "sha256": budget_sha256,
        "authorization": budget["authorization"],
        "capsule_receipt": budget["capsule_receipt"],
        "phase2_publication": publication["receipt"],
        "capsule_complete": True,
        "handoff_kind": "capsule_qualification" if qualification else "authored_capsule_publication",
        "new_author_allowed": False,
        "external_deadline_unix": deadline,
        "phase1_clock_preserved": True,
        **(
            {
                "execution_budget_s": budget["wall_budget_s"],
                "preparation_deadline_unix": budget["authorized_at_unix"] + budget["preparation_budget_s"],
                "authorized_ceiling_unix": deadline,
            }
            if deferred
            else {}
        ),
    }
