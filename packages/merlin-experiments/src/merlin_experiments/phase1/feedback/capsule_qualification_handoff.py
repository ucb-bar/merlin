"""Explicit qualification of retained capsule observations plus a new resource gate.

The original run is read-only. Numerical observations keep their original
producer and source epoch; a separately observed resource verifier supplies no
numerical credit. This route never claims author convergence or a model pass.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import yaml

from merlin.benchharness import hash_tree
from merlin.common import invocation_record as IR
from merlin.common import oot_repo as O

from . import capsule_certification as C
from . import hidden_capsule_policy as HCP
from . import public_scalability_gate as G

SCHEMA = "merlin.phase1.capsule_qualification_handoff.v1"
RECEIPT = "capsule_qualification_handoff.json"
ENV = "MERLIN_IMPORTED_CAPSULE_QUALIFICATION"


def _ordinary(path):
    path = Path(path).absolute()
    if path.resolve() != path or path.is_symlink():
        raise ValueError("capsule qualification requires ordinary selected paths")
    return path


def _evidence(run_dir, descriptor, resource_selection, resource_verdict):
    """Reopen both producer epochs independently; do not modify either one."""
    environment = yaml.safe_load((run_dir / "environment.yaml").read_bytes())
    C.verify_scope("capsules", environment)
    frozen = C._read(C.pin(run_dir / "freeze.json"))
    candidate = frozen["submission_sha256"]
    if hash_tree(run_dir / "submission")["sha256"] != candidate:
        raise ValueError("retained capsule compiler differs from its original freeze")
    hidden = HCP.resolve(descriptor, (environment.get("hidden_capsule_validation") or {}).get("mode", ""))
    HCP.verify_record(hidden, environment.get("hidden_capsule_validation"))
    phases = {
        name: C._phase(run_dir, name, descriptor=descriptor, frozen=frozen, hidden_policy=hidden)
        for name in ("public", "hidden")
    }
    names = [{row["capsule"] for row in C._read(phases[name]["score"])["per_capsule"]} for name in ("public", "hidden")]
    if names[0] & names[1] or phases["public"]["target"] != phases["hidden"]["target"]:
        raise ValueError("capsule qualification requires distinct, consistent complete rosters")
    resources = G.verify_retained_resource_observation(
        run_dir, selection_path=resource_selection, verdict_path=resource_verdict
    )
    C._resources(run_dir, resources["result"], candidate)
    history = frozen.get("oot") or {}
    if (
        Path(history.get("repo", "")) != run_dir / "oot"
        or history.get("package_digest") != candidate
        or not history.get("frozen_commit")
    ):
        raise ValueError("retained qualification has no harness-owned frozen compiler")
    O.verify(run_dir / "oot", history["frozen_commit"], candidate)
    return {
        "schema": SCHEMA,
        "validation_scope": "capsules",
        "qualification_only": True,
        "authoring_converged": False,
        "capsule_complete": True,
        "external_validation_ready": True,
        "external_full_models_complete": False,
        "formal_grade_complete": False,
        "overall_release_qualified": False,
        "phase2_iteration_ready": False,
        "target": phases["public"]["target"],
        "submission_sha256": candidate,
        "original_run_dir": str(run_dir),
        "descriptor": C.pin(descriptor),
        "freeze": C.pin(run_dir / "freeze.json"),
        "environment": C.pin(run_dir / "environment.yaml"),
        "bundle_manifest": C.pin(run_dir / "input_bundle_manifest.yaml"),
        "numerical_producer_epoch": {"source_revision": frozen["repo_sha"], **phases},
        "resource_producer_epoch": resources,
        "hidden_capsule_validation": hidden,
        "oot": {"repo": str(run_dir / "oot"), "commit": history["frozen_commit"], "package_digest": candidate},
        "verifier_source": C.pin(Path(__file__)),
    }


def _handoff(out, evidence, *, create):
    compiler = out / "compiler"
    history = evidence["oot"]
    if create:
        O.export(Path(history["repo"]), history["commit"], compiler)
        for path in (*compiler.rglob("*"), compiler):
            path.chmod(0o555 if path.is_dir() or os.access(path, os.X_OK) else 0o444)
    if hash_tree(compiler)["sha256"] != evidence["submission_sha256"] or any(
        path.is_symlink() or path.stat().st_mode & 0o222 for path in (compiler, *compiler.rglob("*"))
    ):
        raise ValueError("qualification compiler export must be exact, ordinary and read-only")
    return {
        "compiler": {"path": str(compiler), "hash_tree_sha256": evidence["submission_sha256"]},
        "mode": "independently_verified_capsule_qualification",
        "oot": history,
    }


def publish(run_dir, out, *, descriptor, resource_selection, resource_verdict):
    """Create a new qualification receipt; never write into a historical run."""
    run_dir, out, descriptor, resource_selection, resource_verdict = map(
        _ordinary, (run_dir, out, descriptor, resource_selection, resource_verdict)
    )
    if out.exists() or out.is_relative_to(run_dir) or run_dir.is_relative_to(out):
        raise ValueError("capsule qualification needs a fresh, separate owner directory")
    out.mkdir(mode=0o700, parents=True)
    evidence_path = out / "evidence.json"
    arguments = {
        "run_dir": str(run_dir),
        "descriptor": str(descriptor),
        "resource_selection": str(resource_selection),
        "resource_verdict": str(resource_verdict),
    }
    with IR.observe_call(
        out,
        stage="retained_capsule_qualification_verification",
        function=_evidence,
        arguments=arguments,
        inputs=(run_dir / "freeze.json", descriptor, resource_selection, resource_verdict),
        outputs=(evidence_path,),
        dependencies=(Path(C.__file__), Path(G.__file__)),
    ) as observed:
        evidence = _evidence(run_dir, descriptor, resource_selection, resource_verdict)
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
        observed.returned()
    record = {
        **evidence,
        "evidence": C.pin(evidence_path),
        "verification": C.pin(observed.path),
        "verified_at": datetime.now(UTC).isoformat(),
        "handoff": _handoff(out, evidence, create=True),
    }
    path = out / RECEIPT
    path.write_text(json.dumps(record, indent=2) + "\n")
    for member in (evidence_path, path):
        member.chmod(0o444)
    return verify(path, descriptor=descriptor)


def verify(path, *, descriptor):
    """Reopen actual original grading, actual new resources and the exact export."""
    path, descriptor = _ordinary(path), _ordinary(descriptor)
    record = C._read(C.pin(path))
    if path.name != RECEIPT or record.get("schema") != SCHEMA:
        raise ValueError("not an explicit capsule qualification handoff")
    resources = record["resource_producer_epoch"]
    expected = _evidence(
        Path(record["original_run_dir"]),
        descriptor,
        Path(resources["selection"]["path"]),
        Path(resources["verdict"]["path"]),
    )
    evidence = C._read(record["evidence"])
    if evidence != expected:
        raise ValueError("qualification evidence differs from its independently reopened producers")
    observed = IR.verify(Path(record["verification"]["path"]))
    C._read(record["verification"])
    arguments = {
        "run_dir": record["original_run_dir"],
        "descriptor": str(descriptor),
        "resource_selection": resources["selection"]["path"],
        "resource_verdict": resources["verdict"]["path"],
    }
    if (
        observed.get("kind") != "python_call"
        or observed.get("stage") != "retained_capsule_qualification_verification"
        or observed.get("callable") != __name__ + "._evidence"
        or observed.get("arguments") != arguments
        or observed.get("executable", {}).get("sha256") != C.pin(Path(__file__))["sha256"]
        or record["evidence"] not in observed.get("outputs", [])
    ):
        raise ValueError("qualification has no actual independent verifier observation")
    fixed = {
        key: value for key, value in record.items() if key not in {"evidence", "verification", "verified_at", "handoff"}
    }
    if fixed != expected or record["handoff"] != _handoff(path.parent, expected, create=False):
        raise ValueError("qualification handoff changed")
    completed = datetime.fromisoformat(record["verified_at"])
    if completed.tzinfo is None or completed > datetime.now(UTC):
        raise ValueError("qualification verification time is invalid")
    return {**record, "receipt": C.pin(path)}


def selected(run_dir, *, context, candidate, capsules, hidden_capsules, contract, facts):
    """Explicit unpaid external validation may reuse exact capsule observations.

    Absence retains the ordinary measured grade. A selection never authorizes an
    authoring run, a different compiler, roster, contract, facts or descriptor.
    """
    raw = os.environ.get(ENV)
    if raw is None:
        return None
    pointer = json.loads(raw)
    if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"}:
        raise ValueError("imported capsule qualification requires an exact selected pin")
    path = _ordinary(pointer["path"])
    if C.pin(path) != pointer:
        raise ValueError("imported capsule qualification changed")
    environment = yaml.safe_load((run_dir / "environment.yaml").read_bytes())
    if environment.get("qualification_only") is not True or environment.get("validation_scope", "full") != "full":
        raise ValueError("capsule observation reuse is only for unpaid external full validation")
    record = verify(path, descriptor=context.descriptor)
    if hash_tree(candidate)["sha256"] != record["submission_sha256"]:
        raise ValueError("external compiler differs from its qualified capsule compiler")
    original = Path(record["original_run_dir"])
    for phase, roots, labels in (("public", capsules, {"public", "dev"}), ("hidden", hidden_capsules, {"hidden"})):
        invocation = C._read(C.pin(original / ("grading_" + phase) / "capsule_grade_observation.json"))
        previous = C._read(invocation)
        old = C._read(previous["arguments"]["selection"])
        current = C._selection(run_dir, capsules=roots, labels=labels, context=context, contract=contract, facts=facts)
        if (
            old["target"] != current["target"]
            or old["labels"] != current["labels"]
            or old["descriptor"]["sha256"] != current["descriptor"]["sha256"]
            or sorted(row["sha256"] for row in old["roots"]) != sorted(row["sha256"] for row in current["roots"])
            or old["contract"]["sha256"] != current["contract"]["sha256"]
            or [row["sha256"] for row in old["inputs"][2:]] != [row["sha256"] for row in current["inputs"][2:]]
        ):
            raise ValueError("external capsule selection differs from its original exact observations")
    return record
