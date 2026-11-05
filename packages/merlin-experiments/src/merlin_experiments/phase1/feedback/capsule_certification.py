"""Prospective capsule-only certification; no whole-model or convergence authority.

Only the formal grader publishes this receipt, after two observed grades of the
same package. Consumers reopen the observations, selected inputs and frozen OOT
export instead of admitting a saved success flag. Hidden scores stay host-only.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import yaml

from merlin.benchharness import hash_tree
from merlin.common import invocation_record as IR
from merlin.common import oot_repo as O

from . import hidden_capsule_policy as HCP

SCHEMA = "merlin.phase1.capsule_certification.v1"
SCOPES = ("full", "capsules")


def pin(path: Path) -> dict:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"capsule certification requires a regular file: {path}")
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    if any(
        getattr(before, field) != getattr(after, field)
        for field in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_mode",
        )
    ):
        raise ValueError("capsule certification member changed during its read")
    return {"path": str(path), "sha256": digest}


def _read(pin_record: dict) -> dict:
    if pin(Path(pin_record["path"])) != pin_record:
        raise ValueError("capsule certification member changed")
    return json.loads(Path(pin_record["path"]).read_bytes())


def verify_scope(scope: str, environment: dict) -> None:
    """A new argv cannot downgrade a full run or reinterpret historical evidence."""
    if scope not in SCOPES or environment.get("validation_scope", "full") != scope:
        raise ValueError("validation scope differs from the admitted run; start a fresh run")


def _selection(run_dir, *, capsules, labels, context, contract, facts) -> dict:
    roots = [Path(root).absolute() for root in str(capsules).split(",") if root.strip()]
    if not roots or any(root.is_symlink() or not root.is_dir() for root in roots):
        raise ValueError("capsule selection has no direct, nonempty input roots")
    inputs = [run_dir / "environment.yaml", run_dir / "input_bundle_manifest.yaml", context.descriptor]
    if facts is not None:
        inputs.append(Path(facts))
    contract_path = Path(contract or context.repo / "merlin/contract").absolute()
    return {
        "target": context.target,
        "descriptor": pin(context.descriptor),
        "candidate_sha256": hash_tree(run_dir / "submission")["sha256"],
        "package": str((run_dir / "submission").absolute()),
        "labels": sorted(labels),
        "roots": [{"path": str(root), "sha256": hash_tree(root)["sha256"]} for root in roots],
        "contract": {"path": str(contract_path), "sha256": hash_tree(contract_path)["sha256"]},
        "inputs": [pin(path) for path in inputs],
    }


def grade(run_dir: Path, *, capsules, labels, context, contract, facts, timeout_s, hidden_mode="rtl") -> dict:
    """Observe the actual existing grader, preserving its tier and evidence policy."""
    from . import formal

    run_dir = Path(run_dir).absolute()
    phase = "hidden" if labels == {"hidden"} else "public"
    root = run_dir / ("grading_" + phase)
    root.mkdir(exist_ok=True)
    score_path, selection_path = root / "score_capsule.json", root / "capsule_grade_selection.json"
    if score_path.exists() or selection_path.exists():
        raise ValueError("capsule grade refuses stale prior scores; use a fresh run")
    selected = _selection(run_dir, capsules=capsules, labels=labels, context=context, contract=contract, facts=facts)
    selection_path.write_text(json.dumps(selected, indent=2) + "\n")
    arguments = {
        "selection": pin(selection_path),
        "timeout_s": timeout_s,
        "hidden_mode": hidden_mode,
        "no_oracle": False,
    }
    with IR.observe_call(
        root,
        stage="phase1_capsule_" + phase,
        function=grade,
        arguments=arguments,
        inputs=(selection_path, *(Path(item["path"]) for item in selected["inputs"])),
        outputs=(score_path,),
        dependencies=(Path(formal.__file__), Path(formal.CG.__file__), Path(formal.CR.__file__), Path(HCP.__file__)),
    ) as observation:
        score = formal._score(
            selected["package"],
            capsules,
            str(root),
            labels,
            False,
            context=context,
            contract=contract,
            step_timeout_s=timeout_s,
            hidden_capsule_mode=hidden_mode,
        )
        if selected != _selection(
            run_dir,
            capsules=capsules,
            labels=labels,
            context=context,
            contract=contract,
            facts=facts,
        ):
            raise ValueError("candidate or selected capsule inputs changed during grading")
        score_path.write_text(json.dumps(score, indent=2) + "\n")
        observation.returned()
    (root / "capsule_grade_observation.json").write_text(json.dumps(pin(observation.path), indent=2) + "\n")
    return score


def _phase(run_dir: Path, name: str, *, descriptor: Path, frozen: dict, hidden_policy: dict) -> dict:
    from .formal import _phase_manifest

    root = run_dir / ("grading_" + name)
    score_pin = pin(root / "score_capsule.json")
    score = _read(score_pin)
    invocation_pin = json.loads((root / "capsule_grade_observation.json").read_bytes())
    _read(invocation_pin)
    observed = IR.verify(Path(invocation_pin["path"]))
    if (
        observed.get("kind") != "python_call"
        or observed.get("callable") != __name__ + ".grade"
        or observed.get("stage") != "phase1_capsule_" + name
        or (observed.get("executable") or {}).get("sha256") != pin(Path(__file__))["sha256"]
        or score_pin not in observed.get("outputs", [])
    ):
        raise ValueError("capsule score has no actual selected grading observation")
    arguments = observed["arguments"]
    if arguments.get("no_oracle") is not False or arguments.get("hidden_mode") != (
        hidden_policy["mode"] if name == "hidden" else "rtl"
    ):
        raise ValueError("capsule grade used a different oracle policy")
    selection = _read(arguments["selection"])
    if arguments["selection"] not in observed["inputs"]:
        raise ValueError("capsule selection was not observed at the actual grade")
    for item in selection["inputs"]:
        _read_bytes = pin(Path(item["path"]))
        if _read_bytes != item or item not in observed["inputs"]:
            raise ValueError("capsule grade selection input changed")
    if (
        selection.get("descriptor") not in selection["inputs"]
        or selection["descriptor"]["sha256"] != pin(descriptor)["sha256"]
    ):
        raise ValueError("capsule grade belongs to another descriptor")
    for tree in (*selection["roots"], selection["contract"]):
        if hash_tree(Path(tree["path"]))["sha256"] != tree["sha256"]:
            raise ValueError("capsule roster or contract changed")
    if (
        selection["candidate_sha256"] != frozen["submission_sha256"]
        or selection["package"] != str(run_dir / "submission")
        or score.get("package") != selection["package"]
        or selection["labels"] != (["hidden"] if name == "hidden" else ["dev", "public"])
    ):
        raise ValueError("capsule phase was graded against another frozen package or roster")
    functional = name == "hidden" and hidden_policy["mode"] == "functional_spike"
    summary = _phase_manifest(
        score,
        required_tier=hidden_policy["required_tier"] if name == "hidden" else "L3",
        functional_hidden=functional,
    )
    if summary["formal_complete"] is not True or score.get("integrity_exempt") is not False:
        raise ValueError(f"{name} capsule phase incomplete: {summary['completion_failures']}")
    rows = score.get("per_capsule")
    if (
        not isinstance(rows, list)
        or len(rows) != score["n_capsules"]
        or any(
            not isinstance(row, dict)
            or row.get("status") != "pass"
            or row.get("label") not in selection["labels"]
            or not isinstance(row.get("capsule"), str)
            or not row["capsule"]
            for row in rows
        )
    ):
        raise ValueError("capsule score lacks the complete actual passing row roster")
    names = [row["capsule"] for row in rows]
    name_hash = hashlib.sha256(json.dumps(sorted(names), separators=(",", ":")).encode()).hexdigest()
    if len(set(names)) != len(names) or name_hash != score["cohort_admission"]["admitted_name_set_sha256"]:
        raise ValueError("capsule rows differ from the selected admitted-name roster")
    required_tier = hidden_policy["required_tier"] if name == "hidden" else "L3"
    for row in rows:
        if row.get("validated_refusal") is True and row.get("native_executed") is False:
            continue  # Existing grader binds these to the original must-refuse declaration.
        if (
            row.get("numeric") != "pass"
            or (row.get("tiers") or {}).get(required_tier) != "pass"
            or row.get("cert_verdict") is not None
        ):
            raise ValueError("capsule row lacks actual required-tier numeric certification")
    refusals = sum(row.get("validated_refusal") is True for row in rows)
    if refusals != (score.get("expected_refusal_scope") or {}).get("n_validated_refusals", 0):
        raise ValueError("capsule row refusal applicability differs from the selected grader")
    environment = yaml.safe_load((run_dir / "environment.yaml").read_bytes())
    scope = environment.get("task_scope") or {}
    expected = scope.get("held_out_capsules" if name == "hidden" else "required_public_dev_capsules")
    declared = yaml.safe_load(descriptor.read_bytes()).get("grading") or {}
    cohort = declared.get("hidden_capability_admission" if name == "hidden" else "expected_cohort") or {}
    expected = cohort.get("admitted_capsules", expected)
    if type(expected) is not int or expected <= 0 or score.get("n_capsules") != expected:
        raise ValueError("capsule score does not cover the complete selected admitted roster")
    if "source_capsules" in cohort and score["cohort_admission"]["n_source_capsules"] != cohort["source_capsules"]:
        raise ValueError("capsule source roster differs from descriptor")
    return {
        "score": score_pin,
        "invocation": invocation_pin,
        "summary": summary,
        "target": selection["target"],
        "descriptor": selection["descriptor"],
    }


def _resources(run_dir: Path, record: dict, candidate_sha256: str) -> dict:
    from merlin.targetgen.public_codegen_budget import resources_complete

    from . import compiler_profile_runtime, public_scalability_gate

    owner = compiler_profile_runtime.functional_service(run_dir)
    policy = public_scalability_gate.selected_policy(owner) if owner is not None else None
    if policy is None:
        if record != {"status": "not_selected", "required": False, "resource_complete": False}:
            raise ValueError("public resource selection differs from the admitted profile")
        return record
    if record.get("selection") != policy or record.get("compiler_tool_profile_sha256") != owner.profile["sha256"]:
        raise ValueError("public resource selection differs from the admitted profile")
    if record.get("required") is not True or record.get("resource_complete") is not True:
        raise ValueError("public capsule resource gate incomplete")
    if record.get("candidate_sha256") != candidate_sha256:
        raise ValueError("public capsule resource gate belongs to another candidate")
    report = record.get("report")
    if not isinstance(report, dict):
        raise ValueError("public capsule resource gate has no retained complete report")
    actual = _read(report)
    if (
        not resources_complete(actual, policy)
        or actual.get("candidate_sha256") != candidate_sha256
        or actual.get("compiler_tool_profile_sha256") != owner.profile["sha256"]
    ):
        raise ValueError("public capsule resource gate has no retained complete report")
    return record


def _handoff(run_dir: Path, frozen: dict, *, create: bool) -> dict:
    history = frozen.get("oot") or {}
    repo, commit = Path(history.get("repo", "")), history.get("frozen_commit")
    if repo != run_dir / "oot" or not commit or history.get("package_digest") != frozen["submission_sha256"]:
        raise ValueError("capsule handoff requires the live harness-owned frozen OOT commit")
    O.verify(repo, commit, frozen["submission_sha256"])
    root = run_dir / "capsule_handoff"
    compiler = root / "compiler"
    record = {
        "schema": "merlin.phase1.capsule_handoff.v1",
        "compiler": {"path": str(compiler), "hash_tree_sha256": frozen["submission_sha256"]},
        "oot": {"repo": str(repo), "commit": commit, "package_digest": frozen["submission_sha256"]},
        "freeze": pin(run_dir / "freeze.json"),
        "mode": "harness_frozen_capsule_compiler",
        "external_full_models_complete": False,
    }
    manifest = root / "handoff.json"
    if create:
        root.mkdir(mode=0o700)
        O.export(repo, commit, compiler)
        for path in compiler.rglob("*"):
            path.chmod(0o555 if path.is_dir() or os.access(path, os.X_OK) else 0o444)
        compiler.chmod(0o555)
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        manifest.chmod(0o444)
    if _read(pin(manifest)) != record or hash_tree(compiler)["sha256"] != frozen["submission_sha256"]:
        raise ValueError("capsule compiler export differs from the frozen OOT commit")
    if any(path.is_symlink() or path.stat().st_mode & 0o222 for path in (compiler, *compiler.rglob("*"))):
        raise ValueError("capsule compiler export must be direct and read-only")
    return {**record, "manifest": pin(manifest)}


def _evidence(run_dir: Path, descriptor: Path, resources: dict, *, create: bool) -> dict:
    environment = yaml.safe_load((run_dir / "environment.yaml").read_bytes())
    verify_scope("capsules", environment)
    frozen = json.loads((run_dir / "freeze.json").read_bytes())
    if hash_tree(run_dir / "submission")["sha256"] != frozen["submission_sha256"]:
        raise ValueError("capsule submission changed after freeze")
    policy = HCP.resolve(descriptor, (environment.get("hidden_capsule_validation") or {}).get("mode", ""))
    HCP.verify_record(policy, environment.get("hidden_capsule_validation"))
    phases = {
        name: _phase(run_dir, name, descriptor=descriptor, frozen=frozen, hidden_policy=policy)
        for name in ("public", "hidden")
    }
    names = [{row["capsule"] for row in _read(phases[name]["score"])["per_capsule"]} for name in ("public", "hidden")]
    if names[0] & names[1]:
        raise ValueError("public and hidden capsule identities must be distinct")
    from .public_scalability import verify_issued

    scalability = verify_issued(
        run_dir / "grading_public_scalability/issued.json", descriptor=descriptor, candidate=run_dir / "submission"
    )
    return {
        "schema": SCHEMA,
        "validation_scope": "capsules",
        "target": phases["public"]["target"],
        "capsule_complete": True,
        **({"public_scalability": scalability} if isinstance(scalability, dict) else {}),
        "formal_grade_complete": False,
        "external_full_models_complete": False,
        "phase2_iteration_ready": False,
        "submission_sha256": frozen["submission_sha256"],
        "descriptor": phases["public"]["descriptor"],
        "freeze": pin(run_dir / "freeze.json"),
        **phases,
        "hidden_capsule_validation": policy,
        "public_codegen_resources": _resources(run_dir, resources, frozen["submission_sha256"]),
        "handoff": _handoff(run_dir, frozen, create=create),
    }


def publish(run_dir: Path, *, descriptor: Path, resources: dict) -> dict:
    """Publish only a complete capsule gate, before any external model waits."""
    run_dir, descriptor = Path(run_dir).absolute(), Path(descriptor).absolute()
    record = _evidence(run_dir, descriptor, resources, create=True)
    completed = datetime.now(UTC)
    environment = yaml.safe_load((run_dir / "environment.yaml").read_bytes())
    state = run_dir / "qa_loop_state.yaml"
    cumulative = ((yaml.safe_load(state.read_bytes()) or {}).get("cumulative") or {}) if state.is_file() else {}
    started = cumulative.get("started_at")
    elapsed = (completed - datetime.fromisoformat(started)).total_seconds() if started else None
    record["capsule_clock"] = {
        "author_started_at": started,
        "author_state": pin(state) if state.is_file() else None,
        "capsule_completed_at": completed.isoformat(),
        "wall_seconds": elapsed,
        "authoring_only_cost_clock_unchanged": True,
        "run_prepared_at": environment.get("started_at"),
    }
    path = run_dir / "capsule_certification.json"
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    path.chmod(0o444)
    verify(run_dir, descriptor=descriptor)
    manifest = {
        "run_id": run_dir.name,
        "validation_scope": "capsules",
        "submission_sha256": record["submission_sha256"],
        "public_dev": record["public"]["summary"],
        "hidden": record["hidden"]["summary"],
        "hidden_capsule_validation": record["hidden_capsule_validation"],
        "capsule_certification": pin(path),
        "capsule_clock": record["capsule_clock"],
        "completion": {
            "capsule_complete": True,
            "formal_grade_complete": False,
            "external_full_models_complete": False,
            "phase2_iteration_ready": False,
        },
    }
    (run_dir / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (run_dir / "final_report.md").write_text(
        "Capsule validation passed for the frozen compiler. External full-model validation is pending.\n"
    )
    return record


def verify(run_dir: Path, *, descriptor: Path) -> dict:
    """Reopen strict capsule evidence and exported compiler; never infer convergence."""
    run_dir, descriptor = Path(run_dir).absolute(), Path(descriptor).absolute()
    path = run_dir / "capsule_certification.json"
    record = _read(pin(path))
    expected = _evidence(run_dir, descriptor, record.get("public_codegen_resources") or {}, create=False)
    if {key: value for key, value in record.items() if key != "capsule_clock"} != expected:
        raise ValueError("capsule certification differs from its actual frozen evidence")
    clock = record.get("capsule_clock") or {}
    state = run_dir / "qa_loop_state.yaml"
    if clock.get("author_state") != (pin(state) if state.is_file() else None):
        raise ValueError("capsule author clock state changed")
    started = (
        ((yaml.safe_load(state.read_bytes()) or {}).get("cumulative") or {}).get("started_at")
        if state.is_file()
        else None
    )
    if clock.get("author_started_at") != started:
        raise ValueError("capsule author clock differs from its original admitted state")
    completed = datetime.fromisoformat(clock["capsule_completed_at"])
    if completed.tzinfo is None or completed > datetime.now(UTC):
        raise ValueError("capsule completion clock is invalid")
    if clock.get("author_started_at") is not None:
        start = datetime.fromisoformat(clock["author_started_at"])
        if (
            start.tzinfo is None
            or start > completed
            or clock.get("wall_seconds") != (completed - start).total_seconds()
        ):
            raise ValueError("capsule wall clock is invalid")
    return {**record, "receipt": pin(path)}


def _publication_inputs(run_dir, descriptor):
    technical = verify(run_dir, descriptor=descriptor)
    summary_pin, telemetry_pin = pin(run_dir / "qa_loop_summary.yaml"), pin(run_dir / "timing_detailed.json")
    summary = yaml.safe_load(Path(summary_pin["path"]).read_bytes())
    environment = yaml.safe_load((run_dir / "environment.yaml").read_bytes())
    _author_integrity(run_dir, environment, summary)
    telemetry = _read(telemetry_pin)
    if (
        summary.get("validation_scope") != "capsules"
        or summary.get("converged") is not True
        or summary.get("authoring_converged") is not True
        or summary.get("authoring_complete") is not True
        or summary.get("numeric_all_pass") is not True
        or summary.get("workflow_conformant") is not True
        or summary.get("phase2_iteration_ready") is not True
        or summary.get("capsule_complete") is not True
        or summary.get("formal_complete") is not False
        or summary.get("external_full_models_complete") is not False
        or summary.get("operator_seal")
        or (summary.get("telemetry") or {}).get("complete") is not True
        or (telemetry.get("telemetry_integrity") or {}).get("complete") is not True
        or {key: summary["telemetry"].get(key) for key in ("path", "sha256")} != telemetry_pin
        or (summary.get("official_grade") or {}).get("capsule_certification") != technical["receipt"]
    ):
        raise ValueError("capsule publication lacks completed author workflow and telemetry")
    return technical, summary_pin, telemetry_pin


def _author_integrity(run_dir, environment, summary):
    from merlin.targetgen.sandbox.answer_surfaces import audit_hit_is_violation

    mask = environment.get("golden_mask_selftest") or {}
    snapshot = environment.get("bundle_input_snapshot") or {}
    if (
        environment.get("sandbox") != "bwrap"
        or environment.get("isolation_violations") != []
        or mask.get("leaked_answer_files") != []
        or type(mask.get("n_answer_files_masked")) is not int
        or mask["n_answer_files_masked"] <= 0
        or snapshot.get("version") != 4
        or type(snapshot.get("n_files")) is not int
        or snapshot["n_files"] <= 0
        or type(snapshot.get("n_bytes")) is not int
        or snapshot["n_bytes"] <= 0
    ):
        raise ValueError("capsule author publication lacks clean isolation and answer-mask input provenance")
    rounds = summary.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("capsule author publication has no completed audited round")
    finalized = summary.get("finalize")
    if not isinstance(finalized, dict):
        raise ValueError("capsule author publication has no finalization record")
    if not finalized.get("skipped") or (run_dir / "rounds/finalize.transcript.jsonl").exists():
        rounds = [*rounds, finalized]
    for row in rounds:
        hits = row.get("audit_hits") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("answer_access_clean") is not True
            or not isinstance(hits, list)
            or any(audit_hit_is_violation(hit) for hit in hits)
        ):
            raise ValueError("capsule author publication has an incomplete or violated answer-access audit")


def publish_phase2_handoff(
    run_dir: Path, *, descriptor: Path, transcript: Path, endpoint_kind: str, resolved_tools
) -> dict:
    """Seal the final author publication separately from the earlier capsule grade."""
    from .. import authoring_conformance as AC
    from .. import conformance

    answer_surfaces = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    run_dir, descriptor, transcript = Path(run_dir).absolute(), Path(descriptor).absolute(), Path(transcript).absolute()
    technical, summary_pin, telemetry_pin = _publication_inputs(run_dir, descriptor)
    environment = yaml.safe_load((run_dir / "environment.yaml").read_bytes())
    if not transcript.is_relative_to(run_dir / "rounds") or list(resolved_tools) != environment.get("resolved_tools"):
        raise ValueError("capsule author publication has a different transcript or selected tool roster")
    output = run_dir / "capsule_workflow.json"
    if output.exists():
        raise ValueError("capsule publication refuses a stale workflow observation")
    arguments = {
        "transcript": pin(transcript),
        "submission": technical["handoff"]["compiler"],
        "arm": environment["arm"],
        "endpoint_kind": endpoint_kind,
        "resolved_tools": list(resolved_tools),
    }
    with IR.observe_call(
        run_dir / "capsule_handoff",
        stage="phase1_capsule_publication",
        function=AC.workflow_conformance,
        arguments=arguments,
        inputs=(
            transcript,
            Path(technical["receipt"]["path"]),
            Path(summary_pin["path"]),
            Path(telemetry_pin["path"]),
            run_dir / "environment.yaml",
            descriptor,
        ),
        outputs=(output,),
        dependencies=(Path(__file__), Path(conformance.__file__), Path(answer_surfaces.__file__)),
    ) as observation:
        conf, accepted = AC.workflow_conformance(
            transcript,
            Path(arguments["submission"]["path"]),
            arguments["arm"],
            endpoint_kind,
            resolved_tools,
        )
        if not accepted:
            raise ValueError("current frozen compiler workflow no longer conforms")
        output.write_text(json.dumps(conf, indent=2) + "\n")
        observation.returned()
    record = {
        "schema": "merlin.phase1.capsule_publication.v1",
        "validation_scope": "capsules",
        "target": technical["target"],
        "descriptor": technical["descriptor"],
        "capsule_clock": technical["capsule_clock"],
        "capsule_complete": True,
        "phase2_iteration_ready": True,
        "authoring_converged": True,
        "formal_grade_complete": False,
        "external_full_models_complete": False,
        "submission_sha256": technical["submission_sha256"],
        "handoff": technical["handoff"],
        "capsule_certification": technical["receipt"],
        "qa_summary": summary_pin,
        "telemetry": telemetry_pin,
        "workflow": pin(output),
        "workflow_invocation": pin(observation.path),
        "published_at": datetime.now(UTC).isoformat(),
    }
    path = run_dir / "capsule_publication.json"
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    path.chmod(0o444)
    return verify_phase2_handoff(run_dir, descriptor=descriptor)


def verify_phase2_handoff(run_dir: Path, *, descriptor: Path) -> dict:
    """Admit final capsule publication without claiming any external model result."""
    from .. import authoring_conformance as AC

    run_dir, descriptor = Path(run_dir).absolute(), Path(descriptor).absolute()
    path = run_dir / "capsule_publication.json"
    record = _read(pin(path))
    technical, summary_pin, telemetry_pin = _publication_inputs(run_dir, descriptor)
    _read(record["workflow_invocation"])
    invocation = IR.verify(Path(record["workflow_invocation"]["path"]))
    arguments = invocation.get("arguments") or {}
    environment = yaml.safe_load((run_dir / "environment.yaml").read_bytes())
    if (
        record.get("schema") != "merlin.phase1.capsule_publication.v1"
        or record.get("validation_scope") != "capsules"
        or record.get("target") != technical["target"]
        or record.get("descriptor") != technical["descriptor"]
        or record.get("capsule_clock") != technical["capsule_clock"]
        or record.get("phase2_iteration_ready") is not True
        or record.get("capsule_complete") is not True
        or record.get("authoring_converged") is not True
        or record.get("formal_grade_complete") is not False
        or record.get("external_full_models_complete") is not False
        or record.get("submission_sha256") != technical["submission_sha256"]
        or record.get("handoff") != technical["handoff"]
        or record.get("capsule_certification") != technical["receipt"]
        or record.get("qa_summary") != summary_pin
        or record.get("telemetry") != telemetry_pin
        or invocation.get("kind") != "python_call"
        or invocation.get("callable") != AC.__name__ + ".workflow_conformance"
        or (invocation.get("executable") or {}).get("sha256") != pin(Path(AC.__file__))["sha256"]
        or invocation.get("stage") != "phase1_capsule_publication"
        or record.get("workflow") not in invocation.get("outputs", [])
        or arguments.get("submission") != technical["handoff"]["compiler"]
        or arguments.get("arm") != environment.get("arm")
        or arguments.get("resolved_tools") != environment.get("resolved_tools")
    ):
        raise ValueError("capsule publication has no matching actual workflow and frozen handoff")
    for item in (
        technical["receipt"],
        summary_pin,
        telemetry_pin,
        technical["descriptor"],
        pin(run_dir / "environment.yaml"),
    ):
        if item not in invocation["inputs"]:
            raise ValueError("capsule publication did not observe its final selected inputs")
    transcript = Path(arguments["transcript"]["path"])
    if (
        not transcript.is_relative_to(run_dir / "rounds")
        or pin(transcript) != arguments["transcript"]
        or arguments["transcript"] not in invocation["inputs"]
    ):
        raise ValueError("capsule publication workflow transcript changed")
    conf, accepted = AC.workflow_conformance(
        transcript,
        Path(arguments["submission"]["path"]),
        arguments["arm"],
        arguments["endpoint_kind"],
        arguments["resolved_tools"],
    )
    if not accepted or _read(record["workflow"]) != conf:
        raise ValueError("capsule publication workflow does not match the actual frozen compiler")
    return {**record, "receipt": pin(path)}
