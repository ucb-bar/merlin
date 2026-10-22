"""Operator-reviewed library changes through a new native authoring epoch.

No library is edited in place. Only a frozen output of this joint campaign may
continue, with its original empty origin, clock and shared physical ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import yaml

from merlin.common.tree_hash import hash_tree

SCHEMA = "merlin.joint_library_epoch.v1"
PERFORMANCE_ENV = "MERLIN_JOINT_PERFORMANCE_SELECTION"


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents):
        raise ValueError("joint epoch requires direct regular operator files")
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _tier_ledger(environment):
    """Locate optional native evidence without moving its immutable producer paths."""
    workspace = environment.get("workspace_path")
    if workspace is None:
        return None
    workspace = Path(workspace)
    if not workspace.is_absolute() or workspace != workspace.resolve():
        raise ValueError("tier ledger requires the original canonical workspace")
    path = workspace / "qa" / "tier_state.json"
    if not path.exists() and not path.is_symlink():
        return None
    raw, digest = _read(path)
    if len(raw) > 64 * 1024 * 1024 or not isinstance(json.loads(raw), dict):
        raise ValueError("invalid bounded native tier ledger")
    return {"path": str(path), "sha256": digest}


def _calibration_history(root):
    """Pin descriptive native metadata, retaining all embedded original paths."""
    rows, size = [], 0
    for name in ("calibration_candidate_index", "calibration_registry"):
        directory = Path(root) / name
        if directory.is_symlink():
            raise ValueError("calibration history cannot be redirected")
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.json")):
            raw, digest = _read(path)
            size += len(raw)
            if len(rows) >= 8192 or size > 64 * 1024 * 1024 or not isinstance(json.loads(raw), dict):
                raise ValueError("calibration history exceeds native metadata bounds")
            rows.append({"path": str(path), "relative": str(path.relative_to(root)), "sha256": digest})
    return rows


def performance_pin(selection=None):
    """Pin the complete selected runtime, including geometry/model bindings."""
    if selection is None:
        raw = os.environ.get(PERFORMANCE_ENV)
        if not raw:
            return None
        selection = json.loads(raw)
    path = Path(selection["path"]).absolute()
    digest = _read(path)[1]
    if digest != selection.get("sha256"):
        raise ValueError("joint performance selection content differs from its pin")
    return {"path": str(path), "sha256": digest}


def request_stop(run_dir, *, reason):
    """Stop at the next native Codex turn boundary, then join owned jobs/freeze."""
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 4096:
        raise ValueError("epoch stop needs a bounded operator reason")
    path = Path(run_dir) / "joint_epoch_stop.json"
    state = json.loads(_read(path.with_name("joint_campaign.json"))[0])
    if state.get("started_at_epoch") is None:
        raise ValueError("joint epoch stop requires a started author")
    with path.open("x") as stream:
        json.dump({"schema": "merlin.joint_epoch_stop.v1", "reason": reason}, stream)
    path.chmod(0o600)
    publish_stop_notice(run_dir)
    return path


def publish_stop_notice(run_dir):
    """Project a trusted stop request into existing QA without changing a grade."""
    root = Path(run_dir)
    raw, digest = _read(root / "joint_epoch_stop.json")
    request = json.loads(raw)
    if (request.get("schema") != "merlin.joint_epoch_stop.v1"
            or not isinstance(request.get("reason"), str) or not request["reason"].strip()
            or len(request["reason"]) > 4096):
        raise ValueError("invalid bounded operator epoch notice")
    environment = yaml.safe_load(_read(root / "environment.yaml")[0])
    workspace = environment.get("workspace_path")
    if workspace is None:
        return None  # Historical/fixture owners may have no staged author workspace.
    workspace = Path(workspace)
    if not workspace.is_absolute() or workspace != workspace.resolve() or not workspace.is_dir():
        raise ValueError("operator notice needs its original staged author workspace")
    notice = {"schema": "merlin.operator_epoch_notice.v1", "action": "checkpoint_and_return",
              "reason": request["reason"], "stop_request_sha256": digest,
              "instruction": "Finish the current bounded action, save all compiler and generic pass edits, "
                  "and return a checkpoint summary now. Preserve your work; do not restart the compiler. "
                  "The native host will continue this saved compiler with the reviewed inputs and original campaign clock.",
              "changes_grading": False, "private_reference_information": False}
    from .feedback.visible_model_gate import publish
    publish(workspace / "qa" / "verdict.json", {"operator_epoch_notice": notice})
    return notice


def stop_reason(run_dir):
    path = Path(run_dir) / "joint_epoch_stop.json"
    if not path.exists():
        return None
    value = json.loads(_read(path)[0])
    if value.get("schema") != "merlin.joint_epoch_stop.v1" or not str(value.get("reason", "")).strip():
        raise ValueError("invalid operator epoch stop")
    return "operator library epoch: " + value["reason"]


def prepare(previous_run, *, contract_sha256, review_id, output, performance_selection_sha256=None):
    """Pin a completed native epoch; this is review selection, not qualification."""
    root = Path(previous_run).resolve(strict=True)
    if len(contract_sha256) != 64 or any(c not in "0123456789abcdef" for c in contract_sha256):
        raise ValueError("epoch needs the reviewed new library contract SHA")
    if not isinstance(review_id, str) or not review_id.strip():
        raise ValueError("epoch needs an explicit operator review identity")
    pins = {name: _read(root / name)[1] for name in
            ("environment.yaml", "joint_campaign.json", "joint_campaign_summary.json")}
    environment = yaml.safe_load((root / "environment.yaml").read_text())
    summary = json.loads((root / "joint_campaign_summary.json").read_text())
    source = root / "joint_frozen_submission"
    if summary.get("frozen_compiler") != str(source) or summary.get("compiler_sha256") != hash_tree(source)["sha256"]:
        raise ValueError("joint frozen compiler differs from its completion record")
    old_contract = (environment.get("additional_compiler_library") or {}).get("contract_sha256")
    if old_contract == contract_sha256 and performance_selection_sha256 is None:
        raise ValueError("library epoch requires a changed reviewed contract")
    record = {"schema": SCHEMA, "previous_run": str(root), "pins": pins,
              "compiler_sha256": summary["compiler_sha256"], "source": str(source),
              "new_contract_sha256": contract_sha256, "review_id": review_id,
              "lineage_id": environment.get("lineage_id"), "source_run_id": environment.get("run_id")}
    ledger = _tier_ledger(environment)
    if ledger is not None:
        if summary.get("completion_reason") != "library_epoch":
            raise ValueError("tier evidence handoff requires completed native epoch drain")
        record["tier_ledger"] = ledger
    calibration = _calibration_history(root)
    if calibration:
        if summary.get("completion_reason") != "library_epoch":
            raise ValueError("calibration history handoff requires completed native epoch drain")
        record["calibration_history"] = calibration
    if performance_selection_sha256 is not None:
        if len(performance_selection_sha256) != 64 or any(c not in "0123456789abcdef" for c in performance_selection_sha256):
            raise ValueError("epoch timing selection needs its reviewed SHA")
        record["performance_selection_sha256"] = performance_selection_sha256
        state = json.loads((root / "joint_campaign.json").read_text())
        if "performance_selection" not in state:
            raise ValueError("previous epoch did not capture its performance selection")
        previous = state["performance_selection"]
        if previous and previous["sha256"] == performance_selection_sha256:
            raise ValueError("model epoch requires a changed performance selection")
    _verify(record)
    destination = Path(output)
    with destination.open("x") as stream:
        json.dump(record, stream, indent=2)
    destination.chmod(0o444)
    return record


def _verify(record):
    if record.get("schema") == "merlin.joint_interrupted_epoch.v1":
        from .joint_interrupted_epoch import verify
        return verify(record)
    if record.get("schema") != SCHEMA or not record.get("review_id"):
        raise ValueError("invalid joint library epoch review")
    root = Path(record["previous_run"])
    for name in ("environment.yaml", "joint_campaign.json", "joint_campaign_summary.json"):
        if _read(root / name)[1] != record["pins"][name]:
            raise ValueError("previous joint epoch changed after operator review")
    environment = yaml.safe_load((root / "environment.yaml").read_text())
    if "tier_ledger" in record and _tier_ledger(environment) != record["tier_ledger"]:
        raise ValueError("previous tier evidence changed after epoch review")
    if "calibration_history" in record and _calibration_history(root) != record["calibration_history"]:
        raise ValueError("previous calibration history changed after epoch review")
    origin = environment.get("backend_origin")
    if origin != {"schema": "merlin.empty_backend_origin.v1", "initial_files": 0}:
        raise ValueError("joint epoch has no original empty backend origin")
    if not record.get("lineage_id") or record["lineage_id"] != environment.get("lineage_id"):
        raise ValueError("joint epoch lineage changed")
    source = Path(record["source"])
    if source != root / "joint_frozen_submission" or source.is_symlink() or not (source / "manifest.yaml").is_file():
        raise ValueError("joint epoch requires its own complete frozen compiler")
    if any(p.is_symlink() or (not p.is_dir() and not p.is_file()) for p in source.rglob("*")):
        raise ValueError("joint epoch compiler has indirect members")
    if hash_tree(source)["sha256"] != record["compiler_sha256"]:
        raise ValueError("joint epoch compiler changed")
    state = json.loads((root / "joint_campaign.json").read_text())
    if state.get("started_at_epoch") is None:
        raise ValueError("previous joint author never started")
    return state, origin


def selected(options):
    path = getattr(options, "joint_epoch", "")
    if not path:
        return None
    if not options.joint_campaign or not options.fresh_backend or options.seed_submission or options.qualify_submission:
        raise ValueError("joint epoch requires joint/fresh origin and excludes ordinary seeds")
    record = json.loads(_read(path)[0])
    _verify(record)
    timing = record.get("performance_selection_sha256")
    if timing is not None:
        selection = performance_pin()
        if not selection or selection["sha256"] != timing:
            raise ValueError("joint epoch performance selection differs from operator review")
    elif performance_pin() != _verify(record)[0].get("performance_selection"):
        raise ValueError("joint epoch performance selection changed without review")
    return record


def inherited_state(options, policy, policy_sha256):
    record = selected(options)
    if record is None:
        return None
    state, _ = _verify(record)
    if state["policy"] != policy or state["policy_sha256"] != policy_sha256:
        raise ValueError("library epoch cannot change campaign policy or reset budgets")
    if time.time() >= state["started_at_epoch"] + policy["duration_s"] - policy["drain_s"]:
        raise ValueError("joint epoch has no authoring time remaining")
    return {**state, "epoch": record, "performance_selection": performance_pin(),
            "gsim_ledger": state.get("gsim_ledger", str(Path(record["previous_run"]) / "joint_gsim_budget.json")),
            "gsim_correctness_ledger": state.get("gsim_correctness_ledger", str(
                Path(record["previous_run"]) / "joint_gsim_correctness_budget.json"))}


def request_model_epoch(run_dir, *, selection_path, expected_sha256, review_id, output):
    """Trusted calibration service: request native drain, then pin its checkpoint.

    This does not launch another author or qualify the timing model. Call again
    after the existing native owner has joined jobs and written its frozen summary.
    The service must supply its accepted calibration/held-out review identity.
    """
    root = Path(run_dir).resolve(strict=True)
    selection = performance_pin({"path": str(selection_path), "sha256": expected_sha256})
    state = json.loads(_read(root / "joint_campaign.json")[0])
    if state.get("started_at_epoch") is None or time.time() >= (
            state["started_at_epoch"] + state["policy"]["duration_s"] - state["policy"]["drain_s"]):
        raise ValueError("model epoch has no authoring time remaining")
    if "performance_selection" not in state:
        raise ValueError("previous epoch did not capture its performance selection")
    if state["performance_selection"] and state["performance_selection"]["sha256"] == expected_sha256:
        raise ValueError("model epoch requires a changed performance selection")
    if not isinstance(review_id, str) or not review_id.strip():
        raise ValueError("model epoch needs a calibration review identity")
    proposal = {"schema": "merlin.joint_model_epoch_request.v1", "selection": selection,
                "review_id": review_id, "output": str(Path(output).absolute())}
    request = root / "joint_model_epoch_request.json"
    if request.exists():
        if json.loads(_read(request)[0]) != proposal:
            raise ValueError("another model epoch request is already pinned")
    else:
        with request.open("x") as stream:
            json.dump(proposal, stream, indent=2)
        request.chmod(0o600)
    if not (root / "joint_epoch_stop.json").exists():
        request_stop(root, reason="reviewed timing model: " + review_id)
    if not (root / "joint_campaign_summary.json").exists():
        return {"status": "waiting_for_native_drain", "request": str(request)}
    destination = Path(output)
    if destination.exists():
        record = json.loads(_read(destination)[0])
        _verify(record)
        if (record.get("performance_selection_sha256") != expected_sha256 or
                record.get("review_id") != review_id or record.get("previous_run") != str(root)):
            raise ValueError("model epoch output differs from pinned request")
    else:
        environment = yaml.safe_load(_read(root / "environment.yaml")[0])
        prepare(root, contract_sha256=environment["additional_compiler_library"]["contract_sha256"],
                review_id=review_id, output=destination, performance_selection_sha256=expected_sha256)
    return {"status": "ready_for_native_epoch", "joint_epoch": str(destination),
            "performance_selection": selection}


def seed(options, workspace, run_dir):
    from . import run_inputs
    record = selected(options)
    result = run_inputs.seed_submission(workspace, record["source"], run_dir, require_stable_source=True)
    if record.get("schema") == "merlin.joint_interrupted_epoch.v1":
        from .joint_interrupted_epoch import seed_variants
        result["authored_variants_handoff"] = seed_variants(record, workspace)
        result["interrupted_predecessor"] = record["interruption"]
    if "tier_ledger" in record:
        pin = record["tier_ledger"]
        raw, digest = _read(pin["path"])
        if digest != pin["sha256"]:
            raise ValueError("previous tier evidence changed before handoff")
        qa = Path(workspace) / "qa"
        qa.mkdir(exist_ok=True)
        if qa.is_symlink() or qa != qa.resolve():
            raise ValueError("tier evidence destination must be a canonical native QA directory")
        destination = qa / "tier_state.json"
        with destination.open("xb") as stream:
            stream.write(raw)
        destination.chmod(0o600)
        result["tier_evidence_handoff"] = {"source": pin, "destination": str(destination),
            "exact_bytes_preserved": True, "producer_paths_relocated": False,
            "current_compiler_credit_granted": False}
    if "calibration_history" in record:
        for pin in record["calibration_history"]:
            raw, digest = _read(pin["path"])
            if digest != pin["sha256"]:
                raise ValueError("calibration history changed before handoff")
            destination = Path(run_dir) / pin["relative"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.parent != destination.parent.resolve():
                raise ValueError("calibration history destination cannot be redirected")
            with destination.open("xb") as stream:
                stream.write(raw)
            destination.chmod(0o600)
        result["calibration_history_handoff"] = {"members": record["calibration_history"],
            "exact_bytes_preserved": True, "producer_paths_relocated": False,
            "dispatch_or_fit_authority_added": False}
    result["joint_epoch"] = record
    result["lineage"] = {"lineage_id": record["lineage_id"], "source_run_id": record["source_run_id"],
                         "source_tree_sha256": record["compiler_sha256"]}
    (Path(run_dir) / "seed_submission.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stop = commands.add_parser("request-stop")
    stop.add_argument("--run-dir", type=Path, required=True)
    stop.add_argument("--reason", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--previous-run", type=Path, required=True)
    build.add_argument("--contract-sha256", required=True)
    build.add_argument("--review-id", required=True)
    build.add_argument("--performance-selection-sha256")
    build.add_argument("--output", type=Path, required=True)
    values = vars(parser.parse_args())
    command = values.pop("command")
    result = request_stop(**values) if command == "request-stop" else prepare(**values)
    print(json.dumps(str(result) if isinstance(result, Path) else result, indent=2))


if __name__ == "__main__":
    main()
