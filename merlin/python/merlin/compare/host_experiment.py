"""Contract and fail-closed preflight for a four-arm CPU-host compiler experiment.

The experiment owns its paths and task text; this library module only consumes a supplied YAML path.
It enforces the methodological invariants that matter across targets: nested treatments, paper-holdout
isolation, one agent/provider, full-fidelity telemetry, one grader, and unresolved-input refusal.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import statistics as stats
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import repo_root
from merlin.common.schemas import validate_or_raise


_REQUIRED_TELEMETRY = frozenset({
    "raw_events", "arrival_timestamps", "token_subsets", "reasoning_tokens",
    "cache_read_tokens", "cache_write_tokens", "active_time", "wall_time",
    "grader_time", "tool_calls", "billing_mode",
})


def _git_provenance(path: Path) -> dict[str, Any]:
    """Immutable revision plus a content identity for every local modification.

    A list of dirty path names is not an identity: two different AET or Chia patches can touch the
    same files.  Bind the binary diff against ``HEAD`` and every untracked, non-ignored byte as well
    as the commit.  Ignored build/cache trees are not import source and intentionally do not perturb
    a frozen protocol.
    """
    try:
        sha = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True, timeout=30, check=True).stdout
        diff = subprocess.run(
            ["git", "-C", str(path), "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
            capture_output=True, timeout=30, check=True).stdout
        untracked_raw = subprocess.run(
            ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True, timeout=30, check=True).stdout
        untracked_rows = []
        for encoded in sorted(name for name in untracked_raw.split(b"\0") if name):
            relative = encoded.decode("utf-8", errors="surrogateescape")
            source = path / relative
            if source.is_symlink():
                kind, content = "symlink", source.readlink().as_posix().encode("utf-8")
            elif source.is_file():
                kind, content = "file", source.read_bytes()
            else:
                kind, content = "missing_or_non_file", b""
            untracked_rows.append((relative, kind, hashlib.sha256(content).hexdigest()))
        dirty_payload = {
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked": untracked_rows,
        }
        dirty_content_sha256 = hashlib.sha256(json.dumps(
            dirty_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        dirty_paths = status.decode("utf-8", errors="replace").splitlines()
        return {"git_sha": sha, "dirty": bool(status), "dirty_paths": dirty_paths,
                "dirty_content_sha256": dirty_content_sha256,
                "status_sha256": hashlib.sha256(status).hexdigest()}
    except Exception as exc:
        return {"git_sha": None, "dirty": None, "error": str(exc)}


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    return value


def _resolved(value: Any) -> bool:
    return bool(value) and str(value).strip().lower() not in {"unresolved", "none", "null"}


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _protocol_design_replacement_valid(
        *, replacement: dict[str, Any], documents: dict[str, dict[str, Any]],
        artifact_sha256: dict[str, str], cells: Path, arm4_raw_size: int) -> bool:
    """Verify every semantic and content-addressed link in a protocol-design revocation.

    This is deliberately a pure predicate over parsed controller records plus the immutable receipt
    tree.  Preflight reports a single fail-closed invalidity when any link is absent or inconsistent;
    tests can mutate each independently without rewriting historical evidence.
    """
    try:
        campaign = str(replacement["predecessor_campaign_run_id"])
        protocol = str(replacement["predecessor_protocol_inputs_sha256"])
        reason_codes = set(replacement["reason_codes"])
        if (len(replacement["reason_codes"]) != len(reason_codes) or
                set(documents) != {
                    "frozen_protocol", "protocol_claim", "design_audit",
                    "claim_revocation", "campaign_exclusion", "arm1_terminal_outcome",
                    "arm1_grader_result", "arm2_terminal_outcome", "arm2_grader_result",
                    "arm4_run_record", "arm4_controller_cancellation",
                } or any(not _digest(value) for value in artifact_sha256.values())):
            return False

        frozen = _mapping(documents["frozen_protocol"], "predecessor frozen protocol")
        claim = _mapping(documents["protocol_claim"], "predecessor claim")
        audit = _mapping(documents["design_audit"], "predecessor design audit")
        decision = _mapping(audit.get("decision"), "design audit decision")
        arm1_terminal = _mapping(
            documents["arm1_terminal_outcome"], "predecessor Arm1 terminal")
        arm2_terminal = _mapping(
            documents["arm2_terminal_outcome"], "predecessor Arm2 terminal")
        arm1_grader = _mapping(documents["arm1_grader_result"], "predecessor Arm1 grader")
        arm2_grader = _mapping(documents["arm2_grader_result"], "predecessor Arm2 grader")
        arm4_record = _mapping(documents["arm4_run_record"], "predecessor Arm4 run record")
        cancellation = _mapping(
            documents["arm4_controller_cancellation"], "predecessor Arm4 cancellation")
        revocation = _mapping(documents["claim_revocation"], "predecessor claim revocation")
        exclusion = _mapping(documents["campaign_exclusion"], "predecessor exclusion")

        def linked(document: dict[str, Any], key: str, artifact: str, **extra: Any) -> bool:
            row = document.get(key)
            return (isinstance(row, dict) and
                    row.get("sha256") == artifact_sha256.get(artifact) and
                    all(row.get(name) == value for name, value in extra.items()))

        def exact_reasons(document: dict[str, Any]) -> bool:
            reasons = document.get("reason_codes")
            return (isinstance(reasons, list) and len(reasons) == len(reason_codes) and
                    set(reasons) == reason_codes)

        def protocol_failure(terminal: dict[str, Any], grader: dict[str, Any],
                             arm: str, message: str) -> bool:
            checks = terminal.get("checks")
            levels = grader.get("levels")
            if not isinstance(checks, dict) or not isinstance(levels, dict):
                return False
            records = (levels.get("L0") or {}).get("records")
            return (
                terminal.get("arm") == arm and
                terminal.get("terminal_class") == "graded_fail" and
                checks.get("agent_success") is True and
                checks.get("workspace_input_audit") is True and
                checks.get("aet_reconciled") is True and
                checks.get("compiler_seal_status") == "sealed" and
                checks.get("grader_status") == "fail" and
                grader.get("status") == "fail" and
                isinstance(records, list) and len(records) == 143 and
                all(isinstance(row, dict) and row.get("reason") ==
                    "compiler invocation failed" and message in str(row.get("stderr_tail", ""))
                    for row in records))

        freeze = frozen.get("freeze")
        agent = frozen.get("agent")
        plan = agent.get("launch_plan") if isinstance(agent, dict) else None
        if (frozen.get("status") != "protocol_frozen" or not isinstance(freeze, dict) or
                freeze.get("protocol_inputs_sha256") != protocol or
                not isinstance(plan, list) or len(plan) != 16):
            return False

        consumed = ["00.consumed.json", "01.consumed.json", "02.consumed.json"]
        cancelled = [f"{ordinal:02d}.cancelled.json" for ordinal in range(3, 16)]
        expected_receipt_names = consumed + cancelled
        actual_names = sorted(path.name for path in cells.iterdir()) if cells.is_dir() else []
        if (actual_names != sorted(expected_receipt_names) or
                any(path.is_symlink() or not path.is_file() for path in cells.iterdir())):
            return False
        receipt_tree = revocation.get("receipt_tree")
        if (not isinstance(receipt_tree, dict) or set(receipt_tree) != set(expected_receipt_names) or
                any(not _digest(digest) or
                    hashlib.sha256((cells / name).read_bytes()).hexdigest() != digest
                    for name, digest in receipt_tree.items())):
            return False

        exclusion_cells = exclusion.get("cells")
        if not isinstance(exclusion_cells, list) or len(exclusion_cells) != 16:
            return False
        for ordinal, (plan_row, exclusion_row) in enumerate(zip(plan, exclusion_cells)):
            if not isinstance(plan_row, dict) or not isinstance(exclusion_row, dict):
                return False
            expected_name = (f"{ordinal:02d}.consumed.json" if ordinal < 3 else
                             f"{ordinal:02d}.cancelled.json")
            receipt = json.loads((cells / expected_name).read_text(encoding="utf-8"))
            if not isinstance(receipt, dict):
                return False
            expected_run = (f"{campaign}__{plan_row.get('arm')}__"
                            f"r{int(plan_row.get('repeat')):02d}__"
                            f"seed{int(plan_row.get('seed')):03d}")
            expected_state = ("sealed_protocol_pilot" if ordinal < 2 else
                              "controller_cancelled_during_agent" if ordinal == 2 else
                              "not_started_cancelled")
            if (plan_row.get("ordinal") != ordinal or
                    receipt.get("version") != 1 or receipt.get("status") != "authorized" or
                    receipt.get("protocol_inputs_sha256") != protocol or
                    receipt.get("campaign_run_id") != campaign or receipt.get("ordinal") != ordinal or
                    receipt.get("arm") != plan_row.get("arm") or
                    receipt.get("repeat") != plan_row.get("repeat") or
                    receipt.get("seed") != plan_row.get("seed") or
                    receipt.get("run_id") != expected_run or
                    exclusion_row.get("ordinal") != ordinal or
                    exclusion_row.get("arm") != plan_row.get("arm") or
                    exclusion_row.get("repeat") != plan_row.get("repeat") or
                    exclusion_row.get("seed") != plan_row.get("seed") or
                    exclusion_row.get("state") != expected_state or
                    exclusion_row.get("arm_outcome") is not False or
                    exclusion_row.get("receipt") != expected_name or
                    exclusion_row.get("receipt_sha256") != receipt_tree[expected_name]):
                return False

        expected_arm1 = f"{campaign}__arm1_raw_cpp__r00__seed001"
        expected_arm2 = f"{campaign}__arm2_cpp_scaffold__r00__seed001"
        expected_arm4 = f"{campaign}__arm4_agentic_pass_authoring__r00__seed001"
        sealed = audit.get("sealed_cells")
        partial = audit.get("cancelled_partial_cell")
        if (not isinstance(sealed, list) or len(sealed) != 2 or
                [row.get("run_id") for row in sealed if isinstance(row, dict)] !=
                [expected_arm1, expected_arm2] or not isinstance(partial, dict)):
            return False
        audit_links_ok = (
            linked(audit, "frozen_protocol", "frozen_protocol") and
            linked(audit, "protocol_claim", "protocol_claim", status="bound") and
            linked(sealed[0], "terminal_outcome", "arm1_terminal_outcome") and
            linked(sealed[0], "grader_result", "arm1_grader_result") and
            linked(sealed[1], "terminal_outcome", "arm2_terminal_outcome") and
            linked(sealed[1], "grader_result", "arm2_grader_result") and
            partial.get("run_record_sha256") == artifact_sha256["arm4_run_record"] and
            partial.get("raw_events_sha256") == artifact_sha256["arm4_raw_events"])

        exclusion_schedule_ok = (
            exclusion.get("version") == 1 and
            exclusion.get("authority") == "controller_campaign_exclusion_v1" and
            exclusion.get("campaign_run_id") == campaign and
            exclusion.get("protocol_inputs_sha256") == protocol and
            exclusion.get("classification") == "protocol_invalid_pilot" and
            exclusion.get("confirmatory_eligible") is False and
            exclusion.get("excluded_from_arm_outcomes") is True and
            exclusion.get("excluded_from_promotion") is True and
            exclusion.get("excluded_from_holdout_capture") is True and
            exclusion.get("supersedes_terminal_paper_eligibility") is True and
            exclusion.get("treatment_or_provider_started") is True and
            exact_reasons(exclusion) and
            linked(exclusion, "frozen_protocol", "frozen_protocol") and
            linked(exclusion, "protocol_claim", "protocol_claim", status="bound") and
            linked(exclusion, "design_audit", "design_audit") and
            exclusion.get("non_reuse") == {
                "threads": True, "workspaces": True, "submissions": True,
                "policies": True, "search_observations": True,
                "compiler_artifacts": True,
            })
        exclusion_partial = exclusion.get("arm4_partial_evidence")
        exclusion_partial_ok = (
            isinstance(exclusion_partial, dict) and
            exclusion_partial.get("run_record_sha256") == artifact_sha256["arm4_run_record"] and
            exclusion_partial.get("raw_events_sha256") == artifact_sha256["arm4_raw_events"] and
            exclusion_partial.get("controller_cancellation_sha256") ==
            artifact_sha256["arm4_controller_cancellation"] and
            exclusion_partial.get("turn_completed_present") is False and
            exclusion_partial.get("usage_record_present") is False and
            exclusion_partial.get("submission_directory_empty") is True)

        revocation_links_ok = (
            revocation.get("version") == 1 and
            revocation.get("authority") == "controller_protocol_claim_revocation_v1" and
            revocation.get("campaign_run_id") == campaign and
            revocation.get("protocol_inputs_sha256") == protocol and
            revocation.get("consumed_ordinals") == [0, 1, 2] and
            revocation.get("cancelled_ordinals") == list(range(3, 16)) and
            revocation.get("resume_forbidden") is True and exact_reasons(revocation) and
            linked(revocation, "claim", "protocol_claim", historical_status="bound") and
            linked(revocation, "campaign_exclusion", "campaign_exclusion") and
            linked(revocation, "design_audit", "design_audit"))

        cancellation_links_ok = (
            cancellation.get("version") == 1 and cancellation.get("run_id") == expected_arm4 and
            cancellation.get("campaign_run_id") == campaign and
            cancellation.get("protocol_inputs_sha256") == protocol and
            cancellation.get("classification") ==
            "controller_cancelled_after_protocol_invalidation" and
            cancellation.get("terminal_outcome_present") is False and
            cancellation.get("paper_evidence_eligible") is False and
            cancellation.get("promotion_eligible") is False and
            exact_reasons(cancellation) and linked(cancellation, "design_audit", "design_audit"))

        return (
            claim.get("version") == 1 and claim.get("status") == "bound" and
            claim.get("campaign_run_id") == campaign and
            claim.get("protocol_inputs_sha256") == protocol and
            audit.get("version") == 1 and
            audit.get("authority") == "post_start_protocol_design_audit" and
            audit.get("campaign_run_id") == campaign and
            audit.get("protocol_inputs_sha256") == protocol and
            decision.get("classification") == "protocol_design_invalid" and
            decision.get("campaign_stopped") is True and
            decision.get("exclude_all_cells_from_arm_outcomes") is True and
            decision.get("exclude_all_cells_from_promotion") is True and
            decision.get("exclude_all_cells_from_holdout_capture") is True and
            exact_reasons(decision) and audit_links_ok and exclusion_schedule_ok and
            exclusion_partial_ok and
            revocation_links_ok and cancellation_links_ok and
            protocol_failure(arm1_terminal, arm1_grader, "arm1_raw_cpp",
                             "missing numeric capsule field: family") and
            protocol_failure(arm2_terminal, arm2_grader, "arm2_cpp_scaffold",
                             "missing family enum") and
            arm4_record.get("run_id") == expected_arm4 and
            arm4_record.get("arm") == "arm4_agentic_pass_authoring" and
            arm4_record.get("seed") == 1 and arm4_raw_size > 0 and
            partial.get("run_id") == expected_arm4 and
            partial.get("classification") ==
            "controller_cancelled_after_protocol_invalidation" and
            partial.get("terminal_outcome_present") is False and
            partial.get("paper_evidence_eligible") is False and
            partial.get("promotion_eligible") is False)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _git_oid(value: Any) -> bool:
    text = str(value or "")
    return len(text) in {40, 64} and all(char in "0123456789abcdef" for char in text)


def _positive_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(float(value)) and float(value) > 0)


def _tree_sha256(root: Path) -> str:
    rows: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append((relative, "symlink", path.readlink().as_posix()))
        elif path.is_file():
            rows.append((relative, "file", hashlib.sha256(path.read_bytes()).hexdigest()))
    return hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _grader_package_tree_identity(root: Path) -> str:
    """Recompute the source-sealed grader's mode-sensitive package identity."""
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            raise ValueError("calibration package contains .git metadata")
        stat = path.lstat()
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"calibration package contains a non-regular entry: {relative}")
        rows.append((relative.as_posix(), "dir" if path.is_dir() else "file",
                     stat.st_mode & 0o777,
                     None if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()))
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _calibration_sample(rows: list[dict[str, Any]], *, per_family: int,
                        families: list[str]) -> list[dict[str, Any]]:
    """Mirror the source-sealed beam-search semantic sampler for authority validation."""
    def coverage_key(row: dict[str, Any]) -> tuple[str, ...]:
        family = str(row["family"])
        if family == "contraction":
            return family, str(row["operation"]), str(row["dtype"]), str(row["layout"])
        if family in {"elementwise_map", "reduction", "movement_layout", "fusion_epilogue"}:
            return family, str(row["operation"]), str(row["dtype"])
        if family == "runtime_parallel":
            return family, str(row["operation"]), str(row["core_count"])
        raise ValueError(f"unsupported calibration family {family!r}")

    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("calibration validator accepts public train rows only")
        buckets.setdefault(coverage_key(row), []).append(row)
    representatives = [min(values, key=lambda row: str(row["sha256"]))
                       for _, values in sorted(buckets.items())]
    selected = []
    for family in sorted(families):
        selected.extend(sorted(
            (row for row in representatives if row["family"] == family),
            key=lambda row: str(row["sha256"]))[:per_family])
    return selected


def _calibration_private_capsule(row: dict[str, Any], *, nonce: bytes,
                                 phase: str) -> dict[str, Any]:
    """Mirror the source-sealed broker perturbation for independent calibration shapes."""
    private = dict(row)
    shape = dict(row.get("shape", {}))
    for name, value in sorted(shape.items()):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        material = f"{phase}:train:{row['id']}:{name}".encode()
        delta = 1 + int.from_bytes(
            hashlib.sha256(nonce + material).digest()[:2], "big") % 13
        shape[name] = value + delta
    if shape == row.get("shape"):
        raise ValueError("calibration private capsule shape did not change")
    private["shape"] = shape
    identity = {key: private[key] for key in (
        "family", "operation", "dtype", "shape", "layout", "state", "core_count")}
    digest = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    private["sha256"] = digest
    private["id"] = f"private-{private['family']}-{digest[:16]}"
    return private


def _calibration_k1_state_gates():
    """Use the exact source-sealed grader gate functions; do not duplicate their policy."""
    path = (repo_root() /
            "merlin/experiments/cpu_host_compiler_v0/grader.py").resolve()
    spec = importlib.util.spec_from_file_location("merlin_calibration_state_gates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source-sealed K1 environment gates")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._k1_state_ready, module._k1_state_pair_ok


def _calibration_space_problem(recorded: Any) -> str | None:
    """Bind a calibration's optimization space by CONTENT, never by absolute pathname.

    A frozen protocol tree is relocatable — the immutable campaign snapshot lives beside the
    working checkout — so the recorded path need not equal ``repo_root()/merlin/...``.  It must
    still name a readable file whose bytes are this tree's frozen space, so a substituted or
    altered space fails closed.  The digest comes from the space file itself rather than from a
    caller-supplied source map, because a noise authority deliberately omits ``search_space``
    from its own executable source seal.
    """
    expected = (repo_root() /
                "merlin/experiments/cpu_host_compiler_v0/optimization_space_v1.yaml").resolve()
    try:
        if not expected.is_file():
            return "the frozen optimization space is absent from this tree"
        path = Path(str(recorded)).resolve()
        if not path.is_file():
            return "the recorded optimization space is absent"
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashlib.sha256(
                expected.read_bytes()).hexdigest():
            return "the recorded optimization space content is not the frozen space"
    except OSError:
        return "the optimization space could not be read"
    return None


def _validate_calibration_semantics(*, label: str, value: Any, train_sha256: str,
                                    source_sha256: dict[str, str],
                                    space: dict[str, Any],
                                    train_rows: list[dict[str, Any]] | None = None,
                                    noise_authority: dict[str, Any] | None = None,
                                    noise_authority_sha256: str | None = None) -> list[str]:
    """Return fail-closed blockers for one timing-calibration authority artifact."""
    problems: list[str] = []
    if not isinstance(value, dict):
        return [f"{label} calibration is not a JSON object"]
    if label == "noise_calibration":
        expected_checks = {
            "six_families": True, "six_valid_pairs_per_family": True,
            "all_correct": True, "identical_k1_text": True,
            "no_heldout_argument": True,
        }
        if (value.get("version") != 2 or
                value.get("kind") != "cpu_host_k1_order_balanced_aa_noise_calibration" or
                value.get("status") != "pass" or value.get("checks") != expected_checks or
                value.get("paid_work") is not False or value.get("heldout_opened") is not False or
                value.get("protocol_state_mutated") is not False):
            problems.append("A/A noise calibration is not a passing public-only version-2 artifact")
        source_fields = {
            "calibrator_sha256": "noise_calibrator",
            "grader_sha256": "grader",
            "runner_sha256": "search_runner",
            "trusted_harness_sha256": "trusted_harness",
            "k1_monitor_sha256": "k1_monitor",
        }
        if any(value.get(field) != source_sha256.get(name)
               for field, name in source_fields.items()):
            problems.append("A/A noise calibration executable source hashes differ from protocol")
        if value.get("public_train_sha256") != train_sha256:
            problems.append("A/A noise calibration does not bind the exact public train split")
        expected_public_context = {
            "authority": "complete_public_train",
            "capsule_ids": [str(row["id"]) for row in (train_rows or [])],
            "row_count": len(train_rows or []),
            "rows_sha256": hashlib.sha256(json.dumps(
                train_rows or [], sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
        }
        if value.get("public_context") != expected_public_context:
            problems.append("A/A noise calibration did not retain the complete public context")
        derivation = (
            "margin=max(0.02,ceil((exp(max(abs(log(pair_ratio)))+0.005)-1)*1000)/1000); "
            "lower_bound=1/(1+margin)")
        calibration_protocol = {
            "version": 1,
            "confirmation_samples_per_family": int(space["confirmation_samples_per_family"]),
            "confirmation_families": list(space["confirmation_families"]),
            "measurement_repeats": int(space["measurement_repeats"]),
            "board_environment": dict(space["board_environment"]),
            "private_shape_authority":
                "trusted_broker_private_capsule_independent_calibration_nonce",
            "public_context": "complete_public_train",
            "search_package_authority": "driver_private_prebuild",
            "derivation": derivation,
        }
        protocol_sha256 = hashlib.sha256(json.dumps(
            calibration_protocol, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if (value.get("calibration_protocol") != calibration_protocol or
                value.get("calibration_protocol_sha256") != protocol_sha256 or
                value.get("derivation") != derivation):
            problems.append("A/A noise calibration protocol projection differs from frozen inputs")
        submission = Path(str(value.get("submission", ""))).resolve()
        try:
            package_ok = (
                submission.is_dir() and not submission.is_symlink() and
                (submission / "manifest.yaml").is_file() and
                hashlib.sha256((submission / "manifest.yaml").read_bytes()).hexdigest() ==
                value.get("submission_manifest_sha256") and
                _tree_sha256(submission) == value.get("submission_tree_sha256"))
        except OSError:
            package_ok = False
        if not package_ok:
            problems.append("A/A noise calibration package path/hashes are not reproducible")
        else:
            manifest = yaml.safe_load((submission / "manifest.yaml").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or manifest.get("build") != {
                    "command": ["/bin/true"]}:
                problems.append("A/A noise calibration did not measure the prebuilt package")
        noise_sources = {name: digest for name, digest in source_sha256.items()
                         if name != "search_space"}
        if value.get("source_sha256") != noise_sources:
            problems.append("A/A noise calibration lacks the complete executable source seal")
        space_problem = _calibration_space_problem(value.get("space"))
        if space_problem is not None:
            problems.append(
                "A/A noise calibration does not bind the frozen optimization space: "
                f"{space_problem}")
        prebuild_input = Path(str(value.get("prebuild_input_submission", ""))).resolve()
        receipt = value.get("prebuild_receipt")
        try:
            input_ok = (
                prebuild_input.is_dir() and not prebuild_input.is_symlink() and
                prebuild_input != submission and
                hashlib.sha256((prebuild_input / "manifest.yaml").read_bytes()).hexdigest() ==
                value.get("prebuild_input_manifest_sha256") and
                _tree_sha256(prebuild_input) == value.get("prebuild_input_tree_sha256") and
                _grader_package_tree_identity(prebuild_input) ==
                value.get("prebuild_input_package_sha256"))
            receipt_ok = (
                isinstance(receipt, dict) and receipt.get("version") == 1 and
                receipt.get("authority") == "driver_private_prebuild" and
                receipt.get("private_build_override") == ["/bin/true"] and
                receipt.get("submitted_manifest_sha256") ==
                value.get("prebuild_input_manifest_sha256") and
                receipt.get("prebuild_tree_sha256") ==
                value.get("prebuild_input_package_sha256") and
                _digest(receipt.get("built_tree_sha256")) and
                receipt.get("built_tree_sha256") != receipt.get("prebuild_tree_sha256") and
                receipt.get("built_tree_sha256") != receipt.get("sealed_prebuilt_tree_sha256") and
                receipt.get("sealed_prebuilt_tree_sha256") ==
                _grader_package_tree_identity(submission) and
                receipt.get("private_manifest_sha256") ==
                value.get("submission_manifest_sha256") and
                value.get("prebuild_receipt_sha256") == hashlib.sha256(json.dumps(
                    receipt, sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest())
        except (OSError, ValueError):
            input_ok = receipt_ok = False
        if not input_ok or not receipt_ok:
            problems.append("A/A noise calibration prebuild receipt/tree is invalid")
        expected_lineage = {
            "version": 1,
            "stage": "noise_pre_result",
            "pre_result_protocol_sha256": protocol_sha256,
            "raw_input_tree_sha256": value.get("prebuild_input_tree_sha256"),
            "raw_input_package_sha256": value.get("prebuild_input_package_sha256"),
            "output_field": "noise_margin",
        }
        if value.get("calibration_lineage") != expected_lineage:
            problems.append("A/A noise calibration lacks exact pre-result lineage")
        private_authority = value.get("private_shape_calibration")
        expected_private: list[dict[str, Any]] = []
        try:
            nonce = bytes.fromhex(private_authority["nonce_hex"])
            expected_public = _calibration_sample(
                list(train_rows or []),
                per_family=int(space["confirmation_samples_per_family"]),
                families=list(space["confirmation_families"]))
            expected_private = [_calibration_private_capsule(
                row, nonce=nonce, phase="confirm") for row in expected_public]
            records = private_authority["records"]
            private_ok = (
                private_authority.get("version") == 1 and
                private_authority.get("authority") ==
                "trusted_broker_private_capsule_independent_calibration_nonce" and
                len(nonce) == 32 and private_authority.get("phase") == "confirm" and
                private_authority.get("split") == "train" and
                private_authority.get("nonce_sha256") == hashlib.sha256(nonce).hexdigest() and
                private_authority.get("records_sha256") == hashlib.sha256(json.dumps(
                    records, sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest() and
                records == [{"public": public, "private": private}
                            for public, private in zip(
                                expected_public, expected_private, strict=True)])
        except (KeyError, TypeError, ValueError):
            private_ok = False
        if not private_ok:
            problems.append("A/A noise calibration private shapes differ from broker semantics")
        toolchain = value.get("toolchain_identity")
        required_tools = {"python", "bwrap", "spike_gcc", "spike_spike",
                          "spike_objdump", "k1_clang", "k1_objcopy", "ssh", "scp"}
        if isinstance(receipt, dict):
            required_tools.update(
                f"prebuild_command_{index}" for index, _ in enumerate(
                    receipt.get("real_build_commands", ())))
            required_tools.update(
                f"private_build_override_{index}" for index, _ in enumerate(
                    receipt.get("private_build_override", ())))
        if not isinstance(toolchain, dict) or not required_tools <= set(toolchain) or any(
                not isinstance(toolchain[name], dict) or
                not Path(str(toolchain[name].get("path", ""))).resolve().is_file() or
                hashlib.sha256(Path(str(toolchain[name]["path"])).resolve().read_bytes()
                               ).hexdigest() != toolchain[name].get("sha256") or
                Path(str(toolchain[name]["path"])).resolve().stat().st_mode & 0o777 !=
                toolchain[name].get("mode")
                for name in required_tools):
            problems.append("A/A noise calibration timing toolchain identity is incomplete")
        observations, pairs = value.get("observations"), value.get("pairs")
        families = list(space.get("confirmation_families", ()))
        repeats = int(space.get("measurement_repeats", 0))
        pair_orders = ["parent_candidate", "candidate_parent", "candidate_parent",
                       "parent_candidate", "parent_candidate", "candidate_parent"]
        state_ready, state_pair_ok = _calibration_k1_state_gates()
        board_environment = dict(space.get("board_environment", {}))

        def valid_spike_gates(row: dict[str, Any]) -> bool:
            gates = row.get("spike_gates")
            if not isinstance(gates, dict) or set(gates) != {"parent", "candidate"}:
                return False
            for gate in gates.values():
                checks = gate.get("checks") if isinstance(gate, dict) else None
                if (not isinstance(gate, dict) or gate.get("compile_ok") is not True or
                        gate.get("k1_compile_ok") is not True or gate.get("passed") is not True or
                        not isinstance(checks, dict) or any(checks.get(name) is not True for name in (
                            "rvv_correctness", "instruction_evidence", "vlen_256",
                            "cycle_measurement")) or not _digest(gate.get("kernel_text_sha256"))):
                    return False
            return True

        def valid_k1_evidence(evidence: Any, *, observation: dict[str, Any], side: str,
                              seed: int, elapsed_ns: int, calls: int) -> bool:
            if not isinstance(evidence, dict):
                return False
            checks = evidence.get("checks")
            metrics = evidence.get("metrics")
            monitor = evidence.get("monitor")
            expected_digest = observation.get(
                "baseline_code_sha256" if side == "parent" else
                "candidate_code_sha256")
            return (
                evidence.get("capsule") == observation.get("capsule_id") and
                evidence.get("family") == observation.get("family") and
                evidence.get("status") == "pass" and
                evidence.get("seed") == seed and
                isinstance(checks, dict) and bool(checks) and all(checks.values()) and
                isinstance(metrics, dict) and metrics.get("wall_ns") == elapsed_ns and
                metrics.get("calls") == calls and
                evidence.get("kernel_text_sha256") == expected_digest and
                isinstance(evidence.get("receipt_nonce"), int) and
                not isinstance(evidence.get("receipt_nonce"), bool) and
                evidence["receipt_nonce"] > 0 and
                _digest(evidence.get("local_sha256")) and
                evidence.get("remote_sha256") == evidence.get("local_sha256") and
                isinstance(monitor, dict) and monitor.get("returncode") == 0 and
                evidence.get("ssh_returncode") == 0 and
                _positive_number(evidence.get("board_wall_seconds")))

        def valid_condition(condition: Any, *, observation: dict[str, Any],
                            expected_valid: bool, arrays: list[list[int]]) -> bool:
            if not isinstance(condition, dict):
                return False
            pair_id = condition.get("pair_id")
            attempt_id = condition.get("attempt_id")
            if (not isinstance(pair_id, int) or isinstance(pair_id, bool) or
                    not 0 <= pair_id < repeats or
                    not isinstance(attempt_id, int) or isinstance(attempt_id, bool) or
                    attempt_id < 0 or condition.get("order") != pair_orders[pair_id] or
                    condition.get("valid") is not expected_valid):
                return False
            settle = condition.get("settle_probes")
            before, after = condition.get("before"), condition.get("after")
            if (not isinstance(settle, list) or not settle or
                    len(settle) > int(board_environment.get("settle_attempts", 0)) or
                    any(state_ready(probe, board_environment) for probe in settle[:-1]) or
                    before != settle[-1] or not state_ready(settle[-1], board_environment) or
                    state_pair_ok(before, after, board_environment) is not expected_valid):
                return False
            seed = condition.get("seed")
            measurements = condition.get("measurements")
            if (not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0 or
                    not isinstance(measurements, dict) or
                    set(measurements) != {"parent", "candidate"}):
                return False
            for side, elapsed_samples, call_samples in (
                    ("parent", arrays[0], arrays[1]),
                    ("candidate", arrays[2], arrays[3])):
                measurement = measurements.get(side)
                if not isinstance(measurement, dict):
                    return False
                elapsed_ns, calls = measurement.get("elapsed_ns"), measurement.get("calls")
                if (not isinstance(elapsed_ns, int) or isinstance(elapsed_ns, bool) or
                        elapsed_ns <= 0 or not isinstance(calls, int) or
                        isinstance(calls, bool) or calls <= 0 or
                        measurement.get("seed") != seed or
                        not valid_k1_evidence(
                            measurement.get("evidence"), observation=observation,
                            side=side, seed=seed, elapsed_ns=elapsed_ns, calls=calls)):
                    return False
                if expected_valid and (
                        elapsed_ns != elapsed_samples[pair_id] or
                        calls != call_samples[pair_id]):
                    return False
            return True

        expected_pair_rows: list[dict[str, Any]] = []
        raw_observations_ok = True
        if (not isinstance(observations, list) or len(observations) != len(families) or
                {row.get("family") for row in observations if isinstance(row, dict)} !=
                set(families) or
                any(not isinstance(row, dict) or row.get("correctness_ok") is not True or
                    row.get("baseline_code_sha256") != row.get("candidate_code_sha256") or
                    not valid_spike_gates(row) or
                    len(row.get("board_condition_pairs", ())) != repeats
                    for row in observations)):
            problems.append("A/A noise calibration lacks six correct identical-text family observations")
            raw_observations_ok = False
        elif [(row.get("capsule_id"), row.get("family")) for row in observations] != [
                (row["id"], row["family"]) for row in expected_private]:
            problems.append("A/A noise observations differ from the exact private panel")
            raw_observations_ok = False
        if raw_observations_ok:
            for observation in observations:
                arrays = [observation.get(name) for name in (
                    "baseline_elapsed_ns", "baseline_calls",
                    "candidate_elapsed_ns", "candidate_calls")]
                if any(not isinstance(samples, list) or len(samples) != repeats or any(
                        not isinstance(sample, int) or isinstance(sample, bool) or sample <= 0
                        for sample in samples) for samples in arrays):
                    raw_observations_ok = False
                    break
                condition_pairs = observation.get("board_condition_pairs")
                excluded_pairs = observation.get("excluded_board_condition_pairs")
                if (observation.get("pair_orders") != pair_orders or
                        not isinstance(condition_pairs, list) or
                        not isinstance(excluded_pairs, list) or
                        len(excluded_pairs) > int(board_environment.get(
                            "maximum_invalid_pair_replacements_per_capsule", -1)) or
                        any(not valid_condition(
                            condition, observation=observation, expected_valid=True,
                            arrays=arrays) or condition.get("pair_id") != pair_index
                            for pair_index, condition in enumerate(condition_pairs)) or
                        any(not valid_condition(
                            condition, observation=observation, expected_valid=False,
                            arrays=arrays) for condition in excluded_pairs)):
                    raw_observations_ok = False
                    break
                all_attempts = [*condition_pairs, *excluded_pairs]
                attempt_ids = [condition["attempt_id"] for condition in all_attempts]
                chronological = sorted(all_attempts, key=lambda condition: condition["attempt_id"])
                current_pair_id = 0
                transcript_ok = True
                for condition in chronological:
                    if condition["pair_id"] != current_pair_id:
                        transcript_ok = False
                        break
                    if condition["valid"]:
                        current_pair_id += 1
                if (sorted(attempt_ids) != list(range(len(all_attempts))) or
                        [condition["attempt_id"] for condition in condition_pairs] != sorted(
                            condition["attempt_id"] for condition in condition_pairs) or
                        [condition["attempt_id"] for condition in excluded_pairs] != sorted(
                            condition["attempt_id"] for condition in excluded_pairs) or
                        not transcript_ok or current_pair_id != repeats or
                        observation.get("k1_program_count") != 2 * len(all_attempts)):
                    raw_observations_ok = False
                    break
                for pair_index, values in enumerate(zip(*arrays, strict=True)):
                    base_elapsed, base_calls, candidate_elapsed, candidate_calls = values
                    ratio = (base_elapsed / base_calls) / (
                        candidate_elapsed / candidate_calls)
                    expected_pair_rows.append({
                        "capsule_id": observation["capsule_id"],
                        "family": observation["family"],
                        "pair_index": pair_index,
                        "speedup_ratio": ratio,
                        "absolute_unit_deviation": abs(ratio - 1.0),
                    })
        if not raw_observations_ok:
            problems.append("A/A noise calibration lacks positive retained raw K1 samples")
        if not isinstance(pairs, list) or pairs != expected_pair_rows:
            problems.append("A/A noise pairs do not exactly recompute from retained K1 samples")
        elif expected_pair_rows:
            ratios = [float(row["speedup_ratio"]) for row in expected_pair_rows]
            maximum_log = max(abs(math.log(ratio)) for ratio in ratios)
            padded = maximum_log + 0.005
            margin = max(0.02, math.ceil((math.exp(padded) - 1.0) * 1000.0) / 1000.0)
            exact = (
                math.isclose(float(value.get("maximum_absolute_pair_deviation", -1)),
                             max(abs(ratio - 1.0) for ratio in ratios),
                             rel_tol=1e-12, abs_tol=1e-12) and
                math.isclose(float(value.get("maximum_absolute_log_ratio", -1)), maximum_log,
                             rel_tol=1e-12, abs_tol=1e-12) and
                math.isclose(float(value.get("padded_log_half_width", -1)), padded,
                             rel_tol=1e-12, abs_tol=1e-12) and
                float(value.get("derived_noise_margin", -1)) == margin and
                float(value.get("upper_speedup_bound", -1)) == 1.0 + margin and
                float(value.get("lower_speedup_bound", -1)) == 1.0 / (1.0 + margin) and
                float(space.get("noise_margin", -1)) == margin)
            if not exact:
                problems.append(
                    "A/A derived margin/reciprocal bound do not exactly equal optimization_space")
        return problems
    if (value.get("version") != 1 or value.get("paid_work") is not False or
            value.get("heldout_opened") is not False or
            value.get("protocol_state_mutated") is not False):
        problems.append(
            f"{label} calibration must be version 1, public-only, unpaid, and non-mutating")
    if value.get("public_split_sha256") != train_sha256:
        problems.append(f"{label} calibration does not bind the exact public train split")
    expected_public_context = {
        "authority": "complete_public_train",
        "capsule_ids": [str(row["id"]) for row in (train_rows or [])],
        "row_count": len(train_rows or []),
        "rows_sha256": hashlib.sha256(json.dumps(
            train_rows or [], sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
    }
    if value.get("public_context") != expected_public_context:
        problems.append(f"{label} calibration did not retain the complete public context")
    if value.get("source_sha256") != source_sha256:
        problems.append(f"{label} calibration lacks the exact executable source-hash seal")
    if (_calibration_space_problem(value.get("space")) is not None or
            value.get("space_sha256") != source_sha256.get("search_space")):
        problems.append(f"{label} calibration does not bind the exact frozen optimization space")
    submission = Path(str(value.get("submission", ""))).resolve()
    try:
        measured_manifest = yaml.safe_load(
            (submission / "manifest.yaml").read_text(encoding="utf-8"))
        package_ok = (
            submission.is_dir() and not submission.is_symlink() and
            (submission / "manifest.yaml").is_file() and
            hashlib.sha256((submission / "manifest.yaml").read_bytes()).hexdigest() ==
            value.get("submission_manifest_sha256") and
            _tree_sha256(submission) == value.get("submission_tree_sha256"))
    except OSError:
        package_ok = False
    if not package_ok:
        problems.append(f"{label} calibration package path/hashes are not reproducible")
    elif (not isinstance(measured_manifest, dict) or
          measured_manifest.get("build") != {"command": ["/bin/true"]}):
        problems.append(f"{label} calibration did not measure the controller-prebuilt package")
    prebuild_input = Path(str(value.get("prebuild_input_submission", ""))).resolve()
    try:
        prebuild_input_ok = (
            prebuild_input.is_dir() and not prebuild_input.is_symlink() and
            prebuild_input != submission and
            (prebuild_input / "manifest.yaml").is_file() and
            hashlib.sha256((prebuild_input / "manifest.yaml").read_bytes()).hexdigest() ==
            value.get("prebuild_input_manifest_sha256") and
            _tree_sha256(prebuild_input) == value.get("prebuild_input_tree_sha256") and
            _grader_package_tree_identity(prebuild_input) ==
            value.get("prebuild_input_package_sha256"))
    except OSError:
        prebuild_input_ok = False
    receipt = value.get("prebuild_receipt")
    receipt_digest = hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest() if isinstance(receipt, dict) else ""
    receipt_ok = (
        isinstance(receipt, dict) and receipt.get("version") == 1 and
        receipt.get("authority") == "driver_private_prebuild" and
        receipt.get("private_build_override") == ["/bin/true"] and
        receipt.get("submitted_manifest_sha256") ==
        value.get("prebuild_input_manifest_sha256") and
        receipt.get("private_manifest_sha256") ==
        value.get("submission_manifest_sha256") and
        receipt.get("prebuild_tree_sha256") ==
        (value.get("prebuild_input_package_sha256") if prebuild_input_ok else None) and
        _digest(receipt.get("built_tree_sha256")) and
        receipt.get("built_tree_sha256") != receipt.get("prebuild_tree_sha256") and
        receipt.get("built_tree_sha256") != receipt.get("sealed_prebuilt_tree_sha256") and
        receipt.get("sealed_prebuilt_tree_sha256") ==
        (_grader_package_tree_identity(submission) if package_ok else None) and
        isinstance(receipt.get("real_build_commands"), list) and
        bool(receipt.get("real_build_commands")) and
        isinstance(receipt.get("real_build_logs"), list) and
        bool(receipt.get("real_build_logs")) and
        isinstance(receipt.get("built_entrypoint_identity"), list) and
        len(receipt["built_entrypoint_identity"]) == 2 and
        isinstance(receipt["built_entrypoint_identity"][0], int) and
        _digest(receipt["built_entrypoint_identity"][1]) and
        value.get("prebuild_receipt_sha256") == receipt_digest)
    if not prebuild_input_ok or not receipt_ok:
        problems.append(f"{label} calibration lacks a reproducible trusted prebuild receipt")
    expected_cost_lineage = None
    if isinstance(noise_authority, dict) and _digest(noise_authority_sha256):
        expected_cost_lineage = {
            "version": 1,
            "stage": "cost_post_noise_result",
            "predecessor_stage": "noise_pre_result",
            "noise_authority": value.get("calibration_lineage", {}).get(
                "noise_authority") if isinstance(value.get("calibration_lineage"), dict)
                else None,
            "noise_authority_sha256": noise_authority_sha256,
            "pre_result_protocol_sha256": noise_authority.get(
                "calibration_protocol_sha256"),
            "derived_noise_margin": float(noise_authority.get(
                "derived_noise_margin", -1)),
            "raw_input_tree_sha256": noise_authority.get(
                "prebuild_input_tree_sha256"),
            "raw_input_package_sha256": noise_authority.get(
                "prebuild_input_package_sha256"),
            "final_space_sha256": source_sha256.get("search_space"),
        }
    lineage = value.get("calibration_lineage")
    lineage_path_ok = False
    if isinstance(lineage, dict):
        try:
            lineage_path = Path(str(lineage.get("noise_authority", ""))).resolve()
            lineage_path_ok = (
                lineage_path.is_file() and not lineage_path.is_symlink() and
                hashlib.sha256(lineage_path.read_bytes()).hexdigest() ==
                noise_authority_sha256)
        except OSError:
            lineage_path_ok = False
    if (expected_cost_lineage is None or lineage != expected_cost_lineage or
            not lineage_path_ok or
            value.get("prebuild_input_tree_sha256") !=
            noise_authority.get("prebuild_input_tree_sha256") or
            value.get("prebuild_input_package_sha256") !=
            noise_authority.get("prebuild_input_package_sha256") or
            float(space.get("noise_margin", -1)) !=
            float(noise_authority.get("derived_noise_margin", -2))):
        problems.append(
            f"{label} calibration lacks exact A/A predecessor, final-margin, and raw-tree lineage")
    toolchain = value.get("toolchain_identity")
    expected_tools = {
        "python", "bwrap", "spike_gcc", "spike_spike", "spike_objdump",
        *(f"prebuild_command_{index}" for index, _ in enumerate(
            receipt.get("real_build_commands", ()) if isinstance(receipt, dict) else ())),
        *(f"private_build_override_{index}" for index, _ in enumerate(
            receipt.get("private_build_override", ()) if isinstance(receipt, dict) else ())),
    }
    if label == "k1_calibration":
        expected_tools.update({"k1_clang", "k1_objcopy", "ssh", "scp"})
    toolchain_ok = isinstance(toolchain, dict) and expected_tools <= set(toolchain)
    if toolchain_ok:
        for name in expected_tools:
            identity = toolchain[name]
            path = Path(str(identity.get("path", ""))).resolve() \
                if isinstance(identity, dict) else Path("unresolved")
            try:
                valid = (
                    isinstance(identity, dict) and path.is_file() and not path.is_symlink() and
                    hashlib.sha256(path.read_bytes()).hexdigest() == identity.get("sha256") and
                    path.stat().st_mode & 0o777 == identity.get("mode"))
            except OSError:
                valid = False
            if not valid:
                toolchain_ok = False
                break
    if not toolchain_ok:
        problems.append(f"{label} calibration timing toolchain identity is incomplete or stale")

    private_authority = value.get("private_shape_calibration")
    expected_private: list[dict[str, Any]] = []
    phase = "screen" if label == "spike_calibration" else "confirm"
    per_family = int(space[
        "screen_samples_per_family" if phase == "screen"
        else "confirmation_samples_per_family"])
    try:
        nonce_hex = private_authority["nonce_hex"]
        nonce = bytes.fromhex(nonce_hex)
        records = private_authority["records"]
        expected_public = _calibration_sample(
            list(train_rows or []), per_family=per_family,
            families=list(space["confirmation_families"]))
        expected_private = [_calibration_private_capsule(
            row, nonce=nonce, phase=phase) for row in expected_public]
        records_digest = hashlib.sha256(json.dumps(
            records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        private_ok = (
            isinstance(private_authority, dict) and private_authority.get("version") == 1 and
            private_authority.get("authority") ==
            "trusted_broker_private_capsule_independent_calibration_nonce" and
            private_authority.get("phase") == phase and
            private_authority.get("split") == "train" and len(nonce) == 32 and
            private_authority.get("nonce_sha256") == hashlib.sha256(nonce).hexdigest() and
            private_authority.get("records_sha256") == records_digest and
            records == [{"public": public, "private": private}
                        for public, private in zip(
                            expected_public, expected_private, strict=True)])
    except (KeyError, TypeError, ValueError):
        private_ok = False
    if not private_ok:
        problems.append(f"{label} calibration private shapes differ from the trusted broker mechanism")
    budget = _mapping(space.get("budget"), "search budget")
    checks = value.get("checks")
    if label == "k1_calibration":
        if (value.get("kind") != "cpu_host_trusted_search_k1_program_calibration" or
                value.get("status") != "pass"):
            problems.append("K1 calibration has the wrong kind")
        if checks != {"all_passed": True, "max_within_planning_upper": True,
                      "mean_within_expected": True}:
            problems.append("K1 calibration pass/budget checks are not all true")
        declared = value.get("declared")
        expected_declared = {
            "expected_seconds_per_program": float(budget["expected_seconds_per_k1_program"]),
            "planning_upper_seconds_per_program": float(
                budget["planning_upper_seconds_per_k1_program"]),
        }
        if declared != expected_declared:
            problems.append("K1 calibration declared costs differ from the frozen search budget")
        programs, statistics = value.get("programs"), value.get("statistics")
        calibration_capsule = value.get("calibration_capsule")
        expected_capsule = expected_private[0] if expected_private else None
        if calibration_capsule != ({
                key: expected_capsule.get(key)
                for key in ("id", "sha256", "family", "split")
        } if expected_capsule else None):
            problems.append("K1 calibration capsule differs from the trusted private sample")
        trusted = value.get("trusted_evaluation_observations")
        trusted_observation = (trusted[0] if isinstance(trusted, list) and len(trusted) == 1
                               and isinstance(trusted[0], dict) else {})
        valid_pairs = trusted_observation.get("board_condition_pairs")
        excluded_pairs = trusted_observation.get("excluded_board_condition_pairs")
        maximum_replacements = int(space.get("board_environment", {}).get(
            "maximum_invalid_pair_replacements_per_capsule", -1))
        expected_program_count = trusted_observation.get("k1_program_count")
        program_count_bound = (
            isinstance(valid_pairs, list) and
            len(valid_pairs) == int(space["measurement_repeats"]) and
            all(isinstance(pair, dict) and pair.get("valid") is True
                for pair in valid_pairs) and
            isinstance(excluded_pairs, list) and
            len(excluded_pairs) <= maximum_replacements and
            all(isinstance(pair, dict) and pair.get("valid") is False
                for pair in excluded_pairs) and
            type(expected_program_count) is int and
            expected_program_count == 2 * (len(valid_pairs) + len(excluded_pairs))
        )
        if (not isinstance(trusted, list) or len(trusted) != 1 or
                not isinstance(trusted[0], dict) or
                trusted[0].get("capsule_id") != (expected_capsule or {}).get("id") or
                trusted[0].get("family") != (expected_capsule or {}).get("family") or
                trusted[0].get("correctness_ok") is not True or
                not program_count_bound):
            problems.append("K1 calibration trusted observation is incomplete")
        if (not isinstance(programs, list) or not program_count_bound or
                len(programs) != expected_program_count or
                not isinstance(statistics, dict) or statistics.get("count") != len(programs)):
            problems.append(
                "K1 calibration must retain every valid and board-condition-rejected program")
        elif (any(not isinstance(row, dict) or row.get("index") != index or
                  row.get("capsule_id") != (expected_capsule or {}).get("id") or
                  row.get("family") != (expected_capsule or {}).get("family") or
                  row.get("status") != "pass" or
                  not isinstance(row.get("checks"), dict) or
                  not row["checks"] or not all(row["checks"].values()) or
                  not isinstance(row.get("start_monotonic_ns"), int) or
                  not isinstance(row.get("end_monotonic_ns"), int) or
                  row["end_monotonic_ns"] <= row["start_monotonic_ns"] or
                  not _positive_number(row.get("total_seconds")) or
                  not math.isclose(
                      float(row["total_seconds"]),
                      (row["end_monotonic_ns"] - row["start_monotonic_ns"]) / 1e9,
                      rel_tol=1e-12, abs_tol=1e-12) or
                  not isinstance(row.get("evidence"), dict) or
                  row["evidence"].get("capsule") != row.get("capsule_id") or
                  row["evidence"].get("family") != row.get("family") or
                  row["evidence"].get("status") != "pass" or
                  row["evidence"].get("checks") != row.get("checks") or
                  row["evidence"].get("metrics") != row.get("metrics") or
                  row["evidence"].get("monitor") != row.get("monitor") or
                  row["evidence"].get("kernel_text_sha256") !=
                  row.get("kernel_text_sha256") or
                  row["evidence"].get("seed") != row.get("seed") or
                  not isinstance(row["evidence"].get("receipt_nonce"), int) or
                  not _digest(row["evidence"].get("local_sha256")) or
                  row["evidence"].get("remote_sha256") !=
                  row["evidence"].get("local_sha256") or
                  not _positive_number((row.get("metrics") or {}).get("wall_ns")) or
                  not _positive_number((row.get("metrics") or {}).get("calls"))
                  for index, row in enumerate(programs))):
            problems.append("K1 calibration program observations are incomplete or non-passing")
        else:
            totals = [float(row["total_seconds"]) for row in programs]
            mean = sum(totals) / len(totals)
            median = stats.median(totals)
            ordered = sorted(totals)
            p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
            maximum = max(totals)
            if (not math.isclose(float(statistics.get("mean_seconds", -1)), mean,
                                 rel_tol=1e-12, abs_tol=1e-12) or
                    not math.isclose(float(statistics.get("median_seconds", -1)), median,
                                     rel_tol=1e-12, abs_tol=1e-12) or
                    not math.isclose(float(statistics.get("p95_seconds", -1)), p95,
                                     rel_tol=1e-12, abs_tol=1e-12) or
                    not math.isclose(float(statistics.get("max_seconds", -1)), maximum,
                                     rel_tol=1e-12, abs_tol=1e-12) or
                    mean > expected_declared["expected_seconds_per_program"] or
                    maximum > expected_declared["planning_upper_seconds_per_program"]):
                problems.append("K1 calibration statistics do not support the frozen budget")
    elif label == "spike_calibration":
        expected_capsules = int(space["screen_samples_per_family"]) * len(
            space.get("confirmation_families", ()))
        maximum_evaluations = int(budget["maximum_screen_candidate_evaluations"])
        declared = value.get("declared")
        if value.get("kind") != "cpu_host_trusted_search_spike_candidate_calibration":
            problems.append("Spike calibration has the wrong kind")
        if value.get("status") != "pass" or checks != {
                "all_observations_passed": True, "projection_within_expected_budget": True}:
            problems.append("Spike calibration pass/budget checks are not all true")
        if declared != {
                "expected_spike_screen_seconds": float(budget["expected_spike_screen_seconds"]),
                "maximum_candidate_evaluations": maximum_evaluations,
        }:
            problems.append("Spike calibration declaration differs from the frozen search budget")
        candidate = value.get("candidate_evaluation_seconds")
        projected = value.get("projected_max_screen_seconds")
        observations = value.get("observations")
        if (value.get("capsules") != expected_capsules or
                value.get("completed_observations") != expected_capsules or
                value.get("maximum_candidate_evaluations") != maximum_evaluations or
                not isinstance(value.get("start_monotonic_ns"), int) or
                not isinstance(value.get("end_monotonic_ns"), int) or
                value["end_monotonic_ns"] <= value["start_monotonic_ns"] or
                not _positive_number(candidate) or not _positive_number(projected) or
                not math.isclose(
                    float(candidate),
                    (value["end_monotonic_ns"] - value["start_monotonic_ns"]) / 1e9,
                    rel_tol=1e-12, abs_tol=1e-12) or
                not math.isclose(float(projected), float(candidate) * maximum_evaluations,
                                 rel_tol=1e-12, abs_tol=1e-12) or
                float(projected) > float(budget["expected_spike_screen_seconds"])):
            problems.append("Spike calibration measured counts/projection do not support the budget")
        if (not isinstance(observations, list) or len(observations) != expected_capsules or
                any(not isinstance(row, dict) or row.get("correctness_ok") is not True or
                    not isinstance(row.get("capsule_id"), str) or not row.get("capsule_id") or
                    row.get("family") not in space.get("confirmation_families", ())
                    for row in observations)):
            problems.append("Spike calibration lacks complete retained public observations")
        elif (len({row["capsule_id"] for row in observations}) != expected_capsules or
              {(row["capsule_id"], row["family"]) for row in observations} !=
              {(row["id"], row["family"]) for row in expected_private}):
            problems.append("Spike calibration observations differ from the exact private sample")
    elif label == "confirmation_overhead_calibration":
        expected_declared = {
            "package_build": {
                "expected_seconds": float(
                    budget["expected_seconds_per_confirmation_package_build"]),
                "planning_upper_seconds": float(
                    budget["planning_upper_seconds_per_confirmation_package_build"]),
            },
            "compiler_invocation": {
                "expected_seconds": float(
                    budget["expected_seconds_per_confirmation_compiler_invocation"]),
                "planning_upper_seconds": float(
                    budget["planning_upper_seconds_per_confirmation_compiler_invocation"]),
            },
            "spike_check": {
                "expected_seconds": float(
                    budget["expected_seconds_per_confirmation_spike_check"]),
                "planning_upper_seconds": float(
                    budget["planning_upper_seconds_per_confirmation_spike_check"]),
            },
        }
        if value.get("kind") != "cpu_host_confirmation_overhead_calibration":
            problems.append("confirmation-overhead calibration has the wrong kind")
        if value.get("status") != "pass" or checks != {
                "all_toolchain_stages_passed": True, "all_expected_costs_within_budget": True,
                "all_maximum_costs_within_planning_upper": True}:
            problems.append("confirmation-overhead calibration pass/budget checks are not all true")
        if value.get("declared") != expected_declared:
            problems.append(
                "confirmation-overhead calibration declaration differs from the frozen budget")
        capsules = value.get("public_capsules")
        repeats = value.get("calibration_repeats_per_capsule")
        expected_capsules = len(space.get("confirmation_families", ()))
        count = expected_capsules * int(repeats) if isinstance(repeats, int) else -1
        runtime_capsules = sum(
            row.get("family") == "runtime_parallel" for row in expected_private)
        stage_counts = {
            "package_build": count,
            "compiler_invocation": count + runtime_capsules * 2,
            "spike_check": count,
        }
        expected_capsule_rows = [{key: row.get(key) for key in ("id", "sha256", "family")}
                                 for row in expected_private]
        if (not isinstance(capsules, list) or capsules != expected_capsule_rows or
                not isinstance(repeats, int) or repeats != 2 or
                len({row.get("id") for row in capsules if isinstance(row, dict)}) !=
                expected_capsules or
                any(not isinstance(row, dict) or not _digest(row.get("sha256"))
                    for row in capsules) or
                value.get("spike_statuses") != ["pass"] * count):
            problems.append("confirmation-overhead calibration public measurements are incomplete")
        trusted = value.get("trusted_evaluation_observations")
        if (not isinstance(trusted, list) or len(trusted) != expected_capsules or
                any(not isinstance(row, dict) or row.get("correctness_ok") is not True
                    or row.get("calibration_authority") !=
                    "exact_confirmation_pre_k1_stages_without_k1"
                    for row in trusted) or
                {(row.get("capsule_id"), row.get("family")) for row in trusted} !=
                {(row["id"], row["family"]) for row in expected_private}):
            problems.append("confirmation-overhead trusted observations differ from private sample")
        for name, limits in expected_declared.items():
            observation = value.get(name)
            if (not isinstance(observation, dict) or
                    observation.get("count") != stage_counts[name] or
                    not _positive_number(observation.get("mean_seconds")) or
                    not _positive_number(observation.get("max_seconds")) or
                    float(observation["mean_seconds"]) > limits["expected_seconds"] or
                    float(observation["max_seconds"]) > limits["planning_upper_seconds"]):
                problems.append(f"confirmation-overhead {name} measurements exceed/fail the budget")
        stage_observations = value.get("stage_observations")
        if not isinstance(stage_observations, dict) or set(stage_observations) != set(
                expected_declared):
            problems.append("confirmation-overhead lacks complete raw stage observations")
        else:
            for name, rows in stage_observations.items():
                if (not isinstance(rows, list) or len(rows) != stage_counts[name] or
                        any(not isinstance(row, dict) or row.get("index") != index or
                            row.get("stage") != name or
                            row.get("side") not in {"parent", "candidate"} or
                            row.get("capsule_id") not in {
                                capsule["id"] for capsule in expected_private} or
                            row.get("family") != next((
                                capsule["family"] for capsule in expected_private
                                if capsule["id"] == row.get("capsule_id")), None) or
                            row.get("status") != "pass" or
                            not isinstance(row.get("start_monotonic_ns"), int) or
                            not isinstance(row.get("end_monotonic_ns"), int) or
                            row["end_monotonic_ns"] <= row["start_monotonic_ns"] or
                            not _positive_number(row.get("wall_seconds")) or
                            not math.isclose(
                                float(row["wall_seconds"]),
                                (row["end_monotonic_ns"] -
                                 row["start_monotonic_ns"]) / 1e9,
                                rel_tol=1e-12, abs_tol=1e-12) or
                            not isinstance(row.get("evidence"), dict) or
                            not row.get("evidence")
                            for index, row in enumerate(rows))):
                    problems.append(
                        f"confirmation-overhead {name} raw observations are incomplete")
                    continue
                expected_membership = (
                    {(capsule["id"], side, mode)
                     for capsule in expected_private for side in ("parent", "candidate")
                     for mode in (("rvv", "rvv_multicore")
                                  if capsule["family"] == "runtime_parallel" else ("rvv",))}
                    if name == "compiler_invocation" else
                    {(capsule["id"], side, None) for capsule in expected_private
                     for side in ("parent", "candidate")})
                if ({(row["capsule_id"], row["side"], row.get("mode")) for row in rows} !=
                        expected_membership):
                    problems.append(
                        f"confirmation-overhead {name} capsule/stage membership differs")
                    continue
                totals = [float(row["wall_seconds"]) for row in rows]
                summary = value.get(name)
                ordered = sorted(totals)
                median = stats.median(totals)
                p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
                if (not isinstance(summary, dict) or
                        summary.get("count") != len(totals) or
                        not math.isclose(float(summary.get("mean_seconds", -1)),
                                         sum(totals) / len(totals),
                                         rel_tol=1e-12, abs_tol=1e-12) or
                        not math.isclose(float(summary.get("median_seconds", -1)), median,
                                         rel_tol=1e-12, abs_tol=1e-12) or
                        not math.isclose(float(summary.get("p95_seconds", -1)), p95,
                                         rel_tol=1e-12, abs_tol=1e-12) or
                        not math.isclose(float(summary.get("max_seconds", -1)), max(totals),
                                         rel_tol=1e-12, abs_tol=1e-12)):
                    problems.append(
                        f"confirmation-overhead {name} summary differs from raw observations")
    return problems


@dataclass(frozen=True)
class HostArm:
    id: str
    order: int
    capabilities: frozenset[str]
    treatment: str

    @staticmethod
    def parse(raw: Any, index: int) -> "HostArm":
        raw = _mapping(raw, f"arms[{index}]")
        arm = HostArm(
            id=str(raw.get("id", "")),
            order=int(raw.get("order", 0)),
            capabilities=frozenset(str(v) for v in raw.get("capabilities", ()) or ()),
            treatment=str(raw.get("treatment", "")),
        )
        if not arm.id or not arm.treatment or not arm.capabilities:
            raise ValueError(f"arms[{index}] requires id, treatment, and capabilities")
        return arm


@dataclass(frozen=True)
class HostPreflight:
    errors: tuple[str, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    evidence: dict[str, Any]

    @property
    def ready(self) -> bool:
        return not self.errors and not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": "GO" if self.ready else "NO_GO",
            "ready": self.ready,
            "errors": list(self.errors),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class HostExperimentSpec:
    version: int
    status: str
    label: str
    target_contract: str
    dialect_plan: str
    task: str
    development_corpus: dict[str, Any]
    paper_holdouts: tuple[str, ...]
    arms: tuple[HostArm, ...]
    agent: dict[str, Any]
    telemetry: dict[str, Any]
    grading: dict[str, Any]
    search: dict[str, Any]
    analysis: dict[str, Any]
    replacement: dict[str, Any]
    environment: dict[str, Any]
    freeze: dict[str, Any]
    source_path: Path | None = None

    @staticmethod
    def parse(raw: Any, *, source_path: Path | None = None) -> "HostExperimentSpec":
        raw = _mapping(raw, "CPU-host experiment")
        validate_or_raise(raw, "cpu_host_experiment")
        arms = tuple(HostArm.parse(v, i) for i, v in enumerate(raw.get("arms", ())))
        spec = HostExperimentSpec(
            version=int(raw["version"]), status=str(raw["status"]), label=str(raw["label"]),
            target_contract=str(raw["target_contract"]), dialect_plan=str(raw.get("dialect_plan", "")),
            task=str(raw.get("task", "")),
            development_corpus=dict(_mapping(raw["development_corpus"], "development_corpus")),
            paper_holdouts=tuple(str(v) for v in raw.get("paper_holdouts", ()) or ()),
            arms=arms, agent=dict(_mapping(raw["agent"], "agent")),
            telemetry=dict(_mapping(raw["telemetry"], "telemetry")),
            grading=dict(_mapping(raw["grading"], "grading")),
            search=dict(_mapping(raw["search"], "search")),
            analysis=dict(_mapping(raw["analysis"], "analysis")),
            replacement=dict(_mapping(raw.get("replacement", {}), "replacement")),
            environment=dict(_mapping(raw["environment"], "environment")),
            freeze=dict(_mapping(raw["freeze"], "freeze")), source_path=source_path,
        )
        spec._validate()
        return spec

    @staticmethod
    def from_yaml(path: str | Path) -> "HostExperimentSpec":
        source = Path(path).resolve()
        return HostExperimentSpec.parse(yaml.safe_load(source.read_text(encoding="utf-8")),
                                        source_path=source)

    def _validate(self) -> None:
        if self.version != 1 or self.status not in {
                "draft", "protocol_frozen", "campaign_complete",
                "campaign_complete_unpromoted"}:
            raise ValueError(
                "CPU-host experiment must be version 1 and draft, protocol_frozen, "
                "campaign_complete, or campaign_complete_unpromoted")
        if not self.label or not self.target_contract or not self.dialect_plan or not self.task:
            raise ValueError("label, target_contract, dialect_plan, and task are required")
        if self.replacement:
            expected_replacement = {
                "kind", "predecessor_campaign_run_id", "predecessor_protocol_inputs_sha256",
                "scope", "treatment_or_provider_started", "excluded_from_arm_outcomes",
                "reason_codes", "evidence",
            }
            kind = self.replacement.get("kind")
            common_valid = (
                set(self.replacement) == expected_replacement and
                self.replacement.get("scope") == "all_sixteen_cells" and
                self.replacement.get("excluded_from_arm_outcomes") is True and
                isinstance(self.replacement.get("predecessor_campaign_run_id"), str) and
                bool(self.replacement.get("predecessor_campaign_run_id")) and
                bool(_digest(self.replacement.get("predecessor_protocol_inputs_sha256"))))
            infrastructure = (
                kind == "whole_campaign_infrastructure_replacement" and
                self.replacement.get("treatment_or_provider_started") is False and
                set(self.replacement.get("reason_codes", ())) == {
                    "nested_bwrap_uid_map_denied",
                    "generated_workspace_source_identity_drift",
                })
            protocol_design = (
                kind == "whole_campaign_protocol_design_replacement" and
                self.replacement.get("treatment_or_provider_started") is True and
                set(self.replacement.get("reason_codes", ())) == {
                    "undisclosed_capsule_descriptor_abi",
                    "persistent_worker_audit_contract_contradiction",
                })
            if not common_valid or not (infrastructure or protocol_design):
                raise ValueError("replacement does not declare a supported whole-campaign invalidity")
            replacement_evidence = _mapping(
                self.replacement.get("evidence"), "replacement.evidence")
            expected_evidence = ({
                "frozen_protocol", "protocol_claim", "launch", "ordinal1_preflight",
                "ordinal0_run_result", "ordinal0_stderr", "ordinal0_terminal_outcome",
                "ordinal0_raw_events", "ordinal0_token_ledger",
            } if infrastructure else {
                "frozen_protocol", "protocol_claim", "design_audit",
                "claim_revocation", "campaign_exclusion",
                "arm1_terminal_outcome", "arm1_grader_result",
                "arm2_terminal_outcome", "arm2_grader_result",
                "arm4_run_record", "arm4_raw_events", "arm4_controller_cancellation",
            })
            if set(replacement_evidence) != expected_evidence:
                raise ValueError("replacement evidence set is incomplete")
            for name, row in replacement_evidence.items():
                row = _mapping(row, f"replacement.evidence.{name}")
                if (set(row) != {"path", "sha256"} or not _resolved(row.get("path")) or
                        not _digest(row.get("sha256"))):
                    raise ValueError(f"replacement evidence identity is invalid: {name}")
        if len(self.arms) != 4 or [arm.order for arm in self.arms] != [1, 2, 3, 4]:
            raise ValueError("the experiment requires exactly four arms ordered 1,2,3,4")
        if len({arm.id for arm in self.arms}) != 4:
            raise ValueError("arm ids must be unique")
        for previous, current in zip(self.arms, self.arms[1:]):
            if not previous.capabilities < current.capabilities:
                raise ValueError(f"arms must be strictly nested: {previous.id} is not a subset of {current.id}")
        if not self.paper_holdouts or len(set(self.paper_holdouts)) != len(self.paper_holdouts):
            raise ValueError("paper_holdouts must be non-empty and unique")
        agent = self.agent
        if (agent.get("driver"), agent.get("orchestrator"), agent.get("billing")) != (
                "codex", "chia", "subscription_notional"):
            raise ValueError("agent must use Codex, Chia, and subscription_notional billing")
        if (int(agent.get("repeats", 0)) != 4 or
                int(agent.get("active_wall_seconds_per_arm", 0)) < 1):
            raise ValueError("agent must declare exactly four Williams blocks and a positive wall budget")
        if agent.get("schedule") != "continuous":
            raise ValueError("CPU-host arms must use the continuous schedule")
        if agent.get("launch_seed_role") != "campaign_metadata_only_not_provider_sampling":
            raise ValueError("agent.launch_seed must be labeled as campaign metadata, not a provider seed")
        launch_plan = agent.get("launch_plan")
        repeats = int(agent.get("repeats", 0))
        if (isinstance(agent.get("launch_seed"), bool) or
                not isinstance(agent.get("launch_seed"), int)):
            raise ValueError("agent.launch_seed must freeze an integer campaign seed")
        if not isinstance(launch_plan, list) or len(launch_plan) != len(self.arms) * repeats:
            raise ValueError("agent.launch_plan must freeze every arm/repeat exactly once")
        expected_cells = {(arm.id, repeat) for repeat in range(repeats) for arm in self.arms}
        actual_cells: set[tuple[str, int]] = set()
        for ordinal, row in enumerate(launch_plan):
            if (not isinstance(row, dict) or row.get("ordinal") != ordinal or
                    isinstance(row.get("repeat"), bool) or not isinstance(row.get("repeat"), int) or
                    isinstance(row.get("seed"), bool) or not isinstance(row.get("seed"), int)):
                raise ValueError("agent.launch_plan rows require consecutive ordinals and integer repeat/seed")
            actual_cells.add((str(row.get("arm", "")), int(row["repeat"])))
        if actual_cells != expected_cells or len(actual_cells) != len(launch_plan):
            raise ValueError("agent.launch_plan must cover each frozen arm/repeat exactly once")
        for repeat in range(repeats):
            seeds = {int(row["seed"]) for row in launch_plan if int(row["repeat"]) == repeat}
            if len(seeds) != 1:
                raise ValueError("all arms in one repeat must share the same paired seed")
        analysis = self.analysis_plan_config()
        self._validate_analysis_plan(analysis)
        sequences = analysis["design"]["sequences"]
        # The analysis plan is a protocol input, not an editable assertion. Reconstruct the exact
        # four Williams blocks so a reordered or cherry-picked experiment cannot retain the claim.
        expected_launch_plan: list[dict[str, Any]] = []
        for repeat, sequence in enumerate(sequences):
            for arm_id in sequence:
                expected_launch_plan.append({
                    "ordinal": len(expected_launch_plan),
                    "repeat": repeat,
                    "seed": repeat + 1,
                    "arm": arm_id,
                })
        if launch_plan != expected_launch_plan:
            raise ValueError(
                "agent.launch_plan must exactly equal the frozen 4x4 Williams schedule "
                "with canonical paired block identifier repeat+1")
        if self.telemetry.get("sink") != "aet":
            raise ValueError("telemetry sink must be aet")
        missing = sorted(name for name in _REQUIRED_TELEMETRY if self.telemetry.get(name) is not True)
        if missing:
            raise ValueError(f"full-fidelity telemetry fields must be true: {missing}")
        if self.grading.get("same_grader_all_arms") is not True:
            raise ValueError("all four arms must use the same grader")
        levels = [str(v.get("id")) for v in self.grading.get("levels", ()) if isinstance(v, dict)]
        if levels != ["L0", "L1", "L2", "L3"]:
            raise ValueError("grading levels must be exactly L0,L1,L2,L3")
        if self.grading.get("fallback_policy") != "forbid":
            raise ValueError("scored CPU-host runs must forbid fallback")
        if (self.environment.get("require_exact_local_before_each_cell") is not True or
                self.environment.get("require_exact_k1_before_each_cell") is not True):
            raise ValueError(
                "environment must require exact local and K1 identity before every cell")
        if (self.freeze.get("forbid_model_name_dispatch") is not True or
                self.freeze.get("forbid_post_freeze_tuning") is not True or
                int(self.freeze.get("required_empty_sweeps", 0)) != 1 or
                self.freeze.get("require_all_four_arms_complete") is not True):
            raise ValueError(
                "freeze must forbid name dispatch/post-freeze tuning, require one independently "
                "replayed empty sweep, "
                "and require all four arms")
        if self.freeze.get("failure_policy") != {
                "launch_all_scheduled_attempts": True,
                "retry_terminal_outcomes": False,
                "failed_primary_fallback": "forbidden",
                "one_shot_protocol_claim": True,
                "per_cell_atomic_consumption": True,
        }:
            raise ValueError(
                "freeze.failure_policy must launch every cell once and forbid retries/fallback")
        selection = _mapping(self.freeze.get("selection"), "freeze.selection")
        if selection != {
                "method": "predeclared_primary_no_outcome_selection",
                "primary_arm": "arm4_agentic_pass_authoring",
                "primary_repeat_index": 0,
                "eligible_evidence": "none_used_for_final_selection",
                "development_evidence_scope": "generic_train_validation_only",
                "heldout_outcomes_allowed": False,
        }:
            raise ValueError(
                "freeze.selection must predeclare arm4/repeat0 without using heldout outcomes")
        search_arms = [arm.id for arm in self.arms
                       if "deterministic_candidate_search" in arm.capabilities]
        if (int(self.search.get("required_paired_measurements", 0)) != 6 or
                self.search.get("measurement_authority") !=
                "spacemit_k1_elapsed_ns_div_completed_calls" or
                list(self.search.get("trusted_seal_required_for", ())) != search_arms):
            raise ValueError(
                "search arms require six balanced paired trusted K1 measurements and a driver seal")
        space_config = self.search_space_config()
        budget = _mapping(space_config.get("budget"), "search space budget")
        if any(float(budget.get(name, 0)) <= 0 for name in (
                "arm_wall_seconds", "reserved_agent_seconds",
                "expected_seconds_per_k1_program", "planning_upper_seconds_per_k1_program",
                "expected_spike_screen_seconds", "planning_upper_spike_screen_seconds",
                "expected_seconds_per_confirmation_package_build",
                "planning_upper_seconds_per_confirmation_package_build",
                "expected_seconds_per_confirmation_compiler_invocation",
                "planning_upper_seconds_per_confirmation_compiler_invocation",
                "expected_seconds_per_confirmation_spike_check",
                "planning_upper_seconds_per_confirmation_spike_check")):
            raise ValueError("search budget fields must be positive")
        environment = _mapping(space_config.get("board_environment"), "board_environment")
        required_environment = {
            "online", "governor", "frequency_khz", "frequency_core_count",
            "frequency_relative_tolerance", "maximum_temperature_millic",
            "maximum_pair_temperature_delta_millic", "maximum_load_1m",
            "maximum_pair_load_delta", "settle_attempts", "settle_interval_seconds",
            "maximum_invalid_pair_replacements_per_capsule",
        }
        if set(environment) != required_environment:
            raise ValueError("board_environment must contain the exact frozen K1 gate")
        if (environment["online"] != "0-7" or environment["governor"] != "performance" or
                int(environment["frequency_core_count"]) != int(
                    self.grading.get("expected_harts", 0)) or
                not 0 < float(environment["frequency_relative_tolerance"]) < float(
                    space_config.get("noise_margin", 0)) or
                int(environment["settle_attempts"]) * float(
                    environment["settle_interval_seconds"]) > 60 or
                int(environment["maximum_invalid_pair_replacements_per_capsule"]) < 0):
            raise ValueError("board_environment is inconsistent with the target/noise contract")

    def _repo_path(self, value: Any) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else repo_root() / path

    def preflight(self, *, check_environment: bool = True, probe_board: bool = False,
                  require_frozen: bool = False) -> HostPreflight:
        errors: list[str] = []
        blockers: list[str] = []
        warnings: list[str] = []
        evidence: dict[str, Any] = {}
        evidence["experiment_status"] = self.status
        evidence["live_campaign_authorization_checked"] = require_frozen

        if require_frozen and self.status not in {
                "protocol_frozen", "campaign_complete", "campaign_complete_unpromoted"}:
            blockers.append(
                "experiment status is draft; a live campaign requires a frozen protocol")

        paths = {
            # This module implements the freeze/preflight contract and must be bound
            # independently of the thin experiment controller that imports it.
            "experiment_contract": str(Path(__file__).resolve()),
            "host_agent_contract": str(
                repo_root() / "merlin/python/merlin/benchharness/host_agent.py"),
            "capsule_descriptor_contract": str(
                repo_root() / "merlin/python/merlin/benchharness/capsule_descriptor.py"),
            "artifact_lifecycle": str(
                repo_root() / "merlin/python/merlin/common/artifacts.py"),
            "repository_paths": str(
                repo_root() / "merlin/python/merlin/common/paths.py"),
            "schema_validator": str(
                repo_root() / "merlin/python/merlin/common/schemas.py"),
            "agent_sandbox_boundary": str(
                repo_root() / "merlin/python/merlin/targetgen/sandbox/bwrap.py"),
            "corpus_materializer": str(
                repo_root() / "merlin/python/merlin/mining/corpus.py"),
            "corpus_partition_policy": str(
                repo_root() / "merlin/python/merlin/mining/campaign.py"),
            "k1_measurement_adapter": str(
                repo_root() / "merlin/python/merlin/mining/k1.py"),
            "experiment_schema": str(
                repo_root() / "merlin/schemas/cpu_host_experiment.schema.yaml"),
            "frozen_environment_contract": str(
                repo_root() / "merlin/python/merlin/compare/frozen_environment.py"),
            "target_contract": self.target_contract,
            "dialect_plan": self.dialect_plan,
            "task": self.task,
            "development_corpus": self.development_corpus.get("manifest"),
            "agent_runner": self.agent.get("runner"),
            "agent_launcher": self.agent.get("launcher"),
            "protocol_controller": self.agent.get("protocol_controller"),
            "completion_controller": self.agent.get("completion_controller"),
            "grader": self.grading.get("grader"),
            "trusted_harness": self.grading.get("trusted_harness"),
            "k1_monitor": self.grading.get("k1_monitor"),
            "k1_probe_source": self.grading.get("k1_probe_source"),
            "submission_contract": self.grading.get("submission_contract"),
            "search_space": self.search.get("space"),
            "analysis_plan": self.analysis.get("plan"),
            "search_runner": self.search.get("runner"),
            "trusted_evaluator": self.search.get("trusted_evaluator"),
            "trusted_broker": self.search.get("trusted_broker"),
            "trusted_replay": self.search.get("trusted_replay"),
            "cost_calibrator": self.search.get("cost_calibrator"),
            "noise_calibrator": self.search.get("noise_calibrator"),
            "k1_calibration": (self.search.get("calibration") or {}).get("k1_artifact"),
            "spike_calibration": (self.search.get("calibration") or {}).get("spike_artifact"),
            "confirmation_overhead_calibration": (
                self.search.get("calibration") or {}).get("confirmation_overhead_artifact"),
            "noise_calibration": (
                self.search.get("calibration") or {}).get("noise_artifact"),
            # The runner imports this adapter directly, so its bytes are executable protocol input.
            "chia_bridge": self.telemetry.get("bridge"),
        }
        for name, row in sorted((self.replacement.get("evidence") or {}).items()):
            paths[f"replacement_predecessor_{name}"] = row.get("path")
        resolved_paths: dict[str, Path] = {}
        for name, value in paths.items():
            if not _resolved(value):
                blockers.append(f"{name} is unresolved")
                continue
            path = self._repo_path(value)
            resolved_paths[name] = path
            if not path.is_file():
                blockers.append(f"{name} does not exist: {path}")
        evidence["paths"] = {name: str(path) for name, path in resolved_paths.items()}

        replacement_evidence: dict[str, Any] = {}
        for name, row in sorted((self.replacement.get("evidence") or {}).items()):
            path_name = f"replacement_predecessor_{name}"
            path = resolved_paths.get(path_name)
            expected = str(row.get("sha256", ""))
            actual = hashlib.sha256(path.read_bytes()).hexdigest() \
                if path is not None and path.is_file() else None
            replacement_evidence[name] = {
                "path": str(path) if path is not None else None,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
            if actual is not None and actual != expected:
                errors.append(f"replacement predecessor evidence differs: {name}")
        if replacement_evidence:
            predecessor = {
                name: resolved_paths[f"replacement_predecessor_{name}"]
                for name in (self.replacement.get("evidence") or {})
            }
            kind = self.replacement.get("kind")
            if kind == "whole_campaign_infrastructure_replacement":
                objective_no_treatment_start = False
                try:
                    launch = _mapping(json.loads(
                        predecessor["launch"].read_text(encoding="utf-8")),
                        "replacement predecessor launch")
                    claim = _mapping(json.loads(
                        predecessor["protocol_claim"].read_text(encoding="utf-8")),
                        "replacement predecessor claim")
                    run_result = _mapping(json.loads(
                        predecessor["ordinal0_run_result"].read_text(encoding="utf-8")),
                        "replacement predecessor run result")
                    attempts = run_result.get("attempts")
                    usage = _mapping(
                        run_result.get("total_usage"), "replacement predecessor usage")
                    results = launch.get("results")
                    planned = launch.get("planned")
                    objective_no_treatment_start = (
                        launch.get("campaign_run_id") ==
                        self.replacement.get("predecessor_campaign_run_id") and
                        launch.get("protocol_inputs_sha256") ==
                        self.replacement.get("predecessor_protocol_inputs_sha256") and
                        claim.get("protocol_inputs_sha256") ==
                        self.replacement.get("predecessor_protocol_inputs_sha256") and
                        isinstance(planned, list) and len(planned) == 16 and
                        isinstance(results, list) and len(results) == 16 and
                        sum(row.get("attempted") is True for row in results
                            if isinstance(row, dict)) == 1 and
                        sum(row.get("cell_status") == "not_started" for row in results
                            if isinstance(row, dict)) == 15 and
                        run_result.get("thread_id") is None and
                        run_result.get("status") == "failed" and
                        isinstance(attempts, list) and len(attempts) == 1 and
                        isinstance(attempts[0], dict) and
                        attempts[0].get("thread_id") is None and
                        usage.get("reported") is False and
                        predecessor["ordinal0_raw_events"].stat().st_size == 0 and
                        predecessor["ordinal0_token_ledger"].stat().st_size == 0 and
                        "bwrap: setting up uid map: Permission denied" in
                        predecessor["ordinal0_stderr"].read_text(encoding="utf-8"))
                except Exception as exc:
                    errors.append(
                        f"replacement predecessor cannot be objectively audited: {exc}")
                if not objective_no_treatment_start:
                    errors.append(
                        "replacement predecessor evidence does not prove zero treatment/provider start")
                replacement_verification = {
                    "objective_no_treatment_start_verified": objective_no_treatment_start,
                }
            else:
                protocol_design_invalid = False
                try:
                    cells = predecessor["protocol_claim"].with_name(
                        f'{self.replacement.get("predecessor_protocol_inputs_sha256")}.cells')
                    documents = {
                        name: _mapping(
                            yaml.safe_load(path.read_text(encoding="utf-8"))
                            if name == "frozen_protocol" else
                            json.loads(path.read_text(encoding="utf-8")),
                            f"replacement predecessor {name}")
                        for name, path in predecessor.items()
                        if name != "arm4_raw_events"
                    }
                    protocol_design_invalid = _protocol_design_replacement_valid(
                        replacement=self.replacement, documents=documents,
                        artifact_sha256={
                            name: str(row["actual_sha256"])
                            for name, row in replacement_evidence.items()
                            if row.get("actual_sha256") is not None
                        }, cells=cells,
                        arm4_raw_size=predecessor["arm4_raw_events"].stat().st_size)
                except Exception as exc:
                    errors.append(
                        f"replacement predecessor cannot be objectively audited: {exc}")
                if not protocol_design_invalid:
                    errors.append(
                        "replacement predecessor evidence does not prove protocol-design invalidity")
                replacement_verification = {
                    "protocol_design_invalid_verified": protocol_design_invalid,
                    "treatment_or_provider_started": True,
                    "reason_codes": sorted(self.replacement.get("reason_codes", ())),
                }
            evidence["replacement_predecessor"] = {
                "campaign_run_id": self.replacement.get("predecessor_campaign_run_id"),
                "protocol_inputs_sha256": self.replacement.get(
                    "predecessor_protocol_inputs_sha256"),
                "excluded_from_arm_outcomes": self.replacement.get(
                    "excluded_from_arm_outcomes"),
                **replacement_verification,
                "evidence": replacement_evidence,
            }

        environment_manifest: Path | None = None
        environment_value = self.environment.get("manifest")
        if _resolved(environment_value):
            environment_manifest = self._repo_path(environment_value)
            resolved_paths["environment_manifest"] = environment_manifest
            evidence["paths"]["environment_manifest"] = str(environment_manifest)
            try:
                from merlin.compare.frozen_environment import validate_frozen_environment
                environment_check = validate_frozen_environment(
                    environment_manifest,
                    expected_sha256=str(self.environment.get("sha256", "")),
                    source_paths={name: path for name, path in resolved_paths.items()
                                  if name != "environment_manifest" and path.is_file()},
                    agent=self.agent, telemetry=self.telemetry,
                    probe_source=resolved_paths.get("k1_probe_source", Path("unresolved")),
                    check_local=check_environment, check_board=probe_board)
                evidence["frozen_environment"] = environment_check["evidence"]
                errors.extend(environment_check["errors"])
            except Exception as exc:
                errors.append(f"frozen environment could not be validated: {exc}")
        elif self.status in {"protocol_frozen", "campaign_complete",
                             "campaign_complete_unpromoted"} or require_frozen:
            blockers.append("frozen environment manifest is unresolved")
        else:
            warnings.append("environment manifest will be captured atomically at protocol freeze")

        for path_name, digest_name in (
                ("search_space", "space_sha256"),
                ("search_runner", "runner_sha256"),
                ("trusted_evaluator", "trusted_evaluator_sha256"),
                ("trusted_broker", "trusted_broker_sha256"),
                ("trusted_replay", "trusted_replay_sha256"),
                ("cost_calibrator", "cost_calibrator_sha256"),
                ("noise_calibrator", "noise_calibrator_sha256")):
            path = resolved_paths.get(path_name)
            expected = str(self.search.get(digest_name, ""))
            if not _digest(expected):
                blockers.append(f"search.{digest_name} is unresolved")
            elif path and path.is_file():
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                evidence[f"{path_name}_sha256"] = actual
                if actual != expected:
                    errors.append(f"{path_name} digest differs from experiment.yaml")

        analysis_path = resolved_paths.get("analysis_plan")
        expected_analysis = str(self.analysis.get("sha256", ""))
        if not _digest(expected_analysis):
            blockers.append("analysis.sha256 is unresolved")
        elif analysis_path and analysis_path.is_file():
            actual_analysis = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
            evidence["analysis_plan_sha256"] = actual_analysis
            if actual_analysis != expected_analysis:
                errors.append("analysis_plan digest differs from experiment.yaml")

        calibration = _mapping(self.search.get("calibration"), "search calibration")
        for path_name, digest_name in (
                ("k1_calibration", "k1_sha256"),
                ("spike_calibration", "spike_sha256"),
                ("confirmation_overhead_calibration", "confirmation_overhead_sha256"),
                ("noise_calibration", "noise_sha256")):
            path = resolved_paths.get(path_name)
            expected = str(calibration.get(digest_name, ""))
            if not _digest(expected):
                blockers.append(f"search.calibration.{digest_name} is unresolved")
            elif path and path.is_file():
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                evidence[f"{path_name}_sha256"] = actual
                if actual != expected:
                    errors.append(f"{path_name} digest differs from experiment.yaml")

        corpus_path = resolved_paths.get("development_corpus")
        if corpus_path and corpus_path.is_file():
            try:
                expected_digest = str(self.development_corpus.get("sha256", ""))
                actual_digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
                evidence["development_corpus_sha256"] = actual_digest
                if _digest(expected_digest) and actual_digest != expected_digest:
                    errors.append(
                        "development corpus digest differs from experiment.yaml: "
                        f"expected={expected_digest} actual={actual_digest}")
                corpus = _mapping(yaml.safe_load(corpus_path.read_text(encoding="utf-8")), "corpus")
                forbidden = set((corpus.get("paper_model_exclusion") or {}).get("forbidden_workloads") or ())
                if forbidden != set(self.paper_holdouts):
                    errors.append("development corpus exclusion set does not exactly equal paper_holdouts")
                sources = str(corpus.get("families", "")) + str(corpus.get("materialization", ""))
                leaked = sorted(name for name in self.paper_holdouts if name in sources)
                if leaked:
                    errors.append(f"paper model identities leak into selectable corpus content: {leaked}")
            except Exception as exc:  # malformed methodology is an error, not an unavailable resource
                errors.append(f"development corpus could not be validated: {exc}")

        if not _digest(self.development_corpus.get("sha256")):
            blockers.append("development corpus sha256 is unresolved")
        materialized = self.development_corpus.get("materialized_capsules")
        materialized_root: Path | None = None
        if not _resolved(materialized):
            blockers.append("portable RVV capsules are not materialized")
        else:
            materialized_root = self._repo_path(materialized)
            if not materialized_root.is_dir():
                blockers.append(f"materialized capsule root does not exist: {materialized_root}")
            else:
                lock_path = materialized_root / "corpus_lock.yaml"
                if not lock_path.is_file():
                    blockers.append(f"materialized corpus lock is absent: {lock_path}")
                else:
                    try:
                        lock = _mapping(yaml.safe_load(lock_path.read_text(encoding="utf-8")),
                                        "corpus lock")
                        evidence["materialized_corpus"] = {
                            "root": str(materialized_root.resolve()),
                            "corpus_sha256": lock.get("corpus_sha256"),
                            "capsule_count": lock.get("capsule_count"),
                            "split_counts": lock.get("split_counts"),
                        }
                        expected_definition = str(self.development_corpus.get("sha256", ""))
                        if lock.get("definition_sha256") != expected_definition:
                            errors.append("materialized corpus was generated from a different definition")
                        expected_corpus = str(self.development_corpus.get("materialized_sha256", ""))
                        if not _digest(expected_corpus):
                            blockers.append("materialized corpus sha256 is unresolved")
                        elif lock.get("corpus_sha256") != expected_corpus:
                            errors.append("materialized corpus digest differs from experiment.yaml")
                        from merlin.mining.corpus import derive_materialization
                        _, derived = derive_materialization(corpus_path)
                        if any(lock.get(name) != derived[name] for name in (
                                "definition_sha256", "corpus_sha256", "capsule_count",
                                "split_counts", "files")):
                            errors.append(
                                "materialized corpus is not the exact expansion of its frozen definition")
                        files = _mapping(lock.get("files"), "corpus lock files")
                        required = {"public/train.jsonl", "public/validation.jsonl",
                                    "sealed/heldout.jsonl"}
                        if set(files) != required:
                            errors.append("materialized corpus lock has an unexpected split file set")
                        for relpath in required:
                            path = materialized_root / relpath
                            if not path.is_file():
                                blockers.append(f"materialized capsule split is absent: {path}")
                            elif hashlib.sha256(path.read_bytes()).hexdigest() != files.get(relpath):
                                errors.append(f"materialized capsule split digest mismatch: {relpath}")
                        if (materialized_root / "public/heldout.jsonl").exists():
                            errors.append("heldout capsules leaked into the agent-visible public directory")
                    except Exception as exc:
                        errors.append(f"materialized corpus lock could not be validated: {exc}")
        grader_path = resolved_paths.get("grader")
        if materialized_root is not None and materialized_root.is_dir():
            train_path = materialized_root / "public" / "train.jsonl"
            if train_path.is_file():
                source_names = (
                    "cost_calibrator", "noise_calibrator", "grader", "search_runner", "trusted_harness",
                    "k1_monitor", "search_space", "trusted_evaluator", "trusted_broker",
                    "k1_adapter")
                resolved_paths["k1_adapter"] = Path(paths["k1_measurement_adapter"])
                source_sha256 = {
                    name: hashlib.sha256(resolved_paths[name].read_bytes()).hexdigest()
                    for name in source_names
                    if name in resolved_paths and resolved_paths[name].is_file()
                }
                train_sha256 = hashlib.sha256(train_path.read_bytes()).hexdigest()
                train_rows = [json.loads(line) for line in
                              train_path.read_text(encoding="utf-8").splitlines()
                              if line.strip()]
                configured_noise_value: dict[str, Any] | None = None
                configured_noise_sha256: str | None = None
                configured_noise_path = resolved_paths.get("noise_calibration")
                if configured_noise_path is not None and configured_noise_path.is_file():
                    try:
                        loaded_noise = json.loads(
                            configured_noise_path.read_text(encoding="utf-8"))
                        if isinstance(loaded_noise, dict):
                            configured_noise_value = loaded_noise
                            configured_noise_sha256 = hashlib.sha256(
                                configured_noise_path.read_bytes()).hexdigest()
                    except (OSError, json.JSONDecodeError):
                        pass
                for calibration_name in (
                        "k1_calibration", "spike_calibration",
                        "confirmation_overhead_calibration", "noise_calibration"):
                    calibration_path = resolved_paths.get(calibration_name)
                    if calibration_path is None or not calibration_path.is_file():
                        continue
                    try:
                        calibration_value = json.loads(
                            calibration_path.read_text(encoding="utf-8"))
                        blockers.extend(_validate_calibration_semantics(
                            label=calibration_name, value=calibration_value,
                            train_sha256=train_sha256, source_sha256=source_sha256,
                            space=self.search_space_config(), train_rows=train_rows,
                            noise_authority=configured_noise_value,
                            noise_authority_sha256=configured_noise_sha256))
                    except Exception as exc:
                        blockers.append(
                            f"{calibration_name} semantic validation failed closed: {exc}")
        if grader_path and grader_path.is_file():
            if materialized_root is not None and materialized_root.is_dir():
                corpus_check = subprocess.run(
                    [sys.executable, str(grader_path), "--validate-corpus",
                     "--train", str(materialized_root / "public/train.jsonl"),
                     "--validation", str(materialized_root / "public/validation.jsonl"),
                     "--heldout", str(materialized_root / "sealed/heldout.jsonl")],
                    capture_output=True, text=True, timeout=60)
                evidence["grader_corpus_check_returncode"] = corpus_check.returncode
                try:
                    corpus_result = json.loads(corpus_check.stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    corpus_result = {
                        "ready": False,
                        "error": (corpus_check.stderr.strip() or corpus_check.stdout.strip()
                                  or "grader emitted no corpus-check result")[:1000],
                    }
                evidence["grader_corpus_check"] = corpus_result
                if corpus_check.returncode or corpus_result.get("ready") is not True:
                    errors.append(
                        "materialized corpus cannot satisfy declared grader coverage: "
                        f"{corpus_result.get('error', 'unknown corpus-check failure')}")
            try:
                proc = subprocess.run(
                    [sys.executable, str(grader_path), "--self-check"],
                    capture_output=True, text=True, timeout=30)
                evidence["grader_self_check_returncode"] = proc.returncode
                if proc.returncode:
                    blockers.append("deterministic CPU-host grader self-check failed")
                else:
                    self_check = json.loads(proc.stdout.strip().splitlines()[-1]
                                            if proc.stdout.count("\n") == 0 else proc.stdout)
                    evidence["grader_self_check"] = self_check
                    required_levels = [str(value.get("id")) for value in self.grading["levels"]]
                    implemented = set(self_check.get("implemented_levels", ()))
                    missing_levels = [level for level in required_levels if level not in implemented]
                    if missing_levels:
                        blockers.append(
                            f"deterministic CPU-host grader does not implement levels {missing_levels}")
                    if self_check.get("ready") is not True:
                        blockers.append("deterministic CPU-host grader toolchain self-check is not ready")
                    trusted = self_check.get("trusted_search", {})
                    if (trusted.get("outside_sandbox_broker_api") is not True or
                            trusted.get("paired_measurements") != 6 or
                            trusted.get("heldout_argument") is not False):
                        blockers.append("deterministic CPU-host grader lacks the trusted search API")
            except Exception as exc:
                blockers.append(f"deterministic CPU-host grader self-check could not be parsed: {exc}")

        aet_source = Path(str(self.telemetry.get("aet_source", "")))
        chia_source = Path(str(self.telemetry.get("chia_source", "")))
        external_provenance = {
            "aet": _git_provenance(aet_source) if aet_source.is_dir() else None,
            "chia": _git_provenance(chia_source) if chia_source.is_dir() else None,
        }
        evidence["aet_provenance"] = external_provenance["aet"]
        evidence["chia_provenance"] = external_provenance["chia"]
        for name, provenance in external_provenance.items():
            if (not isinstance(provenance, dict) or not _git_oid(provenance.get("git_sha")) or
                    not _digest(provenance.get("dirty_content_sha256"))):
                blockers.append(
                    f"{name} source provenance cannot bind commit plus local content")

        arm_workspace_inputs: dict[str, Any] = {}
        workspace_paths = {
            name: resolved_paths.get(name) for name in (
                "task", "target_contract", "dialect_plan", "submission_contract",
                "search_space", "search_runner", "trusted_evaluator")
        }
        if materialized_root is not None and all(
                path is not None and path.is_file() for path in workspace_paths.values()):
            try:
                from merlin.benchharness.host_agent import stage_host_workspace
                with tempfile.TemporaryDirectory(
                        prefix="merlin-host-protocol-workspaces-") as temporary:
                    root = Path(temporary)
                    for arm in self.arms:
                        staged = stage_host_workspace(
                            root / arm.id,
                            task_path=workspace_paths["task"],
                            target_contract_path=workspace_paths["target_contract"],
                            dialect_plan_path=workspace_paths["dialect_plan"],
                            submission_contract_path=workspace_paths["submission_contract"],
                            public_corpus_dir=materialized_root / "public",
                            search_space_path=workspace_paths["search_space"],
                            search_runner_path=workspace_paths["search_runner"],
                            trusted_evaluator_path=workspace_paths["trusted_evaluator"],
                            arm_id=arm.id, capabilities=arm.capabilities,
                            treatment=arm.treatment)
                        arm_workspace_inputs[arm.id] = {
                            "input_lock_sha256": staged.input_lock_sha256,
                            "file_count": len(staged.input_lock),
                            "input_lock": staged.input_lock,
                        }
            except Exception as exc:
                errors.append(f"arm workspace inputs cannot be prematerialized: {exc}")
        else:
            blockers.append("arm workspace inputs cannot be prematerialized from unresolved paths")
        evidence["arm_workspace_inputs"] = arm_workspace_inputs

        freeze_methodology = {
            name: self.freeze.get(name) for name in (
                "forbid_model_name_dispatch", "forbid_post_freeze_tuning",
                "required_empty_sweeps", "require_all_four_arms_complete", "failure_policy",
                "selection")
        }
        protocol_payload = {
            "version": self.version,
            "label": self.label,
            "paper_holdouts": self.paper_holdouts,
            "arms": [{"id": arm.id, "order": arm.order,
                      "capabilities": sorted(arm.capabilities), "treatment": arm.treatment}
                     for arm in self.arms],
            "agent": self.agent,
            "telemetry": self.telemetry,
            "grading": self.grading,
            "search": self.search,
            "analysis": self.analysis,
            "replacement": self.replacement,
            "environment": self.environment,
            "development_corpus": self.development_corpus,
            "arm_workspace_inputs": arm_workspace_inputs,
            "freeze_methodology": freeze_methodology,
            "external_source_provenance": external_provenance,
            "path_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for name, path in sorted(resolved_paths.items()) if path.is_file()},
        }
        actual_protocol_digest = hashlib.sha256(json.dumps(
            protocol_payload, sort_keys=True, separators=(",", ":"), default=list
        ).encode("utf-8")).hexdigest()
        evidence["protocol_inputs_sha256"] = actual_protocol_digest
        if self.status in {"protocol_frozen", "campaign_complete", "campaign_complete_unpromoted"}:
            expected_protocol = str(self.freeze.get("protocol_inputs_sha256", ""))
            if not _digest(expected_protocol):
                blockers.append("protocol_frozen experiment has no protocol_inputs_sha256")
            elif expected_protocol != actual_protocol_digest:
                errors.append("frozen protocol inputs differ from freeze.protocol_inputs_sha256")

        if self.status in {"campaign_complete", "campaign_complete_unpromoted"}:
            promoted = self.status == "campaign_complete"
            for name in ("selected_policy_sha256", "runtime_sha256", "compiler_sha256"):
                if promoted and not _digest(self.freeze.get(name)):
                    errors.append(f"completed campaign has no valid {name}")
                if not promoted and self.freeze.get(name) != "unresolved":
                    errors.append(f"unpromoted campaign must leave {name} unresolved")
            campaign_record = self.freeze.get("campaign_record")
            if not isinstance(campaign_record, dict):
                errors.append("completed campaign has no embedded campaign_record")
            else:
                actual_campaign_sha = hashlib.sha256(json.dumps(
                    campaign_record, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")).hexdigest()
                if actual_campaign_sha != self.freeze.get("campaign_record_sha256"):
                    errors.append("completed campaign record digest does not match its contents")
                selection = campaign_record.get("selection", {})
                selection_run_id = selection.get("selected_run_id")
                expected_selected = self.freeze.get("selected_run_id") if promoted else None
                promotion = campaign_record.get("promotion", {})
                boundary_receipts = campaign_record.get("block_boundary_receipts")
                if (campaign_record.get("completed_run_count") != 16 or
                        campaign_record.get("analysis_plan_sha256") != self.analysis.get("sha256") or
                        not isinstance(boundary_receipts, list) or len(boundary_receipts) != 4 or
                        selection.get("heldout_outcome_used") is not False or
                        selection.get("selection_outcome_fields_used") != [] or
                        selection_run_id != expected_selected or
                        promotion.get("status") != ("promoted" if promoted else "ineligible")):
                    errors.append(
                        "completed campaign is not an outcome-independent complete 4x4 campaign")
            package_value = self.freeze.get("selected_compiler_package")
            package = Path(str(package_value)).resolve() if _resolved(package_value) else None
            if not promoted and self.freeze.get("selected_compiler_package") != "unresolved":
                errors.append("unpromoted campaign must leave selected compiler package unresolved")
            elif promoted and (package is None or not package.is_dir()):
                errors.append("completed campaign selected compiler package is absent")
            elif promoted:
                try:
                    from merlin.benchharness.host_agent import (
                        _submission_package_digest, _submission_source_digest)
                    manifest = _mapping(yaml.safe_load(
                        (package / "manifest.yaml").read_text(encoding="utf-8")),
                        "selected compiler manifest")
                    policy = (package / str(manifest.get("policy", ""))).resolve()
                    if (not policy.is_relative_to(package) or not policy.is_file() or
                            hashlib.sha256(policy.read_bytes()).hexdigest() !=
                            self.freeze.get("selected_policy_sha256") or
                            _submission_source_digest(package) != self.freeze.get("compiler_sha256") or
                            _submission_package_digest(package) != self.freeze.get("runtime_sha256")):
                        errors.append("completed campaign selected compiler package differs from seal")
                except Exception as exc:
                    errors.append(f"completed campaign compiler seal cannot be verified: {exc}")

        space_config = self.search_space_config()
        budget = space_config["budget"]
        group_counts: dict[str, int] = {}
        for action in space_config.get("actions", ()):
            group = str(action.get("group"))
            group_counts[group] = group_counts.get(group, 0) + 1
        groups = len(group_counts)
        distinct_incumbents = groups + 1
        remaining_actions = sum(group_counts.values())
        maximum_screen_evaluations = 0
        for group_size in sorted(group_counts.values()):
            maximum_screen_evaluations += remaining_actions
            remaining_actions -= group_size
        confirmation_capsules = len(space_config.get("confirmation_families", ())) * int(
            space_config.get("confirmation_samples_per_family", 0))
        confirmation_requests = (
            distinct_incumbents * int(space_config.get("confirmation_width", 0)) * 2)
        package_builds = confirmation_requests * 2
        runtime_confirmation_capsules = (
            int(space_config.get("confirmation_samples_per_family", 0))
            if "runtime_parallel" in space_config.get("confirmation_families", ()) else 0)
        compiler_invocations = confirmation_requests * (
            confirmation_capsules + runtime_confirmation_capsules) * 2
        spike_checks = confirmation_requests * confirmation_capsules * 2
        expected_k1_programs = (
            confirmation_requests * confirmation_capsules * 2 *
            int(space_config.get("measurement_repeats", 0)))
        k1_programs = (
            confirmation_requests * confirmation_capsules * 2 *
            (int(space_config.get("measurement_repeats", 0)) + int(
                _mapping(space_config.get("board_environment"), "board_environment").get(
                    "maximum_invalid_pair_replacements_per_capsule", 0))))
        expected_confirmation_overhead = (
            package_builds * float(
                budget["expected_seconds_per_confirmation_package_build"]) +
            compiler_invocations * float(
                budget["expected_seconds_per_confirmation_compiler_invocation"]) +
            spike_checks * float(budget["expected_seconds_per_confirmation_spike_check"]))
        planning_upper_confirmation_overhead = (
            package_builds * float(
                budget["planning_upper_seconds_per_confirmation_package_build"]) +
            compiler_invocations * float(
                budget["planning_upper_seconds_per_confirmation_compiler_invocation"]) +
            spike_checks * float(
                budget["planning_upper_seconds_per_confirmation_spike_check"]))
        expected_seconds = (
            expected_k1_programs * float(budget["expected_seconds_per_k1_program"]) +
            float(budget["expected_spike_screen_seconds"]) +
            expected_confirmation_overhead)
        planning_upper_seconds = (
            k1_programs * float(budget["planning_upper_seconds_per_k1_program"]) +
            float(budget["planning_upper_spike_screen_seconds"]) +
            planning_upper_confirmation_overhead)
        available_seconds = int(self.agent["active_wall_seconds_per_arm"]) - int(
            budget["reserved_agent_seconds"])
        evidence["trusted_search_budget"] = {
            "action_groups": groups,
            "maximum_distinct_incumbents": distinct_incumbents,
            "maximum_screen_candidate_evaluations": maximum_screen_evaluations,
            "confirmation_capsules_per_split": confirmation_capsules,
            "maximum_confirmation_requests": confirmation_requests,
            "confirmation_package_builds": package_builds,
            "confirmation_compiler_invocations": compiler_invocations,
            "confirmation_spike_checks": spike_checks,
            "expected_k1_program_invocations": expected_k1_programs,
            "k1_program_invocations": k1_programs,
            "expected_confirmation_overhead_seconds": expected_confirmation_overhead,
            "planning_upper_confirmation_overhead_seconds": (
                planning_upper_confirmation_overhead),
            "expected_search_seconds": expected_seconds,
            "planning_upper_search_seconds": planning_upper_seconds,
            "available_search_seconds": available_seconds,
            "fits_declared_arm": planning_upper_seconds <= available_seconds,
        }
        declared = {
            "maximum_distinct_incumbents": distinct_incumbents,
            "maximum_screen_candidate_evaluations": maximum_screen_evaluations,
            "maximum_confirmation_requests": confirmation_requests,
            "confirmation_package_builds": package_builds,
            "confirmation_compiler_invocations": compiler_invocations,
            "confirmation_spike_checks": spike_checks,
            "k1_program_invocations": k1_programs,
            "expected_confirmation_overhead_seconds": expected_confirmation_overhead,
            "planning_upper_confirmation_overhead_seconds": (
                planning_upper_confirmation_overhead),
            "expected_search_seconds": expected_seconds,
            "planning_upper_search_seconds": planning_upper_seconds,
        }
        for name, actual in declared.items():
            if float(budget.get(name, -1)) != float(actual):
                errors.append(f"search budget {name} does not match the frozen search space")
        if int(budget.get("arm_wall_seconds", -1)) != int(
                self.agent["active_wall_seconds_per_arm"]):
            errors.append("search budget arm_wall_seconds differs from the agent wall budget")
        if planning_upper_seconds > available_seconds:
            blockers.append(
                "trusted search planning upper bound exceeds the active arm budget after reserve")

        if check_environment:
            evidence["codex_binary"] = shutil.which("codex")
            if not evidence["codex_binary"]:
                blockers.append("Codex CLI is unavailable")
            else:
                version = subprocess.run([evidence["codex_binary"], "--version"], capture_output=True,
                                         text=True, timeout=30)
                auth = subprocess.run([evidence["codex_binary"], "login", "status"],
                                      capture_output=True, text=True, timeout=30)
                evidence["codex_version"] = version.stdout.strip() or version.stderr.strip()
                evidence["codex_auth_returncode"] = auth.returncode
                evidence["codex_auth_status"] = (auth.stdout.strip() or auth.stderr.strip())[:200]
                if auth.returncode:
                    blockers.append("Codex subscription authentication is unavailable")
                from merlin.benchharness.host_agent import probe_codex_bwrap_runtime
                sandbox_probe = probe_codex_bwrap_runtime(evidence["codex_binary"])
                evidence["codex_bwrap_runtime_probe"] = sandbox_probe
                if sandbox_probe.get("ready") is not True:
                    blockers.append("Codex production bwrap boundary is not executable")
            evidence["aet_importable"] = importlib.util.find_spec("aet") is not None
            if not evidence["aet_importable"]:
                blockers.append("aet is not importable in the main environment")
            else:
                from aet.trajectory.importers.codex import import_codex_run
                evidence["aet_timestamp_sidecar_support"] = (
                    "timestamped" in inspect.signature(import_codex_run).parameters or
                    any(p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in inspect.signature(import_codex_run).parameters.values()))
                if not evidence["aet_timestamp_sidecar_support"]:
                    blockers.append("AET Codex importer lacks timestamp-sidecar support")
            if not aet_source.is_dir():
                blockers.append(f"aet source is unavailable: {aet_source}")
            if not chia_source.is_dir():
                blockers.append(f"Chia source is unavailable: {chia_source}")
            chia_python = self._repo_path(self.telemetry.get("chia_python", ""))
            evidence["chia_python"] = str(chia_python)
            if not chia_python.is_file():
                blockers.append(f"Chia interpreter is unavailable: {chia_python}")
            else:
                check = (
                    "import inspect,json,ray; from chia.models.codex import CodexLLM; "
                    "print(json.dumps({'arrival_timestamps': "
                    "'capture_arrival_timestamps' in inspect.signature(CodexLLM).parameters}))")
                proc = subprocess.run(
                    [str(chia_python), "-c", check], capture_output=True, text=True, timeout=30)
                evidence["chia_import_returncode"] = proc.returncode
                if proc.returncode:
                    blockers.append("the declared Chia interpreter cannot import chia and ray")
                else:
                    chia_features = json.loads(proc.stdout.strip().splitlines()[-1])
                    evidence["chia_features"] = chia_features
                    if chia_features.get("arrival_timestamps") is not True:
                        blockers.append("Chia Codex backend lacks arrival-timestamp capture")
            try:
                from merlin.mining import k1
                evidence["k1_available"] = k1.available()
                if not evidence["k1_available"]:
                    blockers.append("K1 board or cross-toolchain is unavailable")
                elif probe_board:
                    probe = k1.run_arch_probe(resolved_paths["k1_probe_source"])
                    evidence["k1_probe"] = probe
                    values = probe.get("values", {})
                    expected_harts = int(self.grading.get("expected_harts", 0))
                    expected_vlenb = int(self.grading.get("expected_vlenb", 0))
                    if int(values.get("online_harts", -1)) != expected_harts:
                        blockers.append("K1 probe hart count does not match the experiment contract")
                    if int(values.get("vlenb", -1)) != expected_vlenb:
                        blockers.append("K1 probe VLEN does not match the experiment contract")
                    grader_spec = importlib.util.spec_from_file_location(
                        "merlin_host_preflight_grader", resolved_paths["grader"])
                    if grader_spec is None or grader_spec.loader is None:
                        raise RuntimeError("cannot load the frozen grader for its board-state gate")
                    grader_module = importlib.util.module_from_spec(grader_spec)
                    grader_spec.loader.exec_module(grader_module)
                    state_probe = grader_module._probe_k1_state(
                        grader_module._k1_connection())
                    board_environment = _mapping(
                        self.search_space_config().get("board_environment"),
                        "board_environment")
                    state_ready = grader_module._k1_state_ready(
                        state_probe, board_environment)
                    evidence["k1_board_state_probe"] = state_probe
                    evidence["k1_board_state_ready"] = state_ready
                    if not state_ready:
                        blockers.append("K1 board does not satisfy the frozen pre-pair state gate")
            except Exception as exc:
                blockers.append(f"K1 environment probe failed: {exc}")

        return HostPreflight(tuple(errors), tuple(blockers), tuple(warnings), evidence)

    def search_space_config(self) -> dict[str, Any]:
        path = self._repo_path(self.search.get("space", ""))
        if not path.is_file():
            return {}
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}

    def analysis_plan_config(self) -> dict[str, Any]:
        path = self._repo_path(self.analysis.get("plan", ""))
        if not path.is_file():
            return {}
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}

    def _validate_analysis_plan(self, plan: dict[str, Any]) -> None:
        """Validate the frozen small-n estimand and the complete 4x4 Williams design."""
        if plan.get("version") != 1 or plan.get("status") != "frozen_definition":
            raise ValueError("analysis plan must be version 1 with status frozen_definition")
        design = _mapping(plan.get("design"), "analysis design")
        arm_ids = [arm.id for arm in self.arms]
        sequences = design.get("sequences")
        if (design.get("kind") != "balanced_williams_4x4" or design.get("repeats") != 4 or
                design.get("repeat_identifier") != "paired_block_id" or
                design.get("provider_sampling_seeded") is not False or
                design.get("provider_seed_claim_forbidden") is not True or
                design.get("carryover_scope") != "within_block_transitions_only" or
                design.get("cross_block_transition_policy") !=
                "excluded_after_mandatory_washout_and_requalification" or
                not isinstance(sequences, list) or len(sequences) != 4 or
                any(not isinstance(sequence, list) or len(sequence) != 4
                    for sequence in sequences)):
            raise ValueError("analysis design must be the unseeded-provider balanced 4x4 Williams plan")
        if _mapping(design.get("block_boundary"), "analysis block boundary") != {
                "required_before_first_cell": True,
                "authority": "frozen_k1_board_environment_gate",
                "mandatory_washout_seconds_from": "board_environment.settle_interval_seconds",
                "stabilization_attempts_from": "board_environment.settle_attempts",
                "retained_receipt_required": True,
        }:
            raise ValueError("analysis design must requalify K1 at every block boundary")
        expected_sequences = [
            [arm_ids[0], arm_ids[1], arm_ids[3], arm_ids[2]],
            [arm_ids[1], arm_ids[2], arm_ids[0], arm_ids[3]],
            [arm_ids[2], arm_ids[3], arm_ids[1], arm_ids[0]],
            [arm_ids[3], arm_ids[0], arm_ids[2], arm_ids[1]],
        ]
        if sequences != expected_sequences:
            raise ValueError("analysis sequences differ from the predeclared Williams square")
        for position in range(4):
            if {sequence[position] for sequence in sequences} != set(arm_ids):
                raise ValueError("analysis Williams square does not balance every ordinal position")
        carryover = [(left, right) for sequence in sequences
                     for left, right in zip(sequence, sequence[1:])]
        expected_carryover = {(left, right) for left in arm_ids for right in arm_ids if left != right}
        if len(carryover) != 12 or len(set(carryover)) != 12 or set(carryover) != expected_carryover:
            raise ValueError("analysis Williams square must cover all 12 directed carryover pairs once")
        estimand = _mapping(plan.get("estimand"), "analysis estimand")
        if (estimand.get("population") != "all_sixteen_predeclared_arm_block_cells" or
                estimand.get("policy") != "intention_to_treat" or
                estimand.get("primary_unit") != "scheduled_arm_block_cell"):
            raise ValueError("analysis estimand must include all 16 cells by intention to treat")
        endpoints = _mapping(plan.get("endpoints"), "analysis endpoints")
        required_endpoints = {
            "terminal_outcome_counts", "L0_L1_L2_L3_pass_counts", "total_tokens",
            "reasoning_tokens", "cache_read_tokens", "cache_write_tokens",
            "agent_active_seconds", "cell_wall_seconds", "grader_seconds",
        }
        if (not required_endpoints.issubset(set(endpoints.get("primary", ()))) or
                endpoints.get("inferential_significance_claims") != "forbidden" or
                endpoints.get("interpretation") != "descriptive_small_n"):
            raise ValueError("analysis endpoints must retain full-fidelity descriptive outcomes")
        missingness = _mapping(plan.get("missingness"), "analysis missingness")
        if missingness != {
                "treatment_agent_fail": "observed_failure",
                "treatment_build_fail": "observed_failure",
                "treatment_search_fail": "observed_failure",
                "graded_fail": "observed_failure",
                "graded_pass": "observed_success",
                "infrastructure_invalid": "missing_replace_only_by_predeclared_protocol",
                "post_outcome_retry": "forbidden",
        }:
            raise ValueError("analysis missingness must map every treatment outcome without censoring")
        selection = _mapping(plan.get("selection"), "analysis selection")
        if selection != {
                "method": "predeclared_primary_no_outcome_selection",
                "primary_arm": "arm4_agentic_pass_authoring",
                "primary_repeat_index": 0,
                "heldout_outcome_selection": "forbidden",
                "fallback_after_primary_failure": "forbidden",
        }:
            raise ValueError("analysis selection must predeclare Arm4 block 0 without fallback")
        reporting = _mapping(plan.get("reporting"), "analysis reporting")
        if (reporting.get("report_order_and_block") is not True or
                reporting.get("report_block_boundary_receipts") is not True):
            raise ValueError("analysis reporting must retain order, block, and boundary receipts")
