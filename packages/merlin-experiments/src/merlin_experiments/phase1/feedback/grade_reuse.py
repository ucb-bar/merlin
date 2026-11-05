"""Reuse one completed host grade inside its original authoring invocation.

This is not a certificate store or a public-channel reader. The host retains an
in-memory receipt over its own canonical artifacts and rechecks its original
admission, source and cohort before omitting a redundant grade.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from merlin.common.tree_hash import hash_tree


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("grade artifact is absent or a symlink")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _artifacts(root: Path) -> tuple[tuple[Path, str], ...]:
    files, size = [], 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("retained grade has linked artifacts")
        if path.is_file():
            size += path.stat().st_size
            if size > 256 << 20 or len(files) >= 8192:
                raise ValueError("retained grade exceeds bounded artifact inventory")
            files.append((path, _sha(path)))
    return tuple(files)


def _source(path: Path) -> str:
    from merlin_experiments.phase2.retained_revisions import source_inventory

    return source_inventory(path.resolve(), max_bytes=32 << 20, max_files=8192, exclude_generated=True)["sha256"]


def _retained_program(artifacts) -> bool:
    from merlin.targetgen.elf_lanes import PROGRAM_ARTIFACT_SUFFIX

    for path, _ in artifacts:
        if path.name.endswith(PROGRAM_ARTIFACT_SUFFIX):
            return True
        with path.open("rb") as stream:
            header = stream.read(20)
        if len(header) >= 18 and header[:4] == b"\x7fELF" and header[5] in (1, 2):
            endian = "little" if header[5] == 1 else "big"
            if int.from_bytes(header[16:18], endian) in (2, 3):
                return True
    return False


def _binding(inputs, run_dir: Path, public: Path) -> dict:
    from merlin.common.provenance import source_digest
    from merlin.targetgen import tier_cache

    from .compiler_profile_runtime import compiler_profile_binding

    if inputs.compiler_profile_service is None or inputs.verify_grade_inputs is None:
        raise ValueError("no original host admission")
    inputs.verify_grade_inputs()
    owner = compiler_profile_binding(inputs.compiler_profile_service, run_dir=run_dir)
    files = tier_cache.grading_path(inputs.context.target)
    if not owner or not files:
        raise ValueError("original owner or grading instrument is unavailable")
    return {
        "owner": owner,
        "descriptor": _sha(Path(inputs.context.descriptor)),
        "public": hash_tree(public),
        "contract": hash_tree(Path(inputs.contract)) if inputs.contract is not None else None,
        "instrument": source_digest([str(p) for p in files]),
        # Simulator/protocol selection remains the original host's, including
        # environment controls which do not appear in a source-only digest.
        "selection": {key: value for key, value in os.environ.items() if key.startswith("MERLIN_")},
        "target": inputs.context.target,
        "forbidden": inputs.additional_forbidden,
        "policy_roots": [hash_tree(Path(p)) for p in getattr(inputs, "policy_roots", ())],
        "arm": getattr(inputs, "arm", None),
        "readback_policy": getattr(inputs.context, "readback_policy", None),
        "public_build_selection": json.loads(json.dumps(getattr(inputs, "public_build_selection", None))),
        "public_build_budget_s": getattr(inputs, "public_build_budget_s", 0),
        "rtl_facts": _sha(Path(inputs.rtl_facts)) if getattr(inputs, "rtl_facts", None) is not None else None,
    }


@dataclass(frozen=True)
class DriverGradeReceipt:
    candidate: Path
    runs: Path
    public: Path
    source_sha256: str
    binding: dict
    artifacts: tuple[tuple[Path, str], ...]
    rows: dict
    verdict_path: Path | None = None
    verdict_json: str | None = None
    oot_commit: str | None = None

    def current(self, *, submission: Path, inputs, run_dir: Path) -> bool:
        try:
            return (
                _source(submission) == self.source_sha256
                and _source(self.candidate) == self.source_sha256
                and _binding(inputs, run_dir, self.public) == self.binding
                and all(_sha(path) == digest for path, digest in self.artifacts)
            )
        except Exception:  # noqa: BLE001 -- unavailable reuse always falls back to grading
            return False


def retain_driver_grade(*, candidate, runs, public, verdict, inputs, run_dir,
                        verdict_path=None, oot_commit=None, evidence_paths=()):
    """Retain a complete ordinary grade, never a screen or a redacted claim."""
    from merlin.targetgen import capsule_grade as CG
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import expected_refusal as ER

    from .compiler_profile_child import _original_executor

    try:
        if verdict.get("all_pass") is not True or verdict.get("gradeable") is not True:
            return None
        candidate, runs, public, run_dir = map(Path, (candidate, runs, public, run_dir))
        # These paths are selected by LG.grade, not supplied by the author.
        if not candidate.resolve().is_relative_to(run_dir.resolve() / "_qa_work"):
            return None
        if not runs.resolve().is_relative_to(run_dir.resolve() / "_qa_work"):
            return None
        binding = _binding(inputs, run_dir, public)
        source = _source(candidate)
        if not source:
            return None
        executor = _original_executor()
        if (
            executor is None
            or executor.owner is not inputs.compiler_profile_service
            or executor.source_sha256 != source
        ):
            return None
        caps = CR.discover_capsules(public, labels={"public", "dev"}, contract=inputs.contract)
        cohort = {cap["name"]: cap for cap in caps}
        if not cohort or len(cohort) != len(caps):
            return None
        paths = sorted((runs / "runs").glob("*/*/capsule_result.json"))
        artifacts, rows = [], {}
        for path in paths:
            raw = json.loads(path.read_text())
            name = raw.get("capsule")
            if name not in cohort or name in rows:
                return None
            cap = cohort[name]
            if ER.expects_refusal(cap):
                if not ER.validated_result(cap, raw):
                    return None
                row = {
                    "status": "pass",
                    "tiers": {},
                    "validated_refusal": True,
                    "qualification": "expected_refusal",
                    "native_executed": False,
                    "expected_refusal": raw["expected_refusal"],
                }
            else:
                child = raw.get("original_compiler_child")
                if child is not None and (
                    not isinstance(child, dict)
                    or child.get("schema") != "merlin.original_model_child.v1"
                    or child.get("candidate_sha256") != source
                    or any(child.get(key) != binding["owner"][key] for key in ("profile_sha256", "snapshot_sha256"))
                ):
                    return None
                view = CG._candidate_score_view(raw)
                if raw.get("kind") == "model":
                    check = CG._qualified_model_check(raw)
                    if not isinstance(check, dict) or check.get("status") != "pass":
                        return None
                required = set(cap.get("required_oracle_tiers") or [])
                tiers = {
                    tier: record.get("status")
                    for tier, record in (view.get("tiers") or {}).items()
                    if isinstance(record, dict)
                }
                if (
                    view.get("status") != "pass"
                    or view.get("failure")
                    or not required
                    or any(tiers.get(tier) != "pass" for tier in required)
                ):
                    return None
                row = {"status": "pass", "tiers": tiers}
            retained = _artifacts(path.parent)
            if not ER.expects_refusal(cap) and not _retained_program(retained):
                return None
            artifacts.extend(retained)
            rows[name] = row
        if set(rows) != set(cohort):
            return None
        # The completed archive is written by the original grader after all
        # shape/resource checks, separately from its pre-probe snapshot archive.
        payload, completed = None, None
        if verdict_path is not None:
            completed = Path(verdict_path)
            payload = completed.read_text()
            if json.loads(payload) != json.loads(json.dumps(verdict, default=str)):
                return None
        for path in (*evidence_paths, *((completed,) if completed is not None else ())):
            path = Path(path)
            if path.resolve() != path or not path.is_relative_to(run_dir.resolve()):
                return None
            artifacts.extend(_artifacts(path) if path.is_dir() else ((path, _sha(path)),))
        return DriverGradeReceipt(candidate, runs, public, source, binding, tuple(artifacts), rows,
                                  completed, payload, oot_commit)
    except Exception:  # noqa: BLE001 -- optional evidence retention is never a grade gate
        return None


def reuse_checkpoint(receipt, *, submission, inputs, run_dir, eligible, held, cert_tier, attempt):
    """Supply the checkpoint's tally from the same retained full-grade evidence."""
    from .certification import _l3_attempt_tally

    if (
        type(receipt) is not DriverGradeReceipt
        or held
        or set(eligible) != set(receipt.rows)
        or len(eligible) != len(receipt.rows)
        or not receipt.current(submission=Path(submission), inputs=inputs, run_dir=Path(run_dir))
    ):
        return None
    rows, passed, count = _l3_attempt_tally(receipt.rows, [], cert_tier)
    if passed != count or not count:
        return None
    return {
        "attempt": attempt,
        "all_pass": True,
        "n_passed": passed,
        "n_capsules": count,
        "n_rtl_passed": sum(row.get("l3_status") == "pass" for row in rows),
        "n_expected_refusal_passed": sum(row.get("validated_refusal") is True for row in rows),
        "n_not_promoted": 0,
        "cert_tier": cert_tier,
        "per_capsule": rows,
        "rtl_checks": None,
        "evidence_reuse": {
            "stage": "completed_driver_full_grade",
            "measured_now": False,
            "runs_root": str(receipt.runs),
            "source_sha256": receipt.source_sha256,
        },
    }


def reuse_post_turn(receipt, *, ws, run_dir, rnd, inputs):
    """Reopen a completed original full grade, preserving its measured evidence.

    Only the host's in-memory receipt can select this path. Author verdict files,
    fast screens, partial grades and missing final archives cannot authorize it.
    Formal/public/hidden certification remains a separate unchanged consumer.
    """
    from merlin.common import oot_repo as O
    from merlin_experiments.phase1 import oot_history as OH

    from . import loop_grading as LG

    try:
        if type(receipt) is not DriverGradeReceipt or receipt.verdict_json is None:
            return None
        ws, run_dir = Path(ws), Path(run_dir)
        verdict = json.loads(receipt.verdict_json)
        if (
            verdict.get("all_pass") is not True or verdict.get("gradeable") is not True
            or verdict.get("snapshot_current") is not True
            or verdict.get("current_source_eligible") is not True
            or not receipt.current(submission=ws / "submission", inputs=inputs, run_dir=run_dir)
        ):
            return None
        repo = OH.repo_path(run_dir)
        if repo.exists() and (
            receipt.oot_commit is None or O.resolve(repo, "HEAD") != receipt.oot_commit
            or O.tree_digest(repo, receipt.oot_commit) != O.package_digest(receipt.candidate)
        ):
            return None
        # A round still records its exact source under the ordinary OOT owner.
        # This checkpoint is not a new simulator measurement.
        key = f"{rnd:02d}"
        committed = OH.commit_graded(run_dir, receipt.candidate, label="round", key=key,
                                    sandbox_roots=(ws.parent,))
        if repo.exists() and committed is None:
            return None
        verdict["evidence_reuse"] = {
            "stage": "completed_driver_full_grade", "measured_now": False,
            "runs_root": str(receipt.runs), "source_sha256": receipt.source_sha256,
            "completed_verdict": {"path": str(receipt.verdict_path), "sha256": _sha(receipt.verdict_path)},
        }
        if not receipt.current(submission=ws / "submission", inputs=inputs, run_dir=run_dir):
            return None
    except Exception:  # noqa: BLE001 -- changed/unavailable evidence takes ordinary grading
        return None
    # Publication failures propagate; never hide a half-published round by
    # starting another expensive grade against the same owned evidence.
    LG.write_verdict(run_dir / "qa_history" / f"verdict_round_{key}.json", verdict)
    verdict = LG.write_verdict(ws / "qa" / "verdict.json", verdict)
    OH.record_round(run_dir, committed, label="round", key=key, verdict=verdict)
    print(f"[round {rnd}] reused completed current-source full grade; no numerical jobs started", flush=True)
    return {**verdict, **({"oot_commit": committed.commit} if committed is not None else {})}


def post_turn_grade(receipt, ws, run_dir, rnd, no_oracle, timeout, *, inputs, grade_callback):
    """Called only after the background owner has drained; no parallel grade."""
    verdict = None if no_oracle else reuse_post_turn(receipt, ws=ws, run_dir=run_dir, rnd=rnd, inputs=inputs)
    return verdict if verdict is not None else grade_callback(ws, run_dir, rnd, no_oracle, timeout)
