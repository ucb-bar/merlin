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


def retain_driver_grade(*, candidate, runs, public, verdict, inputs, run_dir):
    """Retain a complete ordinary grade, never a screen or a redacted claim."""
    from merlin.targetgen import capsule_grade as CG
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import expected_refusal as ER

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
                child = raw.get("original_compiler_child") or {}
                if (
                    child.get("schema") != "merlin.original_model_child.v1"
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
        return DriverGradeReceipt(candidate, runs, public, source, binding, tuple(artifacts), rows)
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
