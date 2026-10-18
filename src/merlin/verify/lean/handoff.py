"""Fail-closed validation of shared preprocessing before its tensor IR is consumed.

This driver binds a fresh scoped capture to its actual input and returned output, checks every
recorded boundary, and replays support, definedness and the complete chain. Success authorizes
only that tensor boundary under the chain's stated domain and ledger assumptions. Capture of the
original model, OOT compilation and downstream lowering are separate obligations.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import check, plans, stages


@dataclass(frozen=True)
class Options:
    """Explicit source entry, semantic index width and authoritative MLIR environment."""

    function: str
    index_bits: int
    compiler_python: Path

    def __post_init__(self) -> None:
        if not self.function:
            raise ValueError("shared-pass verification requires a source function")
        if type(self.index_bits) is not int or self.index_bits <= 0:
            raise ValueError("shared-pass verification requires a positive index width")


class VerificationError(RuntimeError):
    """The shared tensor boundary was not verified; downstream consumption must stop."""


@dataclass(frozen=True)
class VerifiedPreprocessing:
    """The exact immutable text checked by the chain, and its retained evidence path."""

    text: str
    stats: dict[str, Any]
    report: Path


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inventory(recording: stages.Capture) -> tuple[list[Path], dict[str, str]]:
    if not recording.complete:
        raise VerificationError("shared-pass capture did not complete")
    published = stages.completed(recording.root)
    if not published:
        raise VerificationError("shared-pass capture has no boundaries")
    reservations = sorted(p.name for p in (recording.root / ".reserved").iterdir())
    if reservations != sorted(f"{int(p.name.split('_', 1)[0]):06d}" for p in published):
        raise VerificationError("shared-pass capture has an incomplete boundary")
    if any(p.name.startswith(".writing-") for p in recording.root.iterdir()):
        raise VerificationError("shared-pass capture has an unpublished boundary")
    fingerprints = {}
    for stage in published:
        meta = json.loads((stage / "meta.json").read_bytes())
        if meta.get("schema") != "merlin.lean.stage.v1" or meta.get("run") != recording.run:
            raise VerificationError("shared-pass capture contains another recording run")
        for key in ("before", "after"):
            data = (stage / f"{key}.mlir").read_bytes()
            if _digest(data) != meta[f"{key}_sha256"]:
                raise VerificationError(f"{stage.name}: recorded {key} bytes changed")
            fingerprints[str(stage / f"{key}.mlir")] = _digest(data)
        fingerprints[str(stage / "meta.json")] = _digest((stage / "meta.json").read_bytes())
    fingerprints[str(recording.root / "handoff.json")] = _digest((recording.root / "handoff.json").read_bytes())
    return published, fingerprints


def run(
    source: str,
    output: Path,
    options: Options,
    preprocess: Callable[[str], tuple[str, dict[str, Any]]],
    *,
    plan: plans.Plan | None = None,
) -> VerifiedPreprocessing:
    """Run and verify shared passes, or raise before a caller can consume their output.

    ``preprocess`` must use the normal boundary recorder. The first/last recorded bytes must
    match the supplied source and returned text exactly. Every boundary, including unchanged
    ones, is checked. No unsupported stage or unavailable prerequisite is treated as success.
    The caller consumes the returned text, rather than reopening a mutable handoff file.
    A supplied plan must be resolved independently before execution. Without one, coverage is
    explicitly limited to recorded boundaries and does not establish roster completeness.
    """
    output = output.resolve()
    evidence = output.parent / "lean-shared-passes" / uuid.uuid4().hex
    evidence.mkdir(parents=True)
    report = evidence / "verification.json"
    result: dict[str, Any] = {
        "schema": "merlin.lean.shared_handoff.v1",
        "scope": "recorded source tensor IR through shared preprocessing handoff",
        "coverage_granularity": plan.granularity if plan else "recorded-boundary-only",
        "function": options.function,
        "index_bits": options.index_bits,
        "verdict": "unmeasured",
        "reason": "shared-pass validation has not completed",
    }
    check._atomic_json(report, result)
    manifest = evidence / "expected-plan.json"
    if plan is not None:
        manifest.write_bytes(plan.canonical_bytes())
        result.update(plan=str(manifest), plan_sha256=plan.sha256, expected_boundary_count=len(plan.boundaries))
        check._atomic_json(report, result)
    try:
        with stages.capture(evidence / "stages") as recording:
            target, stats = preprocess(source)
            output.write_text(target, encoding="utf-8")
            stages.record_handoff(output)
        published, frozen = _inventory(recording)
        if plan is not None:
            plans.replay(plan, manifest, recording.root)
        if (published[0] / "before.mlir").read_bytes() != source.encode("utf-8"):
            raise VerificationError("capture does not begin at the supplied source bytes")
        if (published[-1] / "after.mlir").read_bytes() != target.encode("utf-8"):
            raise VerificationError("capture does not end at the returned handoff bytes")
        result["boundaries"] = [p.name for p in published]
        for stage in published:
            receipt = check.check_typed_stage(stage, m2m_python=options.compiler_python, index_bits=options.index_bits)
            if receipt["verdict"] != "verified":
                raise VerificationError(f"{stage.name}: {receipt.get('reason', receipt['verdict'])}")
        program = published[0] / "before.mlir"
        for checker, name in ((check.check_support, "support"), (check.check_definedness, "definedness")):
            receipt = checker(
                program,
                options.function,
                evidence / name,
                m2m_python=options.compiler_python,
                index_bits=options.index_bits,
            )
            if receipt["verdict"] != "verified":
                raise VerificationError(f"source {name}: {receipt.get('reason', receipt['verdict'])}")
        chain_path = evidence / "chain.json"
        chain = check.compose_chain(
            published,
            evidence / "support" / "receipt.json",
            options.function,
            chain_path,
            handoff=recording.root / "handoff.json",
            m2m_python=options.compiler_python,
            definedness=evidence / "definedness" / "receipt.json",
        )
        if chain["verdict"] != "verified":
            raise VerificationError(chain.get("reason", "shared-pass chain did not verify"))
        if _inventory(recording) != (published, frozen) or output.read_bytes() != target.encode("utf-8"):
            raise VerificationError("shared-pass capture or handoff changed during validation")
        if plan is not None:
            plans.replay(plan, manifest, recording.root)
        result.update(
            verdict="verified",
            reason=(
                "every planned preprocessing boundary and source definedness replayed"
                if plan is not None
                else "every captured boundary and source definedness replayed; roster completeness not established"
            ),
            chain=str(chain_path),
            source_sha256=_digest(source.encode("utf-8")),
            handoff_sha256=_digest(target.encode("utf-8")),
            assumptions=chain["assumptions"],
            domain=chain["domain"],
        )
        check._atomic_json(report, result)
        return VerifiedPreprocessing(target, stats, report)
    except Exception as exc:
        result.update(
            verdict="abstracted" if isinstance(exc, (VerificationError, plans.PlanMismatch)) else "unmeasured",
            reason=f"{type(exc).__name__}: {exc}",
        )
        check._atomic_json(report, result)
        raise VerificationError(f"shared-pass verification stopped compilation: {exc}; evidence: {report}") from exc
