"""Immutable static bundle publication and staged member decoding for portfolio reuse.

Callers admit checkpoint authority and recompute scientific readiness. Iterating a
loaded bundle checks one member identity at a time; artifact decoding remains a
separate operation so live admission can occur at the original observation point.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_bytes

from . import contracts as P2_CONTRACTS
from . import reporting
from . import static_identity as SI
from .broker_evidence import _is_sha256


def load_pinned_read_only_mapping(path: Path, digest: str, *, label: str) -> dict[str, Any]:
    """Read one pinned mapping through the shared immutable-JSON byte reader."""
    path = Path(path)
    if (
        not _is_sha256(digest)
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_mode & 0o222
    ):
        raise ValueError(f"{label} is absent, mutable, linked, or changed")
    try:
        document = reporting.read_immutable_json(path, digest, label=label)
    except reporting.ReportingGateError as exc:
        if isinstance(exc.__cause__, (UnicodeDecodeError, ValueError)):
            raise P2_CONTRACTS.StageGateError(f"stage input is unreadable at {path}: {exc.__cause__}") from exc
        raise ValueError(f"{label} is absent, mutable, linked, or changed") from exc
    if not isinstance(document, dict):
        raise P2_CONTRACTS.StageGateError(f"stage input must be a mapping: {path}")
    return document


def atomic_static_write(name: str, record: Mapping[str, Any], *, output: Path) -> Path:
    """Publish one immutable cache object only after its complete payload reaches storage."""
    P2_CONTRACTS.safe_component(name, label="static cache filename")
    path = output / name
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = output / f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
    created = False
    try:
        payload = P2_CONTRACTS.canonical_json(record)
        with temporary.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        # link(2) is atomic and refuses an existing destination; replace(2) would silently
        # overwrite a concurrently published cache object.
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(output, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if created and temporary.exists():
            temporary.unlink()
    return path


def validate_static_artifacts(artifacts: Mapping[str, Any], *, analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimal artifact set used by current host-owned probe preparation."""
    allowed = {
        "lowered_text",
        "decoded_trace",
        "command_buffer",
        "command_buffer_text",
        "candidate_sha256",
        "candidate_lowered_sha256",
        "candidate_command_buffer_sha256",
        "task_instruction_evidence",
        "baseline_artifacts",
    }
    required = {
        "lowered_text",
        "decoded_trace",
        "command_buffer",
        "command_buffer_text",
        "candidate_sha256",
        "candidate_lowered_sha256",
        "candidate_command_buffer_sha256",
        "task_instruction_evidence",
    }
    if not isinstance(artifacts, Mapping) or not required.issubset(artifacts):
        raise ValueError("static analysis bundle lacks required primary artifacts")
    if set(artifacts) - allowed:
        raise ValueError("static analysis bundle contains a non-static artifact field")
    result = copy.deepcopy(dict(artifacts))
    lowered, command_text = result["lowered_text"], result["command_buffer_text"]
    if (
        not isinstance(lowered, str)
        or not isinstance(command_text, str)
        or sha256_bytes(lowered.encode("utf-8")) != result["candidate_lowered_sha256"]
        or sha256_bytes(command_text.encode("utf-8")) != result["candidate_command_buffer_sha256"]
        or result["candidate_sha256"] != analysis.get("candidate_sha256")
        or result["candidate_lowered_sha256"] != (analysis.get("emission") or {}).get("candidate_lowered_sha256")
        or result["candidate_command_buffer_sha256"]
        != (analysis.get("emission") or {}).get("candidate_command_buffer_sha256")
    ):
        raise ValueError("static analysis artifact content binding changed")
    try:
        parsed = json.loads(command_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("static analysis command buffer is malformed") from exc
    if parsed != result["command_buffer"]:
        raise ValueError("static analysis command-buffer representations disagree")
    baseline = result.get("baseline_artifacts")
    if baseline is not None:
        if (
            not isinstance(baseline, Mapping)
            or sha256_bytes(str(baseline.get("lowered_text", "")).encode("utf-8")) != baseline.get("lowered_sha256")
            or sha256_bytes(str(baseline.get("command_buffer_text", "")).encode("utf-8"))
            != baseline.get("command_buffer_sha256")
        ):
            raise ValueError("static analysis baseline artifact content binding changed")
    return result


def persist_static_analysis_bundle(
    record: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    output: Path,
    capsule_sha256s: Sequence[str],
    portfolio_sha256: str,
    portfolio_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Persist only reusable analytical documents and primary emitted artifact bytes."""
    if not artifacts:
        return None  # Lightweight custom analyzers may deliberately expose no artifact sink.
    required_artifacts = {
        "lowered_text",
        "decoded_trace",
        "command_buffer",
        "command_buffer_text",
        "candidate_sha256",
        "candidate_lowered_sha256",
        "candidate_command_buffer_sha256",
        "task_instruction_evidence",
    }
    if not required_artifacts.issubset(artifacts):
        return None  # Test/development analyzers may publish only an auxiliary cache object.
    static_artifacts = {
        key: copy.deepcopy(value)
        for key, value in artifacts.items()
        if key != "interface" and key != "parsed_lowered_module"
    }
    if "baseline_artifacts" in static_artifacts:
        static_artifacts["baseline_artifacts"] = SI.static_only_copy(static_artifacts["baseline_artifacts"])
    static_artifacts = validate_static_artifacts(static_artifacts, analysis=record["analysis"])
    analyses = [record["analysis"], *[member["analysis"] for member in record["portfolio"]["members"][1:]]]
    member_artifacts = []
    by_capsule = dict(portfolio_artifacts or {})
    by_capsule.setdefault(capsule_sha256s[0], artifacts)
    for index, (capsule_sha256, analysis) in enumerate(zip(capsule_sha256s, analyses, strict=True)):
        current = by_capsule.get(capsule_sha256)
        if index == 0:
            if not current:
                # The OBJECTIVE's artifacts are required. Skipping them silently would leave a
                # bundle that looks complete and binds to nothing.
                raise ValueError("static analysis bundle lacks the objective's own artifacts")
            selected = static_artifacts
        elif not current:
            # A TRAINING member that published no primary artifacts is recorded, not fatal.
            # It used to abort the whole run, which is why the portfolio could never be
            # widened: the models with the most optimization room are exactly the ones whose
            # emission is hardest, and one of them failing to publish a reusable artifact set
            # killed the campaign for every member including the objective. The member's
            # ANALYSIS is still carried in `member_analyses` and is still worth having -- a
            # host-lane census is what identifies where a model's cost is, and it does not
            # depend on the artifacts being reusable for probe preparation.
            member_artifacts.append(
                {
                    "capsule_sha256": capsule_sha256,
                    "artifacts": None,
                    "status": "member_published_no_reusable_artifacts",
                    "consequence": (
                        "this member cannot back a probe or a cross-run artifact "
                        "binding; its analysis is still recorded in member_analyses"
                    ),
                }
            )
            continue
        else:
            selected = {
                key: copy.deepcopy(value)
                for key, value in current.items()
                if key != "interface" and key != "parsed_lowered_module"
            }
            if "baseline_artifacts" in selected:
                selected["baseline_artifacts"] = SI.static_only_copy(selected["baseline_artifacts"])
        member_artifacts.append(
            {
                "capsule_sha256": capsule_sha256,
                "artifacts": validate_static_artifacts(selected, analysis=analysis),
                "status": "reusable",
            }
        )
    bundle = {
        "schema": "global_cross_run_static_analysis_bundle_v1",
        "candidate_sha256": record["candidate_sha256"],
        "portfolio_sha256": portfolio_sha256,
        "binding": copy.deepcopy(record["cross_run_static_analysis_binding"]),
        "member_analyses": [SI.static_only_copy(analysis) for analysis in analyses],
        "portfolio_member_artifacts": member_artifacts,
        "excluded_evidence": sorted(SI.DYNAMIC_EVIDENCE_KEYS),
        "full_graph_compiler_invoked_by_import": False,
        "full_model_simulation_executed": False,
    }
    name = f"static_analysis_artifacts_{record['iteration']:04d}.json"
    path = atomic_static_write(name, bundle, output=output)
    return {"path": str(path.resolve()), "sha256": P2_CONTRACTS.sha256_file(path), "schema": bundle["schema"]}


@dataclass(frozen=True)
class StaticCacheMember:
    """One identity-checked analysis whose artifacts require separate admission."""

    analysis: dict[str, Any]
    _artifact_row: Mapping[str, Any]

    def decode_artifacts(self, *, analysis: Mapping[str, Any]) -> dict[str, Any]:
        """Decode after the caller admits the member and recomputes its readiness."""
        if self._artifact_row.get("artifacts") is None:
            raise ValueError(
                "static analysis seed bundle carries a portfolio member with no reusable "
                f"artifacts ({self._artifact_row.get('status')}); re-analyze this candidate instead "
                "of importing the bundle"
            )
        return validate_static_artifacts(self._artifact_row.get("artifacts"), analysis=analysis)


@dataclass(frozen=True)
class LoadedStaticBundle:
    """Validated bundle coverage, with member checks deferred until consumption."""

    _analyses: Sequence[Any]
    _artifact_rows: Sequence[Any]
    _capsule_sha256s: Sequence[str]
    _candidate_sha256: str

    def __iter__(self) -> Iterator[StaticCacheMember]:
        for capsule_sha256, raw_analysis, artifact_row in zip(
            self._capsule_sha256s, self._analyses, self._artifact_rows, strict=True
        ):
            if (
                not isinstance(raw_analysis, Mapping)
                or not isinstance(artifact_row, Mapping)
                or artifact_row.get("capsule_sha256") != capsule_sha256
                or raw_analysis.get("candidate_sha256") != self._candidate_sha256
                or raw_analysis.get("workload", {}).get("capsule_sha256") != capsule_sha256
            ):
                raise ValueError("static analysis seed member identity or order changed")
            yield StaticCacheMember(copy.deepcopy(dict(raw_analysis)), artifact_row)


def load_static_analysis_bundle(
    *,
    checkpoint_parent: Path,
    checkpoint_bundle_ref: object,
    iteration_bundle_ref: object,
    binding: Mapping[str, Any],
    candidate_sha256: str,
    portfolio_sha256: str,
    capsule_sha256s: Sequence[str],
) -> LoadedStaticBundle:
    """Read exact contained bundle bytes, checking coverage before lazy member admission."""
    if not isinstance(checkpoint_bundle_ref, Mapping) or checkpoint_bundle_ref != iteration_bundle_ref:
        raise ValueError("static analysis seed checkpoint has no exact artifact bundle")
    bundle_path_value = checkpoint_bundle_ref.get("path")
    if not isinstance(bundle_path_value, str):
        raise ValueError("static analysis seed artifact bundle path is malformed")
    bundle_path = Path(bundle_path_value)
    if not bundle_path.is_absolute() or bundle_path.parent.resolve() != checkpoint_parent:
        raise ValueError("static analysis seed artifact bundle escaped its experiment")
    bundle = load_pinned_read_only_mapping(
        bundle_path, checkpoint_bundle_ref.get("sha256"), label="static analysis seed artifact bundle"
    )
    analyses = bundle.get("member_analyses")
    artifact_rows = bundle.get("portfolio_member_artifacts")
    if (
        bundle.get("schema") != "global_cross_run_static_analysis_bundle_v1"
        or bundle.get("binding") != binding
        or bundle.get("candidate_sha256") != candidate_sha256
        or bundle.get("portfolio_sha256") != portfolio_sha256
        or not isinstance(analyses, list)
        or not isinstance(artifact_rows, list)
        or len(analyses) != len(capsule_sha256s)
        or len(artifact_rows) != len(capsule_sha256s)
    ):
        raise ValueError("static analysis seed bundle coverage or binding changed")
    return LoadedStaticBundle(analyses, artifact_rows, tuple(capsule_sha256s), candidate_sha256)
