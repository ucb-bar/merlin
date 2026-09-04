"""Validation of capture-owned semantic inference sessions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable

import numpy as np

from merlin.common.schemas import validate_or_raise
from merlin.common.yaml import load_yaml

from .paper import ModelSpec, SessionSpec


def _sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _paper_input_path(bundle: Path, value: object) -> tuple[Path | None, str | None]:
    """Resolve one manifest path without permitting an escape from the frozen bundle."""
    text = str(value or "")
    pure = PurePosixPath(text)
    if (not text or pure.is_absolute() or ".." in pure.parts
            or text != pure.as_posix()):
        return None, f"paper input artifact path is unsafe: {text!r}"
    root = bundle.resolve()
    path = (root / pure).resolve()
    if not path.is_relative_to(root):
        return None, f"paper input artifact escapes its bundle: {text!r}"
    return path, None


def _environment_value(environment: dict[str, Any], suffix: str,
                       model: str) -> tuple[str | None, list[str]]:
    values = [str(value) for key, value in environment.items() if str(key).endswith(suffix)]
    if len(values) != 1:
        return None, [
            f"{model}: paper input environment must contain exactly one *{suffix} value"
        ]
    return values[0], []


def _checkpoint_revisions(checkpoint: object) -> dict[str, str]:
    if not isinstance(checkpoint, dict):
        return {}
    revisions: dict[str, str] = {}
    repo_id, revision = checkpoint.get("repo_id"), checkpoint.get("revision")
    if repo_id is not None and revision is not None:
        revisions[str(repo_id)] = str(revision)
    for component in checkpoint.get("components", ()) or ():
        revisions.update(_checkpoint_revisions(component))
    return revisions


def validate_paper_input_binding(bundle: str | Path,
                                 models: Iterable[ModelSpec]) -> list[str]:
    """Bind capture provenance expectations to the exact prepared-input manifest.

    The whole-tree digest proves which bundle was selected, while this check proves that each
    model's ``expected_provenance`` actually names the input artifact and source recorded by that
    bundle.  Without both checks, a valid but unrelated bundle could be attached to a capture.
    """
    bundle = Path(bundle)
    record_path = bundle / "paper_inputs.json"
    if not record_path.is_file():
        return [f"paper input record is absent: {record_path}"]
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"paper input record is invalid: {exc}"]
    if not isinstance(record, dict) or not isinstance(record.get("models"), dict):
        return ["paper input record models must be a mapping"]

    expected_models = {model.name: model for model in models}
    actual_models = record["models"]
    errors: list[str] = []
    if set(actual_models) != set(expected_models):
        errors.append(
            "paper input record model set differs from the study: "
            f"study={sorted(expected_models)} bundle={sorted(actual_models)}")
    active = record.get("active_holdouts")
    if not isinstance(active, list) or set(map(str, active)) != set(expected_models):
        errors.append("paper input record active_holdouts differs from the study")

    for name, model in expected_models.items():
        detail = actual_models.get(name)
        if not isinstance(detail, dict):
            continue
        environment = detail.get("environment")
        provenance = detail.get("provenance")
        artifacts = detail.get("artifacts")
        if not isinstance(environment, dict) or not isinstance(provenance, dict):
            errors.append(f"{name}: paper input environment/provenance is absent")
            continue
        if not isinstance(artifacts, list):
            errors.append(f"{name}: paper input artifacts must be a list")
            continue

        artifact_digests: dict[str, str] = {}
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                errors.append(f"{name}: paper input artifact record must be a mapping")
                continue
            relative = str(artifact.get("path", ""))
            path, path_error = _paper_input_path(bundle, relative)
            if path_error:
                errors.append(f"{name}: {path_error}")
                continue
            assert path is not None
            recorded_digest = str(artifact.get("sha256", ""))
            if not _sha256(recorded_digest):
                errors.append(f"{name}: artifact {relative!r} has no valid SHA-256")
            elif not path.is_file():
                errors.append(f"{name}: paper input artifact is absent: {relative}")
            elif _sha256_file(path) != recorded_digest:
                errors.append(f"{name}: paper input artifact SHA-256 differs: {relative}")
            if relative in artifact_digests:
                errors.append(f"{name}: duplicate paper input artifact path: {relative}")
            artifact_digests[relative] = recorded_digest

        bound: dict[str, Any] = {}

        def bind_artifact(suffix: str, field: str) -> None:
            value, problems = _environment_value(environment, suffix, name)
            errors.extend(problems)
            if value is None:
                return
            prefix = "{bundle}/"
            if not value.startswith(prefix):
                errors.append(f"{name}: *{suffix} is not relative to the paper input bundle")
                return
            relative = value[len(prefix):]
            if relative not in artifact_digests:
                errors.append(f"{name}: *{suffix} does not name a recorded artifact")
                return
            bound[field] = artifact_digests[relative]

        def bind_source(suffix: str, field: str) -> None:
            value, problems = _environment_value(environment, suffix, name)
            errors.extend(problems)
            if value is None:
                return
            bound[field] = value
            if provenance.get("input_source") != value:
                errors.append(
                    f"{name}: manifest input_source differs from its paper environment")

        if model.session.kind == "autoregressive_decode":
            bind_artifact("_TOKEN_IDS", "token_sha256")
            bind_source("_TOKEN_SOURCE", "token_source")
        elif model.session.kind == "action_chunk":
            bind_artifact("_INPUT_NPZ", "input_sha256")
            bind_source("_INPUT_SOURCE", "input_source")
            revisions = _checkpoint_revisions(provenance.get("checkpoint"))
            if model.checkpoint not in revisions:
                errors.append(f"{name}: paper bundle omits checkpoint {model.checkpoint!r}")
            else:
                bound["checkpoint_revision"] = revisions[model.checkpoint]
        elif model.session.kind == "image_stream":
            bind_artifact("_INPUT_NPZ", "input_sha256")
            bind_source("_INPUT_SOURCE", "input_source")
            preprocessing, problems = _environment_value(
                environment, "_PREPROCESSING", name)
            errors.extend(problems)
            if preprocessing is not None:
                bound["preprocessing"] = preprocessing
                recorded = provenance.get("preprocessing")
                if (not isinstance(recorded, dict)
                        or recorded.get("name") != preprocessing):
                    errors.append(
                        f"{name}: manifest preprocessing differs from its paper environment")
        elif model.session.kind == "recurrent_frames":
            bind_artifact("_SESSION_NPZ", "session_sha256")
            bind_source("_SESSION_SOURCE", "session_source")
            bind_artifact("_CKPT", "checkpoint_sha256")
        else:
            errors.append(f"{name}: unsupported paper input session kind {model.session.kind!r}")

        for key, actual_value in bound.items():
            if key not in model.expected_provenance:
                errors.append(f"{name}: expected_provenance omits bundle-bound field {key!r}")
            elif model.expected_provenance[key] != actual_value:
                errors.append(
                    f"{name}: expected_provenance {key!r} differs from the paper bundle: "
                    f"study={model.expected_provenance[key]!r} bundle={actual_value!r}")
    return errors


def _validate_trajectory_reference(bundle: Path, contract: dict, field: str,
                                   observations: int) -> tuple[dict, list[str]]:
    errors: list[str] = []
    spec = contract.get(field)
    if not isinstance(spec, dict) or spec.get("scope") != "trajectory":
        return {}, [f"{field} must cover the full trajectory"]
    expected_reference = "eager_fp32" if field == "quality" else "eager_same_precision"
    if spec.get("reference") != expected_reference:
        errors.append(
            f"{field} reference must be {expected_reference}, got {spec.get('reference')!r}")
    expected_digest = str(spec.get("reference_sha256", ""))
    if not _sha256(expected_digest):
        errors.append(f"{field} reference has no valid SHA-256")
    golden = bundle / str(spec.get("golden", ""))
    key = str(spec.get("key", ""))
    if not golden.is_file():
        errors.append(f"{field} trajectory golden is absent")
        return spec, errors
    with np.load(golden, mmap_mode="r") as values:
        if key not in values.files:
            errors.append(f"{field} trajectory key {key!r} is absent")
        else:
            array = np.ascontiguousarray(values[key])
            if int(array.shape[0]) != observations:
                errors.append(
                    f"{field} golden has {array.shape[0]} observations, expected {observations}")
            actual_digest = hashlib.sha256(array.tobytes()).hexdigest()
            if _sha256(expected_digest) and actual_digest != expected_digest:
                errors.append(f"{field} reference SHA-256 differs from the trajectory bytes")
    return spec, errors


def _validate_multi_program_data(capture: Path, session: dict, expected: SessionSpec) -> list[str]:
    """Validate every child stream/golden after the ABI graph itself has been checked."""
    from merlin.llvmlower.session_bundle import load as load_multi_program_session

    errors: list[str] = []
    try:
        compiled = load_multi_program_session(capture)
    except (OSError, ValueError) as exc:
        return [f"multi-program session is not executable: {exc}"]
    by_name = {program.name: program for program in compiled.programs}
    quality_program = by_name[compiled.quality_program]
    if quality_program.steps != expected.observations:
        errors.append(
            f"quality program has {quality_program.steps} observations, expected exactly "
            f"{expected.observations}")
    for program in compiled.programs:
        child_path = program.bundle / "session_contract.yaml"
        if not child_path.is_file():
            continue
        child = load_yaml(child_path)
        streams = child.get("streams", ()) or () if isinstance(child, dict) else ()
        if (session.get("kind") == "autoregressive_decode"
                and program.name == compiled.quality_program and not streams):
            errors.append(f"program {program.name}: autoregressive observations need a token stream")
        if streams:
            input_path = program.bundle / str(child.get("inputs", "session_inputs.npz"))
            if not input_path.is_file():
                errors.append(f"program {program.name}: session input corpus is absent")
            else:
                with np.load(input_path, mmap_mode="r") as values:
                    counts = []
                    for stream in streams:
                        key = str(stream.get("key", "")) if isinstance(stream, dict) else ""
                        if key not in values.files:
                            errors.append(f"program {program.name}: stream key {key!r} is absent")
                        else:
                            counts.append(int(values[key].shape[0]))
                    if counts and (len(set(counts)) != 1 or counts[0] != program.steps):
                        errors.append(
                            f"program {program.name}: stream lengths must equal its {program.steps} steps")
        if program.name == compiled.quality_program:
            correctness, correctness_errors = _validate_trajectory_reference(
                program.bundle, child, "correctness", expected.observations)
            quality, quality_errors = _validate_trajectory_reference(
                program.bundle, child, "quality", expected.observations)
            errors.extend(f"program {program.name}: {error}"
                          for error in correctness_errors + quality_errors)
            if (correctness and quality
                    and correctness.get("golden") == quality.get("golden")):
                errors.append(
                    f"program {program.name}: correctness and FP32 quality must use distinct artifacts")
    return errors


def validate_capture_session(capture: str | Path, expected: SessionSpec, *,
                             expected_provenance: dict[str, Any] | None = None
                             ) -> tuple[dict, list[str]]:
    """Check that a capture can support the frozen paper session without synthetic repetition.

    This check is shared by freeze and live preflight.  It deliberately treats ``paper_ready`` as
    evidence supplied by the capture owner: a random checkpoint or synthetic observation stream may
    be useful for compiler bring-up, but cannot be frozen into a paper result.
    """
    capture = Path(capture)
    errors: list[str] = []
    session_path = capture / "session_contract.yaml"
    if not session_path.is_file():
        return {}, [f"session contract is absent: {session_path}"]
    session = load_yaml(session_path)
    try:
        validate_or_raise(session, "session_contract")
    except ValueError as exc:
        return session if isinstance(session, dict) else {}, [str(exc)]
    version = int(session.get("version", 0))
    if version not in {1, 2}:
        errors.append("session contract version must be 1 or 2")
    if session.get("paper_ready") is not True:
        errors.append("session contract is not marked paper_ready=true")
    if session.get("kind") != expected.kind:
        errors.append(f"session kind mismatch: study={expected.kind} capture={session.get('kind')}")
    if list(session.get("stages", ())) != list(expected.stages):
        errors.append("session stages differ from the frozen study")
    provenance = session.get("provenance", {}) or {}
    if not isinstance(provenance, dict):
        errors.append("paper session provenance must be a mapping")
        provenance = {}
    elif session.get("paper_ready") is True:
        if provenance.get("full_checkpoint") is not True:
            errors.append("paper session does not prove a full pretrained checkpoint")
        if expected.kind == "autoregressive_decode":
            if provenance.get("synthetic_tokens") is not False:
                errors.append("paper autoregressive session uses synthetic tokens")
            if not _sha256(provenance.get("token_sha256")):
                errors.append("paper token corpus has no valid SHA-256")
            if str(provenance.get("token_source", "")).startswith("synthetic"):
                errors.append("paper token corpus has no external attribution")
        elif expected.kind == "action_chunk":
            if provenance.get("synthetic_inputs") is not False:
                errors.append("paper action session uses synthetic inputs")
            if not _sha256(provenance.get("input_sha256")):
                errors.append("paper action input corpus has no valid SHA-256")
        elif expected.kind == "image_stream":
            if provenance.get("synthetic_inputs") is not False:
                errors.append("paper image session uses synthetic inputs")
            if not _sha256(provenance.get("input_sha256")):
                errors.append("paper image corpus has no valid SHA-256")
            if provenance.get("preprocessing") != "IMAGENET1K_V2":
                errors.append("paper ResNet corpus does not declare IMAGENET1K_V2 preprocessing")
            if not _sha256(provenance.get("checkpoint_sha256")):
                errors.append("paper ResNet checkpoint has no valid tensor SHA-256")
        elif expected.kind == "recurrent_frames":
            if provenance.get("synthetic_session") is not False:
                errors.append("paper recurrent session uses a synthetic trajectory")
            if not _sha256(provenance.get("session_sha256")):
                errors.append("paper recurrent trajectory has no valid SHA-256")
            if not _sha256(provenance.get("checkpoint_sha256")):
                errors.append("paper recurrent checkpoint has no valid SHA-256")
    for key, expected_value in (expected_provenance or {}).items():
        actual_value = provenance.get(key)
        if actual_value != expected_value:
            errors.append(
                f"session provenance {key!r} differs from the frozen paper input: "
                f"study={expected_value!r} capture={actual_value!r}")
    captured_parameters = session.get("parameters", {}) or {}
    if not isinstance(captured_parameters, dict):
        errors.append("session parameters must be a mapping")
    else:
        for key, value in expected.parameters.items():
            if captured_parameters.get(key) != value:
                errors.append(
                    f"session parameter {key!r} differs: study={value!r} "
                    f"capture={captured_parameters.get(key)!r}")

    timed_stages = expected.parameters.get("timed_stages")
    if timed_stages is not None:
        schedule = session.get("stage_schedule", ()) or ()
        if not isinstance(schedule, list) or not all(isinstance(value, dict) for value in schedule):
            errors.append("session stage_schedule is required for timed_stages")
        else:
            scheduled_names = [str(value.get("name", "")) for value in schedule]
            if scheduled_names != list(expected.stages):
                errors.append("session stage_schedule order differs from the frozen stages")
            actual_timed = [str(value.get("name")) for value in schedule
                            if value.get("timed") is True]
            if actual_timed != list(timed_stages):
                errors.append(
                    f"session timed stages differ: study={list(timed_stages)} "
                    f"capture={actual_timed}")
            for stage in schedule:
                if stage.get("timed") is True and not str(stage.get("execution", "")).startswith(
                        "compiled"):
                    errors.append(
                        f"timed stage {stage.get('name')!r} is not executed by compiled code")

    states = session.get("states", ()) or ()
    state_names = [str(value.get("name")) for value in states if isinstance(value, dict)]
    if len(state_names) != len(set(state_names)):
        errors.append("session ABI has duplicate carried-state names")
    if set(state_names) != set(expected.carried_state):
        errors.append(
            "session carried state differs from study: "
            f"study={sorted(expected.carried_state)} capture={sorted(state_names)}")

    if version == 2:
        errors.extend(_validate_multi_program_data(capture, session, expected))
    else:
        streams = session.get("streams", ()) or ()
        if not streams:
            session_steps = int(session.get("steps", 0) or 0)
            if expected.kind != "action_chunk":
                errors.append("paper session has no per-observation input stream")
            elif session_steps != expected.observations:
                errors.append(
                    f"session has {session_steps} state-transition steps, expected exactly "
                    f"{expected.observations}")
        else:
            stream_path = capture / str(session.get("inputs", "session_inputs.npz"))
            if not stream_path.is_file():
                errors.append(f"session input corpus is absent: {stream_path}")
            else:
                with np.load(stream_path, mmap_mode="r") as stream_data:
                    counts: list[int] = []
                    for stream in streams:
                        key = str(stream.get("key", "")) if isinstance(stream, dict) else ""
                        if key not in stream_data.files:
                            errors.append(f"session stream key {key!r} is absent")
                        else:
                            counts.append(int(stream_data[key].shape[0]))
                    if counts and len(set(counts)) != 1:
                        errors.append(
                            f"session input streams have different lengths: {sorted(set(counts))}")
                    if counts and min(counts) != expected.observations:
                        errors.append(
                            f"session has {min(counts)} observations, expected exactly "
                            f"{expected.observations}")

        correctness, correctness_errors = _validate_trajectory_reference(
            capture, session, "correctness", expected.observations)
        quality, quality_errors = _validate_trajectory_reference(
            capture, session, "quality", expected.observations)
        errors.extend(correctness_errors + quality_errors)
        if correctness and quality and correctness.get("golden") == quality.get("golden"):
            errors.append("correctness and FP32 quality must use distinct artifacts")
    return session, errors
