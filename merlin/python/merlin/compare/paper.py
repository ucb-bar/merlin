"""Version-2 paper-study contracts and fail-closed methodology checks.

The v1 comparison spec is intentionally retained for historical cached results.  A paper study has
stronger semantics: the development corpus is disjoint from the evaluation models, compiler and
runtime artifacts are frozen by digest, sessions declare carried state, and every matrix cell records
build/run/correctness as separate lifecycle facts.
"""
from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from merlin.common.schemas import validate_or_raise


_PRECISIONS = frozenset({"w8a8", "fp32"})
_SESSION_STATE = {
    "autoregressive_decode": frozenset({"kv_cache", "position"}),
    "action_chunk": frozenset({"flow_state", "timestep"}),
    "recurrent_frames": frozenset({"hidden_state", "cell_state"}),
    "image_stream": frozenset(),
}


def _mapping(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping")
    return raw


def _is_lower_hex(value: str, lengths: set[int]) -> bool:
    return len(value) in lengths and all(c in "0123456789abcdef" for c in value)


def _names_unique(values: Iterable[str], where: str) -> None:
    vals = list(values)
    duplicates = sorted({v for v in vals if vals.count(v) > 1})
    if duplicates:
        raise ValueError(f"duplicate {where}: {duplicates}")


def _require_safe_component(value: str, where: str) -> None:
    """Require an identifier that cannot escape an artifact-directory component."""
    allowed = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if (not value or value in {".", ".."} or Path(value).name != value
            or any(character not in allowed for character in value)):
        raise ValueError(f"{where} must be a safe path component, got {value!r}")


@dataclass(frozen=True)
class SessionSpec:
    kind: str
    warmups: int
    observations: int
    stages: tuple[str, ...]
    carried_state: tuple[str, ...]
    parameters: dict[str, Any] = field(default_factory=dict)
    measurement_repeats: int = 1

    @staticmethod
    def parse(raw: Any, where: str) -> "SessionSpec":
        raw = _mapping(raw, where)
        kind = str(raw.get("kind", ""))
        if kind not in _SESSION_STATE:
            raise ValueError(f"{where}.kind must be one of {sorted(_SESSION_STATE)}, got {kind!r}")
        stages = tuple(str(v) for v in raw.get("stages", ()) or ())
        state = tuple(str(v) for v in raw.get("carried_state", ()) or ())
        warmups = int(raw.get("warmups", 0))
        observations = int(raw.get("observations", 0))
        measurement_repeats = int(raw.get("measurement_repeats", 1))
        if warmups < 0 or observations < 1 or measurement_repeats < 1 or not stages:
            raise ValueError(
                f"{where} requires warmups >= 0, observations >= 1, "
                "measurement_repeats >= 1, and stages")
        missing = _SESSION_STATE[kind] - set(state)
        if missing:
            raise ValueError(f"{where} is {kind!r} but omits required carried state {sorted(missing)}")
        return SessionSpec(kind, warmups, observations, stages, state,
                           dict(raw.get("parameters", {}) or {}), measurement_repeats)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "warmups": self.warmups, "observations": self.observations,
                "stages": list(self.stages), "carried_state": list(self.carried_state),
                "parameters": dict(self.parameters),
                "measurement_repeats": self.measurement_repeats}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    capture: str
    checkpoint: str
    artifacts: dict[str, dict[str, str]]
    fidelity: str
    precisions: tuple[str, ...]
    session: SessionSpec
    expected_provenance: dict[str, Any]
    quality: dict[str, Any]
    memory: dict[str, Any]

    @staticmethod
    def parse(raw: Any, index: int) -> "ModelSpec":
        where = f"models[{index}]"
        raw = _mapping(raw, where)
        name = str(raw.get("name", ""))
        capture = str(raw.get("capture", name))
        checkpoint = str(raw.get("checkpoint", ""))
        artifacts = _mapping(raw.get("artifacts"), f"{where}.artifacts")
        fidelity = str(raw.get("fidelity", ""))
        precisions = tuple(str(v) for v in raw.get("precisions", ()) or ())
        if not name or not capture or not checkpoint:
            raise ValueError(f"{where} requires name, capture, and checkpoint")
        _require_safe_component(name, f"{where}.name")
        _require_safe_component(capture, f"{where}.capture")
        if fidelity not in {"full", "compiler_coverage"}:
            raise ValueError(f"{where}.fidelity must be full or compiler_coverage")
        if not precisions or not set(precisions) <= _PRECISIONS:
            raise ValueError(f"{where}.precisions must be a non-empty subset of {sorted(_PRECISIONS)}")
        if set(artifacts) != set(precisions):
            raise ValueError(f"{where}.artifacts must have exactly one entry per declared precision")
        normalized_artifacts: dict[str, dict[str, str]] = {}
        for precision, artifact in artifacts.items():
            artifact = _mapping(artifact, f"{where}.artifacts.{precision}")
            variant, digest = str(artifact.get("variant", "")), str(artifact.get("sha256", ""))
            if not variant or not digest:
                raise ValueError(f"{where}.artifacts.{precision} requires variant and sha256")
            normalized_artifacts[str(precision)] = {"variant": variant, "sha256": digest,
                                                    **({"path": str(artifact["path"])}
                                                       if artifact.get("path") else {})}
        quality = _mapping(raw.get("quality"), f"{where}.quality")
        if not quality.get("reference") or not quality.get("metric"):
            raise ValueError(f"{where}.quality requires reference and metric")
        memory = _mapping(raw.get("memory"), f"{where}.memory")
        policies = set(memory.get("policies", ()) or ())
        if not policies or not policies <= {"resident", "mmap"}:
            raise ValueError(f"{where}.memory.policies must contain resident and/or mmap")
        expected_provenance = raw.get("expected_provenance", {}) or {}
        if not isinstance(expected_provenance, dict):
            raise ValueError(f"{where}.expected_provenance must be a mapping")
        return ModelSpec(name, capture, checkpoint, normalized_artifacts, fidelity, precisions,
                         SessionSpec.parse(raw.get("session"), f"{where}.session"),
                         dict(expected_provenance), dict(quality), dict(memory))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "capture": self.capture, "checkpoint": self.checkpoint,
                "artifacts": {k: dict(v) for k, v in self.artifacts.items()},
                "fidelity": self.fidelity,
                "precisions": list(self.precisions), "session": self.session.to_dict(),
                "expected_provenance": dict(self.expected_provenance),
                "quality": dict(self.quality), "memory": dict(self.memory)}


@dataclass(frozen=True)
class BackendSpec:
    name: str
    kind: str
    runtime: str
    precisions: tuple[str, ...]
    quantization: str
    kernel_scope: str
    adapter: str
    options: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def parse(raw: Any, index: int) -> "BackendSpec":
        where = f"backends[{index}]"
        raw = _mapping(raw, where)
        name, kind, runtime = (str(raw.get(k, "")) for k in ("name", "kind", "runtime"))
        precisions = tuple(str(v) for v in raw.get("precisions", ()) or ())
        quantization, scope = (str(raw.get(k, "")) for k in ("quantization", "kernel_scope"))
        adapter = str(raw.get("adapter", ""))
        if not all((name, kind, runtime, quantization, scope, adapter)):
            raise ValueError(f"{where} omits a required backend field")
        _require_safe_component(name, f"{where}.name")
        if kind not in {"frozen_baseline", "compiler", "kernel_swap", "external_runtime"}:
            raise ValueError(f"{where}.kind is unknown: {kind!r}")
        if not precisions or not set(precisions) <= _PRECISIONS:
            raise ValueError(f"{where}.precisions must be a non-empty subset of {sorted(_PRECISIONS)}")
        return BackendSpec(name, kind, runtime, precisions, quantization, scope, adapter,
                           dict(raw.get("options", {}) or {}))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "runtime": self.runtime,
                "precisions": list(self.precisions), "quantization": self.quantization,
                "kernel_scope": self.kernel_scope, "adapter": self.adapter,
                "options": dict(self.options)}


@dataclass(frozen=True)
class MatrixCell:
    model: ModelSpec
    backend: BackendSpec
    precision: str
    core_count: int

    @property
    def key(self) -> str:
        return f"{self.model.name}/{self.backend.name}/{self.precision}/{self.core_count}c"


@dataclass(frozen=True)
class Preflight:
    errors: tuple[str, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.errors and not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {"ready": self.ready, "errors": list(self.errors),
                "blockers": list(self.blockers), "warnings": list(self.warnings)}


@dataclass(frozen=True)
class PaperStudySpec:
    version: int
    label: str
    status: str
    target: str
    primary_precision: str
    control_precision: str
    core_counts: tuple[int, ...]
    development_corpus: dict[str, Any]
    paper_inputs: dict[str, Any]
    holdout_models: tuple[str, ...]
    freeze: dict[str, Any]
    models: tuple[ModelSpec, ...]
    backends: tuple[BackendSpec, ...]
    reporting: dict[str, Any]
    source_path: Path | None = None

    @staticmethod
    def parse(raw: Any, *, source_path: Path | None = None) -> "PaperStudySpec":
        raw = _mapping(raw, "paper study")
        validate_or_raise(raw, "paper_study")
        if int(raw["version"]) != 2:
            raise ValueError("paper study version must be 2")
        status = str(raw["status"])
        if status not in {"draft", "frozen"}:
            raise ValueError("paper study status must be draft or frozen")
        primary = str(raw["primary_precision"])
        control = str(raw.get("control_precision", "fp32"))
        if primary not in _PRECISIONS or control not in _PRECISIONS or primary == control:
            raise ValueError("primary/control precision must be distinct members of {w8a8, fp32}")
        raw_core_counts = raw.get("core_counts", (1,)) or ()
        if (not isinstance(raw_core_counts, (list, tuple))
                or not raw_core_counts
                or any(type(value) is not int or value < 1 for value in raw_core_counts)
                or len(set(raw_core_counts)) != len(raw_core_counts)):
            raise ValueError("core_counts must contain unique exact positive integers")
        core_counts = tuple(raw_core_counts)
        models = tuple(ModelSpec.parse(v, i) for i, v in enumerate(raw["models"]))
        backends = tuple(BackendSpec.parse(v, i) for i, v in enumerate(raw["backends"]))
        if not models or not backends:
            raise ValueError("paper study requires at least one model and backend")
        _names_unique((m.name for m in models), "model names")
        _names_unique((b.name for b in backends), "backend names")
        holdout = tuple(str(v) for v in raw["holdout_models"])
        _names_unique(holdout, "holdout model names")
        if set(holdout) != {m.name for m in models}:
            raise ValueError("holdout_models must exactly equal the declared paper model names")
        dev = _mapping(raw["development_corpus"], "development_corpus")
        paper_inputs = raw.get("paper_inputs", {"path": "unresolved", "sha256": "unresolved"})
        paper_inputs = _mapping(paper_inputs, "paper_inputs")
        excluded = set(dev.get("excluded_models", ()) or ())
        if not set(holdout) <= excluded:
            raise ValueError("every paper model must be excluded from the development corpus")
        if int(dev.get("convergence_sweeps", 0)) < 2:
            raise ValueError("development_corpus.convergence_sweeps must be at least 2")
        freeze = _mapping(raw["freeze"], "freeze")
        if not freeze.get("forbid_model_name_dispatch") or not freeze.get("forbid_post_freeze_tuning"):
            raise ValueError("freeze must forbid model-name dispatch and post-freeze tuning")
        reporting = dict(raw.get("reporting", {}) or {})
        if reporting.get("same_buffer_repeat_is_diagnostic_only") is not True:
            raise ValueError("paper reporting must mark same-buffer repetition diagnostic-only")
        execution_order = reporting.get("execution_order", {}) or {}
        if (not isinstance(execution_order, dict)
                or execution_order.get("policy") != "deterministic_block_randomized"
                or execution_order.get("block_fields") != ["model", "precision", "core_count"]
                or execution_order.get("randomized_field") != "backend"
                or not _is_lower_hex(str(execution_order.get("seed_sha256", "")), {64})):
            raise ValueError(
                "paper reporting requires a frozen deterministic block-randomized backend order")
        claims = reporting.get("performance_claims", {}) or {}
        band = claims.get("parity_median_ratio_band", ()) if isinstance(claims, dict) else ()
        if (not isinstance(band, list) or len(band) != 2
                or not (0 < float(band[0]) < 1 < float(band[1]))
                or float(claims.get("win_median_ratio_max", 0)) != float(band[0])
                or claims.get("win_requires_nonoverlapping_observed_ranges") is not True
                or claims.get("win_requires_causal_attribution") is not True
                or claims.get("language") != "descriptive_ratio_not_statistical_significance"):
            raise ValueError("paper performance-claim thresholds are incomplete or inconsistent")
        scope_policy = reporting.get("performance_scope_policy", {}) or {}
        if not isinstance(scope_policy, dict):
            raise ValueError("reporting.performance_scope_policy must be a mapping")
        if (scope_policy.get("primary_table") != "end_to_end_continuous_sessions"
                or scope_policy.get("diagnostic_table") != "exact_declared_stage_subsets"
                or scope_policy.get("e2e_claim_requires_all_stages_timed") is not True
                or scope_policy.get("e2e_win_requires_attribution") is not True):
            raise ValueError(
                "paper reporting must use end-to-end continuous sessions as primary, preserve "
                "stage subsets as diagnostics, and require attribution for end-to-end wins")
        for model in models:
            if model.session.measurement_repeats < 3:
                raise ValueError(
                    f"model {model.name}: paper sessions require at least 3 independent repeats")
            timed = tuple(str(value) for value in
                          (model.session.parameters.get("timed_stages", model.session.stages) or ()))
            if timed != model.session.stages:
                raise ValueError(
                    f"model {model.name}: the primary paper session must time every stage in order")
            if model.session.parameters.get("paper_primary_scope") != "end_to_end":
                raise ValueError(
                    f"model {model.name}: session.parameters.paper_primary_scope must be end_to_end")
        return PaperStudySpec(2, str(raw["label"]), status, str(raw["target"]), primary, control,
                              core_counts, dict(dev), dict(paper_inputs), holdout, dict(freeze), models, backends,
                              reporting, source_path)

    @staticmethod
    def from_yaml(path: str | Path) -> "PaperStudySpec":
        source = Path(path).resolve()
        return PaperStudySpec.parse(yaml.safe_load(source.read_text(encoding="utf-8")),
                                    source_path=source)

    def matrix(self) -> tuple[MatrixCell, ...]:
        """All declared, precision-compatible cells; unsupported dtype pairs are never fabricated."""
        cells: list[MatrixCell] = []
        for model in self.models:
            for backend in self.backends:
                for precision in (self.primary_precision, self.control_precision):
                    if precision not in model.precisions or precision not in backend.precisions:
                        continue
                    for cores in self.core_counts:
                        cells.append(MatrixCell(model, backend, precision, cores))
        return tuple(cells)

    def preflight(self) -> Preflight:
        errors: list[str] = []
        blockers: list[str] = []
        warnings: list[str] = []
        if self.status != "frozen":
            blockers.append("study status is draft; freeze the compiler/runtime before live evaluation")
        for key in ("policy_sha256", "runtime_sha256"):
            value = str(self.freeze.get(key, ""))
            if not _is_lower_hex(value, {64}):
                blockers.append(f"freeze.{key} is unresolved or not a SHA-256 digest")
        compiler_sha = str(self.freeze.get("compiler_git_sha", ""))
        if not _is_lower_hex(compiler_sha, set(range(7, 41))):
            blockers.append("freeze.compiler_git_sha is unresolved or invalid")
        if not _is_lower_hex(str(self.freeze.get("compiler_source_sha256", "")), {64}):
            blockers.append("freeze.compiler_source_sha256 is unresolved or invalid")
        if str(self.freeze.get("toolchain_authority_path", "")) in {"", "unresolved"}:
            blockers.append("freeze.toolchain_authority_path is unresolved")
        if not _is_lower_hex(str(
                self.freeze.get("toolchain_authority_sha256", "")), {64}):
            blockers.append("freeze.toolchain_authority_sha256 is unresolved or invalid")
        if str(self.freeze.get("frozen_at", "")) in {"", "unresolved"}:
            blockers.append("freeze.frozen_at is unresolved")
        if not _is_lower_hex(str(self.freeze.get("baseline_sources_sha256", "")), {64}):
            blockers.append("freeze.baseline_sources_sha256 is unresolved")
        measurement_io = self.freeze.get("measurement_io")
        if not isinstance(measurement_io, dict):
            blockers.append("freeze.measurement_io is absent; canonical execution is blocked")
        if any(backend.kind == "external_runtime" for backend in self.backends):
            registration_path = str(
                self.freeze.get("external_package_registration_path", ""))
            if registration_path in {"", "unresolved"}:
                blockers.append("freeze.external_package_registration_path is unresolved")
            if not _is_lower_hex(str(
                    self.freeze.get("external_package_registration_sha256", "")), {64}):
                blockers.append("freeze.external_package_registration_sha256 is unresolved")
        input_path = str(self.paper_inputs.get("path", ""))
        if input_path in {"", "unresolved"}:
            blockers.append("paper_inputs.path is unresolved")
        if not _is_lower_hex(str(self.paper_inputs.get("sha256", "")), {64}):
            blockers.append("paper_inputs.sha256 is unresolved or invalid")
        for model in self.models:
            if not model.expected_provenance:
                blockers.append(f"model {model.name}: expected paper-input provenance is absent")
            for precision, artifact in model.artifacts.items():
                if not _is_lower_hex(artifact["sha256"], {64}):
                    blockers.append(f"model {model.name}/{precision}: capture sha256 is unresolved")
            if model.fidelity != "full":
                warnings.append(f"model {model.name}: compiler-coverage capture cannot support a full-model claim")
        for backend in self.backends:
            if not isinstance(backend.options.get("measurement_contracts"), dict):
                blockers.append(
                    f"backend {backend.name}: generated measurement_contracts are absent")
            if backend.adapter == "merlin_compile":
                if not _is_lower_hex(str(backend.options.get("package_sha256", "")), {64}):
                    blockers.append(f"backend {backend.name}: package_sha256 is unresolved")
            if backend.kind == "kernel_swap":
                if not _is_lower_hex(str(backend.options.get("kernel_source_sha256", "")), {64}):
                    blockers.append(f"backend {backend.name}: kernel_source_sha256 is unresolved")
            if backend.kind == "frozen_baseline":
                source_paths = backend.options.get("source_paths")
                source_digest = str(backend.options.get("kernel_source_sha256", ""))
                if not isinstance(source_paths, list) or not source_paths:
                    warnings.append(
                        f"backend {backend.name}: frozen_baseline causal attribution unavailable; "
                        "source_paths are absent")
                elif not _is_lower_hex(source_digest, {64}):
                    blockers.append(
                        f"backend {backend.name}: declared frozen_baseline source has no "
                        "kernel_source_sha256")
            if backend.kind == "external_runtime":
                if not _is_lower_hex(str(backend.options.get("framework_source_sha256", "")), {64}):
                    blockers.append(f"backend {backend.name}: framework_source_sha256 is unresolved")
                packages = backend.options.get("packages")
                if not isinstance(packages, dict):
                    blockers.append(f"backend {backend.name}: frozen session packages are absent")
                    continue
                for model in self.models:
                    for precision in backend.precisions:
                        if precision not in model.precisions:
                            continue
                        by_precision = packages.get(model.name)
                        row = (by_precision.get(precision)
                               if isinstance(by_precision, dict) else None)
                        if not isinstance(row, dict) or str(row.get("path", "")) in {
                                "", "unresolved"}:
                            blockers.append(
                                f"backend {backend.name}: package path is unresolved for "
                                f"{model.name}/{precision}")
                        elif not _is_lower_hex(str(row.get("sha256", "")), {64}):
                            blockers.append(
                                f"backend {backend.name}: package sha256 is unresolved for "
                                f"{model.name}/{precision}")
                        elif not _is_lower_hex(
                                str(row.get("build_environment_sha256", "")), {64}):
                            blockers.append(
                                f"backend {backend.name}: package build_environment_sha256 is "
                                f"unresolved for {model.name}/{precision}")
        if not any(self.primary_precision in b.precisions and b.kind == "external_runtime"
                   for b in self.backends):
            warnings.append("no external runtime supports the primary precision")
        if not self.matrix():
            errors.append("study matrix is empty")
        return Preflight(tuple(errors), tuple(blockers), tuple(warnings))

    def canonical_dict(self) -> dict[str, Any]:
        return {"version": self.version, "label": self.label, "status": self.status,
                "target": self.target, "primary_precision": self.primary_precision,
                "control_precision": self.control_precision, "core_counts": list(self.core_counts),
                "development_corpus": dict(self.development_corpus),
                "paper_inputs": dict(self.paper_inputs),
                "holdout_models": list(self.holdout_models), "freeze": dict(self.freeze),
                "models": [m.to_dict() for m in self.models],
                "backends": [b.to_dict() for b in self.backends],
                "reporting": dict(self.reporting)}

    def sha256(self) -> str:
        text = yaml.safe_dump(self.canonical_dict(), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_paper_result(result: dict[str, Any]) -> None:
    """Validate schema plus cross-field honesty invariants for one measured matrix cell."""
    validate_or_raise(result, "paper_run_result")
    allowed_top = set({
        "schema_version", "run_id", "timestamp", "git_sha", "study_label", "target",
        "model", "checkpoint", "artifact_sha256", "fidelity", "backend", "runtime",
        "precision", "quantization", "core_count", "session", "lifecycle", "correctness",
        "quality", "timing", "memory", "execution", "provenance", "measurement_receipt",
        "causal_attribution",
    })
    unknown_top = sorted(set(result) - allowed_top)
    if unknown_top:
        raise ValueError(f"paper result is closed; unrecognized top-level fields {unknown_top}")

    def closed(name: str, allowed: set[str]) -> dict[str, Any]:
        value = _mapping(result[name], name)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"{name} is closed; unrecognized fields {unknown}")
        return value

    session_mapping = closed(
        "session", {"kind", "warmups", "observations", "stages", "carried_state",
                    "parameters", "measurement_repeats"})
    parameters = _mapping(session_mapping.get("parameters", {}), "session.parameters")
    allowed_parameters = {
        "batch", "decode_tokens", "diagnostic_timed_stages", "paper_primary_scope",
        "prefill_policy", "prefill_tokens", "timed_stages", "action_horizon",
        "denoise_steps", "prefix_policy", "channels", "height", "width", "sequence_length",
    }
    unknown = sorted(set(parameters) - allowed_parameters)
    if unknown:
        raise ValueError(f"session.parameters is closed; unrecognized fields {unknown}")
    closed("lifecycle", {"built", "ran", "status", "reason"})
    correctness = closed("correctness", {
        "gate_ok", "scope", "steps", "min_cosine", "cosine", "max_relative_error",
        "max_abs", "top1_matches", "top1_agreement", "reference", "thresholds", "status",
        "cycles", "per_step",
    })
    quality = closed("quality", {
        "gate_ok", "metric", "value", "scope", "steps", "reference", "top1_agreement",
        "min_cosine", "per_step",
    })
    timing = closed("timing", {
        "unit", "sample_unit", "scope", "timed_stages", "excluded_stages", "samples",
        "stage_samples", "median", "p95", "drift",
    })
    closed("memory", {"policy", "peak_rss_bytes"})
    closed("execution", {
        "mode", "requested_mode", "fallback_used", "core_count", "requested_core_count",
        "affinity_source", "worker_threads", "worker_thread_source", "semantic_session",
        "same_input_repetition", "n_routed", "n_eligible", "n_candidates",
    })
    closed("provenance", {
        "study_sha256", "compiler_policy_sha256", "compiler_source_sha256", "runtime_sha256",
        "capture_session_identity_sha256", "vlen_bits", "vlen_source", "board_conditions",
        "binary", "package_sha256", "kernel_source_sha256", "framework_source_sha256",
        "framework_package_sha256", "stage_attribution", "stage_attribution_note",
        "external_runtime_protocol",
    })
    thresholds = correctness.get("thresholds")
    if thresholds is not None:
        thresholds = _mapping(thresholds, "correctness.thresholds")
        unknown = sorted(set(thresholds) - {
            "cosine_min", "max_relative_error", "top1_agreement"})
        if unknown:
            raise ValueError(
                f"correctness.thresholds is closed; unrecognized fields {unknown}")
    for section_name, section in (("correctness", correctness), ("quality", quality)):
        per_step = section.get("per_step")
        if per_step is None:
            continue
        if (not isinstance(per_step, list)
                or len(per_step) != int(session_mapping.get("observations") or 0)):
            raise ValueError(f"{section_name}.per_step must cover every observation")
        for index, row in enumerate(per_step):
            row = _mapping(row, f"{section_name}.per_step[{index}]")
            if (set(row) != {"index", "value", "gate_ok"} or row["index"] != index
                    or not isinstance(row["value"], (int, float))
                    or isinstance(row["value"], bool) or type(row["gate_ok"]) is not bool):
                raise ValueError(f"{section_name}.per_step[{index}] is invalid")
    provenance = _mapping(result["provenance"], "provenance")
    conditions = provenance.get("board_conditions")
    if conditions is not None:
        conditions = _mapping(conditions, "provenance.board_conditions")
        unknown = sorted(set(conditions) - {"before", "after"})
        if unknown:
            raise ValueError(
                f"provenance.board_conditions is closed; unrecognized fields {unknown}")
        for endpoint in ("before", "after"):
            if endpoint not in conditions:
                continue
            row = _mapping(conditions[endpoint],
                           f"provenance.board_conditions.{endpoint}")
            unknown = sorted(set(row) - {
                "governor", "current_khz", "max_khz", "max_thermal_millic"})
            if unknown:
                raise ValueError(
                    f"provenance.board_conditions.{endpoint} is closed; "
                    f"unrecognized fields {unknown}")
    if int(result["schema_version"]) != 2:
        raise ValueError("paper result schema_version must be 2")
    # Attribution is populated only after all comparator cells exist.  Its local shape is checked
    # here; the report independently re-derives any claim-ready explanation from the frozen
    # ablation/structural manifest rather than trusting these result bytes.
    if "causal_attribution" in result:
        causal = _mapping(result["causal_attribution"], "causal_attribution")
        unknown = sorted(set(causal) - {"schema_version", "records"})
        if unknown:
            raise ValueError(
                f"causal_attribution is closed; unrecognized fields {unknown}")
        if int(causal.get("schema_version") or 0) != 1:
            raise ValueError("causal_attribution.schema_version must be 1")
        records = causal.get("records")
        if not isinstance(records, list):
            raise ValueError("causal_attribution.records must be a list")
        names: set[str] = set()
        for record in records:
            row = _mapping(record, "causal_attribution.records entry")
            comparator = str(row.get("comparator", ""))
            status = str(row.get("status", ""))
            if not comparator or comparator in names:
                raise ValueError("causal attribution comparators must be nonempty and unique")
            names.add(comparator)
            if status not in {"available", "unavailable"}:
                raise ValueError("causal attribution status must be available or unavailable")
            if status == "unavailable":
                unknown = sorted(set(row) - {"comparator", "status", "reason"})
                if unknown:
                    raise ValueError(
                        f"unavailable causal attribution is closed; unrecognized fields {unknown}")
                if not str(row.get("reason", "")).strip():
                    raise ValueError("unavailable causal attribution requires a reason")
                continue
            unknown = sorted(set(row) - {
                "comparator", "status", "why", "how", "evidence"})
            if unknown:
                raise ValueError(
                    f"available causal attribution is closed; unrecognized fields {unknown}")
            if not str(row.get("why", "")).strip() or not str(row.get("how", "")).strip():
                raise ValueError("available causal attribution requires why and how")
            evidence = _mapping(row.get("evidence"), "causal attribution evidence")
            evidence_fields = {
                "binding_sha256", "ablation_sha256", "structural_sha256",
                "pair_contract_sha256", "pair_evidence_sha256",
                "transformation_delta_sha256", "control_measurement_roots_sha256",
                "treatment_measurement_roots_sha256",
                "generator_source_sha256", "control_artifact_sha256",
                "controller_pair_sha256",
                "treatment_artifact_sha256", "control_raw_log_sha256",
                "treatment_raw_log_sha256", "control_measurement_run_sha256",
                "treatment_measurement_run_sha256", "control_observation_sha256",
                "treatment_observation_sha256", "control_result_sha256",
                "treatment_result_sha256", "structural_generator_source_sha256",
                "structural_inspection_sha256", "structural_result_sha256",
                "control_build_receipt_sha256", "treatment_build_receipt_sha256",
                "control_benchmark_contract_sha256", "treatment_benchmark_contract_sha256",
                "control_analyzed_artifact_sha256",
                "treatment_analyzed_artifact_sha256",
            }
            unknown = sorted(set(evidence) - evidence_fields)
            if unknown:
                raise ValueError(
                    f"causal attribution evidence is closed; unrecognized fields {unknown}")
            for key in ("binding_sha256", "ablation_sha256", "structural_sha256"):
                if not _is_lower_hex(str(evidence.get(key, "")), {64}):
                    raise ValueError(f"available causal attribution has invalid {key}")
    if "measurement_receipt" in result:
        receipt = _mapping(result["measurement_receipt"], "measurement_receipt")
        unknown = sorted(set(receipt) - {"path", "sha256", "aet_run_id", "command_sha256"})
        if unknown:
            raise ValueError(
                f"measurement_receipt is closed; unrecognized fields {unknown}")
        if (not str(receipt.get("path", "")).strip()
                or not _is_lower_hex(str(receipt.get("sha256", "")), {64})
                or not _is_lower_hex(str(receipt.get("command_sha256", "")), {64})
                or receipt.get("aet_run_id") != result.get("run_id")):
            raise ValueError("measurement_receipt must bind a retained receipt to this AET run")
    lifecycle = _mapping(result["lifecycle"], "lifecycle")
    status = lifecycle.get("status")
    if status not in {"pass", "fail", "not_run", "error"}:
        raise ValueError(f"unknown lifecycle status {status!r}")
    if status == "pass":
        if not lifecycle.get("built") or not lifecycle.get("ran"):
            raise ValueError("passing result requires built=true and ran=true")
        if correctness.get("gate_ok") is not True:
            raise ValueError("passing result requires correctness.gate_ok=true")
        if quality.get("gate_ok") is not True:
            raise ValueError("passing result requires quality.gate_ok=true")
        execution = _mapping(result["execution"], "execution")
        if execution.get("fallback_used"):
            raise ValueError("a fallback execution cannot be a passing paper measurement")
        if execution.get("semantic_session") is not True:
            raise ValueError(
                "passing result requires execution.semantic_session=true from the measured runtime")
        if execution.get("same_input_repetition") is not False:
            raise ValueError(
                "passing result requires execution.same_input_repetition=false; repeated identical "
                "inputs are not continuous inference")
        if execution.get("mode") != execution.get("requested_mode"):
            raise ValueError("passing result execution mode differs from requested mode")
        if int(execution.get("core_count") or 0) != int(result["core_count"]):
            raise ValueError("passing result did not execute on the requested core count")
        if int(execution.get("requested_core_count") or 0) != int(result["core_count"]):
            raise ValueError("passing result requested_core_count differs from the matrix cell")
        if execution.get("affinity_source") != "sched_getaffinity":
            raise ValueError(
                "passing result requires an on-device sched_getaffinity core-count observation")
        if result.get("runtime") == "executorch":
            cores = int(result["core_count"])
            if (int(execution.get("worker_threads") or 0) != cores
                    or execution.get("worker_thread_source") != "proc_task_status"):
                raise ValueError(
                    "passing ExecuTorch result requires an exact externally observed "
                    "worker-thread configuration")
        backend = str(result.get("backend", ""))
        if backend in {"merlin_xnnpack", "merlin_openblas"}:
            routed = execution.get("n_routed")
            eligible = execution.get("n_eligible")
            candidates = execution.get("n_candidates")
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                   for value in (routed, eligible, candidates)):
                raise ValueError(
                    "passing kernel-swap result requires integer routed/eligible/candidate counts")
            if eligible <= 0 or routed != eligible or candidates < eligible:
                raise ValueError(
                    "passing kernel-swap result requires complete coverage of a nonempty declared "
                    "eligible GEMM set")
        provenance = _mapping(result["provenance"], "provenance")
        if provenance.get("vlen_source") != "csr" or int(provenance.get("vlen_bits") or 0) != 256:
            raise ValueError("passing K1 result requires CSR-observed VLEN=256")
        conditions = _mapping(provenance.get("board_conditions"),
                              "provenance.board_conditions")
        for endpoint in ("before", "after"):
            observed = _mapping(conditions.get(endpoint),
                                f"provenance.board_conditions.{endpoint}")
            if observed.get("governor") != "performance":
                raise ValueError("passing result requires the K1 performance governor")
            current, maximum = (int(observed.get("current_khz") or 0),
                                int(observed.get("max_khz") or 0))
            if current <= 0 or current != maximum:
                raise ValueError(
                    "passing result requires observed K1 current frequency equal to max frequency")
            if int(observed.get("max_thermal_millic") or 0) <= 0:
                raise ValueError("passing result requires an observed positive K1 temperature")
        memory = _mapping(result["memory"], "memory")
        if int(memory.get("peak_rss_bytes") or 0) <= 0:
            raise ValueError("passing result requires measured positive peak RSS")
    samples = timing.get("samples", [])
    session = _mapping(result["session"], "session")
    stages = [str(value) for value in session.get("stages", ()) or ()]
    timed_stages = [str(value) for value in timing.get("timed_stages", ()) or ()]
    if not stages or not timed_stages or not set(timed_stages) <= set(stages):
        raise ValueError("timing.timed_stages must be a non-empty subset of session.stages")
    expected_scope = "end_to_end" if timed_stages == stages else "stage_subset"
    if timing.get("scope") != expected_scope:
        raise ValueError(
            f"timing.scope must be {expected_scope!r} for stages={stages} "
            f"timed_stages={timed_stages}")
    expected_excluded = [stage for stage in stages if stage not in timed_stages]
    if list(timing.get("excluded_stages", ()) or ()) != expected_excluded:
        raise ValueError("timing.excluded_stages does not match the declared timed stages")
    parameters = session.get("parameters", {}) or {}
    if (isinstance(parameters, dict) and parameters.get("paper_primary_scope") == "end_to_end"
            and timing.get("scope") != "end_to_end"):
        raise ValueError("a primary paper result must time the complete continuous session")
    if lifecycle.get("ran") and not samples:
        raise ValueError("ran=true requires timing samples")
    if lifecycle.get("ran"):
        if timing.get("sample_unit") != "complete_session":
            raise ValueError("paper timing samples must each cover one complete session")
        repeats = int(session.get("measurement_repeats") or 0)
        if len(samples) != repeats:
            raise ValueError(
                f"ran=true requires exactly {repeats} full-session timing samples, "
                f"got {len(samples)}")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in samples):
            raise ValueError("timing samples must be positive integer nanoseconds")
        ordered = sorted(samples)
        n = len(ordered)
        median = (ordered[n // 2] if n % 2
                  else (ordered[n // 2 - 1] + ordered[n // 2]) // 2)
        p95 = ordered[min(n - 1, max(0, int(round(0.95 * (n - 1)))))]
        if timing.get("median") != median or timing.get("p95") != p95:
            raise ValueError("timing median/p95 do not match the recorded samples")
        stage_samples = timing.get("stage_samples", {}) or {}
        if not isinstance(stage_samples, dict):
            raise ValueError("timing.stage_samples must be a mapping when present")
        if stage_samples:
            if set(stage_samples) != set(stages):
                raise ValueError("stage timing diagnostics must cover every session stage")
            for stage, values in stage_samples.items():
                if (not isinstance(values, list) or len(values) != repeats
                        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                               for value in values)):
                    raise ValueError(
                        f"stage timing diagnostics for {stage} must have {repeats} positive samples")
            for index, total in enumerate(samples):
                if sum(stage_samples[stage][index] for stage in stages) > total:
                    raise ValueError("diagnostic stage times cannot exceed complete-session time")
    if status == "pass":
        quality = _mapping(result["quality"], "quality")
        if (quality.get("scope") != "trajectory"
                or int(quality.get("steps") or 0) != int(session.get("observations") or 0)):
            raise ValueError("passing quality gate must cover the exact observation trajectory")
