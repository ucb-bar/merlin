"""The numerics policy's metrics and the ratio rule (contract §4)."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from merlin.common.paths import repo_root

# ---- metrics -----------------------------------------------------------------------------------

RATIO = 1.1
FLOOR_INT8_WORST = 0.5  # LSB
FLOOR_REL = 2.0**-20
FLOOR_KL = 1e-6
#: Metrics where larger is better (agreement); every other float metric is an error, smaller is better.
AGREEMENT = frozenset({"top1_agree", "top5_agree", "cosine"})


def rel_rmse(y: np.ndarray, y_fp: np.ndarray) -> float:
    y, y_fp = np.asarray(y, np.float64), np.asarray(y_fp, np.float64)
    denom = float(np.linalg.norm(y_fp))
    return float(np.linalg.norm(y - y_fp)) / denom if denom else float(np.linalg.norm(y - y_fp))


def worst(y: np.ndarray, y_fp: np.ndarray, scale: float | None = None) -> float:
    """``max|y - y_fp|`` over the output scale (int8 tensors) or ``max|y_fp|`` (float tensors)."""
    y, y_fp = np.asarray(y, np.float64), np.asarray(y_fp, np.float64)
    s = float(scale) if scale else float(np.max(np.abs(y_fp))) or 1.0
    return float(np.max(np.abs(y - y_fp))) / s


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def kl(fp_logits: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """``KL(p_fp || p)`` per distribution (last axis is the vocabulary)."""
    lp, lq = log_softmax(fp_logits), log_softmax(logits)
    return np.sum(np.exp(lp) * (lp - lq), axis=-1)


def topk_agree(fp_logits: np.ndarray, logits: np.ndarray, k: int) -> float:
    """Fraction of rows whose FP top-1 is in the other's top-k."""
    fp, q = (
        np.asarray(fp_logits).reshape(-1, np.shape(fp_logits)[-1]),
        np.asarray(logits).reshape(-1, np.shape(logits)[-1]),
    )
    top1 = fp.argmax(axis=-1)
    topk = np.argsort(-q, axis=-1)[:, :k]
    return float(np.mean([t in row for t, row in zip(top1, topk, strict=True)]))


def model_metrics(kind: str, judged: Sequence[np.ndarray], fp: Sequence[np.ndarray], spec: dict[str, Any]) -> dict:
    """One input's metrics of ``judged`` against the FP reference, per the model's judged kind."""
    if kind == "logits":
        y, y_fp = (
            np.concatenate([np.asarray(a).reshape(1, -1) for a in judged]),
            np.concatenate([np.asarray(a).reshape(1, -1) for a in fp]),
        )
        out = {"rel_rmse": rel_rmse(y, y_fp), "worst": worst(y, y_fp)}
        for k in spec.get("topk", [1]):
            out[f"top{k}_agree"] = topk_agree(y_fp, y, int(k))
        cos = float(np.sum(y * y_fp) / (np.linalg.norm(y) * np.linalg.norm(y_fp)))
        return {**out, "cosine": cos}
    if kind == "next_token_distribution":
        y = np.concatenate([np.asarray(a).reshape(-1, np.shape(a)[-1]) for a in judged])
        y_fp = np.asarray(fp[0]).reshape(-1, np.shape(fp[0])[-1])
        if y.shape != y_fp.shape:
            raise ValueError(f"{y.shape[0]} judged distributions, {y_fp.shape[0]} FP ones")
        divergence = kl(y_fp, y)
        return {
            "kl_worst": float(divergence.max()),
            "kl_mean": float(divergence.mean()),
            "top1_agree": topk_agree(y_fp, y, 1),
            "distributions": int(y.shape[0]),
        }
    if kind == "per_channel":
        y, y_fp = np.asarray(judged[-1], np.float64), np.asarray(fp[0], np.float64)
        first, last = int(spec["channels"]["first"]), int(spec["channels"]["last"])
        rows = {}
        for channel in range(first, last + 1):
            a, b = y[..., channel], y_fp[..., channel]
            rows[str(channel)] = {"rel_rmse": rel_rmse(a, b), "worst": worst(a, b)}
        return {
            "per_channel": rows,
            "rel_rmse_worst": max(r["rel_rmse"] for r in rows.values()),
            "worst_worst": max(r["worst"] for r in rows.values()),
        }
    raise ValueError(f"unknown judged kind {kind!r}")


def ratio_rule(value: float, reference: float, floor: float, ratio: float = RATIO) -> dict[str, Any]:
    """``value <= max(ratio * reference, floor)`` with the threshold it was held to."""
    threshold = max(ratio * reference, floor)
    return {"value": value, "reference": reference, "threshold": threshold, "pass": bool(value <= threshold)}


def worst_of(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Each float metric's worst value over ``rows``: the minimum of an agreement, the maximum of an error."""
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, float)})
    return {
        key: (min if key in AGREEMENT else max)(row[key] for row in rows if isinstance(row.get(key), float))
        for key in keys
    }


def mean_of(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Each float metric's mean over ``rows`` (a per-input agreement of 0 or 1 averages to the split's rate)."""
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, float)})
    return {key: float(np.mean([row[key] for row in rows if isinstance(row.get(key), float)])) for key in keys}


AGGREGATES = {"mean": lambda values: float(np.mean(values)), "max": max, "min": min}


def aggregate(how: str, values: Sequence[float]) -> float:
    """``mean``, ``max``, ``min`` or ``p<NN>`` (the NN-th percentile, linear interpolation)."""
    if how in AGGREGATES:
        return float(AGGREGATES[how](values))
    return float(np.percentile(np.asarray(values, np.float64), float(how[1:])))


def is_aggregate(how: str) -> bool:
    return how in AGGREGATES or (how[:1] == "p" and how[1:].isdigit() and 0 < int(how[1:]) < 100)


def admission(
    rules: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], baseline: Sequence[dict[str, Any]] = ()
) -> dict:
    """Judge an INT8 reference's per-input metric ``rows`` against the models file's admission ``rules``.

    A rule names a ``metric`` of the rows, an ``aggregate`` over the rows (mean, max, min or ``p<NN>``) and one bound:
    ``max`` / ``min`` (absolute), or ``max_ratio_to_baseline`` (at most that multiple of the same aggregate of
    the ``baseline`` rows, the reference precision's own error on the same inputs). A ``report_only`` rule is
    computed and listed with status ``info`` but never decides the verdict.
    """
    checks = []
    for rule in rules:
        metric, how = str(rule["metric"]), str(rule.get("aggregate", "mean"))
        values = [float(r[metric]) for r in rows if isinstance(r.get(metric), (int, float))]
        if not values:
            checks.append({**rule, "status": "unknown", "detail": f"no {metric!r} in the rows"})
            continue
        value = aggregate(how, values)
        if "max_ratio_to_baseline" in rule:
            base = [float(r[metric]) for r in baseline if isinstance(r.get(metric), (int, float))]
            if not base:
                checks.append({**rule, "value": value, "status": "unknown", "detail": "no baseline rows"})
                continue
            bound = float(rule["max_ratio_to_baseline"]) * aggregate(how, base)
            ok = value <= bound
        elif "max" in rule:
            bound, ok = float(rule["max"]), value <= float(rule["max"])
        else:
            bound, ok = float(rule["min"]), value >= float(rule["min"])
        status = "info" if rule.get("report_only") else "pass" if ok else "fail"
        checks.append({**rule, "value": value, "bound": bound, "status": status, "within": bool(ok)})
    statuses = {c["status"] for c in checks if not c.get("report_only")}
    status = "fail" if "fail" in statuses else "unknown" if "unknown" in statuses or not statuses else "pass"
    return {"status": status, "inputs": len(rows), "checks": checks}


# ---- models ------------------------------------------------------------------------------------
# The models file: which programs make up each model, how they chain, and what is judged.
#
# Everything model-specific is data here; the code only knows chain *kinds* and metric *kinds*.

MODELS_SCHEMA = "merlin.evaluation.model.bundles.models.v1"
NUMERICS_SCHEMA = "merlin.evaluation.model.numerics.v1"
LEGACY_FLOAT_POLICY = {"atol": 2e-6, "rtol": 1e-5, "equal_nan": False, "finite_required": True}

CHAIN_KINDS = {"single", "flow", "causal"}
JUDGED_KINDS = {"logits", "next_token_distribution", "per_channel"}
BINDING_SOURCES = ("input", "stream", "tokens", "output", "constant")
# Full mode's differential per model (``full.differential``, contract §2): ``required`` runs the host twin
# on a cache miss; ``cached`` compares with the bundle's twin cache only, a miss leaving the step pending.
DIFFERENTIAL_POLICIES = ("required", "cached")


class ModelsFileError(ValueError):
    """The models file is missing, malformed, or names something it does not define."""


def validate_stream_binding(binding: dict, *, where: str = "stream") -> None:
    """Require an explicit frame axis and rank policy, without implicit reuse."""
    if (
        set(binding) != {"from", "key", "axis", "keepdims"}
        or binding.get("from") != "stream"
        or not isinstance(binding.get("key"), str)
        or not binding["key"]
        or type(binding.get("axis")) is not int
        or type(binding.get("keepdims")) is not bool
    ):
        raise ModelsFileError(f"{where}: stream needs exactly from, key, integer axis and boolean keepdims; no carry")


def selected_host_math_policy(model: Model, full: dict[str, Any] | None = None) -> str:
    """One explicit host arithmetic selection for both execution and reference derivation."""
    from merlin.runtime.host_math import host_math_recipe

    own = model.extra.get("full") or {}
    policy = own.get("host_math_policy", (full or {}).get("host_math_policy", "native"))
    host_math_recipe(policy)  # Validate before creating build or reference artifacts.
    return policy


@dataclass(frozen=True)
class Program:
    name: str
    stage: str | None  # capture sub-stage directory; None when the capture root is the program
    inputs: tuple[dict[str, Any], ...] = ()  # @forward model inputs in order (see the models file header)
    steps: int = 1


@dataclass(frozen=True)
class Model:
    name: str
    workload: str
    chain: dict[str, Any]
    programs: tuple[Program, ...]
    judged: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    def program(self, name: str) -> Program:
        for p in self.programs:
            if p.name == name:
                return p
        raise ModelsFileError(f"model {self.name!r} has no program {name!r}")


@dataclass(frozen=True)
class ModelsFile:
    path: Path
    target: str
    models: dict[str, Model]
    full: dict[str, Any] = field(default_factory=dict)  # how check --mode full builds a program's image
    numerics: dict[str, Any] | None = None  # None preserves historical configurations without this field.

    def model(self, name: str) -> Model:
        if name not in self.models:
            known = ", ".join(sorted(self.models))
            raise ModelsFileError(f"{self.path}: no model {name!r} (declared: {known})")
        return self.models[name]


def selected_float_policy(models: ModelsFile | None = None) -> dict[str, Any]:
    """Read the selected functional comparison; absent historical policy keeps legacy tolerances."""
    numerics = getattr(models, "numerics", None)
    if numerics is None:
        return dict(LEGACY_FLOAT_POLICY)
    if not isinstance(numerics, dict) or numerics.get("version") != NUMERICS_SCHEMA:
        raise ModelsFileError(f"numerics.version must be {NUMERICS_SCHEMA!r}")
    functional = numerics.get("functional")
    required = {"atol", "rtol", "equal_nan", "finite_required", "integer_boolean", "outputs"}
    if not isinstance(functional, dict) or set(functional) != required:
        raise ModelsFileError("numerics.functional requires the complete supported comparison policy")
    for name in ("atol", "rtol"):
        value = functional[name]
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ModelsFileError(f"numerics.functional.{name} must be finite and nonnegative")
    if (
        functional["equal_nan"] is not False
        or functional["finite_required"] is not True
        or functional["integer_boolean"] != "exact"
        or functional["outputs"] != "all_declared_every_step"
    ):
        raise ModelsFileError("numerics.functional requires finite outputs, exact integers/booleans and all steps")
    return {name: float(functional[name]) for name in ("atol", "rtol")} | {
        "equal_nan": False,
        "finite_required": True,
    }


def target_from_contract(contract: str | os.PathLike | None = None) -> str:
    """The target name declared by the selected target contract (``MERLIN_TARGET_CONTRACT``)."""
    source = contract or os.environ.get("MERLIN_TARGET_CONTRACT")
    if not source:
        raise ModelsFileError("no target contract selected: set MERLIN_TARGET_CONTRACT or pass --models")
    data = yaml.safe_load(Path(source).read_text()) or {}
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ModelsFileError(f"{source}: target contract declares no name")
    return name


def default_models_path(target: str) -> Path:
    return repo_root() / "examples/compiler-generation" / target / "numerics" / "models.yaml"


ADMISSION_BOUNDS = ("max", "min", "max_ratio_to_baseline")


def _check_admission(judged: dict[str, Any], where: str) -> None:
    """``admission``: rules judged when a bundle is built; ``reference.precision``: the FP reference's precision
    (``native`` = the checkpoint's own, or ``float32``); ``reference.baseline``: a precision whose own error
    against the reference ``max_ratio_to_baseline`` rules are relative to; ``reference.seeds``: judge every
    input under ``count`` draws of its standard-normal input array ``input`` (draw 0 is the stored one)."""
    for rule in judged.get("admission") or ():
        bounds = [key for key in ADMISSION_BOUNDS if key in rule]
        if "metric" not in rule or len(bounds) != 1:
            raise ModelsFileError(f"{where}.admission: each rule needs a metric and one of {ADMISSION_BOUNDS}")
        if not is_aggregate(str(rule.get("aggregate", "mean"))):
            raise ModelsFileError(f"{where}.admission: aggregate must be mean, max, min or p<NN>")
        if bounds == ["max_ratio_to_baseline"] and not (judged.get("reference") or {}).get("baseline"):
            raise ModelsFileError(f"{where}.admission: a baseline-relative rule needs reference.baseline")
    seeds = (judged.get("reference") or {}).get("seeds")
    if seeds is not None and not (isinstance(seeds.get("input"), str) and int(seeds.get("count", 0)) >= 1):
        raise ModelsFileError(f"{where}.reference.seeds needs an input key and a count >= 1")


def _require(mapping: Any, key: str, where: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise ModelsFileError(f"{where}: missing {key!r}")
    return mapping[key]


def load(path: str | os.PathLike | None = None, *, source_bytes: bytes | None = None) -> ModelsFile:
    """Load a models file, optionally parsing one already captured byte snapshot."""
    if path is None:
        path = default_models_path(target_from_contract())
    path = Path(path)
    if source_bytes is None and not path.is_file():
        raise ModelsFileError(f"models file not found: {path}")
    data = yaml.safe_load(path.read_text() if source_bytes is None else source_bytes) or {}
    if data.get("schema") != MODELS_SCHEMA:
        raise ModelsFileError(f"{path}: schema must be {MODELS_SCHEMA!r}, found {data.get('schema')!r}")
    target = _require(data, "target", str(path))
    models: dict[str, Model] = {}
    for name, entry in (_require(data, "models", str(path)) or {}).items():
        where = f"{path}: models.{name}"
        chain = dict(_require(entry, "chain", where))
        if chain.get("kind") not in CHAIN_KINDS:
            raise ModelsFileError(f"{where}.chain.kind must be one of {sorted(CHAIN_KINDS)}")
        programs = tuple(
            Program(
                name=str(_require(p, "name", f"{where}.programs")),
                stage=p.get("stage"),
                inputs=tuple(dict(i) for i in p.get("inputs") or ()),
                steps=int(p.get("steps", 1)),
            )
            for p in _require(entry, "programs", where)
        )
        names_so_far: set[str] = set()
        for program in programs:
            for binding in program.inputs:
                source = binding.get("from")
                if source not in BINDING_SOURCES:
                    raise ModelsFileError(
                        f"{where}.{program.name}: input source {source!r} is not one of {BINDING_SOURCES}"
                    )
                if source == "stream":
                    validate_stream_binding(binding, where=f"{where}.{program.name}")
                    if program.steps < 1:
                        raise ModelsFileError(f"{where}.{program.name}: stream needs positive declared steps")
                if source == "output" and binding.get("program") not in names_so_far | {program.name}:
                    raise ModelsFileError(f"{where}.{program.name}: binds an output of a later or unknown program")
            names_so_far.add(program.name)
        if not programs:
            raise ModelsFileError(f"{where}: no programs")
        judged = dict(_require(entry, "judged", where))
        if judged.get("kind") not in JUDGED_KINDS:
            raise ModelsFileError(f"{where}.judged.kind must be one of {sorted(JUDGED_KINDS)}")
        _check_admission(judged, f"{where}.judged")
        names = {p.name for p in programs}
        for ref in judged.get("outputs", []):
            if ref.get("program") not in names:
                raise ModelsFileError(f"{where}.judged names unknown program {ref.get('program')!r}")
        differential = (entry.get("full") or {}).get("differential", "required")
        if differential not in DIFFERENTIAL_POLICIES:
            raise ModelsFileError(f"{where}.full.differential must be one of {list(DIFFERENTIAL_POLICIES)}")
        known = {"workload", "chain", "programs", "judged"}
        models[name] = Model(
            name=name,
            workload=str(_require(entry, "workload", where)),
            chain=chain,
            programs=programs,
            judged=judged,
            extra={k: v for k, v in entry.items() if k not in known},
        )
    if not models:
        raise ModelsFileError(f"{path}: declares no models")
    if "numerics" in data and not isinstance(data["numerics"], dict):
        raise ModelsFileError("numerics must be a mapping when selected")
    result = ModelsFile(
        path=path, target=str(target), models=models, full=dict(data.get("full") or {}), numerics=data.get("numerics")
    )
    selected_float_policy(result)
    return result


PROTOCOL_GEOMETRY_SCHEMA = "merlin.evaluation.model.protocol_geometry.v1"


def _protocol_models_bytes(baseline: bytes, protocol: dict) -> bytes:
    """Resolve only explicitly selected chain geometry over unchanged baseline policy.

    Supplied values are the sole inputs. The returned configuration is a new
    selection, not a replacement identity for baseline captures or references.
    No host arithmetic, program bindings, accuracy rules or tolerances change.
    """
    authority = protocol.get("authority") if isinstance(protocol, dict) else None
    selection = authority.get("models_configuration") if isinstance(authority, dict) else None
    if selection != {"schema": PROTOCOL_GEOMETRY_SCHEMA, "chain_source": "protocol.models"}:
        raise ModelsFileError("unsupported explicit protocol models configuration selection")
    data = yaml.safe_load(baseline)
    if not isinstance(data, dict) or data.get("schema") != MODELS_SCHEMA:
        raise ModelsFileError("protocol geometry requires an ordinary baseline models file")
    original, selected = data.get("models"), protocol.get("models")
    if (
        not isinstance(original, dict)
        or not original
        or not isinstance(selected, dict)
        or set(original) != set(selected)
        or any(not isinstance(name, str) or not name for name in original)
    ):
        raise ModelsFileError("protocol geometry requires the exact baseline model roster")
    keys = {"single": {"kind"}, "flow": {"kind", "steps"}, "causal": {"kind", "prefill_tokens", "decode_steps"}}
    for name, entry in original.items():
        declaration = selected[name]
        before = entry.get("chain") if isinstance(entry, dict) else None
        chain = declaration.get("chain") if isinstance(declaration, dict) else None
        kind = before.get("kind") if isinstance(before, dict) else None
        if (
            not isinstance(kind, str)
            or kind not in keys
            or set(before) != keys[kind]
            or not isinstance(chain, dict)
            or chain.get("kind") != kind
            or set(chain) != keys[kind]
            or any(type(value) is not int or value <= 0 for key, value in chain.items() if key != "kind")
        ):
            raise ModelsFileError(
                f"{name}: protocol geometry requires the same chain kind and exact positive dimensions"
            )
        programs, stages = entry.get("programs"), declaration.get("stages")
        if (
            not isinstance(programs, list)
            or not isinstance(stages, dict)
            or any(
                not isinstance(program, dict)
                or not isinstance(program.get("name"), str)
                or not program["name"]
                or type(program.get("steps", 1)) is not int
                or program.get("steps", 1) <= 0
                for program in programs
            )
            or [program.get("name") for program in programs] != list(stages)
            or any(
                not isinstance(stages[program["name"]], dict)
                or type(stages[program["name"]].get("steps")) is not int
                or stages[program["name"]]["steps"] != program.get("steps", 1)
                for program in programs
            )
        ):
            raise ModelsFileError(f"{name}: protocol geometry cannot change baseline programs, order or step counts")
        entry["chain"] = dict(chain)
    resolved = yaml.safe_dump(data, sort_keys=True, allow_unicode=True).encode("utf-8")
    load(Path("<resolved-protocol-models>"), source_bytes=resolved)
    return resolved
