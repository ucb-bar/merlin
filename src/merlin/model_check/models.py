"""The models file: which programs make up each model, how they chain, and what is judged.

Everything model-specific is data here; the code only knows chain *kinds* and metric *kinds*.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import repo_root
from merlin.model_check.metrics import is_aggregate

MODELS_SCHEMA = "merlin.model_check.models.v1"

CHAIN_KINDS = {"single", "flow", "causal"}
JUDGED_KINDS = {"logits", "next_token_distribution", "per_channel"}
BINDING_SOURCES = ("input", "tokens", "output", "constant")


class ModelsFileError(ValueError):
    """The models file is missing, malformed, or names something it does not define."""


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

    def model(self, name: str) -> Model:
        if name not in self.models:
            known = ", ".join(sorted(self.models))
            raise ModelsFileError(f"{self.path}: no model {name!r} (declared: {known})")
        return self.models[name]


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
    return repo_root() / "examples" / target / "numerics" / "models.yaml"


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


def load(path: str | os.PathLike | None = None) -> ModelsFile:
    """Load and validate a models file (default: the selected target's)."""
    if path is None:
        path = default_models_path(target_from_contract())
    path = Path(path)
    if not path.is_file():
        raise ModelsFileError(f"models file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
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
    return ModelsFile(path=path, target=str(target), models=models, full=dict(data.get("full") or {}))
