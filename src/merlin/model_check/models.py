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
    return ModelsFile(path=path, target=str(target), models=models)
