"""Validate definitions before resolving any executable command.

Paths in a definition are relative to the definition, never the invoking cwd.
Adapters own configuration validation and argument construction; YAML cannot
provide a shell string, executable path, Python module, or arbitrary environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import jsonschema
import yaml


class SpecError(ValueError):
    """An experiment definition or frozen execution identity is invalid."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise SpecError(f"duplicate definition key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path: Path) -> dict:
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except (OSError, yaml.YAMLError, TypeError) as exc:
        raise SpecError(f"cannot read {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise SpecError(f"{path}: expected a mapping")
    return document


@dataclass(frozen=True)
class ExperimentSpec:
    """A validated definition with its declaring file retained for path resolution."""

    path: Path
    document: dict

    @property
    def id(self) -> str:
        return self.document["id"]

    @property
    def target(self) -> str:
        return self.document["target"]

    def resolve(self, value: str) -> Path:
        return (self.path.parent / Path(value).expanduser()).resolve()


def catalog(path: Path | None = None) -> dict[str, Path]:
    """Resolve the shared definition catalog without importing execution or CLI code."""
    from merlin.common.paths import repo_root

    source = (path or repo_root() / "experiments/catalog.yaml").resolve()
    doc = read_yaml(source)
    if set(doc) != {"schema_version", "experiments"} or doc["schema_version"] != 1:
        raise SpecError("catalog requires schema_version: 1 and experiments mapping")
    if not isinstance(doc["experiments"], dict):
        raise SpecError("catalog experiments must be a mapping of ids to definition paths")
    result = {}
    for name, value in doc["experiments"].items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise SpecError("catalog ids and definition paths must be strings")
        result[name] = (source.parent / value).resolve()
    return result


def load_spec(path: str | Path) -> ExperimentSpec:
    from .adapters import ADAPTERS

    source = Path(path).expanduser().resolve()
    document = read_yaml(source)
    phases = document.get("phases")
    if isinstance(phases, dict):
        # YAML numeric phase keys are convenient, but JSON snapshots always use strings.
        normalized = {str(key): value for key, value in phases.items()}
        if len(normalized) != len(phases):
            raise SpecError("duplicate phase after normalizing numeric keys")
        document["phases"] = normalized
    schema = json.loads(files("merlin_experiments").joinpath("resources/experiment.schema.json").read_text())
    try:
        jsonschema.Draft202012Validator(schema).validate(document)
    except jsonschema.ValidationError as exc:
        raise SpecError(f"{source}: {exc.message}") from exc
    for number, phase in document["phases"].items():
        adapter = ADAPTERS.get(phase["adapter"])
        if adapter is None or adapter.phase != number:
            raise SpecError(f"phase {number}: unknown or incompatible adapter {phase['adapter']!r}")
        if phase.get("mode") != adapter.mode:
            raise SpecError(f"phase {number}: {adapter.name} requires mode {adapter.mode!r}")
        adapter.validate(phase["config"])
    return ExperimentSpec(source, document)
