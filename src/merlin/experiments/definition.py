"""Validate definitions before resolving any executable command.

Paths in a definition are relative to the definition, never the invoking cwd.
Adapters own configuration validation and argument construction; YAML cannot
provide a shell string, executable path, Python module, or arbitrary environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

import yaml

from merlin.common.paths import GENERATION_SUITES as ENGINE_STAGES
from merlin.common.paths import generation_run_roots

STAGE_ENGINES = {"test-generation": "0", "backend-generation": "1"}


def engine_definitions(document: dict) -> dict:
    """Normalize named authored stages once; numeric IDs remain legacy engine protocol."""
    if document.get("schema_version") == 2:
        return {STAGE_ENGINES[name]: value for name, value in document["stages"].items()}
    return document.get("phases", {})


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

    @property
    def prohibited_instruction_roles(self) -> list[str]:
        """The declared roles no candidate program may emit; an empty list when none is declared."""
        return list((self.document.get("policy") or {}).get("prohibited_instruction_roles") or ())


def backend_launch_inputs(
    spec: ExperimentSpec,
    selected: list[str],
    *,
    corpus_seal: Path | None,
    joint_campaign: Path | None,
    oracle_timing: Path | None,
    driver: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    provider: str | None = None,
) -> dict[str, str]:
    """Resolve the two reviewed backend launch inputs without inventing campaign defaults."""
    named_backend = spec.document["schema_version"] == 2 and "1" in selected
    if named_backend and corpus_seal is None:
        raise SpecError(
            "backend-generation requires a fresh reviewed --corpus-seal and --bundle-manifest; "
            "run test-generation, review its release, then select --workflow backend-generation"
        )
    overrides = {
        name: value
        for name, value in (("driver", driver), ("model", model), ("effort", effort), ("provider", provider))
        if value is not None
    }
    if overrides and selected != ["1"]:
        raise SpecError("backend launch overrides require --workflow backend-generation")
    defaults = engine_definitions(spec.document).get("1", {}).get("config", {})
    for name, path in (("joint_campaign", joint_campaign), ("oracle_timing", oracle_timing)):
        flag = "--backend-" + name.replace("_", "-")
        if path is not None:
            if selected != ["1"]:
                raise SpecError("backend input overrides require --workflow backend-generation")
            selected_path = path.expanduser().absolute()
            if selected_path.is_symlink() or not selected_path.is_file():
                raise SpecError(f"{flag} must select an existing ordinary file: {selected_path}")
            overrides[name] = str(selected_path)
        elif named_backend:
            selected_path = spec.resolve(defaults[name]) if defaults.get(name) else None
            if selected_path is None or selected_path.is_symlink() or not selected_path.is_file():
                raise SpecError(f"backend-generation requires reviewed {flag} FILE (or its definition field)")
    return overrides


def selected_engines(document: dict, *, workflow: str | None = None, phase: str | None = None) -> tuple[str, list[str]]:
    """Select named work once; explicit numeric selection is the legacy Python interface."""
    if workflow is not None and phase is not None:
        raise SpecError("select workflow or the legacy phase argument, never both")
    if workflow is not None and workflow not in (*STAGE_ENGINES, "all"):
        raise SpecError(f"unknown workflow {workflow!r}; select test-generation, backend-generation or all")
    engine = phase if phase is not None else STAGE_ENGINES.get(workflow or "all", "all")
    declared = engine_definitions(document)
    selected = sorted(declared) if engine == "all" else [engine]
    if any(number not in declared for number in selected):
        raise SpecError(f"definition does not declare selected workflow {workflow or engine!r}")
    return engine, selected


def require_plan_selection(plan: dict, *, workflow: str | None = None, phase: str | None = None) -> None:
    """A run may assert its selection; admitted plan inputs are never silently rebuilt."""
    if workflow is None and phase is None:
        return
    _, selected = selected_engines(plan["spec"], workflow=workflow, phase=phase)
    if set(selected) != set(plan["phases"]):
        raise SpecError("requested workflow differs from the resolved plan; resolve that workflow first")


def default_run_dir(spec: ExperimentSpec, phase: str | None = None, *, workflow: str | None = None) -> Path:
    """Place new single-workflow runs in their named suite; keep multi-workflow runs together."""
    from merlin.common.artifacts import phase_run_id
    from merlin.common.paths import runs_dir

    _, selected = selected_engines(spec.document, workflow=workflow, phase=phase)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if len(selected) == 1:
        base = generation_run_roots(spec.target, selected[0])[0]
        candidate = base / phase_run_id(spec.id, timestamp=stamp)
        return candidate if not candidate.exists() else base / f"{candidate.name}_{uuid4().hex[:6]}"
    return runs_dir() / spec.target / spec.id / f"{stamp}_{uuid4().hex[:8]}"


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
    from merlin.experiments.adapters import ADAPTERS

    source = Path(path).expanduser().resolve()
    document = read_yaml(source)
    phases = document.get("phases")
    if isinstance(phases, dict):
        # YAML numeric phase keys are convenient, but JSON snapshots always use strings.
        normalized = {str(key): value for key, value in phases.items()}
        if len(normalized) != len(phases):
            raise SpecError("duplicate phase after normalizing numeric keys")
        document["phases"] = normalized
    schema = json.loads(files("merlin.experiments").joinpath("resources/experiment.schema.json").read_text())
    # Imported here: interpreters that only run broker workers (the model capture Python) need not
    # carry the schema validator just because they import this package.
    import jsonschema

    try:
        jsonschema.Draft202012Validator(schema).validate(document)
    except jsonschema.ValidationError as exc:
        raise SpecError(f"{source}: {exc.message}") from exc
    roles = (document.get("policy") or {}).get("prohibited_instruction_roles")
    if roles is not None:
        from merlin.analysis.instruction_roles import ROLES

        unknown = sorted(set(roles) - set(ROLES))
        if unknown:
            raise SpecError(
                f"{source}: policy.prohibited_instruction_roles names unknown role(s) {unknown}; "
                f"the closed vocabulary is {sorted(ROLES)}"
            )
    for number, phase in engine_definitions(document).items():
        adapter = ADAPTERS.get(phase["adapter"])
        if adapter is None or adapter.phase != number:
            raise SpecError(f"phase {number}: unknown or incompatible adapter {phase['adapter']!r}")
        if phase.get("mode") != adapter.mode:
            raise SpecError(f"phase {number}: {adapter.name} requires mode {adapter.mode!r}")
        adapter.validate(phase["config"])
        if document["schema_version"] == 2 and number == "1":
            config = phase["config"]
            if not config.get("joint_campaign"):
                raise SpecError(
                    "backend-generation requires joint_campaign: POLICY_JSON for the combined "
                    "correctness/performance engine; use a reviewed policy and select fresh_backend "
                    "or its reviewed authored checkpoint"
                )
            if config.get("schedule", "continuous") != "continuous":
                raise SpecError("backend-generation requires the native continuous schedule")
    return ExperimentSpec(source, document)


# Existing callers and persisted objects retain their original import identities.
# The compatibility module is the same module object, so legacy global patches
# still affect these actual implementations. Source custody pins this file too.
for _definition in (
    SpecError,
    _UniqueLoader,
    _mapping,
    read_yaml,
    ExperimentSpec,
    default_run_dir,
    catalog,
    load_spec,
):
    _definition.__module__ = "merlin.experiments.spec"
for _member in vars(ExperimentSpec).values():
    _function = _member.fget if isinstance(_member, property) else _member
    if callable(_function) and getattr(_function, "__module__", None) == __name__:
        _function.__module__ = "merlin.experiments.spec"
del _definition, _member, _function
