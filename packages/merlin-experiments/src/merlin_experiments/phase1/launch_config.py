"""One checked-in operator configuration per Phase 1 launch: render its inputs and verify drift.

A launch config (``merlin.phase1.launch_config.v1``) and the reviewed Phase 0 release config it
names (``merlin.phase0.release_config.v1``) are the single source of one launch's controller
argv, complete process environment, CPU policy, native runtime pins, target selection, numerics
lineage, instruction selection, operator errata and exclusions. Machine roots are
``${VARIABLES}``: the built-ins ``TASK_ROOT``, ``PREP`` and ``RUN_ID`` come from the command
line, ``HOME`` from the invoking environment, and ``--var NAME=VALUE`` overrides a declared
default (for example an earlier release version). The target is configuration data; nothing
here selects or assumes one.

``render`` writes the operator inputs: the environment script, ``launch-argv.json`` (argv,
affinity, supervisor environment and pins), ``release-prepare.json`` (the exact
``merlin_experiments.corpus.release.prepare`` arguments), a byte-identical errata copy and, from
``--plan-template``, a launch plan. ``verify`` compares prepared inputs -- and optionally a live
controller's ``/proc`` command line, environment and affinity -- with the config, and lists every
drift: a missing, extra or forbidden variable, a wrong flag value, a wrong hash, a broken numerics
lineage or a reachable exclusion. Neither starts a run, reads a credential, nor writes below the
inputs it checks.

    python -m merlin_experiments.phase1.launch_config render --config C --task-root T --prep P [--run-id R] [--out D]
    python -m merlin_experiments.phase1.launch_config verify --config C --task-root T --prep P [--controller-pid N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import jsonschema
import yaml

from merlin.common.digest import sha256_bytes, sha256_file
from merlin_experiments.spec import SpecError, read_yaml

LAUNCH_SCHEMA = "merlin.phase1.launch_config.v1"
RELEASE_SCHEMA = "merlin.phase0.release_config.v1"
ARGV_SCHEMA = "merlin.phase1.launch_argv.v1"
PREPARE_SCHEMA = "merlin.phase0.release_prepare_call.v1"
REPORT_SCHEMA = "merlin.phase1.launch_config_report.v1"
PREPARE_FUNCTION = "merlin_experiments.corpus.release.prepare"

#: Variables supplied by the invocation's own options, never by a config default or ``--var``.
BUILTINS = ("TASK_ROOT", "PREP", "RUN_ID")

#: The installed controller's option for a driver's concurrent-subagent cap. A driver without an
#: entry refuses ``author.max_subagents`` instead of silently dropping it.
#: TODO(author-subagent-cap): when a driver-neutral author subagent option lands in
#: ``merlin_experiments.phase1.options``, bind ``author.max_subagents`` to it here (one place).
SUBAGENT_FLAGS = {"codex": "--codex-max-subagents"}

#: Explicit client-runtime selections of a driver's ``author.<driver>`` section, in argv order.
CLIENT_FLAGS = {
    "codex": (
        ("binary", "--codex-binary"),
        ("auth_source", "--codex-auth-source"),
        ("home_root", "--codex-home-root"),
    )
}

#: Characters a rendered shell value may carry unquoted.
_SHELL_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./,:=+@%")
_NAME_START = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_NAME_REST = _NAME_START | frozenset("0123456789")


class ConfigError(ValueError):
    """A launch or release config is invalid, inconsistent or cannot be resolved."""


@dataclass(frozen=True)
class Prepend:
    """A search-path variable whose value starts with ``prefix`` and keeps the inherited value."""

    prefix: str

    def accepts(self, value: str) -> bool:
        return value == self.prefix or value.startswith(self.prefix + ":")

    def __str__(self) -> str:
        return f"{self.prefix}:<inherited>"


@dataclass(frozen=True)
class Drift:
    kind: str
    subject: str
    expected: object = None
    actual: object = None

    def line(self) -> str:
        return f"DRIFT {self.kind:<15} {self.subject}: expected {_show(self.expected)}, found {_show(self.actual)}"

    def record(self) -> dict:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "expected": _plain(self.expected),
            "actual": _plain(self.actual),
        }


@dataclass(frozen=True)
class LaunchConfig:
    path: Path
    sha256: str
    document: dict
    release_path: Path | None = None
    release_sha256: str | None = None
    release: dict | None = None


@dataclass(frozen=True)
class Launch:
    """One config resolved against concrete roots: everything the controller must receive."""

    config: LaunchConfig
    variables: dict[str, str]
    overrides: dict[str, str]
    run_id: str
    argv: tuple[str, ...]
    environment: dict[str, str | Prepend]
    launch_environment: dict[str, str]
    pins: dict[str, tuple[str, str]]
    manager_cpus: tuple[int, ...]

    @property
    def document(self) -> dict:
        return self.config.document

    def expand(self, text: str) -> str:
        return _substitute(text, self.variables)

    def path(self, text: str) -> Path:
        return Path(self.expand(text))

    @property
    def placeholder(self) -> str:
        return self.document["launcher"]["run_id_placeholder"]


@dataclass
class Report:
    checked: list[str] = field(default_factory=list)
    drifts: list[Drift] = field(default_factory=list)

    def ok(self, subject: str) -> None:
        self.checked.append(subject)

    def drift(self, kind: str, subject: str, expected: object = None, actual: object = None) -> None:
        self.drifts.append(Drift(kind, subject, expected, actual))

    def expect(self, subject: str, expected: object, actual: object, kind: str = "value") -> bool:
        if expected == actual:
            self.ok(subject)
            return True
        self.drift(kind, subject, expected, actual)
        return False


# --------------------------------------------------------------------------- loading


def _schema(name: str) -> dict:
    return json.loads(files("merlin_experiments").joinpath(f"resources/{name}").read_text(encoding="utf-8"))


def _read(path: Path, schema: str) -> dict:
    try:
        document = read_yaml(path)
    except SpecError as exc:
        raise ConfigError(str(exc)) from exc
    try:
        jsonschema.Draft202012Validator(_schema(schema)).validate(document)
    except jsonschema.ValidationError as exc:
        where = "/".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ConfigError(f"{path}: {where}: {exc.message}") from exc
    return document


def load(path: str | Path) -> LaunchConfig:
    """Read and validate a launch config and the release config it names; refuse inconsistency."""
    source = Path(os.path.abspath(Path(path).expanduser()))
    document = _read(source, "phase1_launch_config.schema.json")
    release_path = release = release_sha = None
    if document.get("release_config"):
        release_path = Path(os.path.normpath(source.parent / document["release_config"]))
        release = _read(release_path, "phase0_release_config.schema.json")
        release_sha = sha256_file(release_path)
        if release["target"] != document["identity"]["target"]:
            raise ConfigError(
                f"release config target {release['target']!r} differs from launch target "
                f"{document['identity']['target']!r}"
            )
    _check_identity(document)
    _check_author(document)
    for name in ("atol", "rtol"):
        value = document.get("full_model_verification", {}).get(name, 0)
        if not math.isfinite(value):
            raise ConfigError(f"full_model_verification.{name} must be finite")
    _environment_names(document)
    errata = document.get("errata")
    if errata:
        bytes_ = (source.parent / errata["source"]).read_bytes()
        if sha256_bytes(bytes_) != errata["sha256"]:
            raise ConfigError(f"errata source {errata['source']} does not match errata.sha256")
    return LaunchConfig(source, sha256_file(source), document, release_path, release_sha, release)


def _check_identity(document: dict) -> None:
    from .levels import level_for_phase1, materialize_level

    identity = document["identity"]
    selection = {"level": identity["level"], "arm": identity["arm"], "treatment": identity["treatment"]}
    try:
        materialize_level(selection)
    except ValueError as exc:
        raise ConfigError(f"identity: {exc}") from exc
    labeled = level_for_phase1({**selection, "bundle": identity["bundle"]})
    if labeled != identity["level"]:
        raise ConfigError(
            f"identity: bundle {identity['bundle']!r} with this arm/treatment is {labeled!r}, not {identity['level']!r}"
        )


def _check_author(document: dict) -> None:
    author = document["author"]
    driver = author["driver"]
    if "max_subagents" in author and driver not in SUBAGENT_FLAGS:
        raise ConfigError(f"author.max_subagents has no controller option for driver {driver!r}")
    for section in CLIENT_FLAGS:
        if section in author and section != driver:
            raise ConfigError(f"author.{section} selects a client for driver {section!r}, not {driver!r}")


def _environment_names(document: dict) -> list[str]:
    names: list[str] = []
    for entries in document["environment"].values():
        names += list(entries)
    for key in ("author", "compute_pool"):
        if document["cpu"].get(key):
            names.append(document["cpu"][key]["env"])
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"environment variables declared more than once: {duplicates}")
    policy = document.get("environment_policy") or {}
    forbidden = set(policy.get("forbidden") or ()) & (set(names) | set(document.get("launch_environment") or ()))
    if forbidden:
        raise ConfigError(f"forbidden variables are also declared: {sorted(forbidden)}")
    return names


# --------------------------------------------------------------------------- variables


def _is_name(text: str) -> bool:
    return bool(text) and text[0] in _NAME_START and all(char in _NAME_REST for char in text[1:])


def _expand(text: str, lookup: Callable[[str], str]) -> str:
    pieces, cursor = [], 0
    while (start := text.find("${", cursor)) >= 0:
        end = text.find("}", start + 2)
        name = text[start + 2 : end] if end >= 0 else ""
        if not _is_name(name):
            raise ConfigError(f"malformed variable reference in {text!r}")
        pieces += [text[cursor:start], lookup(name)]
        cursor = end + 1
    pieces.append(text[cursor:])
    return "".join(pieces)


def _substitute(text: str, variables: Mapping[str, str]) -> str:
    def lookup(name: str) -> str:
        if name not in variables:
            raise ConfigError(f"undefined variable ${{{name}}} in {text!r}")
        return variables[name]

    return _expand(text, lookup)


def resolve_variables(
    config: LaunchConfig,
    *,
    task_root: str | Path,
    prep: str | Path,
    run_id: str | None = None,
    overrides: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Every declared variable expanded; absolute results are path-normalized."""
    environ = os.environ if environ is None else environ
    overrides = dict(overrides or {})
    builtin = sorted(set(overrides) & set(BUILTINS))
    if builtin:
        raise ConfigError(f"{builtin} are set by their own options, not --var")
    declared: dict[str, str] = {}
    for document in (config.release, config.document):
        for name, value in ((document or {}).get("variables") or {}).items():
            if name in BUILTINS:
                raise ConfigError(f"config may not redefine built-in variable {name}")
            if declared.get(name, value) != value:
                raise ConfigError(f"variable {name} is declared differently by the launch and release configs")
            declared[name] = value
    unknown = sorted(set(overrides) - set(declared) - {"HOME"})
    if unknown:
        raise ConfigError(f"--var names undeclared variables: {unknown}")
    raw = {**declared, **overrides}
    if "HOME" not in raw and environ.get("HOME"):
        raw["HOME"] = environ["HOME"]
    resolved = {
        "TASK_ROOT": os.path.normpath(os.path.abspath(task_root)),
        "PREP": os.path.normpath(os.path.abspath(prep)),
        "RUN_ID": run_id or config.document["launcher"]["run_id_placeholder"],
    }
    active: list[str] = []

    def lookup(name: str) -> str:
        if name in resolved:
            return resolved[name]
        if name not in raw:
            raise ConfigError(f"undefined variable ${{{name}}}")
        if name in active:
            raise ConfigError(f"variable cycle: {' -> '.join([*active, name])}")
        active.append(name)
        value = _expand(raw[name], lookup)
        active.pop()
        resolved[name] = os.path.normpath(value) if value.startswith("/") else value
        return resolved[name]

    for name in raw:
        lookup(name)
    return resolved


# --------------------------------------------------------------------------- resolution


def build(
    config: LaunchConfig,
    *,
    task_root: str | Path,
    prep: str | Path,
    run_id: str | None = None,
    overrides: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Launch:
    """Resolve a loaded config against concrete roots without touching any of the named files."""
    variables = resolve_variables(
        config, task_root=task_root, prep=prep, run_id=run_id, overrides=overrides, environ=environ
    )

    def x(text: str) -> str:
        return _substitute(text, variables)

    document = config.document
    environment: dict[str, str | Prepend] = {}
    for entries in document["environment"].values():
        for name, value in entries.items():
            environment[name] = Prepend(x(value["prepend"])) if isinstance(value, dict) else x(value)
    for key in ("author", "compute_pool"):
        roster = document["cpu"].get(key)
        if roster:
            environment[roster["env"]] = ",".join(str(cpu) for cpu in roster["cpus"])
    return Launch(
        config=config,
        variables=variables,
        overrides=dict(overrides or {}),
        run_id=variables["RUN_ID"],
        argv=_argv(document, x, variables["RUN_ID"]),
        environment=environment,
        launch_environment={name: x(value) for name, value in (document.get("launch_environment") or {}).items()},
        pins=_pins(config, x),
        manager_cpus=tuple(document["cpu"]["manager"]),
    )


def _argv(document: dict, x: Callable[[str], str], run_id: str) -> tuple[str, ...]:
    identity, inputs, author = document["identity"], document["inputs"], document["author"]
    budgets, origin, cpu = document["budgets"], document["origin"], document["cpu"]
    driver = author["driver"]
    pairs = [
        ("--run-id", run_id),
        ("--descriptor", x(inputs["descriptor"])),
        ("--repo", x(inputs["repo"])),
        ("--level", identity["level"]),
        ("--bundle", identity["bundle"]),
        ("--bundle-manifest", x(inputs["bundle_manifest"])),
        ("--corpus-seal", x(inputs["corpus_seal"])),
        ("--oracle-timing", x(inputs["oracle_timing"]["path"])),
    ]
    if document.get("validation_scope", "full") != "full":
        pairs.append(("--validation-scope", document["validation_scope"]))
    if "private_full_model_spec" in inputs:
        pairs.append(("--private-full-model-spec", x(inputs["private_full_model_spec"])))
    pairs += [("--driver", driver), ("--model", x(author["model"])), ("--effort", author["effort"])]
    client = author.get(driver) or {}
    pairs += [(flag, x(client[key])) for key, flag in CLIENT_FLAGS.get(driver, ()) if key in client]
    if "max_subagents" in author:
        pairs.append((SUBAGENT_FLAGS[driver], str(author["max_subagents"])))
    pairs += [
        ("--sandbox", author["sandbox"]),
        ("--experiment", identity["experiment"]),
        ("--schedule", identity["schedule"]),
        ("--max-wall-s", str(budgets["authoring_s"])),
        ("--round-timeout", str(budgets["authoring_s"])),
        ("--grade-interval", str(budgets["grade_interval_s"])),
    ]
    for key, flag in (("qa_timeout_s", "--qa-timeout"), ("model_budget_s", "--model-budget-s")):
        if key in budgets:
            pairs.append((flag, str(budgets[key])))
    jobs = len(cpu["compute_pool"]["cpus"]) if cpu["sim_max_jobs"] == "pool" else cpu["sim_max_jobs"]
    pairs.append(("--sim-max-jobs", str(jobs)))
    if "sim_per_capsule_timeout_s" in budgets:
        pairs.append(("--sim-per-capsule-timeout", str(budgets["sim_per_capsule_timeout_s"])))
    if document.get("errata"):
        pairs.append(("--operator-errata", x(document["errata"]["path"])))
    if document.get("instruction_selection"):
        pairs.append(("--instruction-selection", x(document["instruction_selection"]["path"])))
    if origin["seed_submission"] is not None:
        pairs.append(("--seed-submission", x(origin["seed_submission"])))
    launcher = document["launcher"]
    argv = [x(launcher["python"]), "-m", launcher["module"]]
    for flag, value in pairs:
        argv += [flag, value]
    if origin["resume"]:
        argv.append("--resume")
    if author.get("live_conformance_feedback", False):
        argv.append("--live-conformance-feedback")
    return tuple(argv)


def _pins(config: LaunchConfig, x: Callable[[str], str]) -> dict[str, tuple[str, str]]:
    document, pins = config.document, {}
    for section in ("runtime_pins", "target_selection"):
        for name, pin in (document.get(section) or {}).items():
            pins[f"{section}.{name}"] = (x(pin["path"]), pin["sha256"])
    pin = document["inputs"]["oracle_timing"]
    pins["inputs.oracle_timing"] = (x(pin["path"]), pin["sha256"])
    selection = document.get("instruction_selection")
    if selection:
        pins["instruction_selection"] = (x(selection["path"]), selection["sha256"])
        pins["instruction_selection.approval"] = (x(selection["approval"]["path"]), selection["approval"]["sha256"])
    if document.get("errata"):
        pins["errata"] = (x(document["errata"]["path"]), document["errata"]["sha256"])
    release = config.release or {}
    for key in ("kernel_timing_selection", "lean_toolchain_manifest", "phase1_policy_descriptor"):
        pin = (release.get("prepare") or {}).get(key)
        if pin:
            pins[f"release.prepare.{key}"] = (x(pin["path"]), pin["sha256"])
    pin = (release.get("seal") or {}).get("post_experiment_descriptor")
    if pin:
        pins["release.seal.post_experiment_descriptor"] = (x(pin["path"]), pin["sha256"])
    return pins


# --------------------------------------------------------------------------- rendering


def _shell_value(name: str, value: str | Prepend) -> str:
    if isinstance(value, Prepend):
        if any(char in value.prefix for char in '"$`\\'):
            raise ConfigError(f"{name}: a search-path prefix cannot contain shell metacharacters")
        return f'"{value.prefix}${{{name}:+:${name}}}"'
    if value and all(char in _SHELL_SAFE for char in value):
        return value
    return shlex.quote(value)


def environment_script(launch: Launch) -> str:
    """The environment file a supervisor sources before it starts the controller."""
    config = launch.config
    lines = [
        "# Rendered by merlin_experiments.phase1.launch_config; change the config, not this file.",
        f"# config {config.path} sha256 {config.sha256}",
    ]
    if config.release_path:
        lines.append(f"# release config {config.release_path} sha256 {config.release_sha256}")
    lines += [f"export {name}={_shell_value(name, value)}" for name, value in launch.environment.items()]
    return "\n".join(lines) + "\n"


def parse_environment_script(text: str) -> tuple[dict[str, str | Prepend], list[str]]:
    """Exported values of a rendered (or hand-written) environment file, and every unparsed line."""
    values: dict[str, str | Prepend] = {}
    problems: list[str] = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        name, equals, value = rest.strip().partition("=")
        if head != "export" or not equals or not _is_name(name):
            problems.append(f"line {number}: {raw!r}")
            continue
        try:
            words = shlex.split(value, comments=True)
        except ValueError:
            words = []
        if len(words) != 1:
            problems.append(f"line {number}: {raw!r}")
            continue
        if name in values:
            problems.append(f"line {number}: {name} exported again")
        values[name] = _search_path(name, words[0]) if not value.startswith("'") else words[0]
    return values, problems


def _search_path(name: str, word: str) -> str | Prepend:
    for suffix in (f"${{{name}:+:${name}}}", f":${name}", f":${{{name}}}"):
        if word.endswith(suffix) and len(word) > len(suffix):
            return Prepend(word[: -len(suffix)])
    return word


def launch_record(launch: Launch) -> dict:
    config, document = launch.config, launch.document
    return {
        "schema": ARGV_SCHEMA,
        "id": document["id"],
        "config": {"path": str(config.path), "sha256": config.sha256},
        "release_config": (
            {"path": str(config.release_path), "sha256": config.release_sha256} if config.release_path else None
        ),
        "variables": launch.variables,
        "overrides": launch.overrides,
        "run_id": launch.run_id,
        "cwd": launch.expand(document["launcher"]["cwd"]),
        "manager_cpus": list(launch.manager_cpus),
        "argv": list(launch.argv),
        "environment_file": launch.expand(document["launcher"]["environment_file"]),
        "launch_environment": launch.launch_environment,
        "runtime_pins": {path: sha for label, (path, sha) in launch.pins.items() if label.startswith("runtime_pins.")},
        "budgets": dict(document["budgets"]),
        "origin": dict(document["origin"]),
    }


def release_prepare_call(launch: Launch) -> dict | None:
    """The exact ``release.prepare`` arguments the release config selects, as JSON data."""
    release = launch.config.release
    if not release:
        return None
    prepare, seal = release["prepare"], release["seal"]
    kwargs: dict[str, object] = {"generated_only": prepare["generated_only"]}
    if "capsule_scope" in prepare:
        kwargs["capsule_scope"] = prepare["capsule_scope"]
    for key in ("selected_levels", "selected_variants", "optional_levels"):
        if key in prepare:
            kwargs[key] = list(prepare[key])
    for key in ("lean_toolchain_manifest", "phase1_policy_descriptor"):
        if key in prepare:
            kwargs[key] = launch.expand(prepare[key]["path"])
    if "kernel_timing_selection" in prepare:
        pin = prepare["kernel_timing_selection"]
        kwargs["kernel_timing_selection"] = [launch.expand(pin["path"]), pin["sha256"]]
    if "host_index_bits" in prepare:
        kwargs["host_index_bits"] = prepare["host_index_bits"]
    return {
        "schema": PREPARE_SCHEMA,
        "function": PREPARE_FUNCTION,
        "config": {"path": str(launch.config.release_path), "sha256": launch.config.release_sha256},
        "run_dir": launch.expand(prepare["run_dir"]),
        "output": launch.expand(prepare["output"]),
        "kwargs": kwargs,
        "expect": dict(release.get("expect") or {}),
        "seal": {
            "descriptor": launch.expand(seal["descriptor"]),
            "seal": launch.expand(seal["seal"]),
            "reviewed_by": seal["reviewed_by"],
            "review_note": seal["review_note"],
        },
    }


def prepare_arguments(call: Mapping) -> tuple[tuple, dict]:
    """``(args, kwargs)`` for ``release.prepare(*args, **kwargs)`` from a rendered call record."""
    if call.get("schema") != PREPARE_SCHEMA or call.get("function") != PREPARE_FUNCTION:
        raise ConfigError("not a rendered release preparation call")
    kwargs = dict(call["kwargs"])
    for key in ("selected_levels", "selected_variants", "optional_levels"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    for key in ("lean_toolchain_manifest", "phase1_policy_descriptor"):
        if key in kwargs:
            kwargs[key] = Path(kwargs[key])
    if "kernel_timing_selection" in kwargs:
        path, digest = kwargs["kernel_timing_selection"]
        kwargs["kernel_timing_selection"] = (Path(path), digest)
    return (Path(call["run_dir"]), Path(call["output"])), kwargs


def launch_plan(launch: Launch, template: Mapping) -> dict:
    """A launch plan from an operator template with every config-owned field replaced."""
    document = launch.document
    origin = document["origin"]
    plan = dict(template)
    plan.update(
        run_id=launch.run_id,
        environment_file=launch.expand(document["launcher"]["environment_file"]),
        manager_affinity=list(launch.manager_cpus),
        argv=list(launch.argv),
        seed_submission=None if origin["seed_submission"] is None else launch.expand(origin["seed_submission"]),
        resume=origin["resume"],
        initial_submission=origin["initial_submission"],
        proposed_authoring_budget_s=document["budgets"]["authoring_s"],
        selected_runtime_hashes=launch_record(launch)["runtime_pins"],
        launch_config={"path": str(launch.config.path), "sha256": launch.config.sha256},
    )
    return plan


def render(
    launch: Launch, out: str | Path, *, plan_template: str | Path | None = None, force: bool = False
) -> dict[str, str]:
    """Write the operator inputs under ``out``; refuse to replace different existing bytes."""
    out = Path(out)
    document = launch.document
    products: dict[str, bytes] = {
        launch.path(document["launcher"]["environment_file"]).name: environment_script(launch).encode(),
        "launch-argv.json": _json(launch_record(launch)),
    }
    call = release_prepare_call(launch)
    if call:
        products["release-prepare.json"] = _json(call)
    errata = document.get("errata")
    if errata:
        products[launch.path(errata["path"]).name] = (launch.config.path.parent / errata["source"]).read_bytes()
    if plan_template is not None:
        template = json.loads(Path(plan_template).read_text(encoding="utf-8"))
        products[launch.path(document["launcher"]["plan"]).name] = _json(launch_plan(launch, template))
    changed = sorted(
        name for name, data in products.items() if (out / name).exists() and (out / name).read_bytes() != data
    )
    if changed and not force:
        raise ConfigError(f"refusing to replace different existing files in {out}: {changed}; pass --force")
    out.mkdir(parents=True, exist_ok=True)
    for name, data in products.items():
        target = out / name
        staging = target.with_name(f".{name}.render")
        staging.write_bytes(data)
        staging.chmod(0o600)
        staging.replace(target)
    return {name: sha256_bytes(data) for name, data in products.items()}


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


# --------------------------------------------------------------------------- verification


def _show(value: object) -> str:
    if value is None:
        return "<absent>"
    return repr(str(value)) if isinstance(value, Prepend) else repr(value)


def _plain(value: object) -> object:
    return {"prepend": value.prefix} if isinstance(value, Prepend) else value


def _load_json(path: Path, report: Report, subject: str) -> dict | None:
    if not path.is_file():
        report.drift("missing_file", subject, str(path), None)
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        report.drift("unreadable", subject, "JSON", f"{type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, dict):
        report.drift("unreadable", subject, "a JSON object", type(value).__name__)
        return None
    return value


def _load_yaml(path: Path, report: Report, subject: str) -> dict | None:
    if not path.is_file():
        report.drift("missing_file", subject, str(path), None)
        return None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        report.drift("unreadable", subject, "YAML", f"{type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, dict):
        report.drift("unreadable", subject, "a YAML mapping", type(value).__name__)
        return None
    return value


def _dig(value: object, *keys: str) -> object:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _same_path(a: object, b: object) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return a == b
    return os.path.normpath(a) == os.path.normpath(b) or os.path.realpath(a) == os.path.realpath(b)


def _within(path: str | Path, root: str | Path) -> bool:
    return Path(os.path.normpath(path)).is_relative_to(Path(os.path.normpath(root)))


def split_argv(argv: Iterable[str]) -> tuple[list[str], dict[str, list[str | None]], list[str]]:
    """``(interpreter, -m, module)`` head, the ``--flag value`` occurrences, and stray words."""
    argv = list(argv)
    flags: dict[str, list[str | None]] = {}
    strays: list[str] = []
    index = 3
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            strays.append(token)
            index += 1
            continue
        name, equals, inline = token.partition("=")
        if equals:
            value, index = inline, index + 1
        elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            value, index = argv[index + 1], index + 2
        else:
            value, index = None, index + 1
        flags.setdefault(name, []).append(value)
    return argv[:3], flags, strays


def _compare_argv(launch: Launch, actual: list[str], subject: str, report: Report, *, template: bool) -> None:
    head, expected, _ = split_argv(launch.argv)
    found_head, found, strays = split_argv(actual)
    report.expect(f"{subject} interpreter/module", " ".join(head), " ".join(found_head), "argv_head")
    for word in strays:
        report.drift("extra_argument", subject, None, word)
    for flag, values in found.items():
        if len(values) > 1:
            report.drift("duplicate_flag", f"{subject} {flag}", 1, len(values))
    for flag, values in expected.items():
        if flag not in found:
            report.drift("missing_flag", f"{subject} {flag}", values[0], None)
        elif template and flag == "--run-id" and found[flag][-1] == launch.placeholder:
            report.ok(f"{subject} {flag} (template)")
        else:
            report.expect(f"{subject} {flag}", values[0], found[flag][-1], "flag_value")
    for flag in found:
        if flag not in expected:
            report.drift("extra_flag", f"{subject} {flag}", None, found[flag][-1])


def _compare_environment(
    expected: Mapping[str, str | Prepend],
    actual: Mapping[str, str | Prepend],
    subject: str,
    report: Report,
    policy: Mapping,
    *,
    live: bool,
) -> None:
    """A file must say exactly the config; a live process may also inherit unowned variables.

    In a live environment a search path matches when it starts with the configured prefix;
    in a file it must keep the inherited value (``{prepend: ...}``), not replace it.
    """
    forbidden = set(policy.get("forbidden") or ())
    owned = tuple(policy.get("owned_prefixes") or ())
    for name, want in expected.items():
        if name not in actual:
            report.drift("missing_env", f"{subject} {name}", want, None)
            continue
        found = actual[name]
        matches = want.accepts(found) if live and isinstance(want, Prepend) else want == found
        if matches:
            report.ok(f"{subject} {name}")
        else:
            report.drift("env_value", f"{subject} {name}", want, found)
    for name, found in actual.items():
        if name in expected:
            continue
        if name in forbidden:
            report.drift("forbidden_env", f"{subject} {name}", "unset", found)
        elif not live or (owned and name.startswith(owned)):
            report.drift("extra_env", f"{subject} {name}", None, found)


def _check_pins(launch: Launch, report: Report) -> None:
    for label, (path, digest) in launch.pins.items():
        target = Path(path)
        if not target.is_file():
            report.drift("missing_file", label, path, None)
        else:
            report.expect(f"{label} sha256", digest, sha256_file(target), "hash")
    selection = launch.document.get("instruction_selection")
    if selection:
        approval = _load_json(launch.path(selection["approval"]["path"]), Report(), "approval")
        if approval is not None:
            report.expect(
                "instruction_selection.approval names sha256",
                selection["sha256"],
                approval.get("selection_sha256"),
                "lineage",
            )
            if _same_path(launch.expand(selection["path"]), approval.get("selection")):
                report.ok("instruction_selection.approval names path")
            else:
                report.drift(
                    "lineage",
                    "instruction_selection.approval names path",
                    launch.expand(selection["path"]),
                    approval.get("selection"),
                )


def _check_environment_file(launch: Launch, path: Path, report: Report) -> None:
    if not path.is_file():
        report.drift("missing_file", "environment file", str(path), None)
        return
    actual, problems = parse_environment_script(path.read_text(encoding="utf-8"))
    for problem in problems:
        report.drift("env_syntax", f"environment file {path.name}", "export NAME=VALUE", problem)
    policy = launch.document.get("environment_policy") or {}
    _compare_environment(launch.environment, actual, "environment file", report, policy, live=False)


def _check_plan(launch: Launch, path: Path, report: Report) -> dict | None:
    plan = _load_json(path, report, "launch plan")
    if plan is None:
        return None
    _compare_argv(launch, list(plan.get("argv") or []), "plan argv", report, template=True)
    document = launch.document
    origin = document["origin"]
    seed = None if origin["seed_submission"] is None else launch.expand(origin["seed_submission"])
    for key, want in (
        ("environment_file", launch.expand(document["launcher"]["environment_file"])),
        ("manager_affinity", list(launch.manager_cpus)),
        ("seed_submission", seed),
        ("resume", origin["resume"]),
        ("initial_submission", origin["initial_submission"]),
        ("proposed_authoring_budget_s", document["budgets"]["authoring_s"]),
    ):
        report.expect(f"plan {key}", want, plan.get(key, "<absent>"), "plan_field")
    if "selected_runtime_hashes" in plan:
        report.expect(
            "plan selected_runtime_hashes",
            launch_record(launch)["runtime_pins"],
            plan["selected_runtime_hashes"],
            "hash",
        )
    return plan


def _check_numerics(launch: Launch, plan: dict | None, report: Report) -> None:
    numerics = launch.document.get("numerics")
    if not numerics:
        return
    approved = numerics["approved"]
    bundle = launch.path(approved["bundle"])
    manifest = bundle / "MANIFEST.json"
    if manifest.is_file():
        report.expect(
            f"numerics {approved['id']} manifest sha256", approved["manifest_sha256"], sha256_file(manifest), "hash"
        )
    else:
        report.drift("missing_file", f"numerics {approved['id']} manifest", str(manifest), None)
    signoff = approved["signoff"]
    source = launch.path(signoff["source"])
    if not source.is_file():
        report.drift("missing_file", "numerics sign-off source", str(source), None)
    else:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        needles = (signoff["marker"], approved["manifest_sha256"], signoff["verdict"])
        if any(all(needle in line for needle in needles) for line in lines):
            report.ok("numerics sign-off")
        else:
            report.drift("lineage", "numerics sign-off", " + ".join(needles), "no line with all three")
    launch_bundle = launch.expand(numerics["launch_bundle"])
    launch_manifest = Path(launch_bundle) / "MANIFEST.json"
    launch_sha = sha256_file(launch_manifest) if launch_manifest.is_file() else None
    if launch_sha is None:
        report.drift("missing_file", "numerics launch bundle manifest", str(launch_manifest), None)
    elif plan is not None:
        report.expect("plan manifest_sha256", launch_sha, plan.get("manifest_sha256"), "lineage")
    if numerics.get("reuse_proof"):
        proof = _load_json(launch.path(numerics["reuse_proof"]), report, "numerics reuse proof")
        if proof is not None:
            if _same_path(str(bundle), proof.get("parent_bundle")):
                report.ok("reuse proof parent bundle")
            else:
                report.drift("lineage", "reuse proof parent bundle", str(bundle), proof.get("parent_bundle"))
            report.expect(
                "reuse proof parent manifest sha256",
                approved["manifest_sha256"],
                proof.get("parent_manifest_sha256"),
                "lineage",
            )
            report.expect("reuse proof changed original files", [], proof.get("changed_original_files"), "lineage")
    spec = launch.document["inputs"].get("private_full_model_spec")
    models = _load_yaml(launch.path(spec), report, "private full-model spec") if spec else None
    if models is not None:
        named = _dig(models, "numerics_bundle", "path")
        _expect_path(report, "private full-model spec numerics_bundle.path", launch_bundle, named)
    if numerics.get("finalization_report"):
        status = _load_json(launch.path(numerics["finalization_report"]), report, "finalization report")
        if status is not None:
            _expect_path(report, "finalization report bundle", launch_bundle, status.get("bundle"))
            if launch.config.release:
                release = launch.expand(launch.config.release["prepare"]["output"])
                _expect_path(report, "finalization report release", release, status.get("release"))
            if launch_sha:
                report.expect(
                    "finalization report manifest_sha256", launch_sha, status.get("manifest_sha256"), "lineage"
                )
    if numerics.get("verify_report"):
        verified = _load_json(launch.path(numerics["verify_report"]), report, "bundle verify report")
        if verified is not None:
            report.expect("bundle verify status", "pass", verified.get("status"), "lineage")
            if launch_sha:
                report.expect(
                    "bundle verify manifest_sha256",
                    launch_sha,
                    _dig(verified, "bundle", "verify", "manifest_sha256"),
                    "lineage",
                )


def _expect_path(report: Report, subject: str, expected: str, actual: object) -> None:
    if _same_path(expected, actual):
        report.ok(subject)
    else:
        report.drift("lineage", subject, expected, actual)


def _check_release(launch: Launch, report: Report) -> None:
    release = launch.config.release
    if not release:
        return
    prepare, seal = release["prepare"], release["seal"]
    root = launch.path(prepare["output"])
    record = _load_json(root / "private" / "preparation.json", report, "release preparation record")
    if record is not None:
        _expect_path(report, "release source_run", launch.expand(prepare["run_dir"]), record.get("source_run"))
        if "kernel_timing_selection" in prepare:
            found = _dig(record, "scaffolding", "capability_contract", "kernel_timing_selection", "sha256")
            report.expect(
                "release kernel timing selection sha256", prepare["kernel_timing_selection"]["sha256"], found, "hash"
            )
        if "phase1_policy_descriptor" in prepare:
            found = _dig(record, "phase1_policy", "sha256")
            report.expect("release phase1 policy sha256", prepare["phase1_policy_descriptor"]["sha256"], found, "hash")
        if "selected_variants" in prepare:
            found = _dig(record, "scaffolding", "bundle_generation", "selection", "variants")
            report.expect("release selected variants", list(prepare["selected_variants"]), found)
        if prepare["generated_only"]:
            report.expect("release assembly mode", "generated_only", _dig(record, "assembly", "mode"))
        if prepare.get("capsule_scope") == "accelerator":
            report.expect("release capsule scope", "accelerator", _dig(record, "assembly", "capsule_scope", "kind"))
    sealed = _load_json(launch.path(seal["seal"]), report, "release seal")
    if sealed is not None:
        report.expect("release seal reviewed_by", seal["reviewed_by"], _dig(sealed, "review", "reviewed_by"))
        report.expect("release seal review note", seal["review_note"], _dig(sealed, "review", "note"))
    descriptor = launch.path(seal["descriptor"])
    if descriptor.is_file():
        report.ok("release descriptor")
    else:
        report.drift("missing_file", "release descriptor", str(descriptor), None)
    bundle = _load_yaml(launch.path(launch.document["inputs"]["bundle_manifest"]), report, "bundle manifest")
    if bundle is not None:
        report.expect("bundle manifest bundle_id", launch.document["identity"]["bundle"], bundle.get("bundle_id"))


def _path_values(launch: Launch) -> list[tuple[str, str]]:
    values = [("argv", word) for word in launch.argv]
    for name, value in [*launch.environment.items(), *launch.launch_environment.items()]:
        text = value.prefix if isinstance(value, Prepend) else value
        values += [(f"environment {name}", part) for part in text.split(":")]
    return [(where, value) for where, value in values if value.startswith("/")]


def _check_exclusions(launch: Launch, report: Report) -> None:
    exclusions = launch.document.get("exclusions") or ()
    repo = launch.path(launch.document["inputs"]["repo"])
    excluded = [(entry["id"], repo / relative) for entry in exclusions for relative in entry["paths"]]
    for where, value in _path_values(launch):
        for name, root in excluded:
            if _within(value, root):
                report.drift("exclusion", f"{name} named by {where}", "unreferenced", value)
    bundle = _load_yaml(launch.path(launch.document["inputs"]["bundle_manifest"]), Report(), "bundle manifest")
    if bundle is not None:
        for key in ("allowed", "host_inputs"):
            for row in bundle.get(key) or ():
                raw = row.get("path") if isinstance(row, dict) else None
                if not isinstance(raw, str):
                    report.drift("unreadable", f"bundle {key} row", "a path", row)
                    continue
                granted = Path(raw) if Path(raw).is_absolute() else repo / raw
                for name, root in excluded:
                    if _within(granted, root) or _within(root, granted):
                        report.drift("exclusion", f"{name} reachable from bundle {key}", "unreachable", raw)
        report.ok("bundle grants exclude operator-only paths")
    digests = {digest: entry["id"] for entry in exclusions for digest in entry.get("content_sha256") or ()}
    digests[launch.config.sha256] = "launch config"
    if launch.config.release_sha256:
        digests[launch.config.release_sha256] = "release config"
    if launch.document.get("errata"):
        digests[launch.document["errata"]["sha256"]] = "operator errata"
    if launch.config.release:
        payload = launch.path(launch.config.release["prepare"]["output"]) / "payload"
        if payload.is_dir():
            scanned = 0
            for path in sorted(payload.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    scanned += 1
                    owner = digests.get(sha256_file(path))
                    if owner:
                        report.drift(
                            "exclusion", f"{owner} bytes in release payload", "absent", str(path.relative_to(payload))
                        )
            report.ok(f"release payload content ({scanned} files)")


def read_process(pid: int) -> tuple[list[str], dict[str, str], set[int]]:
    """A live process's exact argv, initial environment and CPU affinity."""
    base = Path("/proc") / str(pid)
    argv = [part.decode("utf-8", "surrogateescape") for part in (base / "cmdline").read_bytes().split(b"\0")]
    if argv and argv[-1] == "":
        argv.pop()
    environment = {}
    for item in (base / "environ").read_bytes().split(b"\0"):
        name, equals, value = item.partition(b"=")
        if equals:
            environment[name.decode("utf-8", "surrogateescape")] = value.decode("utf-8", "surrogateescape")
    return argv, environment, set(os.sched_getaffinity(pid))


def _check_process(launch: Launch, pid: int, report: Report, reader=read_process) -> None:
    try:
        argv, environment, cpus = reader(pid)
    except OSError as exc:
        report.drift("process", f"controller {pid}", "readable /proc entry", f"{type(exc).__name__}: {exc}")
        return
    _compare_argv(launch, argv, f"controller {pid} argv", report, template=False)
    expected = {**launch.environment, **launch.launch_environment}
    policy = launch.document.get("environment_policy") or {}
    _compare_environment(expected, environment, f"controller {pid} environment", report, policy, live=True)
    report.expect(f"controller {pid} affinity", sorted(launch.manager_cpus), sorted(cpus), "affinity")


def verify(
    launch: Launch,
    *,
    plan: str | Path | None = None,
    environment_file: str | Path | None = None,
    controller_pid: int | None = None,
    reader=read_process,
) -> Report:
    """Every drift between the config and the prepared inputs (and, optionally, a live controller)."""
    report = Report()
    launcher = launch.document["launcher"]
    environment_file = Path(environment_file) if environment_file else launch.path(launcher["environment_file"])
    _check_pins(launch, report)
    _check_environment_file(launch, environment_file, report)
    selected = _check_plan(launch, Path(plan) if plan else launch.path(launcher["plan"]), report)
    _check_numerics(launch, selected, report)
    _check_release(launch, report)
    _check_exclusions(launch, report)
    if controller_pid is not None:
        _check_process(launch, controller_pid, report, reader)
    return report


def report_record(launch: Launch, report: Report) -> dict:
    return {
        "schema": REPORT_SCHEMA,
        "id": launch.document["id"],
        "config": {"path": str(launch.config.path), "sha256": launch.config.sha256},
        "overrides": launch.overrides,
        "run_id": launch.run_id,
        "status": "pass" if not report.drifts else "drift",
        "checked": len(report.checked),
        "drifts": [drift.record() for drift in report.drifts],
        "passed": list(report.checked),
    }


# --------------------------------------------------------------------------- command line


def _overrides(values: list[str]) -> dict[str, str]:
    result = {}
    for item in values:
        name, equals, value = item.partition("=")
        if not equals or not _is_name(name):
            raise ConfigError(f"--var expects NAME=VALUE, got {item!r}")
        result[name] = value
    return result


def _process_run_id(pid: int, reader=read_process) -> str | None:
    try:
        argv = reader(pid)[0]
    except OSError:
        return None
    values = split_argv(argv)[1].get("--run-id") or [None]
    return values[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("render", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path, help="launch config YAML")
        command.add_argument("--task-root", required=True, type=Path, help="value of ${TASK_ROOT}")
        command.add_argument("--prep", required=True, type=Path, help="value of ${PREP}: the operator prep directory")
        command.add_argument("--run-id", help="value of ${RUN_ID}; default: the config's run-id placeholder")
        command.add_argument("--var", action="append", default=[], metavar="NAME=VALUE", help="override a variable")
    render_command = commands.choices["render"]
    render_command.add_argument("--out", type=Path, help="output directory (default: --prep)")
    render_command.add_argument("--plan-template", type=Path, help="operator plan whose config-owned fields to replace")
    render_command.add_argument("--force", action="store_true", help="replace existing files with different bytes")
    verify_command = commands.choices["verify"]
    verify_command.add_argument("--plan", type=Path, help="launch plan (default: the config's launcher.plan)")
    verify_command.add_argument("--env-file", type=Path, help="environment file (default: launcher.environment_file)")
    verify_command.add_argument("--controller-pid", type=int, help="also compare a live controller's /proc entry")
    verify_command.add_argument("--json", action="store_true", help="print the JSON report")
    verify_command.add_argument("--report", type=Path, help="also write the JSON report here (outside --prep)")
    args = parser.parse_args(argv)
    try:
        config = load(args.config)
        run_id = args.run_id
        if args.command == "verify" and run_id is None and args.controller_pid is not None:
            run_id = _process_run_id(args.controller_pid)
        launch = build(config, task_root=args.task_root, prep=args.prep, run_id=run_id, overrides=_overrides(args.var))
        if args.command == "render":
            written = render(launch, args.out or args.prep, plan_template=args.plan_template, force=args.force)
            print(json.dumps({"status": "rendered", "out": str(args.out or args.prep), "files": written}, indent=2))
            return 0
        if args.report is not None and _within(os.path.abspath(args.report), launch.variables["PREP"]):
            raise ConfigError("--report must not write inside the prep directory it verifies")
        report = verify(launch, plan=args.plan, environment_file=args.env_file, controller_pid=args.controller_pid)
    except (ValueError, OSError) as exc:  # ConfigError, or an unreadable plan template
        print(f"launch config: {exc}", file=sys.stderr)
        return 2
    record = report_record(launch, report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(_json(record))
    if args.json:
        print(json.dumps(record, indent=2))
    else:
        overrides = " ".join(f"{name}={value}" for name, value in launch.overrides.items()) or "none"
        print(f"launch config {launch.document['id']} sha256 {config.sha256[:12]} (overrides: {overrides})")
        for drift in report.drifts:
            print(drift.line())
        print(f"{len(report.checked)} checks passed, {len(report.drifts)} drift(s)")
    return 0 if not report.drifts else 1


if __name__ == "__main__":
    raise SystemExit(main())
