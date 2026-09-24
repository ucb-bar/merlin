"""Typed compatibility adapters for the existing production phase entrypoints.

This module does not import the legacy engines: even their imports can discover
hardware, mutate environment, or require optional tools. It builds argv only.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from .spec import SpecError

PHASE0_MODULE = "merlin_experiments.phase0"
PHASE1_MODULE = "merlin_experiments.phase1"


def phase1_entrypoint() -> Path:
    """Resolve the installed command, never a target-local executable alias."""
    from merlin.common.paths import module_source_path

    path = module_source_path(PHASE1_MODULE + ".__main__")
    if path.is_symlink() or not path.is_file():
        raise SpecError("installed phase-1 entrypoint is absent or symlinked")
    return path.resolve()


def phase0_sources() -> tuple[Path, tuple[Path, ...]]:
    """Resolve the installed implementation, excluding interpreter cache products."""
    from merlin.common.paths import module_source_path

    entrypoint = module_source_path(PHASE0_MODULE + ".__main__")
    package = entrypoint.parent
    if package.is_symlink():
        raise SpecError("phase-0 implementation package may not be a symlink")
    sources = []
    for member in sorted(package.rglob("*")):
        if "__pycache__" in member.relative_to(package).parts:
            continue
        if member.is_symlink() or not (member.is_file() or member.is_dir()):
            raise SpecError("phase-0 implementation contains symlinked or nonregular entries")
        if member.is_file() and member.suffix == ".py":
            sources.append(member.resolve())
    if entrypoint.is_symlink() or entrypoint.resolve() not in sources:
        raise SpecError("phase-0 module has no authoritative Python entrypoint")
    return entrypoint.resolve(), tuple(sources)


def phase0_startup_inputs() -> dict[str, Path]:
    """Exact package startup and historical-citation binding, not a dependency crawl."""
    from merlin.common.paths import module_source_path

    initializer = module_source_path("merlin_experiments")
    inputs = {
        "package": initializer,
        "spec": module_source_path("merlin_experiments.spec"),
        "specir_integration": module_source_path("merlin.integrations.specir"),
        "provenance_binding": initializer.parent / "resources/legacy_entrypoints.json",
    }
    if any(path.is_symlink() or not path.is_file() for path in inputs.values()):
        raise SpecError("phase-0 startup/provenance input is missing or symlinked")
    return {name: path.resolve() for name, path in inputs.items()}


def _relative_entrypoint(value) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise SpecError("legacy entrypoint must be a nonempty relative Python source path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise SpecError(f"legacy entrypoint must remain inside its checkout: {value!r}")
    return value


@lru_cache(maxsize=1)
def _legacy_entrypoints() -> dict:
    """Execution-location data, never target facts or user-supplied commands."""
    source = files("merlin_experiments").joinpath("resources", "legacy_entrypoints.json")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SpecError(f"bundled legacy entrypoints are unavailable or invalid: {exc}") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "entrypoints"}
        or document["schema_version"] != 1
        or not isinstance(document["entrypoints"], dict)
    ):
        raise SpecError("bundled legacy entrypoints must use schema_version: 1")
    for adapter, variants in document["entrypoints"].items():
        if not isinstance(adapter, str) or not isinstance(variants, dict) or "default" not in variants:
            raise SpecError("each legacy adapter binding needs a default entrypoint")
        for variant, path in variants.items():
            if not isinstance(variant, str) or not variant:
                raise SpecError("legacy entrypoint variant must be named")
            _relative_entrypoint(path)
    return document["entrypoints"]


def _legacy_script(adapter: str, variant: str = "default") -> str:
    try:
        return _legacy_entrypoints()[adapter][variant]
    except KeyError as exc:
        raise SpecError(f"bundled legacy entrypoint binding missing: {adapter}/{variant}") from exc


def _entrypoint_path(root: Path, script: str) -> Path:
    relative = _relative_entrypoint(script)
    checkout = root.resolve()
    resolved = (checkout / relative).resolve()
    if not resolved.is_relative_to(checkout):
        raise SpecError(f"legacy entrypoint escapes its checkout: {script!r}")
    if not resolved.is_file():
        raise SpecError(f"legacy phase entrypoint missing: {resolved}; configure its source checkout")
    return resolved


@dataclass(frozen=True)
class Option:
    kind: str = "text"
    required: bool = False
    choices: tuple[str, ...] = ()
    flag: str | None = None

    def validate(self, name: str, value) -> None:
        if self.kind == "bool":
            valid = isinstance(value, bool)
        elif self.kind == "positive":
            valid = type(value) is int and value > 0
        elif self.kind == "strings":
            valid = isinstance(value, list) and all(isinstance(v, str) and v for v in value)
        else:
            valid = isinstance(value, str) and bool(value.strip()) and "\x00" not in value
        if not valid or (self.choices and value not in self.choices):
            raise SpecError(f"invalid {name}: expected {self.kind}" + (f" in {self.choices}" if self.choices else ""))
        if name.endswith("sha256") and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            raise SpecError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class Adapter:
    name: str
    phase: str
    script: str
    options: dict[str, Option] = field(default_factory=dict)
    mode: str | None = None
    resume: str = "retry"
    module: str | None = None

    def validate(self, config: dict) -> None:
        unknown = set(config) - set(self.options)
        missing = {name for name, option in self.options.items() if option.required} - set(config)
        if unknown or missing:
            raise SpecError(f"{self.name}: unknown options {sorted(unknown)}; missing options {sorted(missing)}")
        for name, value in config.items():
            self.options[name].validate(name, value)
        if self.module == PHASE1_MODULE:
            installed_inputs = {"bundle", "bundle_manifest", "oracle_timing"}
            missing = installed_inputs - config.keys()
            if missing:
                raise SpecError(f"installed Phase 1 requires explicit inputs: {sorted(missing)}")
        if (
            self.name == "capsule_bench"
            and config.get("treatment") == "rtlchecks"
            and config.get("arm") != "merlin_assisted"
        ):
            raise SpecError("rtlchecks treatment requires arm: merlin_assisted")
        for name in (
            "external_objective",
            "optimization_baseline",
            "mechanism_catalog",
            "mechanism_work_order",
            "functional_gsim_certificate",
        ):
            if (name in config) != (name + "_sha256" in config):
                raise SpecError(f"{name} and {name}_sha256 must be provided together")
        if self.name == "model_portfolio" and "objective_capsule" in config and "external_objective" in config:
            raise SpecError("model_portfolio accepts either objective_capsule or external_objective")
        if self.name == "capsule_derivation" and "profile" in config:
            profile = config["profile"]
            if Path(profile).name != profile or profile in (".", ".."):
                raise SpecError("profile must be one profile name, without a path")
        if self.name == "capsule_derivation":
            explicit = {
                "recipe",
                "performance_template",
                "conformance_spec",
                "synth_profile",
                "smt_profile",
                "hidden_profile",
            }
            if explicit & config.keys():
                if not {"recipe", "performance_template"} <= config.keys():
                    raise SpecError("explicit phase-0 inputs require recipe and performance_template together")
                if "profiles_root" in config:
                    raise SpecError("explicit phase-0 recipe inputs are mutually exclusive with profiles_root")
                if config.get("comparison_manifest"):
                    raise SpecError("explicit phase-0 recipe does not support comparison_manifest")

    def resolve(self, spec, config: dict, root: Path, run_dir: Path) -> dict:
        from merlin.common.paths import out_dir, python_import_roots

        values = dict(config)
        inputs = {}
        for name, option in self.options.items():
            if name in values and option.kind in ("input", "workspace", "path"):
                values[name] = str(spec.resolve(values[name]))
                if option.kind == "input":
                    inputs[name] = values[name]
        if self.name == "measured_claims":
            from . import measured_launch

            return measured_launch.resolve(spec, values, inputs, run_dir)
        if self.name == "model_portfolio":
            from . import portfolio_catalog

            return portfolio_catalog.resolve(spec, values, inputs, run_dir)
        script_name = self.script
        module = self.module
        if module:
            if self.name == "capsule_derivation" and module == PHASE0_MODULE:
                script, _ = phase0_sources()
            elif self.name == "capsule_bench" and module == PHASE1_MODULE:
                script = phase1_entrypoint()
            else:
                raise SpecError("unsupported installed phase module")
            argv = [sys.executable, "-m", module]
        else:
            script = _entrypoint_path(root, script_name)
            argv = [sys.executable, str(script)]
        env = {
            "MERLIN_REPO_ROOT": str(root),
            "MERLIN_OUT_ROOT": str(out_dir().resolve()),
            "PYTHONPATH": os.pathsep.join(
                [*(str(path) for path in python_import_roots()), *filter(None, [os.environ.get("PYTHONPATH")])]
            ),
        }
        if module:
            # `-m` must not pick up a same-named package from an experiment cwd.
            # Use the same extension import root that supplied the pinned module.
            env["PYTHONSAFEPATH"] = "1"
            env["PYTHONPATH"] = os.pathsep.join([str(script.parent.parent.parent), env["PYTHONPATH"]])
        if self.name == "capsule_derivation":
            profile = values.get("profile", spec.target)
            if self.module and "profiles_root" not in values and "recipe" not in values:
                raise SpecError(
                    "phase-0 orchestration requires explicit recipe inputs or an explicit profiles_root input"
                )
            argv += ["--target", profile, "--output-root", str(run_dir / "phase0" / "capsules")]
        elif self.name == "capsule_bench":
            env["MERLIN_TARGET_EXPERIMENT"] = values["descriptor"]
            env["MERLIN_CORPUS_SEAL"] = values.get("corpus_seal", "")
            argv += ["--run-id", run_dir.name, "--schedule", "continuous", "--sandbox", "bwrap"]
            if module == PHASE1_MODULE:
                argv += ["--repo", str(root), "--descriptor", values["descriptor"]]
        elif self.name == "measured_claims":
            argv += ["--experiment-id", spec.id, "--root", str(run_dir / "phase2")]
        elif self.name == "model_portfolio":
            argv += ["--output", str(run_dir / "phase2" / "segment-0001")]
        if "descriptor" in values:
            env["MERLIN_TARGET_EXPERIMENT"] = values["descriptor"]
        for name, value in values.items():
            option = self.options[name]
            if option.flag == "":
                continue
            flag = option.flag or "--" + name.replace("_", "-")
            if option.kind == "bool":
                if value:
                    argv.append(flag)
            elif option.kind == "strings":
                for item in value:
                    argv += [flag, item]
            else:
                argv += [flag, str(value)]
        engine_output = str(run_dir / "phase2") if self.phase == "2" else None
        if self.name == "capsule_derivation":
            engine_output = str(run_dir / "phase0" / "capsules")
        if self.name == "capsule_bench":
            from merlin.common.paths import runs_dir

            engine_output = str(runs_dir() / spec.target / "capsule-bench" / values["arm"] / run_dir.name)
        return {
            "adapter": self.name,
            "mode": self.mode,
            "argv": argv,
            "env": env,
            "cwd": str(root),
            "inputs": inputs,
            **(
                {"requires_reviewed_corpus": values.get("require_reviewed_corpus", False)}
                if self.name == "capsule_bench"
                else {}
            ),
            "entrypoint": str(script),
            "module": module,
            "resume_policy": self.resume,
            "engine_output": engine_output,
            "workspaces": [
                values[name] for name, option in self.options.items() if name in values and option.kind == "workspace"
            ],
        }


_TEXT = Option()
_POSITIVE = Option("positive")
_INPUT = Option("input")
_REQUIRED_INPUT = Option("input", True)
_REQUIRED_TEXT = Option(required=True)
_REQUIRED_POSITIVE = Option("positive", True)

ADAPTERS = {
    "capsule_derivation": Adapter(
        "capsule_derivation",
        "0",
        _legacy_script("capsule_derivation"),
        {
            "descriptor": Option("input", True),
            "profile": Option(flag=""),
            "profiles_root": Option("input"),
            "recipe": Option("input"),
            "performance_template": Option("input"),
            "conformance_spec": Option("input"),
            "synth_profile": Option("input"),
            "smt_profile": Option("input"),
            "hidden_profile": Option("input"),
            "comparison_manifest": Option("bool"),
        },
        module=PHASE0_MODULE,
    ),
    "capsule_bench": Adapter(
        "capsule_bench",
        "1",
        _legacy_script("capsule_bench"),
        {
            "descriptor": Option("input", True, flag=""),
            "corpus_seal": Option("input", flag=""),
            "require_reviewed_corpus": Option("bool", flag=""),
            "arm": Option(required=True, choices=("raw_baseline", "cpp_merlininfra", "merlin_assisted")),
            "treatment": Option(choices=("baseline", "rtlchecks")),
            "model": _REQUIRED_TEXT,
            "effort": _REQUIRED_TEXT,
            "driver": Option(choices=("auto", "converse", "claudecode", "opencode", "codex")),
            "provider": Option(choices=("subscription", "bedrock")),
            "max_wall_s": _REQUIRED_POSITIVE,
            "round_timeout": _REQUIRED_POSITIVE,
            "grade_interval": _POSITIVE,
            "qa_timeout": _POSITIVE,
            "sim_max_jobs": _POSITIVE,
            "model_budget_s": _POSITIVE,
            "plateau_rounds": _POSITIVE,
            "bundle": _TEXT,
            "bundle_manifest": _INPUT,
            "oracle_timing": _INPUT,
            "experiment": Option(choices=("full", "realistic")),
            "with_tool": Option("strings"),
            "without_tool": Option("strings"),
        },
        resume="native_flag",
        module=PHASE1_MODULE,
    ),
    "measured_claims": Adapter(
        "measured_claims",
        "2",
        _legacy_script("measured_claims"),
        {
            "source_root": Option("path", required=True),
            "contract_root": _REQUIRED_INPUT,
            "functional_runs_root": Option("path", required=True),
            "stage_root": Option("path", required=True),
            "measurement_root": Option("path", required=True),
            "holdout_catalog": _REQUIRED_INPUT,
            "core_package_root": Option("path", required=True),
            "experiments_package_root": Option("path", required=True),
            "experiments_namespace_root": Option("path", required=True),
            "managed_native_endpoint": Option("path", required=True),
            "suite": _REQUIRED_TEXT,
            "codex_slots": _POSITIVE,
            "gsim_slots": _POSITIVE,
            "functional_run_id": _REQUIRED_TEXT,
            "functional_submission_sha256": _REQUIRED_TEXT,
            "descriptor": _REQUIRED_INPUT,
            "rtl_facts": _REQUIRED_INPUT,
            "perf_profile": _REQUIRED_INPUT,
            "gsim_certificate": _REQUIRED_INPUT,
            "gsim_certificate_sha256": _REQUIRED_TEXT,
            "functional_gsim_certificate": _INPUT,
            "functional_gsim_certificate_sha256": _TEXT,
            "model": _REQUIRED_TEXT,
            "effort": _REQUIRED_TEXT,
            "wall_budget_seconds": _REQUIRED_POSITIVE,
            "rounds": _REQUIRED_POSITIVE,
            "round_timeout_seconds": _REQUIRED_POSITIVE,
            "max_tool_calls": _REQUIRED_POSITIVE,
            "tool_timeout_seconds": _REQUIRED_POSITIVE,
            "measurement_timeout": _POSITIVE,
            "smoke_replicates": _POSITIVE,
            "holdout_count": _POSITIVE,
            "generalization_count": _POSITIVE,
            "perf_capsules": _TEXT,
            "perf_families": _TEXT,
            "telemetry_price_table": _INPUT,
        },
        mode="measured_claims",
        resume="native_chain",
        module="merlin_experiments.phase2.chia_envelope_cli",
    ),
    "model_portfolio": Adapter(
        "model_portfolio",
        "2",
        _legacy_script("model_portfolio"),
        {
            "deployment": _REQUIRED_INPUT,
            "campaign_config": _REQUIRED_INPUT,
            "candidate": Option("workspace", True),
            "objective_capsule": _TEXT,
            "portfolio_capsule": Option("strings"),
            "external_objective": _INPUT,
            "external_objective_sha256": _TEXT,
            "optimization_baseline": _INPUT,
            "optimization_baseline_sha256": _TEXT,
            "functional_gate": _INPUT,
            "mechanism_catalog": _INPUT,
            "mechanism_catalog_sha256": _TEXT,
            "mechanism_work_order": _INPUT,
            "mechanism_work_order_sha256": _TEXT,
            "round_seconds": _REQUIRED_POSITIVE,
            "iteration_seconds": _POSITIVE,
            "max_tool_calls": _REQUIRED_POSITIVE,
            "max_rounds": _REQUIRED_POSITIVE,
            "total_authoring_seconds": _REQUIRED_POSITIVE,
            "portfolio_analysis_workers": _POSITIVE,
        },
        mode="model_portfolio",
        resume="checkpoint_segment",
        module="merlin_experiments.phase2.portfolio_cli",
    ),
}
