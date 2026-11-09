"""Installed portfolio catalog commands and their immutable deployment selections."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from merlin.common.paths import module_source_path

from .spec import SpecError

MODULE = "merlin_experiments.phase2.portfolio_cli"
PREFIX = "phase2:portfolio:"


def _command(values: dict, *, target: str, run_dir: str, storage_root: str, template: bool) -> dict:
    from .adapters import ADAPTERS
    from .phase2.portfolio_cli import load_deployment

    adapter = ADAPTERS["model_portfolio"]
    adapter.validate(values)
    entrypoint = module_source_path(MODULE)
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise SpecError("portfolio entrypoint must be an ordinary source file")
    entrypoint = entrypoint.resolve()
    deployment = None if template else load_deployment(Path(values["deployment"]))
    if deployment is not None and (deployment["target"] != target or deployment["output_root"] != storage_root):
        raise SpecError("portfolio deployment target/storage differs from experiment ownership")
    roots = (
        []
        if deployment is None
        else [str(Path(deployment["source_root"]) / root) for root in deployment["python_roots"]]
    )
    argv = [sys.executable, "-m", MODULE, "--output", str(Path(run_dir) / "phase2/segment-0001")]
    for name, value in values.items():
        option = adapter.options[name]
        flag = option.flag or "--" + name.replace("_", "-")
        if option.kind == "bool":
            if value:
                argv.append(flag)
        elif option.kind == "strings":
            for item in value:
                argv.extend((flag, item))
        else:
            argv.extend((flag, str(value)))
    return {
        "adapter": "model_portfolio",
        "mode": "model_portfolio",
        "module": MODULE,
        "entrypoint": str(entrypoint),
        "argv": argv,
        "cwd": str(entrypoint.parent) if deployment is None else deployment["source_root"],
        "env": {
            "MERLIN_OUT_ROOT": storage_root,
            "PYTHONPATH": os.pathsep.join(roots),
            "PYTHONSAFEPATH": "1",
            "PYTHONNOUSERSITE": "1",
        },
        "inputs": {name: value for name, value in values.items() if adapter.options[name].kind == "input"},
        "workspaces": [value for name, value in values.items() if adapter.options[name].kind == "workspace"],
        "resume_policy": "checkpoint_segment",
        "engine_output": str(Path(run_dir) / "phase2"),
        "mutable_outputs": [] if deployment is None else [deployment["lease_path"]],
        "portfolio_launch": {
            "version": 1,
            "values": values,
            "target": target,
            "run_dir": run_dir,
            "storage_root": storage_root,
            "template": template,
        },
    }


def resolve(spec, values: dict, inputs: dict, run_dir: Path) -> dict:
    from merlin.common.paths import out_dir

    try:
        command = _command(
            values,
            target=spec.target,
            run_dir=str(run_dir),
            storage_root=str(out_dir().resolve()),
            template=spec.document.get("kind") == "template",
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SpecError(f"invalid portfolio deployment: {exc}") from exc
    if command["inputs"] != inputs:
        raise SpecError("portfolio immutable input classification differs")
    return command


def source_inputs(command: dict) -> dict[str, str]:
    from .phase2.portfolio_cli import deployment_source_inputs

    if command["portfolio_launch"]["template"]:
        return {}
    try:
        inputs = deployment_source_inputs(Path(command["inputs"]["deployment"]))
        raw = Path(command["inputs"]["campaign_config"]).read_bytes()
        document = json.loads(raw)
        config = document["campaigns"][0]["config"] if "campaigns" in document else document.get("config", document)
        for name in ("descriptor", "telemetry_price_table"):
            if name not in config:
                if name == "descriptor":
                    raise ValueError("campaign descriptor is required")
                continue
            path = Path(config[name])
            if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
                raise ValueError(f"campaign {name} must be a canonical ordinary file")
            inputs["campaign:" + name] = str(path)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise SpecError(f"portfolio source admission failed: {exc}") from exc
    return {PREFIX + key: value for key, value in inputs.items()}


def verify_plan(plan: dict) -> None:
    for command in plan["phases"].values():
        if command["adapter"] != "model_portfolio":
            continue
        if command.get("module") is None:
            if command["argv"][1:2] != [command["entrypoint"]]:
                raise SpecError("historical portfolio command differs from its recorded script")
            continue
        record = command.get("portfolio_launch")
        if not isinstance(record, dict) or record.get("version") != 1:
            raise SpecError("installed portfolio deployment record is absent or unsupported")
        if any(record.get(key) != plan.get(key) for key in ("target", "run_dir", "storage_root")):
            raise SpecError("portfolio command differs from experiment ownership")
        if record.get("template") is not (plan["spec"].get("kind") == "template"):
            raise SpecError("portfolio template classification differs from experiment")
        try:
            expected = _command(
                **{key: record[key] for key in ("values", "target", "run_dir", "storage_root", "template")}
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise SpecError(f"portfolio selection cannot be restored: {exc}") from exc
        if command != expected:
            raise SpecError("portfolio command differs from its frozen selection")
        expected_inputs = source_inputs(command)
        actual = {key: value for key, value in plan["input_paths"].items() if key.startswith(PREFIX)}
        if expected_inputs != actual:
            raise SpecError("portfolio source membership changed or is absent")
        writable = [Path(plan["run_dir"]), Path(command["engine_output"])]
        writable.extend(Path(value) for value in command["workspaces"] + command["mutable_outputs"])
        for value in (*expected_inputs.values(), *command["inputs"].values()):
            source = Path(value).resolve()
            if any(path.resolve().is_relative_to(source) or source.is_relative_to(path.resolve()) for path in writable):
                raise SpecError("portfolio mutable output/workspace overlaps frozen input")
        if "inputs" in plan and any(
            plan["inputs"].get(key, {}).get("path") != value for key, value in expected_inputs.items()
        ):
            raise SpecError("portfolio sources lack frozen fingerprints")
