"""Measured-claims catalog command and installed source admission.

Deployment locations are frozen selections, not recursively hashed mutable roots.
Python membership covers the declared core and experiments owners; this does not
attest external tools or prove managed service or scientific readiness.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from merlin.common.paths import module_source_path
from merlin.common.source_membership import SourceMembershipError, python_members

from .spec import SpecError

MODULE = "merlin_experiments.phase2.chia_envelope_cli"
COORDINATOR = "merlin_experiments.phase2.checkpoint_cli"
OWNER = "merlin_experiments.phase2.chia_envelope"
PREFIX = "phase2:installed:"
ENVELOPE_FIELDS = {"managed_native_endpoint", "codex_slots", "gsim_slots"}


def _source(name: str) -> Path:
    path = module_source_path(name)
    if path.is_symlink() or not path.is_file():
        raise SpecError(f"measured launch source is missing or symlinked: {name}")
    return path.resolve()


def _command(values: dict, *, experiment: str, target: str, run_dir: str, storage_root: str) -> dict:
    from .adapters import ADAPTERS

    ADAPTERS["measured_claims"].validate(values)
    entrypoint = _source(MODULE)
    wrapper = _source(OWNER)
    argv = [
        sys.executable,
        "-m",
        MODULE,
        "--driver-python",
        sys.executable,
        "--cwd",
        values["source_root"],
        "--suite",
        values["suite"],
        "--target",
        target,
        "--orchestration-run-id",
        Path(run_dir).name,
        "--managed-native-endpoint",
        values["managed_native_endpoint"],
    ]
    for name in ("codex_slots", "gsim_slots"):
        if name in values:
            argv += ["--" + name.replace("_", "-"), str(values[name])]
    argv += [
        "--",
        "--experiment-id",
        experiment,
        "--root",
        str(Path(run_dir) / "phase2"),
        "--chia-wrapper",
        str(wrapper),
    ]
    for name, value in values.items():
        if name not in ENVELOPE_FIELDS:
            argv += ["--" + name.replace("_", "-"), str(value)]
    roots = [
        str(Path(values[name]).parent)
        for name in ("experiments_package_root", "experiments_namespace_root", "core_package_root")
    ]
    inputs = {
        name: value for name, value in values.items() if ADAPTERS["measured_claims"].options[name].kind == "input"
    }
    return {
        "adapter": "measured_claims",
        "mode": "measured_claims",
        "module": MODULE,
        "entrypoint": str(entrypoint),
        "argv": argv,
        "cwd": values["source_root"],
        "env": {
            "MERLIN_REPO_ROOT": values["source_root"],
            "MERLIN_OUT_ROOT": storage_root,
            "MERLIN_TARGET_EXPERIMENT": values["descriptor"],
            "PYTHONSAFEPATH": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(dict.fromkeys(roots)),
        },
        "inputs": inputs,
        "workspaces": [],
        "resume_policy": "native_chain",
        "engine_output": str(Path(run_dir) / "phase2"),
        "mutable_outputs": [values["stage_root"], values["measurement_root"]],
        "input_owner_roots": [
            values[name] for name in ("core_package_root", "experiments_package_root", "experiments_namespace_root")
        ],
        "managed_endpoint": values["managed_native_endpoint"],
        "measured_launch": {
            "version": 1,
            "values": values,
            "experiment": experiment,
            "target": target,
            "run_dir": run_dir,
            "storage_root": storage_root,
        },
    }


def resolve(spec, values: dict, inputs: dict, run_dir: Path) -> dict:
    from merlin.common.paths import out_dir

    command = _command(
        values, experiment=spec.id, target=spec.target, run_dir=str(run_dir), storage_root=str(out_dir().resolve())
    )
    if command["inputs"] != inputs:
        raise SpecError("measured launch immutable input classification differs")
    return command


def source_inputs(command: dict) -> dict[str, str]:
    """Verify declared owners against actual resolution and loaded module origins."""
    values = command["measured_launch"]["values"]
    owners = {}
    paths = {PREFIX + "interpreter": command.get("argv", [sys.executable])[0]}
    try:
        for role, namespace, field in (
            ("core", "merlin", "core_package_root"),
            ("experiments", "merlin_experiments", "experiments_package_root"),
            ("experiments_namespace", "merlin", "experiments_namespace_root"),
        ):
            for relative, path in python_members(Path(values[field]), label="measured launch").items():
                parts = list(Path(relative).with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts.pop()
                name = ".".join([namespace, *parts])
                if name in owners and owners[name] != path:
                    raise SpecError(f"measured launch has duplicate Python ownership: {name}")
                owners[name] = path
                paths[PREFIX + role + ":" + relative] = str(path)
    except SourceMembershipError as exc:
        raise SpecError(str(exc)) from exc
    for name in (
        "merlin",
        "merlin.common.paths",
        "merlin.common.source_membership",
        "merlin_experiments",
        MODULE,
        OWNER,
        COORDINATOR,
        "merlin_experiments.execution.chia_native",
        "merlin_experiments.execution.chia_group",
        "merlin.targetgen.capsule_runner",
        "merlin.benchharness.chia_bridge",
    ):
        if owners.get(name) != _source(name):
            raise SpecError(f"measured launch declared root differs from installed source owner: {name}")
    for name in owners:
        loaded = sys.modules.get(name)
        if loaded is None:
            continue
        canonical = getattr(loaded, "__name__", name)
        origin = getattr(loaded, "__file__", None)
        if (
            origin is None
            or canonical not in owners
            or sys.modules.get(canonical) is not loaded
            or Path(origin).resolve() != owners[canonical]
        ):
            raise SpecError(f"measured launch loaded source origin differs: {name}")
    for resource in ("experiment.schema.json", "legacy_entrypoints.json"):
        path = Path(values["experiments_package_root"]) / "resources" / resource
        if path.is_symlink() or not path.is_file():
            raise SpecError(f"measured launch resource is absent or linked: {path}")
        paths[PREFIX + "resource:" + resource] = str(path)
    return paths


def execution_environment(command: dict) -> dict[str, str]:
    """Use the frozen installed roots even if the invoking ambient selection changed."""
    environment = dict(os.environ, **command["env"])
    if command.get("module") in {MODULE, "merlin_experiments.phase2.portfolio_cli"}:
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONUSERBASE", None)
    return environment


def _verify_locations(command: dict) -> None:
    """Recheck frozen canonical selections before launch; not a race-free mount claim."""
    from .adapters import ADAPTERS

    values = command["measured_launch"]["values"]
    selected = [
        value for name, value in values.items() if ADAPTERS["measured_claims"].options[name].kind in ("path", "input")
    ]
    selected += [command["measured_launch"]["run_dir"], command["engine_output"]]
    try:
        for value in selected:
            path = Path(value)
            if not path.is_absolute() or path.resolve() != path:
                raise SpecError(f"measured launch canonical location changed: {path}")
    except (OSError, RuntimeError) as exc:
        raise SpecError(f"measured launch canonical location cannot be resolved: {exc}") from exc
    immutable = [Path(value) for value in (*command["inputs"].values(), *command["input_owner_roots"])]
    mutable = [
        Path(value)
        for value in (
            *command["mutable_outputs"],
            command["managed_endpoint"],
            command["engine_output"],
            command["measured_launch"]["run_dir"],
        )
    ]
    for output in mutable:
        for source in immutable:
            if output == source or output.is_relative_to(source) or source.is_relative_to(output):
                raise SpecError(f"measured launch mutable location {output} overlaps frozen input {source}")


def verify_plan(plan: dict) -> None:
    for command in plan["phases"].values():
        if command["adapter"] != "measured_claims":
            continue
        if command.get("module") is None:
            if command["argv"][1:2] != [command["entrypoint"]]:
                raise SpecError("historical measured command differs from its recorded script")
            continue  # Historical native plans retain their original command and inputs.
        record = command.get("measured_launch")
        if not isinstance(record, dict) or record.get("version") != 1:
            raise SpecError("installed measured launch selection is absent or unsupported")
        if any(record.get(key) != plan.get(key) for key in ("experiment", "target", "run_dir", "storage_root")):
            raise SpecError("installed measured launch differs from its experiment ownership")
        expected = _command(
            record["values"],
            experiment=record["experiment"],
            target=record["target"],
            run_dir=record["run_dir"],
            storage_root=record["storage_root"],
        )
        if command != expected:
            raise SpecError("installed measured command differs from its frozen selection")
        _verify_locations(command)
        expected_inputs = source_inputs(command)
        expected_inputs.update({"phase2:" + name: path for name, path in command["inputs"].items()})
        actual = {
            name: path
            for name, path in plan["input_paths"].items()
            if name.startswith(PREFIX) or name in {"phase2:" + key for key in command["inputs"]}
        }
        if expected_inputs != actual:
            raise SpecError("installed measured source membership changed or is absent")
        if "inputs" in plan and any(
            plan["inputs"].get(name, {}).get("path") != path for name, path in expected_inputs.items()
        ):
            raise SpecError("installed measured sources lack frozen fingerprints")
