"""Observe owned compiler/link invocations and their actual file inputs/outputs.

This is compilation evidence, not a hermetic toolchain lock or a cache identity.
Headers, library resolution, imported providers and non-project environment need
their own closure. An explicit linker trace may bind named symbol suppliers;
none are inferred by default. Callers name inputs, never a target ABI here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from merlin.common import invocation_record as I
from merlin.common.digest import is_sha256, sha256_file
from merlin.common.jsonio import strict_json_equal, write_pretty_json

from . import codegen_env
from .link_supplier_trace import observe_link_suppliers, verify_link_suppliers

FILENAME = "compilation_recipe.json"
SCHEMA = "merlin.compilation_recipe.v2"
LEGACY_SCHEMA = "merlin.compilation_recipe.v1"
SCOPE = "caller-owned compilation and link; provider compilation and toolchain dependency closure unknown"
_NON_FILE_VALUE_FLAGS = frozenset({"-I", "-isystem", "-iquote", "-idirafter", "--sysroot", "-b"})


def _identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


class CompilationRecipe:
    """One build's observed commands, completed only after the caller's audits."""

    def __init__(self, work: Path, *, producer: Path):
        self.path = work / FILENAME
        self.record = {
            "schema": SCHEMA,
            "scope": SCOPE,
            "producer": _identity(producer),
            "status": "prepared",
            "commands": [],
        }
        write_pretty_json(self.path, self.record)

    def bind_preparation(self, name: str, path: Path) -> None:
        """Bind a caller-selected numerical preparation input before compilation."""
        if self.record["status"] != "prepared" or not name or not path.is_file():
            raise ValueError("compilation preparation is absent or already invoked")
        self.record.setdefault("preparation", {})[name] = _identity(path)
        write_pretty_json(self.path, self.record)

    def record_link_suppliers(self, expected_suppliers: Mapping[str, Path], result: Any) -> None:
        """Retain actual linker diagnostics before declaring this build complete.

        A missing or ambiguous definition leaves the recipe invoked. The
        independently selected expected supplier is not inferred from an
        archive merely appearing in the link argv.
        """
        commands = self.record["commands"]
        if (
            self.record["status"] != "invoked"
            or not commands
            or commands[-1]["status"] != "returned"
            or "link_suppliers" in self.record
        ):
            raise ValueError("link supplier proof needs one returned final link")
        link = commands[-1]
        if (
            getattr(result, "returncode", None) != 0
            or [str(argument) for argument in getattr(result, "args", ())] != link["argv"]
        ):
            raise ValueError("link supplier proof has no matching returned invocation")
        observation = observe_link_suppliers(
            argv=link["argv"],
            input_identities=link["inputs"],
            expected_suppliers=expected_suppliers,
            stderr=result.stderr,
        )
        self.record["link_suppliers"] = observation
        write_pretty_json(self.path, self.record)

    def run(
        self,
        argv: Sequence[str | Path],
        *,
        runner: Callable[..., Any],
        inputs: Iterable[Path],
        output: Path,
        cwd: Path | None = None,
    ) -> Any:
        first = len(self.record["commands"])
        with I.observe_timings() as events:
            try:
                with I.timed_event("phase", "compilation command"):
                    return self._run(argv, runner=runner, inputs=inputs, output=output, cwd=cwd)
            finally:
                if len(self.record["commands"]) > first:
                    self.record["commands"][first]["timing"] = I.timing_document(events)
                    write_pretty_json(self.path, self.record)

    @contextmanager
    def phase(self, name: str):
        """Observe a caller-owned Python/preparation phase, including nested known tools."""
        if not isinstance(name, str) or not name:
            raise ValueError("compilation timing phase needs an explicit name")
        with I.observe_timings() as events:
            try:
                with I.timed_event("phase", name):
                    yield
            finally:
                self.record.setdefault("phases", []).append({"name": name, "timing": I.timing_document(events)})
                write_pretty_json(self.path, self.record)

    def _run(self, argv, *, runner, inputs, output, cwd):
        """Launch the original command after retaining its explicit input bytes.

        A command error retains captured textual diagnostics and a failed record
        without a returned output, even when an older artifact remains there.
        """
        executable = shutil.which(str(argv[0]))
        if executable is None:
            raise FileNotFoundError(str(argv[0]))
        environment = codegen_env.snapshot()
        command = {
            "argv": [str(argument) for argument in argv],
            "cwd": str((cwd or Path.cwd()).resolve()),
            "executable": _identity(Path(executable)),
            "environment": environment,
            "environment_digest": codegen_env.digest(environment),
            "inputs": [_identity(path) for path in inputs],
            "requested_output": str(output.resolve()),
            "status": "invoked",
        }
        self.record["commands"].append(command)
        self.record["status"] = "invoked"
        write_pretty_json(self.path, self.record)
        started = time.monotonic()
        try:
            with I.timed_event("phase", "command runner"):
                result = runner(argv, **({"cwd": cwd} if cwd is not None else {}))
        except Exception as failure:
            captured = getattr(failure, "captured_process", None)
            expired = getattr(failure, "captured_timeout", None)
            if isinstance(failure, subprocess.TimeoutExpired):
                expired = failure
            self._retain_failure(command, captured, expired, time.monotonic() - started)
            raise
        if result.returncode != 0:
            self._retain_failure(command, result, None, time.monotonic() - started)
            raise RuntimeError(f"compilation command returned {result.returncode}")
        command.update(status="returned", output=_identity(output))
        write_pretty_json(self.path, self.record)
        return result

    def _retain_failure(self, command, captured, expired, elapsed):
        """Retain runner diagnostics; this cannot complete or certify a product."""
        command.update(status="timeout" if expired is not None else "failed", elapsed_s=elapsed)
        if captured is not None:
            command["returncode"] = captured.returncode
        if expired is not None:
            command["timeout_s"] = expired.timeout
        observation = expired if expired is not None else captured
        if observation is not None:
            directory = self.path.parent / "compiler-diagnostics" / uuid.uuid4().hex
            directory.mkdir(parents=True)
            diagnostics = {}
            for name in ("stdout", "stderr"):
                value = getattr(observation, name, None)
                if value is not None:
                    path = directory / (name + ".bin")
                    path.write_bytes(value.encode("utf-8") if isinstance(value, str) else value)
                    diagnostics[name] = _identity(path)
            command["captured_diagnostics"] = diagnostics
        self.record["status"] = command["status"]
        write_pretty_json(self.path, self.record)

    def completed(self, executable: Path) -> None:
        """Bind the final executable only after successful linking and audits."""
        self.completed_product(executable, kind="executable")

    def completed_product(self, product: Path, *, kind: str) -> None:
        """Record a completed compiler product without labeling IR as a binary.

        The legacy executable record is unchanged. Intermediate LLVM IR and
        objects need their own normal codegen/link qualification afterwards.
        """
        if kind not in {"executable", "object", "llvm_ir"}:
            raise ValueError("explicit supported compilation product kind required")
        if not self.record["commands"] or any(command["status"] != "returned" for command in self.record["commands"]):
            raise ValueError("compilation has no complete successful command sequence")
        if any(
            not Path(identity["path"]).is_file() or _identity(Path(identity["path"])) != identity
            for identity in self.record.get("preparation", {}).values()
        ):
            raise ValueError("compilation preparation changed before completion")
        _recheck_weight_preparation(self.record.get("preparation", {}))
        self.record.update(status="completed", **{kind: _identity(product)})
        write_pretty_json(self.path, self.record)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"compilation recipe has duplicate JSON field {key!r}")
        result[key] = value
    return result


def _recheck_weight_preparation(preparation):
    if "weight_prepack" in preparation:
        from .weight_prepack import revalidate_build_preparation

        revalidate_build_preparation(Path(preparation["weight_prepack"]["path"]))


def _ordinary_path(value: object, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("compilation recipe has an empty path")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("compilation recipe path is indirect or not absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("compilation recipe path is absent") from exc
    if resolved != path or not (path.is_dir() if directory else path.is_file()):
        raise ValueError("compilation recipe path is indirect or not ordinary")
    return path


def _checked_identity(value: object, label: str, *, defer_hash: bool = False) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256", "bytes"}
        or not is_sha256(value.get("sha256"))
        or type(value.get("bytes")) is not int
        or value["bytes"] < 0
    ):
        raise ValueError(f"compilation recipe {label} has no exact file identity")
    path = _ordinary_path(value["path"])
    if path.stat().st_size != value["bytes"] or (not defer_hash and sha256_file(path) != value["sha256"]):
        raise ValueError(f"compilation recipe {label} bytes changed")
    return value


def _argument_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    try:
        return (path if path.is_absolute() else cwd / path).resolve(strict=True)
    except OSError as exc:
        raise ValueError("compilation recipe argv names an absent direct file") from exc


def _checked_command(
    value: object,
    prior_outputs: set[str],
    *,
    final: bool,
    defer_output_hash: bool = False,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) - {"timing"}
        != {
            "argv",
            "cwd",
            "executable",
            "environment",
            "environment_digest",
            "inputs",
            "requested_output",
            "status",
            "output",
        }
        or value.get("status") != "returned"
    ):
        raise ValueError("compilation recipe has an incomplete command")
    if "timing" in value:
        I.verify_timing_document(value["timing"], root_label="compilation command")
        if value["timing"]["events"][-1]["status"] != "returned":
            raise ValueError("compilation recipe timing reports a failed returned command")
    argv = value["argv"]
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv):
        raise ValueError("compilation recipe command has no complete argv")
    cwd = _ordinary_path(value["cwd"], directory=True)
    executable = _checked_identity(value["executable"], "command executable")
    if not Path(argv[0]).is_absolute() or _argument_path(argv[0], cwd) != Path(executable["path"]):
        raise ValueError("compilation recipe command selected another executable")
    environment = value["environment"]
    if (
        not isinstance(environment, dict)
        or any(not isinstance(key, str) or not isinstance(item, str) for key, item in environment.items())
        or codegen_env.snapshot(environment) != environment
        or value["environment_digest"] != codegen_env.digest(environment)
    ):
        raise ValueError("compilation recipe command environment digest differs")
    inputs = value["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("compilation recipe command has no explicit inputs")
    input_paths = []
    for item in inputs:
        input_paths.append(_checked_identity(item, "input")["path"])
    if len(set(input_paths)) != len(input_paths):
        raise ValueError("compilation recipe command repeats an input identity")
    output = _checked_identity(value["output"], "output", defer_hash=defer_output_hash)
    output_path = output["path"]
    if (
        value["requested_output"] != output_path
        or output_path in input_paths
        or output_path in prior_outputs
        or argv.count("-o") != 1
        or argv.index("-o") + 1 >= len(argv)
        or _argument_path(argv[argv.index("-o") + 1], cwd) != Path(output_path)
    ):
        raise ValueError("compilation recipe command output differs from its invocation")
    direct_inputs = set()
    for index, arg in enumerate(argv[1:], start=1):
        if arg.startswith("-") or argv[index - 1] in _NON_FILE_VALUE_FLAGS | {"-o"}:
            continue
        path = _argument_path(arg, cwd)
        if str(path) != output_path and str(path) not in input_paths:
            raise ValueError("compilation recipe argv has an unrecorded direct file input")
        if str(path) in input_paths:
            direct_inputs.add(str(path))
    if final and direct_inputs != set(input_paths):
        raise ValueError("compilation recipe link input has no direct argv reference")
    prior_outputs.add(output_path)
    return value


def verify_completed_recipe(
    recipe_path: Path,
    *,
    executable: Path,
    expected_recipe_sha256: str,
    expected_link_suppliers: Mapping[str, Path] | None = None,
    terminal_read_policy: str = "once",
) -> dict[str, Any]:
    """Recheck a completed v1/v2 observation against current explicit file bytes.

    Imported headers, ambient tool dependencies and numerical behavior are not
    inferred. A supplier claim requires the opt-in trace plus an independent
    ``expected_link_suppliers`` selection; argv archive presence alone is not proof.

    ``once`` defers only the terminal command output's content read to the
    existing final executable check after exact metadata equality. No tool or
    callback runs between these checks. ``repeated`` retains both reads as an
    explicit comparison mode. Every call freshly hashes the final executable;
    neither mode supplies a cache or protects against concurrent file writers.
    """
    if type(terminal_read_policy) is not str or terminal_read_policy not in ("once", "repeated"):
        raise ValueError("compilation recipe terminal read policy must be once or repeated")
    if not is_sha256(expected_recipe_sha256):
        raise ValueError("compilation recipe has no selected full digest")
    recipe = _ordinary_path(str(recipe_path))
    if recipe.name != FILENAME or sha256_file(recipe) != expected_recipe_sha256:
        raise ValueError("compilation recipe bytes differ from selected digest")
    selected_elf = _ordinary_path(str(executable))
    try:
        record = json.loads(recipe.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("compilation recipe is not valid JSON") from exc
    if (
        not isinstance(record, dict)
        or not {"schema", "scope", "producer", "status", "commands", "executable"}.issubset(record)
        or set(record)
        - {"schema", "scope", "producer", "status", "commands", "executable", "preparation", "link_suppliers", "phases"}
        or record.get("schema") not in (SCHEMA, LEGACY_SCHEMA)
        or record.get("scope") != SCOPE
        or record.get("status") != "completed"
    ):
        raise ValueError("compilation recipe is not a completed observation")
    phases = record.get("phases", [])
    if type(phases) is not list:
        raise ValueError("compilation recipe phase observations are malformed")
    for phase in phases:
        if (
            not isinstance(phase, dict)
            or set(phase) != {"name", "timing"}
            or not isinstance(phase["name"], str)
            or not phase["name"]
        ):
            raise ValueError("compilation recipe phase observation has no name")
        I.verify_timing_document(phase["timing"], root_label=phase["name"])
    _checked_identity(record["producer"], "producer")
    preparation = record.get("preparation", {})
    if not isinstance(preparation, dict) or any(not isinstance(key, str) or not key for key in preparation):
        raise ValueError("compilation recipe preparation is malformed")
    preparation_paths = []
    for identity in preparation.values():
        preparation_paths.append(_checked_identity(identity, "preparation")["path"])
    if len(set(preparation_paths)) != len(preparation_paths):
        raise ValueError("compilation recipe repeats a preparation identity")
    _recheck_weight_preparation(preparation)
    commands = record["commands"]
    if not isinstance(commands, list) or not commands:
        raise ValueError("compilation recipe has no returned commands")
    if record["schema"] == LEGACY_SCHEMA and (
        "phases" in record or any(isinstance(command, dict) and "timing" in command for command in commands)
    ):
        raise ValueError("compilation recipe timing requires the v2 schema")
    outputs: set[str] = set()
    for index, command in enumerate(commands):
        final_command = index == len(commands) - 1
        _checked_command(
            command,
            outputs,
            final=final_command,
            defer_output_hash=final_command and terminal_read_policy == "once",
        )
    final = record["executable"]
    if not strict_json_equal(final, commands[-1]["output"]) or final["path"] != str(selected_elf):
        raise ValueError("compilation recipe final executable differs from link output")
    _checked_identity(final, "final executable")
    observation = record.get("link_suppliers")
    if expected_link_suppliers is not None and observation is None:
        raise ValueError("compilation recipe has no requested link supplier proof")
    if observation is not None:
        verify_link_suppliers(
            observation,
            argv=commands[-1]["argv"],
            input_identities=commands[-1]["inputs"],
            expected_suppliers=expected_link_suppliers,
        )
    if sha256_file(recipe) != expected_recipe_sha256:
        raise ValueError("compilation recipe changed during verification")
    return record
