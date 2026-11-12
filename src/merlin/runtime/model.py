"""Select explicit engines and execute admitted saved images without choosing acceptance policy."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.compiler.build.modeloptions import BaremetalModelError, _sha
from merlin.runtime import modelinputs as MI
from merlin.runtime.backends import spike as spike_backend
from merlin.runtime.backends import spike_model


def _retain_simulator_failure_output(
    exc: subprocess.TimeoutExpired | subprocess.CalledProcessError,
    output: Path,
    receipt: dict[str, Any],
) -> None:
    """Keep incomplete process streams as byte evidence, never as model OUT."""
    artifacts: dict[str, Any] = {"scope": "diagnostic_partial_simulator_output_only"}
    for stream in ("stdout", "stderr"):
        value = getattr(exc, stream, None)
        if stream == "stdout" and value is None:
            value = getattr(exc, "output", None)
        if value is None:
            continue
        data = value if isinstance(value, bytes) else value.encode("utf-8", errors="surrogateescape")
        path = output / f"simulator_{stream}.partial.bin"
        path.write_bytes(data)
        artifacts[stream] = {"path": str(path), "sha256": _sha(path), "bytes": len(data)}
    receipt["failure_artifacts"] = artifacts


def _spike_extension_backend(target: str):
    """The target backend that runs ELFs on the functional simulator WITH its accelerator extension.

    ``None`` for a target whose backend declares no extension (a host-only image then runs on the
    plain simulator); an extension that is declared but cannot be resolved refuses the run."""
    from merlin.runtime.backends import base

    try:
        backend = base.get_backend(target)
    except KeyError:
        return None
    if not callable(getattr(backend, "spike_extension", None)) or not callable(getattr(backend, "run_elf", None)):
        return None
    return backend


def _native_engine(target: str, run: str, facts: dict[str, str]):
    from merlin.development.targetgen.oracle_policy import selected_l3_engine_report

    selection = selected_l3_engine_report(target)
    if not selection.get("available") or selection.get("engine") != run:
        raise BaremetalModelError(f"requested {run} is not the selected elaborated-RTL engine: {selection}")
    backend, citation, revalidate, prepare = MI.native_engine(target, run, facts)
    if not callable(getattr(backend, "run_elf", None)):
        raise BaremetalModelError(f"selected backend has no public run_elf for {target}")
    return backend, {"selection": selection, "citation": citation}, revalidate, prepare


@dataclass(frozen=True)
class SelectRuntimeResult:
    native: bool
    facts: dict | None
    backend: object
    selection: dict | None
    revalidate: Callable | None
    prepare: Callable | None


def select_runtime(
    *,
    native_engine,
    receipt,
    rtl_facts,
    run,
    selected,
    target,
):
    """Resolve the explicitly requested engine and its original byte-bound implementation."""
    facts = None
    native = run in {"gsim", "verilator"}
    if native:
        if rtl_facts is None or not selected.rtl_sim_config:
            raise BaremetalModelError("native RTL execution requires explicit facts and board RTL config")
        facts = MI.selected_firrtl(rtl_facts, target=target, config=selected.rtl_sim_config)
        ambient = os.environ.get("MERLIN_RTL_FACTS", "").strip()
        if ambient and _sha(Path(ambient)) != facts["sha256"]:
            raise BaremetalModelError("ambient RTL facts differ from explicitly selected native facts")
        backend, selection, revalidate, prepare = native_engine(target, run, facts)
        receipt["inputs"]["rtl_facts_identity"] = facts
    else:
        backend, selection, revalidate, prepare = None, None, None, None
    if run == "spike" and not spike_backend.available():
        raise BaremetalModelError("Spike simulator or bare-metal cross-toolchain is unavailable")
    return SelectRuntimeResult(
        native=native, facts=facts, backend=backend, selection=selection, revalidate=revalidate, prepare=prepare
    )


@dataclass(frozen=True)
class LaunchSavedResult:
    console: str | bytes


def launch_saved(
    *,
    backend,
    built,
    device,
    elf,
    elf_sha,
    extension_backend,
    full,
    marches,
    output_path,
    prepare,
    receipt,
    revalidate,
    run,
    selected,
    selection,
    target,
    timeout_s,
):
    """Execute one admitted image; retain partial process output on failure, never acceptance."""
    extension = extension_backend(target) if run == "spike" else None
    if backend is None and extension is not None:
        # The functional model WITH the target's accelerator extension: a device-routed image
        # issues accelerator instructions the plain simulator traps on.
        flags, libdir = extension.spike_extension()
        receipt["output"]["spike_extension"] = {"flags": list(flags), "library_dir": str(libdir)}
        try:
            console = extension.run_elf(elf, simulator="spike", timeout=timeout_s, capture_bytes=full)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            _retain_simulator_failure_output(exc, output_path, receipt)
            raise
    elif backend is None and full:
        if device is not None:
            raise BaremetalModelError("a device-routed image needs the target's spike extension")
        try:
            console = spike_model.run_raw(
                elf,
                harts=selected.harts,
                mem_bytes=built["mem_bytes"],
                isa=marches[0],
                timeout=timeout_s,
                vlen=built.get("vlen"),
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            _retain_simulator_failure_output(exc, output_path, receipt)
            raise
    elif backend is None:
        if device is not None:
            raise BaremetalModelError("a device-routed image needs the target's spike extension")
        try:
            result = spike_model.run(
                elf,
                harts=selected.harts,
                mem_bytes=built["mem_bytes"],
                isa=marches[0],
                timeout=timeout_s,
                vlen=built.get("vlen"),
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            _retain_simulator_failure_output(exc, output_path, receipt)
            raise
        console = str(result.get("console", ""))
    else:
        revalidate()
        command = None
        if run == "gsim":
            if not callable(prepare):
                raise BaremetalModelError("GSIM backend has no byte-bound command preparer")
            command = prepare(elf, expected_elf_sha256=elf_sha, expected_engine_provenance=selection["citation"])
            command_check = command.revalidate()
            command_evidence = command.to_evidence()
            receipt["output"]["native_command"] = {
                "schema": command_evidence["schema"],
                "command_sha256": command_check["command_sha256"],
                "emulator_argv": command_evidence["emulator_argv"],
                "max_cycles": command_evidence["max_cycles"],
            }
        if run == "gsim":
            from merlin.development.targetgen.rtl_engine_policy import gsim_runtime_slot

            slot = gsim_runtime_slot(wait_timeout_s=timeout_s)
        else:
            slot = nullcontext()
        with slot:
            try:
                console = backend.run_elf(elf, simulator=run, timeout=timeout_s, capture_bytes=full)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                _retain_simulator_failure_output(exc, output_path, receipt)
                raise
        if command is not None:
            command.revalidate()
        revalidate()
    return LaunchSavedResult(console=console)


def observe_profiles(
    *,
    console,
    full,
    group_profile,
    op_profile,
    output_path,
    profile_table,
    receipt,
):
    """Decode requested runtime observations against the actual emitted profile tables."""
    if op_profile:
        from merlin.runtime.operation_profile import summarize
        from merlin.runtime.out_bin import binary_console_diagnostics

        text = (
            binary_console_diagnostics(console).decode("utf-8", errors="replace")
            if full
            else (console.decode("utf-8", errors="replace") if isinstance(console, bytes) else console)
        )
        profile = summarize(json.loads(profile_table.read_text()), text)
        profile_path = output_path / "operation_profile.json"
        profile_path.write_text(json.dumps(profile, indent=2) + "\n")
        receipt["output"]["operation_profile"] = {"path": str(profile_path), "sha256": _sha(profile_path)}
    if group_profile:
        from merlin.runtime.whole_model_readback import parse_group_profile

        text = console.decode("utf-8", errors="replace") if isinstance(console, bytes) else console
        if full:
            from merlin.runtime.out_bin import binary_console_diagnostics

            text = binary_console_diagnostics(console).decode("utf-8", errors="replace")
        profile = parse_group_profile(text)
        if profile is None:
            raise BaremetalModelError("the group-profiled image printed no group profile")
        (output_path / "group_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
        receipt["output"]["group_profile"] = {
            "path": str(output_path / "group_profile.json"),
            "calls": profile["calls"],
            "dropped": profile["dropped"],
            "group_cycles": sum(row["cycles"] for row in profile["groups"]),
            "gap_cycles": sum(row["gap_before"] for row in profile["groups"]) + profile["tail_gap"],
        }
