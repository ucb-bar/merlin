"""Execute a provider-declared simulator without losing its machine contract.

The provider owns the ISA, devices, memory map and qualification. This helper
checks the declared bytes, runs an argv list with owned timeout cleanup, and
returns raw output plus a receipt; process exit never grants numerical or timing
certification.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


class DeviceSimulatorError(RuntimeError):
    """An explicitly selected simulator is unavailable or violates its contract."""


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_rich(descriptor: dict) -> bool:
    """Whether reducing this descriptor to binary+ISA would lose machine facts."""
    return any(
        k in descriptor
        for k in ("per_hart_isa", "launch_args", "memory_regions", "execution_input_sha256", "native_binary")
    )


def metadata(descriptor: dict) -> dict:
    """Keep provider qualification flags, including explicit false values."""
    keys = (
        "binary",
        "binary_sha256",
        "native_binary",
        "native_binary_sha256",
        "provenance",
        "per_hart_isa",
        "matrix_hart_ids",
        "memory_regions",
        "timing_available",
        "timing_status",
        "timing_scope",
        "execution_mode",
        "counter_semantics",
        "timing_calibrated",
        "accuracy_2percent_certified",
        "whole_model_certified",
        "full_model_certified",
        "rtl_certified",
        "hart2_timing_accuracy_certified",
        "qualification",
    )
    return {
        key: value
        for key, value in descriptor.items()
        if key in keys or key.endswith("_certified") or key.startswith("qualification")
    }


def execution_inputs(descriptor: dict) -> dict[str, str]:
    expected = descriptor.get("execution_input_sha256", {})
    if not isinstance(expected, dict):
        raise DeviceSimulatorError("execution_input_sha256 must be a mapping")
    expected = dict(expected)
    for key in ("binary", "native_binary"):
        path, digest = descriptor.get(key), descriptor.get(key + "_sha256")
        if key == "native_binary" and path is None and digest is None:
            continue
        if not isinstance(path, str) or not isinstance(digest, str):
            raise DeviceSimulatorError("Missing executable identity: " + key)
        if path in expected and expected[path] != digest:
            raise DeviceSimulatorError("Conflicting executable identity: " + key)
        expected[path] = digest
    check_inputs(expected)
    return expected


def check_inputs(inputs: dict[str, str]) -> None:
    for name, digest in inputs.items():
        if (
            not isinstance(name, str)
            or not Path(name).is_absolute()
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or not Path(name).is_file()
            or sha(name) != digest
        ):
            raise DeviceSimulatorError("Simulator input changed or missing: " + str(name))


def argv(
    descriptor: dict,
    elf: str | Path,
    *,
    isa: str,
    harts: int | None = None,
    active_hart_ids: tuple[int, ...] | None = None,
    memory: tuple[int, int] | None = None,
) -> list[str]:
    """Construct the exact machine; worker harts are distinct from physical harts."""
    if descriptor.get("available") is not True:
        raise DeviceSimulatorError(str(descriptor.get("reason") or "Explicit simulator unavailable"))
    per_hart = descriptor.get("per_hart_isa")
    if per_hart is not None:
        if (
            not isinstance(per_hart, dict)
            or not per_hart
            or set(per_hart) != {str(i) for i in range(len(per_hart))}
            or any(not isinstance(v, str) or not v for v in per_hart.values())
        ):
            raise DeviceSimulatorError("per_hart_isa must name every contiguous physical hart")
        if harts is not None and harts != len(per_hart):
            raise DeviceSimulatorError("Image CPU count differs from the simulator's physical hart count")
        harts = len(per_hart)
    if harts is None:
        harts = 1
    if type(harts) is not int or harts < 1:
        raise DeviceSimulatorError("Positive physical hart count required")
    allowed = descriptor.get("matrix_hart_ids")
    if allowed is not None:
        if (
            not isinstance(allowed, (list, tuple))
            or not allowed
            or len(set(allowed)) != len(allowed)
            or any(type(h) is not int or not 0 <= h < harts for h in allowed)
        ):
            raise DeviceSimulatorError("Invalid matrix_hart_ids")
        if active_hart_ids is None:
            raise DeviceSimulatorError("Explicit active_hart_ids required for a restricted matrix unit")
    if active_hart_ids is not None:
        if (
            not isinstance(active_hart_ids, (list, tuple))
            or not active_hart_ids
            or any(type(h) is not int for h in active_hart_ids)
            or len(set(active_hart_ids)) != len(active_hart_ids)
            or any(type(h) is not int or not 0 <= h < harts for h in active_hart_ids)
            or allowed is not None
            and any(h not in allowed for h in active_hart_ids)
        ):
            raise DeviceSimulatorError("Requested worker hart lacks the selected matrix unit")
    if not isinstance(isa, str) or not isa:
        raise DeviceSimulatorError("Explicit base ISA required")
    extensions = descriptor.get("isa_extensions", [])
    if not isinstance(extensions, list) or any(not isinstance(x, str) or not x for x in extensions):
        raise DeviceSimulatorError("isa_extensions must be a string list")
    machine_isa = "_".join(dict.fromkeys([*isa.split("_"), *extensions]))
    result = [descriptor["binary"], "--isa=" + machine_isa, "-p" + str(harts)]
    if memory is not None and (
        not isinstance(memory, (tuple, list))
        or len(memory) != 2
        or any(type(v) is not int for v in memory)
        or memory[0] < 0
        or memory[1] <= 0
        or sum(memory) > 1 << 64
    ):
        raise DeviceSimulatorError("Invalid image memory budget")
    regions = descriptor.get("memory_regions", [memory] if memory is not None else None)
    if regions is not None:
        if (
            not isinstance(regions, list)
            or not regions
            or any(
                not isinstance(r, (list, tuple))
                or len(r) != 2
                or any(type(v) is not int for v in r)
                or r[0] < 0
                or r[1] <= 0
                or r[0] + r[1] > 1 << 64
                for r in regions
            )
        ):
            raise DeviceSimulatorError("Invalid simulator memory regions")
        ordered = sorted(regions)
        if any(a[0] + a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise DeviceSimulatorError("Overlapping simulator memory regions")
        if memory is not None and not any(r[0] <= memory[0] and memory[0] + memory[1] <= r[0] + r[1] for r in regions):
            raise DeviceSimulatorError("Image DRAM budget is outside the declared simulator memory map")
        result.append("-m" + ",".join(f"{base:#x}:{size:#x}" for base, size in regions))
    arguments = descriptor.get("launch_args", [])
    if (
        not isinstance(arguments, list)
        or any(not isinstance(a, str) or not a or "\0" in a for a in arguments)
        or any(a.startswith(("--isa=", "--hart-isa=", "--hartids=", "-p", "-m")) for a in arguments)
    ):
        raise DeviceSimulatorError("Invalid or conflicting provider launch arguments")
    result.extend(arguments)
    if per_hart is not None:
        result.extend("--hart-isa=" + str(h) + ":" + per_hart[str(h)] for h in range(harts))
    result.append(str(Path(elf).resolve()))
    return result


def run(
    descriptor: dict,
    elf: str | Path,
    *,
    isa: str,
    harts: int | None = None,
    active_hart_ids: tuple[int, ...] | None = None,
    memory: tuple[int, int] | None = None,
    timeout: float = 300,
    output: Path | None = None,
) -> dict[str, Any]:
    """Execute with byte checks before/after and cleanup confined to this process group."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise DeviceSimulatorError("Finite timeout between zero and 86400 seconds required")
    elf = Path(elf).resolve()
    receipt = dict(
        schema="merlin.device_simulator_execution.v1",
        status="preflight",
        hardware_contacted=False,
        numerical_certification=False,
        timing_certification=False,
        simulator=metadata(descriptor),
    )
    output = Path(output).resolve() if output is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)

    def save():
        if output is not None:
            (output / "run.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")

    proc = None
    started = time.monotonic()
    save()
    try:
        inputs = execution_inputs(descriptor)
        inputs[str(Path(__file__).resolve())] = sha(__file__)
        inputs[str(elf)] = sha(elf)
        command = argv(descriptor, elf, isa=isa, harts=harts, active_hart_ids=active_hart_ids, memory=memory)
        env = dict(os.environ)
        overrides = descriptor.get("environment", {})
        if not isinstance(overrides, dict) or any(
            not isinstance(k, str) or v is not None and not isinstance(v, str) for k, v in overrides.items()
        ):
            raise DeviceSimulatorError("Invalid simulator environment")
        for name, value in overrides.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        receipt.update(
            status="running",
            argv=command,
            input_sha256=inputs,
            ELF_sha256=inputs[str(elf)],
            active_hart_ids=list(active_hart_ids) if active_hart_ids is not None else None,
            timeout_seconds=timeout,
        )
        save()
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(0.2)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        if output is not None:
            (output / "stdout").write_bytes(stdout)
            (output / "stderr").write_bytes(stderr)
            receipt["output_sha256"] = {name: sha(output / name) for name in ("stdout", "stderr")}
        receipt.update(
            status="passed_guest_exit" if proc.returncode == 0 and not timed_out else "failed_guest_exit",
            returncode=proc.returncode,
            timed_out=timed_out,
        )
        check_inputs(inputs)
        receipt.update(unchanged_inputs_after_execution=True, host_seconds=time.monotonic() - started)
        return {
            **receipt,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "receipt": str(output / "run.json") if output is not None else None,
        }
    except BaseException as exc:
        receipt.update(status="refused_or_failed", error=str(exc))
        raise
    finally:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        receipt["host_seconds"] = time.monotonic() - started
        save()
