"""Execute an existing native Linux ELF with explicit initial CPU affinity.

Header corroboration and observed process exit establish no numerical, device,
ISA-extension or exclusive-resource qualification. No build or simulator occurs.
"""

from __future__ import annotations

import base64
import json
import math
import os
import signal
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from merlin.compiler.artifact import Artifact

from .service import RunRequest, RunResult

name = "merlin.linux"


def _header(path):
    """Corroborate ELF header fields; the kernel loader still admits actual execution."""
    with path.open("rb") as stream:
        header = stream.read(64)
    if (
        len(header) < 52
        or header[:4] != b"\x7fELF"
        or header[4] not in (1, 2)
        or header[5:7] != b"\x01\x01"
        or header[7] not in (0, 3)
        or header[8] != 0
    ):
        raise ValueError("native Linux execution requires a supported little-endian ELF header")
    bits = {1: 32, 2: 64}[header[4]]
    if bits == 64 and len(header) < 64:
        raise ValueError("truncated native Linux ELF64 header")
    kind, machine, version = struct.unpack_from("<HHI", header, 16)
    entry, phoff = struct.unpack_from("<QQ" if bits == 64 else "<II", header, 24)
    flags, size, phsize, count = struct.unpack_from("<IHHH", header, 48 if bits == 64 else 36)
    if (
        kind not in (2, 3)
        or version != 1
        or not entry
        or size != (64 if bits == 64 else 52)
        or phsize != (56 if bits == 64 else 32)
        or not count
        or phoff < size
        or phoff + phsize * count > path.stat().st_size
    ):
        raise ValueError("native Linux input requires an executable ELF header and program table")
    return {"bits": bits, "machine": machine, "flags": flags, "type": kind}


def _selection(request):
    if sys.platform != "linux" or not hasattr(os, "sched_getaffinity") or os.execve not in os.supports_fd:
        raise ValueError("merlin.linux requires native Linux affinity and file-descriptor exec support")
    request.deployment.__post_init__()
    if request.deployment.platform != "linux" or request.artifact.kind != "elf":
        raise ValueError("merlin.linux requires an ELF and an explicit Linux Deployment")
    if (
        type(request.harts) is not int
        or request.harts != request.deployment.harts
        or not 1 <= request.harts <= 1024
        or isinstance(request.timeout, bool)
        or not isinstance(request.timeout, (int, float))
        or not math.isfinite(request.timeout)
        or request.timeout <= 0
    ):
        raise ValueError("Linux execution requires matching bounded harts and a positive finite timeout")
    if type(request.arguments) is not tuple or any(type(v) is not str or "\x00" in v for v in request.arguments):
        raise ValueError("Linux process arguments must be strings without NUL")
    request.artifact.verify()
    if not os.access(request.artifact.path, os.X_OK):
        raise ValueError("selected Linux ELF must already be executable")
    image, host = _header(request.artifact.path), _header(Path("/proc/self/exe"))
    machine = os.uname().machine
    if (image["bits"], image["machine"]) != (host["bits"], host["machine"]):
        raise ValueError("Linux ELF architecture differs from the executing host")
    deployment = request.deployment
    if machine == "x86_64" and image["machine"] == 62 and image["bits"] == 64:
        if deployment.isa != "x86_64" or deployment.abi != "sysv" or image["flags"] != 0:
            raise ValueError("Linux ELF differs from the selected x86_64/sysv ABI")
    elif machine in {"riscv32", "riscv64"} and image["machine"] == 243:
        abis = {32: {"ilp32": 0, "ilp32f": 2, "ilp32d": 4}, 64: {"lp64": 0, "lp64f": 2, "lp64d": 4, "lp64q": 6}}
        if (
            not deployment.isa.startswith(f"rv{image['bits']}")
            or deployment.abi not in abis[image["bits"]]
            or image["flags"] & 8
            or image["flags"] & 6 != abis[image["bits"]][deployment.abi]
            or image["flags"] & 6 != host["flags"] & 6
        ):
            raise ValueError("Linux ELF differs from the executing host and selected RISC-V psABI")
    else:
        raise ValueError("this executing Linux architecture has no admitted ELF header/ABI route")
    available = sorted(os.sched_getaffinity(0))
    if request.harts > len(available):
        raise ValueError("Linux hart request exceeds the executing process's available CPU affinity")
    return image, tuple(available[: request.harts])


def validate(request: RunRequest) -> None:
    _selection(request)


def _stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _execute(command, request, descriptors):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=request.artifact.path.parent,
        pass_fds=descriptors,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=request.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop_group(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.output or b"", error.stderr or b""
            process.stdout.close()
            process.stderr.close()
            process.wait(timeout=5)
    except BaseException:
        _stop_group(process)
        process.communicate(timeout=5)
        raise
    return process, stdout, stderr, timed_out


def run(request: RunRequest) -> RunResult:
    image, cpus = _selection(request)
    worker = Artifact.from_file(Path(__file__).with_name("linux_worker.py"), kind="python", role="runtime_worker")
    interpreter = Artifact.from_file(Path(sys.executable), kind="tool", role="python")
    image_fd = os.open(request.artifact.path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        # A separate unlinked file keeps worker observations out of program
        # streams and cannot block a large affinity observation behind a pipe.
        with tempfile.TemporaryFile() as report:
            command = [
                str(interpreter.path),
                "-I",
                "-B",
                str(worker.path),
                str(image_fd),
                str(report.fileno()),
                request.artifact.sha256,
                json.dumps(cpus),
                str(request.artifact.path),
                *request.arguments,
            ]
            worker.verify()
            interpreter.verify()
            request.artifact.verify()
            process, stdout, stderr, timed_out = _execute(command, request, (image_fd, report.fileno()))
            report.seek(0)
            observations = report.read(32769)
    finally:
        os.close(image_fd)
    worker.verify()
    interpreter.verify()
    request.artifact.verify()
    if len(observations) > 32768:
        raise ValueError("Linux worker observations exceed the bounded protocol")
    records = [json.loads(line) for line in observations.splitlines()]
    ready = next((row for row in records if row.get("stage") == "exec_requested"), None)
    failed = any(row.get("stage") == "exec_failed" for row in records)
    if ready is not None and (
        ready.get("pid") != process.pid
        or ready.get("affinity") != list(cpus)
        or ready.get("image_sha256") != request.artifact.sha256
    ):
        raise ValueError("Linux execution observations differ from the selected image or CPU affinity")
    if process.returncode == 0 and (ready is None or failed):
        raise ValueError("Linux success requires the original worker's exec observation")
    return RunResult(
        name,
        request.artifact.sha256,
        "timed_out" if timed_out else "completed" if process.returncode == 0 else "failed",
        stdout.decode("utf-8", errors="replace"),
        {
            "exit_code": process.returncode,
            "stderr": stderr.decode("utf-8", errors="replace"),
            "stdout_base64": base64.b64encode(stdout).decode("ascii"),
            "stderr_base64": base64.b64encode(stderr).decode("ascii"),
            "argv": [str(request.artifact.path), *request.arguments],
            "elf_header": image,
            "harts": request.harts,
            "requested_cpu_affinity": list(cpus),
            "initial_cpu_affinity": ready["affinity"] if ready is not None else None,
            "harts_meaning": "initial CPU affinity count; no thread count or exclusive reservation",
            "worker": {"path": str(worker.path), "sha256": worker.sha256},
            "interpreter": {"path": str(interpreter.path), "sha256": interpreter.sha256},
            "process_observations": records,
            "timeout_process_group_cleanup": timed_out,
            "verification": "not_requested",
            "device_execution_verified": False,
            "admission_scope": "native Linux ELF header/ABI and process observations; ISA extensions unqualified",
        },
    )
