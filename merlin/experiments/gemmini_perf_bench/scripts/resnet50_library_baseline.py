#!/usr/bin/env python3
"""Build and measure Gemmini's hand-written ResNet-50 library baseline on FireSim.

The upstream ``imagenet/resnet50.c`` accepts ``os|ws|cpu`` and ``conv|matmul`` on argv, but a
bare-metal FireSim workload receives no argv.  This driver snapshots the upstream sources and builds
one tiny wrapper translation unit per requested arm.  Each wrapper calls the unmodified upstream
program with a fixed argv, so all arms use the same model, parameters, compiler flags and timing
decomposition without editing the external checkout.

The default comparison is the one needed to attribute Merlin's whole-model gap:

* ``ws_conv`` — the library's native LOOP_CONV_WS path;
* ``ws_matmul`` — the library using im2col + the same matmul strategy as Merlin;
* ``cpu_matmul`` — the built-in no-accelerator control.

Every queued execution performs one unmeasured warm-up inference and then one measured inference in
the same process.  It pins the ELF, source snapshot, compiler, Chipyard checkout, FireSim hardware
config, queue job and raw UART.  A timing is admitted only when both phases print ``PASS``, every
declared measured cycle bucket is present, and the buckets sum exactly to ``Total cycles``.  The
queue's atomic ``kill -> infrasetup -> runworkload -> kill`` operation is mandatory; this driver
never invokes FireSim directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_HERE = Path(__file__).resolve()
_root = os.environ.get("MERLIN_REPO_ROOT", "").strip()
if not _root:
    _root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=_HERE.parent,
        capture_output=True, text=True, check=False,
    ).stdout.strip()
REPO = Path(_root).expanduser().resolve() if _root else _HERE.parents[4]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.common.paths import ext_path  # noqa: E402


ARM_ARGS: dict[str, tuple[str, str]] = {
    f"{dataflow}_{lowering}": (dataflow, lowering)
    for dataflow in ("os", "ws", "cpu")
    for lowering in ("conv", "matmul")
}
DEFAULT_ARMS = ("ws_conv", "ws_matmul", "cpu_matmul")
DEFAULT_HW_CONFIG = "alveo_u250_firesim_shuttle_gemmini_opu"
DEFAULT_REPETITIONS = 3
WORKLOAD = "merlin-perfbench"
BOOTBINARY = "merlin-perfbench.elf"
QUEUE_CWD_LAUNCHER = _HERE.with_name("firesim_queue_cwd_launcher") / "firesim"
FIRESIM_MAKE_LAUNCHER = QUEUE_CWD_LAUNCHER.with_name("make")
FIRESIM_LIFECYCLE = (
    "firesim kill", "firesim infrasetup", "firesim runworkload", "firesim kill",
)
_WARM_BEGIN = "MERLIN_PROFILE warmup begin"
_WARM_END = "MERLIN_PROFILE warmup end rc=0"
_MEASURED_BEGIN = "MERLIN_PROFILE measured begin"
_MEASURED_END = "MERLIN_PROFILE measured end rc=0"

_JOB_RE = re.compile(r"job_id=(\d+)")
_CYCLE_FIELDS = {
    "total": "Total cycles",
    "matmul": "Matmul cycles",
    "im2col": "Im2col cycles",
    "conv": "Conv cycles",
    "pooling": "Pooling cycles",
    "depthwise_conv": "Depthwise convolution cycles",
    "res_add": "Res add cycles",
    "other": "Other cycles",
}
_CYCLE_RES = {
    key: re.compile(rf"^{re.escape(label)}:\s*(\d+)\s*\(", re.MULTILINE)
    for key, label in _CYCLE_FIELDS.items()
}
_COMPONENTS = tuple(key for key in _CYCLE_FIELDS if key != "total")


class BaselineError(RuntimeError):
    """The baseline could not produce an attributable measurement."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _profile_phase(text: str, begin: str, end: str) -> str:
    """Return one explicitly delimited profile phase, refusing duplicates or reordered markers."""
    if text.count(begin) != 1 or text.count(end) != 1:
        raise BaselineError(f"UART does not contain exactly one {begin!r}/{end!r} pair")
    before, separator, tail = text.partition(begin)
    if not separator:
        raise BaselineError(f"UART is missing {begin!r}")
    body, separator, after = tail.partition(end)
    if not separator or begin in before or begin in body or end in after:
        raise BaselineError(f"UART has malformed {begin!r}/{end!r} boundaries")
    return body


def parse_uart(text: str) -> dict[str, Any]:
    """Parse only the post-warm-up compute-cycle decomposition, failing closed on any drift."""
    warm = _profile_phase(text, _WARM_BEGIN, _WARM_END)
    measured = _profile_phase(text, _MEASURED_BEGIN, _MEASURED_END)
    if text.index(_WARM_END) > text.index(_MEASURED_BEGIN):
        raise BaselineError("measured inference began before warm-up completed")
    if re.search(r"^FAIL\s*$", warm, re.MULTILINE):
        raise BaselineError("upstream ResNet-50 warm-up reported FAIL")
    if not re.search(r"^PASS\s*$", warm, re.MULTILINE):
        raise BaselineError("upstream ResNet-50 warm-up did not report PASS")
    if re.search(r"^FAIL\s*$", measured, re.MULTILINE):
        raise BaselineError("upstream ResNet-50 reported FAIL")
    if not re.search(r"^PASS\s*$", measured, re.MULTILINE):
        raise BaselineError("upstream ResNet-50 did not report PASS")

    cycles: dict[str, int] = {}
    for key, pattern in _CYCLE_RES.items():
        matches = pattern.findall(measured)
        if len(matches) > 1:
            raise BaselineError(f"measured phase repeats {_CYCLE_FIELDS[key]!r}")
        match = pattern.search(measured)
        if match is None:
            raise BaselineError(f"UART is missing {_CYCLE_FIELDS[key]!r}")
        cycles[key] = int(match.group(1))
    component_sum = sum(cycles[key] for key in _COMPONENTS)
    if component_sum != cycles["total"]:
        raise BaselineError(
            f"cycle buckets sum to {component_sum}, not Total cycles {cycles['total']}")
    return {
        "profile": {"warmup_runs": 1, "measured_runs": 1,
                    "recorded_scope": "post-warm-up compute-cycle decomposition"},
        "cycles": cycles,
        "component_sum": component_sum,
        "component_percent": {
            key: (100.0 * cycles[key] / cycles["total"] if cycles["total"] else 0.0)
            for key in _COMPONENTS
        },
    }


def render_wrapper(arm: str) -> str:
    """Return a wrapper that fixes argv and executes one warm plus one measured inference."""
    try:
        dataflow, lowering = ARM_ARGS[arm]
    except KeyError as exc:
        raise BaselineError(f"unknown arm {arm!r}; choose from {sorted(ARM_ARGS)}") from exc
    return f'''/* Generated by resnet50_library_baseline.py; upstream source remains unmodified. */
#define main merlin_resnet50_upstream_main
#include "../source/imagenet/resnet50_profiled.c"
#undef main

int main(void) {{
    char arg0[] = "resnet50";
    char arg1[] = "{dataflow}";
    char arg2[] = "{lowering}";
    char *argv[] = {{arg0, arg1, arg2, 0}};
    printf("{_WARM_BEGIN}\\n");
    int warmup_rc = merlin_resnet50_upstream_main(3, argv);
    printf("MERLIN_PROFILE warmup end rc=%d\\n", warmup_rc);
    if (warmup_rc != 0) return warmup_rc;
    printf("{_MEASURED_BEGIN}\\n");
    int measured_rc = merlin_resnet50_upstream_main(3, argv);
    printf("MERLIN_PROFILE measured end rc=%d\\n", measured_rc);
    return measured_rc;
}}
'''


def render_profile_source(upstream: str) -> str:
    """Make the upstream entrypoint return after success so the wrapper can invoke it twice.

    The original snapshot stays byte-identical.  Only the terminal success ``exit(0)`` is replaced;
    all early error exits remain fail-fast.  Refuse if the exact source seam is not unique.
    """
    terminal = '    printf("PASS\\n");\n    exit(0);\n}\n'
    replacement = '    printf("PASS\\n");\n    return 0;\n}\n'
    if upstream.count(terminal) != 1:
        raise BaselineError("upstream ResNet-50 terminal success seam is absent or ambiguous")
    return upstream.replace(terminal, replacement)


def _source_paths(source_root: Path) -> list[Path]:
    required = [
        source_root / "imagenet/resnet50.c",
        source_root / "imagenet/resnet50_params.h",
        source_root / "imagenet/images.h",
        source_root / "riscv-tests/benchmarks/common/crt.S",
        source_root / "riscv-tests/benchmarks/common/syscalls.c",
        source_root / "riscv-tests/benchmarks/common/test.ld",
        source_root / "riscv-tests/benchmarks/common/util.h",
        source_root / "riscv-tests/env/encoding.h",
    ]
    required.extend(sorted((source_root / "include").glob("*.h")))
    required.extend(sorted((source_root / "rocc-software/src").glob("*.h")))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise BaselineError("Gemmini source tree is incomplete: " + ", ".join(missing))
    return sorted(set(path.resolve() for path in required))


def _git_provenance(path: Path) -> dict[str, Any]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True,
                              check=False)

    head = git("rev-parse", "HEAD")
    if head.returncode:
        return {"root": str(path.resolve()), "is_git": False}
    top = git("rev-parse", "--show-toplevel").stdout.strip()
    status = git("status", "--porcelain=v1", "--untracked-files=no").stdout
    diff = git("diff", "--binary", "HEAD").stdout.encode()
    return {
        "root": top,
        "is_git": True,
        "head": head.stdout.strip(),
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "tracked_diff_sha256": _sha256_bytes(diff),
    }


def snapshot_sources(source_root: Path, destination: Path) -> dict[str, Any]:
    """Copy the build inputs into the run so a later live-checkout change cannot alter an arm."""
    source_root = source_root.expanduser().resolve()
    files = _source_paths(source_root)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for source in files:
        relative = source.relative_to(source_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        source_hash = _sha256(source)
        snapshot_hash = _sha256(target)
        if snapshot_hash != source_hash:
            raise BaselineError(f"source snapshot drift for {relative}")
        manifest.append({"path": relative.as_posix(), "sha256": snapshot_hash,
                         "bytes": target.stat().st_size})
    receipt = {
        "external_root": str(source_root),
        "external_git": _git_provenance(source_root),
        "files": manifest,
        "aggregate_sha256": _sha256_bytes(
            "".join(f"{row['path']}\0{row['sha256']}\n" for row in manifest).encode()),
    }
    _write_json(destination.parent / "source_manifest.json", receipt)
    return receipt


def resolve_compiler(source_root: Path, requested: str | None) -> Path:
    candidates: list[Path] = []
    if requested:
        candidates.append(Path(requested).expanduser())
    configured = os.environ.get("MERLIN_RISCV_GCC", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    if found := shutil.which("riscv64-unknown-elf-gcc"):
        candidates.append(Path(found))
    for parent in (source_root, *source_root.parents):
        candidates.append(parent / ".conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise BaselineError(
        "riscv64-unknown-elf-gcc not found; pass --compiler or set MERLIN_RISCV_GCC")


def build_command(compiler: Path, snapshot: Path, wrapper: Path, elf: Path) -> list[str]:
    common = snapshot / "riscv-tests/benchmarks/common"
    return [
        str(compiler),
        "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany", "-std=gnu99", "-O2",
        "-ffast-math", "-fno-common", "-fno-builtin-printf",
        "-fno-tree-loop-distribute-patterns", "-march=rv64gc", "-Wa,-march=rv64gc",
        "-lm", "-lgcc", f"-I{snapshot / 'riscv-tests'}", f"-I{snapshot / 'riscv-tests/env'}",
        f"-I{snapshot}", f"-I{common}", "-DID_STRING=", "-Wno-incompatible-pointer-types",
        "-nostdlib", "-nostartfiles", "-static", "-T", str(common / "test.ld"),
        "-DBAREMETAL=1", str(wrapper), "-o", str(elf), str(common / "syscalls.c"),
        str(common / "crt.S"),
    ]


def build_arm(arm: str, run_dir: Path, compiler: Path, source_manifest: Mapping[str, Any]) -> dict:
    wrappers = run_dir / "wrappers"
    elfs = run_dir / "elfs"
    wrappers.mkdir(parents=True, exist_ok=True)
    elfs.mkdir(parents=True, exist_ok=True)
    upstream = run_dir / "source/imagenet/resnet50.c"
    profiled = run_dir / "source/imagenet/resnet50_profiled.c"
    expected_profiled = render_profile_source(upstream.read_text(encoding="utf-8"))
    if profiled.is_file() and profiled.read_text(encoding="utf-8") != expected_profiled:
        raise BaselineError(f"refusing to overwrite drifted profile source {profiled}")
    if not profiled.exists():
        profiled.write_text(expected_profiled, encoding="utf-8")
    wrapper = wrappers / f"{arm}.c"
    expected_wrapper = render_wrapper(arm)
    if wrapper.is_file() and wrapper.read_text(encoding="utf-8") != expected_wrapper:
        raise BaselineError(f"refusing to overwrite drifted wrapper {wrapper}")
    if not wrapper.exists():
        wrapper.write_text(expected_wrapper, encoding="utf-8")
    elf = elfs / f"resnet50_{arm}.elf"
    command = build_command(compiler, run_dir / "source", wrapper, elf)
    started = time.time()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    build_log = run_dir / "build_logs" / f"{arm}.log"
    build_log.parent.mkdir(parents=True, exist_ok=True)
    build_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode or not elf.is_file():
        raise BaselineError(f"build failed for {arm}; see {build_log}")
    if elf.read_bytes()[:4] != b"\x7fELF":
        raise BaselineError(f"compiler did not produce an ELF for {arm}")
    version = subprocess.run([str(compiler), "--version"], capture_output=True, text=True,
                             check=False).stdout.splitlines()
    receipt = {
        "arm": arm,
        "argv": list(ARM_ARGS[arm]),
        "source_aggregate_sha256": source_manifest["aggregate_sha256"],
        "upstream_resnet50_sha256": _sha256(upstream),
        "profile_source_sha256": _sha256(profiled),
        "profile_source_transform": "terminal success exit(0) -> return 0; warm+measured in wrapper",
        "wrapper_sha256": _sha256(wrapper),
        "compiler": str(compiler),
        "compiler_sha256": _sha256(compiler),
        "compiler_version": version[:2],
        "command": command,
        "returncode": completed.returncode,
        "build_wall_s": round(time.time() - started, 3),
        "elf": str(elf.resolve()),
        "elf_sha256": _sha256(elf),
        "elf_bytes": elf.stat().st_size,
    }
    _write_json(run_dir / "build_receipts" / f"{arm}.json", receipt)
    return receipt


def queue_command(*, queue: Path, chipyard: Path, elf: Path, hw_config: str,
                  timeout: int, priority: int) -> list[str]:
    return [
        str(queue), "runworkload-full", "--chipyard", str(chipyard),
        "--workload", WORKLOAD, "--bootbinary", BOOTBINARY,
        "--stage-from", str(elf), "--hw-config", hw_config,
        "--priority", str(priority), "--project", "merlin-resnet50-baseline",
        "--timeout", str(timeout),
    ]


def queue_client_environment() -> dict[str, str]:
    """Let a cross-user queue daemon retain its own account identity.

    ``runworkload-full`` intentionally forwards HOME/USER/LOGNAME when clients supply them.  That is
    correct for a same-user daemon, but a shared daemon cannot read another user's private conda
    configuration.  Omitting only those identity variables makes the daemon keep its own readable
    home while the queue still derives the submitting user from the client's uid via getpass.
    """
    for launcher in (QUEUE_CWD_LAUNCHER, FIRESIM_MAKE_LAUNCHER):
        if launcher.is_symlink() or not launcher.is_file() or not os.access(launcher, os.X_OK):
            raise BaselineError(
                f"FireSim queue launcher is missing, symlinked, or not executable: {launcher}")
    env = dict(os.environ)
    for name in ("HOME", "USER", "LOGNAME"):
        env.pop(name, None)
    # A cross-user queue daemon selects a writable per-job deploy overlay as
    # its cwd.  Older queue daemons nevertheless resolve the bare `firesim`
    # command from the submitter's original deploy directory; FireSim then
    # chdirs back there based on __file__ and defeats the overlay.  Put a tiny
    # pinned launcher first in PATH so `firesim` means `./firesim`, preserving
    # the daemon-selected cwd for logs, workload staging, and results.
    inherited_path = env.get("PATH", "")
    env["PATH"] = str(QUEUE_CWD_LAUNCHER.parent) + (
        os.pathsep + inherited_path if inherited_path else "")
    return env


def validate_queue_help(text: str) -> dict[str, Any]:
    """Prove the selected queue operation owns the required atomic FireSim lifecycle."""
    compact = " ".join(text.split())
    required = "kill -> infrasetup -> runworkload -> kill sequence"
    if required not in compact:
        raise BaselineError(
            "FireSim queue runworkload-full help does not declare the required " + required)
    return {"queue_operation": "runworkload-full", "firesim_lifecycle": list(FIRESIM_LIFECYCLE),
            "contract_help_sha256": _sha256_bytes(text.encode("utf-8"))}


def inspect_queue_contract(queue: Path) -> dict[str, Any]:
    # argparse prints a subparser's descriptive contract in the top-level help,
    # while the subcommand help contains its concrete arguments.  Preserve and
    # validate both: the former proves lifecycle ownership; the latter proves
    # that this installed queue accepts the atomic operation we will submit.
    top_level = subprocess.run(
        [str(queue), "--help"], capture_output=True, text=True, check=False)
    operation = subprocess.run(
        [str(queue), "runworkload-full", "--help"], capture_output=True, text=True, check=False)
    if top_level.returncode or operation.returncode:
        raise BaselineError("FireSim queue could not describe runworkload-full")
    return validate_queue_help(
        top_level.stdout + top_level.stderr + operation.stdout + operation.stderr)


def _queue_job_log(queue: Path, job_id: int) -> Path:
    path = queue.parent.parent / "jobs" / str(job_id) / "stdout.log"
    if path.is_symlink() or not path.is_file():
        raise BaselineError(f"FireSim queue job {job_id} has no plain daemon stdout log")
    return path.resolve()


def validate_queue_phases(text: str) -> list[str]:
    """Require queue evidence for the locked lifecycle phases in execution order."""
    labels = ("STAGING", "INFRASETUP", "RUNNING", "TEARDOWN")
    cursor = 0
    for label in labels:
        marker = f"=== [firesim-queue] phase={label}"
        index = text.find(marker, cursor)
        if index < 0:
            raise BaselineError(f"queue daemon log is missing ordered phase {label}")
        cursor = index + len(marker)
    return list(labels)


def _find_uart(queue: Path, chipyard: Path, job_id: int) -> Path:
    """Find UART in either the queue's cross-user overlay or the native deploy tree."""
    roots = (
        queue.parent.parent / "jobs" / str(job_id) / "deploy_overlay/results-workload",
        chipyard / "sims/firesim/deploy/results-workload",
    )
    for results_root in roots:
        candidates = sorted(
            results_root.glob(f"*-{WORKLOAD}-q{job_id}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for directory in candidates:
            uart = directory / f"{WORKLOAD}0/uartlog"
            if uart.is_file():
                return uart
    raise BaselineError(f"no per-job UART found for FireSim queue job {job_id}")


def run_arm(arm: str, repetition: int, run_dir: Path, build: Mapping[str, Any], *, queue: Path,
            chipyard: Path, hw_config: str, timeout: int, priority: int) -> dict[str, Any]:
    elf = Path(str(build["elf"]))
    if _sha256(elf) != build["elf_sha256"]:
        raise BaselineError(f"ELF drift before run for {arm}")
    command = queue_command(queue=queue, chipyard=chipyard, elf=elf, hw_config=hw_config,
                            timeout=timeout, priority=priority)
    started = time.time()
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout + 600, check=False, env=queue_client_environment())
    queue_log = run_dir / "queue_logs" / f"{arm}.rep-{repetition:02d}.log"
    queue_log.parent.mkdir(parents=True, exist_ok=True)
    queue_log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode or "terminal state=DONE" not in completed.stdout:
        raise BaselineError(f"FireSim queue job for {arm} did not finish DONE; see {queue_log}")
    match = _JOB_RE.search(completed.stdout)
    if match is None:
        raise BaselineError(f"FireSim queue did not report a job id for {arm}")
    job_id = int(match.group(1))
    source_uart = _find_uart(queue, chipyard, job_id)
    uart_text = source_uart.read_text(encoding="utf-8", errors="replace")
    uart = run_dir / "uarts" / f"{arm}.rep-{repetition:02d}.uartlog"
    uart.parent.mkdir(parents=True, exist_ok=True)
    uart.write_text(uart_text, encoding="utf-8")
    parsed = parse_uart(uart_text)
    daemon_log = _queue_job_log(queue, job_id)
    phases = validate_queue_phases(daemon_log.read_text(encoding="utf-8", errors="replace"))
    queue_job_root = queue.parent.parent / "jobs" / str(job_id)
    runtime_config = queue_job_root / "config_runtime.yaml"
    expected_simulation_dir = str((queue_job_root / "simulation").resolve())
    if runtime_config.is_symlink() or not runtime_config.is_file():
        raise BaselineError(f"queue job {job_id} has no plain runtime config")
    simulation_fields = [
        line.split(":", 1)[1].strip()
        for line in runtime_config.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("default_simulation_dir:")
    ]
    if simulation_fields != [expected_simulation_dir]:
        raise BaselineError(
            f"queue job {job_id} did not isolate its simulation directory: {simulation_fields}")
    result = {
        "status": "pass",
        "arm": arm,
        "repetition": repetition,
        "argv": list(ARM_ARGS[arm]),
        "job_id": job_id,
        "hw_config": hw_config,
        "workload": WORKLOAD,
        "bootbinary": BOOTBINARY,
        "queue_command": command,
        "queue_wall_s": round(time.time() - started, 3),
        "queue_phases": phases,
        "firesim_lifecycle": list(FIRESIM_LIFECYCLE),
        "queue_daemon_log": str(daemon_log),
        "queue_daemon_log_sha256": _sha256(daemon_log),
        "queue_runtime_config": str(runtime_config.resolve()),
        "queue_runtime_config_sha256": _sha256(runtime_config),
        "queue_simulation_dir": expected_simulation_dir,
        "elf_sha256": build["elf_sha256"],
        "uart": str(uart.resolve()),
        "uart_sha256": _sha256(uart),
        **parsed,
    }
    _write_json(run_dir / "run_receipts" / f"{arm}.rep-{repetition:02d}.json", result)
    return result


def aggregate_arm(repetitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the median-total repetition, retaining its internally exact cycle decomposition."""
    if not repetitions or any(row.get("status") != "pass" for row in repetitions):
        raise BaselineError("cannot aggregate an absent or failed repetition")
    ranked = sorted(repetitions, key=lambda row: (
        int(row["cycles"]["total"]), int(row["repetition"])))
    representative = ranked[(len(ranked) - 1) // 2]
    totals = [int(row["cycles"]["total"]) for row in repetitions]
    return {
        "status": "pass", "arm": representative["arm"],
        "repetition_count": len(repetitions),
        "representative_repetition": representative["repetition"],
        "selection": "median total compute cycles; decomposition from that exact repetition",
        "cycles": dict(representative["cycles"]),
        "component_sum": representative["component_sum"],
        "component_percent": dict(representative["component_percent"]),
        "total_cycle_distribution": {
            "values": totals, "min": min(totals), "max": max(totals),
            "median": statistics.median(totals),
        },
        "repetitions": list(repetitions),
    }


def load_repetition_receipt(arm: str, repetition: int, run_dir: Path,
                            build: Mapping[str, Any], *, queue: Path, chipyard: Path,
                            hw_config: str, timeout: int, priority: int) -> dict[str, Any] | None:
    """Adopt a completed repetition only after revalidating every mutable input and result."""
    path = run_dir / "run_receipts" / f"{arm}.rep-{repetition:02d}.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise BaselineError(f"resume receipt is not a plain file: {path}")
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"resume receipt is unreadable: {path}") from exc
    elf = Path(str(build["elf"]))
    expected_command = queue_command(
        queue=queue, chipyard=chipyard, elf=elf, hw_config=hw_config,
        timeout=timeout, priority=priority)
    if (row.get("status"), row.get("arm"), row.get("repetition")) != (
            "pass", arm, repetition):
        raise BaselineError(f"resume receipt identity/status drifted: {path}")
    if (row.get("elf_sha256") != build["elf_sha256"]
            or _sha256(elf) != build["elf_sha256"]
            or row.get("hw_config") != hw_config
            or row.get("queue_command") != expected_command
            or row.get("firesim_lifecycle") != list(FIRESIM_LIFECYCLE)):
        raise BaselineError(f"resume receipt inputs drifted: {path}")
    uart = Path(str(row.get("uart") or ""))
    daemon_log = Path(str(row.get("queue_daemon_log") or ""))
    runtime_config = Path(str(row.get("queue_runtime_config") or ""))
    if (uart.is_symlink() or not uart.is_file() or _sha256(uart) != row.get("uart_sha256")):
        raise BaselineError(f"resume UART drifted: {path}")
    if (daemon_log.is_symlink() or not daemon_log.is_file()
            or _sha256(daemon_log) != row.get("queue_daemon_log_sha256")):
        raise BaselineError(f"resume queue daemon log drifted: {path}")
    if (runtime_config.is_symlink() or not runtime_config.is_file()
            or _sha256(runtime_config) != row.get("queue_runtime_config_sha256")):
        raise BaselineError(f"resume queue runtime config drifted: {path}")
    parsed = parse_uart(uart.read_text(encoding="utf-8", errors="replace"))
    if (parsed.get("cycles") != row.get("cycles")
            or validate_queue_phases(
                daemon_log.read_text(encoding="utf-8", errors="replace"))
            != row.get("queue_phases")):
        raise BaselineError(f"resume measurement does not reproduce: {path}")
    return row


def comparison(results: Mapping[str, Mapping[str, Any]]) -> dict[str, float | int]:
    """Compute only comparisons supported by like-for-like successful arms."""
    output: dict[str, float | int] = {}
    if "ws_conv" in results and "ws_matmul" in results:
        conv = int(results["ws_conv"]["cycles"]["total"])
        matmul = int(results["ws_matmul"]["cycles"]["total"])
        output["ws_matmul_over_ws_conv"] = matmul / conv
        output["ws_conv_minus_ws_matmul_cycles"] = conv - matmul
        output["ws_native_conv_cycles"] = int(results["ws_conv"]["cycles"]["conv"])
        output["ws_explicit_im2col_cycles"] = int(results["ws_matmul"]["cycles"]["im2col"])
    if "cpu_matmul" in results and "ws_matmul" in results:
        cpu = int(results["cpu_matmul"]["cycles"]["total"])
        ws = int(results["ws_matmul"]["cycles"]["total"])
        output["cpu_matmul_over_ws_matmul"] = cpu / ws
    return output


def _parse_arms(value: str) -> list[str]:
    arms = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(arms) - set(ARM_ARGS))
    if unknown:
        raise BaselineError(f"unknown arm(s) {unknown}; choose from {sorted(ARM_ARGS)}")
    if len(set(arms)) != len(arms):
        raise BaselineError("each arm may be named only once")
    return arms


def _default_out() -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return REPO / f"out/artifacts/perf-bench/gemmini/resnet50_library_baseline_{timestamp}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemmini-rocc-tests", required=True, type=Path,
                        help="external gemmini-rocc-tests checkout to snapshot")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--compiler", default=None)
    parser.add_argument("--chipyard", type=Path, default=None)
    parser.add_argument("--queue", type=Path, default=None)
    parser.add_argument("--hw-config", default=DEFAULT_HW_CONFIG)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--priority", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)

    source_root = args.gemmini_rocc_tests.expanduser().resolve()
    run_dir = (args.out or _default_out()).expanduser().resolve()
    if run_dir.exists() and not run_dir.is_dir():
        raise BaselineError(f"output path is not a directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    arms = _parse_arms(args.arms)
    if args.repetitions < 1:
        raise BaselineError("repetitions must be positive")
    source_manifest = snapshot_sources(source_root, run_dir / "source")
    compiler = resolve_compiler(source_root, args.compiler)
    builds = {arm: build_arm(arm, run_dir, compiler, source_manifest) for arm in arms}

    root_receipt = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "prepared" if args.prepare_only else "running",
        "arms": arms,
        "measurement": {
            "warmup_inferences_per_execution": 1,
            "measured_inferences_per_execution": 1,
            "repetitions": args.repetitions,
            "recorded_scope": "compute-cycle decomposition only",
        },
        "source_manifest": "source_manifest.json",
        "source_aggregate_sha256": source_manifest["aggregate_sha256"],
        "builds": builds,
    }
    _write_json(run_dir / "results.json", root_receipt)
    if args.prepare_only:
        print(f"prepared {len(arms)} arms under {run_dir}")
        return 0

    chipyard = (args.chipyard or ext_path("chipyard")).expanduser().resolve()
    queue = (args.queue or (ext_path("firesim_queue") / "bin/firesim-queue")).expanduser().resolve()
    if not queue.is_file():
        raise BaselineError(f"FireSim queue executable not found: {queue}")
    queue_contract = inspect_queue_contract(queue)
    runs: dict[str, dict[str, Any]] = {}
    root_receipt.update(
        chipyard=_git_provenance(chipyard), queue=str(queue), queue_sha256=_sha256(queue),
        queue_cwd_launcher=str(QUEUE_CWD_LAUNCHER),
        queue_cwd_launcher_sha256=_sha256(QUEUE_CWD_LAUNCHER),
        firesim_make_launcher=str(FIRESIM_MAKE_LAUNCHER),
        firesim_make_launcher_sha256=_sha256(FIRESIM_MAKE_LAUNCHER),
        queue_contract=queue_contract, hw_config=args.hw_config, runs=runs,
    )
    measured: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    for repetition in range(1, args.repetitions + 1):
        for arm in arms:
            try:
                row = load_repetition_receipt(
                    arm, repetition, run_dir, builds[arm], queue=queue, chipyard=chipyard,
                    hw_config=args.hw_config, timeout=args.timeout, priority=args.priority)
                if row is None:
                    row = run_arm(
                        arm, repetition, run_dir, builds[arm], queue=queue, chipyard=chipyard,
                        hw_config=args.hw_config, timeout=args.timeout, priority=args.priority)
                measured[arm].append(row)
                print(f"[{arm} rep {repetition}] measured Total cycles: "
                      f"{row['cycles']['total']:,}", flush=True)
            except Exception as exc:  # noqa: BLE001 - persist completed repetitions before failing.
                runs[arm] = {"status": "error", "completed_repetitions": measured[arm],
                             "error": f"{type(exc).__name__}: {exc}"}
                root_receipt["status"] = "error"
                root_receipt["comparison"] = comparison(
                    {key: row for key, row in runs.items() if row.get("status") == "pass"})
                _write_json(run_dir / "results.json", root_receipt)
                raise
            runs.update({name: aggregate_arm(rows) for name, rows in measured.items() if rows})
            root_receipt["comparison"] = comparison(runs)
            _write_json(run_dir / "results.json", root_receipt)
    root_receipt["status"] = "pass"
    root_receipt["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    root_receipt["comparison"] = comparison(runs)
    _write_json(run_dir / "results.json", root_receipt)
    print(f"wrote {run_dir / 'results.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
