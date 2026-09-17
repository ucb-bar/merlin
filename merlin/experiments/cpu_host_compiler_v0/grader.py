#!/usr/bin/env python3
"""Hermetic, deterministic L0--L3 grader for the four CPU-host compiler arms.

The grader is intentionally submission-agnostic. It builds a package before reading held-out rows,
then invokes the built compiler in a fresh, networkless sandbox for one capsule at a time. L0 checks the
compiler contract, L1 executes randomized scalar semantics under sanitizers, L2 checks RVV tail semantics
and instruction evidence on Spike, and L3 checks the same artifacts on K1 silicon. Structural evidence can
therefore never award an executable pass by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import secrets
import shlex
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from merlin.benchharness import capsule_descriptor


IMPLEMENTED_LEVELS = ("L0", "L1", "L2", "L3")
MODES = ("scalar", "rvv", "rvv_multicore")
FAMILY_CODE = capsule_descriptor.FAMILY_CODE
DTYPE_CODE = capsule_descriptor.DTYPE_CODE
LAYOUT_CODE = capsule_descriptor.LAYOUT_CODE
OPERATION_CODE = capsule_descriptor.OPERATION_CODE
SEMANTIC_OP_CODE = capsule_descriptor.SEMANTIC_OPERATION_CODE
KIND_CODE = capsule_descriptor.KIND_CODE


class GradeError(RuntimeError):
    pass


class SandboxTimeout(GradeError):
    """A submitted compiler/kernel exhausted a declared treatment timeout."""

    def __init__(self, seconds: int):
        super().__init__(f"sandboxed command timed out after {seconds}s")
        self.seconds = seconds


class TreatmentSubmissionFailure(GradeError):
    """The authored package does not satisfy the submission contract."""

    failure_class = "treatment_agent_fail"


class TreatmentBuildFailure(GradeError):
    """A submitted build command failed or exhausted its treatment-owned timeout."""

    failure_class = "treatment_build_fail"

    def __init__(self, reason: str, evidence: dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence


class TrustedEvaluationFailure(GradeError):
    """A controller-owned correctness gate failed before performance measurement."""

    failure_class = "trusted_evaluation_fail"

    def __init__(self, reason: str, evidence: dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence


def _repo_root() -> Path:
    """Discover the checkout root without depending on this file's nesting depth."""
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "merlin" / "python").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise GradeError("could not discover repository root from grader location")


def _llvm_tool(name: str) -> Path:
    tool = _repo_root() / "third_party" / "llvm-install" / "bin" / name
    if not tool.is_file():
        raise GradeError(f"trusted LLVM tool is absent: {tool}")
    return tool


def _native_cc() -> Path:
    # The repository LLVM build intentionally omits compiler-rt. L1 needs ASan/UBSan, so use the
    # host distribution clang whose sanitizer runtime is part of the trusted grader environment.
    value = shutil.which("clang")
    if not value:
        raise GradeError("host clang with sanitizer runtimes is absent")
    return Path(value)


def _setting(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    dotenv = _repo_root() / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _spike_tools() -> dict[str, Path]:
    chipyard = Path(_setting("MERLIN_CHIPYARD") or "/path/to/chipyard")
    bin_dir = chipyard / ".conda-env" / "riscv-tools" / "bin"
    gcc = Path(_setting("MERLIN_RISCV_GCC") or bin_dir / "riscv64-unknown-elf-gcc")
    spike = Path(_setting("MERLIN_SPIKE") or bin_dir / "spike")
    objdump = gcc.with_name("riscv64-unknown-elf-objdump")
    return {"gcc": gcc, "spike": spike, "objdump": objdump}


def _k1_cc() -> Path | None:
    configured = _setting("MERLIN_K1_TOOLCHAIN")
    if not configured:
        return None
    root = Path(configured)
    candidates = [root / "bin" / "clang"]
    if root.is_dir():
        candidates.extend(path / "bin" / "clang" for path in sorted(
            root.glob("spacemit-toolchain-linux-glibc-*")))
    return next((path for path in candidates if path.is_file()), None)


def _k1_connection() -> dict[str, str]:
    return {"host": _setting("MERLIN_K1_HOST"), "port": _setting("MERLIN_K1_SSH_PORT"),
            "key": _setting("MERLIN_K1_SSH_KEY")}


def _ssh_argv(connection: dict[str, str]) -> list[str]:
    argv = ["ssh", "-i", connection["key"]]
    if connection["port"]:
        argv += ["-p", connection["port"]]
    return argv + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   "-o", "StrictHostKeyChecking=no", connection["host"]]


def _scp_argv(connection: dict[str, str]) -> list[str]:
    argv = ["scp", "-i", connection["key"]]
    if connection["port"]:
        argv += ["-P", connection["port"]]
    return argv + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   "-o", "StrictHostKeyChecking=no"]


@contextmanager
def _k1_lock(connection: dict[str, str]):
    # Serialize against every other Merlin K1 user, not merely other CPU-host graders. A private
    # experiment lock could overlap a mining/baseline process and poison the wall-time authority.
    from merlin.mining import k1

    if connection["host"] != k1.K1_HOST:
        raise GradeError("CPU-host grader and canonical K1 adapter resolve different board hosts")
    with k1.board_lock():
        yield


def _probe_k1_state(connection: dict[str, str]) -> dict[str, Any]:
    """Capture board state around one balanced timing pair from outside the measured process."""
    script = r'''
import glob,json,os,time
def read(path):
  try:
    return open(path,encoding="utf-8").read().strip()
  except OSError:
    return None
def values(pattern):
  return {path:read(path) for path in sorted(glob.glob(pattern))}
print(json.dumps({
 "monotonic_ns":time.monotonic_ns(),
 "online":read("/sys/devices/system/cpu/online"),
 "governors":values("/sys/devices/system/cpu/cpufreq/policy*/scaling_governor"),
 "frequencies_khz":values("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"),
 "temperatures_millic":values("/sys/class/thermal/thermal_zone*/temp"),
 "loadavg":read("/proc/loadavg")
},sort_keys=True))
'''
    started = time.monotonic_ns()
    proc = subprocess.run(_ssh_argv(connection) + [
        f"exec /usr/bin/python3 -c {shlex.quote(script)}"], capture_output=True, text=True,
        timeout=20)
    try:
        state = json.loads(proc.stdout.strip()) if proc.returncode == 0 else {}
    except json.JSONDecodeError:
        state = {}
    return {"authority": "driver_ssh_sysfs_procfs", "controller_monotonic_ns": started,
            "returncode": proc.returncode, "state": state,
            "stderr_tail": proc.stderr[-1000:]}


def _k1_state_ready(probe: dict[str, Any], contract: dict[str, Any]) -> bool:
    """Apply the frozen pre-measurement environmental gate to one external probe."""
    if probe.get("returncode") != 0:
        return False
    state = probe.get("state", {})
    if not isinstance(state, dict) or state.get("online") != contract.get("online"):
        return False
    try:
        governors = state["governors"]
        frequencies = [int(value) for value in state["frequencies_khz"].values()]
        temperatures = [int(value) for value in state["temperatures_millic"].values()]
        load = float(str(state["loadavg"]).split()[0])
        expected_frequency = int(contract["frequency_khz"])
        tolerance = float(contract["frequency_relative_tolerance"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return (isinstance(governors, dict) and bool(governors) and
            set(governors.values()) == {contract.get("governor")} and
            len(frequencies) == int(contract.get("frequency_core_count", -1)) and
            all(abs(value - expected_frequency) / expected_frequency <= tolerance
                for value in frequencies) and bool(temperatures) and
            max(temperatures) <= int(contract["maximum_temperature_millic"]) and
            load <= float(contract["maximum_load_1m"]))


def _k1_state_pair_ok(
        before: dict[str, Any], after: dict[str, Any], contract: dict[str, Any]) -> bool:
    if not _k1_state_ready(before, contract) or not _k1_state_ready(after, contract):
        return False
    left, right = before["state"], after["state"]
    try:
        temperature_delta = max(
            abs(int(left["temperatures_millic"][key]) -
                int(right["temperatures_millic"][key]))
            for key in left["temperatures_millic"] if key in right["temperatures_millic"])
        load_delta = abs(float(str(left["loadavg"]).split()[0]) -
                         float(str(right["loadavg"]).split()[0]))
    except (KeyError, TypeError, ValueError):
        return False
    return (left.get("online") == right.get("online") and
            left.get("governors") == right.get("governors") and
            temperature_delta <= int(contract["maximum_pair_temperature_delta_millic"]) and
            load_delta <= float(contract["maximum_pair_load_delta"]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_nonce() -> int:
    """Return a non-zero marker selected only after submitted source has been frozen."""
    return secrets.randbits(63) or 1


def _trusted_result_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines()
            if line.startswith("MERLIN_TRUSTED_RESULT ")]


def _native_receipt_ok(stdout: str, *, seed: int, nonce: int) -> bool:
    expected = (f"MERLIN_TRUSTED_RESULT version=1 seed={seed} nonce={nonce} "
                "memory=1 numeric=1")
    return _trusted_result_lines(stdout) == [expected]


def _k1_timing_authority(
        metrics: dict[str, int], harts: int, output_count: int | None = None,
) -> tuple[bool, bool]:
    """Validate retained per-call correctness and the one excluded thread-audit call."""
    measured_calls = metrics.get("calls", 0)
    per_call_correctness = (
        measured_calls >= 20 and 1 <= metrics.get("audit_call", 0) <= 20 and
        metrics.get("correctness_checks") == measured_calls + 1 and
        metrics.get("audit_wall_ns", 0) > 0 and metrics.get("audit_time_ticks", 0) > 0)
    counter_accounting = (
        metrics.get("pthread_create_attempts") ==
        metrics.get("pthread_creates", 0) + metrics.get("pthread_create_failures", 0) and
        metrics.get("pthread_affinity_attempts") ==
        metrics.get("pthread_affinity_successes", 0) +
        metrics.get("pthread_affinity_failures", 0))
    output_count = metrics.get("audit_output_elements", -1) if output_count is None else output_count
    floor_count = output_count // harts if harts > 0 else -1
    ceil_count = (output_count + harts - 1) // harts if harts > 0 else -1
    shard_attribution = (
        output_count >= harts and metrics.get("audit_output_elements") == output_count and
        metrics.get("audit_serialized_callbacks") == harts - 1 and
        metrics.get("audit_output_coverage") == output_count and
        metrics.get("audit_ownership_violations") == 0 and
        metrics.get("audit_owner_min_elements") >= floor_count and
        metrics.get("audit_owner_max_elements") <= ceil_count and
        metrics.get("audit_balanced_shards") == 1)
    if harts == 1:
        audit_attribution = counter_accounting and shard_attribution and all(
            metrics.get(name, 0) == 0 for name in (
            "pinned_hart_mask", "worker_hart_mask", "productive_worker_hart_mask",
            "pthread_create_attempts", "pthread_creates", "pthread_create_failures",
            "pthread_completions", "pthread_affinity_attempts",
            "pthread_affinity_successes", "pthread_affinity_failures",
            "minimum_worker_cpu_ns", "counterfactual_create_attempts",
            "counterfactual_creates", "counterfactual_create_failures",
            "counterfactual_suppressed_starts")) and metrics.get(
                "counterfactual_worker_dependence") == 1
    else:
        expected_hart_mask = (1 << harts) - 1
        worker_mask = expected_hart_mask & ~1
        audit_attribution = counter_accounting and shard_attribution and (
            metrics.get("pinned_hart_mask") == expected_hart_mask and
            metrics.get("worker_hart_mask") == worker_mask and
            metrics.get("productive_worker_hart_mask") == worker_mask and
            metrics.get("pthread_create_attempts") == harts - 1 and
            metrics.get("pthread_creates") == harts - 1 and
            metrics.get("pthread_create_failures") == 0 and
            metrics.get("pthread_completions") == harts - 1 and
            metrics.get("pthread_affinity_attempts") == harts and
            metrics.get("pthread_affinity_successes") == harts and
            metrics.get("pthread_affinity_failures") == 0 and
            metrics.get("minimum_worker_cpu_ns", 0) >= 100 and
            metrics.get("counterfactual_create_attempts") == harts - 1 and
            metrics.get("counterfactual_creates") == harts - 1 and
            metrics.get("counterfactual_create_failures") == 0 and
            metrics.get("counterfactual_suppressed_starts") == harts - 1 and
            metrics.get("counterfactual_worker_dependence") == 1)
    return per_call_correctness, audit_attribution


def _kernel_source_is_receipt_isolated(source: str) -> bool:
    """Forbid submitted code from addressing the trusted result/termination interfaces.

    Linux executions additionally run the kernel in a stdout-silenced child with shared guarded
    buffers.  This static gate is still required for bare-metal Spike, where there is no process
    boundary, and keeps all three levels on one explicit ABI.
    """
    dangerous = re.compile(
        r"\b(?:_?exit|abort|quick_exit|printf|fprintf|sprintf|snprintf|vprintf|puts|putchar|"
        r"vfprintf|vsprintf|vsnprintf|dprintf|vdprintf|fwrite|fputs|fputc|fflush|perror|"
        r"write|writev|pwrite|pwritev|send|sendto|sendmsg|syscall|fork|vfork|clone|clone3|"
        r"kill|raise|longjmp|setjmp|dlsym|dlopen|sched_setaffinity|"
        r"pthread_attr_setaffinity_np|__real_pthread_[A-Za-z0-9_]*|"
        r"htif_[A-Za-z0-9_]*|tohost|fromhost|ecall|scall)\b|"
        r"MERLIN_TRUSTED_RESULT|K1_METRIC|PASS\s+seed")
    return dangerous.search(source) is None


def _submission_digest(root: Path, *, include_policy: bool) -> str:
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise GradeError("submission manifest must be a mapping")
    policy = Path(str(manifest.get("policy", ""))).as_posix()
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise GradeError(f"submission symlink is forbidden: {path.relative_to(root)}")
        relative = path.relative_to(root)
        if (not path.is_file() or any(part == ".git" for part in relative.parts)
                or (not include_policy and relative.as_posix() == policy)):
            continue
        rows.append((relative.as_posix(), _sha256(path)))
    encoded = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_compiler_seal(submission: Path, seal_path: Path | None) -> dict[str, Any]:
    if seal_path is None:
        return {"status": "not_required_by_non_campaign_caller"}
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load((submission / "manifest.yaml").read_text(encoding="utf-8"))
    policy = (submission / str(manifest.get("policy", ""))).resolve()
    checks = {
        "sealed": seal.get("status") == "sealed",
        "policy_sha256": policy.is_file() and seal.get("policy_sha256") == _sha256(policy),
        "compiler_source_sha256": seal.get("compiler_source_sha256") ==
                                  _submission_digest(submission, include_policy=False),
        "compiler_package_sha256": seal.get("compiler_package_sha256") ==
                                   _submission_digest(submission, include_policy=True),
    }
    if not all(checks.values()):
        raise GradeError(f"post-campaign compiler seal failed before heldout access: {checks}")
    return {"status": "pass", "checks": checks, "seal_sha256": _sha256(seal_path)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GradeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise GradeError(f"{path}:{line_number}: capsule must be a mapping")
            rows.append(row)
    return rows


def _validate_corpus(train: Path, validation: Path, heldout: Path) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": _load_jsonl(train), "validation": _load_jsonl(validation),
              "heldout": _load_jsonl(heldout)}
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            required = {"id", "sha256", "split", "family", "operation", "dtype", "shape",
                        "layout", "state", "core_count"}
            if not required <= set(row):
                raise GradeError(f"{split} row omits {sorted(required - set(row))}")
            if row["split"] != split:
                raise GradeError(f"capsule {row['id']} declares split {row['split']}, read from {split}")
            identity = {key: row[key] for key in (
                "family", "operation", "dtype", "shape", "layout", "state", "core_count")}
            canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            if digest != row["sha256"] or not str(row["id"]).endswith(digest[:16]):
                raise GradeError(f"capsule identity digest mismatch: {row['id']}")
            if row["id"] in seen:
                raise GradeError(f"capsule {row['id']} occurs in {seen[row['id']]} and {split}")
            seen[row["id"]] = split
            if row["family"] not in FAMILY_CODE:
                raise GradeError(f"unknown capsule family {row['family']!r}")
    return splits


def _coverage_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Semantic coverage dimensions, deliberately excluding exact shape and capsule identity."""
    family = str(row["family"])
    if family == "contraction":
        return family, str(row["operation"]), str(row["dtype"]), str(row["layout"])
    if family in {"elementwise_map", "reduction", "movement_layout", "fusion_epilogue"}:
        return family, str(row["operation"]), str(row["dtype"])
    if family == "runtime_parallel":
        return family, str(row["operation"])
    raise GradeError(f"unknown capsule family {family!r}")


def _select_semantic_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
    missing = sorted(set(FAMILY_CODE) - set(by_family))
    if missing:
        raise GradeError(f"split omits required families {missing}")
    # One content-addressed representative per operation/type/layout semantic bucket. Content
    # ordering, not file ordering, makes selection invariant to JSONL shuffling.
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_coverage_key(row)].append(row)
    return [min(buckets[key], key=lambda row: row["sha256"]) for key in sorted(buckets)]


def _is_vector_tail(row: dict[str, Any]) -> bool:
    plan = _buffer_plan(row)
    family = row["family"]
    if family in {"contraction", "fusion_epilogue"}:
        extent = int(plan["dim1"])
        lanes = 8  # output is f32/i32
    else:
        extent = int(plan["output_count"] if family == "movement_layout" else plan["dim0"])
        output_kind = str(plan["output_kind"])
        lanes = 32 if output_kind == "int8" else 8
    return extent % lanes != 0


def _select_tail_coverage(
        rows: list[dict[str, Any]], *, split_name: str = "heldout") -> list[dict[str, Any]]:
    selected = []
    for family in FAMILY_CODE:
        candidates = [row for row in rows if row["family"] == family and _is_vector_tail(row)]
        if family == "runtime_parallel" and not candidates:
            candidates = [row for row in rows if row["family"] == family]
        if not candidates:
            raise GradeError(f"{split_name} split has no RVV tail case for {family}")
        # Keep simulator cost bounded while choosing solely from generic capsule properties.
        selected.append(min(candidates, key=lambda row: (
            sum(int(_buffer_plan(row)[name]) for name in (
                "input0_count", "input1_count", "input2_count", "output_count")),
            row["sha256"])))
    return selected


def _select_multicore(
        rows: list[dict[str, Any]], *, split_name: str = "heldout") -> dict[str, Any]:
    candidates = [row for row in rows if row["family"] == "runtime_parallel"
                  and row["operation"] != "single_hart" and int(row["core_count"]) > 1]
    if not candidates:
        raise GradeError(f"{split_name} split has no genuine multicore runtime capsule")
    maximum = max(int(row["core_count"]) for row in candidates)
    return min((row for row in candidates if int(row["core_count"]) == maximum),
               key=lambda row: row["sha256"])


def validate_corpus_for_grading(
        train: Path, validation: Path, heldout: Path) -> dict[str, Any]:
    """Prove all frozen splits can exercise the declared grader before an agent runs.

    Only aggregate structural coverage leaves this trusted boundary. Exact capsule
    identities remain sealed from the agent.
    """
    splits = _validate_corpus(train, validation, heldout)
    coverage: dict[str, Any] = {}
    for split_name, rows in splits.items():
        semantic = _select_semantic_coverage(rows)
        tails = _select_tail_coverage(rows, split_name=split_name)
        multicore = _select_multicore(rows, split_name=split_name)
        coverage[split_name] = {
            "capsule_count": len(rows),
            "families": sorted({str(row["family"]) for row in semantic}),
            "tail_families": sorted({str(row["family"]) for row in tails}),
            "genuine_multicore": (
                multicore["family"] == "runtime_parallel"
                and multicore["operation"] != "single_hart"
                and int(multicore["core_count"]) > 1
            ),
        }
    return {"version": 1, "ready": True, "coverage": coverage}


def _codes(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    tables = {
        "family": FAMILY_CODE,
        "operation": OPERATION_CODE,
        "dtype": DTYPE_CODE,
        "layout": LAYOUT_CODE,
    }
    if field not in tables:
        raise GradeError(f"no stable public enum table for {field!r}")
    table = tables[field]
    unknown = sorted({str(row[field]) for row in rows} - set(table))
    if unknown:
        raise GradeError(f"unknown {field} values outside public ABI: {unknown}")
    # Return the complete table so numeric ABI codes cannot depend on which split or subset happens
    # to be passed by a trusted caller.
    return dict(table)


def _dims(row: dict[str, Any]) -> tuple[int, int, int, int]:
    try:
        return capsule_descriptor.dimensions(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise GradeError(f"invalid public capsule dimensions: {exc}") from exc


def _buffer_plan(row: dict[str, Any]) -> dict[str, int | str]:
    """Resolve the exact kernel buffers from a frozen generic capsule."""
    try:
        return capsule_descriptor.buffer_plan(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise GradeError(f"invalid public capsule buffer plan: {exc}") from exc


def _capsule_mlir(row: dict[str, Any], operation_codes: dict[str, int]) -> str:
    # Keep the historical parameter while every trusted caller migrates as one protocol unit.  The
    # canonical renderer is intentionally the sole numeric-ABI authority and never consults a
    # caller-provided, split-dependent mapping.
    del operation_codes
    try:
        return capsule_descriptor.render_capsule_mlir(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise GradeError(f"cannot render public capsule descriptor: {exc}") from exc


def _sandbox_prefix(*, package: Path, writable_package: bool, extra_ro: tuple[Path, ...] = (),
                    extra_rw: tuple[Path, ...] = (), canonical_io: bool = False) -> list[str]:
    if shutil.which("bwrap") is None:
        raise GradeError("bubblewrap is required for untrusted compiler grading")
    argv = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-net", "--clearenv",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/scratch", "--tmpfs", "/scratch2", "--tmpfs", "/tmp"]
    package_mount = "/package" if canonical_io else str(package)
    if canonical_io:
        argv += ["--dir", "/package", "--dir", "/work"]
    argv += ["--bind" if writable_package else "--ro-bind", str(package), package_mount]
    for index, path in enumerate(extra_ro):
        target = f"/work/ro_{index}" if canonical_io else str(path)
        argv += ["--ro-bind", str(path), target]
    for index, path in enumerate(extra_rw):
        target = "/work/output" if canonical_io and index == 0 else (
            f"/work/rw_{index}" if canonical_io else str(path))
        argv += ["--bind", str(path), target]
    llvm = _repo_root() / "third_party" / "llvm-install"
    if llvm.is_dir():
        argv += ["--ro-bind", str(llvm), str(llvm)]
    path = f"{llvm / 'bin'}:/usr/bin:/bin" if llvm.is_dir() else "/usr/bin:/bin"
    return argv + ["--setenv", "PATH", path, "--setenv", "HOME", "/tmp",
                   "--setenv", "LANG", "C", "--setenv", "LC_ALL", "C",
                   "--chdir", package_mount]


def _execution_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _run_sandbox(argv: list[str], *, timeout: int, limit_execution: bool = False
                 ) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              preexec_fn=_execution_limits if limit_execution else None)
    except subprocess.TimeoutExpired as exc:
        raise SandboxTimeout(timeout) from exc
    if proc.returncode and "Operation not permitted" in proc.stderr:
        raise GradeError("kernel denied the required bubblewrap isolation")
    return proc


def _native_defines(row: dict[str, Any], operation_codes: dict[str, int], *, harts: int = 1
                    ) -> list[str]:
    plan = _buffer_plan(row)
    values = {
        "MERLIN_FAMILY": FAMILY_CODE[row["family"]],
        "MERLIN_OPERATION_CODE": operation_codes[row["operation"]],
        "MERLIN_SEMANTIC_OP": SEMANTIC_OP_CODE[row["operation"]],
        "MERLIN_DTYPE_CODE": DTYPE_CODE[row["dtype"]],
        "MERLIN_LAYOUT": LAYOUT_CODE[row["layout"]],
        "MERLIN_HARTS": harts,
        "MERLIN_VLEN_BITS": 256,
        "MERLIN_DIM0": plan["dim0"], "MERLIN_DIM1": plan["dim1"],
        "MERLIN_DIM2": plan["dim2"], "MERLIN_STATE0": plan["state0"],
        "MERLIN_INPUT0_KIND": KIND_CODE[str(plan["input0_kind"])],
        "MERLIN_INPUT1_KIND": KIND_CODE[str(plan["input1_kind"])],
        "MERLIN_INPUT2_KIND": KIND_CODE[str(plan["input2_kind"])],
        "MERLIN_OUTPUT_KIND": KIND_CODE[str(plan["output_kind"])],
        "MERLIN_INPUT0_COUNT": plan["input0_count"],
        "MERLIN_INPUT1_COUNT": plan["input1_count"],
        "MERLIN_INPUT2_COUNT": plan["input2_count"],
        "MERLIN_OUTPUT_COUNT": plan["output_count"],
    }
    return [f"-D{key}={value}" for key, value in values.items()]


def _grade_native(row: dict[str, Any], compile_record: dict[str, Any],
                  operation_codes: dict[str, int], root: Path) -> dict[str, Any]:
    """L1: independently build and execute the scalar artifact against randomized goldens."""
    record: dict[str, Any] = {"capsule": row["id"], "family": row["family"], "mode": "scalar"}
    if not compile_record.get("ok"):
        return {**record, "status": "fail", "reason": "L0 scalar artifact failed"}
    work = root / row["id"]
    work.mkdir(parents=True)
    executable = work / "capsule_native"
    kernel = Path(str(compile_record["_kernel_path"]))
    harness = Path(__file__).resolve().with_name("trusted_harness.c")
    nonce = _receipt_nonce()
    common = [str(_native_cc()), "-std=c11", "-O1", "-g", "-fno-omit-frame-pointer",
              "-fsanitize=address,undefined", *_native_defines(row, operation_codes)]
    kernel_o, harness_o = work / "kernel.o", work / "trusted_harness.o"
    commands = [
        [*common, "-c", str(kernel), "-o", str(kernel_o)],
        [*common, f"-DMERLIN_RECEIPT_NONCE={nonce}ULL", "-c", str(harness),
         "-o", str(harness_o)],
        [str(_native_cc()), "-fsanitize=address,undefined", str(harness_o), str(kernel_o),
         "-lm", "-o", str(executable)],
    ]
    started = time.monotonic()
    build_logs = []
    for stage, command in enumerate(commands):
        stage_started = time.monotonic()
        try:
            compile_proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return {**record, "status": "fail", "reason": "trusted native build timed out",
                    "failed_stage_index": stage, "timeout_seconds": 120,
                    "build_logs": build_logs}
        build_logs.append({"returncode": compile_proc.returncode,
                           "wall_seconds": time.monotonic() - stage_started,
                           "stderr_tail": compile_proc.stderr[-4000:]})
        if compile_proc.returncode:
            break
    record["build_wall_seconds"] = time.monotonic() - started
    record["build_stderr_tail"] = compile_proc.stderr[-4000:]
    record["build_returncode"] = compile_proc.returncode
    if compile_proc.returncode:
        return {**record, "status": "fail", "reason": "trusted native build failed",
                "failed_stage_index": stage, "build_logs": build_logs}
    trials = []
    for _ in range(3):
        # The seed is intentionally created after kernel.c exists. Recording it makes every failed
        # or successful trial exactly replayable without exposing a pre-codegen constant.
        seed = secrets.randbits(63) or 1
        started = time.monotonic()
        try:
            proc = _run_sandbox(
                _sandbox_prefix(package=work, writable_package=False) +
                ["/usr/bin/env", "ASAN_OPTIONS=detect_leaks=0:abort_on_error=1",
                 "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1", str(executable), str(seed)],
                timeout=45, limit_execution=True)
        except SandboxTimeout:
            return {**record, "status": "fail", "reason": "trusted native execution timed out",
                    "trials": trials, "timed_out_trial_index": len(trials),
                    "timed_out_seed": seed, "timeout_seconds": 45}
        trial = {"seed": seed, "returncode": proc.returncode,
                 "wall_seconds": time.monotonic() - started,
                 "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-4000:]}
        trial["ok"] = proc.returncode == 0 and _native_receipt_ok(
            proc.stdout, seed=seed, nonce=nonce)
        trials.append(trial)
    ok = all(trial["ok"] for trial in trials)
    record.update(status="pass" if ok else "fail", trials=trials, receipt_nonce=nonce,
                  checks={"numeric_correctness": ok, "memory_safety": ok,
                          "post_codegen_random_seeds": True,
                          "trusted_parent_receipts": ok})
    return record


def _useful_vector_dataflow(disassembly: str) -> dict[str, Any]:
    """Conservatively prove input-vector-compute-output flow inside the ABI entry point.

    Merely finding a vector opcode is insufficient: a treatment can add dead or semantics-neutral
    vector work beside an unchanged scalar kernel.  This recognizer follows ABI input/output pointer
    aliases plus vector-register dependencies and accepts only a computed vector value that reaches an
    output store (or a reduced vector value extracted and scalar-stored to output).
    """
    in_function = False
    instructions: list[tuple[int, str, list[str]]] = []
    for line in disassembly.splitlines():
        symbol = re.search(r"<([^>]+)>:\s*$", line)
        if symbol:
            if in_function:
                if symbol.group(1).startswith(".L"):
                    continue
                break
            in_function = symbol.group(1) == "merlin_capsule_run"
            continue
        if not in_function:
            continue
        instruction = re.match(r"^\s*([0-9a-f]+):\s*(.*)$", line)
        if not instruction:
            continue
        fields = instruction.group(2).split()
        if len(fields) < 2 or re.fullmatch(r"[0-9a-f]+", fields[0]) is None:
            continue
        # GNU objdump may tab-separate mnemonic and operands, while LLVM objdump commonly emits
        # individual two-hex-digit instruction bytes. Parse the address/encoding prefix first;
        # taking the final tab column mistakes `v28,(a0)` for the mnemonic.
        mnemonic_index = 1
        while (mnemonic_index < len(fields) and len(fields[mnemonic_index]) == 2 and
               re.fullmatch(r"[0-9a-f]{2}", fields[mnemonic_index])):
            mnemonic_index += 1
        if mnemonic_index >= len(fields):
            continue
        mnemonic = fields[mnemonic_index]
        operand_text = " ".join(fields[mnemonic_index + 1:])
        operands = [operand.strip() for operand in operand_text.split(",") if operand.strip()]
        instructions.append((int(instruction.group(1), 16), mnemonic.lower(), operands))

    input_ptrs = {"a1", "a2", "a3"}
    output_ptrs = {"a4"}
    vector_loaded: set[str] = set()
    vector_computed: set[str] = set()
    vector_dependencies: dict[str, set[int]] = {}
    scalar_results: set[str] = set()
    scalar_dependencies: dict[str, set[int]] = {}
    source_loads: list[str] = []
    output_vector_stores: list[str] = []
    output_scalar_stores: list[str] = []
    output_scalar_overwrites: list[str] = []
    useful_paths: list[set[int]] = []
    useful_vector_store_addresses: list[int] = []
    vector_mnemonics: set[str] = set()

    def base_register(operand: str) -> str | None:
        match = re.search(r"\(([a-z][a-z0-9]*)\)", operand)
        return match.group(1) if match else None

    def vector_registers(operands: list[str]) -> list[str]:
        return [operand for operand in operands if re.fullmatch(r"v(?:[0-9]|[12][0-9]|3[01])", operand)]

    scalar_aliases = {"mv", "addi", "addiw", "slli", "srli", "srai", "sext.w", "zext.w"}
    scalar_binary = {"add", "sub", "and", "or", "xor"}
    scalar_stores = {"sb", "sh", "sw", "sd", "fsw", "fsd"}
    for address, mnemonic, operands in instructions:
        if mnemonic in scalar_aliases and len(operands) >= 2:
            destination, source = operands[0], operands[1]
            for aliases in (input_ptrs, output_ptrs):
                if source in aliases:
                    aliases.add(destination)
                else:
                    aliases.discard(destination)
        elif mnemonic in scalar_binary and len(operands) >= 3:
            destination, sources = operands[0], operands[1:3]
            for aliases in (input_ptrs, output_ptrs):
                if any(source in aliases for source in sources):
                    aliases.add(destination)
                else:
                    aliases.discard(destination)

        if mnemonic.startswith("v"):
            vector_mnemonics.add(mnemonic)
        vregs = vector_registers(operands)
        if mnemonic.startswith("vl") and vregs and any(
                base_register(operand) in input_ptrs | output_ptrs for operand in operands):
            vector_loaded.add(vregs[0])
            vector_dependencies[vregs[0]] = {address}
            source_loads.append(mnemonic)
            continue
        if mnemonic.startswith("vs") and vregs and any(
                base_register(operand) in output_ptrs for operand in operands):
            if vregs[0] in vector_computed:
                output_vector_stores.append(mnemonic)
                useful_paths.append({*vector_dependencies.get(vregs[0], set()), address})
                useful_vector_store_addresses.append(address)
            continue
        if mnemonic in {"vmv.x.s", "vfmv.f.s"} and len(operands) >= 2:
            if operands[1] in vector_computed:
                scalar_results.add(operands[0])
                scalar_dependencies[operands[0]] = {
                    *vector_dependencies.get(operands[1], set()), address}
            continue
        if mnemonic in scalar_stores and operands:
            if (useful_vector_store_addresses and any(
                    base_register(operand) in output_ptrs for operand in operands[1:])):
                output_scalar_overwrites.append(mnemonic)
            if (operands[0] in scalar_results and any(
                    base_register(operand) in output_ptrs for operand in operands[1:])):
                output_scalar_stores.append(mnemonic)
                useful_paths.append({*scalar_dependencies.get(operands[0], set()), address})
            continue
        # Actual memory loads/stores have already continued above. Prefixes alone are not enough:
        # compute instructions such as vsext.vf2 and vslide* also begin with `vs`/`vl`.
        if mnemonic.startswith("v") and not mnemonic.startswith("vset") and vregs:
            destination = vregs[0]
            sources = vregs[1:]
            # Accumulating instructions read their destination as an implicit/explicit source.
            if "macc" in mnemonic or "red" in mnemonic:
                sources = [destination, *sources]
            identity = ((mnemonic in {"vadd.vx", "vsub.vx", "vxor.vx", "vor.vx"} and
                         len(operands) >= 3 and operands[2] == "zero") or
                        (mnemonic in {"vadd.vi", "vsub.vi"} and len(operands) >= 3 and
                         operands[2] in {"0", "0x0"}))
            if (not identity and
                    any(source in vector_loaded or source in vector_computed for source in sources)):
                vector_computed.add(destination)
                dependencies = {address}
                for source in sources:
                    dependencies.update(vector_dependencies.get(source, set()))
                vector_dependencies[destination] = dependencies

    useful = bool(source_loads and useful_paths and not output_scalar_overwrites)
    required_execution_pcs = (sorted(min(useful_paths, key=lambda path: (len(path), sorted(path))))
                              if useful_paths else [])
    return {
        "version": 1,
        "function_found": in_function,
        "useful": useful,
        "source_vector_loads": source_loads,
        "computed_vector_registers": sorted(vector_computed),
        "output_vector_stores": output_vector_stores,
        "output_scalar_stores": output_scalar_stores,
        "output_scalar_overwrites": output_scalar_overwrites,
        "required_execution_pcs": required_execution_pcs,
        "vector_instructions": sorted(vector_mnemonics),
    }


def _grade_spike(row: dict[str, Any], compile_record: dict[str, Any],
                 operation_codes: dict[str, int], root: Path) -> dict[str, Any]:
    """L2: execute an RVV tail case on Spike and require instructions from kernel.o itself."""
    record: dict[str, Any] = {"capsule": row["id"], "family": row["family"], "mode": "rvv",
                              "tail_case": _is_vector_tail(row)}
    tools = _spike_tools()
    absent = [name for name, path in tools.items() if not path.is_file()]
    if absent:
        raise GradeError(f"Spike tools absent: {absent}")
    if not compile_record.get("ok"):
        return {**record, "status": "fail", "reason": "L0 RVV artifact failed"}
    work = root / row["id"]
    work.mkdir(parents=True)
    gcc, objdump, spike = tools["gcc"], tools["objdump"], tools["spike"]
    flags = ["-march=rv64gcv_zfh_zvfh_zvl256b", "-mabi=lp64d", "-mcmodel=medany",
             "-O2", "-fno-tree-vectorize", "-ffreestanding"]
    kernel = Path(str(compile_record["_kernel_path"]))
    kernel_o = work / "kernel.o"
    commands = [[str(gcc), *flags, "-c", str(kernel), "-o", str(kernel_o)]]
    harness_dir = _repo_root() / "merlin" / "runtime" / "baremetal" / "spike"
    seed = secrets.randbits(63) or 1
    nonce = _receipt_nonce()
    harness_o = work / "harness.o"
    commands.append([str(gcc), *flags, *_native_defines(row, operation_codes),
                     "-DMERLIN_FREESTANDING=1", f"-DMERLIN_SEED={seed}ULL",
                     f"-DMERLIN_RECEIPT_NONCE={nonce}ULL",
                     "-I", str(harness_dir), "-c",
                     str(Path(__file__).resolve().with_name("trusted_harness.c")),
                     "-o", str(harness_o)])
    objects = [kernel_o, harness_o]
    for name, source in (("crt.o", harness_dir / "crt.S"),
                         ("htif.o", harness_dir / "htif.c"),
                         ("libc.o", harness_dir / "libc_min.c")):
        output = work / name
        commands.append([str(gcc), *flags, "-I", str(harness_dir), "-c", str(source),
                         "-o", str(output)])
        objects.append(output)
    elf = work / "capsule.elf"
    commands.append([str(gcc), *flags, "-nostdlib", "-nostartfiles", "-T",
                     str(harness_dir / "link.ld"), *(str(path) for path in objects),
                     "-lm", "-lgcc", "-o", str(elf)])
    build_logs = []
    for stage_index, command in enumerate(commands):
        started = time.monotonic()
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return {**record, "status": "fail", "reason": "Spike build timed out",
                    "build_logs": build_logs, "failed_stage_index": stage_index,
                    "timeout_seconds": 120}
        build_logs.append({"returncode": proc.returncode,
                           "wall_seconds": time.monotonic() - started,
                           "stderr_tail": proc.stderr[-3000:]})
        if proc.returncode:
            return {**record, "status": "fail", "reason": "Spike build failed",
                    "build_logs": build_logs, "failed_stage_index": stage_index}
    disassembly = subprocess.run([str(objdump), "-d", str(kernel_o)], capture_output=True,
                                 text=True, timeout=60)
    vector_dataflow = _useful_vector_dataflow(disassembly.stdout)
    vector_instructions = vector_dataflow["vector_instructions"]
    objcopy = gcc.with_name("riscv64-unknown-elf-objcopy")
    text_binary = work / "kernel.text.bin"
    text_extract = subprocess.run(
        [str(objcopy), "-O", "binary", "--only-section=.text*", str(kernel_o), str(text_binary)],
        capture_output=True, text=True, timeout=60)
    kernel_text_sha256 = (_sha256(text_binary)
                          if text_extract.returncode == 0 and text_binary.is_file() and
                          text_binary.stat().st_size > 0 else None)
    started = time.monotonic()
    try:
        proc = _run_sandbox(
            _sandbox_prefix(package=work, writable_package=False, extra_ro=(spike,)) +
            [str(spike), "-l", "--isa=rv64gcv_zfh_zvfh_zvl256b", "-p1", str(elf)],
            timeout=180, limit_execution=True)
    except SandboxTimeout:
        return {**record, "status": "fail", "reason": "Spike execution timed out",
                "seed": seed, "vector_instructions": vector_instructions,
                "vector_dataflow": vector_dataflow,
                "kernel_text_sha256": kernel_text_sha256,
                "build_logs": build_logs, "timeout_seconds": 180,
                "wall_seconds": time.monotonic() - started}
    linked_disassembly = subprocess.run([str(objdump), "-d", str(elf)], capture_output=True,
                                        text=True, timeout=60)
    linked_dataflow = _useful_vector_dataflow(linked_disassembly.stdout)
    executed_pcs = {int(value, 16) for value in re.findall(
        r"core\s+\d+:\s+0x([0-9a-fA-F]+)", proc.stderr)}
    required_pcs = set(linked_dataflow["required_execution_pcs"])
    executed_vector_dataflow = bool(required_pcs and required_pcs <= executed_pcs)
    required_pc_trace_lines = [line for line in proc.stderr.splitlines()
                               if any(re.search(rf"\b0x0*{pc:x}\b", line, re.IGNORECASE)
                                      for pc in required_pcs)]
    receipt_pattern = re.compile(
        rf"^MERLIN_TRUSTED_RESULT version=1 seed={seed} nonce={nonce} "
        rf"vlenb=32 cycles=([1-9][0-9]*) calls=20$")
    receipt_matches = [receipt_pattern.fullmatch(line) for line in _trusted_result_lines(proc.stdout)]
    trusted_receipt = len(receipt_matches) == 1 and receipt_matches[0] is not None
    checks = {"rvv_correctness": proc.returncode == 0 and trusted_receipt,
              "instruction_evidence": vector_dataflow["useful"] and
                                      linked_dataflow["useful"] and
                                      executed_vector_dataflow and
                                      kernel_text_sha256 is not None,
              "tail_case": bool(record["tail_case"]) or row["family"] == "runtime_parallel",
              "vlen_256": "vlenb=32" in proc.stdout}
    cycle_match = re.search(r"\bcycles=([0-9]+)\b", proc.stdout)
    spike_cycles = int(cycle_match.group(1)) if cycle_match else 0
    checks["cycle_measurement"] = spike_cycles > 0
    record.update(status="pass" if all(checks.values()) else "fail", seed=seed, checks=checks,
                  vector_instructions=vector_instructions, vector_dataflow=vector_dataflow,
                  linked_vector_dataflow=linked_dataflow,
                  executed_vector_dataflow=executed_vector_dataflow,
                  trusted_receipt=trusted_receipt,
                  receipt_nonce=nonce,
                  required_pc_trace_lines=required_pc_trace_lines,
                  spike_trace_sha256=hashlib.sha256(proc.stderr.encode()).hexdigest(),
                  kernel_text_sha256=kernel_text_sha256, build_logs=build_logs,
                  spike_cycles=spike_cycles, spike_returncode=proc.returncode,
                  wall_seconds=time.monotonic() - started, stdout_tail=proc.stdout[-4000:],
                  stderr_tail=proc.stderr[-4000:])
    return record


def _grade_k1(row: dict[str, Any], compile_record: dict[str, Any],
              operation_codes: dict[str, int], root: Path, *, seed: int | None = None
              ) -> dict[str, Any]:
    """L3: cross-build and run one exact-mode artifact under an independent K1 monitor."""
    mode = str(compile_record["mode"])
    harts = int(compile_record.get("metadata", {}).get("harts", -1))
    record: dict[str, Any] = {"capsule": row["id"], "family": row["family"],
                              "mode": mode, "harts": harts}
    cc, connection = _k1_cc(), _k1_connection()
    if cc is None:
        raise GradeError("SpacemiT compiler is absent")
    if not connection["host"] or not Path(connection["key"]).is_file():
        raise GradeError("K1 SSH configuration is absent")
    if not compile_record.get("ok"):
        return {**record, "status": "fail", "reason": f"L0 {mode} artifact failed"}
    work = root / f"{row['id']}_{mode}"
    work.mkdir(parents=True)
    binary = work / "capsule_k1"
    seed = (secrets.randbits(63) or 1) if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1 or seed >= 2**63:
        raise GradeError("trusted K1 seed must be an integer in [1, 2^63)")
    nonce = _receipt_nonce()
    kernel = Path(str(compile_record["_kernel_path"]))
    common = [str(cc), "--target=riscv64-unknown-linux-gnu", "-march=rv64gcv_zfh_zvfh",
              "-mabi=lp64d", "-O3", "-DNDEBUG", "-pthread", "-DMERLIN_K1_LINUX=1",
              *_native_defines(row, operation_codes, harts=harts)]
    kernel_o, harness_o = work / "kernel.o", work / "trusted_harness.o"
    commands = [
        [*common, "-c", str(kernel), "-o", str(kernel_o)],
        [*common, f"-DMERLIN_RECEIPT_NONCE={nonce}ULL", "-c",
         str(Path(__file__).resolve().with_name("trusted_harness.c")), "-o", str(harness_o)],
        [*common, str(harness_o), str(kernel_o), "-lm", "-pthread",
         "-Wl,--wrap=pthread_create", "-Wl,--wrap=pthread_join",
         "-Wl,--wrap=pthread_setaffinity_np",
         "-o", str(binary)],
    ]
    started = time.monotonic()
    build_logs = []
    for stage, command in enumerate(commands):
        stage_started = time.monotonic()
        try:
            build = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return {**record, "status": "fail", "reason": "K1 cross-build timed out",
                    "failed_stage_index": stage, "timeout_seconds": 180,
                    "build_logs": build_logs}
        build_logs.append({"returncode": build.returncode,
                           "wall_seconds": time.monotonic() - stage_started,
                           "stderr_tail": build.stderr[-4000:]})
        if build.returncode:
            break
    record.update(build_wall_seconds=time.monotonic()-started,
                  build_stderr_tail=build.stderr[-4000:], build_returncode=build.returncode)
    if build.returncode:
        return {**record, "status": "fail", "reason": "K1 cross-build failed",
                "failed_stage_index": stage, "build_logs": build_logs}
    objcopy = cc.with_name("llvm-objcopy")
    text_binary = work / "kernel.text.bin"
    text_extract = subprocess.run(
        [str(objcopy), "-O", "binary", "--only-section=.text*", str(kernel_o),
         str(text_binary)], capture_output=True, text=True, timeout=60)
    if (text_extract.returncode != 0 or not text_binary.is_file() or
            text_binary.stat().st_size == 0):
        raise GradeError("K1 kernel .text extraction failed")
    record["kernel_text_sha256"] = _sha256(text_binary)
    local_sha = _sha256(binary)
    remote = f"/tmp/merlin_host_grade_{secrets.token_hex(12)}"
    monitor = Path(__file__).resolve().with_name("k1_monitor.py")
    ssh = _ssh_argv(connection)
    try:
        create = subprocess.run(ssh + [f"install -d -m 0755 {shlex.quote(remote)}"],
                                capture_output=True, text=True, timeout=30)
        if create.returncode:
            raise GradeError(
                f"K1 remote directory creation failed (ssh return code {create.returncode})")
        upload = subprocess.run(_scp_argv(connection) + [str(binary), str(monitor),
                                 f"{connection['host']}:{remote}/"], capture_output=True,
                                text=True, timeout=60)
        if upload.returncode:
            raise GradeError(f"K1 upload failed (scp return code {upload.returncode})")
        remote_binary, remote_monitor = f"{remote}/capsule_k1", f"{remote}/k1_monitor.py"
        prepare_command = (f"chmod 0555 {shlex.quote(remote_binary)} "
                           f"{shlex.quote(remote_monitor)} && sha256sum {shlex.quote(remote_binary)}")
        prepared = subprocess.run(ssh + [prepare_command], capture_output=True, text=True, timeout=30)
        remote_sha = prepared.stdout.split()[0] if prepared.returncode == 0 and prepared.stdout else ""
        if remote_sha != local_sha:
            raise GradeError("K1 upload digest mismatch")
        run_command = (
            f"cd {shlex.quote(remote)} && exec /usr/bin/unshare -n -- /usr/bin/timeout -k 5 90 "
            f"/usr/bin/python3 ./k1_monitor.py ./capsule_k1 {seed} {harts}")
        started = time.monotonic()
        run = subprocess.run(ssh + [run_command], capture_output=True, text=True, timeout=105)
        board_wall = time.monotonic() - started
    finally:
        subprocess.run(ssh + [f"rm -rf -- {shlex.quote(remote)}"], capture_output=True,
                       text=True, timeout=30)
    marker = next((line.removeprefix("MERLIN_K1_MONITOR ") for line in run.stdout.splitlines()
                   if line.startswith("MERLIN_K1_MONITOR ")), None)
    if marker is None:
        raise GradeError(f"K1 monitor emitted no result (ssh return code {run.returncode})")
    try:
        report = json.loads(marker)
    except json.JSONDecodeError as exc:
        raise GradeError(f"invalid K1 monitor JSON: {exc}") from exc
    child_stdout = str(report.get("child_stdout", ""))
    metrics: dict[str, int] = {}
    for line in child_stdout.splitlines():
        match = re.fullmatch(r"K1_METRIC ([a-z_]+) ([0-9]+)", line)
        if match:
            metrics[match.group(1)] = int(match.group(2))
    expected_affinity = "0" if harts == 1 else f"0-{harts-1}"
    expected_cpus = list(range(harts))
    expected_hart_mask = (1 << harts) - 1
    per_call_correctness, audit_attribution = _k1_timing_authority(
        metrics, harts, int(_buffer_plan(row)["output_count"]))
    receipt_line = (f"MERLIN_TRUSTED_RESULT version=1 seed={seed} nonce={nonce} "
                    "memory=1 numeric=1")
    trusted_receipt = _trusted_result_lines(child_stdout) == [receipt_line]
    checks = {
        "exact_mode": compile_record.get("metadata", {}).get("actual_mode") == mode,
        "no_fallback": compile_record.get("metadata", {}).get("fallback_used") is False,
        "numeric_correctness": report.get("returncode") == 0 and trusted_receipt,
        "trusted_parent_receipt": trusted_receipt,
        "per_call_correctness": per_call_correctness,
        "csr_vlen": metrics.get("vlenb") == 32,
        "exact_affinity": metrics.get("affinity_count") == harts and
                          report.get("affinity_samples") == [expected_affinity],
        # The monitor process plus the measured child and at most harts-1 worker threads may be
        # live at once. Ephemeral thread pools need not place all workers in one sampling instant;
        # cumulative task-local affinity and nanosecond schedstat evidence below proves every core.
        "exact_task_count": 2 <= int(report.get("max_tasks", 0)) <= harts + 1 and
                            int(report.get("tids_observed", 0)) >= harts + 1,
        "active_harts": audit_attribution and ((
            report.get("pinned_affinities_observed") == expected_cpus and
            report.get("pinned_runtime_cpus") == expected_cpus
        ) if harts == 1 else True),
        "audit_attribution": audit_attribution,
        "wall_time": metrics.get("wall_ns", 0) > 0 and report.get("wall_ns", 0) > 0,
        "peak_rss": max(metrics.get("peak_rss_kb", 0), int(report.get("peak_rss_kb", 0))) > 0,
        "upload_integrity": remote_sha == local_sha,
    }
    record.update(status="pass" if run.returncode == 0 and all(checks.values()) else "fail",
                  receipt_nonce=nonce,
                  seed=seed, checks=checks, metrics=metrics, monitor=report,
                  local_sha256=local_sha, remote_sha256=remote_sha,
                  board_wall_seconds=board_wall, ssh_returncode=run.returncode,
                  ssh_stderr_tail=run.stderr[-4000:])
    return record


def _command(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(part, str) for part in value):
        raise GradeError(f"{where} must be a non-empty string argument array")
    if any("\x00" in part or "\n" in part for part in value):
        raise GradeError(f"{where} contains forbidden control characters")
    return list(value)


def _package_tree_identity(package: Path) -> str:
    rows = []
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if ".git" in relative.parts:
            raise GradeError("submission .git metadata is forbidden")
        stat = path.lstat()
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise GradeError(f"submission contains a non-regular entry: {relative}")
        rows.append((relative.as_posix(), "dir" if path.is_dir() else "file",
                     stat.st_mode & 0o777, None if path.is_dir() else _sha256(path)))
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _compiler_entrypoint(package: Path, command: list[str]) -> Path:
    first = command[0]
    interpreters = {"python3", "/usr/bin/python3", "/usr/bin/python3.12"}
    raw = command[1] if first in interpreters and len(command) > 1 else first
    relative = Path(raw)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise GradeError("compiler entrypoint must be a package-relative file")
    entrypoint = (package / relative).resolve()
    if not entrypoint.is_relative_to(package.resolve()):
        raise GradeError("compiler entrypoint escapes the package")
    return entrypoint


def _entrypoint_identity(path: Path) -> tuple[int, str] | None:
    if not path.is_file() or path.is_symlink():
        return None
    return path.stat().st_mode & 0o777, _sha256(path)


def _validated_manifest(package: Path, *, require_compiler: bool = False) -> dict[str, Any]:
    manifest_path = package / "manifest.yaml"
    if not manifest_path.is_file():
        raise GradeError("submission/manifest.yaml is absent")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or int(manifest.get("version", 0)) != 1:
        raise GradeError("submission manifest must be a version-1 mapping")
    build = manifest.get("build")
    compiler = manifest.get("compiler")
    if not isinstance(build, dict) or not isinstance(compiler, dict):
        raise GradeError("manifest requires build and compiler mappings")
    _command(build.get("command"), "build.command")
    if build.get("then") is not None:
        _command(build.get("then"), "build.then")
    policy = (package / str(manifest.get("policy", ""))).resolve()
    if not policy.is_relative_to(package.resolve()) or not policy.is_file():
        raise GradeError("manifest policy file is absent")
    compiler_command = _command(compiler.get("command"), "compiler.command")
    entrypoint = _compiler_entrypoint(package, compiler_command)
    if require_compiler:
        if not entrypoint.is_file() or entrypoint.is_symlink():
            raise GradeError("built compiler entrypoint is not a regular package file")
        if compiler_command[0] not in {"python3", "/usr/bin/python3", "/usr/bin/python3.12"} \
                and not os.access(entrypoint, os.X_OK):
            raise GradeError("built compiler entrypoint is not executable")
    required = {"{input_mlir}", "{output_dir}", "{mode}", "{harts}", "{vlen_bits}"}
    text = "\n".join(compiler_command)
    missing = sorted(value for value in required if value not in text)
    if missing:
        raise GradeError(f"compiler.command omits substitutions {missing}")
    return manifest


def _build(package: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        manifest = _validated_manifest(package)
    except GradeError as exc:
        raise TreatmentSubmissionFailure(str(exc)) from exc
    build = manifest["build"]
    commands = [_command(build.get("command"), "build.command")]
    if build.get("then") is not None:
        commands.append(_command(build.get("then"), "build.then"))
    logs = []
    for stage, command in enumerate(commands):
        started = time.monotonic()
        try:
            proc = _run_sandbox(
                _sandbox_prefix(package=package, writable_package=True) + command,
                timeout=1800)
        except SandboxTimeout as exc:
            logs.append({"command": command, "returncode": None,
                         "wall_seconds": time.monotonic() - started,
                         "stdout_tail": "", "stderr_tail": "",
                         "outcome": "timeout"})
            raise TreatmentBuildFailure(
                "submission build timed out",
                {"commands": logs, "failed_stage_index": stage,
                 "timeout_seconds": exc.seconds}) from exc
        logs.append({"command": command, "returncode": proc.returncode,
                     "wall_seconds": time.monotonic() - started,
                     "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]})
        if proc.returncode:
            raise TreatmentBuildFailure(
                "submission build failed",
                {"commands": logs, "failed_stage_index": stage,
                 "returncode": proc.returncode})
    try:
        manifest = _validated_manifest(package, require_compiler=True)
    except GradeError as exc:
        raise TreatmentBuildFailure(
            "submission build did not produce a valid compiler",
            {"commands": logs, "failed_stage_index": len(commands) - 1,
             "contract_error": str(exc)}) from exc
    policy = (package / str(manifest["policy"])).resolve()
    return manifest, {"commands": logs, "policy_sha256": _sha256(policy)}


def prepare_prebuilt_search_package(*, submission: Path, destination: Path,
                                    build_override: list[str]) -> dict[str, Any]:
    """Build one private package, then install a controller-owned candidate-time no-op build.

    The submitted manifest remains a reproducible real build and is what the final grader sees.  Search
    candidates are cloned only from this private, already-built tree, preventing candidate-time rebuilds
    without asking the treatment to lie in its deliverable contract.
    """
    if build_override != ["/bin/true"]:
        raise GradeError("private search build override must be exactly ['/bin/true']")
    if destination.exists():
        raise GradeError("private prebuilt search destination already exists")
    shutil.copytree(submission.resolve(), destination, symlinks=False)
    before_tree = _package_tree_identity(destination)
    before = _sha256(destination / "manifest.yaml")
    submitted_manifest = _validated_manifest(destination)
    entrypoint = _compiler_entrypoint(
        destination, _command(submitted_manifest["compiler"]["command"], "compiler.command"))
    before_entrypoint = _entrypoint_identity(entrypoint)
    manifest, build = _build(destination)
    real_commands = [entry["command"] for entry in build["commands"]]
    built_tree = _package_tree_identity(destination)
    if built_tree == before_tree:
        raise GradeError("submitted search build made no package change")
    built_entrypoint = _entrypoint_identity(entrypoint)
    if built_entrypoint is None or built_entrypoint == before_entrypoint:
        raise GradeError("submitted search build did not produce or update its compiler entrypoint")
    private_manifest = dict(manifest)
    private_manifest["build"] = {"command": list(build_override)}
    (destination / "manifest.yaml").write_text(
        yaml.safe_dump(private_manifest, sort_keys=False), encoding="utf-8")
    _validated_manifest(destination)
    sealed_prebuilt_tree = _package_tree_identity(destination)
    return {
        "version": 1,
        "authority": "driver_private_prebuild",
        "submitted_manifest_sha256": before,
        "private_manifest_sha256": _sha256(destination / "manifest.yaml"),
        "real_build_commands": real_commands,
        "real_build_logs": build["commands"],
        "prebuild_tree_sha256": before_tree,
        "built_tree_sha256": built_tree,
        "sealed_prebuilt_tree_sha256": sealed_prebuilt_tree,
        "submitted_entrypoint_identity": before_entrypoint,
        "built_entrypoint_identity": built_entrypoint,
        "private_build_override": list(build_override),
        "policy_sha256": build["policy_sha256"],
    }


def _freeze_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else (0o555 if os.access(path, os.X_OK) else 0o444))
    root.chmod(0o555)


def _compile_one(package: Path, manifest: dict[str, Any], row: dict[str, Any], mode: str,
                 operation_codes: dict[str, int], root: Path) -> dict[str, Any]:
    work = root / f"{row['id']}_{mode}"
    input_path, output_dir = work / "input.mlir", work / "output"
    output_dir.mkdir(parents=True)
    input_path.write_text(_capsule_mlir(row, operation_codes), encoding="utf-8")
    harts = int(row["core_count"]) if mode == "rvv_multicore" else 1
    substitutions = {"{input_mlir}": "/work/ro_0", "{output_dir}": "/work/output",
                     "{mode}": mode, "{harts}": str(harts), "{vlen_bits}": "256"}
    command = []
    for part in _command(manifest["compiler"]["command"], "compiler.command"):
        for key, value in substitutions.items():
            part = part.replace(key, value)
        if "{" in part or "}" in part:
            raise GradeError(f"unknown compiler command substitution in {part!r}")
        command.append(part)
    started = time.monotonic()
    try:
        proc = _run_sandbox(
            _sandbox_prefix(package=package, writable_package=False,
                            extra_ro=(input_path,), extra_rw=(output_dir,),
                            canonical_io=True) + command,
            timeout=300)
    except SandboxTimeout:
        return {"capsule": row["id"], "family": row["family"], "mode": mode,
                "ok": False, "reason": "compiler invocation timed out",
                "timeout_seconds": 300, "wall_seconds": time.monotonic() - started}
    record = {"capsule": row["id"], "family": row["family"], "mode": mode,
              "returncode": proc.returncode, "wall_seconds": time.monotonic() - started,
              "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
    if proc.returncode:
        record.update(ok=False, reason="compiler invocation failed")
        return record
    required = {name: output_dir / name for name in ("kernel.c", "lowered.mlir", "metadata.json")}
    absent = [name for name, path in required.items() if not path.is_file()]
    if absent:
        record.update(ok=False, reason=f"compiler omitted outputs {absent}")
        return record
    try:
        metadata = json.loads(required["metadata.json"].read_text(encoding="utf-8"))
    except Exception as exc:
        record.update(ok=False, reason=f"metadata is invalid: {exc}")
        return record
    source = required["kernel.c"].read_text(encoding="utf-8")
    vector_mode = mode != "scalar"
    vlen_policy = metadata.get("vlen_policy")
    tail_policy = metadata.get("tail_policy")
    checks = {
        "version": metadata.get("version") == 1,
        "capsule_sha256": metadata.get("capsule_sha256") == row["sha256"],
        "requested_mode": metadata.get("requested_mode") == mode,
        "actual_mode": metadata.get("actual_mode") == mode,
        "fallback_forbidden": metadata.get("fallback_used") is False,
        "harts": int(metadata.get("harts", -1)) == harts,
        "vlen_bits": int(metadata.get("vlen_bits", -1)) == 256,
        "source_sha256": metadata.get("source_sha256") == _sha256(required["kernel.c"]),
        "transformations": isinstance(metadata.get("transformations"), list)
                           and bool(metadata.get("transformations")),
        "vlen_policy": (vlen_policy == "not_applicable" if not vector_mode else
                        vlen_policy in {"scalable_vl", "runtime_verified_fixed"}),
        "tail_policy": (tail_policy == "not_applicable" if not vector_mode else
                        tail_policy in {"dynamic_vl", "explicit_mask"}),
        "kernel_symbol": "merlin_capsule_run" in source,
        "no_main": re.search(r"\bmain\s*\(", source) is None,
        "no_process_wrappers": re.search(
            r"\b(?:fork|vfork|exec[lvpe]*|system|popen|syscall)\s*\(", source) is None,
        "trusted_receipt_isolation": _kernel_source_is_receipt_isolated(source),
        "no_constructors": re.search(r"__attribute__\s*\(\([^)]*(?:constructor|destructor)",
                                     source) is None,
        "source_size": len(source.encode()) <= 8 * 1024 * 1024,
        "lowered_changed": required["lowered.mlir"].read_bytes() != input_path.read_bytes(),
    }
    clang = _llvm_tool("clang")
    if vector_mode:
        if mode == "rvv_multicore" and _k1_cc() is not None:
            syntax_command = [str(_k1_cc()), "--target=riscv64-unknown-linux-gnu", "-std=c11",
                              "-march=rv64gcv_zfh_zvfh", "-mabi=lp64d", "-pthread",
                              "-fsyntax-only", str(required["kernel.c"])]
        else:
            rvv_gcc = _spike_tools()["gcc"]
            syntax_command = [str(rvv_gcc), "-std=c11", "-march=rv64gcv_zfh_zvfh_zvl256b",
                              "-mabi=lp64d", "-fsyntax-only", str(required["kernel.c"])]
    else:
        syntax_command = [str(clang), "-std=c11", "-fsyntax-only", str(required["kernel.c"])]
    artifact_evidence = {
        "checks": checks, "metadata": metadata,
        "source_sha256": _sha256(required["kernel.c"]),
        "source_size_bytes": len(source.encode()),
        "input_mlir_sha256": _sha256(input_path),
        "lowered_mlir_sha256": _sha256(required["lowered.mlir"]),
        "buffer_plan": _buffer_plan(row),
    }
    try:
        syntax = subprocess.run(syntax_command, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        checks.update(c_syntax=False, mlir_verifier=False)
        record.update(
            ok=False, reason="C syntax check timed out", timeout_seconds=60,
            timed_out_stage="c_syntax", syntax_returncode=None,
            verifier_returncode=None,
            syntax_stderr=(exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes)
                           else (exc.stderr or ""))[-2000:], verifier_stderr="",
            **artifact_evidence)
        return record
    checks["c_syntax"] = syntax.returncode == 0
    mlir_opt = _llvm_tool("mlir-opt")
    try:
        verifier = subprocess.run(
            [str(mlir_opt), str(required["lowered.mlir"]), "-o", os.devnull],
            capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        checks["mlir_verifier"] = False
        record.update(
            ok=False, reason="MLIR verifier timed out", timeout_seconds=60,
            timed_out_stage="mlir_verifier", syntax_returncode=syntax.returncode,
            verifier_returncode=None, syntax_stderr=syntax.stderr[-2000:],
            verifier_stderr=(exc.stderr.decode(errors="replace")
                             if isinstance(exc.stderr, bytes) else (exc.stderr or ""))[-2000:],
            **artifact_evidence)
        return record
    checks["mlir_verifier"] = verifier.returncode == 0
    record.update(ok=all(checks.values()), checks=checks, metadata=metadata,
                  source_sha256=_sha256(required["kernel.c"]),
                  source_size_bytes=len(source.encode()),
                  input_mlir_sha256=_sha256(input_path),
                  lowered_mlir_sha256=_sha256(required["lowered.mlir"]),
                  syntax_returncode=syntax.returncode, verifier_returncode=verifier.returncode,
                  syntax_stderr=syntax.stderr[-2000:], verifier_stderr=verifier.stderr[-2000:],
                  buffer_plan=_buffer_plan(row), _kernel_path=str(required["kernel.c"]))
    return record


def _scan_forbidden(package: Path, heldout: list[dict[str, Any]]) -> list[str]:
    needles = [str(row["id"]) for row in heldout] + [str(row["sha256"]) for row in heldout]
    hits = []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(needle in text for needle in needles):
            hits.append(path.relative_to(package).as_posix())
    return hits


def _install_search_policy(package: Path, candidate: dict[str, Any]) -> None:
    """Replace only the declared policy in a private package copy before building it."""
    manifest_path = package / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise GradeError("submission manifest is not a mapping")
    relative = Path(str(manifest.get("policy", "")))
    policy = (package / relative).resolve()
    if not relative.parts or not policy.is_relative_to(package.resolve()):
        raise GradeError("manifest policy path escapes the submission package")
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _search_spike_correct(record: dict[str, Any]) -> bool:
    """Spike search gate: semantics/instructions/VLEN, without requiring a selected tail case."""
    checks = record.get("checks") if isinstance(record.get("checks"), dict) else {}
    return all(checks.get(name) is True for name in (
        "rvv_correctness", "instruction_evidence", "vlen_256", "cycle_measurement"))


def _spike_gate_evidence(record: dict[str, Any], *, compile_ok: bool) -> dict[str, Any]:
    """Retain the exact evidence needed to explain a pre-K1 Spike decision."""
    return {
        "compile_ok": compile_ok,
        "status": record.get("status"),
        "checks": record.get("checks"),
        "seed": record.get("seed"),
        "receipt_nonce": record.get("receipt_nonce"),
        "trusted_receipt": record.get("trusted_receipt"),
        "spike_returncode": record.get("spike_returncode"),
        "spike_cycles": record.get("spike_cycles"),
        "kernel_text_sha256": record.get("kernel_text_sha256"),
        "vector_dataflow": record.get("vector_dataflow"),
        "linked_vector_dataflow": record.get("linked_vector_dataflow"),
        "executed_vector_dataflow": record.get("executed_vector_dataflow"),
        "required_pc_trace_lines": record.get("required_pc_trace_lines"),
        "reason": record.get("reason"),
    }


def _require_pre_k1_spike_gates(
    *, capsules: list[dict[str, Any]], compiled: dict[tuple[str, str], dict[str, Any]],
    compiled_k1: dict[tuple[str, str], dict[str, Any]],
    spike_records: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    gates: dict[str, dict[str, dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for row in capsules:
        capsule_id = str(row["id"])
        gates[capsule_id] = {}
        for label in ("parent", "candidate"):
            compile_ok = bool(compiled[label, capsule_id].get("ok"))
            k1_compile_ok = bool(compiled_k1[label, capsule_id].get("ok"))
            evidence = _spike_gate_evidence(
                spike_records[label, capsule_id], compile_ok=compile_ok)
            evidence["k1_compile_ok"] = k1_compile_ok
            evidence["passed"] = (
                compile_ok and k1_compile_ok and
                _search_spike_correct(spike_records[label, capsule_id]))
            gates[capsule_id][label] = evidence
            if not evidence["passed"]:
                failures.append({"capsule_id": capsule_id, "family": row["family"],
                                 "arm": label, **evidence})
    if failures:
        raise TrustedEvaluationFailure(
            "pre-K1 Spike correctness/instruction gate failed; no K1 timing was started",
            {"version": 1, "stage": "pre_k1_spike_gate", "k1_programs_started": 0,
             "failures": failures, "spike_gates": gates})
    return gates


@contextmanager
def _trusted_stage_cap(seconds: float | None, label: str,
                       deadline_monotonic_ns: int | None = None):
    """Hard-stop a confirmation stage while preserving the broker's outer wall alarm."""
    if seconds is None:
        yield
        return
    limit = float(seconds)
    if limit <= 0:
        raise GradeError(f"trusted {label} stage cap must be positive")
    if deadline_monotonic_ns is not None:
        remaining = (deadline_monotonic_ns - time.monotonic_ns()) / 1e9
        if remaining <= 0:
            raise TimeoutError("trusted search wall deadline expired")
        limit = min(limit, remaining)
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(_signum, _frame):
        raise TimeoutError(f"trusted {label} stage cap exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            restored = max(1e-6, previous_timer[0] - (time.monotonic() - started))
            signal.setitimer(signal.ITIMER_REAL, restored, previous_timer[1])


def evaluate_public_policy_k1(
    *,
    submission: Path,
    capsules: list[dict[str, Any]],
    parent: dict[str, Any],
    candidate: dict[str, Any],
    repeats: int,
    public_rows: list[dict[str, Any]],
    board_environment: dict[str, Any],
    deadline_monotonic_ns: int | None = None,
    stage_caps: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Trusted search evaluator used only by the outside-sandbox broker.

    Parent and child are built from the same submission snapshot. Each capsule first clears Spike
    semantic/instruction checks, then runs as a frozen balanced sequence of paired measurements on
    K1. The function has no held-out argument or filesystem lookup.
    """
    if repeats != 6:
        raise GradeError("trusted CPU-host search requires exactly six balanced paired measurements")
    operation_codes = _codes(public_rows, "operation")
    caps = dict(stage_caps or {})
    with tempfile.TemporaryDirectory(prefix="merlin-host-trusted-search-") as temporary:
        root = Path(temporary)
        packages: dict[str, Path] = {}
        manifests: dict[str, dict[str, Any]] = {}
        for label, policy in (("parent", parent), ("candidate", candidate)):
            package = root / label / "submission"
            shutil.copytree(submission.resolve(), package, symlinks=False)
            _install_search_policy(package, policy)
            with _trusted_stage_cap(
                    caps.get("package_build"), "package-build", deadline_monotonic_ns):
                manifest, _ = _build(package)
            _freeze_tree(package)
            packages[label] = package
            manifests[label] = manifest

        compiled: dict[tuple[str, str], dict[str, Any]] = {}
        compiled_k1: dict[tuple[str, str], dict[str, Any]] = {}
        for label in ("parent", "candidate"):
            compile_root = root / label / "compile"
            compile_root.mkdir()
            for row in capsules:
                with _trusted_stage_cap(
                        caps.get("compiler_invocation"), "compiler-invocation",
                        deadline_monotonic_ns):
                    compiled[label, str(row["id"])] = _compile_one(
                        packages[label], manifests[label], row, "rvv", operation_codes,
                        compile_root)
                if row.get("family") == "runtime_parallel":
                    with _trusted_stage_cap(
                            caps.get("compiler_invocation"), "compiler-invocation",
                            deadline_monotonic_ns):
                        compiled_k1[label, str(row["id"])] = _compile_one(
                            packages[label], manifests[label], row, "rvv_multicore",
                            operation_codes, compile_root)
                else:
                    compiled_k1[label, str(row["id"])] = compiled[label, str(row["id"])]

        spike_ok: dict[tuple[str, str], bool] = {}
        spike_records: dict[tuple[str, str], dict[str, Any]] = {}
        for label in ("parent", "candidate"):
            spike_root = root / label / "spike"
            spike_root.mkdir()
            for row in capsules:
                with _trusted_stage_cap(
                        caps.get("spike_check"), "Spike-check", deadline_monotonic_ns):
                    record = _grade_spike(
                        row, compiled[label, str(row["id"])], operation_codes, spike_root)
                spike_records[label, str(row["id"])] = record
                spike_ok[label, str(row["id"])] = _search_spike_correct(record)

        # This gate deliberately precedes `_k1_lock` and every `_grade_k1` call: correctness and
        # executed vector evidence are prerequisites, never post-hoc annotations on paid timing.
        spike_gates = _require_pre_k1_spike_gates(
            capsules=capsules, compiled=compiled, compiled_k1=compiled_k1,
            spike_records=spike_records)

        observations = []
        connection = _k1_connection()
        with _k1_lock(connection):
            for row in capsules:
                capsule_id = str(row["id"])
                baseline_elapsed_ns: list[int] = []
                baseline_calls: list[int] = []
                candidate_elapsed_ns: list[int] = []
                candidate_calls: list[int] = []
                condition_pairs: list[dict[str, Any]] = []
                k1_code_sha256: dict[str, str] = {}
                k1_ok = True
                pair_orders = ["parent_candidate", "candidate_parent", "candidate_parent",
                               "parent_candidate", "parent_candidate", "candidate_parent"]
                excluded_condition_pairs: list[dict[str, Any]] = []
                pair_attempt = 0
                repeat = 0
                maximum_replacements = int(
                    board_environment["maximum_invalid_pair_replacements_per_capsule"])
                while repeat < repeats:
                    pair_order = pair_orders[repeat]
                    settle_probes: list[dict[str, Any]] = []
                    for settle_attempt in range(int(board_environment["settle_attempts"])):
                        before = _probe_k1_state(connection)
                        settle_probes.append(before)
                        if _k1_state_ready(before, board_environment):
                            break
                        if settle_attempt + 1 < int(board_environment["settle_attempts"]):
                            time.sleep(float(board_environment["settle_interval_seconds"]))
                    else:
                        raise GradeError("K1 did not enter the frozen pre-pair environment")
                    pair_seed = secrets.randbits(63) or 1
                    labels = (("parent", baseline_elapsed_ns, baseline_calls),
                              ("candidate", candidate_elapsed_ns, candidate_calls)) \
                        if pair_order == "parent_candidate" else \
                        (("candidate", candidate_elapsed_ns, candidate_calls),
                         ("parent", baseline_elapsed_ns, baseline_calls))
                    pair_measurements: dict[str, dict[str, int]] = {}
                    for label, elapsed_samples, call_samples in labels:
                        if (deadline_monotonic_ns is not None and
                                time.monotonic_ns() >= deadline_monotonic_ns):
                            raise GradeError("trusted K1 search reached its frozen wall deadline")
                        run_root = (root / label / "k1" /
                                    f"pair_{repeat}_attempt_{pair_attempt}")
                        run_root.mkdir(parents=True, exist_ok=True)
                        with _trusted_stage_cap(
                                caps.get("k1_program"), "K1-program",
                                deadline_monotonic_ns):
                            result = _grade_k1(
                                row, compiled_k1[label, capsule_id], operation_codes, run_root,
                                seed=pair_seed)
                        elapsed_ns = result.get("metrics", {}).get("wall_ns")
                        calls = result.get("metrics", {}).get("calls")
                        if (result.get("status") != "pass" or result.get("seed") != pair_seed or
                                not isinstance(elapsed_ns, int) or elapsed_ns <= 0 or
                                not isinstance(calls, int) or calls <= 0):
                            raise GradeError(
                                f"trusted K1 {label} measurement failed for {capsule_id}: "
                                f"{result.get('reason') or result.get('checks')}")
                        digest = result.get("kernel_text_sha256")
                        if not isinstance(digest, str) or len(digest) != 64:
                            raise GradeError("trusted K1 result lacks its kernel .text digest")
                        previous_digest = k1_code_sha256.setdefault(label, digest)
                        if previous_digest != digest:
                            raise GradeError("trusted K1 kernel .text changed between paired repeats")
                        pair_measurements[label] = {
                            "elapsed_ns": elapsed_ns, "calls": calls, "seed": pair_seed,
                            "evidence": json.loads(json.dumps(result)),
                        }
                        k1_ok = k1_ok and result.get("status") == "pass"
                    after = _probe_k1_state(connection)
                    condition_ok = _k1_state_pair_ok(before, after, board_environment)
                    condition_record = {
                        "pair_id": repeat, "attempt_id": pair_attempt, "order": pair_order,
                        "seed": pair_seed, "settle_probes": settle_probes,
                        "measurements": pair_measurements, "before": before, "after": after,
                        "valid": condition_ok}
                    pair_attempt += 1
                    if not condition_ok:
                        excluded_condition_pairs.append(condition_record)
                        if len(excluded_condition_pairs) > maximum_replacements:
                            raise GradeError(
                                "K1 exceeded the frozen invalid-environment replacement limit")
                        continue
                    condition_pairs.append(condition_record)
                    baseline_elapsed_ns.append(pair_measurements["parent"]["elapsed_ns"])
                    baseline_calls.append(pair_measurements["parent"]["calls"])
                    candidate_elapsed_ns.append(pair_measurements["candidate"]["elapsed_ns"])
                    candidate_calls.append(pair_measurements["candidate"]["calls"])
                    repeat += 1
                base_record = compiled["parent", capsule_id]
                candidate_record = compiled["candidate", capsule_id]
                observations.append({
                    "capsule_id": capsule_id,
                    "family": row["family"],
                    "baseline_elapsed_ns": baseline_elapsed_ns,
                    "baseline_calls": baseline_calls,
                    "candidate_elapsed_ns": candidate_elapsed_ns,
                    "candidate_calls": candidate_calls,
                    "correctness_ok": bool(
                        base_record.get("ok") and candidate_record.get("ok") and
                        compiled_k1["parent", capsule_id].get("ok") and
                        compiled_k1["candidate", capsule_id].get("ok") and
                        spike_ok["parent", capsule_id] and
                        spike_ok["candidate", capsule_id] and k1_ok),
                    "baseline_code_sha256": k1_code_sha256.get("parent"),
                    "candidate_code_sha256": k1_code_sha256.get("candidate"),
                    "code_digest_authority": "measured_k1_kernel_object_text_section",
                    "timing_authority": "spacemit_k1_elapsed_ns_div_completed_calls",
                    "correctness_authority": "spike_rv64gcv_and_k1_trusted_harness",
                    "spike_gates": spike_gates[capsule_id],
                    "parent_candidate_sha256": parent["candidate_sha256"],
                    "candidate_sha256": candidate["candidate_sha256"],
                    "pair_orders": pair_orders,
                    "board_condition_pairs": condition_pairs,
                    "excluded_board_condition_pairs": excluded_condition_pairs,
                    "k1_program_count": pair_attempt * 2,
                })
        return observations


def evaluate_public_policy_spike(
    *,
    submission: Path,
    capsules: list[dict[str, Any]],
    parent: dict[str, Any],
    candidate: dict[str, Any],
    public_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cheap trusted screening: one deterministic Spike cycle count per public capsule."""
    operation_codes = _codes(public_rows, "operation")
    with tempfile.TemporaryDirectory(prefix="merlin-host-trusted-screen-") as temporary:
        root = Path(temporary)
        compiled: dict[tuple[str, str], dict[str, Any]] = {}
        for label, policy in (("parent", parent), ("candidate", candidate)):
            package = root / label / "submission"
            shutil.copytree(submission.resolve(), package, symlinks=False)
            _install_search_policy(package, policy)
            manifest, _ = _build(package)
            _freeze_tree(package)
            compile_root = root / label / "compile"; compile_root.mkdir()
            for row in capsules:
                compiled[label, str(row["id"])] = _compile_one(
                    package, manifest, row, "rvv", operation_codes, compile_root)

        observations = []
        for row in capsules:
            capsule_id = str(row["id"])
            results = {}
            for label in ("parent", "candidate"):
                spike_root = root / label / "spike"; spike_root.mkdir(exist_ok=True)
                results[label] = _grade_spike(
                    row, compiled[label, capsule_id], operation_codes, spike_root)
            base_record = compiled["parent", capsule_id]
            candidate_record = compiled["candidate", capsule_id]
            observations.append({
                "capsule_id": capsule_id,
                "family": row["family"],
                "baseline_cycles": int(results["parent"].get("spike_cycles", 0)),
                "candidate_cycles": int(results["candidate"].get("spike_cycles", 0)),
                "correctness_ok": bool(
                    base_record.get("ok") and candidate_record.get("ok") and
                    _search_spike_correct(results["parent"]) and
                    _search_spike_correct(results["candidate"])),
                "baseline_code_sha256": results["parent"].get("kernel_text_sha256"),
                "candidate_code_sha256": results["candidate"].get("kernel_text_sha256"),
                "code_digest_authority": "compiled_kernel_object_text_section",
                "screen_authority": "spike_rv64gcv_mcycle_trusted_harness",
                "parent_candidate_sha256": parent["candidate_sha256"],
                "candidate_sha256": candidate["candidate_sha256"],
            })
        return observations


def evaluate_public_policy_confirmation_stages(
    *,
    submission: Path,
    capsules: list[dict[str, Any]],
    parent: dict[str, Any],
    candidate: dict[str, Any],
    public_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run exactly the confirmation pre-K1 build/compile/Spike path, without paid K1 work.

    This is a calibration authority, not a cheaper search evaluator.  In particular, every
    runtime-parallel capsule is compiled in both ``rvv`` and ``rvv_multicore`` modes exactly as in
    :func:`evaluate_public_policy_k1`, and the same pre-K1 Spike gate is applied before returning.
    """
    operation_codes = _codes(public_rows, "operation")
    with tempfile.TemporaryDirectory(prefix="merlin-host-trusted-confirmation-calibration-") \
            as temporary:
        root = Path(temporary)
        packages: dict[str, Path] = {}
        manifests: dict[str, dict[str, Any]] = {}
        for label, policy in (("parent", parent), ("candidate", candidate)):
            package = root / label / "submission"
            shutil.copytree(submission.resolve(), package, symlinks=False)
            _install_search_policy(package, policy)
            manifest, _ = _build(package)
            _freeze_tree(package)
            packages[label] = package
            manifests[label] = manifest

        compiled: dict[tuple[str, str], dict[str, Any]] = {}
        compiled_k1: dict[tuple[str, str], dict[str, Any]] = {}
        for label in ("parent", "candidate"):
            compile_root = root / label / "compile"
            compile_root.mkdir()
            for row in capsules:
                capsule_id = str(row["id"])
                compiled[label, capsule_id] = _compile_one(
                    packages[label], manifests[label], row, "rvv", operation_codes,
                    compile_root)
                if row.get("family") == "runtime_parallel":
                    compiled_k1[label, capsule_id] = _compile_one(
                        packages[label], manifests[label], row, "rvv_multicore",
                        operation_codes, compile_root)
                else:
                    compiled_k1[label, capsule_id] = compiled[label, capsule_id]

        spike_records: dict[tuple[str, str], dict[str, Any]] = {}
        for label in ("parent", "candidate"):
            spike_root = root / label / "spike"
            spike_root.mkdir()
            for row in capsules:
                capsule_id = str(row["id"])
                spike_records[label, capsule_id] = _grade_spike(
                    row, compiled[label, capsule_id], operation_codes, spike_root)

        spike_gates = _require_pre_k1_spike_gates(
            capsules=capsules, compiled=compiled, compiled_k1=compiled_k1,
            spike_records=spike_records)
        return [{
            "capsule_id": str(row["id"]),
            "family": row["family"],
            "correctness_ok": True,
            "baseline_code_sha256": spike_records[
                "parent", str(row["id"])].get("kernel_text_sha256"),
            "candidate_code_sha256": spike_records[
                "candidate", str(row["id"])].get("kernel_text_sha256"),
            "spike_gates": spike_gates[str(row["id"])],
            "parent_candidate_sha256": parent["candidate_sha256"],
            "candidate_sha256": candidate["candidate_sha256"],
            "calibration_authority": "exact_confirmation_pre_k1_stages_without_k1",
        } for row in capsules]


def grade(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    contracts = {}
    for name, path in (("target_contract", args.target_contract),
                       ("dialect_plan", args.dialect_plan)):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise GradeError(f"{name} must be a YAML mapping")
        contracts[name] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    compiler_seal = _verify_compiler_seal(
        args.submission.resolve(), getattr(args, "compiler_seal", None))
    with tempfile.TemporaryDirectory(prefix="merlin-host-grader-") as temporary:
        root = Path(temporary)
        package = root / "submission"
        links = [path.relative_to(args.submission).as_posix()
                 for path in args.submission.rglob("*") if path.is_symlink()]
        if links:
            raise GradeError(f"submission symlinks are forbidden: {links[:8]}")
        metadata = [path.relative_to(args.submission).as_posix()
                    for path in args.submission.rglob("*")
                    if ".git" in path.relative_to(args.submission).parts]
        if metadata:
            raise GradeError(f"submission .git metadata is forbidden: {metadata[:8]}")
        shutil.copytree(args.submission.resolve(), package, symlinks=False)
        # Build while the sealed corpus has not been opened. The sandbox mounts only this copied
        # package and the toolchain, so build scripts cannot inspect any split through the host FS.
        manifest, build = _build(package)
        trusted_search = {"status": "not_required"}
        if getattr(args, "trusted_search_seal", None) is not None:
            seal = json.loads(args.trusted_search_seal.read_text(encoding="utf-8"))
            checks = {
                "driver_verified": seal.get("status") == "pass",
                "policy_byte_match": seal.get("selected_policy_sha256") == build["policy_sha256"],
                "independent_convergence_sweep":
                    seal.get("checks", {}).get("independent_convergence_sweep") is True,
                "deterministic_replay": seal.get("checks", {}).get("deterministic_replay") is True,
                "heldout_never_opened": seal.get("checks", {}).get("heldout_never_opened") is True,
            }
            if not all(checks.values()):
                raise GradeError(f"trusted search seal failed before heldout access: {checks}")
            trusted_search = {"status": "pass", "checks": checks,
                              "seal_sha256": _sha256(args.trusted_search_seal)}
        splits = _validate_corpus(args.train, args.validation, args.heldout)
        operation_codes = _codes(
            [*splits["train"], *splits["validation"], *splits["heldout"]], "operation")
        forbidden = _scan_forbidden(package, splits["heldout"])
        if forbidden:
            raise GradeError(f"submission contains sealed capsule identities: {forbidden}")
        _freeze_tree(package)
        semantic_selected = _select_semantic_coverage(splits["heldout"])
        tail_selected = _select_tail_coverage(splits["heldout"], split_name="heldout")
        multicore_row = _select_multicore(splits["heldout"], split_name="heldout")
        selected_by_id = {row["id"]: row for row in
                          [*semantic_selected, *tail_selected, multicore_row]}
        selected = [selected_by_id[key] for key in sorted(selected_by_id)]
        compile_root = root / "compile"; compile_root.mkdir()
        records = []
        for row in selected:
            records.append(_compile_one(package, manifest, row, "scalar", operation_codes,
                                        compile_root))
            records.append(_compile_one(package, manifest, row, "rvv", operation_codes,
                                        compile_root))
        records.append(_compile_one(package, manifest, multicore_row, "rvv_multicore",
                                    operation_codes, compile_root))
        paired = defaultdict(dict)
        for record in records:
            paired[record["capsule"]][record["mode"]] = record
        code_changes = {
            capsule: (modes.get("scalar", {}).get("source_sha256") !=
                      modes.get("rvv", {}).get("source_sha256"))
            for capsule, modes in paired.items()
        }
        l0_ok = all(record.get("ok") for record in records) and all(code_changes.values())
        native_root = root / "native"; native_root.mkdir()
        by_capsule = {record["capsule"]: record for record in records if record["mode"] == "scalar"}
        l1_records = [_grade_native(row, by_capsule[row["id"]], operation_codes, native_root)
                      for row in selected]
        l1_ok = all(record["status"] == "pass" for record in l1_records)
        rvv_by_capsule = {record["capsule"]: record for record in records if record["mode"] == "rvv"}
        spike_root = root / "spike"; spike_root.mkdir()
        l2_records = [_grade_spike(row, rvv_by_capsule[row["id"]], operation_codes, spike_root)
                      for row in tail_selected]
        l2_ok = all(record["status"] == "pass" for record in l2_records)
        if getattr(args, "run_l3", True):
            k1_root = root / "k1"; k1_root.mkdir()
            scalar_by_capsule = {record["capsule"]: record for record in records
                                 if record["mode"] == "scalar"}
            multicore_record = next(record for record in records
                                    if record["mode"] == "rvv_multicore")
            connection = _k1_connection()
            with _k1_lock(connection):
                l3_records = []
                for row in tail_selected:
                    l3_records.append(_grade_k1(row, scalar_by_capsule[row["id"]],
                                                operation_codes, k1_root))
                    l3_records.append(_grade_k1(row, rvv_by_capsule[row["id"]],
                                                operation_codes, k1_root))
                l3_records.append(_grade_k1(multicore_row, multicore_record,
                                            operation_codes, k1_root))
            l3_ok = all(record["status"] == "pass" for record in l3_records)
            l3 = {"status": "pass" if l3_ok else "fail", "records": l3_records,
                  "authority": "spacemit_k1_linux_csr_and_proc_monitor"}
        else:
            l3 = {"status": "not_run", "reason": "disabled by non-scoring caller"}
        public_records = [{key: value for key, value in record.items() if not key.startswith("_")}
                          for record in records]
        levels = {
            "L0": {"status": "pass" if l0_ok else "fail", "records": public_records,
                   "scalar_rvv_source_change": code_changes},
            "L1": {"status": "pass" if l1_ok else "fail", "records": l1_records,
                   "authority": "native_scalar_reference_with_asan_ubsan_and_guards"},
            "L2": {"status": "pass" if l2_ok else "fail", "records": l2_records,
                   "authority": "spike_rv64gcv_vlen256"},
            "L3": l3,
        }
        return {"version": 1, "status": "pass" if all(
                    value["status"] == "pass" for value in levels.values()) else "fail",
                "implemented_levels": list(IMPLEMENTED_LEVELS), "levels": levels,
                "build": build, "selected_capsules": [row["id"] for row in selected],
                "tail_capsules": [row["id"] for row in tail_selected],
                "multicore_capsule": multicore_row["id"],
                "trusted_search": trusted_search,
                "compiler_seal": compiler_seal,
                "contracts": contracts,
                "wall_seconds": time.monotonic() - started}


def _self_check() -> dict[str, Any]:
    repo = _repo_root()
    llvm = repo / "third_party" / "llvm-install" / "bin"
    spike = _spike_tools(); connection = _k1_connection(); k1_cc = _k1_cc()
    native_cc = shutil.which("clang")
    descriptor_rows = capsule_descriptor.conformance_rows()
    descriptor_checks: list[dict[str, Any]] = []
    descriptor_ready = (
        len(descriptor_rows) == len(FAMILY_CODE) and
        {str(row["family"]) for row in descriptor_rows} == set(FAMILY_CODE) and
        _codes(list(descriptor_rows), "operation") == OPERATION_CODE)
    if (llvm / "mlir-opt").is_file():
        with tempfile.TemporaryDirectory(prefix="merlin-descriptor-self-check-") as temporary:
            for row in descriptor_rows:
                path = Path(temporary) / f'{row["family"]}.mlir'
                try:
                    rendered = capsule_descriptor.render_capsule_mlir(row)
                    path.write_text(rendered, encoding="utf-8")
                    checked = subprocess.run(
                        [str(llvm / "mlir-opt"), "--allow-unregistered-dialect", str(path)],
                        capture_output=True, text=True, timeout=30)
                    ok = checked.returncode == 0
                    descriptor_checks.append({
                        "family": row["family"], "verified": ok,
                        "descriptor_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                        "stderr_tail": checked.stderr[-1000:],
                    })
                    descriptor_ready = descriptor_ready and ok
                except (OSError, subprocess.TimeoutExpired, TypeError, ValueError) as exc:
                    descriptor_ready = False
                    descriptor_checks.append({
                        "family": row.get("family"), "verified": False,
                        "error": str(exc),
                    })
    else:
        descriptor_ready = False
    board_reachable = False
    if connection["host"] and Path(connection["key"]).is_file() and shutil.which("ssh"):
        try:
            probe = subprocess.run(_ssh_argv(connection) + ["true"], capture_output=True,
                                   text=True, timeout=15)
            board_reachable = probe.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            pass
    level_ready = {
        "L0": bool(shutil.which("bwrap") and (llvm / "clang").is_file()
                   and (llvm / "mlir-opt").is_file() and descriptor_ready),
        "L1": bool(native_cc and Path(native_cc).is_file()
                   and Path(__file__).with_name("trusted_harness.c").is_file()),
        "L2": all(path.is_file() for path in spike.values()),
        "L3": bool(k1_cc and board_reachable and
                   Path(__file__).with_name("k1_monitor.py").is_file()),
    }
    return {"version": 1, "implemented_levels": list(IMPLEMENTED_LEVELS),
            "level_ready": level_ready, "bwrap": shutil.which("bwrap"),
            "clang": str(llvm / "clang"), "native_clang": native_cc,
            "mlir_opt": str(llvm / "mlir-opt"),
            "capsule_descriptor": {
                "path": str(Path(capsule_descriptor.__file__).resolve()),
                "sha256": _sha256(Path(capsule_descriptor.__file__).resolve()),
                "stable_operation_table_sha256": hashlib.sha256(json.dumps(
                    OPERATION_CODE, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "fixture_count": len(descriptor_checks),
                "families": sorted(str(row["family"]) for row in descriptor_rows),
                "checks": descriptor_checks,
                "ready": descriptor_ready,
            },
            "spike": {name: str(path) for name, path in spike.items()},
            "k1_compiler": str(k1_cc) if k1_cc else None,
            "k1_board_reachable": board_reachable,
            "trusted_search": {
                "outside_sandbox_broker_api": callable(evaluate_public_policy_k1),
                "paired_measurements": 6,
                "authorities": ["spike_rv64gcv", "spacemit_k1_linux"],
                "heldout_argument": False,
            },
            "ready": all(level_ready.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--validate-corpus", action="store_true")
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--target-contract", type=Path)
    parser.add_argument("--dialect-plan", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trusted-search-seal", type=Path)
    parser.add_argument("--compiler-seal", type=Path)
    args = parser.parse_args(argv)
    if args.self_check:
        print(json.dumps(_self_check(), indent=2))
        return 0
    if args.validate_corpus:
        if any(value is None for value in (args.train, args.validation, args.heldout)):
            parser.error("corpus validation requires train, validation, and heldout")
        try:
            result = validate_corpus_for_grading(args.train, args.validation, args.heldout)
            returncode = 0
        except Exception as exc:
            result = {"version": 1, "ready": False,
                      "error": f"{type(exc).__name__}: {exc}"}
            returncode = 2
        print(json.dumps(result, sort_keys=True))
        return returncode
    required = (args.submission, args.target_contract, args.dialect_plan, args.train,
                args.validation, args.heldout, args.output)
    if any(value is None for value in required):
        parser.error("grading requires submission, contracts, all three splits, and output")
    result: dict[str, Any]
    try:
        result = grade(args)
        returncode = 0 if result["status"] == "pass" else 1
    except TreatmentBuildFailure as exc:
        result = {"version": 1, "status": "treatment_build_fail",
                  "failure_class": exc.failure_class,
                  "implemented_levels": list(IMPLEMENTED_LEVELS),
                  "reason": exc.reason, "build_failure": exc.evidence}
        returncode = 1
    except TreatmentSubmissionFailure as exc:
        result = {"version": 1, "status": "treatment_agent_fail",
                  "failure_class": exc.failure_class,
                  "implemented_levels": list(IMPLEMENTED_LEVELS),
                  "reason": str(exc)}
        returncode = 1
    except Exception as exc:  # fail closed, but always leave a machine-readable result
        result = {"version": 1, "status": "error", "failure_class": "harness_invalid",
                  "implemented_levels": list(IMPLEMENTED_LEVELS),
                  "error": f"{type(exc).__name__}: {exc}"}
        returncode = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
