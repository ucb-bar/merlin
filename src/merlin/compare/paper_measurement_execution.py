"""Execute admitted paper cells, probe the board and compare complete outputs."""

from __future__ import annotations

import json
import math
import platform
import struct
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .paper_ablation_generator import TRUSTED_K1_PROBE_SOURCE_SHA256
from .paper_measurement_build import _build_executable
from .paper_measurement_contract import _contract, _session_manifest
from .paper_measurement_process import _run_iteration
from .paper_measurement_protocol import (
    _FRAME_MAGIC,
    CONTROLLER_ID,
    _bound_file,
    _canonical_sha,
    _closed,
    _sha_bytes,
    _sha_file,
    entry_source,
)
from .paper_measurement_study import _validate_study_root
from .paper_model_object_builder import executorch_session_resources
from .paper_session_abi import decode_request, decode_response, descriptor_from_dict


def _compile_probe(source: Path, target: str, destination: Path, timeout: float) -> None:
    if _sha_file(source) != TRUSTED_K1_PROBE_SOURCE_SHA256 or source.suffix != ".c":
        raise ValueError("board probe differs from the shipped trusted K1 CSR/sysfs source")
    flags = ["-march=rv64gcv", "-mabi=lp64d"] if target == "k1" else []
    if target not in {"k1", "unit-test"}:
        raise ValueError("paper measurement controller supports only k1 (or unit-test)")
    if target == "k1" and not platform.machine().lower().startswith("riscv"):
        raise ValueError("K1 measurements must run the controller locally on the RISC-V board")
    completed = subprocess.run(
        ["/usr/bin/cc", "-O2", "-std=c11", *flags, str(source), "-o", str(destination)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0 or not destination.is_file() or not destination.read_bytes().startswith(b"\x7fELF"):
        raise ValueError(f"trusted board probe build failed: {completed.stderr[-500:]}")


def _probe(executable: Path, target: str, timeout: float) -> tuple[str, dict[str, Any]]:
    mode = "--unit-test-json" if target == "unit-test" else "--json"
    completed = subprocess.run([str(executable), mode], capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        raise ValueError(f"trusted board probe failed: {completed.stderr[-500:]}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("trusted board probe did not emit JSON") from error
    value = _closed(
        value,
        {
            "schema_version",
            "kind",
            "identity",
            "vlen_bits",
            "vlen_source",
            "governor",
            "current_khz",
            "max_khz",
            "max_thermal_millic",
        },
        "trusted board probe output",
    )
    if (
        value["schema_version"] != 1
        or value["kind"] != "merlin_board_probe_v1"
        or value["vlen_source"] != "csr"
        or value["governor"] != "performance"
        or value["current_khz"] != value["max_khz"]
        or any(
            type(value[field]) is not int or value[field] <= 0
            for field in ("vlen_bits", "current_khz", "max_khz", "max_thermal_millic")
        )
    ):
        raise ValueError("trusted board probe does not establish the locked K1 regime")
    return completed.stdout, dict(value)


def _frequency_rows(target: str, core_ids: list[int], base: Mapping[str, Any]) -> list[dict]:
    rows = []
    for core in core_ids:
        if target == "unit-test":
            governor = str(base["governor"])
            current, maximum = int(base["current_khz"]), int(base["max_khz"])
        else:
            cpufreq = Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq")
            try:
                governor = (cpufreq / "scaling_governor").read_text().strip()
                current = int((cpufreq / "scaling_cur_freq").read_text().strip())
                maximum = int((cpufreq / "cpuinfo_max_freq").read_text().strip())
            except (OSError, ValueError) as error:
                raise ValueError(f"cannot establish frequency for requested CPU {core}") from error
        if governor != "performance" or current != maximum or current <= 0:
            raise ValueError(f"requested CPU {core} is outside the frozen frequency regime")
        rows.append({"core_id": core, "governor": governor, "current_khz": current, "max_khz": maximum})
    return rows


def _frames(value: bytes, expected_steps: int, label: str) -> list[bytes]:
    if not value.startswith(_FRAME_MAGIC):
        raise ValueError(f"{label} is not a controller framed trajectory")
    offset, frames = len(_FRAME_MAGIC), []
    while offset < len(value):
        if len(value) - offset < 8:
            raise ValueError(f"{label} has a truncated frame header")
        length = struct.unpack_from("<Q", value, offset)[0]
        offset += 8
        if length <= 0 or length > len(value) - offset:
            raise ValueError(f"{label} has an invalid frame length")
        frames.append(value[offset : offset + length])
        offset += length
    if len(frames) != expected_steps:
        raise ValueError(f"{label} has {len(frames)} frames, expected exactly {expected_steps}")
    return frames


def _frame_score(observed: bytes, reference: bytes, kind: str) -> float:
    if kind == "bytes_exact":
        return 1.0 if observed == reference else 0.0
    if kind == "float32_cosine":
        if len(observed) != len(reference) or not observed or len(observed) % 4:
            return -1.0
        lhs = struct.unpack(f"<{len(observed) // 4}f", observed)
        rhs = struct.unpack(f"<{len(reference) // 4}f", reference)
        dot = sum(a * b for a, b in zip(lhs, rhs))
        norm = math.sqrt(sum(a * a for a in lhs) * sum(b * b for b in rhs))
        return dot / norm if norm and all(math.isfinite(v) for v in (*lhs, *rhs)) else -1.0
    if len(observed) != len(reference) or not observed or len(observed) % 8:
        return -1.0
    lhs = struct.unpack(f"<{len(observed) // 8}q", observed)
    rhs = struct.unpack(f"<{len(reference) // 8}q", reference)
    return sum(a == b for a, b in zip(lhs, rhs)) / len(lhs)


def _oracle(observed: bytes, reference: bytes, contract: Mapping[str, Any], *, descriptor=None) -> tuple[dict, dict]:
    oracle = contract["oracle"]
    kind, threshold = oracle["kind"], float(oracle["threshold"])
    if "session_abi" in contract:
        if descriptor is None:
            raise ValueError("MRLNSES2 oracle requires its bound public descriptor")
        lhs_response = decode_response(observed, expected_descriptor=descriptor)
        rhs_response = decode_response(reference, expected_descriptor=descriptor)
        lhs = [frame.payload for frame in lhs_response.outputs]
        rhs = [frame.payload for frame in rhs_response.outputs]
        if len(lhs) != int(oracle["steps"]):
            raise ValueError("MRLNSES2 response trajectory differs from the frozen oracle")
    else:
        lhs = _frames(observed, int(oracle["steps"]), "observed output")
        rhs = _frames(reference, int(oracle["steps"]), "reference output")
    scores = [_frame_score(got, want, kind) for got, want in zip(lhs, rhs, strict=True)]
    per_step = [{"index": index, "value": score, "gate_ok": score >= threshold} for index, score in enumerate(scores)]
    value = min(scores)
    gate = all(row["gate_ok"] for row in per_step)
    common = {
        "gate_ok": gate,
        "scope": oracle["scope"],
        "steps": oracle["steps"],
        "reference": "controller_bound_reference",
        "per_step": per_step,
    }
    return ({**common, "status": "pass" if gate else "fail"}, {**common, "metric": oracle["metric"], "value": value})


def _run_contract(contract_path: Path) -> dict[str, Any]:
    contract, cell = _contract(contract_path)
    root = contract_path.parent
    deadline_ns = time.monotonic_ns() + contract["timeout_seconds"] * 1_000_000_000

    def remaining_seconds() -> float:
        remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining <= 0:
            raise TimeoutError("paper measurement exceeded its whole-cell deadline")
        return remaining

    with tempfile.TemporaryDirectory(prefix="merlin-paper-controller-") as temporary_value:
        temporary = Path(temporary_value)
        executable = temporary / "cell_executable"
        # Executable materialization, including registry-owned model-object regeneration or sealed
        # ExecuTorch producer verification, must be complete
        # before the controller opens the canonical study's private input/reference authorities.
        # `_contract` above validates only the closed plan shape and does not resolve those files.
        build_receipt = _build_executable(root, contract, executable, remaining_seconds)
        _validate_study_root(root, contract)
        artifact = _bound_file(root, contract["artifact"], "cell artifact")
        if contract["registry_id"] != "unit_test_v1":
            model_build_input = _bound_file(root, contract["build"]["inputs"]["model_artifact"], "build model artifact")
            if _sha_file(model_build_input) != _sha_file(artifact):
                raise ValueError("production build did not consume the bound runtime model artifact")
        inputs = {name: _bound_file(root, ref, f"cell input {name}") for name, ref in contract["inputs"].items()}
        reference = _bound_file(root, contract["reference_output"], "reference output")
        executorch_package = None
        if contract["registry_id"] == "executorch_v1" and contract["target"] != "unit-test":
            compiler_input = _bound_file(root, contract["build"]["inputs"]["compiler_input"], "compiler input")
            sealed = executorch_session_resources(compiler_input, include_private=True)
            from merlin.evaluation.frameworks.executorch_session import load_session_package

            executorch_package = load_session_package(sealed.package_root)
            logical_stages = list(executorch_package.plan.logical_stages or executorch_package.plan.stages)
            timed_stages = list(
                executorch_package.plan.parameters.get("timed_stages", logical_stages) or logical_stages
            )
            if (
                _sha_file(sealed.runner) != _sha_file(executable)
                or executorch_package.plan.warmups != contract["warmup_iterations"]
                or executorch_package.plan.repeats != contract["measured_iterations"]
                or executorch_package.plan.observations != contract["session"]["observations"]
                or timed_stages != contract["timing"]["timed_stages"]
                or executorch_package.plan.stage_attribution != contract["frozen_provenance"].get("stage_attribution")
            ):
                raise ValueError("sealed ExecuTorch executable/session cadence differs from measurement contract")
        session_descriptor = None
        session_request = None
        if "session_abi" in contract:
            descriptor_path = _bound_file(root, contract["session_abi"]["descriptor"], "MRLNSES2 descriptor")
            session_descriptor = descriptor_from_dict(json.loads(descriptor_path.read_text(encoding="ascii")))
            session_request = inputs["session_request"].read_bytes()
            decode_request(session_request, expected_descriptor=session_descriptor)
            decode_response(reference.read_bytes(), expected_descriptor=session_descriptor)
        # ``artifact_sha256`` is the frozen capture/tree identity used by the paper matrix.  The
        # concrete package is independently byte-bound by ``artifact``; these identities differ
        # because real captures are directory digests, not one file's byte digest.
        _session_manifest(
            root, contract["session_manifest"], contract["session"], inputs, descriptor=session_descriptor
        )
        probe_source = _bound_file(root, contract["board_probe_source"], "board probe source")
        probe_executable = temporary / "paper_k1_board_probe"
        _compile_probe(probe_source, contract["target"], probe_executable, remaining_seconds())
        before_text, before = _probe(probe_executable, contract["target"], remaining_seconds())
        core_ids = list(contract["execution"]["core_ids"])
        before_cores = _frequency_rows(contract["target"], core_ids, before)
        replacements = {
            "{executable}": str(executable),
            "{artifact}": str(artifact),
            "{package_root}": (str(executorch_package.root) if executorch_package is not None else ""),
            "{core_count}": str(len(core_ids)),
            **{f"{{input:{name}}}": str(value) for name, value in inputs.items()},
        }
        iterations, outputs, stdouts, stderrs, measured = [], [], [], [], []
        started_all = time.monotonic_ns()
        external_stage_samples = None
        if executorch_package is not None:
            observation = temporary / "executorch-trajectory.bin"
            replacements["{observation}"] = str(observation)
            argv = [replacements.get(part, part) for part in contract["argv"]]
            row, trajectory, stdout, stderr = _run_iteration(
                argv,
                cwd=root,
                environment=contract["environment"],
                core_ids=core_ids,
                observation=observation,
                timeout=remaining_seconds(),
                phase="sealed_session",
                index=0,
                allow_stdout=True,
            )
            from merlin.evaluation.frameworks.executorch_session import parse_session_console

            parsed = parse_session_console(
                (stdout + stderr).decode(errors="replace"),
                executorch_package.plan,
                requested_cores=len(core_ids),
                trajectory=observation,
            )
            element_bytes = executorch_package.plan.observation_output.tensor.nbytes
            if len(trajectory) != element_bytes * contract["session"]["observations"]:
                raise ValueError("ExecuTorch trajectory byte count differs from its session ABI")
            framed = _FRAME_MAGIC + b"".join(
                struct.pack("<Q", element_bytes) + trajectory[offset : offset + element_bytes]
                for offset in range(0, len(trajectory), element_bytes)
            )
            # The receipt's observation identity is the canonical framed trajectory consumed by
            # the independent oracle, not the runner's transport-only raw tensor concatenation.
            row["observation_sha256"] = _sha_bytes(framed)
            iterations.append(row)
            outputs.append(framed)
            stdouts.append(stdout)
            stderrs.append(stderr)
            measured.extend(parsed.samples)
            external_stage_samples = (
                {}
                if executorch_package.plan.stage_attribution == "opaque_whole_forward"
                else {name: list(values) for name, values in parsed.stage_samples.items()}
            )
        else:
            total = contract["warmup_iterations"] + contract["measured_iterations"]
            for ordinal in range(total):
                phase = "warmup" if ordinal < contract["warmup_iterations"] else "measured"
                index = ordinal if phase == "warmup" else ordinal - contract["warmup_iterations"]
                observation = temporary / f"observation-{phase}-{index}.bin"
                replacements["{observation}"] = str(observation)
                argv = [replacements.get(part, part) for part in contract["argv"]]
                row, output, stdout, stderr = _run_iteration(
                    argv,
                    cwd=root,
                    environment=contract["environment"],
                    core_ids=core_ids,
                    observation=observation,
                    timeout=remaining_seconds(),
                    phase=phase,
                    index=index,
                    request=session_request,
                )
                iterations.append(row)
                outputs.append(output)
                stdouts.append(stdout)
                stderrs.append(stderr)
                if phase == "measured":
                    measured.append(row["elapsed_ns"])
        ended_all = time.monotonic_ns()
        after_text, after = _probe(probe_executable, contract["target"], remaining_seconds())
        after_cores = _frequency_rows(contract["target"], core_ids, after)
    if len({_sha_bytes(output) for output in outputs}) != 1:
        raise ValueError("functional output changed across complete-session repetitions")
    if len({_sha_bytes(v) for v in stdouts}) != 1 or len({_sha_bytes(v) for v in stderrs}) != 1:
        raise ValueError("process capture changed across complete-session repetitions")
    if before["identity"] != after["identity"] or before["vlen_bits"] != after["vlen_bits"]:
        raise ValueError("board identity or VLEN changed during measurement")
    correctness, quality = _oracle(outputs[0], reference.read_bytes(), contract, descriptor=session_descriptor)
    status = "pass" if correctness["gate_ok"] and quality["gate_ok"] else "fail"
    ordered, count = sorted(measured), len(measured)
    median = ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) // 2
    p95 = ordered[min(count - 1, max(0, int(round(0.95 * (count - 1)))))]
    max_tasks = max(row["max_task_count"] for row in iterations)
    execution = {
        "mode": contract["execution"]["mode"],
        "requested_mode": contract["execution"]["mode"],
        "fallback_used": False,
        "core_count": len(core_ids),
        "requested_core_count": len(core_ids),
        "affinity_source": "sched_getaffinity",
        "semantic_session": True,
        "same_input_repetition": False,
    }
    if contract["execution"]["require_worker_threads"]:
        if max_tasks != len(core_ids):
            raise ValueError("externally observed task count differs from requested worker threads")
        execution.update(worker_threads=max_tasks, worker_thread_source="proc_task_status")
    board_conditions = {
        endpoint: {
            "governor": "performance",
            "current_khz": min(row["current_khz"] for row in rows),
            "max_khz": min(row["max_khz"] for row in rows),
            "max_thermal_millic": board["max_thermal_millic"],
        }
        for endpoint, rows, board in (("before", before_cores, before), ("after", after_cores, after))
    }
    runtime_argv = [
        (
            {
                "{executable}": "<rebuilt-executable>",
                "{artifact}": str(artifact),
                "{package_root}": "<retained-executorch-package>",
                "{core_count}": str(len(core_ids)),
                **{f"{{input:{name}}}": str(value) for name, value in inputs.items()},
                "{observation}": "<controller-observation>",
            }
        ).get(part, part)
        for part in contract["argv"]
    ]
    return {
        "schema_version": 5,
        "kind": "paper_controller_raw_measurement_v5",
        "status": "complete",
        "run_id": contract["run_id"],
        "study_sha256": contract["study_sha256"],
        "cell": cell,
        "contract_sha256": _sha_file(contract_path),
        "build_receipt": build_receipt,
        "driver": {
            "id": CONTROLLER_ID,
            "source_sha256": _sha_file(entry_source()),
            "argv": runtime_argv,
            "command_sha256": _canonical_sha(runtime_argv),
            "exit_code": 0,
            "started_monotonic_ns": started_all,
            "ended_monotonic_ns": ended_all,
            "stdout_sha256": _sha_bytes(stdouts[0]),
            "stderr_sha256": _sha_bytes(stderrs[0]),
            "iterations": iterations,
        },
        "artifact_sha256": contract["artifact_sha256"],
        "functional_output_sha256": _sha_bytes(outputs[0]),
        # ``built`` is the longstanding normalized lifecycle name.  For ExecuTorch it means that
        # the sealed executable was materialized and verified, not that K1 compiled it; the build
        # receipt's ``operation`` makes that distinction machine-checkable.
        "lifecycle": {
            "built": True,
            "ran": True,
            "status": status,
            "reason": None if status == "pass" else "controller output oracle failed",
        },
        "session": contract["session"],
        "correctness": correctness,
        "quality": quality,
        "memory": {
            "policy": contract["memory_policy"],
            "peak_rss_bytes": max(row["peak_rss_bytes"] for row in iterations),
        },
        "timing": {
            **dict(contract["timing"]),
            **({"stage_samples": external_stage_samples} if external_stage_samples is not None else {}),
            "samples": measured,
            "median": median,
            "p95": p95,
        },
        "execution": execution,
        "provenance": {
            **contract["frozen_provenance"],
            "binary": build_receipt["executable_sha256"],
            "vlen_bits": before["vlen_bits"],
            "vlen_source": before["vlen_source"],
            "board_conditions": board_conditions,
        },
        "board_receipts": {
            "before": {"probe": before_text, "cores": before_cores},
            "after": {"probe": after_text, "cores": after_cores},
        },
        "_captured_stdout": stdouts[0],
        "_captured_stderr": stderrs[0],
    }
