"""Trusted, non-agentic producers and analyzers for paper causal evidence.

``benchmark`` executes a predeclared executable plus an untimed, source-pinned K1 board probe and
retains its raw JSON outputs. ``inspect`` derives structural counters from measured binaries. ``observe`` and
``summarize`` replay the raw benchmark receipt into canonical result files. None of the replay
actions accepts authored samples, counters, or explanation prose.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import yaml


GENERATOR_ID = "merlin.compare.paper_ablation_summary_v2"
MEASUREMENT_TOOL_ID = "merlin.compare.paper_ablation_benchmark_v2"
STRUCTURAL_TOOL_ID = "merlin.compare.paper_structural_analyzer_v2"
TRUSTED_K1_PROBE_SOURCE_SHA256 = (
    "9f48d7a584578bab3657fe069b293ae8990dbd9acfde1beab0944ea389bc479b")

_HEX = frozenset("0123456789abcdef")
_STRUCTURAL_MECHANISMS = {
    # The compiler/runtime emits this named audit marker at every retained runtime-dispatch site.
    # The trusted analyzer counts the marker directly in the reproduced measured executable.
    "runtime_dispatch_markers": (
        "retained runtime-dispatch sites", "lower_is_better",
        b"MERLIN_RUNTIME_DISPATCH_MARKER"),
}
_PAIR_SENTINEL = "/* MERLIN_TYPED_TRANSFORM:runtime_dispatch_elimination_v1 */"
_PAIR_FRAGMENTS = {
    "control": "runtime_dispatch(value);\n  runtime_dispatch(value);",
    "treatment": "runtime_dispatch(value);",
}
PAIR_CONTROLLER_ID = "merlin.compare.paper_causal_pair_controller_v1"
_IDENTITY_FIELDS = (
    "binding_sha256", "variant", "target", "model", "precision", "core_count",
    "backend", "package_sha256", "source_sha256", "runtime_sha256", "capture_sha256",
    "capture_session_identity_sha256", "session_protocol_sha256", "artifact_sha256",
    "run_id", "command_sha256", "metric", "direction",
)
_RAW_FIELDS = frozenset({
    "schema_version", "kind", "status", *_IDENTITY_FIELDS,
    "producer", "build_receipt_sha256", "executable_sha256", "execution_argv",
    "benchmark_contract_sha256", "samples_ns", "functional_stdout_sha256",
    "measurement_clock", "warmup_iterations", "measured_iterations", "board_probe_argv",
    "board_probe_command_sha256", "board_receipts",
})
_OBSERVATION_FIELDS = frozenset({
    "schema_version", "kind", "status", *_IDENTITY_FIELDS, "producer_id",
    "producer_source_sha256", "build_receipt_sha256", "executable_sha256",
    "benchmark_contract_sha256", "board_receipts_sha256", "samples",
})


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_sha(path: Path) -> str:
    """Match ``freeze.sha256_paths([one_file])`` without importing the package when copied."""
    digest = hashlib.sha256()
    digest.update(b"F\0" + f"0:{path.name}".encode("utf-8") + b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
    return digest.hexdigest()


def _content_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in _HEX for character in text)


def _closed(value: object, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    extra, missing = sorted(set(value) - fields), sorted(fields - set(value))
    if extra or missing:
        raise ValueError(f"{label} is closed; unrecognized={extra} missing={missing}")
    return value


def _json_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be retained JSON stdout text")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not JSON: {error}") from error
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} must decode to a mapping")
    return parsed


def _validate_samples(values: object, label: str) -> list[int]:
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                   for value in values)):
        raise ValueError(f"{label} requires positive integer samples")
    return list(values)


def _board_receipt(stdout: object, label: str) -> dict[str, Any]:
    raw = _closed(_json_mapping(stdout, label), frozenset({
        "schema_version", "kind", "identity", "vlen_bits", "vlen_source", "governor",
        "current_khz", "max_khz", "max_thermal_millic",
    }), label)
    if (raw["schema_version"] != 1 or raw["kind"] != "merlin_board_probe_v1"
            or not str(raw["identity"]).strip() or raw["vlen_source"] != "csr"):
        raise ValueError(f"{label} has invalid identity/VLEN provenance")
    numeric = ("vlen_bits", "current_khz", "max_khz", "max_thermal_millic")
    if any(not isinstance(raw[key], int) or isinstance(raw[key], bool) or raw[key] <= 0
           for key in numeric):
        raise ValueError(f"{label} has invalid numeric board observations")
    if raw["governor"] != "performance" or raw["current_khz"] != raw["max_khz"]:
        raise ValueError(f"{label} does not establish the locked performance-frequency regime")
    return dict(raw)


def observe(raw_log: Mapping[str, Any], *, generator_source_sha256: str) -> dict[str, Any]:
    """Derive samples and board facts from a closed production benchmark receipt."""
    raw_log = _closed(raw_log, _RAW_FIELDS, "raw benchmark receipt")
    if (raw_log["schema_version"] != 2 or raw_log["kind"] != "frozen_ablation_raw_log"
            or raw_log["status"] != "pass"):
        raise ValueError("raw benchmark receipt must be a passing schema-v2 receipt")
    producer = _closed(raw_log["producer"], frozenset({
        "id", "source_sha256", "command",
    }), "raw benchmark receipt producer")
    if (producer["id"] != MEASUREMENT_TOOL_ID
            or producer["source_sha256"] != generator_source_sha256
            or not _is_sha(generator_source_sha256)):
        raise ValueError("raw benchmark receipt producer is not the bound trusted tool")
    if producer["command"] != [
            "python3", "paper_ablation_generator.py", "benchmark", "{contract}", "{raw_log}"]:
        raise ValueError("raw benchmark receipt producer command is not canonical")
    for field in ("binding_sha256", "package_sha256", "source_sha256", "runtime_sha256",
                  "capture_sha256", "capture_session_identity_sha256", "session_protocol_sha256",
                  "artifact_sha256", "command_sha256", "build_receipt_sha256",
                  "executable_sha256", "benchmark_contract_sha256",
                  "board_probe_command_sha256"):
        if not _is_sha(raw_log[field]):
            raise ValueError(f"raw benchmark receipt has invalid {field}")
    argv = raw_log["execution_argv"]
    probe_argv = raw_log["board_probe_argv"]
    for name, command, digest in (("execution", argv, raw_log["command_sha256"]),
                                  ("board probe", probe_argv,
                                   raw_log["board_probe_command_sha256"])):
        if (not isinstance(command, list) or not command
                or any(not isinstance(part, str) or not part for part in command)
                or _canonical_sha(command) != digest):
            raise ValueError(f"raw benchmark receipt {name} argv/digest differs")
    samples = _validate_samples(raw_log["samples_ns"], "trusted benchmark receipt")
    if (raw_log["measurement_clock"] != "time.monotonic_ns"
            or not _is_sha(raw_log["functional_stdout_sha256"])
            or not isinstance(raw_log["warmup_iterations"], int)
            or isinstance(raw_log["warmup_iterations"], bool)
            or raw_log["warmup_iterations"] < 0
            or not isinstance(raw_log["measured_iterations"], int)
            or isinstance(raw_log["measured_iterations"], bool)
            or raw_log["measured_iterations"] != len(samples)):
        raise ValueError("benchmark samples lack trusted monotonic-clock production provenance")
    receipts = _closed(raw_log["board_receipts"], frozenset({"before", "after"}),
                       "board receipts")
    board = {endpoint: _board_receipt(receipts[endpoint], f"board {endpoint} receipt")
             for endpoint in ("before", "after")}
    if board["before"]["identity"] != board["after"]["identity"]:
        raise ValueError("board identity changed across benchmark endpoints")
    return {
        "schema_version": 2,
        "kind": "frozen_ablation_observation", "status": "pass",
        **{field: raw_log[field] for field in _IDENTITY_FIELDS},
        "producer_id": MEASUREMENT_TOOL_ID,
        "producer_source_sha256": generator_source_sha256,
        "build_receipt_sha256": raw_log["build_receipt_sha256"],
        "executable_sha256": raw_log["executable_sha256"],
        "benchmark_contract_sha256": raw_log["benchmark_contract_sha256"],
        "board_receipts_sha256": _canonical_sha(receipts),
        "samples": samples,
    }


def summarize(observation: Mapping[str, Any], *, generator_source_sha256: str) -> dict[str, Any]:
    """Build the canonical result from a closed trusted observation."""
    observation = _closed(observation, _OBSERVATION_FIELDS, "ablation observation")
    if (observation["schema_version"] != 2
            or observation["kind"] != "frozen_ablation_observation"
            or observation["status"] != "pass"
            or observation["producer_id"] != MEASUREMENT_TOOL_ID
            or observation["producer_source_sha256"] != generator_source_sha256):
        raise ValueError("ablation observation is not from the bound trusted producer")
    samples = _validate_samples(observation["samples"], "ablation observation")
    ordered = sorted(samples)
    median = (ordered[len(ordered) // 2] if len(ordered) % 2
              else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) // 2)
    return {
        "schema_version": 2,
        "kind": "frozen_ablation_result", "status": "pass",
        "variant": observation["variant"],
        "binding_sha256": observation["binding_sha256"],
        "artifact_sha256": observation["artifact_sha256"],
        "build_receipt_sha256": observation["build_receipt_sha256"],
        "executable_sha256": observation["executable_sha256"],
        "benchmark_contract_sha256": observation["benchmark_contract_sha256"],
        "board_receipts_sha256": observation["board_receipts_sha256"],
        "generator_id": GENERATOR_ID,
        "generator_source_sha256": generator_source_sha256,
        "metric": observation["metric"], "direction": observation["direction"],
        "samples": samples, "median": median,
    }


def _read_bound(root: Path, value: object, label: str) -> tuple[Path, str]:
    value = _closed(value, frozenset({"path", "sha256"}), label)
    relative = Path(str(value["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}.path must be relative and contained")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label}.path escapes the inspection root") from error
    if not path.is_file() or _path_sha(path) != value["sha256"]:
        raise ValueError(f"{label} digest differs from retained artifact")
    return path, str(value["sha256"])


def inspect(inspection: Mapping[str, Any], *, root: Path,
            generator_source_sha256: str) -> dict[str, Any]:
    """Derive counters and prose directly from both exact measured executable artifacts."""
    inspection = _closed(inspection, frozenset({
        "schema_version", "kind", "status", "binding_sha256", "ablation_sha256", "mechanism",
        "intervention_id",
        "control_artifact_sha256", "treatment_artifact_sha256", "control_package_sha256",
        "treatment_package_sha256", "control_source_sha256", "treatment_source_sha256",
        "control_build_receipt_sha256", "treatment_build_receipt_sha256",
        "control_measurement_run_sha256", "treatment_measurement_run_sha256",
        "control_artifact", "treatment_artifact",
    }), "structural inspection contract")
    if (inspection["schema_version"] != 2
            or inspection["kind"] != "frozen_structural_inspection_contract"
            or inspection["status"] != "pass"):
        raise ValueError("structural inspection must be a passing schema-v2 contract")
    mechanism = str(inspection["mechanism"])
    if mechanism not in _STRUCTURAL_MECHANISMS:
        raise ValueError(f"unsupported structural mechanism {mechanism!r}")
    for field in set(inspection) - {"schema_version", "kind", "status", "mechanism",
                                      "control_artifact", "treatment_artifact",
                                      "intervention_id"}:
        if not _is_sha(inspection[field]):
            raise ValueError(f"structural inspection has invalid {field}")
    control_path, control_sha = _read_bound(root, inspection["control_artifact"],
                                             "control_artifact")
    treatment_path, treatment_sha = _read_bound(root, inspection["treatment_artifact"],
                                                 "treatment_artifact")
    if (control_sha != inspection["control_artifact_sha256"]
            or treatment_sha != inspection["treatment_artifact_sha256"]):
        raise ValueError("structural analyzer artifacts differ from the measured executables")
    label, direction, _retired_marker = _STRUCTURAL_MECHANISMS[mechanism]
    control = _dispatch_call_count(control_path)
    treatment = _dispatch_call_count(treatment_path)
    improved = treatment < control if direction == "lower_is_better" else treatment > control
    if not improved:
        raise ValueError("structural treatment does not improve the trace-derived mechanism")
    return {
        "schema_version": 2, "kind": "frozen_structural_result", "status": "pass",
        "binding_sha256": inspection["binding_sha256"],
        "ablation_sha256": inspection["ablation_sha256"], "mechanism": mechanism,
        "intervention_id": inspection["intervention_id"],
        "generator_id": STRUCTURAL_TOOL_ID,
        "generator_source_sha256": generator_source_sha256,
        "control_analyzed_artifact_sha256": control_sha,
        "treatment_analyzed_artifact_sha256": treatment_sha,
        "facts": [{"metric": mechanism, "control": control, "treatment": treatment,
                   "direction": direction, "delta": abs(control - treatment)}],
        "why": f"The trusted analyzer derives an improvement in {label} from the bound binaries.",
        "how": f"The bound treatment changes {label} from {control} to {treatment}.",
    }


def _dispatch_call_count(executable: Path) -> int:
    """Count executable dataflow edges to ``runtime_dispatch`` in trusted disassembly.

    Plain byte strings and source comments are deliberately not evidence.  The analyzer requires an
    ELF and asks the separately installed system objdump to decode call/jump instructions whose
    resolved target symbol is exactly ``runtime_dispatch``.
    """
    if not executable.read_bytes().startswith(b"\x7fELF"):
        raise ValueError("structural analyzer requires compiled ELF artifacts")
    completed = subprocess.run(
        ["/usr/bin/objdump", "-d", "--no-show-raw-insn", str(executable)],
        capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode != 0:
        raise ValueError(f"trusted objdump failed: {completed.stderr[-500:]}")
    count = 0
    for line in completed.stdout.splitlines():
        if "<" not in line or ">" not in line:
            continue
        prefix, target = line.split("<", 1)
        symbol = target.split(">", 1)[0].split("+", 1)[0]
        words = prefix.replace(":", " ").split()
        if symbol == "runtime_dispatch" and any(
                word in {"call", "callq", "jal", "jalr"} for word in words):
            count += 1
    return count


def _pair_contract(path: Path) -> tuple[dict[str, Any], Path, Path]:
    root = path.parent
    contract = _closed(_load(path), frozenset({
        "schema_version", "kind", "status", "binding_sha256", "target", "intervention_id",
        "canonical_source", "dispatch_package", "compiler_sha256", "objdump_sha256",
        "timeout_seconds",
        "warmup_iterations", "measured_iterations",
    }), "causal pair contract")
    if (contract["schema_version"] != 1
            or contract["kind"] != "paper_causal_pair_contract_v1"
            or contract["status"] != "ready"
            or contract["intervention_id"] != "runtime_dispatch_elimination_v1"
            or contract["target"] not in {"unit-test", "k1"}
            or (contract["target"] == "k1"
                and not platform.machine().lower().startswith("riscv"))
            or not _is_sha(contract["binding_sha256"])
            or not _is_sha(contract["compiler_sha256"])
            or _content_sha(Path("/usr/bin/cc")) != contract["compiler_sha256"]
            or not _is_sha(contract["objdump_sha256"])
            or _content_sha(Path("/usr/bin/objdump")) != contract["objdump_sha256"]
            or any(not isinstance(contract[field], int) or isinstance(contract[field], bool)
                   for field in ("timeout_seconds", "warmup_iterations", "measured_iterations"))
            or contract["timeout_seconds"] <= 0 or contract["warmup_iterations"] < 0
            or contract["measured_iterations"] <= 0):
        raise ValueError(
            "causal pair contract is invalid, off-target, or does not pin compiler/analyzer")
    canonical, _ = _read_bound(root, contract["canonical_source"], "canonical source")
    package, _ = _read_bound(root, contract["dispatch_package"], "dispatch package")
    source_text = canonical.read_text(encoding="utf-8")
    if (source_text.count(_PAIR_SENTINEL) != 1
            or any(token in source_text for token in ("__FILE__", "__DATE__", "__TIME__"))):
        raise ValueError("canonical source must contain exactly one reproducible typed transform site")
    return dict(contract), canonical, package


def _compile_pair_arm(source: Path, package: Path, output: Path, *, timeout: int) -> list[str]:
    argv = [
        "/usr/bin/cc", "-O2", "-std=c11", "-fno-lto", "-fno-inline",
        "-fno-ident", "-Wl,--build-id=none", str(source), str(package), "-o", str(output),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False,
                               env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    if (completed.returncode != 0 or not output.is_file()
            or not output.read_bytes().startswith(b"\x7fELF")):
        raise ValueError(f"typed causal-pair build failed: {completed.stderr[-500:]}")
    return argv


def _run_pair_arm(executable: Path, *, warmups: int, iterations: int,
                  timeout: int) -> tuple[list[int], str]:
    stdout_digests: list[str] = []
    samples: list[int] = []
    for index in range(warmups + iterations):
        started = time.monotonic_ns()
        completed = subprocess.run([str(executable)], capture_output=True, text=True,
                                   timeout=timeout, check=False,
                                   env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
        elapsed = time.monotonic_ns() - started
        if completed.returncode != 0 or completed.stderr:
            raise ValueError(
                f"causal-pair executable failed rc={completed.returncode}: {completed.stderr[-500:]}")
        stdout_digests.append(_functional_receipt(completed.stdout))
        if index >= warmups:
            samples.append(elapsed)
    if len(set(stdout_digests)) != 1:
        raise ValueError("causal-pair functional result changed across repetitions")
    return samples, stdout_digests[0]


def produce_causal_pair(contract_path: str | Path, output_dir: str | Path) -> Path:
    """Construct, compile, execute, and structurally inspect one controller-owned causal pair."""
    contract_path = Path(contract_path).resolve()
    contract, canonical, package = _pair_contract(contract_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    retained_canonical, retained_package = inputs / "canonical.c", inputs / "dispatch.c"
    shutil.copy2(canonical, retained_canonical)
    shutil.copy2(package, retained_package)
    retained_contract = dict(contract)
    retained_contract["canonical_source"] = {
        "path": retained_canonical.relative_to(output).as_posix(),
        "sha256": _path_sha(retained_canonical),
    }
    retained_contract["dispatch_package"] = {
        "path": retained_package.relative_to(output).as_posix(),
        "sha256": _path_sha(retained_package),
    }
    retained_contract_path = output / "pair-contract.yaml"
    retained_contract_path.write_text(yaml.safe_dump(retained_contract, sort_keys=True),
                                      encoding="utf-8")
    canonical_text = retained_canonical.read_text(encoding="utf-8")
    arm_rows: dict[str, dict[str, Any]] = {}
    functional: dict[str, str] = {}
    for arm in ("control", "treatment"):
        arm_dir = output / arm
        arm_dir.mkdir()
        generated = arm_dir / "generated.c"
        generated.write_text(canonical_text.replace(_PAIR_SENTINEL, _PAIR_FRAGMENTS[arm]),
                             encoding="utf-8")
        executable = arm_dir / "program"
        compile_argv = _compile_pair_arm(
            generated, retained_package, executable, timeout=contract["timeout_seconds"])
        samples, functional[arm] = _run_pair_arm(
            executable, warmups=contract["warmup_iterations"],
            iterations=contract["measured_iterations"], timeout=contract["timeout_seconds"])
        disassembly = arm_dir / "disassembly.txt"
        completed = subprocess.run(
            ["/usr/bin/objdump", "-d", "--no-show-raw-insn", str(executable)],
            capture_output=True, text=True, timeout=contract["timeout_seconds"], check=False)
        if completed.returncode != 0:
            raise ValueError(f"causal-pair objdump failed: {completed.stderr[-500:]}")
        disassembly.write_text(completed.stdout, encoding="utf-8")
        arm_rows[arm] = {
            "generated_source": {"path": generated.relative_to(output).as_posix(),
                                 "sha256": _path_sha(generated)},
            "executable": {"path": executable.relative_to(output).as_posix(),
                           "sha256": _path_sha(executable)},
            "disassembly": {"path": disassembly.relative_to(output).as_posix(),
                            "sha256": _path_sha(disassembly)},
            "compile_argv_shape": [
                "/usr/bin/cc", "-O2", "-std=c11", "-fno-lto", "-fno-inline",
                "-fno-ident", "-Wl,--build-id=none", "{source}", "{package}", "-o", "{output}"],
            "compile_argv_sha256": _canonical_sha([
                part if part not in {str(generated), str(retained_package), str(executable)}
                else ({str(generated): "{source}", str(retained_package): "{package}",
                       str(executable): "{output}"}[part]) for part in compile_argv]),
            "samples_ns": samples,
            "functional_stdout_sha256": functional[arm],
            "runtime_dispatch_calls": _dispatch_call_count(executable),
        }
    if functional["control"] != functional["treatment"]:
        raise ValueError("causal pair control/treatment functional outputs differ")
    receipt = {
        "schema_version": 1, "kind": "paper_causal_pair_receipt_v1", "status": "complete",
        "controller_id": PAIR_CONTROLLER_ID,
        "binding_sha256": contract["binding_sha256"],
        "intervention_id": contract["intervention_id"],
        "compiler_sha256": contract["compiler_sha256"],
        "objdump_sha256": contract["objdump_sha256"],
        "contract": {"path": retained_contract_path.name,
                     "sha256": _path_sha(retained_contract_path)},
        "canonical_source_sha256": _path_sha(retained_canonical),
        "dispatch_package_sha256": _path_sha(retained_package),
        "normalized_source_sha256": _content_sha(retained_canonical),
        "functional_stdout_sha256": functional["control"],
        "arms": arm_rows,
    }
    receipt_path = output / "pair-receipt.yaml"
    receipt_path.write_text(yaml.safe_dump(receipt, sort_keys=True), encoding="utf-8")
    return receipt_path


def verify_causal_pair(receipt_path: str | Path, *, expected_binding_sha256: str) -> dict[str, Any]:
    """Regenerate a pair from its one canonical source and verify every retained executable."""
    receipt_path = Path(receipt_path).resolve()
    root = receipt_path.parent
    receipt = _closed(_load(receipt_path), frozenset({
        "schema_version", "kind", "status", "controller_id", "binding_sha256",
        "intervention_id", "compiler_sha256", "objdump_sha256", "contract",
        "canonical_source_sha256",
        "dispatch_package_sha256", "normalized_source_sha256",
        "functional_stdout_sha256", "arms",
    }), "causal pair receipt")
    if (receipt["schema_version"] != 1 or receipt["kind"] != "paper_causal_pair_receipt_v1"
            or receipt["status"] != "complete" or receipt["controller_id"] != PAIR_CONTROLLER_ID
            or receipt["binding_sha256"] != expected_binding_sha256
            or receipt["intervention_id"] != "runtime_dispatch_elimination_v1"):
        raise ValueError("causal pair receipt identity differs")
    contract_path, _ = _read_bound(root, receipt["contract"], "causal pair contract")
    contract, canonical, package = _pair_contract(contract_path)
    if (contract["binding_sha256"] != expected_binding_sha256
            or receipt["compiler_sha256"] != contract["compiler_sha256"]
            or receipt["objdump_sha256"] != contract["objdump_sha256"]
            or receipt["canonical_source_sha256"] != _path_sha(canonical)
            or receipt["dispatch_package_sha256"] != _path_sha(package)
            or receipt["normalized_source_sha256"] != _content_sha(canonical)):
        raise ValueError("causal pair contract/source/package binding differs")
    arms = _closed(receipt["arms"], frozenset({"control", "treatment"}), "causal pair arms")
    functional: dict[str, str] = {}
    values: dict[str, int] = {}
    medians: dict[str, int] = {}
    canonical_text = canonical.read_text(encoding="utf-8")
    expected_shape = [
        "/usr/bin/cc", "-O2", "-std=c11", "-fno-lto", "-fno-inline", "-fno-ident",
        "-Wl,--build-id=none", "{source}", "{package}", "-o", "{output}"]
    for arm in ("control", "treatment"):
        row = _closed(arms[arm], frozenset({
            "generated_source", "executable", "disassembly", "compile_argv_shape",
            "compile_argv_sha256", "samples_ns", "functional_stdout_sha256",
            "runtime_dispatch_calls",
        }), f"causal pair {arm}")
        generated, _ = _read_bound(root, row["generated_source"], f"{arm} generated source")
        executable, executable_sha = _read_bound(root, row["executable"], f"{arm} executable")
        disassembly, _ = _read_bound(root, row["disassembly"], f"{arm} disassembly")
        expected_source = canonical_text.replace(_PAIR_SENTINEL, _PAIR_FRAGMENTS[arm])
        if (generated.read_text(encoding="utf-8") != expected_source
                or generated.read_text(encoding="utf-8").replace(
                    _PAIR_FRAGMENTS[arm], _PAIR_SENTINEL, 1) != canonical_text
                or row["compile_argv_shape"] != expected_shape
                or row["compile_argv_sha256"] != _canonical_sha(expected_shape)):
            raise ValueError("causal pair has a non-intervention source/config/build difference")
        if disassembly.read_text(encoding="utf-8").count("<runtime_dispatch>") < 1:
            raise ValueError("retained disassembly does not contain runtime_dispatch dataflow")
        samples = _validate_samples(row["samples_ns"], f"causal pair {arm}")
        if len(samples) != contract["measured_iterations"]:
            raise ValueError("causal pair sample count differs from its contract")
        with tempfile.TemporaryDirectory(prefix=f"merlin-pair-{arm}-") as temporary:
            temporary_root = Path(temporary)
            replay_source = temporary_root / "generated.c"
            replay_source.write_text(expected_source, encoding="utf-8")
            replay_executable = temporary_root / "program"
            _compile_pair_arm(replay_source, package, replay_executable,
                              timeout=contract["timeout_seconds"])
            if _content_sha(replay_executable) != _content_sha(executable):
                raise ValueError("causal pair build replay differs from retained executable")
            replay_samples, replay_functional = _run_pair_arm(
                replay_executable, warmups=contract["warmup_iterations"],
                iterations=contract["measured_iterations"], timeout=contract["timeout_seconds"])
        value = _dispatch_call_count(executable)
        if row["runtime_dispatch_calls"] != value:
            raise ValueError("causal pair structural counter is not derived from compiled dataflow")
        retained_median = sorted(samples)[len(samples) // 2]
        replay_median = sorted(replay_samples)[len(replay_samples) // 2]
        # This schema-v1 micro-pair is now unit-test compatibility evidence only (production K1
        # attribution requires full-model schema v2).  Keep its replay magnitude gate consistent
        # with the legacy benchmark replay above; a 34% process-launch window flakes badly under a
        # parallel test load and has no production-claim value.
        if not (0.25 <= retained_median / replay_median <= 4.0):
            raise ValueError("causal pair timing does not reproduce")
        functional[arm] = replay_functional
        values[arm] = value
        medians[arm] = replay_median
        if executable_sha != row["executable"]["sha256"]:
            raise ValueError("causal pair executable binding differs")
    if (functional["control"] != functional["treatment"]
            or functional["control"] != receipt["functional_stdout_sha256"]):
        raise ValueError("causal pair control/treatment functional outputs differ")
    if values != {"control": 2, "treatment": 1}:
        raise ValueError("typed intervention did not produce exactly the declared dataflow delta")
    return {
        "control_artifact_sha256": arms["control"]["executable"]["sha256"],
        "treatment_artifact_sha256": arms["treatment"]["executable"]["sha256"],
        "control_artifact_content_sha256": _content_sha(
            root / arms["control"]["executable"]["path"]),
        "treatment_artifact_content_sha256": _content_sha(
            root / arms["treatment"]["executable"]["path"]),
        "control_runtime_dispatch_calls": values["control"],
        "treatment_runtime_dispatch_calls": values["treatment"],
        "control_replay_median_ns": medians["control"],
        "treatment_replay_median_ns": medians["treatment"],
        "functional_stdout_sha256": functional["control"],
        "treatment_improved": medians["treatment"] < medians["control"],
    }


def explain(inspection: Mapping[str, Any], *, generator_source_sha256: str) -> dict[str, Any]:
    """Reject the retired authored-count API; callers must use :func:`inspect`."""
    del inspection, generator_source_sha256
    raise ValueError("structural counters require a retained trace artifact and inspection contract")


def _run_command(argv: list[str], *, cwd: Path, environment: Mapping[str, str],
                 timeout_seconds: int, label: str) -> str:
    completed = subprocess.run(argv, cwd=cwd, env={**os.environ, **dict(environment)},
                               capture_output=True, text=True, timeout=timeout_seconds,
                               check=False)
    if completed.returncode != 0:
        raise ValueError(f"{label} failed rc={completed.returncode}: {completed.stderr[-500:]}")
    return completed.stdout


def _functional_receipt(stdout: str) -> str:
    receipt = _closed(_json_mapping(stdout, "benchmark functional stdout"), frozenset({
        "schema_version", "kind", "status", "output_sha256",
    }), "benchmark functional stdout")
    if (receipt["schema_version"] != 1
            or receipt["kind"] != "merlin_continuous_session_completion_v1"
            or receipt["status"] != "pass" or not _is_sha(receipt["output_sha256"])):
        raise ValueError("benchmark executable did not emit a valid functional completion receipt")
    return hashlib.sha256(stdout.encode("utf-8")).hexdigest()


def _verify_build_receipt(build: Mapping[str, Any], *, root: Path, executable: Path,
                          executable_sha: str, contract: Mapping[str, Any]) -> None:
    """Replay the exact retained build invocation and reproduce the measured executable bytes."""
    build = _closed(build, frozenset({
        "schema_version", "kind", "status", "backend", "package_sha256", "source_sha256",
        "executable_sha256", "package", "source", "invocation",
    }), "build receipt")
    if (build["schema_version"] != 2 or build["kind"] != "paper_executable_build_receipt_v2"
            or build["status"] != "pass" or build["backend"] != contract["backend"]
            or build["package_sha256"] != contract["package_sha256"]
            or build["source_sha256"] != contract["source_sha256"]
            or build["executable_sha256"] != executable_sha):
        raise ValueError("executable differs from exact build/package receipt")
    invocation = _closed(build["invocation"], frozenset({
        "tool", "argv", "cwd", "environment", "timeout_seconds",
    }), "build invocation")
    tool, _tool_sha = _read_bound(root, invocation["tool"], "build tool")
    package, package_sha = _read_bound(root, build["package"], "build package input")
    source, source_sha = _read_bound(root, build["source"], "build source input")
    if package_sha != contract["package_sha256"] or source_sha != contract["source_sha256"]:
        raise ValueError("build inputs differ from the frozen package/source roots")
    argv_template = invocation["argv"]
    if (not isinstance(argv_template, list) or not argv_template
            or argv_template[0] != "{tool}" or argv_template.count("{output}") != 1
            or argv_template.count("{package}") != 1 or argv_template.count("{source}") != 1
            or any(not isinstance(part, str) or not part for part in argv_template)
            or not isinstance(invocation["environment"], Mapping)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in invocation["environment"].items())
            or not isinstance(invocation["timeout_seconds"], int)
            or isinstance(invocation["timeout_seconds"], bool)
            or invocation["timeout_seconds"] <= 0):
        raise ValueError("build receipt does not contain an exact replayable invocation")
    cwd = (root / str(invocation["cwd"])).resolve()
    try:
        cwd.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("build invocation cwd escapes the receipt root") from error
    with tempfile.TemporaryDirectory(prefix="merlin-build-replay-") as temp:
        output = Path(temp) / executable.name
        replacements = {"{tool}": str(tool), "{package}": str(package),
                        "{source}": str(source), "{output}": str(output)}
        argv = [replacements.get(part, part) for part in argv_template]
        _run_command(argv, cwd=cwd, environment=invocation["environment"],
                     timeout_seconds=invocation["timeout_seconds"], label="build replay")
        if not output.is_file() or _path_sha(output) != executable_sha:
            raise ValueError("build replay did not reproduce the measured executable")


def benchmark(contract: Mapping[str, Any], *, root: Path,
              generator_source_sha256: str) -> dict[str, Any]:
    """Execute one production ablation contract and return its closed raw receipt."""
    fields = frozenset({
        "schema_version", "kind", "status", *_IDENTITY_FIELDS,
        "build_receipt", "executable", "execution", "board_probe",
    }) - {"command_sha256"}
    contract = _closed(contract, fields, "benchmark contract")
    if (contract["schema_version"] != 2
            or contract["kind"] != "paper_ablation_benchmark_contract_v2"
            or contract["status"] != "ready"):
        raise ValueError("benchmark contract must be a ready schema-v2 contract")
    executable, executable_sha = _read_bound(root, contract["executable"], "executable")
    build_path, build_sha = _read_bound(root, contract["build_receipt"], "build_receipt")
    build = yaml.safe_load(build_path.read_text(encoding="utf-8"))
    _verify_build_receipt(build, root=root, executable=executable,
                          executable_sha=executable_sha, contract=contract)
    execution = _closed(contract["execution"], frozenset({
        "argv", "cwd", "environment", "timeout_seconds", "warmup_iterations",
        "measured_iterations"}), "execution contract")
    board_probe = _closed(contract["board_probe"], frozenset({
        "authority", "source", "environment", "timeout_seconds"}),
        "board probe contract")
    if board_probe["authority"] != "merlin_trusted_k1_csr_sysfs_probe_v1":
        raise ValueError("board probe must use the source-pinned trusted K1 probe")
    probe_source, _probe_source_sha = _read_bound(
        root, board_probe["source"], "trusted K1 board probe source")
    if (_content_sha(probe_source) != TRUSTED_K1_PROBE_SOURCE_SHA256
            or probe_source.suffix != ".c"):
        raise ValueError("board probe source differs from the separately shipped trusted K1 probe")
    for label, value in (("execution", execution), ("board probe", board_probe)):
        if (label == "execution" and (not isinstance(value["argv"], list)
                or not value["argv"]
                or any(not isinstance(part, str) or not part for part in value["argv"]))
                or not isinstance(value["environment"], Mapping)
                or any(not isinstance(key, str) or not isinstance(item, str)
                       for key, item in value["environment"].items())
                or not isinstance(value["timeout_seconds"], int)
                or value["timeout_seconds"] <= 0):
            raise ValueError(f"{label} contract is invalid")
    if (not isinstance(execution["warmup_iterations"], int)
            or isinstance(execution["warmup_iterations"], bool)
            or execution["warmup_iterations"] < 0
            or not isinstance(execution["measured_iterations"], int)
            or isinstance(execution["measured_iterations"], bool)
            or execution["measured_iterations"] <= 0):
        raise ValueError("execution contract requires explicit warmup/measured iteration counts")
    argv = [str(executable) if part == "{executable}" else part for part in execution["argv"]]
    if argv[0] != str(executable):
        raise ValueError("execution argv must invoke the exact retained executable first")
    execution_cwd = (root / str(execution["cwd"])).resolve()
    for label, cwd in (("execution", execution_cwd),):
        try:
            cwd.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"{label} cwd escapes the contract root") from error
    probe_mode = "--unit-test-json" if contract["target"] == "unit-test" else "--json"
    if contract["target"] not in {"unit-test", "k1"}:
        raise ValueError("trusted K1 paper probe supports only target=k1")
    logical_probe_argv = ["merlin-trusted-k1-board-probe", probe_mode]
    with tempfile.TemporaryDirectory(prefix="merlin-k1-probe-") as probe_temp:
        probe_executable = Path(probe_temp) / "paper_k1_board_probe"
        target_flags = (["-march=rv64gcv", "-mabi=lp64d"]
                        if contract["target"] == "k1" else [])
        compile_argv = ["/usr/bin/cc", "-O2", "-std=c11", *target_flags,
                        str(probe_source), "-o", str(probe_executable)]
        _run_command(compile_argv, cwd=root, environment={},
                     timeout_seconds=board_probe["timeout_seconds"],
                     label="trusted K1 board probe build")
        if (not probe_executable.is_file()
                or not probe_executable.read_bytes().startswith(b"\x7fELF")):
            raise ValueError("trusted K1 board probe build did not produce an ELF executable")
        probe_argv = [str(probe_executable), probe_mode]
        before = _run_command(probe_argv, cwd=root,
                              environment=board_probe["environment"],
                              timeout_seconds=board_probe["timeout_seconds"],
                              label="board before probe")
        functional_digests: list[str] = []
        for _index in range(execution["warmup_iterations"]):
            stdout = _run_command(argv, cwd=execution_cwd, environment=execution["environment"],
                                  timeout_seconds=execution["timeout_seconds"],
                                  label="benchmark warmup executable")
            functional_digests.append(_functional_receipt(stdout))
        samples: list[int] = []
        for _index in range(execution["measured_iterations"]):
            started = time.monotonic_ns()
            stdout = _run_command(argv, cwd=execution_cwd, environment=execution["environment"],
                                  timeout_seconds=execution["timeout_seconds"],
                                  label="benchmark measured executable")
            elapsed = time.monotonic_ns() - started
            functional_digests.append(_functional_receipt(stdout))
            samples.append(elapsed)
        if len(set(functional_digests)) != 1:
            raise ValueError("benchmark functional output changed across the continuous session")
        after = _run_command(probe_argv, cwd=root,
                             environment=board_probe["environment"],
                             timeout_seconds=board_probe["timeout_seconds"],
                             label="board after probe")
    receipt = {
        "schema_version": 2, "kind": "frozen_ablation_raw_log", "status": "pass",
        **{field: contract[field] for field in _IDENTITY_FIELDS if field != "command_sha256"},
        "command_sha256": _canonical_sha(argv),
        "producer": {"id": MEASUREMENT_TOOL_ID, "source_sha256": generator_source_sha256,
                     "command": ["python3", "paper_ablation_generator.py", "benchmark",
                                 "{contract}", "{raw_log}"]},
        "build_receipt_sha256": build_sha, "executable_sha256": executable_sha,
        "benchmark_contract_sha256": _canonical_sha(contract),
        "execution_argv": argv, "samples_ns": samples,
        "functional_stdout_sha256": functional_digests[0],
        "measurement_clock": "time.monotonic_ns",
        "warmup_iterations": execution["warmup_iterations"],
        "measured_iterations": execution["measured_iterations"],
        "board_probe_argv": logical_probe_argv,
        "board_probe_command_sha256": _canonical_sha(logical_probe_argv),
        "board_receipts": {"before": before, "after": after},
    }
    observe(receipt, generator_source_sha256=generator_source_sha256)
    return receipt


def _load(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit("input must be a mapping")
    return document


def _source_sha() -> str:
    return _path_sha(Path(__file__))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3 or args[0] not in {
            "benchmark", "observe", "summarize", "inspect", "produce-pair"}:
        raise SystemExit(
            "usage: paper_ablation_generator.py "
            "{benchmark|observe|summarize|inspect|produce-pair} "
            "INPUT.yaml OUTPUT.json")
    action, source_value, output_value = args
    source, output = Path(source_value).resolve(), Path(output_value).resolve()
    if action == "produce-pair":
        print(produce_causal_pair(source, output))
        return 0
    document = _load(source)
    source_sha = _source_sha()
    if action == "benchmark":
        result = benchmark(document, root=source.parent, generator_source_sha256=source_sha)
    elif action == "observe":
        result = observe(document, generator_source_sha256=source_sha)
    elif action == "summarize":
        result = summarize(document, generator_source_sha256=source_sha)
    else:
        result = inspect(document, root=source.parent, generator_source_sha256=source_sha)
    output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
                      encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess replay tests
    raise SystemExit(main())
