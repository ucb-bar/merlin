"""Validate retained paper receipts, native tracking records and semantic replay."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .paper_measurement_build import _validate_build_receipt
from .paper_measurement_contract import _contract
from .paper_measurement_execution import _run_contract
from .paper_measurement_protocol import (
    _ISSUED_RECEIPTS,
    CONTROLLER_ID,
    _bound_file,
    _canonical_sha,
    _closed,
    _is_sha,
    _sha_file,
    _validate_issuance,
    entry_source,
    validate_controller_sources,
)


def _validate_board_receipts(contract: Mapping[str, Any], raw: Mapping[str, Any]) -> None:
    receipts = _closed(raw["board_receipts"], {"before", "after"}, "board receipts")
    conditions = raw["provenance"]["board_conditions"]
    for endpoint in ("before", "after"):
        entry = _closed(receipts[endpoint], {"probe", "cores"}, f"{endpoint} board receipt")
        try:
            probe = json.loads(entry["probe"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"{endpoint} board probe receipt is invalid") from error
        probe = _closed(
            probe,
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
            f"{endpoint} board probe receipt",
        )
        rows = entry["cores"]
        if not isinstance(rows, list) or len(rows) != len(contract["execution"]["core_ids"]):
            raise ValueError("board receipt does not cover every requested core")
        observed = []
        for row in rows:
            row = _closed(row, {"core_id", "governor", "current_khz", "max_khz"}, f"{endpoint} core frequency receipt")
            observed.append(row["core_id"])
            if row["governor"] != "performance" or row["current_khz"] != row["max_khz"] or row["current_khz"] <= 0:
                raise ValueError("board receipt contains an unlocked requested core")
        aggregate = conditions[endpoint]
        if (
            observed != contract["execution"]["core_ids"]
            or probe["vlen_bits"] != raw["provenance"]["vlen_bits"]
            or probe["vlen_source"] != raw["provenance"]["vlen_source"]
            or aggregate["current_khz"] != min(row["current_khz"] for row in rows)
            or aggregate["max_khz"] != min(row["max_khz"] for row in rows)
            or aggregate["max_thermal_millic"] != probe["max_thermal_millic"]
        ):
            raise ValueError("board receipt does not reconcile to result provenance")


def _validate_native_aet(
    receipt_path: Path, receipt: Mapping[str, Any], raw_sha: str, contract: Mapping[str, Any], expected_status: str
) -> tuple[str, str]:
    lifecycle = _closed(receipt["aet_lifecycle"], {"run_record", "events"}, "AET lifecycle")
    run_ref = _closed(lifecycle["run_record"], {"path", "sha256"}, "AET run record ref")
    event_ref = _closed(lifecycle["events"], {"path", "sha256"}, "AET events ref")
    run_path = _bound_file(receipt_path.parent, run_ref, "AET run record")
    event_path = _bound_file(receipt_path.parent, event_ref, "AET events")
    run = _closed(
        json.loads(run_path.read_text(encoding="utf-8")),
        {
            "schema_version",
            "run_id",
            "project",
            "suite",
            "target",
            "method",
            "seed",
            "tracking_mode",
            "created_at",
            "agentic",
            "token_accounting",
            "study_sha256",
            "cell_sha256",
            "raw_measurement_sha256",
            "benchmark_command_sha256",
        },
        "native AET run record",
    )
    try:
        created_at = datetime.fromisoformat(str(run["created_at"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("native AET created_at is not an ISO timestamp") from error
    if (
        created_at.tzinfo is None
        or run["schema_version"] != "1.1"
        or run["run_id"] != contract["run_id"]
        or run["project"] != "merlin"
        or run["suite"] != "paper-cell"
        or run["target"] != contract["target"]
        or run["method"] != "benchmark"
        or run["seed"] != 0
        or run["tracking_mode"] != "local"
        or run["agentic"] is not False
        or run["token_accounting"] != "not_applicable"
        or run["study_sha256"] != contract["study_sha256"]
        or run["cell_sha256"] != _canonical_sha(contract["cell"])
        or run["raw_measurement_sha256"] != raw_sha
        or run["benchmark_command_sha256"] != receipt["command_sha256"]
    ):
        raise ValueError("native AET run record does not bind controller measurement")
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(events) != 2:
        raise ValueError("native AET lifecycle must contain exactly benchmark+finish events")
    previous_ts, ids = None, set()
    for sequence, event in enumerate(events, 1):
        event = _closed(
            event,
            {
                "ts",
                "run_id",
                "project",
                "suite",
                "target",
                "method",
                "seed",
                "event",
                "event_id",
                "sequence",
                "payload",
            },
            f"native AET event {sequence}",
        )
        try:
            timestamp = datetime.fromisoformat(str(event["ts"]).replace("Z", "+00:00"))
            event_id = str(uuid.UUID(str(event["event_id"])))
        except ValueError as error:
            raise ValueError("native AET event timestamp/id is invalid") from error
        if (
            timestamp.tzinfo is None
            or (previous_ts is not None and timestamp < previous_ts)
            or event_id in ids
            or event["sequence"] != sequence
            or event["run_id"] != contract["run_id"]
            or event["project"] != "merlin"
            or event["suite"] != "paper-cell"
            or event["target"] != contract["target"]
            or event["method"] != "benchmark"
            or event["seed"] != 0
        ):
            raise ValueError("native AET event identity/order differs")
        previous_ts, ids = timestamp, ids | {event_id}
    benchmark = _closed(
        events[0]["payload"], {"status", "command_sha256", "raw_measurement_sha256"}, "AET benchmark payload"
    )
    finish = _closed(events[1]["payload"], {"status", "message"}, "AET finish payload")
    if (
        events[0]["event"] != "benchmark.completed"
        or benchmark["status"] != "complete"
        or benchmark["command_sha256"] != receipt["command_sha256"]
        or benchmark["raw_measurement_sha256"] != raw_sha
        or events[1]["event"] != "run.finished"
        or finish["status"] != ("ok" if expected_status == "pass" else "fail")
    ):
        raise ValueError("native AET lifecycle is not finalized for controller measurement")
    return str(run_ref["sha256"]), str(event_ref["sha256"])


def verify_receipt(
    receipt_path: str | Path,
    *,
    expected_result: Mapping[str, Any],
    expected_study_sha256: str,
    require_live_issuance: bool = False,
    trusted_issuance_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate/replay a receipt, requiring an external anchor in a fresh process.

    Local public keys and signatures prove possession only; a same-user editor can replace that
    entire self-contained root.  Therefore an issuance known to this live controller process is
    accepted directly, while a fresh process must receive the fingerprint from a separately
    frozen/notarized channel.
    """
    receipt_path = Path(receipt_path).resolve()
    issued = _ISSUED_RECEIPTS.get(receipt_path)
    document = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    if isinstance(document, Mapping) and document.get("schema_version") == 6:
        raise ValueError("legacy v6 measurement receipts require their original pinned controller source version")
    receipt = _closed(
        document,
        {
            "schema_version",
            "kind",
            "status",
            "controller_id",
            "study_sha256",
            "run_id",
            "cell",
            "command_sha256",
            "controller_source",
            "controller_sources",
            "contract",
            "raw_measurement",
            "issuance",
            "aet_lifecycle",
        },
        "controller measurement receipt",
    )
    cell = _closed(receipt["cell"], {"model", "backend", "precision", "core_count"}, "controller receipt cell")
    expected_cell = {key: expected_result.get(key) for key in ("model", "backend", "precision", "core_count")}
    if (
        receipt["schema_version"] != 7
        or receipt["kind"] != "paper_controller_measurement_receipt_v7"
        or receipt["status"] != "finalized"
        or receipt["controller_id"] != CONTROLLER_ID
        or receipt["study_sha256"] != expected_study_sha256
        or receipt["run_id"] != expected_result.get("run_id")
        or dict(cell) != expected_cell
        or not _is_sha(receipt["command_sha256"])
    ):
        raise ValueError("controller receipt identity differs from paper result")
    if require_live_issuance and (issued is None or issued[0] != _sha_file(receipt_path)):
        raise ValueError("controller receipt changed after production")
    source = _bound_file(receipt_path.parent, receipt["controller_source"], "controller source")
    if source.read_bytes() != entry_source().read_bytes():
        raise ValueError("receipt was not produced by this trusted controller version")
    validate_controller_sources(receipt_path.parent, receipt["controller_sources"])
    contract_path = _bound_file(receipt_path.parent, receipt["contract"], "measurement contract")
    contract, contract_cell = _contract(contract_path)
    if (
        contract["study_sha256"] != expected_study_sha256
        or contract["run_id"] != expected_result.get("run_id")
        or contract_cell != expected_cell
        or contract["artifact_sha256"] != expected_result.get("artifact_sha256")
    ):
        raise ValueError("controller contract differs from expected paper cell")
    expected_identity = {
        key: expected_result.get(key)
        for key in (
            "timestamp",
            "git_sha",
            "study_label",
            "target",
            "model",
            "checkpoint",
            "fidelity",
            "backend",
            "runtime",
            "precision",
            "quantization",
            "core_count",
        )
    }
    if contract["result_identity"] != expected_identity:
        raise ValueError("controller contract full result identity differs")
    raw_path = _bound_file(receipt_path.parent, receipt["raw_measurement"], "raw measurement")
    fingerprint = _validate_issuance(receipt_path, receipt, _sha_file(raw_path), _sha_file(contract_path))
    if trusted_issuance_fingerprint is not None and (
        not _is_sha(trusted_issuance_fingerprint) or trusted_issuance_fingerprint != fingerprint
    ):
        raise ValueError("controller issuance differs from externally notarized fingerprint")
    if issued is None and trusted_issuance_fingerprint is None:
        raise ValueError("fresh-process receipt authority requires an externally notarized issuance fingerprint")
    raw = _closed(
        json.loads(raw_path.read_text(encoding="utf-8")),
        {
            "schema_version",
            "kind",
            "status",
            "run_id",
            "study_sha256",
            "cell",
            "contract_sha256",
            "build_receipt",
            "driver",
            "artifact_sha256",
            "functional_output_sha256",
            "lifecycle",
            "session",
            "correctness",
            "quality",
            "memory",
            "timing",
            "execution",
            "provenance",
            "board_receipts",
            "process_capture",
        },
        "controller raw measurement",
    )
    if (
        raw["schema_version"] != 5
        or raw["kind"] != "paper_controller_raw_measurement_v5"
        or raw["status"] != "complete"
        or raw["contract_sha256"] != _sha_file(contract_path)
        or raw["run_id"] != expected_result.get("run_id")
        or raw["study_sha256"] != expected_study_sha256
        or raw["cell"] != expected_cell
        or raw["artifact_sha256"] != expected_result.get("artifact_sha256")
    ):
        raise ValueError("controller raw measurement identity differs")
    if issued is not None and issued != (_sha_file(receipt_path), _sha_file(raw_path), _sha_file(contract_path)):
        raise ValueError("controller-issued receipt/raw/contract changed after production")
    retained_build = _validate_build_receipt(contract_path.parent, contract, raw["build_receipt"])
    _validate_board_receipts(contract, raw)
    for field in ("session", "lifecycle", "correctness", "quality", "memory", "timing", "execution", "provenance"):
        if raw[field] != expected_result.get(field):
            raise ValueError(f"controller raw {field} differs from normalized result")
    capture = _closed(raw["process_capture"], {"stdout", "stderr"}, "process capture")
    stdout_path = _bound_file(receipt_path.parent, capture["stdout"], "captured stdout")
    stderr_path = _bound_file(receipt_path.parent, capture["stderr"], "captured stderr")
    driver = _closed(
        raw["driver"],
        {
            "id",
            "source_sha256",
            "argv",
            "command_sha256",
            "exit_code",
            "started_monotonic_ns",
            "ended_monotonic_ns",
            "stdout_sha256",
            "stderr_sha256",
            "iterations",
        },
        "controller raw driver",
    )
    if (
        driver["id"] != CONTROLLER_ID
        or driver["source_sha256"] != _sha_file(source)
        or driver["command_sha256"] != receipt["command_sha256"]
        or driver["command_sha256"] != _canonical_sha(driver["argv"])
        or driver["exit_code"] != 0
        or driver["ended_monotonic_ns"] <= driver["started_monotonic_ns"]
        or driver["stdout_sha256"] != _sha_file(stdout_path)
        or driver["stderr_sha256"] != _sha_file(stderr_path)
    ):
        raise ValueError("controller driver evidence is invalid")
    iterations = driver["iterations"]
    sealed_session = contract["registry_id"] == "executorch_v1" and contract["target"] != "unit-test"
    expected_processes = 1 if sealed_session else contract["warmup_iterations"] + contract["measured_iterations"]
    if not isinstance(iterations, list) or len(iterations) != expected_processes:
        raise ValueError("controller iteration trace is incomplete")
    measured = []
    for ordinal, item in enumerate(iterations):
        item = _closed(
            item,
            {
                "phase",
                "index",
                "started_monotonic_ns",
                "ended_monotonic_ns",
                "elapsed_ns",
                "exit_code",
                "stdout_sha256",
                "stderr_sha256",
                "observation_sha256",
                "peak_rss_bytes",
                "affinity_core_ids",
                "max_task_count",
                "simultaneous_running_core_ids",
                "cpu_time_ticks_by_core",
            },
            f"controller iteration {ordinal}",
        )
        if sealed_session:
            # One continuously resident process owns all warmups, measured repeats, and recurrent
            # state.  Per-repeat timing is separately emitted by the sealed runner; inventing one
            # OS-process trace row per internal repeat would be false evidence.
            expected_phase, expected_index = "sealed_session", 0
        else:
            expected_phase = "warmup" if ordinal < contract["warmup_iterations"] else "measured"
            expected_index = ordinal if expected_phase == "warmup" else ordinal - contract["warmup_iterations"]
        if (
            item["phase"] != expected_phase
            or item["index"] != expected_index
            or item["exit_code"] != 0
            or item["ended_monotonic_ns"] <= item["started_monotonic_ns"]
            or item["elapsed_ns"] != item["ended_monotonic_ns"] - item["started_monotonic_ns"]
            or item["stdout_sha256"] != driver["stdout_sha256"]
            or item["stderr_sha256"] != driver["stderr_sha256"]
            or item["observation_sha256"] != raw["functional_output_sha256"]
            or item["peak_rss_bytes"] <= 0
            or item["affinity_core_ids"] != contract["execution"]["core_ids"]
            or item["simultaneous_running_core_ids"] != contract["execution"]["core_ids"]
            or set(item["cpu_time_ticks_by_core"]) != {str(core) for core in contract["execution"]["core_ids"]}
            or any(type(ticks) is not int or ticks <= 0 for ticks in item["cpu_time_ticks_by_core"].values())
        ):
            raise ValueError("controller iteration trace does not reconcile")
        if expected_phase == "measured":
            measured.append(item["elapsed_ns"])
    timing = _closed(
        raw["timing"],
        {
            "unit",
            "sample_unit",
            "scope",
            "timed_stages",
            "excluded_stages",
            "stage_samples",
            "samples",
            "median",
            "p95",
        },
        "controller timing",
    )
    if sealed_session:
        measured = timing["samples"]
        stage_samples = timing["stage_samples"]
        opaque = contract["frozen_provenance"].get("stage_attribution") == "opaque_whole_forward"
        measured_valid = (
            isinstance(measured, list)
            and len(measured) == contract["measured_iterations"]
            and all(type(sample) is int and sample > 0 for sample in measured)
        )
        stage_evidence_invalid = stage_samples != {}
        if measured_valid and not opaque:
            stage_evidence_invalid = (
                not isinstance(stage_samples, Mapping)
                or set(stage_samples) != set(timing["timed_stages"])
                or any(
                    not isinstance(samples, list)
                    or len(samples) != contract["measured_iterations"]
                    or any(type(sample) is not int or sample <= 0 for sample in samples)
                    for samples in stage_samples.values()
                )
            )
            if not stage_evidence_invalid:
                stage_evidence_invalid = any(
                    sum(stage_samples[stage][index] for stage in timing["timed_stages"]) != measured[index]
                    for index in range(len(measured))
                )
        if not measured_valid or stage_evidence_invalid:
            raise ValueError("sealed ExecuTorch per-repeat timing evidence is incomplete")
    ordered, count = sorted(measured), len(measured)
    median = ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) // 2
    p95 = ordered[min(count - 1, max(0, int(round(0.95 * (count - 1)))))]
    if (
        measured != timing["samples"]
        or timing["median"] != median
        or timing["p95"] != p95
        or sum(measured) > driver["ended_monotonic_ns"] - driver["started_monotonic_ns"]
    ):
        raise ValueError("controller timing summary does not reconcile")
    run_sha, event_sha = _validate_native_aet(
        receipt_path, receipt, _sha_file(raw_path), contract, str(raw["lifecycle"]["status"])
    )
    replay = _run_contract(contract_path)
    replay.pop("_captured_stdout")
    replay.pop("_captured_stderr")
    for field in (
        "cell",
        "artifact_sha256",
        "functional_output_sha256",
        "session",
        "lifecycle",
        "correctness",
        "quality",
        "execution",
        "provenance",
    ):
        if replay[field] != raw[field]:
            raise ValueError(f"controller replay {field} differs from retained measurement")
    replay_build = _validate_build_receipt(contract_path.parent, contract, replay["build_receipt"])
    for field in (
        "registry_id",
        "exit_code",
        "executable_sha256",
        "tool_sha256",
        "source_sha256",
        "input_sha256",
        "object_regeneration",
    ):
        if field not in retained_build and field not in replay_build:
            continue
        if replay_build[field] != retained_build[field]:
            raise ValueError("controller replay build receipt differs from retained build")
    if raw["memory"]["policy"] != replay["memory"]["policy"]:
        raise ValueError("controller replay memory policy differs")
    # Timing/RSS naturally vary.  Their primary bytes are authenticated by the detached issuance
    # root above.  Replay establishes reproducibility of build, semantics, output, and environment;
    # it must never make rewritten primary timing samples authentic through a tolerance window.
    return {
        "cell": "/".join((str(cell["model"]), str(cell["backend"]), str(cell["precision"]), f"{cell['core_count']}c")),
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha_file(receipt_path),
        "command_sha256": receipt["command_sha256"],
        "raw_measurement_sha256": _sha_file(raw_path),
        "aet_run_record_sha256": run_sha,
        "aet_events_sha256": event_sha,
        "build_receipt_sha256": _canonical_sha(raw["build_receipt"]),
        "issuance_fingerprint": fingerprint,
        "reproducibility": {"status": "semantic_replay_passed", "primary_timing_used": False},
    }
