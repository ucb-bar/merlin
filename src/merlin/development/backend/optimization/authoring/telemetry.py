"""Strict Codex telemetry for Phase 2, independent of native authoring controllers.

Prepare the implementation/price declaration, collect lossless round evidence, and
finalize into AET-owned trajectories and ledgers. These operations never launch an agent.
"""

from __future__ import annotations

import inspect
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from merlin.common import access as shared_access_policy
from merlin.common import digest
from merlin.common import digest as source_membership
from merlin.common.digest import hash_tree
from merlin.common.digest import sha256_bytes as _sha256
from merlin.common.paths import module_source_path
from merlin.development.backend.optimization import contracts
from merlin.development.backend.optimization.author_interface import prompt_inputs as PP
from merlin.development.backend.optimization.campaign import functional_boundary as PC
from merlin.development.backend.optimization.claims import affine_reduction as PK
from merlin.development.backend.optimization.contracts import (
    ROUND_DEADLINE_EXIT,
    StageGateError,
)
from merlin.development.backend.optimization.contracts import (
    canonical_json as _canonical_json,
)
from merlin.development.backend.optimization.contracts import (
    require_executable as _require_executable,
)
from merlin.development.backend.optimization.contracts import (
    sha256_file as _sha256_file,
)
from merlin.development.backend.optimization.contracts import (
    write_json as _write_json,
)
from merlin.development.backend.optimization.measurement import gsim_gate as GATE
from merlin.development.targetgen.sandbox import grants as BW
from merlin.development.targetgen.sandbox import toolchain as TC
from merlin.development.targetgen.sandbox.answer_surfaces import answer_surfaces
from merlin.experiments import definition as historical_target_access_policy
from merlin.target.registry.descriptor_loading import load_target_experiment

# ---- telemetry ---------------------------------------------------------------------------------


def prepare(
    *, model: str, authoring_stage: Path, price_table: Path | None = None, codex_binary: str | Path | None = None
) -> dict[str, Any]:
    """Prove the exact raw-Codex -> token ledger -> AET trajectory path before a paid turn.

    This is deliberately strict for the performance experiment even though the shared AET bridge is
    soft for ordinary developer runs.  It performs an in-memory parser canary and pins every parser
    implementation plus the notional price input; it neither creates a run directory nor launches an
    agent.
    """
    from merlin.development.backend.optimization.authoring.telemetry_accounting import (
        ACCOUNTING_POLICY,
        SOURCE_POLICY_VERSION,
        _declared_price_rate,
        _import_codex_driver,
        _package_source_record,
        _price_snapshot_document,
        _snapshot,
        _telemetry_source_record,
    )

    try:
        from aet.tracking.run_logger import EvalRunLogger  # noqa: PLC0415
        from aet.trajectory import billing, cost, price_snapshot, pricing
        from aet.trajectory.classify import ActivityClassifier, ActivityConfig  # noqa: PLC0415
        from aet.trajectory.codex import CodexNormalizer  # noqa: PLC0415
        from aet.trajectory.importers.codex import build_trajectory_from_run  # noqa: PLC0415
        from aet.trajectory.reconcile import reconcile_codex  # noqa: PLC0415

        from merlin.development.backend import transcript_tokens as ET  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise StageGateError(f"required AET/Codex telemetry stack is unavailable: {exc}") from exc
    CA = _import_codex_driver()
    try:
        from merlin.development.backend.providers import agent_bridge as model_bridge  # noqa: PLC0415

        resolved_model = str(CA.resolve_model(model) or "").strip()
    except Exception as exc:  # noqa: BLE001
        raise StageGateError(f"Codex model resolution preflight failed: {exc}") from exc
    if not resolved_model:
        raise StageGateError("Codex model resolution produced an empty model identity")
    codex_path = _require_executable(str(codex_binary or os.environ.get("CODEX_BIN") or "codex"), label="Codex")
    if price_table is None:
        try:
            from merlin.common.paths import _dotenv  # noqa: PLC0415

            raw_price = os.environ.get("AET_PRICE_TABLE") or _dotenv().get("AET_PRICE_TABLE") or ""
        except Exception:  # noqa: BLE001
            raw_price = os.environ.get("AET_PRICE_TABLE", "")
        price_table = Path(raw_price) if raw_price else None
    if price_table is None:
        raise StageGateError("AET_PRICE_TABLE must explicitly pin subscription notional pricing")
    price_table = Path(price_table)
    if price_table.is_symlink() or not price_table.is_file():
        raise StageGateError(f"telemetry price table is absent, linked, or non-regular: {price_table}")
    price_table = price_table.resolve()
    rate = _declared_price_rate(price_table, resolved_model)
    if rate is None:
        raise StageGateError(f"telemetry price table has no exact substring rate for {resolved_model!r}")
    snapshot_document = _price_snapshot_document(price_table)
    snapshot = _snapshot(snapshot_document)

    # A real schema canary, in memory: unknown event kinds remain reconcilable, token subset
    # arithmetic is exercised, and no filesystem side effect is needed for preflight.
    canary_events = [
        {"type": "thread.started", "thread_id": "telemetry-canary"},
        {"type": "turn.started"},
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 17,
                "cached_input_tokens": 5,
                "cache_write_input_tokens": 2,
                "output_tokens": 7,
                "reasoning_output_tokens": 3,
            },
        },
    ]
    normalizer = CodexNormalizer()
    normalizer.feed_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in canary_events))
    run = normalizer.result()
    cfg = ActivityConfig()
    trajectory = build_trajectory_from_run(
        run,
        run_id="telemetry-canary",
        classifier=ActivityClassifier(cfg),
        classifier_cfg=cfg.to_dict(),
        model=resolved_model,
        snapshot=snapshot,
        billing_row={"provider": "openai", "billing_mode": "subscription"},
        calculated_at="",
    )
    reconciliation = reconcile_codex(run, trajectory)
    parsed = ET._codex_usage(canary_events)  # same reader used by parse_agent_transcript
    if (
        not reconciliation.get("ok")
        or not parsed
        or parsed.get("usage_complete") is not True
        or parsed.get("tokens_input") != 10
        or parsed.get("tokens_cached") != 5
        or parsed.get("tokens_cache_write") != 2
        or parsed.get("tokens_output") != 7
    ):
        raise StageGateError("raw Codex/AET telemetry parser canary failed")
    return {
        "schema_version": 5,
        "source_policy_version": SOURCE_POLICY_VERSION,
        "accounting_policy": ACCOUNTING_POLICY,
        "required": True,
        "driver": "codex",
        "raw_capture": "durable_jsonl_before_interpretation_plus_timestamp_sidecar",
        "accounting": "raw_codex_turn_usage_nonoverlapping_token_buckets",
        "activity": "aet_native_codex_structured_tool_spans",
        "aet_reconciliation_required": True,
        "billing_mode": "subscription_notional",
        "model_resolution": {
            "requested_model": model,
            "resolved_model": resolved_model,
            # resolve_model consults this before every native/bridged/default route.  Preserve the
            # exact ambient input so a resume cannot silently redirect an otherwise identical slug.
            "codex_model_map": os.environ.get("CODEX_MODEL_MAP", ""),
        },
        "price_table": {
            "path": str(price_table),
            "sha256": _sha256_file(price_table),
            "model": resolved_model,
            "requested_model": model,
            "rate_per_million": list(rate),
            "snapshot": {"document": snapshot_document, "sha256": snapshot.sha256()},
        },
        "sources": {
            "performance_package_sources": _package_source_record(),
            "python_source_membership": _telemetry_source_record(source_membership, label="Python source membership"),
            "aet_price_snapshot": _telemetry_source_record(price_snapshot, label="AET price snapshot"),
            "aet_pricing": _telemetry_source_record(pricing, label="AET pricing"),
            "aet_cost": _telemetry_source_record(cost, label="AET cost record"),
            "aet_billing": _telemetry_source_record(billing, label="AET billing classification"),
            "performance_telemetry": _telemetry_source_record(prepare, label="performance telemetry"),
            "performance_contracts": _telemetry_source_record(contracts, label="performance evidence contracts"),
            "shared_digest": _telemetry_source_record(digest, label="shared digest primitives"),
            "codex_binary": {"path": str(codex_path), "sha256": _sha256_file(codex_path)},
            "performance_authoring_stage": _telemetry_source_record(
                authoring_stage, label="performance authoring stage"
            ),
            "performance_campaign": _telemetry_source_record(PC, label="performance campaign helper"),
            "performance_gsim_gate": _telemetry_source_record(GATE, label="performance GSIM gate"),
            "performance_pk_claim": _telemetry_source_record(PK, label="performance PK claim"),
            "performance_prompt": _telemetry_source_record(PP, label="performance prompt contract"),
            "codex_driver": {
                **_telemetry_source_record(CA, label="Codex driver"),
                "support": {
                    "codex_runtime": _telemetry_source_record(
                        module_source_path("merlin.development.backend.providers.codex_runtime"), label="Codex runtime"
                    )
                },
            },
            "codex_model_bridge": _telemetry_source_record(model_bridge, label="Codex model bridge"),
            "benchharness": _telemetry_source_record(hash_tree, label="candidate tree hashing"),
            "sandbox_bwrap": _telemetry_source_record(BW, label="agent bwrap policy"),
            "sandbox_toolchain": _telemetry_source_record(TC, label="agent toolchain policy"),
            "sandbox_answer_surfaces": _telemetry_source_record(answer_surfaces, label="answer-surface policy"),
            "shared_access_policy": _telemetry_source_record(
                shared_access_policy, label="shared access and audit policy"
            ),
            "historical_target_access_policy": _telemetry_source_record(
                historical_target_access_policy, label="historical target access policy"
            ),
            "historical_target_access_data": {
                "path": str(historical_target_access_policy.resource_path()),
                "sha256": _sha256_file(historical_target_access_policy.resource_path()),
            },
            "target_experiment_loader": _telemetry_source_record(
                load_target_experiment, label="target experiment loader"
            ),
            "experiment_tokens": _telemetry_source_record(ET, label="token accounting"),
            "aet_codex_normalizer": _telemetry_source_record(CodexNormalizer, label="AET Codex normalizer"),
            "aet_codex_importer": _telemetry_source_record(build_trajectory_from_run, label="AET Codex importer"),
            "aet_reconciliation": _telemetry_source_record(reconcile_codex, label="AET reconciliation"),
            "aet_activity_classifier": _telemetry_source_record(ActivityClassifier, label="AET activity classifier"),
            "aet_canonical_logger": _telemetry_source_record(EvalRunLogger, label="AET canonical logger"),
        },
    }


def collect_round(
    stage_root: Path, round_index: int, *, model: str, agent_exit_code: int, preflight_record: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and hash the driver's raw, timestamped and summary artifacts for one round.

    A DEADLINE EXIT IS A BUDGET, NOT A BROKEN INSTRUMENT. Two of the checks below can never be met by
    a round the harness itself killed at its deadline: ``round_NN.final.txt`` is written by the driver
    only when it emits a final assistant message, and ``summary.timed_out`` is then true. Treating
    those as malformed telemetry cost this campaign 2 of its 3 paid rounds -- each refused here, which
    made the round non-authored, which discarded its compiler edits and restarted the next round from
    the initial seed. Phase 1 has always admitted the same exit as ``stopped_by`` on exactly the
    evidence a clean round already produced.

    So a deadline exit yields COMPLETE telemetry with ``deadline_reached: True`` and the absent final
    message NAMED, and every other integrity check is unchanged and still fatal. The distinction being
    drawn is between "the record is a lie" -- a rewritten sidecar, a discontinuous sequence, counts
    that disagree -- and "the record is honest and the round ran out of time". The first must refuse;
    the second is a result.
    """
    from datetime import datetime  # noqa: PLC0415

    from merlin.development.backend.optimization.authoring.telemetry_accounting import (
        ACCOUNTING_POLICY,
        _accounting,
        _verified_snapshot,
    )

    snapshot = _verified_snapshot(preflight_record, model)
    rounds = stage_root / "rounds"
    paths = {
        "raw": rounds / f"round_{round_index:02d}.codex_events.raw.jsonl",
        "timestamped": rounds / f"round_{round_index:02d}.codex_events.timestamped.jsonl",
        "summary": rounds / f"round_{round_index:02d}.codex_summary.json",
        "stderr": rounds / f"round_{round_index:02d}.codex_stderr.log",
        "prompt": rounds / f"round_{round_index:02d}.prompt.txt",
        "final": rounds / f"round_{round_index:02d}.final.txt",
    }
    # The driver writes the final message itself, at the end of its turn; a round killed at the
    # deadline has no end of turn. Absent-and-explained, never absent-and-assumed.
    deadline = int(agent_exit_code) == ROUND_DEADLINE_EXIT
    absent: dict[str, str] = {}
    for label, path in list(paths.items()):
        if path.is_symlink() or not path.is_file():
            if label == "final" and deadline:
                absent[label] = (
                    f"{path.name} is absent because the round reached its deadline "
                    f"before the driver emitted a final message"
                )
                del paths[label]
                continue
            raise StageGateError(f"Codex round {round_index} lacks real {label} telemetry: {path}")
    try:
        raw_lines = paths["raw"].read_text(encoding="utf-8").splitlines()
        stamped = [
            json.loads(line) for line in paths["timestamped"].read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    except (UnicodeError, ValueError) as exc:
        raise StageGateError(f"Codex round {round_index} telemetry is malformed: {exc}") from exc
    if not raw_lines or len(stamped) != len(raw_lines):
        raise StageGateError(f"Codex round {round_index} raw/timestamped event counts differ or are zero")
    for sequence, (raw_line, wrapper) in enumerate(zip(raw_lines, stamped, strict=True), start=1):
        if not isinstance(wrapper, Mapping) or wrapper.get("seq") != sequence:
            raise StageGateError(f"Codex round {round_index} timestamp sequence is discontinuous")
        try:
            datetime.fromisoformat(str(wrapper["arrived_at"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise StageGateError(f"Codex round {round_index} has an invalid arrival timestamp") from exc
        try:
            event = json.loads(raw_line)
        except ValueError:
            if wrapper.get("unparsed") != raw_line or "event" in wrapper:
                raise StageGateError(f"Codex round {round_index} sidecar changed raw line {sequence}")
        else:
            if wrapper.get("event") != event or "unparsed" in wrapper:
                raise StageGateError(f"Codex round {round_index} sidecar changed raw event {sequence}")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    # `timed_out` is REQUIRED to agree with the exit code in both directions, rather than merely
    # relaxed: a round that reports a timeout without the deadline exit, or the deadline exit without
    # a reported timeout, has a summary that disagrees with what happened, and that is the malformed
    # case this gate is for.
    if (
        not isinstance(summary, Mapping)
        or summary.get("billing_mode") != "subscription_notional"
        or summary.get("exit_code") != agent_exit_code
        or summary.get("usage_complete") is not True
        or summary.get("timed_out") is not deadline
        or not isinstance(summary.get("wall_s"), (int, float))
        or float(summary["wall_s"]) <= 0
    ):
        raise StageGateError(f"Codex round {round_index} usage/timing summary is incomplete")
    accounting = _accounting(paths["raw"], model=model, snapshot=snapshot)
    if (
        accounting.get("available") is not True
        or accounting.get("usage_complete") is not True
        or not isinstance(accounting.get("tokens_total"), int)
        or accounting["tokens_total"] <= 0
    ):
        raise StageGateError(f"Codex round {round_index} token accounting is incomplete")
    for path in paths.values():
        path.chmod(0o444)
    return {
        "accounting_policy": ACCOUNTING_POLICY,
        "price_snapshot_sha256": snapshot.sha256(),
        # Stated positively so a consumer never has to read the ABSENCE of a refusal as completeness;
        # the caller's failure path writes `complete: False` with a reason, and these two are the only
        # two shapes.
        "complete": True,
        "event_count": len(raw_lines),
        "summary": dict(summary),
        "accounting": accounting,
        "deadline_reached": deadline,
        # Named, never omitted: an absent key reads as "this artifact does not apply", and only an
        # explicit entry says "it is missing and here is why that is admissible".
        "absent_artifacts": dict(absent),
        "artifacts": {
            label: {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for label, path in paths.items()
        },
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def finalize(
    stage_root: Path,
    round_records: Sequence[Mapping[str, Any]],
    *,
    model: str,
    target: str,
    suite: str,
    run_id: str,
    preflight_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce fail-closed AET, cost, activity, token-ledger and tool-ledger artifacts."""
    from merlin.development.backend import transcript_tokens as ET  # noqa: PLC0415
    from merlin.development.backend.optimization.authoring.telemetry_accounting import (
        ACCOUNTING_POLICY,
        _accounting,
        _verified_snapshot,
        _verify_raw_accounting,
    )

    snapshot = _verified_snapshot(preflight_record, model)
    from datetime import datetime  # noqa: PLC0415

    try:
        from aet.trajectory.classify import ActivityClassifier, ActivityConfig  # noqa: PLC0415
        from aet.trajectory.codex import CodexNormalizer  # noqa: PLC0415
        from aet.trajectory.importers.codex import build_trajectory_from_run  # noqa: PLC0415
        from aet.trajectory.reconcile import (  # noqa: PLC0415
            reconcile_codex,
            token_ledger_rows,
            tool_ledger_rows,
        )
    except Exception as exc:  # noqa: BLE001
        raise StageGateError(f"required AET telemetry finalizer is unavailable: {exc}") from exc

    root = stage_root / "telemetry"
    raw_dir, ts_dir = root / "raw", root / "timestamped"
    raw_dir.mkdir(parents=True, exist_ok=False)
    ts_dir.mkdir(parents=True, exist_ok=False)
    snapshot_path = root / "price_snapshot.json"
    snapshot_path.write_bytes(snapshot.raw_bytes)
    combined_raw = root / "codex_events.raw.jsonl"
    total_wall = 0.0
    normalizer = CodexNormalizer()
    first_arrival = previous_arrival = None
    with combined_raw.open("wb") as combined:
        for row in round_records:
            index = int(row["round"])
            evidence = row.get("telemetry") or {}
            if (
                evidence.get("accounting_policy") != ACCOUNTING_POLICY
                or evidence.get("price_snapshot_sha256") != snapshot.sha256()
            ):
                raise StageGateError(f"Codex round {index} differs from the pinned price authority")
            artifacts = evidence.get("artifacts") or {}
            raw = Path(str((artifacts.get("raw") or {}).get("path") or ""))
            stamped_path = Path(str((artifacts.get("timestamped") or {}).get("path") or ""))
            if _sha256_file(raw) != (artifacts.get("raw") or {}).get("sha256") or _sha256_file(stamped_path) != (
                artifacts.get("timestamped") or {}
            ).get("sha256"):
                raise StageGateError(f"Codex round {index} telemetry changed before AET import")
            _verify_raw_accounting(evidence.get("accounting") or {}, raw, model=model, snapshot=snapshot)
            raw_payload = raw.read_bytes()
            combined.write(raw_payload)
            if raw_payload and not raw_payload.endswith(b"\n"):
                combined.write(b"\n")
            destination = raw_dir / f"events.{index:02d}.jsonl"
            destination.write_bytes(raw_payload)
            raw_lines = raw.read_text(encoding="utf-8").splitlines()
            wrappers = [
                json.loads(line) for line in stamped_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            for line, wrapper in zip(raw_lines, wrappers, strict=True):
                try:
                    arrival = datetime.fromisoformat(wrapper["arrived_at"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise StageGateError(f"Codex round {index} lacks an observed arrival timestamp") from exc
                if arrival.utcoffset() is None or (previous_arrival is not None and arrival < previous_arrival):
                    raise StageGateError("Codex arrival timestamps must be timezone-aware and nondecreasing")
                if first_arrival is None:
                    first_arrival = arrival
                normalizer.feed_line(line, t_s=(arrival - first_arrival).total_seconds())
                previous_arrival = arrival
            _write_jsonl(
                ts_dir / destination.name,
                [
                    {"ts": wrapper["arrived_at"], "line": line}
                    for line, wrapper in zip(raw_lines, wrappers, strict=True)
                ],
            )
            total_wall += float((evidence.get("summary") or {}).get("wall_s", 0.0))

    normalized = normalizer.result()
    config = ActivityConfig()
    trajectory = build_trajectory_from_run(
        normalized,
        run_id=run_id,
        classifier=ActivityClassifier(config),
        classifier_cfg=config.to_dict(),
        model=model,
        snapshot=snapshot,
        billing_row={"provider": "openai", "billing_mode": "subscription"},
        calculated_at="",
    )
    reconciliation = reconcile_codex(normalized, trajectory, admin_usd=None)
    expected_raw_events = sum(int((row.get("telemetry") or {}).get("event_count", 0)) for row in round_records)
    if (
        reconciliation.get("ok") is not True
        or reconciliation["raw_events"].get("raw_event_count") != expected_raw_events
        or normalized.raw_event_count != expected_raw_events
        or reconciliation["token_ledger"].get("num_turns") < len(round_records)
    ):
        raise StageGateError("AET failed to reconcile the complete raw Codex event stream")
    tool_rows = tool_ledger_rows(normalized)
    if not tool_rows:
        raise StageGateError("AET found zero structured tool calls in the performance agent run")
    trajectory_path = stage_root / "metrics" / "trajectory.json"
    trajectory.to_json(trajectory_path)
    reconciliation_path = root / "aet_reconciliation.json"
    _write_json(reconciliation_path, reconciliation)
    token_ledger = stage_root / "metrics" / "token_ledger.jsonl"
    tool_ledger = stage_root / "agent" / "tools.jsonl"
    _write_jsonl(token_ledger, token_ledger_rows(normalized))
    _write_jsonl(tool_ledger, tool_rows)

    accounting = _accounting(combined_raw, model=model, snapshot=snapshot)
    if trajectory.cost != accounting["cost_provenance"]:
        raise StageGateError("AET trajectory and token ledger price authority differ")
    if accounting.get("available") is not True or accounting.get("usage_complete") is not True:
        raise StageGateError("combined raw Codex token/cost accounting is incomplete")
    if (
        not isinstance(accounting.get("subscription_notional_usd"), (int, float))
        or accounting["subscription_notional_usd"] <= 0
    ):
        raise StageGateError("combined raw Codex usage lacks pinned subscription-notional cost")
    accounting["tool_calls"] = len(tool_rows)
    accounting["subagent_tool_calls_tracked"] = False
    cost_path = stage_root / "cost_time_toolcalls.yaml"
    ET.write_cost_yaml(accounting, cost_path, wall_time_seconds=round(total_wall, 3), model=model, exit_code=0)

    # Native AET run-store rows make the trial discoverable by `aet spend`.  A ChatGPT seat has
    # real spend 0 here; the separately named notional metric cannot be summed into a billed budget.
    try:
        from aet.tracking.run_logger import EvalRunLogger  # noqa: PLC0415

        logger = EvalRunLogger.start(
            project="merlin",
            suite=suite,
            target=target,
            method="agentic_perf_trial",
            seed=0,
            run_id=run_id,
            run_path=stage_root,
            tracking_mode="local",
        )
        logger.log_token_usage(
            input_tokens=int(accounting.get("tokens_input", 0)),
            output_tokens=int(accounting.get("tokens_output", 0)),
            cache_creation_tokens=int(accounting.get("tokens_cache_write", 0)),
            cache_read_tokens=int(accounting.get("tokens_cached", 0)),
            model=model,
        )
        logger.log_cost(0.0, model=model)
        logger.log_param("billing_mode", "subscription_notional")
        if accounting.get("subscription_notional_usd") is not None:
            logger.log_metric("cost.subscription_notional_usd", float(accounting["subscription_notional_usd"]))
        logger.log_agent_turns(len(normalized.turns))
        logger.close()
    except Exception as exc:  # noqa: BLE001
        raise StageGateError(f"AET canonical run logger failed: {exc}") from exc
    metrics_log = stage_root / "logs" / "metrics.jsonl"
    if metrics_log.is_symlink() or not metrics_log.is_file() or metrics_log.stat().st_size <= 0:
        raise StageGateError("AET canonical metrics log was not materialized")

    durations: dict[str, float] = {}
    for band in trajectory.bands:
        durations[band.category] = durations.get(band.category, 0.0) + band.duration_s
    classified = sum(durations.values())
    if classified <= 0 or trajectory.duration_s <= 0:
        raise StageGateError("AET produced no duration-bearing classified activity spans")
    activity = {
        "schema_version": 2,
        "basis": "aet_native_codex_structured_tool_spans",
        "denominator": "sum_of_classified_tool_span_seconds_including_overlap",
        "is_wall_time_partition": False,
        "overlapping_tool_spans_allowed": True,
        "classified_seconds": round(classified, 6),
        "trajectory_wall_seconds": trajectory.duration_s,
        "agent_round_wall_seconds": round(total_wall, 3),
        "classified_span_occupancy_ratio": (
            round(classified / trajectory.duration_s, 6) if trajectory.duration_s > 0 else None
        ),
        "occupancy_ratio_may_exceed_one": True,
        "subagent_tool_calls_tracked": False,
        "seconds_by_category": {key: round(value, 6) for key, value in sorted(durations.items())},
        "share_by_category": (
            {key: round(value / classified, 8) for key, value in sorted(durations.items())} if classified else {}
        ),
        "note": (
            "shares partition classified tool-span seconds, not wall time; simultaneous tool "
            "spans each contribute their full duration, so classified_span_occupancy_ratio may "
            "exceed 1; unclassified wall is not relabeled as thinking"
        ),
    }
    activity_path = root / "activity_share.json"
    _write_json(activity_path, activity)
    preflight_path = root / "preflight.json"
    _write_json(preflight_path, dict(preflight_record))
    artifact_paths = {
        "combined_raw": combined_raw,
        "trajectory": trajectory_path,
        "reconciliation": reconciliation_path,
        "token_ledger": token_ledger,
        "tool_ledger": tool_ledger,
        "cost_time_toolcalls": cost_path,
        "activity_share": activity_path,
        "preflight": preflight_path,
        "aet_metrics_log": metrics_log,
        "price_snapshot": snapshot_path,
    }
    for path in (*raw_dir.iterdir(), *ts_dir.iterdir(), *artifact_paths.values()):
        path.chmod(0o444)
    return {
        "policy_version": 3,
        "accounting_policy": ACCOUNTING_POLICY,
        "price_snapshot_sha256": snapshot.sha256(),
        "required": True,
        "driver": "codex",
        "raw_event_count": normalized.raw_event_count,
        "tool_call_count": len(tool_rows),
        "rounds_with_complete_usage": len(round_records),
        "subagent_tool_calls_tracked": False,
        "billing_mode": "subscription_notional",
        "accounting": accounting,
        "activity_share": activity,
        "aet_reconciliation": reconciliation,
        "preflight_sha256": _sha256(_canonical_json(preflight_record)),
        "artifacts": {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in artifact_paths.items()},
    }
