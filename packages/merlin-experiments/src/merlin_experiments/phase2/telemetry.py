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

from merlin.benchharness import hash_tree
from merlin.common import access as shared_access_policy
from merlin.common import digest, source_membership
from merlin.common.digest import sha256_bytes as _sha256
from merlin.common.paths import module_source_path
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces
from merlin.targetgen.target_experiment import load_target_experiment
from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2 import contracts
from merlin_experiments.phase2 import gsim_gate as GATE
from merlin_experiments.phase2 import prompt as PP
from merlin_experiments.phase2.claims import pk as PK
from merlin_experiments.phase2.contracts import (
    ROUND_DEADLINE_EXIT,
    StageGateError,
)
from merlin_experiments.phase2.contracts import (
    canonical_json as _canonical_json,
)
from merlin_experiments.phase2.contracts import (
    require_executable as _require_executable,
)
from merlin_experiments.phase2.contracts import (
    sha256_file as _sha256_file,
)
from merlin_experiments.phase2.contracts import (
    write_json as _write_json,
)

LEGACY_TREATMENT_SOURCES = frozenset(
    {
        "codex_binary",
        "performance_authoring_stage",
        "performance_campaign",
        "performance_gsim_gate",
        "performance_pk_claim",
        "performance_prompt",
        "codex_driver",
        "codex_model_bridge",
        "benchharness",
        "sandbox_bwrap",
        "sandbox_toolchain",
        "sandbox_answer_surfaces",
        "shared_access_policy",
        "target_experiment_loader",
        "experiment_tokens",
        "aet_codex_normalizer",
        "aet_codex_importer",
        "aet_reconciliation",
        "aet_activity_classifier",
        "aet_canonical_logger",
    }
)


SOURCE_POLICY_VERSION = 4
ACCOUNTING_POLICY = "aet_explicit_price_snapshot_v1"
PACKAGED_TREATMENT_SOURCES = LEGACY_TREATMENT_SOURCES | frozenset(
    {"performance_telemetry", "performance_contracts", "shared_digest"}
)
EXPLICIT_PRICE_TREATMENT_SOURCES = PACKAGED_TREATMENT_SOURCES | frozenset(
    {"aet_price_snapshot", "aet_pricing", "aet_cost", "aet_billing"}
)
TREATMENT_SOURCES = EXPLICIT_PRICE_TREATMENT_SOURCES | frozenset(
    {"performance_package_sources", "python_source_membership"}
)


def _package_source_record(directory: Path | None = None) -> dict[str, Any]:
    """Embed source membership in the existing preflight, never a second seal."""
    if directory is None:
        entry = module_source_path("merlin_experiments.phase2")
        _sha256_file(entry)
        directory = entry.parent
    try:
        members = source_membership.python_members(directory, label="phase-2")
    except source_membership.SourceMembershipError as exc:
        raise StageGateError(str(exc)) from exc
    hashes = {name: _sha256_file(path) for name, path in members.items()}
    result = {"path": str(directory.resolve()), "sha256": _sha256(_canonical_json(hashes)), "members": hashes}
    _validate_package_source_record(result)
    return result


def _validate_package_source_record(record: object) -> None:
    """Decode historical closure evidence without accessing live source files."""
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "members"}:
        raise StageGateError("performance package source closure is malformed")
    root, members = record["path"], record["members"]
    if (
        not isinstance(root, str)
        or "\0" in root
        or not Path(root).is_absolute()
        or str(Path(root)) != root
        or ".." in Path(root).parts
        or not isinstance(members, Mapping)
        or "__init__.py" not in members
    ):
        raise StageGateError("performance package source closure has invalid root or membership")
    for name, sha in members.items():
        if (
            not isinstance(name, str)
            or "\0" in name
            or PurePosixPath(name).is_absolute()
            or str(PurePosixPath(name)) != name
            or any(part in {"..", "__pycache__"} for part in PurePosixPath(name).parts)
            or PurePosixPath(name).suffix != ".py"
            or "\\" in name
            or not digest.is_sha256(sha)
        ):
            raise StageGateError("performance package source closure has unsafe or malformed member")
    if not digest.is_sha256(record["sha256"]) or _sha256(_canonical_json(dict(members))) != record["sha256"]:
        raise StageGateError("performance package source closure digest changed")


def treatment_identity(
    preflight: Mapping[str, Any],
    *,
    authoring_stage_sha256: str | None = None,
    codex_binary_sha256: str | None = None,
) -> dict[str, Any]:
    """Decode exact versioned source roles without rewriting historical declarations."""
    if not isinstance(preflight, Mapping):
        raise StageGateError("performance telemetry declaration is not an object")
    version = preflight.get("schema_version")
    policy = preflight.get("source_policy_version")
    if type(version) is int and version == 1 and policy is None:
        required = LEGACY_TREATMENT_SOURCES
    elif type(version) is int and version == 2 and type(policy) is int and policy == 2:
        required = PACKAGED_TREATMENT_SOURCES
    elif type(version) is int and version == 3 and type(policy) is int and policy == 3:
        required = EXPLICIT_PRICE_TREATMENT_SOURCES
    elif type(version) is int and version == 4 and type(policy) is int and policy == SOURCE_POLICY_VERSION:
        required = TREATMENT_SOURCES
    else:
        raise StageGateError("unsupported performance telemetry source policy")
    sources = preflight.get("sources")
    if (
        not isinstance(sources, Mapping)
        or set(sources) != required
        or any(
            not isinstance(source, Mapping) or not digest.is_sha256(source.get("sha256")) for source in sources.values()
        )
    ):
        raise StageGateError("performance telemetry treatment source identity is incomplete")
    if version == 4:
        _validate_package_source_record(sources["performance_package_sources"])
    source_sha256 = {str(name): str(source["sha256"]) for name, source in sorted(sources.items())}
    resolution = preflight.get("model_resolution")
    if (
        not isinstance(resolution, Mapping)
        or not isinstance(resolution.get("requested_model"), str)
        or not resolution.get("requested_model")
        or not isinstance(resolution.get("resolved_model"), str)
        or not resolution.get("resolved_model")
        or not isinstance(resolution.get("codex_model_map"), str)
    ):
        raise StageGateError("agent telemetry preflight lacks its exact model resolution identity")
    if (
        authoring_stage_sha256 is not None and source_sha256["performance_authoring_stage"] != authoring_stage_sha256
    ) or (codex_binary_sha256 is not None and source_sha256["codex_binary"] != codex_binary_sha256):
        raise StageGateError("performance candidate executable/source identities differ from telemetry preflight")
    return {
        "telemetry_preflight_sha256": _sha256(_canonical_json(preflight)),
        "codex_binary_sha256": source_sha256["codex_binary"],
        "authoring_stage_sha256": source_sha256["performance_authoring_stage"],
        "telemetry_source_sha256": source_sha256,
        "requested_model": str(resolution["requested_model"]),
        "resolved_model": str(resolution["resolved_model"]),
        "codex_model_map": str(resolution["codex_model_map"]),
    }


def _import_codex_driver():
    from merlin_experiments.phase1.providers import codex_agent  # noqa: PLC0415

    return codex_agent


def _telemetry_source_record(obj: object, *, label: str) -> dict[str, str]:
    source = str(obj) if isinstance(obj, Path) else inspect.getsourcefile(obj)
    if not source:
        raise StageGateError(f"{label} has no inspectable source file")
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise StageGateError(f"{label} source is absent, linked, or non-regular: {path}")
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256_file(path)}


def _declared_price_rate(path: Path, model: str) -> tuple[float, float, float, float] | None:
    """Legacy input matching, without a second cost calculation or fallback table."""
    matches = [(key, rate) for key, rate in _normalized_rates(path).items() if key in model.lower()]
    return max(matches, key=lambda item: (len(item[0]), item[0]))[1] if matches else None


def _normalized_rates(path: Path) -> dict[str, tuple[float, float, float, float]]:
    """Adapt only legacy input syntax/cache defaults; AET owns all dollar calculations."""
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping) or not document:
        raise StageGateError("telemetry price table must be a nonempty rate mapping")
    rates = {}
    for key, value in document.items():
        lowered = str(key).lower()
        if not lowered or lowered in rates:
            raise StageGateError("telemetry price keys must be nonempty and case-insensitively unique")
        try:
            if isinstance(value, Mapping):
                input_rate, output_rate = float(value["input"]), float(value["output"])
                cache_read = float(value.get("cache_read", input_rate * 0.10))
                cache_write = float(value.get("cache_creation", value.get("cache_write", input_rate * 1.25)))
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                values = [float(item) for item in value]
                if not 2 <= len(values) <= 4:
                    raise ValueError("rate requires two to four buckets")
                input_rate, output_rate = values[:2]
                cache_read = values[2] if len(values) >= 3 else input_rate * 0.10
                cache_write = values[3] if len(values) >= 4 else input_rate * 1.25
            else:
                raise ValueError("rate must be a mapping or sequence")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise StageGateError(f"malformed telemetry price rate: {key}") from exc
        rate = (input_rate, output_rate, cache_read, cache_write)
        if any(not math.isfinite(number) or number < 0 for number in rate):
            raise StageGateError("telemetry price rates must be finite and nonnegative")
        rates[lowered] = rate
    return rates


def _price_snapshot_document(path: Path) -> dict[str, Any]:
    return {
        "price_table_id": "merlin-legacy-" + _sha256_file(path),
        "provider": "openai",
        "verified": False,
        "unit": "per_million_tokens",
        "normalization_policy": "merlin_legacy_cache_defaults_v1",
        "rounding_policy": {"aet_cost_decimals": 6, "legacy_display_decimals": 4},
        "rates": {key: list(rate) for key, rate in sorted(_normalized_rates(path).items())},
    }


def _snapshot(document: Mapping[str, Any]):
    from aet.trajectory.price_snapshot import PriceSnapshot

    return PriceSnapshot(
        price_table_id=document["price_table_id"],
        rates={key: tuple(rate) for key, rate in document["rates"].items()},
        provider="openai",
        verified=False,
        raw_bytes=_canonical_json(document),
    )


def read_preflight(path: Path) -> dict[str, Any]:
    """Read the host declaration, never resolving a symlink or inventing a live fallback."""
    _sha256_file(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise StageGateError("performance telemetry declaration is not an object")
    treatment_identity(record)
    return record


def _verified_snapshot(preflight: Mapping[str, Any], model: str):
    identity = treatment_identity(preflight)
    if preflight.get("accounting_policy") != ACCOUNTING_POLICY or preflight.get("schema_version") not in (3, 4):
        raise StageGateError(
            "new telemetry requires explicit price-snapshot qualification; preserve historical records"
        )
    if identity["resolved_model"] != model:
        raise StageGateError("telemetry model differs from the preflight price identity")
    price = preflight.get("price_table") or {}
    path = Path(str(price.get("path") or ""))
    if _sha256_file(path) != price.get("sha256"):
        raise StageGateError("pinned telemetry price table changed before accounting")
    declared = price.get("snapshot") or {}
    document = _price_snapshot_document(path)
    if document != declared.get("document") or _sha256(_canonical_json(document)) != declared.get("sha256"):
        raise StageGateError("normalized telemetry price snapshot differs from pinned input")
    for label, source in preflight["sources"].items():
        if label == "performance_package_sources":
            if _package_source_record(Path(source["path"])) != source:
                raise StageGateError("pinned telemetry implementation changed: performance_package_sources")
            if Path(source["path"]) != module_source_path("merlin_experiments.phase2").parent.resolve():
                raise StageGateError("pinned performance package differs from the active source owner")
            continue
        if _sha256_file(Path(str(source.get("path") or ""))) != source["sha256"]:
            raise StageGateError(f"pinned telemetry implementation changed: {label}")
    return _snapshot(document)


def _accounting(path: Path, *, model: str, snapshot) -> dict[str, Any]:
    from aet.trajectory.codex import CodexNormalizer
    from aet.trajectory.price_snapshot import cost_record_for

    from merlin.targetgen import experiment_tokens as ET

    record = ET.parse_agent_transcript(
        path, driver="codex", model=model, billing_mode=ET.SUBSCRIPTION_NOTIONAL, include_cost=False
    )
    normalizer = CodexNormalizer()
    normalizer.feed_text(path.read_text(encoding="utf-8"))
    totals = normalizer.result().totals()
    buckets = {
        "tokens_input": "uncached_input_tokens",
        "tokens_cached": "cached_input_tokens",
        "tokens_cache_write": "cache_write_input_tokens",
        "tokens_output": "output_tokens",
    }
    if (
        record.get("available") is not True
        or record.get("usage_complete") is not True
        or any(record.get(left, 0) != (totals.get(right) or 0) for left, right in buckets.items())
    ):
        raise StageGateError("raw Codex token evidence does not reconcile with AET accounting")
    cost = cost_record_for(
        totals,
        snapshot=snapshot,
        model_requested=model,
        model_resolved=model,
        billing_row={"provider": "openai", "billing_mode": "subscription"},
        calculated_at="",
    )
    if cost.is_unpriced or cost.kind != "subscription_notional":
        raise StageGateError("pinned AET snapshot cannot price complete Codex usage")
    record.update(
        estimated_cost_usd=None,
        subscription_notional_usd=round(cost.value_usd, 4),
        cost_unavailable_reason=(
            "subscription_notional: a subscription seat is not billed per token; any dollar figure is "
            "what the same traffic would have cost metered, not money spent"
        ),
        cost_provenance=cost.to_dict(),
    )
    return record


def artifact_names(evidence: Mapping[str, Any]) -> set[str]:
    """Exact artifact inventory for historical and explicit-price telemetry."""
    names = {
        "combined_raw",
        "trajectory",
        "reconciliation",
        "token_ledger",
        "tool_ledger",
        "cost_time_toolcalls",
        "activity_share",
        "preflight",
        "aet_metrics_log",
    }
    version = evidence.get("policy_version")
    if version is None and "accounting_policy" not in evidence and "price_snapshot_sha256" not in evidence:
        return names
    if type(version) is not int or version != 3 or evidence.get("accounting_policy") != ACCOUNTING_POLICY:
        raise StageGateError("unsupported performance telemetry accounting policy")
    if not digest.is_sha256(evidence.get("price_snapshot_sha256")):
        raise StageGateError("performance telemetry price snapshot identity is absent")
    return names | {"price_snapshot"}


def verify_price_evidence(
    evidence: Mapping[str, Any], rounds: Sequence[Mapping[str, Any]], preflight: Mapping[str, Any]
) -> None:
    """Verify sealed price roles without consulting current prices or rewriting history."""
    artifact_names(evidence)
    identity = treatment_identity(preflight)
    if evidence.get("policy_version") is None:
        if preflight.get("schema_version") not in (1, 2):
            raise StageGateError("historical telemetry cannot omit current price evidence")
        return
    if preflight.get("schema_version") not in (3, 4) or preflight.get("accounting_policy") != ACCOUNTING_POLICY:
        raise StageGateError("telemetry price policy differs from preflight")
    declared = (preflight.get("price_table") or {}).get("snapshot") or {}
    price_sha = evidence["price_snapshot_sha256"]
    path = Path(str(((evidence.get("artifacts") or {}).get("price_snapshot") or {}).get("path") or ""))
    if (
        declared.get("sha256") != price_sha
        or _sha256(_canonical_json(declared.get("document"))) != price_sha
        or _sha256_file(path) != price_sha
    ):
        raise StageGateError("sealed price snapshot differs from preflight")
    model = identity["resolved_model"]
    snapshot = _snapshot(declared["document"])
    for item in [evidence, *[(row.get("telemetry") or {}) for row in rounds]]:
        accounting = item.get("accounting") or {}
        if not isinstance(accounting, Mapping) or not isinstance(accounting.get("cost_provenance"), Mapping):
            raise StageGateError("sealed price accounting is malformed")
        cost = accounting.get("cost_provenance") or {}
        amount = cost.get("value_usd")
        if (
            item.get("accounting_policy") != ACCOUNTING_POLICY
            or item.get("price_snapshot_sha256") != price_sha
            or cost.get("price_table_sha256") != price_sha
            or cost.get("kind") != "subscription_notional"
            or cost.get("model_requested") != model
            or cost.get("model_resolved") != model
            or accounting.get("model") != model
            or type(amount) not in (float, int)
            or not math.isfinite(amount)
            or amount < 0
            or accounting.get("subscription_notional_usd") != round(amount, 4)
            or accounting.get("estimated_cost_usd") is not None
        ):
            raise StageGateError("sealed round/final accounting uses a different price authority")
        artifacts = item.get("artifacts") or {}
        raw_record = artifacts.get("combined_raw" if item is evidence else "raw") or {}
        raw_path = Path(str(raw_record.get("path") or ""))
        if _sha256_file(raw_path) != raw_record.get("sha256"):
            raise StageGateError("sealed accounting raw evidence changed")
        _verify_raw_accounting(accounting, raw_path, model=model, snapshot=snapshot)
    trajectory_path = Path(evidence["artifacts"]["trajectory"]["path"])
    _sha256_file(trajectory_path)
    if json.loads(trajectory_path.read_text(encoding="utf-8")).get("cost") != evidence["accounting"]["cost_provenance"]:
        raise StageGateError("sealed trajectory and token ledger price authority differ")


def _verify_raw_accounting(accounting: Mapping[str, Any], raw: Path, *, model: str, snapshot) -> None:
    """Reprice each raw stream independently; never sum rounded per-round amounts."""
    expected = _accounting(raw, model=model, snapshot=snapshot)
    fields = (
        "model",
        "tokens_input",
        "tokens_fresh_input",
        "tokens_cached",
        "tokens_cache_write",
        "tokens_output",
        "tokens_total",
        "usage_complete",
        "cost_provenance",
        "estimated_cost_usd",
        "subscription_notional_usd",
    )
    if any(accounting.get(field) != expected.get(field) for field in fields):
        raise StageGateError("telemetry accounting differs from its raw evidence and pinned model/price authority")


def prepare(
    *, model: str, authoring_stage: Path, price_table: Path | None = None, codex_binary: str | Path | None = None
) -> dict[str, Any]:
    """Prove the exact raw-Codex -> token ledger -> AET trajectory path before a paid turn.

    This is deliberately strict for the performance experiment even though the shared AET bridge is
    soft for ordinary developer runs.  It performs an in-memory parser canary and pins every parser
    implementation plus the notional price input; it neither creates a run directory nor launches an
    agent.
    """
    try:
        from aet.tracking.run_logger import EvalRunLogger  # noqa: PLC0415
        from aet.trajectory import billing, cost, price_snapshot, pricing
        from aet.trajectory.classify import ActivityClassifier, ActivityConfig  # noqa: PLC0415
        from aet.trajectory.codex import CodexNormalizer  # noqa: PLC0415
        from aet.trajectory.importers.codex import build_trajectory_from_run  # noqa: PLC0415
        from aet.trajectory.reconcile import reconcile_codex  # noqa: PLC0415

        from merlin.targetgen import experiment_tokens as ET  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise StageGateError(f"required AET/Codex telemetry stack is unavailable: {exc}") from exc
    CA = _import_codex_driver()
    try:
        from merlin_experiments.phase1.providers import agent_bridge as model_bridge  # noqa: PLC0415

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
        "schema_version": 4,
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
            "codex_driver": _telemetry_source_record(CA, label="Codex driver"),
            "codex_model_bridge": _telemetry_source_record(model_bridge, label="Codex model bridge"),
            "benchharness": _telemetry_source_record(hash_tree, label="candidate tree hashing"),
            "sandbox_bwrap": _telemetry_source_record(BW, label="agent bwrap policy"),
            "sandbox_toolchain": _telemetry_source_record(TC, label="agent toolchain policy"),
            "sandbox_answer_surfaces": _telemetry_source_record(answer_surfaces, label="answer-surface policy"),
            "shared_access_policy": _telemetry_source_record(
                shared_access_policy, label="shared access and audit policy"
            ),
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
    from merlin.targetgen import experiment_tokens as ET  # noqa: PLC0415

    snapshot = _verified_snapshot(preflight_record, model)
    try:
        from aet.trajectory.importers.codex import import_codex_run  # noqa: PLC0415
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
            _write_jsonl(
                ts_dir / destination.name,
                [
                    {"ts": wrapper["arrived_at"], "line": line}
                    for line, wrapper in zip(raw_lines, wrappers, strict=True)
                ],
            )
            total_wall += float((evidence.get("summary") or {}).get("wall_s", 0.0))

    trajectory, normalized = import_codex_run(
        raw_dir,
        timestamped=ts_dir,
        run_id=run_id,
        model=model,
        billing_mode="subscription",
        provider="openai",
        calculated_at="",
        price_snapshot=snapshot_path,
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
