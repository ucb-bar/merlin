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

SOURCE_POLICY_VERSION = 5

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

TARGET_ACCESS_TREATMENT_SOURCES = TREATMENT_SOURCES | frozenset(
    {"historical_target_access_policy", "historical_target_access_data"}
)


def _package_source_record(directory: Path | None = None) -> dict[str, Any]:
    """Embed source membership in the existing preflight, never a second seal."""
    if directory is None:
        entry = module_source_path("merlin.development.backend.optimization")
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
    elif type(version) is int and version == 4 and type(policy) is int and policy == 4:
        required = TREATMENT_SOURCES
    elif type(version) is int and version == 5 and type(policy) is int and policy == SOURCE_POLICY_VERSION:
        required = TARGET_ACCESS_TREATMENT_SOURCES
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
    if version in (4, 5):
        _validate_package_source_record(sources["performance_package_sources"])
    driver = sources["codex_driver"]
    if "support" in driver:
        support = driver["support"]
        if not isinstance(support, Mapping) or set(support) != {"codex_runtime"}:
            raise StageGateError("Codex driver support source identity is incomplete")
        runtime = support["codex_runtime"]
        if (
            set(driver) != {"path", "sha256", "support"}
            or not isinstance(runtime, Mapping)
            or set(runtime) != {"path", "sha256"}
            or not isinstance(runtime["path"], str)
            or not runtime["path"]
            or not digest.is_sha256(runtime["sha256"])
        ):
            raise StageGateError("Codex driver runtime source identity is malformed")
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
    from merlin.development.backend.providers import codex_launch_codex_agent as codex_agent  # noqa: PLC0415

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
    if preflight.get("accounting_policy") != ACCOUNTING_POLICY or preflight.get("schema_version") not in (3, 4, 5):
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
            if Path(source["path"]) != module_source_path("merlin.development.backend.optimization").parent.resolve():
                raise StageGateError("pinned performance package differs from the active source owner")
            continue
        if _sha256_file(Path(str(source.get("path") or ""))) != source["sha256"]:
            raise StageGateError(f"pinned telemetry implementation changed: {label}")
        if label == "codex_driver" and "support" in source:
            runtime = source["support"]["codex_runtime"]
            active = module_source_path("merlin.development.backend.providers.codex_runtime").resolve()
            if Path(runtime["path"]) != active or _sha256_file(active) != runtime["sha256"]:
                raise StageGateError("pinned telemetry implementation changed: codex_runtime")
    return _snapshot(document)


def _accounting(path: Path, *, model: str, snapshot) -> dict[str, Any]:
    from aet.trajectory.codex import CodexNormalizer
    from aet.trajectory.price_snapshot import cost_record_for

    from merlin.development.backend import transcript_tokens as ET

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
    if preflight.get("schema_version") not in (3, 4, 5) or preflight.get("accounting_policy") != ACCOUNTING_POLICY:
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
