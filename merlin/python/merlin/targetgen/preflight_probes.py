"""One descriptor-driven behavioral-probe protocol for every code-generation layer.

A probe adapter may compile/run scalar code, RVV, or a target dialect.  Core infrastructure sees only a
target-owned fixture plus operation identities (``{domain, dialect, operation}``) and receives normalized
observations.  Concrete lowering and execution stay with the adapter that owns that environment; adding a
target or dialect never adds a branch here.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Mapping


class ProbeProtocolError(ValueError):
    """A declaration, adapter, or observation did not satisfy the generic probe protocol."""


def _operation_key(value: Any) -> tuple[str, str, str]:
    if not isinstance(value, Mapping):
        raise ProbeProtocolError("operation identity must be a mapping")
    domain, dialect, operation = value.get("domain"), value.get("dialect"), value.get("operation")
    if (not isinstance(domain, str) or not domain.strip()
            or not isinstance(dialect, str) or not dialect.strip()
            or not isinstance(operation, str) or not operation.strip()):
        raise ProbeProtocolError("operation identity requires non-empty domain, dialect, and operation")
    return domain.strip(), dialect.strip(), operation.strip()


def _load_adapter(reference: str) -> Callable:
    """Resolve one trusted descriptor's ``module:callable`` adapter reference."""
    module_name, sep, attr = str(reference).partition(":")
    if not sep or not module_name or not attr:
        raise ProbeProtocolError(f"invalid probe adapter reference {reference!r}")
    module = importlib.import_module(module_name)
    adapter = getattr(module, attr, None)
    if not callable(adapter):
        raise ProbeProtocolError(f"probe adapter {reference!r} is not callable")
    return adapter


def _operation_contract_for_experiment(te) -> dict[str, Any]:
    """Load operation declarations from the descriptor-selected/discovered target contract."""
    from .operation_capabilities import operation_contract_for_target
    from .target_experiment import declared_vs_resolved_contract, load_capability_manifest

    declared, _resolved, verdict = declared_vs_resolved_contract(te)
    if verdict == "mismatch":
        raise ProbeProtocolError(
            "descriptor-declared and registry-resolved target contracts disagree")
    explicit = declared if verdict == "declared_only" else None
    manifest = load_capability_manifest(te.target, contract_path=explicit)
    return operation_contract_for_target(te.target, manifest.contract)


def _unknown_observations(probe, detail: str) -> list[dict[str, Any]]:
    return [{**dict(operation), "status": "unknown",
             "evidence": {"kind": "preflight_adapter", "detail": detail}}
            for operation in probe.requirements["operations"]]


def _normalize_result(probe, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProbeProtocolError("adapter result must be a mapping")
    observations = raw.get("observations")
    if not isinstance(observations, list):
        raise ProbeProtocolError("adapter result must contain an observations list")
    required = [_operation_key(op) for op in probe.requirements["operations"]]
    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise ProbeProtocolError(f"observation {index} must be a mapping")
        key = _operation_key(observation)
        if key in observed:
            raise ProbeProtocolError(f"adapter returned duplicate observation for {key}")
        status = observation.get("status")
        if status not in {"supported", "unsupported", "unknown"}:
            raise ProbeProtocolError(f"observation for {key} has invalid status {status!r}")
        evidence = observation.get("evidence")
        if (not isinstance(evidence, Mapping) or not str(evidence.get("kind") or "").strip()
                or not str(evidence.get("detail") or "").strip()):
            raise ProbeProtocolError(f"observation for {key} requires evidence.kind and evidence.detail")
        observed[key] = dict(observation)
    missing = [key for key in required if key not in observed]
    extra = [key for key in observed if key not in required]
    if missing or extra:
        raise ProbeProtocolError(
            f"adapter observations differ from requirements (missing={missing}, extra={extra})")
    normalized = [observed[key] for key in required]
    return {
        "ok": all(observation["status"] == "supported" for observation in normalized),
        "reason": str(raw.get("reason") or "adapter returned behavioral observations"),
        "observations": normalized,
    }


def run_declared_capability_probes(
        te, *, workdir, timeout: int = 600,
        operation_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute every declared probe and return normalized, operation-addressable observations.

    Adapters receive the same keyword interface regardless of their layer:
    ``adapter(te=, probe=, workdir=, timeout=)``. They own compilation/execution and return
    ``{reason, observations:[{domain,dialect,operation,status,evidence:{kind,detail}}]}``.
    """
    probes = tuple(getattr(te, "preflight_capability_probes", ()) or ())
    if not probes:
        return {"ok": True, "probes": [], "reason": "n/a (no capability probes declared)"}
    from .operation_capabilities import merge_operation_contracts, merge_operation_observations

    try:
        declared = merge_operation_contracts(
            operation_contract if operation_contract is not None
            else _operation_contract_for_experiment(te))
    except Exception as exc:  # noqa: BLE001 -- unavailable declarations make every claim unknown
        detail = f"operation capability contract unavailable: {type(exc).__name__}: {exc}"
        rows = [{
            "ok": False,
            "reason": detail,
            "observations": _unknown_observations(probe, detail),
            "capability": probe.capability,
            "adapter": probe.adapter,
            "fixture": dict(probe.fixture),
        } for probe in probes]
        return {"ok": False, "probes": rows, "reason": detail}
    declared_ids = {_operation_key(operation) for operation in declared["operations"]}
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    mergeable_observations: list[dict[str, Any]] = []
    for index, probe in enumerate(probes):
        probe_dir = root / f"probe_{index:02d}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        try:
            required_ids = {_operation_key(operation)
                            for operation in probe.requirements["operations"]}
            undeclared = sorted(required_ids - declared_ids)
            if undeclared:
                raise ProbeProtocolError(
                    f"required operations are not declared by the operation contract: {undeclared}")
            adapter = _load_adapter(probe.adapter)
            normalized = _normalize_result(
                probe, adapter(te=te, probe=probe, workdir=probe_dir, timeout=timeout))
        except Exception as exc:  # noqa: BLE001 -- a broken/unavailable probe fails closed with evidence
            detail = f"{type(exc).__name__}: {exc}"
            normalized = {"ok": False, "reason": detail,
                          "observations": _unknown_observations(probe, detail)}
        mergeable_observations.extend(
            observation for observation in normalized["observations"]
            if _operation_key(observation) in declared_ids)
        rows.append({**normalized, "capability": probe.capability, "adapter": probe.adapter,
                     "fixture": dict(probe.fixture)})
    ok = all(row["ok"] for row in rows)
    observed_contract = merge_operation_observations(declared, mergeable_observations)
    return {"ok": ok, "probes": rows, "operation_capabilities": observed_contract,
            "reason": ("all declared capability probes passed" if ok
                       else "one or more declared capability probes failed")}
