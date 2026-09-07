"""Predeclared semantic sampling for functional engine qualification only.

The certificate still contains observations of exact workloads. This declaration says which
observations admit a functional regrade; it never widens the tuning engine's timing envelope.
Producer and consumer independently apply the same deterministic policy to their own cohort.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

import perf_gsim_gate as GATE
import produce_gsim_certificate as PRODUCER

SCHEMA = "merlin.functional-coverage.v1"
POLICY = "minimum-per-operation-semantics.v1"


class CoverageError(ValueError):
    """The declared sample does not cover the independently derived cohort."""


def _costs(model: Mapping[str, Any] | None, pins: Mapping | None) -> dict[str, float]:
    if model is None:
        return {}
    if not isinstance(model, Mapping) or model.get("engine") != GATE.REFERENCE_ENGINE:
        raise CoverageError("sample cost model must name the reference engine")
    expected = {name: pins[name]["sha256"] for name in sorted(GATE.REQUIRED_PINS)} if pins else None
    if expected is None or model.get("pins") != expected:
        raise CoverageError("sample cost model pins differ from the qualification engines")
    values = model.get("seconds")
    if not isinstance(values, Mapping) or not values:
        raise CoverageError("sample cost model has no workload estimates")
    if any(not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, (float, int))
           or not math.isfinite(v) or v < 0 for k, v in values.items()):
        raise CoverageError("sample cost estimates must be finite nonnegative seconds")
    return dict(values)


def derive(cases: Sequence[tuple[str, Path]], *, reference_cost_model: Mapping | None = None,
           pins: Mapping | None = None) -> dict[str, Any]:
    """Select one representative per stratum without reading outputs or capture availability.

    Cost estimates are frozen run inputs. If a stratum is only partly priced, use input element
    counts for the WHOLE stratum rather than letting missing estimates lose to known ones.
    """
    costs = _costs(reference_cost_model, pins)
    strata: dict[str, dict[str, Any]] = {}
    identities: set[str] = set()
    for identity, manifest in cases:
        workload = PRODUCER.derive_workload(manifest)
        if GATE.workload_sha256(workload) != identity:
            raise CoverageError("sample descriptor differs from its frozen workload identity")
        if identity in identities:
            continue
        identities.add(identity)
        semantic = {key: workload[key] for key in ("operation", "semantics")}
        key = hashlib.sha256(GATE.canonical_json(semantic).encode()).hexdigest()
        descriptor = yaml.safe_load(Path(manifest).read_text(encoding="utf-8"))
        elements = sum(math.prod(tensor["shape"]) for tensor in descriptor["inputs"])
        stratum = strata.setdefault(key, {"stratum_sha256": key, **semantic, "elements": {}})
        stratum["elements"][identity] = elements
    if not identities:
        raise CoverageError("cannot qualify an empty functional cohort")
    rows, selected = [], []
    for key, stratum in sorted(strata.items()):
        elements = stratum.pop("elements")
        measured = all(identity in costs for identity in elements)
        scores = {identity: costs[identity] if measured else elements[identity]
                  for identity in sorted(elements)}
        representative = min(scores, key=lambda identity: (scores[identity], identity))
        selected.append(representative)
        rows.append({**stratum, "workload_sha256": sorted(elements),
                     "ranking_basis": "reference_seconds" if measured else "input_elements_proxy",
                     "ranking_values": scores, "selected": representative})
    return {"schema": SCHEMA, "mode": "stratified", "policy": POLICY,
            "reference_cost_model": reference_cost_model,
            "strata": rows, "selected": sorted(selected),
            "unsampled": sorted(identities - set(selected)),
            "limits": "Output agreement on selected workloads only; omitted shapes and cycle "
                      "equivalence are not independently verified."}


def verify(coverage: Any, cases: Sequence[tuple[str, Path]], members: set[str], *,
           pins: Mapping | None = None) -> dict[str, Any]:
    """Missing metadata means exact; only an explicitly valid sample relaxes missing members."""
    full = {identity for identity, _ in cases}
    extras = members - full
    if extras:
        raise CoverageError("functional certificate extras=" + ", ".join(sorted(extras)))
    if coverage is None:
        if members != full:
            raise CoverageError("functional certificate is not the exact admitted public+hidden cohort: "
                                "missing=" + ", ".join(sorted(full - members)))
        return {"mode": "exact", "selected": sorted(full), "unsampled": []}
    if not isinstance(coverage, Mapping) or coverage.get("mode") != "stratified":
        raise CoverageError("unknown functional coverage mode")
    expected = derive(cases, reference_cost_model=coverage.get("reference_cost_model"), pins=pins)
    if coverage != expected:
        raise CoverageError("functional sample differs from independently derived strata and selection")
    if members != set(expected["selected"]):
        raise CoverageError("functional sample is missing a stratum or differs from the sealed selection")
    return expected
