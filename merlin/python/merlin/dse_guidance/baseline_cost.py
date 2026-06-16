"""Parse a baseline cost breakdown and keep it honest.

The DSE axis triage is computed against a measured/modelled latency breakdown: a total and a
set of named cost components, each tagged with an evidence source. Two invariants:

  * Component sums are reconciled with the stated total. If they disagree we add an explicit
    ``residual`` component (tagged ``assumed``) and warn — never silently normalize away the
    discrepancy.
  * Every component carries an evidence tag. A component without a ``metadata_source`` entry
    defaults to ``assumed`` (the weakest tag), so an untagged number can never masquerade as
    a measured one.

Component names are the canonical set the axis catalog reduces against (``compute``,
``dma_memory``, ``packing``, ``cpu_dispatch``, ``sync``, ``intermediate_materialization``,
``capacity_spill``, ``other``). They may be given with or without a trailing ``_ms``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.common import schemas
from merlin.common.yaml import load_yaml

_SCHEMA = "baseline_cost"

# Canonical components the axis interventions reduce. ``other``/``residual`` are catch-alls.
CANONICAL_COMPONENTS: tuple[str, ...] = (
    "compute",
    "dma_memory",
    "packing",
    "cpu_dispatch",
    "sync",
    "intermediate_materialization",
    "capacity_spill",
    "other",
)
# Reconciliation tolerance (ms) between Σcomponents and the stated total.
_SUM_TOL_MS = 1e-6


def _strip_ms(name: str) -> str:
    return name[:-3] if name.endswith("_ms") else name


@dataclass
class BaselineCost:
    workload: str
    baseline_total_ms: float
    target_total_ms: float | None
    components: dict[str, float]              # component -> cost (canonical names, no _ms suffix)
    evidence: dict[str, str]                  # component -> evidence tag
    unit: str = "ms"                          # "ms" for measured/modelled; e.g. "cycles" for synth
    # Optional role-attributed sub-breakdowns. These let residency benefit be charged to the
    # repeated action head only (not the once-per-replan backbone) — the backbone/head fix.
    head_components: dict[str, float] = field(default_factory=dict)
    loop_invariant_components: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def component(self, name: str) -> float:
        return float(self.components.get(_strip_ms(name), 0.0))

    def evidence_for(self, name: str) -> str:
        return self.evidence.get(_strip_ms(name), "assumed")

    @property
    def has_head_breakdown(self) -> bool:
        """True iff the baseline separates the repeated-head cost from the backbone."""
        return bool(self.head_components)

    def head_component(self, name: str) -> float:
        return float(self.head_components.get(_strip_ms(name), 0.0))

    def loop_invariant_component(self, name: str) -> float:
        return float(self.loop_invariant_components.get(_strip_ms(name), 0.0))

    @property
    def target_gap_ms(self) -> float | None:
        """``baseline_total - target_total`` (None when no target was provided)."""
        if self.target_total_ms is None:
            return None
        return self.baseline_total_ms - self.target_total_ms


def parse(doc: dict) -> BaselineCost:
    """Validate and normalize a baseline cost mapping into a :class:`BaselineCost`."""
    schemas.validate_or_raise(doc, _SCHEMA)
    baseline = doc.get("baseline") or {}
    raw_components = baseline.get("components") or {}

    components: dict[str, float] = {}
    for raw_name, value in raw_components.items():
        if value is None:
            continue
        components[_strip_ms(str(raw_name))] = float(value)

    raw_sources = doc.get("metadata_source") or {}
    evidence: dict[str, str] = {}
    for comp in components:
        # metadata_source keys may be given with or without _ms.
        tag = raw_sources.get(f"{comp}_ms", raw_sources.get(comp))
        evidence[comp] = str(tag) if tag else "assumed"

    warnings: list[str] = []
    comp_sum = sum(components.values())
    total = baseline.get("total_ms")
    if total is None:
        total = comp_sum
        warnings.append("baseline.total_ms missing; using sum of components")
    total = float(total)

    # Reconcile: add an explicit residual rather than silently absorbing the difference.
    diff = total - comp_sum
    if abs(diff) > _SUM_TOL_MS:
        components["residual"] = components.get("residual", 0.0) + diff
        evidence.setdefault("residual", "assumed")
        warnings.append(
            f"components sum to {comp_sum:g} ms but total_ms={total:g} ms; "
            f"added residual={diff:g} ms (tagged assumed)"
        )

    target = doc.get("target") or {}
    target_total = target.get("total_ms")
    target_total = float(target_total) if target_total is not None else None

    def _subdict(key: str) -> dict[str, float]:
        raw = baseline.get(key) or {}
        return {_strip_ms(str(k)): float(v) for k, v in raw.items() if v is not None}

    head = _subdict("repeated_head")
    loop_inv = _subdict("loop_invariant")
    # Validate the head sub-breakdown does not exceed the whole (a head component cannot cost
    # more than the same component overall) — warn rather than silently accept.
    for comp, val in head.items():
        whole = components.get(comp, 0.0)
        if val > whole + _SUM_TOL_MS:
            warnings.append(f"repeated_head.{comp}={val:g} exceeds total {comp}={whole:g}")

    return BaselineCost(
        workload=str(doc["workload"]),
        baseline_total_ms=total,
        target_total_ms=target_total,
        components=components,
        evidence=evidence,
        unit=str(baseline.get("unit", "ms")),
        head_components=head,
        loop_invariant_components=loop_inv,
        warnings=warnings,
    )


def load(path) -> BaselineCost:
    """Load and parse a baseline cost breakdown from a YAML file."""
    return parse(load_yaml(path))
