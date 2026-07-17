"""LAYER 3 — ATTRIBUTION: the new glue that automates the manual ``kernel_breakdown.md``.

For each workload where an ``ours`` config TRAILS an expert (xnnpack/openblas), it:
  1. takes the measured Δ from layer 1 (ours wall vs expert wall -> "ours is X% of expert"),
  2. runs ``cca_compare.compare(expert_cca, ours_cca)`` (layer 2) -> typed ``Divergence``s,
  3. routes each via ``action_catalog.build_catalog`` -> ``CompilerAction``s,
  4. PAIRS them: "ours is X% of expert; structural deltas = {…}; routed actions = […]".

REUSES ``kernels.cca_compare`` and ``kernels.action_catalog`` unchanged.

HONEST framing (recorded in the report): static CCA decode gives the RANKING of factors, not exact
cycle fractions — there are no K1 perf counters here. The ``.vf``-vs-``.vv`` divergence (the
contraction/accumulator-residency form) is exactly the openvla/rdt2 gap driver the manual breakdown
identified; this module surfaces it automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from merlin.kernels.action_catalog import CompilerAction, build_catalog
from merlin.kernels.cca_compare import Divergence, compare

from .empirical import Measurement
from .spec import Config, Workload


@dataclass
class Attribution:
    """One ours-vs-expert attribution for a (workload, ours_config, expert_config)."""
    workload: str
    ours_config: str
    expert_config: str
    measured: dict[str, Any]          # ours/expert walls, ratio, pct-of-expert, ours_faster
    divergences: list[Divergence] = field(default_factory=list)
    actions: list[CompilerAction] = field(default_factory=list)
    unrouted: list[Divergence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _measured_pair(ours_m: Measurement, exp_m: Measurement) -> dict | None:
    if ours_m.status != "measured" or exp_m.status != "measured":
        return None
    if not ours_m.value or not exp_m.value:
        return None
    # lower wall = faster. ratio = ours/expert; <1 means ours is faster.
    ratio = ours_m.value / exp_m.value
    pct_of_expert = round(100.0 * exp_m.value / ours_m.value)  # ours speed as % of expert speed
    return {
        "ours_value": ours_m.value,
        "expert_value": exp_m.value,
        "ratio_ours_over_expert": ratio,
        "pct_of_expert": pct_of_expert,
        "ours_faster": ratio < 1.0,
        "metric_lower_is_better": True,
    }


def attribute(
    spec,
    measurements: dict[tuple[str, str], Measurement],
    ccas: dict[str, Any],          # {config_name: CCA|None} for the representative shape
    *,
    workload_ccas: dict[tuple[str, str], Any] | None = None,
) -> list[Attribution]:
    """Build the attribution records. For every (ours, expert) pair on every workload that has both
    measured, emit an Attribution pairing the measured gap with the CCA divergences + routed
    actions. ``workload_ccas`` lets the caller pass per-(config, workload) CCAs (preferred); falls
    back to the representative ``ccas``."""
    ours_cfgs = [c for c in spec.configs if c.kind == "ours"]
    expert_cfgs = [c for c in spec.configs if c.kind == "kernel_backend"]
    workload_ccas = workload_ccas or {}

    def cca_of(cfg_name: str, wl_name: str):
        if (cfg_name, wl_name) in workload_ccas:
            return workload_ccas[(cfg_name, wl_name)]
        return ccas.get(cfg_name)

    out: list[Attribution] = []
    for wl in spec.workloads:
        for oc in ours_cfgs:
            for ec in expert_cfgs:
                om = measurements.get((oc.name, wl.name))
                em = measurements.get((ec.name, wl.name))
                if om is None or em is None:
                    continue
                meas = _measured_pair(om, em)
                if meas is None:
                    continue
                ours_cca = cca_of(oc.name, wl.name)
                exp_cca = cca_of(ec.name, wl.name)
                notes: list[str] = []
                divs: list[Divergence] = []
                actions: list[CompilerAction] = []
                unrouted: list[Divergence] = []
                if ours_cca is None or exp_cca is None:
                    notes.append(
                        "no CCA on one side (baseline/scalar or undecoded); measured-only attribution")
                else:
                    ev = [f"{ec.name}:{exp_cca.provenance.get('decode_kernel', ec.name)}"]
                    divs = compare(exp_cca, ours_cca, evidence=ev)
                    # cite the .vf-vs-.vv counts when present (the kernel_breakdown.md evidence).
                    vf = (exp_cca.provenance.get("fma_loop_vfmacc_vf"),
                          ours_cca.provenance.get("fma_loop_vfmacc_vf"))
                    vv = (exp_cca.provenance.get("fma_loop_vfmacc_vf"),
                          ours_cca.provenance.get("fma_loop_vfmacc_vv"))
                    if (ours_cca.provenance.get("fma_loop_vfmacc_vv") and
                            not ours_cca.provenance.get("fma_loop_vfmacc_vf") and
                            exp_cca.provenance.get("fma_loop_vfmacc_vf")):
                        notes.append(
                            "vfmacc form: expert emits .vf (broadcast A scalar; "
                            f"vf={exp_cca.provenance.get('fma_loop_vfmacc_vf')}, vv=0); "
                            f"ours emits .vv (vf=0, vv={ours_cca.provenance.get('fma_loop_vfmacc_vv')}) "
                            "-> the per-K broadcast-ladder gap driver (kernel_breakdown.md).")
                    actions, unrouted = build_catalog(divs)
                if not meas["ours_faster"]:
                    notes.append(
                        f"ours trails {ec.name}: {meas['pct_of_expert']}% of expert speed "
                        f"(ratio {meas['ratio_ours_over_expert']:.2f}x of expert wall).")
                else:
                    notes.append(
                        f"ours BEATS {ec.name}: {meas['ratio_ours_over_expert']:.2f}x its wall "
                        f"({round(1.0/meas['ratio_ours_over_expert'],2)}x faster).")
                notes.append(
                    "static CCA decode gives the RANKING of structural factors, not exact cycle "
                    "fractions (no K1 perf counters).")
                out.append(Attribution(
                    workload=wl.name, ours_config=oc.name, expert_config=ec.name,
                    measured=meas, divergences=divs, actions=actions,
                    unrouted=unrouted, notes=notes))
    return out


def gap_driver_axes(attrs: list[Attribution]) -> set[str]:
    """The union of divergence axes flagged across all trailing attributions (the gap drivers)."""
    axes: set[str] = set()
    for a in attrs:
        if not a.measured.get("ours_faster"):
            axes.update(d.axis for d in a.divergences)
    return axes


# --- whole-model, region-by-region cross-compiler alignment (C7) ------------------------------
# Both Merlin and ExecuTorch(=XNNPACK) descend from the SAME captured model, so their regions carry
# the SAME model-layer provenance key (region_id / fqn). Joining on it lets us line up
# attention.3-vs-attention.3 across the two compilers instead of collapsing to one whole-model number.

@dataclass
class RegionAlignment:
    """One model layer compared across two compilers, matched by shared provenance."""
    key: str                          # the join key (region_id, else fqn, else region name)
    fqn: str
    role: str
    label: str                        # region label / bucket (from the profile name)
    ours_wall_ns: int | None
    expert_wall_ns: int | None
    wall_ratio: float | None          # ours/expert; <1 means Merlin is faster on THIS region
    ours_cos: float | None            # per-region equivalence (None = not scored, honest)
    expert_cos: float | None
    presence: str                     # "both" | "ours_only" | "expert_only"
    note: str = ""


def _region_key(r) -> str:
    return getattr(r, "region_id", "") or getattr(r, "fqn", "") or getattr(r, "name", "")


def align_regions(ours_regions, expert_regions, *, ours_name: str = "merlin",
                  expert_name: str = "executorch") -> list[RegionAlignment]:
    """Align two per-region profile lists (e.g. a Merlin ``BaselineResult.regions`` and an ExecuTorch
    one) by their shared ``region_id``/``fqn`` provenance. Emits one row per model layer, with the
    per-region wall ratio and per-region equivalence — and, crucially, flags a layer present on only
    ONE side (the delegation heterogeneity a whole-model number hides: a region ET left scalar, or a
    Merlin region ET fused away). Presence-asymmetry is surfaced, never averaged in."""
    ours = {_region_key(r): r for r in ours_regions if _region_key(r)}
    expert = {_region_key(r): r for r in expert_regions if _region_key(r)}
    rows: list[RegionAlignment] = []
    for k in dict.fromkeys([*ours, *expert]):
        o, e = ours.get(k), expert.get(k)
        base = o or e
        ow = getattr(o, "wall_ns", None) if o else None
        ew = getattr(e, "wall_ns", None) if e else None
        ratio = (ow / ew) if (ow and ew) else None
        presence = "both" if (o and e) else (f"{ours_name}_only" if o else f"{expert_name}_only")
        note = "" if presence == "both" else (
            f"region present only in {ours_name if o else expert_name} "
            "(delegation/vectorization heterogeneity — not comparable as a single whole-model number)")
        rows.append(RegionAlignment(
            key=k, fqn=getattr(base, "fqn", ""), role=getattr(base, "role", ""),
            label=getattr(base, "name", ""), ours_wall_ns=ow, expert_wall_ns=ew, wall_ratio=ratio,
            ours_cos=getattr(o, "cos", None) if o else None,
            expert_cos=getattr(e, "cos", None) if e else None,
            presence=presence, note=note))
    return rows
