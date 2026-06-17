"""Workload-family clustering — group VLAs by the system abstractions they imply.

Not every VLA wants the same accelerator. A future DSE engine should instantiate a different
search space per *workload family* rather than treating all VLAs as one. The recovered workload
class (:data:`topology.CLASS_*`) already separates iterative-denoise heads, autoregressive decode,
and single-shot regression. This module clusters the recaptured workloads into families and unions
the DSE axes each family's members suggest — the per-family enabled-axis set the search-space
template (:mod:`.search_space`) then turns into knobs.

Structural only: a family's enabled axes are the union of its members' candidate axes. No speedup,
ranking, or cycle number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.dse_guidance import topology as TOP

# Recovered workload class -> a short family label a DSE engine keys its search space on.
_FAMILY = {
    TOP.CLASS_FLOW_MATCHING: "iterative_denoise",
    TOP.CLASS_AUTOREGRESSIVE: "token_decode",
    TOP.CLASS_REGRESSION_PARALLEL: "single_shot",
    TOP.CLASS_UNKNOWN: "unknown",
}


def family_of(workload_class: str) -> str:
    return _FAMILY.get(workload_class, "unknown")


@dataclass
class FamilyRow:
    workload: str
    family: str
    enabled_axes: list[str] = field(default_factory=list)


def _enabled_axes(pkg) -> list[str]:
    return [c.axis for c in pkg.get("cands", [])]


def family_rows(packages) -> list[FamilyRow]:
    rows = []
    for p in packages:
        rows.append(FamilyRow(workload=p["case"].workload,
                              family=family_of(p["case"].topo.workload_class),
                              enabled_axes=_enabled_axes(p)))
    return sorted(rows, key=lambda r: (r.family, r.workload))


def family_axis_sets(packages) -> dict[str, set[str]]:
    """Family label -> union of DSE axes its member workloads suggest."""
    out: dict[str, set[str]] = {}
    for p in packages:
        fam = family_of(p["case"].topo.workload_class)
        out.setdefault(fam, set()).update(_enabled_axes(p))
    return out


def workload_family_csv(packages) -> str:
    from merlin.dse_guidance.case_study import _csv
    rows = [{"workload": r.workload, "family": r.family,
             "enabled_axes": "; ".join(r.enabled_axes)} for r in family_rows(packages)]
    return _csv(rows, ["workload", "family", "enabled_axes"])
