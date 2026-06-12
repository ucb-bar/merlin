"""Emit a design_pressure dict (conforming to ``design_pressure.schema.yaml``)."""
from __future__ import annotations

from typing import Iterable

from merlin.common import schemas


def emit_design_pressure(
    workload: str,
    cutpoints: Iterable[str],
    rpv: dict,
    candidate_contracts: Iterable[str],
    validate: bool = True,
) -> dict:
    """Build a schema-shaped design_pressure report.

    ``rpv`` is the Region Pressure Vector (see ``pressure_vector.compute_rpv``); its flat
    ``metrics`` populate the report's ``metrics`` field and its class breakdown is carried in
    ``pressure_classes`` for downstream readers.
    """
    report = {
        "workload": workload,
        "cutpoints": list(cutpoints),
        "metrics": dict(rpv.get("metrics", {})),
        "pressure_classes": dict(rpv.get("classes", {})),
        "candidate_contracts": list(candidate_contracts),
    }
    if validate:
        schemas.validate_or_raise(report, "design_pressure")
    return report
