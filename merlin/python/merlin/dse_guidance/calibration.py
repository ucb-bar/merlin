"""One calibration anchor: prediction vs measurement.

The point is a single *measured* anchor that ties a predicted quantity to a measured one, not
a full validation suite. We predict structural quantities (e.g. dispatch count =
dispatches_per_step * K) and compare against the measured value ingested from ``aet`` coupling
data. If no measurement is available we emit nothing — a calibration anchor with no
measurement would be a fabricated number, which the workstream forbids.

Output rows: ``workload, quantity, predicted, measured, error_pct, evidence_type``.
"""
from __future__ import annotations

import csv
import io

from merlin.dse_guidance.aet_ingest import CpuCoupling

_COLUMNS = ["workload", "quantity", "predicted", "measured", "error_pct", "evidence_type"]


def anchor_rows(workload: str, coupling: CpuCoupling | None,
                dispatches_per_step: int, K: int, num_regions: int = 1) -> list[dict]:
    """Build calibration rows from measured coupling vs structural predictions. May be empty."""
    if coupling is None:
        return []
    per = coupling.per_replan(dispatches_per_step, K, num_regions=num_regions)
    rows: list[dict] = []

    op = per.get("op_level")
    if op is not None:
        predicted = int(dispatches_per_step) * max(int(K), 1)
        measured = int(op["num_dispatches"])
        rows.append(_row(workload, "dispatch_count_op_level", predicted, measured,
                         op.get("source", "measured")))

    # cpu_dispatch_ms per replan: predicted from op-level measured per-dispatch cost is
    # definitionally equal, so the meaningful anchor is the op-level vs batched ratio.
    if op is not None and per.get("batched") is not None:
        ba = per["batched"]
        if ba["cpu_dispatch_ms"] > 0:
            predicted_ratio = (int(dispatches_per_step) * max(int(K), 1)) / max(ba["num_dispatches"], 1)
            measured_ratio = (op["cpu_dispatch_ms"] / ba["cpu_dispatch_ms"]
                              if ba["cpu_dispatch_ms"] else 0.0)
            rows.append(_row(workload, "dispatch_overhead_ratio",
                             round(predicted_ratio, 4), round(measured_ratio, 4),
                             per.get("source", "measured")))
    return rows


def _row(workload: str, quantity: str, predicted, measured, evidence_type: str) -> dict:
    error_pct = None
    if isinstance(measured, (int, float)) and measured:
        error_pct = round(abs(predicted - measured) / abs(measured) * 100.0, 4)
    return {
        "workload": workload, "quantity": quantity,
        "predicted": predicted, "measured": measured,
        "error_pct": error_pct, "evidence_type": evidence_type,
    }


def anchor_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_COLUMNS)
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r[k]) for k in _COLUMNS})
    return buf.getvalue()
