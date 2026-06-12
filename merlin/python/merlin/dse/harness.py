"""Evaluation harness over (workload x feature): build variants, cost them, emit dse_result.

Each cell is independent: a Region Pressure Vector + a feature -> the 4 variant plans -> the
analytical cost model -> a ``dse_result`` artifact. M1 runs cells in a single process; the
parallel fan-out (process pool / Workflow) is an M2 ergonomics concern. Reuses the
``dse_result`` schema so all approaches stay directly comparable.
"""
from __future__ import annotations

from pathlib import Path

from merlin.common import schemas
from merlin.common.artifacts import yaml_artifact
from merlin.dse.cost_model import evaluate_cost
from merlin.dse.hardware_space import default_cost_model
from merlin.dse.variants import VARIANTS, build_variants


def evaluate_feature(workload: str, rpv: dict, feature: str,
                     cost_model: dict | None = None) -> dict:
    """Build the 4 variants for ``feature`` and return a schema-shaped ``dse_result``."""
    cm = cost_model or default_cost_model()
    plans = build_variants(rpv, feature)
    results = {}
    for name in VARIANTS:
        cost = evaluate_cost(rpv, plans[name], cm)
        results[name] = {"cycles": cost["cycles"], "energy": cost["energy"]}
    result = {
        "workload": workload,
        "feature": feature,
        "variants": list(VARIANTS),
        "cost_model": dict(cm),
        "results": results,
    }
    schemas.validate_or_raise(result, "dse_result")
    return result


def run_matrix(cells: list[tuple[str, dict]], features: list[str],
               cost_model: dict | None = None,
               out_base: str | Path | None = None) -> list[dict]:
    """Evaluate every (cell x feature). ``cells`` is a list of ``(workload_name, rpv)``.

    Writes ``<out_base>/<workload>/<feature>/dse_result.yaml`` when ``out_base`` is given.
    """
    cm = cost_model or default_cost_model()
    out: list[dict] = []
    for workload, rpv in cells:
        for feature in features:
            res = evaluate_feature(workload, rpv, feature, cm)
            out.append(res)
            if out_base is not None:
                yaml_artifact(
                    f"{workload}/{feature}/dse_result.yaml", res,
                    header=f"dse_result: {workload} / {feature}",
                ).write(out_base)
    return out
