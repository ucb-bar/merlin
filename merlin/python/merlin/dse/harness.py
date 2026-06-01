"""Parallel evaluation harness over the (workload x strategy) matrix.

Each cell is independent: same input IR -> the strategy's xDSL pipeline -> simulator/cost
model -> a `dse_result` artifact under output/dse/<exp>/<strategy_id>/. Cells run in parallel
(process pool, or the orchestration Workflow tool). Reuses the existing dse_result /
exploitability_report schemas so all approaches are directly comparable.

Placeholder module. No real logic yet.
"""
from __future__ import annotations


def run_matrix(*args, **kwargs):
    """TODO: fan out (workload x strategy), evaluate each, collect dse_result artifacts."""
    raise NotImplementedError("run_matrix is a scaffold stub; not implemented yet.")
