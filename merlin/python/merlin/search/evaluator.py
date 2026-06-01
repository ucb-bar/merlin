"""Shared scoring for all three search methods.

Scoring rubric (early-work priority order: correctness first, speed last):

    score =  correctness
           + compile_success
           + verifier_success
           + workload_coverage
           + compiler_exploitability
           + speedup_or_cost_improvement
           - complexity_penalty

    priority: correctness > compile_success > coverage > exploitability > speedup

Delegates compilation+measurement to merlin.dse.harness. Returns a Score record. Prioritizing
correctness/compile/coverage prevents search from finding fast-but-invalid junk.

Placeholder module. No real logic yet.
"""
from __future__ import annotations


def evaluate(*args, **kwargs):
    """TODO: compile + verify + simulate a candidate; return a Score per the rubric above."""
    raise NotImplementedError("evaluate is a scaffold stub; not implemented yet.")
