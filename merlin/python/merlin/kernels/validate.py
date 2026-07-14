"""Stage-D held-out validation: do policies hold on benchmark workloads and shape regimes?

A motif that cleared the promotion gate is only a *candidate*. This module exercises each
policy against the semantic-memory benchmarks — including the negative control and the
``capacity_stress_reuse`` ``resident_store_bytes`` sweep — and reports, per workload, whether
the policy **holds**, **fails** (incl. correctly silent on a negative control), or is
**n/a** (workload lacks the facts to decide). No kernel is executed; this is symbolic
generalization testing, the structural precursor to measured speedups in a later session.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from merlin.kernels.policy import _coerce

_DTYPE_BYTES = {"i8": 1, "u8": 1, "i32": 4, "f16": 2, "bf16": 2, "f32": 4}


def _bench_dir() -> Path:
    # MERLIN_BENCH_DIR is the benchmarks ROOT (unified with the dse_guidance sites); the workload
    # specs live under semantic_memory/. bench_dir() resolves in-repo canonical or bundled _data.
    from merlin.common.paths import bench_dir
    return bench_dir() / "semantic_memory"


def _load(name: str) -> dict:
    with (_bench_dir() / f"{name}.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def benchmark_facts(bench: dict) -> dict:
    """Flatten a benchmark workload into the symbolic facts a policy `when` references."""
    facts: dict = {}
    reuse = bench.get("reuse", {}) or {}
    facts.update(reuse)
    if "rhs_mutable" not in facts:
        facts["rhs_mutable"] = False  # benchmark weights are immutable unless stated
    ops = bench.get("ops") or []
    if ops:
        facts["op"] = ops[0]
    seq = bench.get("op_sequence")
    if seq:
        facts["has_epilogue"] = len(seq) > 1
        facts["accumulator_live_across_epilogue"] = True
    # K from the weight tensor (W shape is [K, N])
    W = (bench.get("tensors") or {}).get("W") or (bench.get("tensors") or {}).get("W_j")
    if W and isinstance(W.get("shape"), list) and len(W["shape"]) == 2:
        facts["K"] = W["shape"][0]
    return facts


def apply_status(when: dict, facts: dict) -> str:
    """Return 'holds' | 'fails' | 'n/a' for a `when` clause against facts.

    Stricter than ``policy.evaluate_when``: 'holds' requires *every* condition present and
    passing; a present-but-failing condition is 'fails'; missing facts make it 'n/a'.
    """
    present = [k for k in when if k in facts]
    if not present:
        return "n/a"
    for key in present:
        cond_s = str(when[key]).strip()
        fact = _coerce(facts[key])
        ok = True
        for op in (">=", "<=", ">", "<", "=="):
            if cond_s.startswith(op):
                try:
                    rhs = _coerce(cond_s[len(op):].strip())
                    ok = {">=": fact >= rhs, "<=": fact <= rhs, ">": fact > rhs,
                          "<": fact < rhs, "==": fact == rhs}[op]
                except TypeError:
                    ok = False
                break
        else:
            ok = (str(fact) in cond_s.split("|")) if "|" in cond_s else (fact == _coerce(cond_s))
        if not ok:
            return "fails"
    return "holds" if len(present) == len(when) else "n/a"


def capacity_sweep(bench: dict, dtype: str = "i8") -> list[dict]:
    """For a capacity-stress workload, report whether residency fits each store budget."""
    reuse = bench.get("reuse", {}) or {}
    W = (bench.get("tensors") or {}).get("W_j") or (bench.get("tensors") or {}).get("W") or {}
    shape = W.get("shape", [])
    if len(shape) != 2:
        return []
    rhs_bytes = shape[0] * shape[1] * _DTYPE_BYTES.get(dtype, 1)
    distinct = reuse.get("distinct_weights", 1)
    footprint = rhs_bytes * distinct
    out = []
    for budget in (bench.get("parameters", {}) or {}).get("resident_store_bytes", []):
        out.append({"resident_store_bytes": budget, "footprint_bytes": footprint,
                    "fits": footprint <= budget})
    return out


# Which benchmark workloads each policy should be checked against, and which is the control.
_POLICY_WORKLOADS = {
    "packed_rhs_policy": ["repeated_rhs_matmul", "no_reuse_matmul", "capacity_stress_reuse"],
    "accumulator_commit_policy": ["matmul_bias_requant_relu", "no_reuse_matmul"],
}
_NEGATIVE_CONTROL = "no_reuse_matmul"

# Symbolic shape-regime sweep (spec §4/§6): a policy must be checked across reuse counts,
# K regimes and tail-heaviness — and stay silent on the negative controls — before it can
# be called shape-general. Facts a policy's `when` never references are skipped by
# apply_status, so only shape-relevant policies get a matrix.
_SHAPE_FACT_KEYS = {"rhs_reuse_count", "K", "k_tail_heavy", "rhs_size_bytes"}
_SWEEP_BASE = {  # positive-template facts so non-shape conditions are satisfiable
    "packed_rhs_policy": {"rhs_mutable": False},
    "accumulator_commit_policy": {"op": "matmul", "has_epilogue": True,
                                  "accumulator_live_across_epilogue": True},
}
_SWEEP_REUSE = (1, 2, 4, 8, 16)
_SWEEP_K = (64, 1024)


def shape_sweep(rule: dict) -> dict | None:
    """Evaluate ``rule`` over the symbolic regime grid; None if shape-independent."""
    when = rule["when"]
    if not set(when) & _SHAPE_FACT_KEYS:
        return None
    base = _SWEEP_BASE.get(rule["policy"], {})
    cells = []
    for r in _SWEEP_REUSE:
        for k in _SWEEP_K:
            for tail in (False, True):
                kk = k + 8 if tail else k  # tail-heavy K is genuinely non-divisible
                facts = {**base, "rhs_reuse_count": r, "K": kk, "k_tail_heavy": tail}
                cells.append({"reuse": r, "K": kk, "tail_heavy": tail,
                              "status": apply_status(when, facts)})
    neg = {
        "mutable_rhs": apply_status(
            when, {**base, "rhs_mutable": True, "rhs_reuse_count": 8,
                   "K": 1024, "k_tail_heavy": False}),
        "no_reuse": apply_status(
            when, {**base, "rhs_reuse_count": 1, "K": 1024, "k_tail_heavy": False}),
    }
    neg = {name: ("correctly_silent" if st == "fails" else f"LEAK({st})")
           for name, st in neg.items()}
    return {"cells": cells,
            "fires": sum(1 for c in cells if c["status"] == "holds"),
            "negative_controls": neg}


def validate_policies(rules: list[dict]) -> dict[str, dict]:
    """Per-policy benchmark verdicts + capacity sweep + the shape-regime matrix."""
    report: dict[str, dict] = {}
    for rule in rules:
        name = rule["policy"]
        entry: dict = {"workloads": {}}
        for wl in _POLICY_WORKLOADS.get(name, []):
            bench = _load(wl)
            status = apply_status(rule["when"], benchmark_facts(bench))
            # A 'fails' on the negative control is the *desired* outcome.
            if wl == _NEGATIVE_CONTROL:
                status = "correctly_silent" if status == "fails" else f"LEAK({status})"
            entry["workloads"][wl] = status
            if wl == "capacity_stress_reuse":
                entry["capacity_sweep"] = capacity_sweep(bench)
        sweep = shape_sweep(rule)
        entry["regime_matrix"] = sweep if sweep else "shape_independent"
        report[name] = entry
    return report
