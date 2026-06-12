"""Promotion gate + benchmark validation of emitted policies."""
import os

import yaml

from merlin.kernels import policy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BENCH = os.path.join(ROOT, "merlin", "benchmarks", "semantic_memory")


def _stat(sources, kernels):
    s = policy.MotifStat(kernel_count=kernels)
    s.sources = set(sources)
    s.evidence_ids = {f"{x}_t_op" for x in sources}
    return s


def test_promotion_gate():
    assert policy.is_promotable(_stat(["a", "b"], 2))      # >=2 sources
    assert policy.is_promotable(_stat(["a"], 10))          # >=10 kernels
    assert not policy.is_promotable(_stat(["a"], 3))       # single source, too few


def test_promote_emits_expected_rules():
    stats = {
        "packed_rhs": _stat(["xnnpack", "autocomp"], 900),
        "accumulator_commit": _stat(["xnnpack", "autocomp"], 400),
        "vector_length_polymorphic": _stat(["xnnpack"], 545),
        "weight_stationary_dataflow": _stat(["autocomp"], 800),
    }
    res = policy.promote(stats, min_kernels=10)
    names = {r["policy"] for r in res.rules}
    assert {"packed_rhs_policy", "accumulator_commit_policy", "vl_agnostic_loop_policy",
            "weight_stationary_dataflow_policy"} <= names
    assert len(res.rules) >= 3
    cnames = {c["name"] for c in res.candidates}
    assert "resident_packed_tensor" in cnames and "accumulator_commit" in cnames


def _bench(name):
    with open(os.path.join(BENCH, f"{name}.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_packed_rhs_policy_fires_on_positive_and_silent_on_negative():
    rule = next(r for r in policy.promote(
        {"packed_rhs": _stat(["xnnpack", "autocomp"], 900)}).rules
        if r["policy"] == "packed_rhs_policy")
    pos = _bench("repeated_rhs_matmul")["reuse"]   # rhs_reuse_count: 8, rhs_mutable: false
    neg = _bench("no_reuse_matmul")["reuse"]       # rhs_reuse_count: 1
    assert policy.evaluate_when(rule["when"], pos) is True
    assert policy.evaluate_when(rule["when"], neg) is False


def test_accumulator_commit_fires_on_epilogue_workload():
    rule = next(r for r in policy.promote(
        {"accumulator_commit": _stat(["xnnpack", "autocomp"], 400)}).rules
        if r["policy"] == "accumulator_commit_policy")
    bench = _bench("matmul_bias_requant_relu")
    facts = {"op": bench["ops"][0], "has_epilogue": len(bench["op_sequence"]) > 1,
             "accumulator_live_across_epilogue": True}
    assert policy.evaluate_when(rule["when"], facts) is True
