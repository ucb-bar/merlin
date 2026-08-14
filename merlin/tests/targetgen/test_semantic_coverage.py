"""Foundations for Target-Conditioned Semantic Coverage (Acceleratable Region Recall):
the canonical family vocabulary, the ``semantic_capabilities`` contract block, and the independent
eligibility oracle (the ARR denominator)."""
from __future__ import annotations

from merlin.common.paths import repo_root
from merlin.targetgen import compute_units as cu
from merlin.targetgen import coverage_certificate as cert
from merlin.targetgen import coverage_report as cov
from merlin.targetgen import eligibility as el
from merlin.targetgen import routing as rt
from merlin.targetgen import semantic_families as sf


# --- A1: canonical semantic-family vocabulary ---------------------------------------------------

def test_family_vocabulary_is_consistent():
    assert sf.check() == []
    assert set(sf.PRIMITIVES) <= sf.FAMILIES
    # composites decompose only to primitives
    for comp, parts in sf.COMPOSITES.items():
        assert all(p in sf.PRIMITIVES for p in parts)


def test_family_mapping_from_prov_and_op():
    assert sf.from_prov("contraction") == "contraction"
    assert sf.from_prov("normalization") == "normalization"
    assert sf.from_prov("elementwise") == "elementwise_map"
    assert sf.from_op("matmul") == "contraction"
    assert sf.from_op("softmax") == "softmax"
    assert sf.from_op("transpose") == "movement"
    # fail-closed on the unknown
    assert sf.from_prov(None) is None
    assert sf.from_op("gather_scatter") is None
    assert sf.primitives_of("attention") == ("contraction", "reduction", "elementwise_map")
    assert sf.primitives_of("contraction") == ("contraction",)


# --- A2: semantic_capabilities in the capability contract ---------------------------------------

def _contract() -> dict:
    return {"compute_units": [
        {"name": "simt", "kind": "simt", "dtypes": ["fp16", "fp32"],
         "ops": ["matmul", "elementwise"], "contains": ["mx"],
         "semantic_capabilities": [
             {"family": "contraction", "dtypes": ["fp16", "fp32"], "ranks": [2, 3]},
             {"family": "reduction", "dtypes": ["fp16", "fp32"]},
             {"family": "elementwise_map", "dtypes": ["fp16", "fp32"]},
             {"family": "movement", "dtypes": ["fp16", "fp32"]},
         ]},
        {"name": "mx", "kind": "systolic", "dtypes": ["mxfp8", "bf16"], "ops": ["matmul"],
         "semantic_capabilities": [{"family": "contraction", "dtypes": ["mxfp8", "bf16"]}]},
    ]}


def test_semantic_capabilities_parse_and_fold():
    units = cu.compute_units(_contract())
    # composition folds the mx PE's contraction dtypes up into the simt cluster
    eff = cu.effective(units[0], units)
    fams = {c.family for c in eff.semantic_capabilities}
    assert fams == {"contraction", "reduction", "elementwise_map", "movement"}
    cmap = cu.semantic_capability_map(units)
    # merged contraction is the union across both units
    assert set(cmap["contraction"].dtypes) == {"fp16", "fp32", "mxfp8", "bf16"}


def test_backward_compat_no_semantic_block():
    units = cu.compute_units({"compute_units": [{"name": "m", "kind": "systolic", "dtypes": ["int8"]}]})
    assert units[0].semantic_capabilities == ()
    assert cu.semantic_capability_map(units) == {}


# --- A3: independent eligibility oracle (the ARR denominator) ------------------------------------

def test_eligibility_truth_table():
    cm = el.capability_map_from_contract(_contract())
    D = el.RegionDescriptor

    assert el.is_eligible(D(op="matmul", in_dtype="bf16", rank=2), cm).eligible          # contraction
    assert el.is_eligible(D(op="softmax", in_dtype="fp16"), cm).eligible                 # composite via prims
    assert el.is_eligible(D(op="reduce", in_dtype="fp32"), cm).eligible

    # ineligible cases, each with a reason
    v = el.is_eligible(D(op="matmul", in_dtype="int8", rank=2), cm)
    assert not v.eligible and "int8" in v.reason                                         # dtype not supported
    v = el.is_eligible(D(op="matmul", in_dtype="bf16", rank=5), cm)
    assert not v.eligible and "rank" in v.reason
    v = el.is_eligible(D(op="gather", in_dtype="fp16"), cm)
    assert not v.eligible and v.family is None                                           # fail-closed


# --- B1: coverage certificate (numerator=routing decision, denominator=eligibility) ------------

def test_coverage_certificate_surfaces_false_fallback_and_ineligible():
    # Mesh routing binds only bf16/int8; the eligibility oracle declares contraction eligible for
    # bf16 AND fp16. So an fp16 matmul is eligible-but-not-accelerated (false_fallback = compiler gap),
    # and an int8 matmul routes to the mesh but the oracle rejects it (accelerated_ineligible).
    contract = {"compute_units": [
        {"name": "mesh", "kind": "systolic", "dtypes": ["bf16", "int8"], "ops": ["matmul"],
         "semantic_capabilities": [{"family": "contraction", "dtypes": ["bf16", "fp16"]}]},
    ]}
    units = cu.compute_units(contract)
    cap_map = el.capability_map_from_contract(contract)
    demands = [
        rt.OpDemand(op="matmul", in_fmt="bf16", weight_fmt="bf16", site="L0", m=16, k=16, n=16),  # accel+eligible
        rt.OpDemand(op="matmul", in_fmt="fp16", weight_fmt="fp16", site="L1", m=16, k=16, n=16),  # eligible, gapped
        rt.OpDemand(op="matmul", in_fmt="int8", weight_fmt="int8", site="L2", m=16, k=16, n=16),  # accel, ineligible
    ]
    plan = rt.route_plan_on(demands, units)
    c = cert.build(plan, cap_map, target="synthetic")

    assert c["n_regions"] == 3
    assert c["n_eligible"] == 2                       # bf16 + fp16 contraction
    assert c["n_accelerated"] == 2                    # bf16 + int8 routed to mesh
    assert c["n_eligible_accelerated"] == 1           # only bf16 is both
    assert c["false_fallback_count"] == 1             # fp16: hw-capable but compiler left it off-mesh
    assert c["accelerated_ineligible_count"] == 1     # int8: accelerated but oracle rejects
    m = c["metrics"]
    assert m["acceleratable_region_recall"] == 0.5    # 1 of 2 eligible accelerated
    assert m["acceleration_precision"] == 0.5         # 1 of 2 accelerated were eligible


# --- B2: suite-level ARR in the coverage aggregate ----------------------------------------------

def test_suite_acceleratable_coverage():
    # Three capsules with author-eligibility overrides (target=None -> oracle empty, overrides drive it).
    # A2 accelerated (passed L2/spike); A7 eligible but only L1 passed = false_fallback; SORT ineligible.
    caps = [
        {"name": "A2", "operation": {"op": "matmul"}, "numeric_policy": {"dtype": "bf16"},
         "semantic": {"eligible": True, "must_accelerate": True}},
        {"name": "A7", "operation": {"op": "matmul"}, "numeric_policy": {"dtype": "bf16"},
         "semantic": {"eligible": True, "must_accelerate": True}},
        {"name": "SORT", "operation": {"op": "reduce"}, "numeric_policy": {"dtype": "fp32"},
         "semantic": {"eligible": False, "fallback_allowed": True}},
    ]
    results = [
        {"capsule": "A2", "kind": "isa", "label": "public", "status": "pass",
         "tiers": {"L2": {"status": "pass"}}},
        {"capsule": "A7", "kind": "isa", "label": "public", "status": "fail",
         "tiers": {"L1": {"status": "pass"}, "L2": {"status": "fail"}}},
        {"capsule": "SORT", "kind": "isa", "label": "public", "status": "pass",
         "tiers": {"L0": {"status": "pass"}}},
    ]
    cap_by_name = {c["name"]: c for c in caps}
    ac = cov._acceleratable_coverage(results, cap_by_name, target=None)
    assert ac["n_eligible"] == 2
    assert ac["n_eligible_accelerated"] == 1
    assert ac["false_fallback"] == ["A7"]
    assert ac["acceleratable_region_recall"] == 0.5
    # A7 is eligible + must_accelerate but did not accelerate -> a must_accelerate violation.
    assert ac["must_accelerate_violations"] == ["A7"]
    assert ac["must_accelerate_pass"] is False
    # full aggregate surfaces it too
    full = cov.aggregate(results, capsules=caps, target=None)
    assert full["acceleratable_coverage"]["false_fallback"] == ["A7"]


def test_eligibility_does_not_depend_on_routing():
    # The denominator must be independent of the compiler's routing/lowering. Walk the AST and assert
    # no import (module- or function-level) pulls in routing — so a future edit can't silently create
    # the circularity that would make ARR trivially 1.0. (The docstring may *mention* routing.)
    import ast

    src = (repo_root() / "merlin" / "python" / "merlin" / "targetgen" / "eligibility.py").read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported.add(base)
            imported.update(f"{base}.{a.name}" for a in node.names)
    assert not any("routing" in m for m in imported), \
        f"eligibility oracle must not import routing (ARR would be trivially 1.0); imports={sorted(imported)}"
