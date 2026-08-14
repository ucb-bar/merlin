"""Foundations for Target-Conditioned Semantic Coverage (Acceleratable Region Recall):
the canonical family vocabulary, the ``semantic_capabilities`` contract block, and the independent
eligibility oracle (the ARR denominator)."""
from __future__ import annotations

from merlin.common.paths import repo_root
from merlin.targetgen import compute_units as cu
from merlin.targetgen import eligibility as el
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
