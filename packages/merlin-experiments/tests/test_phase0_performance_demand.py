"""Performance cases derive from captured programs' contraction inventories, by declared policy."""

from __future__ import annotations

import json

import pytest
import yaml

from merlin_experiments.phase0 import case_redundancy as R
from merlin_experiments.phase0 import performance_demand as D

FACTS = {"facts": {"arrays": [{"name": "mesh", "rows": 4, "cols": 4}]}}


def policy(**override):
    base = {
        "schema": D.POLICY_SCHEMA,
        "coverage_target": 0.9,
        "min_scored_share": 0.01,
        "class_edges": {"M": [1, 4], "K": [8], "N": [8]},
        "holdout": "one_per_class",
        "admissibility": {"logical_storage_budget_bytes": None, "max_case_macs": None},
        "name_prefix": "PD",
    }
    return {**base, **override}


def projection(shapes, *, sha="a" * 64, model="net", program="main"):
    """A value-free prepared-graph projection holding ``count`` int8 matmuls of each shape."""
    nodes, edges = [], []
    for index, (m, k, n, count) in enumerate(shapes):
        for j in range(count):
            node = f"n{index}_{j}"
            nodes.append(
                {"id": node, "target": "aten._int_mm.default", "results": [{"dtype": "int32", "shape": [m, n]}]}
            )
            edges.append({"consumer_node_id": node, "argument_path": "args/0", "dtype": "int8", "shape": [m, k]})
            edges.append({"consumer_node_id": node, "argument_path": "args/1", "dtype": "int8", "shape": [k, n]})
    return {
        "model": model,
        "program": program,
        "source_sha256": sha,
        "graphs": [{"stage": "prepared", "graph_sha256": "b" * 64, "nodes": nodes, "edges": edges}],
    }


def program(shapes, **kwargs):
    record = projection(shapes, **kwargs)
    return {"model": record["model"], "program": record["program"], "projection": record}


# Shares: 64x64x64 (50%), 64x64x32 (25%), 64x32x32 (12.5%), 64x16x32 (6.25%), 1x64x64 (tiny).
SHAPES = [(64, 64, 64, 2), (64, 64, 32, 2), (64, 32, 32, 2), (64, 16, 32, 2), (1, 64, 64, 1)]


def test_classes_come_from_the_facts_tile_and_declared_edges():
    edges = {"M": [1, 4], "K": [4], "N": [4]}
    assert D.tile_edge(FACTS) == 4
    assert D.regime_class((1, 8, 64), tile=4, edges=edges) == "M=1/K<=4t/N>4t"
    assert D.regime_class((3, 16, 17), tile=4, edges=edges) == "M<=1t/K<=4t/N>4t"
    with pytest.raises(D.PerformanceDemandError):
        D.tile_edge({"facts": {"arrays": [{"rows": 4, "cols": 8}]}})


def test_selection_ranks_by_mac_share_until_the_target_and_holds_out_one_shape_per_class():
    plan = D.derive([program(SHAPES)], policy=policy(), facts=FACTS)
    public = [m["shape"] for m in plan["members"] if m["split"] == "public"]
    hidden = [m["shape"] for m in plan["members"] if m["split"] == "hidden"]
    # 64x32x32 and 64x16x32 share the class M>4t/K<=8t/N<=8t: its smallest is held out.
    assert hidden == [[64, 16, 32]]
    assert public[:2] == [[64, 64, 64], [64, 64, 32]] and [64, 32, 32] in public
    coverage = plan["coverage"]["net/main"]
    assert coverage["status"] == "target_met" and coverage["exact_public_share"] >= 0.9
    assert coverage["hidden_share"] == pytest.approx(64 * 16 * 32 * 2 / sum(m * k * n * c for m, k, n, c in SHAPES))
    member = next(m for m in plan["members"] if m["shape"] == [64, 64, 64])
    assert member["represents"][0]["model"] == "net" and member["represents"][0]["class"] == "M>4t/K>8t/N>8t"
    assert member["name"] == "PD_net_main_m64k64n64_raw"
    assert len(plan["sha256"]) == 64
    assert D.derive([program(SHAPES)], policy=policy(), facts=FACTS) == plan  # deterministic


def test_an_inadmissible_layer_gets_its_nearest_representative_with_the_same_k_and_n():
    bounds = {"logical_storage_budget_bytes": None, "max_case_macs": 60 * 64 * 64}
    plan = D.derive([program([(66, 64, 64, 1)])], policy=policy(admissibility=bounds), facts=FACTS)
    (member,) = plan["members"]
    # Positions shrink in whole tiles and keep the model's residue modulo the tile (66 mod 4 = 2).
    assert member["shape"] == [58, 64, 64] and member["admission"] == "representative"
    assert member["exceeded_bound"] == "max_case_macs" and member["source_shape"] == [66, 64, 64]
    assert plan["coverage"]["net/main"]["represented_public_share"] == 1.0


def test_existing_cases_are_reused_and_never_held_out():
    existing = {"OLD_case": [64, 16, 32]}
    plan = D.derive([program(SHAPES)], policy=policy(), facts=FACTS, existing=existing)
    reused = next(m for m in plan["members"] if m["satisfied_by_existing"])
    assert reused["name"] == "OLD_case" and reused["split"] == "public"
    assert [m["shape"] for m in plan["members"] if m["split"] == "hidden"] == [[64, 32, 32]]


def test_a_shape_two_programs_need_is_one_member_serving_both():
    first = program([(64, 64, 64, 1), (8, 8, 8, 1)], model="a", program="x")
    second = program([(64, 64, 64, 1)], model="b", program="y", sha="c" * 64)
    plan = D.derive([first, second], policy=policy(holdout="none"), facts=FACTS)
    shared = [m for m in plan["members"] if m["shape"] == [64, 64, 64]]
    assert len(shared) == 1 and {r["model"] for r in shared[0]["represents"]} == {"a", "b"}


def test_policy_declares_every_knob():
    with pytest.raises(D.PerformanceDemandError):
        D.load_policy({k: v for k, v in policy().items() if k != "coverage_target"})
    with pytest.raises(D.PerformanceDemandError):
        D.load_policy(policy(class_edges={"M": [4, 1], "K": [4], "N": [4]}))


IFACE = """module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<4x4xi8>
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<4x4xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<4x4xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<4x4xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<4x4xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
"""


def template(tmp_path):
    root = tmp_path / "template"
    root.mkdir(parents=True)
    cap = {
        "name": "T",
        "kind": "layer",
        "source_role": "handauthored_compiler_test",
        "label": "public",
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": "W", "role": "weight", "shape": [4, 4], "dtype": "i8"},
            {"name": "A0", "role": "input", "shape": [4, 4], "dtype": "i8"},
        ],
        "operation": {
            "op": "matmul",
            "attributes": {"lhs": "A0", "weight": "W", "out": "Y0", "epilogue": [], "output_dtype": "i32"},
        },
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "expected": {"modes": {"relu": False, "acc_scale": False, "i8": False}},
        "required_oracle_tiers": ["L0", "L1", "L2"],
        "stimulus_range": [-4, 3],
    }
    (root / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False))
    (root / "capsule.interface.mlir").write_text(IFACE)
    return root


def test_sources_and_hidden_profile_carry_the_represented_class(tmp_path):
    plan = D.derive([program(SHAPES)], policy=policy(), facts=FACTS)
    written = D.write_public_sources(plan, template=template(tmp_path), destination=tmp_path / "sources")
    assert len(written) == sum(m["split"] == "public" for m in plan["members"])
    cap = yaml.safe_load((written[0] / "capsule.yaml").read_text())
    assert cap["performance_scope"] == "scored" and cap["label"] == "public"
    assert cap[D.BLOCK]["represents"][0]["class"] and "net/main" in cap["source_reference"]
    assert {i["name"]: i["shape"] for i in cap["inputs"]} == {"W": [64, 64], "A0": [64, 64]}
    assert "tensor<64x64xi8>" in (written[0] / "capsule.interface.mlir").read_text()
    binding = json.loads((written[0] / "source-shape-binding.json").read_text())
    assert binding["plan_sha256"] == plan["sha256"] and binding["full_model"] is False
    profile = D.hidden_profile(plan, template=template(tmp_path / "again"))
    (entry,) = profile["capsules"]
    assert (entry["cat"], entry["label"], entry["M"], entry["K"], entry["N"]) == ("hidden", "hidden", 64, 16, 32)
    assert entry["stimulus_range"] == [-4, 3]
    assert entry["source_role"] == "model_derived"
    # A holdout is a generated hidden member: no cap below the target's tiers, so a certification
    # floor holds it to direct certification like every other generated member.
    assert "max_oracle_tier" not in entry
    from merlin_experiments.phase0.certification_floor import require_direct_tier

    floor = require_direct_tier([{"name": entry["name"], **entry}], floor="L3", declared_tiers=["L0", "L1", "L2", "L3"])
    assert floor["below_floor"] == []
    # A derived source is its own scorer once redundancy runs against an older raw copy of the same shape.
    older = dict(yaml.safe_load((written[0] / "capsule.yaml").read_text()), name="OLDER")
    rows = [
        {"name": cap["name"], "cohort": "scalability", "capsule": cap},
        {"name": "OLDER", "cohort": "scalability", "capsule": {**older, "stimulus_range": [-8, 7]}},
    ]
    record = R.derive(rows, demand=plan)
    assert record["performance_scope"] == {cap["name"]: "scored", "OLDER": "excluded"}
