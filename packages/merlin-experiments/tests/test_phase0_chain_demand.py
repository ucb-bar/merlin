"""Connected chain cases derive from captured graphs' structure and are written as one program each."""

from __future__ import annotations

import pytest
from merlin_experiments.phase0 import chain_demand as D
from merlin_experiments.phase0 import chain_graph as G

FACTS = {"facts": {"arrays": [{"name": "mesh", "rows": 4, "cols": 4}]}}


class Trace:
    """A minimal frontend trace: parameters, an input and call_function nodes with tensor edges."""

    def __init__(self):
        self.nodes, self.edges, self.count = [], [], 0
        self.input = self._placeholder("USER_INPUT", [1, 3, 16, 16])

    def _placeholder(self, kind, shape):
        node = f"g:original:root:n{self.count}"
        self.count += 1
        self.nodes.append({"id": node, "op": "placeholder", "input_kind": kind, "target": "x",
                           "results": [{"kind": "tensor", "dtype": "float32", "shape": list(shape)}]})
        return node, list(shape)

    def param(self, shape):
        return self._placeholder("PARAMETER", shape)

    def call(self, target, operands, shape, args=()):
        node = f"g:original:root:n{self.count}"
        self.count += 1
        refs = []
        for index, (producer, pshape) in enumerate(operands):
            self.edges.append({"consumer_node_id": node, "producer_node_id": producer, "argument_path": f"args/{index}",
                               "shape": list(pshape), "dtype": "float32", "value_kind": "tensor"})
            refs.append({"node_id": producer})
        self.nodes.append({"id": node, "op": "call_function", "target": target, "ordinal": self.count,
                           "args": [*refs, *args],
                           "results": [{"kind": "tensor", "dtype": "float32", "shape": list(shape)}]})
        return node, list(shape)

    def graph(self):
        return G.read({"graphs": {"original": {"sha256": "c" * 64, "nodes": self.nodes, "edges": self.edges}}})


def stem_trace():
    """conv 7x7/2 -> batch norm -> relu -> max pool 3x3/2, then a bottleneck-style 1x1 conv + add."""
    t = Trace()
    conv = t.call("aten.conv2d.default", [t.input, t.param([8, 3, 7, 7])], [1, 8, 8, 8], [None, [2, 2], [3, 3]])
    norm = t.call("aten.batch_norm.default", [conv, t.param([8]), t.param([8])], [1, 8, 8, 8])
    relu = t.call("aten.relu_.default", [norm], [1, 8, 8, 8])
    pool = t.call("aten.max_pool2d.default", [relu], [1, 8, 4, 4], [[3, 3], [2, 2], [1, 1]])
    point = t.call("aten.conv2d.default", [pool, t.param([8, 8, 1, 1])], [1, 8, 4, 4])
    folded = t.call("aten.batch_norm.default", [point, t.param([8]), t.param([8])], [1, 8, 4, 4])
    add = t.call("aten.add_.Tensor", [folded, pool], [1, 8, 4, 4])
    t.call("aten.relu_.default", [add], [1, 8, 4, 4])
    return t


def decoder_trace():
    """Q/K/V siblings, grouped-query attention over expanded keys/values, and a gated feed-forward."""
    t = Trace()
    x = t.call("aten.mul.Tensor", [t.input, t.param([1, 3, 16, 16])], [1, 3, 16, 16])
    x = t.call("aten.reshape.default", [x], [1, 12, 64], [[1, 12, 64]])
    q = t.call("aten.linear.default", [x, t.param([64, 64])], [1, 12, 64])
    k = t.call("aten.linear.default", [x, t.param([16, 64])], [1, 12, 16])
    v = t.call("aten.linear.default", [x, t.param([16, 64])], [1, 12, 16])
    q4 = t.call("aten.view.default", [q], [1, 4, 12, 16], [[1, 4, 12, 16]])
    heads = []
    for value in (k, v):
        base = t.call("aten.view.default", [value], [1, 1, 12, 16], [[1, 1, 12, 16]])
        wide = t.call("aten.expand.default", [base], [1, 4, 12, 16], [[1, 4, 12, 16]])
        heads.append(wide)
    attn = t.call("aten.scaled_dot_product_attention.default", [q4, *heads], [1, 4, 12, 16])
    flat = t.call("aten.reshape.default", [attn], [1, 12, 64], [[1, 12, 64]])
    out = t.call("aten.linear.default", [flat, t.param([64, 64])], [1, 12, 64])
    res = t.call("aten.add.Tensor", [x, out], [1, 12, 64])
    gate = t.call("aten.linear.default", [res, t.param([96, 64])], [1, 12, 96])
    up = t.call("aten.linear.default", [res, t.param([96, 64])], [1, 12, 96])
    act = t.call("aten.silu.default", [gate], [1, 12, 96])
    prod = t.call("aten.mul.Tensor", [act, up], [1, 12, 96])
    t.call("aten.linear.default", [prod, t.param([64, 96])], [1, 12, 64])
    return t


def test_stem_and_bottleneck_chains_are_found_by_structure():
    rows = {r["family"]: r for r in G.inventory(stem_trace().graph())}
    pool = rows["conv_requant_pool"]["instance"]
    assert pool["affine"] and pool["relu"]
    assert pool["producer"]["kind"] == "convolution" and pool["producer"]["stride"] == [2, 2]
    assert pool["pool"] == {"size": [3, 3], "stride": [2, 2], "padding": [1, 1], "out": [4, 4]}
    residual = rows["contraction_residual"]["instance"]
    assert residual["producer"]["kind"] == "pointwise" and residual["relu_after_add"]
    assert (residual["producer"]["M"], residual["producer"]["K"], residual["producer"]["N"]) == (16, 8, 8)


def test_decoder_siblings_feed_and_shared_kv_heads_are_found():
    rows = G.inventory(decoder_trace().graph())
    found = {r["family"]: r["instance"] for r in rows}
    assert found["shared_input_siblings"]["N"] == [64, 16, 16]
    feed = found["sibling_feed"]
    assert (feed["M"], feed["K"], feed["H"], feed["N"], feed["gated"]) == (12, 64, 96, 64, True)
    heads = found["shared_kv_heads"]
    assert (heads["query_heads"], heads["kv_heads"], heads["S"], heads["L"], heads["d"]) == (4, 1, 12, 12, 16)
    assert found["contraction_residual"]["producer"]["kind"] == "linear"


def test_an_ungrouped_attention_is_not_a_shared_kv_chain():
    t = Trace()
    x = t.call("aten.reshape.default", [t.input], [1, 4, 12, 16], [[1, 4, 12, 16]])
    t.call("aten.scaled_dot_product_attention.default", [x, x, x], [1, 4, 12, 16])
    assert not [r for r in G.inventory(t.graph()) if r["family"] == "shared_kv_heads"]


def policy(**override):
    base = {
        "schema": D.POLICY_SCHEMA,
        "families": list(G.FAMILIES),
        "performance": {"members_per_group": 2, "minimum_input_elements": 2000, "max_program_nodes": 64,
                        "admissibility": {"logical_storage_budget_bytes": 1 << 24, "max_case_macs": 1 << 30}},
        "correctness": {"reduced_tiles": 2, "edge_tail": 1},
        "stimulus_range": [-4, 3],
        "readout_rms": 24,
        "name_prefix": "CH",
    }
    return {**base, **override}


def rows(claim):
    return [{"model": "net", "program": "main", "graph": decoder_trace().graph(), "trace_sha256": "a" * 64,
             "claim": claim}]


def test_claim_programs_give_performance_shapes_only():
    plan = D.derive(rows(True), policy=policy(), facts=FACTS)
    roles = {m["role"] for m in plan["members"]}
    assert roles == {"performance"}
    assert any(d.get("reason", "").startswith("no public application") for d in plan["deferred"])


def test_public_programs_give_correctness_variants_on_the_tile_scale():
    plan = D.derive(rows(False), policy=policy(), facts=FACTS)
    by_name = {m["name"]: m for m in plan["members"]}
    assert "CH_sibling_feed_net_main_m12k64h96n64" in by_name
    reduced = by_name["CH_sibling_feed_reduced_m8k8h8n8"]
    edge = by_name["CH_sibling_feed_edge_m5k5h5n5"]
    assert reduced["cohort"] == edge["cohort"] == "generated"
    assert by_name["CH_sibling_feed_net_main_m12k64h96n64"]["cohort"] == "scalability"
    assert plan["sha256"] == D.derive(rows(False), policy=policy(), facts=FACTS)["sha256"]


def test_member_entries_declare_their_chain_and_cohort():
    plan = D.derive(rows(False), policy=policy(), facts=FACTS)
    entries = {m["name"]: D.member_entry(m, plan["sha256"], plan["policy"], narrow_bits=8) for m in plan["members"]}
    feed = entries["CH_sibling_feed_net_main_m12k64h96n64"]
    assert feed["op"] == "sibling_feed" and feed["cat"] == "sources" and feed["performance_scope"] == "scored"
    assert feed[D.BLOCK]["chain"][-1] == "matmul" and feed[D.BLOCK]["plan_sha256"] == plan["sha256"]
    small = entries["CH_sibling_feed_reduced_m8k8h8n8"]
    assert small["cat"] == "model_slices" and "performance_scope" not in small
    assert all(e["source_role"] == "model_derived" and e["label"] == "public" for e in entries.values())


def test_readout_scale_is_a_power_of_two_and_falls_with_depth():
    shallow, deep = D.readout_scale(16, policy()), D.readout_scale(4096, policy())
    assert shallow > deep
    for value in (shallow, deep):
        assert value == 2.0 ** round(__import__("math").log2(value))


def test_policy_declares_every_knob():
    with pytest.raises(D.ChainDemandError):
        D.load_policy({k: v for k, v in policy().items() if k != "readout_rms"})
    with pytest.raises(D.ChainDemandError):
        D.load_policy(policy(families=["unknown_family"]))


def test_pool_variants_keep_the_window_geometry():
    row = {r["family"]: r for r in G.inventory(stem_trace().graph())}["conv_requant_pool"]
    shapes = dict(D.variants(row, tile=4, policy=policy()))
    assert shapes["reduced"]["producer"]["out_height"] == 8
    assert (shapes["edge"]["producer"]["out_height"], shapes["edge"]["producer"]["out_width"]) == (7, 5)
    assert shapes["edge"]["producer"]["cout"] == 5
    window = shapes["window"]["producer"]
    assert (window["out_height"], window["height"]) == (3, 5)


def test_chain_profile_refuses_entries_it_did_not_derive(tmp_path):
    import yaml
    from merlin_experiments.phase0.profiles import chain_profile_entries

    plan = D.derive(rows(False), policy=policy(), facts=FACTS)
    good = [D.member_entry(m, plan["sha256"], plan["policy"], narrow_bits=8)
            for m in plan["members"] if m["cohort"] == "generated"]
    path = tmp_path / "chain-profile.yaml"
    path.write_text(yaml.safe_dump({"capsules": good}))
    assert [e["name"] for e in chain_profile_entries(path)] == [e["name"] for e in good]
    for bad in ({"cat": "hidden"}, {"source_role": "derived_sweep"}, {D.BLOCK: {}}):
        path.write_text(yaml.safe_dump({"capsules": [{**good[0], **bad}]}))
        with pytest.raises(ValueError, match="public model-derived"):
            chain_profile_entries(path)


def test_epoch_takes_a_chain_plan_with_its_sources(tmp_path):
    from merlin_experiments.phase0.performance_corpus import assemble_epoch

    with pytest.raises(ValueError, match="both a chain plan"):
        assemble_epoch(tmp_path, plan=tmp_path / "plan.json", sources=tmp_path, output=tmp_path / "out",
                       evidence_root=tmp_path, target="t", chain_plan=tmp_path / "chains.json")


def test_a_tensor_program_entry_is_screened_by_its_written_program():
    from merlin_experiments.phase0.software_screen import screen_entry

    plan = D.derive(rows(False), policy=policy(), facts=FACTS)
    entry = next(D.member_entry(m, plan["sha256"], plan["policy"], narrow_bits=8)
                 for m in plan["members"] if m["family"] == "shared_input_siblings")
    decision = screen_entry({"schema": "merlin.software_spec.v1", "operations": {}}, entry)
    assert decision["status"] == "unknown" and "per-operation" in decision["reason"]
