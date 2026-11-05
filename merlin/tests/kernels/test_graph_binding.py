"""Named bodies constrain hypotheses without granting source or numeric equality."""

from merlin.kernels.graph_binding import graph_binding_hypotheses


def _region(identity, path, target, producer=None, consumer=None):
    value = dict(id=identity + ":v0", shape=[2, 3], dtype="i8")
    return dict(
        region_id=identity,
        operation=dict(module_stack={"m": [path, "Module"]}, target=target),
        inputs=[] if producer is None else [dict(producer_node_id=producer, value=dict(id=producer + ":v0"))],
        outputs=[value],
        output_consumers={value["id"]: [] if consumer is None else [dict(node_id=consumer)]},
    )


def test_body_aliases_and_connected_fusion_remain_hypotheses():
    graph = dict(
        schema="merlin.capture_graph_regions.v1",
        regions=[
            _region("a", "model.block.0.projection", "aten.mm.default", consumer="b"),
            _region("b", "model.block.0.activation", "aten.relu.default", producer="a", consumer="output"),
            _region("c", "model.block.1.projection", "aten.mm.default", consumer="output"),
        ],
    )
    listing = dict(
        regions=[
            dict(
                region_id=1,
                body_match=dict(
                    basis="complete_listed_body_bytes",
                    contiguous_body=True,
                    matches=[
                        dict(symbol="vendor_block_0_projection_native"),
                        dict(symbol="vendor_block_1_projection_native"),
                    ],
                ),
            )
        ]
    )
    result = graph_binding_hypotheses(graph, listing)
    row = result["graph_bindings"][0]
    assert row["candidate_region_ids"] == ["a", "c"]
    assert [r["graph_region_ids"] for r in row["candidate_bindings"]] == [["a"], ["a", "b"], ["c"]]
    assert row["candidate_bindings"][1]["outputs"][0]["id"] == "b:v0"
    assert row["numeric_match_status"] == "unknown" and row["source_region"] is None


def test_indexed_operator_keeps_both_bases_and_unmatched_keeps_whole_roster():
    graph = dict(
        schema="merlin.capture_graph_regions.v1",
        regions=[
            _region("a", "model.first", "aten.add_.Tensor", consumer="output"),
            _region("b", "model.second", "aten.add_.Tensor", consumer="output"),
        ],
    )
    listing = dict(
        regions=[
            dict(region_id=0),
            dict(
                region_id=1,
                body_match=dict(
                    basis="complete_listed_body_bytes",
                    contiguous_body=True,
                    matches=[dict(symbol="vendor_add_1_native")],
                ),
            ),
            dict(region_id=2),
        ]
    )
    rows = graph_binding_hypotheses(graph, listing)["graph_bindings"]
    assert len(rows) == 3 and rows[0]["status"] == rows[2]["status"] == "unbound"
    assert rows[0]["candidate_region_ids"] == ["a", "b"]
    assert {basis["index_base"] for c in rows[1]["candidate_bindings"] for basis in c["basis"]} == {0, 1}


def test_partial_body_never_authorizes_named_association():
    graph = dict(schema="merlin.capture_graph_regions.v1", regions=[_region("a", "model.first", "custom.first")])
    listing = dict(
        regions=[
            dict(
                region_id=1,
                body_match=dict(basis="prefix_bytes", contiguous_body=True, matches=[dict(symbol="vendor_first")]),
            )
        ]
    )
    assert graph_binding_hypotheses(graph, listing)["graph_bindings"][0]["candidate_bindings"] == []


def test_root_owned_calls_do_not_hide_nested_module_associations():
    graph = dict(
        schema="merlin.capture_graph_regions.v1",
        regions=[
            _region("root", "model", "custom.root"),
            _region("parent", "model.branch", "custom.parent"),
            _region("child", "model.branch.projection", "custom.projection"),
        ],
    )
    listing = dict(
        regions=[
            dict(
                region_id=1,
                body_match=dict(
                    basis="complete_listed_body_bytes",
                    contiguous_body=True,
                    matches=[dict(symbol="vendor_branch_projection_native")],
                ),
            )
        ]
    )
    assert graph_binding_hypotheses(graph, listing)["graph_bindings"][0]["candidate_region_ids"] == ["child"]
