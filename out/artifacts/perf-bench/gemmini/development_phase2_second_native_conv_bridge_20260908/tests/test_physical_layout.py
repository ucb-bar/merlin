#!/usr/bin/env python3
"""Tests for target-neutral graph-level physical-layout planning."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "compiler"))

from mlir_oot.lowering.physical_layout import (
    LayoutGraph, LayoutOp, LayoutPlanningError, LayoutValue, solve,
)
from mlir_oot.frontend.capture_layout import RegionLayoutPolicy, extract_region_layout_graph
from mlir_oot.frontend.parse import parse_module


LAYOUTS = ("NCHW", "NHWC")


def value(name: str, shape=(1, 16, 8, 8), dtype="i8") -> LayoutValue:
    return LayoutValue(name, tuple(shape), dtype)


def polymorphic(name: str, kind: str, inputs, outputs, preferred=None) -> LayoutOp:
    return LayoutOp(name, kind, tuple(inputs), tuple(outputs), True, LAYOUTS, preferred)


def boundary(name: str, inputs=(), outputs=()) -> LayoutOp:
    return LayoutOp(name, "boundary", tuple(inputs), tuple(outputs), False, ("NCHW",))


class PhysicalLayoutTest(unittest.TestCase):
    def test_provenance_extractor_builds_residual_agreement_constraint(self) -> None:
        module = parse_module(r'''builtin.module {
          func.func @forward(%x: tensor<1x4x2x2xi8>) -> tensor<1x4x2x2xi8> {
            %left = "fixture.op"(%x) {prov.region_id = "conv_left", prov.op = "conv"}
              : (tensor<1x4x2x2xi8>) -> tensor<1x4x2x2xi8>
            %right = "fixture.op"(%x) {prov.region_id = "conv_right", prov.op = "conv"}
              : (tensor<1x4x2x2xi8>) -> tensor<1x4x2x2xi8>
            %sum = "fixture.op"(%left, %right) {prov.region_id = "merge", prov.op = "add"}
              : (tensor<1x4x2x2xi8>, tensor<1x4x2x2xi8>) -> tensor<1x4x2x2xi8>
            %q = "fixture.op"(%sum) {prov.region_id = "quant", prov.op = "quantize"}
              : (tensor<1x4x2x2xi8>) -> tensor<1x4x2x2xi8>
            func.return %q : tensor<1x4x2x2xi8>
          }
        }''')
        policy = RegionLayoutPolicy(
            layouts=LAYOUTS,
            canonical_layout="NCHW",
            preferred_accelerator_layout="NHWC",
            accelerator_ops=frozenset({"conv"}),
            preserving_ops=frozenset({"quantize"}),
            residual_ops=frozenset({"add"}),
            view_ops=frozenset({"view"}),
            fixed_layout_ops=frozenset(),
        )
        graph, census = extract_region_layout_graph(module, policy)
        plan = solve(graph)
        self.assertEqual(census["included_op_counts"]["accelerator"], 2)
        self.assertEqual(census["included_op_counts"]["residual"], 1)
        self.assertEqual(set(plan.assignments.values()), {"NHWC"})
        self.assertEqual([(item.op, item.port) for item in plan.conversions], [
            ("graph_input_0", "output"),
            ("graph_output_0", "input"),
        ])

    def test_conv_elementwise_quant_chain_converts_only_at_abi_boundaries(self) -> None:
        values = {name: value(name) for name in ("x", "a", "b", "c", "y")}
        graph = LayoutGraph(LAYOUTS, "NCHW", values, (
            boundary("input", outputs=("x",)),
            polymorphic("conv0", "accelerator", ("x",), ("a",), "NHWC"),
            polymorphic("relu", "layout_preserving", ("a",), ("b",)),
            polymorphic("quantize", "layout_preserving", ("b",), ("c",)),
            polymorphic("conv1", "accelerator", ("c",), ("y",), "NHWC"),
            boundary("output", inputs=("y",)),
        ))
        plan = solve(graph)
        self.assertEqual(set(plan.assignments.values()), {"NHWC"})
        self.assertEqual([(item.op, item.port) for item in plan.conversions],
                         [("input", "output"), ("output", "input")])
        self.assertFalse(any(item.op in {"relu", "quantize"} for item in plan.conversions))

    def test_residual_merge_forces_both_branches_and_output_to_agree(self) -> None:
        values = {name: value(name) for name in ("x", "left", "right", "sum", "y")}
        graph = LayoutGraph(LAYOUTS, "NCHW", values, (
            polymorphic("left_conv", "accelerator", ("x",), ("left",), "NHWC"),
            polymorphic("right_conv", "accelerator", ("x",), ("right",), "NHWC"),
            polymorphic("residual", "residual", ("left", "right"), ("sum",)),
            polymorphic("consumer", "accelerator", ("sum",), ("y",), "NHWC"),
        ))
        plan = solve(graph)
        self.assertEqual({plan.assignments[name] for name in values}, {"NHWC"})
        self.assertEqual(len(plan.components), 1)
        self.assertEqual(plan.conversions, [])

    def test_rank_changing_view_is_a_real_boundary(self) -> None:
        values = {
            "before": value("before", (1, 16, 1, 1)),
            "after": value("after", (1, 16, 1, 1)),
            "sink": value("sink", (1, 16, 1, 1)),
        }
        graph = LayoutGraph(LAYOUTS, "NCHW", values, (
            polymorphic("producer", "accelerator", (), ("before",), "NHWC"),
            boundary("rank_changing_view", inputs=("before",), outputs=("after",)),
            polymorphic("consumer", "accelerator", ("after",), ("sink",), "NHWC"),
        ))
        plan = solve(graph)
        self.assertEqual([(item.op, item.port) for item in plan.conversions], [
            ("rank_changing_view", "input"),
            ("rank_changing_view", "output"),
        ])

    def test_shape_identical_view_can_join_components(self) -> None:
        values = {name: value(name) for name in ("before", "after", "sink")}
        graph = LayoutGraph(LAYOUTS, "NCHW", values, (
            polymorphic("producer", "accelerator", (), ("before",), "NHWC"),
            polymorphic("identity_view", "layout_preserving_view",
                        ("before",), ("after",)),
            polymorphic("consumer", "accelerator", ("after",), ("sink",), "NHWC"),
        ))
        plan = solve(graph)
        self.assertEqual(len(plan.components), 1)
        self.assertEqual(plan.conversions, [])

    def test_incompatible_joined_operator_domains_fail_closed(self) -> None:
        values = {name: value(name) for name in ("x", "y")}
        graph = LayoutGraph(LAYOUTS, "NCHW", values, (
            LayoutOp("first", "fixed", ("x",), ("y",), True, ("NCHW",)),
            LayoutOp("second", "fixed", ("x",), ("y",), True, ("NHWC",)),
        ))
        with self.assertRaisesRegex(LayoutPlanningError, "incompatible operator layouts"):
            solve(graph)

    def test_canonical_structural_graph_keeps_all_residuals_and_convs_together(self) -> None:
        path = BUNDLE / "validation/canonical_resnet50/physical_layout_graph.json"
        if not path.is_file():
            self.skipTest("canonical structural graph was not packaged")
        graph = LayoutGraph.from_dict(json.loads(path.read_text()))
        plan = solve(graph)
        accelerator = [op for op in graph.ops if op.kind == "accelerator"]
        residuals = [op for op in graph.ops if op.kind == "residual"]
        self.assertEqual(len(accelerator), 53)
        self.assertEqual(len(residuals), 16)
        self.assertTrue(all(plan.assignments[name] == "NHWC"
                            for op in accelerator for name in op.ports))
        for op in residuals:
            self.assertEqual({plan.assignments[name] for name in op.ports}, {"NHWC"})


if __name__ == "__main__":
    unittest.main()
