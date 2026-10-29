"""Isolated retained/synthetic data tests for strict IR-field association.

No native/compiler/proof invocation, SDK reads or compiler-validity claim. Wrong
SSA/observer/ABI mutations are well-formed JSON, with compiler validity unmeasured.
"""

from __future__ import annotations

import argparse
import copy
import json
import runpy
import unittest
from pathlib import Path


class ProjectionTests(unittest.TestCase):
    def reject_module(self, document):
        with self.assertRaises(ValueError):
            term = self.projection["module_term"](document, self.serializer)
            self.projection["require_literal"](self.source, "actualBefore", "Module", term)

    def reject_raw(self, kernel):
        with self.assertRaises(ValueError):
            term = self.projection["raw_kernel"](kernel, self.serializer)
            self.projection["require_literal"](self.source, "actualKernel", "RawKernel", term)

    def reject_plan(self, plan, locators):
        with self.assertRaises((ValueError, KeyError)):
            term, _ = self.projection["bind_plan"](plan, locators, self.document, self.serializer)
            self.projection["require_literal"](self.source, "actualPlan", "Plan", term)

    def test_genuine_complete_source(self):
        term = self.projection["module_term"](self.document, self.serializer)
        self.projection["require_literal"](self.source, "actualBefore", "Module", term)

    def test_genuine_complete_kernel(self):
        term = self.projection["raw_kernel"](self.kernel, self.serializer)
        self.projection["require_literal"](self.source, "actualKernel", "RawKernel", term)

    def test_genuine_plan_locators(self):
        term, mapped = self.projection["bind_plan"](self.plan, self.locators, self.document, self.serializer)
        self.projection["require_literal"](self.source, "actualPlan", "Plan", term)
        self.assertEqual(mapped["actual_value_ids"], [2, 0, 3, 9, 1, 16, 21])
        self.assertEqual([row["body_index"] for row in mapped["stage_rows"]], [4, 5, 6])

    def test_hidden_nested_type_field(self):
        d = copy.deepcopy(self.document)
        d["module"]["ops"][0]["regions"][0]["blocks"][0]["args"][0]["type"]["ignored"] = True
        self.reject_module(d)

    def test_hidden_attribute_field(self):
        d = copy.deepcopy(self.document)
        d["module"]["ops"][0]["attrs"]["sym_name"]["candidate"] = True
        self.reject_module(d)

    def test_hidden_ssa_value_field(self):
        d = copy.deepcopy(self.document)
        d["module"]["ops"][0]["regions"][0]["blocks"][0]["args"][0]["fresh"] = True
        self.reject_module(d)

    def test_hidden_operation_field(self):
        d = copy.deepcopy(self.document)
        d["module"]["ops"][0]["effects"] = []
        self.reject_module(d)

    def test_same_width_producer_operand(self):
        d = copy.deepcopy(self.document)
        body = d["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
        body[4]["regions"][0]["blocks"][0]["ops"][1]["operands"][1] = 10
        self.reject_module(d)

    def test_changed_unchanged_observer(self):
        d = copy.deepcopy(self.document)
        d["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"][8]["operands"][0] = 9
        self.reject_module(d)

    def test_swapped_same_type_return_observers(self):
        d = copy.deepcopy(self.document)
        returned = d["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"][-1]["operands"]
        returned[0], returned[1] = returned[1], returned[0]
        self.reject_module(d)

    def test_wrong_source_window(self):
        k = copy.deepcopy(self.kernel)
        k["body"][2]["args"][1][1]["value"] = 6
        self.reject_raw(k)

    def test_missing_second_consumer(self):
        k = copy.deepcopy(self.kernel)
        k["body"].pop()
        self.reject_raw(k)

    def test_wrong_abi_output_order(self):
        k = copy.deepcopy(self.kernel)
        k["args"][-1], k["args"][-2] = k["args"][-2], k["args"][-1]
        self.reject_raw(k)

    def test_async_field_retained(self):
        k = copy.deepcopy(self.kernel)
        k["body"][0]["awaits"] = ["completion"]
        self.reject_raw(k)

    def test_hidden_raw_expression(self):
        k = copy.deepcopy(self.kernel)
        k["body"][0]["args"][0][1]["overflow"] = "none"
        self.reject_raw(k)

    def test_stale_same_type_plan_value(self):
        locators = copy.deepcopy(self.locators)
        locators[3]["owner_ordinal"] = 10
        self.reject_plan(self.plan, locators)

    def test_missing_plan_value(self):
        locators = copy.deepcopy(self.locators)
        locators[3]["owner_ordinal"] = 99999
        self.reject_plan(self.plan, locators)

    def test_missing_plan_output(self):
        p = copy.deepcopy(self.plan)
        p["outputs"].pop()
        self.reject_plan(p, self.locators)

    def test_hidden_plan_metadata(self):
        p = copy.deepcopy(self.plan)
        p["values"][0]["qualified"] = True
        self.reject_plan(p, self.locators)

    def test_ambiguous_duplicate_literal(self):
        term = self.projection["module_term"](self.document, self.serializer)
        line = f"def actualBefore : Module := {term}"
        with self.assertRaises(ValueError):
            self.projection["require_literal"](self.source + "\n" + line + "\n", "actualBefore", "Module", term)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    api = runpy.run_path(str(args.helper))
    projection = runpy.run_path(str(args.projection))
    root = args.root.resolve(strict=True)
    base = root / api["BASE"]
    _, evidence = api["original_evidence"](root, "qualification-v1-originals/verified-i8/EVIDENCE.json", api["EVIDENCE"])
    serializer_pin = dict(
        path=str(base / "qualification-v1/source-snapshot/src/merlin/verify/lean_graph_regions_declared_ast.py"),
        bytes=projection["SERIALIZER"][0], sha256=projection["SERIALIZER"][1],
    )
    api["read_bound"](root, serializer_pin)
    ProjectionTests.serializer = runpy.run_path(serializer_pin["path"])
    ProjectionTests.projection = projection
    ProjectionTests.document = json.loads(api["read_bound"](root, evidence["exports"]["before"]))
    ProjectionTests.kernel = json.loads(api["read_bound"](root, evidence["originals"]["actual-output-kernel.json"]))
    ProjectionTests.plan = json.loads(api["read_bound"](root, evidence["originals"]["original-connected-plan.json"]))
    ProjectionTests.locators = json.loads(
        api["read_bound"](root, evidence["originals"]["original-connected-plan-values.json"])
    )
    ProjectionTests.source = (base / "retained-literal-binding-v3/Generated.lean").read_text()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProjectionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (args.destination / "RESULT.json").write_text(json.dumps(dict(
        scope="SOURCE DATA ONLY; JSON mutation/compiler validity unmeasured",
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), passed=result.wasSuccessful(), native_invocations=0,
        sdk_binary_reads=0, qualified=False, selected_runtime_hold=True,
    ), indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
