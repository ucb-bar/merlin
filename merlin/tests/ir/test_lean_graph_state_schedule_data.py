"""SOURCE-ONLY stdlib tests on retained/synthetic metadata; never SDK/binaries.

Run with system Python -I -S and explicit MERLIN_REPO_ROOT during the shared
runtime hold. This intentional stdlib-only harness must not import project
initializers/tool discovery. It provides NO Lean or runtime qualification.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(os.environ.get("MERLIN_REPO_ROOT", os.getcwd())).resolve()
SOURCE = ROOT / "src/merlin/verify"
BASE = ROOT / "out/artifacts/verification/graph-state-schedule"


def owned_module(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


errors = owned_module("lean_graph_state_schedule_errors")
schema = owned_module("lean_graph_state_schedule_receipt_schema")


def extract(path, names, namespace):
    """Load only selected function ASTs, discarding every project import/global."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def reference(name="source.py"):
    return {"path": str(BASE / "synthetic" / name), "sha256": "0" * 64, "bytes": 7}


def receipt():
    value = {key: None for key in schema.KEYS}
    value.update(schema=schema.SCHEMA, version=1, root=str(BASE / "synthetic"),
                 settings={"factor": 2, "tile": 3, "index_bits": 32, "count": 1, "minimum_depth": 1},
                 mode="graph", model=None, qualification_state="UNQUALIFIED", claims=dict(schema.CLAIMS),
                 roots=["Merlin.GraphStateScheduleInstance.schedule_complete"])
    for key in ("originals", "orders", "snapshots", "order_snapshots"):
        value[key] = [reference(key + "0.mlir"), reference(key + "1.mlir")]
    for key in ("sources", "files", "dependencies"):
        value[key] = [reference(key + ".py")]
    for key in ("journal", "journal_snapshot", "report", "capture", "profile", "query", "object"):
        value[key] = reference(key + ".json")
    return value


class SchemaTests(unittest.TestCase):
    def test_synthetic_receipt_shape_is_only_unqualified_metadata(self):
        result = schema.shape(receipt())
        self.assertEqual(result["qualification_state"], "UNQUALIFIED")
        self.assertIs(result["claims"]["qualified"], False)

    def test_closed_version_and_claim_refusals(self):
        changes = (
            ("version", True), ("version", 0), ("schema", "legacy"),
            ("mode", "approved"), ("qualification_state", "QUALIFIED"), ("roots", []),
        )
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(errors.ScheduleDomainRefusal):
                candidate = receipt()
                candidate[key] = value
                schema.shape(candidate)
        for key in schema.CLAIMS:
            if type(schema.CLAIMS[key]) is bool:
                candidate = receipt()
                candidate["claims"][key] = not schema.CLAIMS[key]
                with self.subTest(claim=key), self.assertRaises(errors.ScheduleDomainRefusal):
                    schema.shape(candidate)

    def test_reference_types_are_exact(self):
        for key, value in (("bytes", True), ("bytes", -1), ("sha256", "A" * 64), ("path", "relative")):
            candidate = reference()
            candidate[key] = value
            with self.subTest(field=key), self.assertRaises(errors.ScheduleDomainRefusal):
                schema.reference_row(candidate)

    def test_json_duplicate_float_nonfinite_refused(self):
        for text in ('{"x":1,"x":2}', '{"x":1.0}', '{"x":NaN}', 'not json'):
            with self.subTest(text=text), self.assertRaises(errors.ScheduleDomainRefusal):
                schema.strict_json(text)

    def test_retained_current_order_journal_shape(self):
        candidates = sorted({p.resolve() for p in (BASE / "development/same-current-block-v1").glob("*/producer/actual/REQUESTS.json")})
        self.assertEqual(len(candidates), 1, "explicit retained metadata fixture missing/ambiguous; no producer fallback")
        path = candidates[0]
        rows = schema.journal_shape(schema.strict_json(path.read_text()))
        self.assertEqual(rows[0]["block_index"], rows[1]["block_index"])
        self.assertNotEqual(rows[0]["witness"]["operation_sha256"], rows[1]["witness"]["operation_sha256"])
        for mutate in ("bool", "duplicate", "request-witness"):
            bad = copy.deepcopy(rows)
            if mutate == "bool":
                bad[0]["order"][0] = True
            elif mutate == "duplicate":
                bad[0]["witness"]["edges"].append(bad[0]["witness"]["edges"][0])
            else:
                bad[1]["witness"]["order"] = bad[0]["order"]
            with self.subTest(kind=mutate), self.assertRaises(errors.ScheduleDomainRefusal):
                schema.journal_shape(bad)

    def test_model_cannot_assume_source_success_or_approve_native(self):
        candidate = receipt()
        candidate["mode"] = "initialized-scalar-provider-v1"
        candidate["model"] = {"source_success_assumed": True}
        with self.assertRaises(errors.ScheduleDomainRefusal):
            schema.shape(candidate)
        candidate["model"] = {"source_success_assumed": False, "native_or_poison_fidelity_proved": True}
        with self.assertRaises(errors.ScheduleDomainRefusal):
            schema.shape(candidate)


class HoldAndErrorTests(unittest.TestCase):
    def test_guard_stops_before_any_dependency_or_sdk_action(self):
        events = []
        namespace = {"artifacts_dir": lambda: ROOT / "out/artifacts", "ARTIFACT": "verification/graph-state-schedule",
                     "build_dir": lambda: events.append("unexpected build dependency access")}
        extract(SOURCE / "lean_graph_state_schedule_kernel.py", {"guard"}, namespace)
        with self.assertRaisesRegex(RuntimeError, "SHARED SELECTED RUNTIME HOLD"):
            namespace["guard"]()
        self.assertEqual(events, [])

    def test_domain_adapter_never_relabels_identity_or_author_value_error(self):
        namespace = {"ScheduleDomainRefusal": errors.ScheduleDomainRefusal,
                     "ScheduleUnavailable": errors.ScheduleUnavailable,
                     "IR_DOMAIN": frozenset({"unsupported actual CFG; no unreachable block can be dropped"})}
        extract(SOURCE / "lean_graph_state_schedule_projection.py", {"domain_call"}, namespace)
        for reason in ("producer/runtime identity changed during intake", "submitted buffers changed", "unexpected author bug"):
            def broken():
                raise ValueError(reason)
            with self.subTest(reason=reason), self.assertRaises(errors.ScheduleUnavailable):
                namespace["domain_call"](broken)
        def domain():
            raise ValueError("unsupported actual CFG; no unreachable block can be dropped")
        with self.assertRaises(errors.ScheduleDomainRefusal):
            namespace["domain_call"](domain)

    def test_identity_adapter_is_unavailable_not_domain(self):
        namespace = {"ScheduleIdentityUnavailable": errors.ScheduleIdentityUnavailable}
        extract(SOURCE / "lean_graph_state_schedule_capture.py", {"identity_call"}, namespace)
        def identity():
            raise ValueError("producer/runtime identity changed during intake")
        with self.assertRaises(errors.ScheduleIdentityUnavailable):
            namespace["identity_call"](identity)

    def test_unexpected_prefix_author_error_is_not_an_ordinary_refusal(self):
        # Source-level inspection plus adapter test above, without importing old
        # project initializers or opening any recorded tool/runtime reference.
        tree = ast.parse((SOURCE / "lean_graph_state_schedule_emit.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prefix_author")
        names = [n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn("ScheduleUnavailable", names)
        strings = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        self.assertNotIn("producer/runtime identity changed during intake", strings)


if __name__ == "__main__":
    unittest.main()
