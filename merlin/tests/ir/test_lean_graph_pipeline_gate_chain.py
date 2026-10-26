"""Bounded SDK-free chain controls. All kernel/custody results are MOCKED."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import time
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("old_gate_tests", Path(__file__).with_name("test_lean_graph_pipeline_gate.py"))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

SOURCE = '''builtin.module {
  func.func private @observe(i8) -> i8
  func.func @main(%a: i8, %b: i8) -> i8 {
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %st = arith.constant 1 : index
    %r = scf.for %i = %lb to %ub step %st iter_args(%c = %a) -> (i8) {
      %x = arith.addi %a, %b : i8
      %y = arith.addi %a, %b : i8
      %z = arith.muli %x, %y : i8
      %v = arith.addi %c, %z : i8
      %o = func.call @observe(%v) : (i8) -> i8
      scf.yield %o : i8
    }
    %out = func.call @observe(%r) : (i8) -> i8
    func.return %out : i8
  }
  func.func @other(%a: i8) -> i8 {
    func.return %a : i8
  }
}'''


def reindex_source():
    source = SOURCE.replace('func.func private @observe',
        'func.func private @touch(memref<4xi8>) -> memref<4xi8>\n  func.func private @observe')
    source = source.replace('%lb = arith.constant', '%dead = arith.constant 7 : i16\n    %lb = arith.constant')
    return source[:-1] + '''  func.func @opaque(%m: memref<4xi8>) -> memref<4xi8> {
          %r = func.call @touch(%m) : (memref<4xi8>) -> memref<4xi8>
          func.return %r : memref<4xi8>
        }
        }'''


class Chain(unittest.TestCase):
    def setUp(self):
        from merlin.verify import lean_graph_pipeline_gate_chain as chain
        from merlin.verify import lean_graph_pipeline_gate_chain_live as live
        from merlin.verify import lean_graph_pipeline_gate_chain_models as models
        from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
        from merlin.transforms.certificates import parse_ir
        self.chain, self.live, self.models = chain, live, models
        self.options = PipelineOptions(index_bits=64, normalize_shapes=False, simplify_integers=False,
                                       reuse_buffers=False, eliminate_redundancy=True, hoist_invariants=True)
        self.pipeline, self.parse = apply_common_pipeline, parse_ir
        self.changes = self.pipeline(self.parse(SOURCE), self.options, plan_only=True).changes
        self.assertEqual(tuple(c.kind for c in self.changes), chain.ORDER)

    def test_actual_common_model_full_roster_direct_and_composed_query(self):
        p = self.chain.prepare(SOURCE, self.changes)
        projections = [json.loads(s) for s in p.projections]
        self.assertEqual(len(projections[0]["functions"]), 3)
        self.assertTrue(all(x["registry"] == projections[0]["registry"] for x in projections))
        self.assertEqual(projections[0]["functions"][1]["after"], projections[1]["functions"][1]["before"])
        self.assertIn("merlin_chain_middle_join", p.query)
        self.assertIn("merlin_chain_composed", p.query)
        self.assertIn("merlin_chain_transform", p.query)
        self.assertNotIn("#print", p.query)
        self.assertTrue(self.chain.matches(p.describe(), SOURCE, self.changes))
        witness, roots, status = self.models.witness(p)
        self.assertIn("source_success", witness)
        self.assertIn(".error (.external 17,8)", witness)
        self.assertFalse(status["source_success_assumed"])
        self.assertEqual(len(roots), len(set(roots)))

    def test_wrong_middle_order_missing_and_certificate_refuse(self):
        a, b = self.changes
        wrong = old.rebound(b, before_ir=b.before_ir.replace("3 : index", "4 : index"))
        for changes in ((a,), (b, a), (a, wrong), (a, replace(b, certificate=a.certificate))):
            with self.subTest(changes=len(changes)), self.assertRaises(ValueError):
                self.chain.prepare(SOURCE, changes)
        with self.assertRaises(ValueError):
            self.chain.prepare(SOURCE.replace("3 : index", "4 : index"), self.changes)

    def test_valid_altered_final_observer_and_callee_change_query_not_authority(self):
        # Parse/verify each changed module and refresh all certificate hashes.
        # Preparing a different query is NOT an observed Lean refusal.
        good = self.chain.prepare(SOURCE, self.changes)
        final = self.changes[-1].after_ir
        mutations = (final.replace("arith.muli", "arith.addi"),
                     final.replace('"func.return"(%out)', '"func.return"(%r)'),
                     final.replace('"func.call"(%r)', '"func.call"(%a)'))
        for text in mutations:
            self.assertNotEqual(text, final)
            self.parse(text).verify()
            changed = (*self.changes[:-1], old.rebound(self.changes[-1], after_ir=text))
            with self.assertRaises(ValueError):
                self.chain.prepare(SOURCE, changed)
            # Independent endpoint obligation still sees the semantic edit;
            # no composed producer certificate is manufactured for it.
            from merlin.verify.lean_graph_state_scalar_capture import project
            from merlin.verify.lean_graph_pipeline_gate_projection import complete_query
            projection = project(self.parse(SOURCE), self.parse(text))
            query, _ = complete_query(projection)
            original_query, _ = complete_query(json.loads(good.projections[-1]))
            self.assertNotEqual(query, original_query)

    def test_saved_registry_roster_domain_and_source_forgeries_refuse(self):
        p = self.chain.prepare(SOURCE, self.changes)
        for key, value in (("domain", "local"), ("boundaries", []), ("sources", []), ("buffers", list(reversed(p.buffers)))):
            forged = p.describe(); forged[key] = value
            self.assertFalse(self.chain.matches(forged, SOURCE, self.changes))
        forged = p.describe()
        projected = json.loads(forged["projections"][0]); projected["registry"].reverse()
        forged["projections"] = [json.dumps(projected), *forged["projections"][1:]]
        self.assertFalse(self.chain.matches(forged, SOURCE, self.changes))

    def test_interface_and_external_callee_roster_refuse(self):
        final = self.changes[-1].after_ir
        for text in (final.replace('sym_name = "other"', 'sym_name = "renamed"'),
                     final.replace('sym_name = "observe"', 'sym_name = "different_observer"').replace("@observe", "@different_observer")):
            self.assertNotEqual(text, final)
            self.parse(text).verify()
            with self.assertRaises(ValueError):
                self.chain.prepare(SOURCE, (self.changes[0], old.rebound(self.changes[1], after_ir=text)))

    def test_actual_dce_changes_independent_opaque_type_numbers(self):
        source = reindex_source()
        changes = self.pipeline(self.parse(source), self.options, plan_only=True).changes
        from merlin.verify.lean_graph_pipeline_gate_projection import prepare
        separate = [json.loads(prepare(c).projection) for c in changes]
        self.assertNotEqual(separate[0]['types'], separate[1]['types'])
        self.assertNotEqual(separate[0]['registry'], separate[1]['registry'])
        common = self.chain.prepare(source, changes)
        projected = [json.loads(p) for p in common.projections]
        self.assertTrue(all(p['registry'] == projected[0]['registry'] for p in projected))
        self.assertTrue(all(p['types'] == projected[0]['types'] for p in projected))

    def test_public_hold_and_saved_receipt_cannot_grant_chain(self):
        p = self.chain.prepare(SOURCE, self.changes)
        with self.assertRaises(RuntimeError):
            self.live.verify(SOURCE, self.changes, toolchain=object(), deadline=time.monotonic()+30)
        fake = self.live.EndpointCompletion(p.identity, "a"*64, p.roots, "b"*64, "c"*64)
        self.assertFalse(self.live.accepted(fake, SOURCE, self.changes, toolchain=object()))

    def test_actual_service_wrapper_requires_endpoint_and_leaves_v1_output_untouched(self):
        case = old.ServiceBoundary(); case.setUp()
        self.addCleanup(case.doCleanups)
        case.activate_mock_methods()
        request = {**case.request, "source_ir": SOURCE,
                   "options": {**case.request["options"], "hoist_invariants": True}}
        # Both per-boundary positives are insufficient while endpoint held.
        with self.assertRaises(RuntimeError):
            self.live.apply_service(case.service, request, work_root=case.context.workspace, timeout_s=60)
        native = self.live.native
        with patch.object(native, "check", native._check_with_context), patch.object(
                native, "revalidate", native._revalidate_with_context):
            result = self.live.apply_service(case.service, request, work_root=case.context.workspace, timeout_s=60)
            self.assertTrue(self.live.eligible_delivery(case.binding, result, SOURCE))
            self.assertEqual(result["compiler_pass_chain"], "not_established")
            self.assertFalse(case.service.compiler_eligible(result, source_ir=SOURCE))
            self.assertFalse(self.live.eligible_delivery(case.binding, dict(result), SOURCE))

    def test_actual_checked_pipeline_plus_live_endpoint_mock_only(self):
        fixture = old.WholeBoundary(); fixture.setUp()
        Context, _, context, checked = fixture.context_fixture()
        # All imports precede service/source binding. Constructors/guards must
        # never reach the runtime. Only real producer/parser and host code run.
        from merlin.verify import lean_graph_pipeline_gate_service as service
        from merlin.verify import lean_graph_pipeline_gate as gate
        from merlin.verify.lean_graph_pipeline_gate_projection import sha
        native = self.live.native
        with patch.object(Context, "check_query", side_effect=checked), patch.object(
                Context, "revalidate", side_effect=lambda x: x), patch.object(
                Context, "_guard", side_effect=AssertionError("no SDK")), patch.object(
                Context, "from_review", side_effect=AssertionError("no Context construction")), patch.object(
                gate.native, "check", gate.native._check_with_context), patch.object(
                gate.native, "revalidate", gate.native._revalidate_with_context), patch.object(
                native, "check", native._check_with_context), patch.object(
                native, "revalidate", native._revalidate_with_context):
            binding = service.bind_service_context(context, compiler_profile_sha256="a"*64, timeout_s=30)
            receipts = []
            def verifier(change):
                result = gate.verify(change, toolchain=context, deadline=time.monotonic()+30)
                receipts.append(result)
                return result
            result = self.pipeline(self.parse(SOURCE), self.options, verifier=verifier, plan_only=False)
            changes = result.changes
            endpoint = self.live.verify(SOURCE, changes, toolchain=context, deadline=time.monotonic()+30)
            p = self.chain.prepare(SOURCE, changes)
            document = {"source_sha256": sha(SOURCE), "transformed_ir": changes[-1].after_ir,
                        "plan_only": False, "chain_prepared_identity": p.identity, "compiler_pass_chain": "not_established"}
            def record(value=endpoint, roster=None, doc=None):
                self.live.record_delivery(binding, document if doc is None else doc, SOURCE, changes,
                                          tuple(receipts) if roster is None else roster, value, deadline=time.monotonic()+30)
            for endpoint_bad in (receipts[-1], replace(endpoint), asdict(endpoint), None):
                with self.assertRaises(RuntimeError): record(endpoint_bad)
            with self.assertRaises(RuntimeError): record(roster=(receipts[-1],))
            with self.assertRaises(RuntimeError): record(roster=tuple(reversed(receipts)))
            with self.assertRaises(RuntimeError): record(doc={**document, "plan_only": True})
            record()
            self.assertTrue(self.live.eligible_delivery(binding, document, SOURCE))
            self.assertFalse(self.live.eligible_delivery(binding, copy.deepcopy(document), SOURCE))
            document["transformed_ir"] += "\n"
            self.assertFalse(self.live.eligible_delivery(binding, document, SOURCE))
            document["transformed_ir"] = changes[-1].after_ir
            context.held = True
            self.assertFalse(self.live.eligible_delivery(binding, document, SOURCE))
        # Removing unit patches must restore the native hold and authority refusal.
        self.assertFalse(self.live.eligible_delivery(binding, document, SOURCE))


def retain(destination):
    test = Chain(); test.setUp()
    p = test.chain.prepare(SOURCE, test.changes)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "original.mlir").write_text(SOURCE)
    for i, text in enumerate((test.changes[0].before_ir, test.changes[0].after_ir, test.changes[1].after_ir)):
        (destination / f"buffer-{i}.mlir").write_text(text)
    (destination / "PREPARED.json").write_text(json.dumps(p.describe(), indent=2)+"\n")
    (destination / "CHANGES.json").write_text(json.dumps([asdict(c) for c in test.changes], indent=2)+"\n")
    (destination / "Endpoint.lean").write_text(p.query)
    query, roots, info = test.models.witness(p)
    (destination / "Witness.lean").write_text(query)
    (destination / "WITNESS.json").write_text(json.dumps({**info, "roots": roots}, indent=2)+"\n")
    request = test.live.native.request(p)
    (destination / "REQUEST.json").write_text(json.dumps({**asdict(request), "query": request.query.decode()}, indent=2)+"\n")
    # Real descriptor/type-renumbering gap, with every source and projection.
    source = reindex_source()
    changes = test.pipeline(test.parse(source), test.options, plan_only=True).changes
    common = test.chain.prepare(source, changes)
    from merlin.verify.lean_graph_pipeline_gate_projection import prepare, complete_query
    from merlin.verify.lean_graph_state_scalar_capture import project
    (destination / "reindex-original.mlir").write_text(source)
    (destination / "REINDEX.json").write_text(json.dumps({
        "changes": [asdict(c) for c in changes],
        "independent_projections": [json.loads(prepare(c).projection) for c in changes],
        "common_prepared": common.describe(), "compiler_pass_chain": "not_established",
    }, indent=2)+"\n")
    final = test.changes[1].after_ir
    negative = final.replace('"func.call"(%r)', '"func.call"(%a)')
    test.parse(negative).verify()
    projection = project(test.parse(SOURCE), test.parse(negative))
    query, roots = complete_query(projection)
    (destination / "negative-observer.mlir").write_text(negative)
    (destination / "NegativeObserver.lean").write_text(query)
    (destination / "NEGATIVE.json").write_text(json.dumps({
        "status": "UNCOMPILED", "kernel_refusal_observed": False,
        "mutation": "final observer receives original a instead of loop-carried r",
        "parse_verify": True, "roots": roots, "projection": projection,
        "producer_certificate": None,
    }, indent=2)+"\n")


if __name__ == "__main__":
    def source_only_hold(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if ("/original-runtime-candidate" in path or "/toolchain/lib/lean/" in path
                    or "/toolchain/bin/" in path or path.endswith((".olean", ".ilean"))):
                raise AssertionError("chain SDK-free suite cannot open native images or Lean objects")
    sys.addaudithook(source_only_hold)
    old.isolated_owners(*(Path(p).resolve() for p in sys.argv[1:3]))
    args = sys.argv[3:]
    if args and args[0] == "--retain":
        retain(Path(args[1])); args = args[2:]
    unittest.main(argv=[sys.argv[0], *args])
