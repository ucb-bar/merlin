"""SDK-free actual producer tests; checker completions are explicitly mocked.

Run directly with an isolated Python interpreter and two explicit source roots.
No pytest conftest, optional solver initializer, Lean discovery or native runner.
"""
from __future__ import annotations

import json
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


def isolated_owners(primary, producer):
    for name, relative in (("merlin", "merlin"), ("merlin.verify", "merlin/verify")):
        module = types.ModuleType(name)
        module.__path__ = [str(producer / "src" / relative), str(primary / "src" / relative)]
        sys.modules[name] = module

    def hold(event, args):
        if event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}:
            raise AssertionError("SDK-free tests cannot launch child processes")
        if event == "open" and isinstance(args[0], (str, bytes)):
            name = str(args[0])
            if "/.elan/" in name or "/fresh-environment/" in name or "libleanshared" in name:
                raise AssertionError("SDK-free tests cannot read held/archived runtime")
    sys.addaudithook(hold)


SHAPE = '''builtin.module {
  func.func @main(%a: memref<4xi8>) -> memref<4xi8> {
    %b = "memref.cast"(%a) : (memref<4xi8>) -> memref<4xi8>
    func.return %b : memref<4xi8>
  }
}'''
INTEGER = '''builtin.module {
  func.func @main(%a: i8) -> i8 {
    %z = arith.constant 0 : i8
    %b = arith.addi %a, %z : i8
    func.return %b : i8
  }
}'''
SCALAR = '''builtin.module {
  func.func private @observe(i8) -> i8
  func.func @main(%a: i8, %b: i8) -> i8 {
    %x = arith.addi %a, %b : i8
    %y = arith.addi %a, %b : i8
    %z = arith.muli %x, %y : i8
    %r = func.call @observe(%z) : (i8) -> i8
    func.return %r : i8
  }
  func.func @other(%a: i8) -> i8 {
    func.return %a : i8
  }
}'''
LOOP = '''builtin.module {
  func.func @main(%a: i8, %b: i8) -> i8 {
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %st = arith.constant 1 : index
    %r = scf.for %i = %lb to %ub step %st iter_args(%c = %a) -> (i8) {
      %x = arith.addi %a, %b : i8
      %v = arith.muli %c, %x : i8
      scf.yield %v : i8
    }
    func.return %r : i8
  }
}'''


def rebound(change, **changes):
    """Valid-data adversary refreshes all textual certificate identities too."""
    from merlin.verify.lean_graph_pipeline_gate_projection import sha
    value = replace(change, **changes)
    return replace(value, certificate=replace(value.certificate, kind=value.kind,
        before_ir_sha256=sha(value.before_ir), after_ir_sha256=sha(value.after_ir)))


class LegacyAuthority(unittest.TestCase):
    def setUp(self):
        from merlin.verify import lean_checker
        from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
        from merlin.transforms.certificates import parse_ir
        self.checker, self.options, self.pipeline, self.parse = (
            lean_checker, PipelineOptions, apply_common_pipeline, parse_ir)
        self.queries = []

    def mock_kernel(self, query, receipt, toolchain, **kwargs):
        self.queries.append(query)
        result = replace(receipt, status="verified", toolchain_sha256="0" * 64)
        self.checker._COMPLETED[id(result)] = result
        return result

    def change(self, text, shape=False):
        return self.pipeline(self.parse(text), self.options(
            index_bits=64, normalize_shapes=shape, simplify_integers=not shape,
            reuse_buffers=False), plan_only=True).changes[0]

    def test_local_receipt_has_no_whole_program_authority(self):
        diagnostic = getattr(self.checker, "verify_pipeline_local_obligations",
                             self.checker.verify_pipeline_change)
        for text, shape in ((SHAPE, True), (INTEGER, False)):
            with self.subTest(shape=shape), patch.object(self.checker, "_check_generated", self.mock_kernel):
                change = self.change(text, shape)
                receipt = diagnostic(change, toolchain=None)
                self.assertEqual(receipt.status, "local_checked", receipt.reason)
                self.assertFalse(receipt.verified)
                self.assertFalse(self.checker.accepted_receipt(receipt, change.before_ir, change.after_ir))
                self.assertFalse(self.checker.accepted_pipeline_receipt(receipt, change))
        self.assertIn("id value = value", self.queries[0])
        self.assertIn("∀ x : BitVec", self.queries[1])

    def test_actual_pipeline_rejects_legacy_completion(self):
        diagnostic = getattr(self.checker, "verify_pipeline_local_obligations",
                             self.checker.verify_pipeline_change)
        with patch.object(self.checker, "_check_generated", self.mock_kernel):
            with self.assertRaisesRegex(ValueError, "completion|whole-program"):
                self.pipeline(self.parse(SHAPE), self.options(
                    index_bits=64, simplify_integers=False, reuse_buffers=False),
                    verifier=lambda change: diagnostic(change, toolchain=None))

    def test_arbitrary_callback_never_grants_authority(self):
        for fake in (True, {"status": "verified"}, object(), None):
            with self.subTest(fake=type(fake).__name__), self.assertRaises(ValueError):
                self.pipeline(self.parse(SHAPE), self.options(
                    index_bits=64, simplify_integers=False, reuse_buffers=False), verifier=lambda _: fake)


class WholeBoundary(unittest.TestCase):
    def setUp(self):
        from merlin.verify import lean_graph_pipeline_gate as gate
        from merlin.verify import lean_graph_pipeline_gate_projection as projection
        from merlin.verify import lean_checker
        from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
        from merlin.transforms.certificates import parse_ir
        self.gate, self.mapping, self.checker = gate, projection, lean_checker
        self.parse, self.pipeline = parse_ir, apply_common_pipeline
        self.options = PipelineOptions(index_bits=64, normalize_shapes=False,
                                       simplify_integers=False, reuse_buffers=False,
                                       eliminate_redundancy=True)
        self.change = self.pipeline(self.parse(SCALAR), self.options, plan_only=True).changes[0]

    def mock_kernel(self, prepared, **kwargs):
        return self.gate.native.KernelCompletion(
            prepared.identity, self.mapping.sha(prepared.query), prepared.roots,
            ("propext", "Quot.sound", "Classical.choice"), "1" * 64, "2" * 64)

    def test_actual_changed_projection_and_complete_roster(self):
        prepared = self.mapping.prepare(self.change)
        self.assertNotEqual(prepared.source, prepared.target)
        projected = json.loads(prepared.projection)
        self.assertEqual([f["name"] for f in projected["functions"]], ["observe", "main", "other"])
        self.assertIn("PreservesEntries model beforeEntries afterEntries", prepared.query)
        self.assertIn("F1.actual Value World Error model", prepared.query)
        self.assertIn("F2.actual Value World Error model", prepared.query)
        self.assertEqual(prepared.roots[-1], "merlin_transform")
        self.assertEqual(prepared.schema, "merlin.graph_pipeline_prepared.v3")
        self.assertNotIn("#print", prepared.query)
        self.assertNotIn("#check", prepared.query)
        self.assertTrue(any(ref.path.endswith("lean_graph_input_admission.py") for ref in prepared.sources))
        self.assertTrue(any(ref.path.endswith("xdsl/parser/core.py") for ref in prepared.sources))
        for filename in ("lean_checker.py", "pipeline.py", "pure_scalar.py", "certificates.py",
                         "lean_graph_pipeline_gate_projection.py", "lean_graph_pipeline_gate_emit.py"):
            self.assertTrue(any(Path(ref.path).name == filename for ref in prepared.sources), filename)
        self.assertNotIn("sorry", prepared.query)

    def test_real_gate_is_unavailable_without_native_launch(self):
        report = self.checker.verify_pipeline_change(self.change, toolchain=object())
        self.assertIs(type(report), self.gate.GateReport)
        self.assertEqual(report.status, "unavailable")
        self.assertIn("held", report.reason)
        self.assertFalse(self.checker.accepted_pipeline_receipt(report, self.change))

    def test_actual_pipeline_consumes_mocked_complete_route_only(self):
        with patch.object(self.gate.native, "check", self.mock_kernel), patch.object(
                self.gate.native, "revalidate", return_value=True):
            result = self.pipeline(self.parse(SCALAR), self.options,
                verifier=lambda change: self.checker.verify_pipeline_change(change, toolchain=None))
            self.assertFalse(result.plan_only)
            self.assertEqual(len(result.changes), 1)
        # This is host wiring evidence with a mocked kernel, NOT Lean acceptance.

    def test_saved_copied_wrong_program_and_forged_reports_refused(self):
        with patch.object(self.gate.native, "check", self.mock_kernel), patch.object(
                self.gate.native, "revalidate", return_value=True):
            receipt = self.gate.verify(self.change)
            self.assertIs(type(receipt), self.gate.WholeProgramCompletion)
            self.assertTrue(self.gate.accepted(receipt, self.change))
            for copied in (replace(receipt), {"status": "verified"}, True, None):
                self.assertFalse(self.gate.accepted(copied, self.change))
            for changed in (
                rebound(self.change, after_ir=self.change.before_ir),
                rebound(self.change, before_ir=self.change.after_ir),
                replace(self.change, kind="exact_integer_simplification"),
            ):
                self.assertFalse(self.gate.accepted(receipt, changed))
            for field, value in (("domain", "shape"), ("relation", "local"), ("premises", ()),
                                 ("roots", ("id",)), ("query_sha256", "0" * 64)):
                self.assertFalse(self.gate.accepted(replace(receipt, **{field: value}), self.change))

    def test_hold_blocks_even_previous_live_unit_completion(self):
        with patch.object(self.gate.native, "check", self.mock_kernel):
            receipt = self.gate.verify(self.change)
        self.assertFalse(self.gate.accepted(receipt, self.change))

    def test_skipped_wrong_root_foreign_axiom_wrong_query_no_authority(self):
        for mutation in (
            lambda p: None,
            lambda p: {"status": "verified"},
            lambda p: replace(self.mock_kernel(p), checked_roots=("local",)),
            lambda p: replace(self.mock_kernel(p), prepared_identity="0" * 64),
            lambda p: replace(self.mock_kernel(p), query_sha256="0" * 64),
            lambda p: replace(self.mock_kernel(p), axioms=("Lean.trustCompiler",)),
            lambda p: replace(self.mock_kernel(p), toolchain_identity=""),
        ):
            with self.subTest(mutation=mutation), patch.object(self.gate.native, "check", side_effect=lambda p, **kw: mutation(p)):
                report = self.gate.verify(self.change)
                self.assertEqual(report.status, "unavailable")
                self.assertFalse(self.gate.accepted(report, self.change))

    def test_unexpected_author_identity_valueerror_is_unavailable(self):
        with patch.object(self.gate, "prepare", side_effect=ValueError("selected dependency changed")):
            report = self.gate.verify(self.change)
        self.assertEqual(report.status, "unavailable")

    def test_native_checker_failure_is_not_domain_refusal(self):
        with patch.object(self.gate.native, "check", side_effect=OSError("held checker cannot start")):
            report = self.gate.verify(self.change)
        self.assertEqual(report.status, "unavailable")

    def test_postcheck_changed_projection_refuses_live_completion(self):
        prepared = self.mapping.prepare(self.change)
        changed = replace(prepared, projection=prepared.projection + " ")
        with patch.object(self.gate, "prepare", side_effect=[prepared, changed]), patch.object(
                self.gate.native, "check", self.mock_kernel):
            report = self.gate.verify(self.change)
        self.assertEqual(report.status, "unavailable")

    def test_unsupported_kind_does_not_call_checker(self):
        with patch.object(self.gate.native, "check", side_effect=AssertionError("not reached")):
            report = self.gate.verify(replace(self.change, kind="shape_view_normalization"))
        self.assertEqual(report.status, "unsupported")

    def test_actual_licm_empty_and_nonempty_loop_candidates(self):
        from merlin.transforms.pipeline import PipelineOptions
        for text in (LOOP, LOOP.replace("constant 3 : index", "constant 0 : index")):
            with self.subTest(empty="constant 3" not in text):
                options = PipelineOptions(index_bits=64, normalize_shapes=False,
                    simplify_integers=False, reuse_buffers=False, hoist_invariants=True)
                result = self.pipeline(self.parse(text), options, plan_only=True)
                self.assertEqual(len(result.changes), 1)
                prepared = self.mapping.prepare(result.changes[0])
                self.assertNotEqual(prepared.source, prepared.target)
                self.assertIn("(.loop", prepared.query)
                self.assertIn("common finite abstract SCF", " ".join(prepared.premises))

    def test_wrong_result_and_observer_are_complete_valid_different_queries(self):
        prepared = self.mapping.prepare(self.change)
        wrong_result = self.change.after_ir.replace("arith.muli", "arith.subi")
        wrong_observer = self.change.after_ir.replace("@observe", "@changed_observer").replace(
            'sym_name = "observe"', 'sym_name = "changed_observer"')
        for text in (wrong_result, wrong_observer):
            self.assertNotEqual(text, self.change.after_ir)
            self.parse(text).verify()  # Compiler-valid negative, not malformed text.
            altered = rebound(self.change, after_ir=text)
            if text == wrong_observer:
                # Declaration/context changes are recognized by the host mapper.
                with self.assertRaises(self.mapping.GateUnsupported):
                    self.mapping.prepare(altered)
            else:
                candidate = self.mapping.prepare(altered)
                self.assertNotEqual(candidate.query, prepared.query)
                self.assertNotEqual(candidate.target, prepared.target)
                self.assertEqual(self.gate.verify(altered).status, "unavailable")
                # Independent Lean semantic refusal is pending, not simulated.

    def test_changed_unchanged_function_is_in_query(self):
        after = self.change.after_ir
        modified = after.replace('"func.return"(%a)',
            '%neg = "arith.subi"(%a, %a) : (i8, i8) -> i8\n    "func.return"(%neg)')
        self.assertNotEqual(after, modified)
        self.parse(modified).verify()
        original = self.mapping.prepare(self.change)
        altered = self.mapping.prepare(rebound(self.change, after_ir=modified))
        self.assertNotEqual(original.query, altered.query)

    def test_resource_metadata_refused_before_parser(self):
        from merlin.transforms import certificates
        changed = rebound(self.change, before_ir=self.change.before_ir + '\n{-# dialect_resources: {} #-}')
        with patch.object(certificates, "parse_ir", side_effect=AssertionError("parse not permitted")):
            report = self.gate.verify(changed)
        self.assertEqual(report.status, "unsupported")
        self.assertIn("file-metadata", report.reason)

    def test_indirect_calls_refused_even_when_hidden_in_other_function(self):
        suffix = '''
  func.func @local(%x: i8) -> i8 { func.return %x : i8 }
  func.func @indirect(%x: i8) -> i8 {
    %f = "func.constant"() {value = @local} : () -> ((i8) -> i8)
    %v = "func.call_indirect"(%f, %x) : ((i8) -> i8, i8) -> i8
    func.return %v : i8
  }
'''
        text = SCALAR.rstrip()[:-1] + suffix + "}"
        self.parse(text).verify()
        changed = rebound(self.change, before_ir=text, after_ir=text)
        report = self.gate.verify(changed)
        self.assertEqual(report.status, "unsupported")
        self.assertIn("input admission", report.reason)

    def test_held_default_does_not_return_changed_pipeline(self):
        with self.assertRaisesRegex(ValueError, "whole-program"):
            self.pipeline(self.parse(SCALAR), self.options,
                verifier=lambda change: self.checker.verify_pipeline_change(change, toolchain=None))

    def test_arithmetic_sensitive_witness_candidate_uses_actual_source(self):
        from merlin.verify.lean_graph_pipeline_gate_models import witness
        prepared = self.mapping.prepare(self.change)
        query, roots, metadata = witness(prepared)
        self.assertEqual([r["result"]["values"] for r in metadata["success_proposals"]], [[64], [81]])
        self.assertFalse(metadata["proof_checked"])
        self.assertFalse(metadata["model_approved"])
        self.assertIn("arguments.headD 0", query)
        self.assertIn("source_initialized", query)
        self.assertNotIn("#print", query)
        self.assertNotIn("#check", query)
        self.assertIn("Merlin.GraphPipelineGateWitness.target_success1", roots)

    def test_original_primitive_renderer_extraction_is_exact(self):
        from merlin.verify import lean_graph_pipeline_gate_emit as emitter
        original = Path(emitter.__file__).with_name("lean_graph_state_scalar.py").read_text()
        extracted = Path(emitter.__file__).read_text()
        self.assertEqual(extracted[extracted.index("def integer("):].strip(),
            original[original.index("def integer("):original.index("\ndef sources():")].strip())

    def test_replay_requires_a_fresh_checker_not_saved_json(self):
        from dataclasses import asdict
        prepared = self.mapping.prepare(self.change)
        saved = json.loads(json.dumps(asdict(prepared)))
        report = self.gate.replay(self.change, saved)
        self.assertEqual(report.status, "unavailable")
        self.assertIn("held", report.reason)
        with patch.object(self.gate.native, "check", self.mock_kernel) as checker, patch.object(
                self.gate.native, "revalidate", return_value=True):
            completion = self.gate.replay(self.change, saved)
            self.assertIs(type(completion), self.gate.WholeProgramCompletion)
            self.assertTrue(self.gate.accepted(completion, self.change))

    def test_coherent_rehashed_forgery_does_not_select_queries_or_source_paths(self):
        from dataclasses import asdict
        saved = asdict(self.mapping.prepare(self.change))
        saved["query"] = "theorem merlin_transform : True := True.intro"
        saved["roots"] = ("merlin_transform",)
        saved["sources"] = ({"path": "/.elan/forged.lean", "sha256": "0" * 64, "size": 0},)
        with patch.object(self.gate.native, "check", side_effect=AssertionError("forged work must not launch")):
            report = self.gate.replay(self.change, saved)
        self.assertEqual(report.status, "unavailable")

    def test_pipeline_serialization_is_descriptive_not_issuance(self):
        with patch.object(self.gate.native, "check", self.mock_kernel):
            receipt = self.gate.verify(self.change)
        document = receipt.to_dict()
        self.assertEqual(document["status"], "conditional_model_relation")
        self.assertFalse(document["review_approval"])
        self.assertFalse(self.gate.accepted(document, self.change))

    def test_projection_and_author_failures_have_narrow_classification(self):
        from merlin.verify import lean_graph_state_scalar_capture as capture
        for message, status in (
            ("complete function interface/attributes changed", "unsupported"),
            ("actual producer/parser imports escaped pinned source", "unavailable"),
            ("selected dependency changed while its bytes were observed", "unavailable"),
        ):
            with self.subTest(message=message), patch.object(capture, "project", side_effect=ValueError(message)):
                self.assertEqual(self.gate.verify(self.change).status, status)
        with patch.object(self.mapping, "complete_query", side_effect=ValueError("bad author implementation")):
            self.assertEqual(self.gate.verify(self.change).status, "unavailable")

    def test_live_receipt_stale_mapper_sources_refused(self):
        prepared = self.mapping.prepare(self.change)
        with patch.object(self.gate, "prepare", return_value=prepared), patch.object(
                self.gate.native, "check", self.mock_kernel):
            receipt = self.gate.verify(self.change)
        sources = (replace(prepared.sources[0], sha256="f" * 64), *prepared.sources[1:])
        with patch.object(self.gate, "prepare", return_value=replace(prepared, sources=sources)), patch.object(
                self.gate.native, "revalidate", side_effect=AssertionError("stale source must refuse before runtime")):
            self.assertFalse(self.gate.accepted(receipt, self.change))

    def test_valid_changed_observer_keeps_roster_but_changes_lean_program(self):
        text = SCALAR.replace("func.func private @observe(i8) -> i8",
            "func.func private @observe(i8) -> i8\n  func.func private @other_observer(i8) -> i8")
        change = self.pipeline(self.parse(text), self.options, plan_only=True).changes[0]
        after = change.after_ir.replace("callee = @observe", "callee = @other_observer")
        self.assertNotEqual(after, change.after_ir)
        self.parse(after).verify()
        positive = self.mapping.prepare(change)
        negative = self.mapping.prepare(rebound(change, after_ir=after))
        self.assertNotEqual(positive.query, negative.query)
        with patch.object(self.gate.native, "check", self.mock_kernel):
            receipt = self.gate.verify(change)
        with patch.object(self.gate.native, "revalidate", return_value=True):
            self.assertFalse(self.gate.accepted(receipt, rebound(change, after_ir=after)))

    def test_wrong_scope_never_launches_checker(self):
        from merlin.transforms.certificates import TRANSFORM_SCOPES
        certificate = replace(self.change.certificate, scope=TRANSFORM_SCOPES["shape_view_normalization"])
        with patch.object(self.gate.native, "check", side_effect=AssertionError("unsupported must not launch")):
            report = self.gate.verify(replace(self.change, certificate=certificate))
        self.assertEqual(report.status, "unsupported")

    def test_stale_certificate_digests_refuse_before_checker(self):
        for field in ("before_ir_sha256", "after_ir_sha256"):
            certificate = replace(self.change.certificate, **{field: "f" * 64})
            with self.subTest(field=field), patch.object(self.gate.native, "check", side_effect=AssertionError("stale")):
                report = self.gate.verify(replace(self.change, certificate=certificate))
                self.assertEqual(report.status, "unsupported")

    def test_valid_flag_change_is_not_a_pure_same_program(self):
        after = self.change.after_ir.replace("#arith.overflow<none>", "#arith.overflow<nsw>", 1)
        self.assertNotEqual(after, self.change.after_ir)
        self.parse(after).verify()
        before_projection = json.loads(self.mapping.prepare(self.change).projection)
        after_projection = json.loads(self.mapping.prepare(rebound(self.change, after_ir=after)).projection)
        self.assertNotEqual(before_projection["registry"], after_projection["registry"])
        self.assertNotEqual(before_projection["functions"][1]["after"], after_projection["functions"][1]["after"])

    def test_actual_two_changed_stages_keep_exact_middle_and_complete_queries(self):
        from merlin.transforms.pipeline import PipelineOptions
        text = LOOP.replace("%x = arith.addi %a, %b : i8", """%x = arith.addi %a, %b : i8
      %duplicate = arith.addi %a, %b : i8
      %dead = arith.muli %x, %duplicate : i8""")
        options = PipelineOptions(index_bits=64, normalize_shapes=False,
            simplify_integers=False, reuse_buffers=False, eliminate_redundancy=True, hoist_invariants=True)
        result = self.pipeline(self.parse(text), options, plan_only=True)
        self.assertEqual([c.kind for c in result.changes], ["scalar_redundancy_elimination", "scalar_loop_invariant_motion"])
        first, second = result.changes
        self.assertEqual(first.after_ir, second.before_ir)
        projections = [self.mapping.prepare(c) for c in result.changes]
        self.assertTrue(all(c.source != c.target for c in projections))
        for candidate in projections:
            self.assertIn("PreservesEntries", candidate.query)
        self.assertEqual(json.loads(projections[0].projection)["registry"], json.loads(projections[1].projection)["registry"])
        # This is actual stage/host mapping evidence; composed native acceptance
        # is not inferred from equal registry JSON or equal middle hashes.

    def test_common_context_request_binds_exact_full_root_query_and_sources(self):
        prepared = self.mapping.prepare(self.change)
        request = self.gate.native.request(prepared)
        self.assertEqual(request.query, prepared.query.encode())
        self.assertEqual(request.final_root, "merlin_transform")
        self.assertEqual(request.support_roots, prepared.roots[:-1])
        self.assertEqual(len(request.project_sources), 7)
        self.assertIn("Merlin.GraphStateGateEntries", [r[0] for r in request.project_sources])
        with self.assertRaises(self.mapping.GateUnavailable):
            self.gate.native.request(replace(prepared, roots=("localLemma",)))
        with self.assertRaises(self.mapping.GateUnavailable):
            self.gate.native.request(replace(prepared, sources=()))
        with self.assertRaises(self.mapping.GateUnavailable):
            self.gate.native.request(replace(prepared, schema="merlin.graph_pipeline_prepared.v1"))
        with self.assertRaises(self.mapping.GateUnavailable):
            self.gate.native.request(replace(prepared, schema="merlin.graph_pipeline_prepared.v2"))

    def test_minimal_roster_keeps_exact_frozen_observations_and_evaluator(self):
        directory = Path(self.mapping.__file__).parent / "lean/Merlin"
        old = (directory / "GraphStatePipelineProgram.lean").read_text()
        new = (directory / "GraphStateGateEntries.lean").read_text()
        self.assertEqual([line for line in new.splitlines() if line.startswith("import ")],
                         ["import Merlin.GraphStateScalar"])
        # Source/data correspondence ONLY, not a substitute for compiling Lean.
        old_body = old[old.index("namespace Merlin.GraphState.Pipeline"):]
        old_body = old_body.replace("Merlin.GraphState.Pipeline", "Merlin.GraphState.GateEntries")
        old_body = old_body.replace("  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr\n", "")
        old_body = old_body.replace("(outputs : first.outputs = second.outputs)",
                                    "(_outputs : first.outputs = second.outputs)")
        self.assertEqual(new[new.index("namespace Merlin.GraphState.GateEntries"):], old_body)
        prepared = self.mapping.prepare(self.change)
        sources = {Path(ref.path).stem for ref in prepared.sources if ref.path.endswith(".lean")}
        self.assertEqual(sources, {"GraphStateGateEntries", "GraphStateScalar", "GraphStateScalarContract",
                                  "GraphStateScalarNormalize", "GraphStateScalarSound",
                                  "GraphStateScalarTyping", "GraphStateScalarInitialized"})
        self.assertNotIn("Merlin.GraphState.Pipeline", prepared.query)
        self.assertEqual(prepared.domain, "complete_scalar_roster_equal_outcome_v1")
        self.assertEqual(prepared.relation, "all_defined_entries_equal_returns_errors_terminal_world")

    def test_plan_only_requires_exact_boolean_and_grants_no_receipt(self):
        from merlin.transforms.certificates import ir_text
        for flag in ("false", "true", 0, 1, None, [], {}):
            original = self.parse(SCALAR)
            before = ir_text(original)
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "explicit boolean"):
                self.pipeline(original, self.options, plan_only=flag)
            self.assertEqual(ir_text(original), before)
        original = self.parse(SCALAR)
        before = ir_text(original)
        planned = self.pipeline(original, self.options, plan_only=True)
        self.assertIs(planned.plan_only, True)
        self.assertEqual(ir_text(original), before)
        self.assertNotEqual(ir_text(planned.module), before)
        self.assertFalse(self.checker.accepted_pipeline_receipt(planned, planned.changes[0]))

    def test_mid_chain_rejection_preserves_original_module_and_input_context(self):
        from merlin.transforms.pipeline import PipelineOptions
        from merlin.transforms.certificates import ir_text
        text = LOOP.replace("%x = arith.addi %a, %b : i8", """%x = arith.addi %a, %b : i8
      %duplicate = arith.addi %a, %b : i8
      %dead = arith.muli %x, %duplicate : i8""")
        options = PipelineOptions(index_bits=64, normalize_shapes=False, simplify_integers=False,
            reuse_buffers=False, eliminate_redundancy=True, hoist_invariants=True)
        original = self.parse(text)
        before = ir_text(original)
        calls = []
        def reject_second(change):
            calls.append(change)
            if len(calls) == 1:
                return self.gate.verify(change)
            return self.gate.GateReport("not_established", "mocked later-stage refusal")
        with patch.object(self.gate.native, "check", self.mock_kernel), patch.object(
                self.gate.native, "revalidate", return_value=True):
            with self.assertRaisesRegex(ValueError, "whole-program"):
                self.pipeline(original, options, plan_only=False, verifier=reject_second)
        self.assertEqual([c.kind for c in calls], ["scalar_redundancy_elimination", "scalar_loop_invariant_motion"])
        self.assertEqual(calls[0].after_ir, calls[1].before_ir)
        self.assertEqual(ir_text(original), before)
        original.verify()
        # Neither failed invocation nor its partial clone is returned; callbacks
        # received immutable text changes. External callback side effects are not
        # claimed transactional, and this is host evidence, not composed Lean proof.

    def context_fixture(self):
        from merlin.verify import lean_graph_pipeline_gate_provider as provider
        controller, data, _ = provider.selected()
        Context, CheckedQuery, FileRef = controller.Context, controller.CheckedQuery, data.FileRef
        prepared = self.mapping.prepare(self.change)
        request = self.gate.native.request(prepared)
        context = object.__new__(Context)  # No from_review, filesystem or SDK reads.
        context.context_id = "1" * 64
        context.held = False
        context.workspace = Path("/mock/graph-context")
        context.compile_audit_roots = dict(provider._COMPILE_ROOTS)
        context.review = {role: FileRef(str(Path(provider.__file__).resolve().parents[3] / path), digest, size)
                          for role, (path, digest, size) in provider._REVIEW_PINS.items()}
        context.controls = tuple(FileRef(str(provider._DIRECTORY / (name + ".py")), digest, size)
                                 for name, digest, size in provider._PINS)
        context.originals = {name: FileRef(path, digest, size) for name, path, digest, size in request.project_sources}

        def checked(module, query, *, final_root, support_roots):
            return CheckedQuery(context.context_id, module,
                FileRef("/mock/query.lean", self.mapping.sha(query), len(query)),
                FileRef("/mock/query.olean", "2" * 64, 1), final_root, support_roots,
                tuple((name, ("propext",)) for name in (final_root, *support_roots)),
                FileRef("/mock/audit.json", "3" * 64, 1),
                FileRef("/mock/types.txt", "4" * 64, 1),
                FileRef("/mock/imports.json", "5" * 64, 1), (), object())
        return Context, prepared, context, checked

    def test_inactive_common_adapter_with_mocked_typed_provider(self):
        Context, prepared, context, checked = self.context_fixture()
        with patch.object(Context, "check_query", side_effect=checked), patch.object(
                Context, "revalidate", side_effect=lambda value: value), patch.object(
                Context, "_guard", side_effect=AssertionError("no native guard in SDK-free test")):
            result = self.gate.native._check_with_context(prepared, toolchain=context)
            self.assertEqual(result.prepared_identity, prepared.identity)
            self.assertTrue(self.gate.native._revalidate_with_context(result, toolchain=context))
            with self.assertRaises(self.mapping.GateUnavailable):
                self.gate.native._revalidate_with_context(result, toolchain=object())
        with self.assertRaises(self.mapping.GateUnavailable):
            self.gate.native.check(prepared, toolchain=context)  # Public entry stays held.

    def test_inactive_adapter_wrong_typed_report_refuses(self):
        Context, prepared, context, checked = self.context_fixture()
        from merlin.verify.lean_graph_pipeline_gate_provider import selected
        FileRef = selected()[1].FileRef
        for update in (
            {"final_root": "localLemma"}, {"context_id": "f" * 64}, {"support_roots": ()},
            {"query": FileRef("/mock/query.lean", "f" * 64, 3)}, {"axioms": ()},
        ):
            def wrong(*args, **kwargs):
                return replace(checked(*args, **kwargs), **update)
            with self.subTest(update=update), patch.object(Context, "check_query", side_effect=wrong), patch.object(
                    Context, "revalidate", side_effect=AssertionError("wrong report must not revalidate")):
                with self.assertRaises(self.mapping.GateUnavailable):
                    self.gate.native._check_with_context(prepared, toolchain=context)

    def test_inactive_adapter_refusal_vs_infrastructure_and_deadline(self):
        from merlin.verify.lean_graph_pipeline_gate_provider import selected
        provider = selected()[0]
        ProofRefused, InfrastructureStop = provider.ProofRefused, provider.InfrastructureStop
        Context, prepared, context, _ = self.context_fixture()
        for failure, expected in ((ProofRefused("ordinary diagnostic"), self.mapping.GateNotEstablished),
                                  (InfrastructureStop("binding changed"), self.mapping.GateUnavailable)):
            with self.subTest(failure=failure), patch.object(Context, "check_query", side_effect=failure):
                with self.assertRaises(expected):
                    self.gate.native._check_with_context(prepared, toolchain=context)
        with patch.object(Context, "check_query", side_effect=AssertionError("expired budget must not launch")):
            with self.assertRaises(self.mapping.GateUnavailable):
                self.gate.native._check_with_context(prepared, toolchain=context, deadline=0)
        with patch.object(self.gate.native, "check", side_effect=self.mapping.GateNotEstablished("ordinary diagnostic")):
            self.assertEqual(self.gate.verify(self.change).status, "not_established")

    def test_selected_frozen_provider_refuses_other_class_and_mixed_refs(self):
        from merlin.verify import lean_graph_pipeline_gate_provider as provider
        from merlin.verify.lean_graph_original_image_context import Context as LiveContext
        Context, prepared, context, _ = self.context_fixture()
        self.assertIsNot(Context, LiveContext)
        with self.assertRaises(self.mapping.GateUnavailable):
            provider.bound_context(object.__new__(LiveContext))
        for controls in ((), context.controls[:-1], tuple(reversed(context.controls))):
            with self.subTest(controls=controls), patch.object(context, "controls", controls):
                with self.assertRaises(self.mapping.GateUnavailable):
                    provider.bound_context(context)
        for review in ({}, {k: v for k, v in context.review.items() if k != "finite_review"},
                       {**context.review, "probe": context.review["base_intent"]}):
            with self.subTest(review=review), patch.object(context, "review", review):
                with self.assertRaises(self.mapping.GateUnavailable):
                    provider.bound_context(context)
        closure = {source.path for source in self.mapping.prepare(self.change).sources}
        self.assertTrue(all(str(provider._DIRECTORY / (name + ".py")) in closure
                            for name, _, _ in provider._PINS))

    def test_selected_provider_source_and_module_changes_refuse_before_native(self):
        from merlin.verify import lean_graph_pipeline_gate_provider as provider
        controller, data, _ = provider.selected()
        with patch.object(controller, "data", object()):
            with self.assertRaises(self.mapping.GateUnavailable):
                provider.selected()
        name, digest, size = provider._PINS[0]
        with patch.object(provider, "_PINS", ((name, "0" * 64, size), *provider._PINS[1:])):
            with self.assertRaises(self.mapping.GateUnavailable):
                provider.selected()
        with patch.dict(sys.modules, {controller.__name__: object()}):
            with self.assertRaises(self.mapping.GateUnavailable):
                provider.selected()

    def test_source_only_context_recipe_is_exact_and_cannot_construct_or_launch(self):
        from merlin.verify import lean_graph_pipeline_gate_provider as provider
        Context, prepared, context, _ = self.context_fixture()
        with patch.object(Context, "from_review", side_effect=AssertionError("recipe must not construct")), patch.object(
                Context, "_guard", side_effect=AssertionError("recipe must not inspect runtime")):
            args = provider.context_arguments(prepared, workspace="/mock/fresh-gate-workspace")
        self.assertEqual(args["project_sources"], context.originals)
        self.assertEqual(args["controller_sources"], context.controls)
        self.assertEqual(args["review"], {k: v for k, v in context.review.items()
                                         if k in {"metadata", "copy", "probe", "trace"}})
        self.assertEqual(sum(map(len, args["compile_audit_roots"].values())), 11)
        self.assertTrue(all(type(roots) is tuple and roots for roots in args["compile_audit_roots"].values()))
        self.assertEqual(set(args["compile_audit_roots"]), {
            "Merlin.GraphStateScalar", "Merlin.GraphStateScalarInitialized",
            "Merlin.GraphStateScalarSound", "Merlin.GraphStateScalarTyping"})
        with self.assertRaises(self.mapping.GateUnavailable):
            provider.context_arguments(replace(prepared, sources=prepared.sources + (
                self.mapping.Source("/mock/Merlin/Extra.lean", "0" * 64, 1),)), workspace="/mock/fresh")
        with patch.object(context, "originals", {**context.originals, "Merlin.Extra": object()}):
            with self.assertRaises(self.mapping.GateUnavailable):
                self.gate.native._check_with_context(prepared, toolchain=context)

    def test_actual_enforced_pipeline_through_inactive_frozen_adapter_mock_only(self):
        Context, prepared, context, checked = self.context_fixture()
        calls = []
        def observe(*args, **kwargs):
            calls.append((args, kwargs))
            return checked(*args, **kwargs)
        # ONLY unit wiring: public entry remains held outside this scoped patch.
        # No Context constructor, runtime guard or subprocess may execute.
        with patch.object(Context, "check_query", side_effect=observe), patch.object(
                Context, "revalidate", side_effect=lambda value: value), patch.object(
                Context, "_guard", side_effect=AssertionError("no runtime reads")), patch.object(
                self.gate.native, "check", self.gate.native._check_with_context), patch.object(
                self.gate.native, "revalidate", self.gate.native._revalidate_with_context):
            receipts = []
            def verifier(change):
                receipt = self.checker.verify_pipeline_change(change, toolchain=context)
                receipts.append(receipt)
                return receipt
            result = self.pipeline(self.parse(SCALAR), self.options, plan_only=False, verifier=verifier)
            self.assertFalse(result.plan_only)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result.changes[0].after_ir, self.change.after_ir)
            self.assertTrue(self.checker.accepted_pipeline_receipt(receipts[0], result.changes[0]))
            self.assertFalse(self.checker.accepted_pipeline_receipt(receipts[0].to_dict(), result.changes[0]))
            self.assertFalse(self.checker.accepted_pipeline_receipt(replace(receipts[0]), result.changes[0]))
            changed = rebound(result.changes[0], after_ir=result.changes[0].after_ir.replace("arith.muli", "arith.subi"))
            self.assertFalse(self.checker.accepted_pipeline_receipt(receipts[0], changed))
        self.assertFalse(self.checker.accepted_pipeline_receipt(receipts[0], result.changes[0]))


class ServiceBoundary(unittest.TestCase):
    """Actual service, gate and pipeline; release custody/native methods mocked.

    The graph binding factory and its source/typed-context checks run normally.
    No real bundle snapshot verification or runtime checking is claimed here.
    """

    def setUp(self):
        import contextlib
        import tempfile
        from merlin.targetgen import common_transforms as service
        from merlin.verify import lean_graph_pipeline_gate_service as selection
        self.case = WholeBoundary()
        self.case.setUp()
        self.Context, _, self.context, self.checked = self.case.context_fixture()
        self.service_module, self.selection = service, selection
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.manifest = directory / "host-bundle.yaml"
        self.manifest.write_text("{}\n")
        self.workspace = directory / "workspace"
        self.profile = {"sha256": "a" * 64, "host_index_bits": 64,
                        "common_substrate": "merlin.common_compiler_substrate.v2",
                        "common_sources": [], "automatic_checker_sources": []}
        self.stack.enter_context(patch.object(service, "verify_tool_profile", return_value=self.profile))
        self.stack.enter_context(patch.object(service.BW, "verify_bundle_snapshot", return_value={"content_sha256": "b" * 64}))
        self.stack.enter_context(patch.object(service.BW, "snapshot_marker_digest", return_value="c" * 64))
        self.stack.enter_context(patch.object(service.BW, "verified_snapshot_scope", side_effect=lambda *a, **k: contextlib.nullcontext()))
        self.stack.enter_context(patch.object(service.CommonTransformService, "_toolchain", side_effect=AssertionError("no legacy SDK fallback")))
        self.stack.enter_context(patch.object(self.Context, "from_review", side_effect=AssertionError("no service context construction")))
        self.stack.enter_context(patch.object(self.Context, "_guard", side_effect=AssertionError("no runtime reads")))
        self.binding = selection.bind_service_context(self.context, compiler_profile_sha256=self.profile["sha256"], timeout_s=120)
        self.service = self.make_service(self.binding)
        self.request = {"schema": service.SCHEMA, "source_ir": SCALAR, "options": {
            "normalize_shapes": False, "simplify_integers": False, "reuse_buffers": False,
            "eliminate_redundancy": True}}

    def make_service(self, binding):
        return self.service_module.CommonTransformService(workspace=self.workspace, bundle_manifest=self.manifest,
            repo=Path(self.case.mapping.__file__).resolve().parents[3], graph_binding=binding)

    def activate_mock_methods(self, check=None):
        self.stack.enter_context(patch.object(self.Context, "check_query", side_effect=check or self.checked))
        self.stack.enter_context(patch.object(self.Context, "revalidate", side_effect=lambda value: value))
        self.stack.enter_context(patch.object(self.case.gate.native, "check", self.case.gate.native._check_with_context))
        self.stack.enter_context(patch.object(self.case.gate.native, "revalidate", self.case.gate.native._revalidate_with_context))

    def test_actual_service_gate_and_live_compiler_eligibility_mocked(self):
        import time
        self.activate_mock_methods()
        document = self.service.apply(self.request, work_root=self.context.workspace)
        self.assertEqual(document["status"], "conditional_model_relation")
        self.assertNotEqual(document["transformed_ir"], self.request["source_ir"])
        self.assertEqual(len(document["receipts"]), 1)
        self.assertEqual(document["compiler_pass_chain"], "not_established")
        self.assertTrue(self.service.compiler_eligible(document, source_ir=SCALAR))
        self.assertFalse(self.service.compiler_eligible(json.loads(json.dumps(document)), source_ir=SCALAR))
        self.assertFalse(self.service.compiler_eligible(document, source_ir=SCALAR + "\n"))
        recorded = self.selection._DELIVERIES[id(document)]
        with self.assertRaises(ValueError):
            self.selection.record_delivery(self.binding, dict(document), SCALAR.replace("arith.addi", "arith.subi", 1),
                recorded[4], recorded[5], deadline=time.monotonic() + 60)
        with self.assertRaises(TimeoutError):
            self.selection.record_delivery(self.binding, dict(document), SCALAR, recorded[4], recorded[5], deadline=0)
        document["transformed_ir"] += "\n"
        self.assertFalse(self.service.compiler_eligible(document, source_ir=SCALAR))

    def test_service_refuses_missing_saved_legacy_profile_and_unsupported_options(self):
        for value in (self.binding.describe(), replace(self.binding), object()):
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                self.make_service(value)
        missing = self.make_service(None)
        with self.assertRaises(ValueError):
            missing.apply(self.request, work_root=self.context.workspace)
        for changes in ({"normalize_shapes": True}, {"unroll_factors": [2]}, {"plan_only": True}):
            request = {**self.request, "options": {**self.request["options"], **changes}}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.service.apply(request, work_root=self.context.workspace)
        with self.assertRaises(ValueError):
            self.service.apply({**self.request, "graph_binding": self.binding.describe()}, work_root=self.context.workspace)
        with self.assertRaises(ValueError):
            self.service.apply(self.request, work_root=Path("/mock/other-context"))
        with self.assertRaisesRegex(ValueError, "no legacy fallback"):
            self.service.check({"schema": "merlin.common_check_request.v1", "before_ir": SCALAR,
                "after_ir": SCALAR, "entry": "main"}, work_root=self.context.workspace)

    def test_service_native_hold_and_new_context_hold_cannot_grant_eligibility(self):
        with self.assertRaisesRegex(ValueError, "held"):
            self.service.apply(self.request, work_root=self.context.workspace)
        self.activate_mock_methods()
        document = self.service.apply(self.request, work_root=self.context.workspace)
        self.context.held = True
        self.assertFalse(self.service.compiler_eligible(document, source_ir=SCALAR))
        with self.assertRaises(ValueError):
            self.service.apply(self.request, work_root=self.context.workspace)

    def test_service_second_boundary_refusal_has_no_deliverable(self):
        calls = []
        provider = __import__("merlin.verify.lean_graph_pipeline_gate_provider", fromlist=["selected"])
        def checked(*args, **kwargs):
            calls.append(args[0])
            if len(calls) == 2:
                raise provider.selected()[0].ProofRefused("mocked second-boundary refusal")
            return self.checked(*args, **kwargs)
        self.activate_mock_methods(checked)
        source = LOOP.replace("%x = arith.addi %a, %b : i8", """%x = arith.addi %a, %b : i8
      %duplicate = arith.addi %a, %b : i8
      %dead = arith.muli %x, %duplicate : i8""")
        request = {**self.request, "source_ir": source,
                   "options": {**self.request["options"], "hoist_invariants": True}}
        count = len(self.selection._DELIVERIES)
        with self.assertRaisesRegex(ValueError, "second-boundary"):
            self.service.apply(request, work_root=self.context.workspace)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.selection._DELIVERIES), count)
        self.assertEqual(request["source_ir"], source)


def retain_examples(destination):
    """Retain actual producer bytes and uncompiled queries; never a receipt seal."""
    from dataclasses import asdict
    from merlin.verify.lean_graph_pipeline_gate_models import witness
    case = WholeBoundary()
    case.setUp()
    destination.mkdir(parents=True, exist_ok=False)
    change = case.change
    prepared = case.mapping.prepare(change)
    for name, text in (("input.mlir", SCALAR), ("before.mlir", change.before_ir),
                       ("after.mlir", change.after_ir), ("Check.lean", prepared.query)):
        (destination / name).write_text(text)
    (destination / "projection.json").write_text(prepared.projection + "\n")
    (destination / "PREPARED.json").write_text(json.dumps(asdict(prepared), indent=2) + "\n")
    query, roots, model = witness(prepared)
    (destination / "ArithmeticWitness.lean").write_text(query)
    (destination / "WITNESS.json").write_text(json.dumps({**model, "roots": roots}, indent=2) + "\n")
    wrong = change.after_ir.replace("arith.muli", "arith.subi")
    case.parse(wrong).verify()
    wrong_prepared = case.mapping.prepare(rebound(change, after_ir=wrong))
    (destination / "wrong-result.mlir").write_text(wrong)
    (destination / "WrongResult.lean").write_text(wrong_prepared.query)
    status = case.gate.verify(change).to_dict()
    (destination / "GATE-OUTPUT.json").write_text(json.dumps(status, indent=2) + "\n")
    (destination / "STATUS.json").write_text(json.dumps({
        "producer": "corrective4499 planning APIs; complete actual changed CSE boundary",
        "actual_parser_verify": True, "query_state": "UNCOMPILED", "qualification": "NONE",
        "corrected_producer_source": "PREPARED.json sources; not historical4499 qualification",
        "gate_status": status, "wrong_result": "compiler-valid; changed independent after query; Lean rejection PENDING",
        "mocked_pipeline_tests": "separate unit evidence, not real Lean acceptance",
    }, indent=2) + "\n")
    observer = destination / "changed-observer"
    observer.mkdir()
    source = SCALAR.replace("func.func private @observe(i8) -> i8",
        "func.func private @observe(i8) -> i8\n  func.func private @other_observer(i8) -> i8")
    observed = case.pipeline(case.parse(source), case.options, plan_only=True).changes[0]
    wrong = observed.after_ir.replace("callee = @observe", "callee = @other_observer")
    case.parse(wrong).verify()
    good, bad = case.mapping.prepare(observed), case.mapping.prepare(rebound(observed, after_ir=wrong))
    for name, text in (("before.mlir", observed.before_ir), ("correct-after.mlir", observed.after_ir),
                       ("wrong-after.mlir", wrong), ("Positive.lean", good.query), ("Negative.lean", bad.query)):
        (observer / name).write_text(text)
    (observer / "STATUS.json").write_text(json.dumps({
        "state": "UNCOMPILED_VALID_NEGATIVE", "same_roster": True, "actual_parser_verify": True,
        "changed_query": good.query != bad.query, "kernel_refusal_observed": False,
        "positive_prepared_identity": good.identity, "negative_prepared_identity": bad.identity,
    }, indent=2) + "\n")


if __name__ == "__main__":
    primary, producer = (Path(value).resolve() for value in sys.argv[1:3])
    isolated_owners(primary, producer)
    args = sys.argv[3:]
    if args and args[0] == "--retain-example":
        retain_examples(Path(args[1]).resolve())
        args = args[2:]
    unittest.main(argv=[sys.argv[0], *args])
