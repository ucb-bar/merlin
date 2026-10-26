"""SDK-free original-image controller tests; no native invocation authority."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if not os.environ.get("MERLIN_IMAGE_CONTEXT_DATA_TEST"):
    raise unittest.SkipTest("requires the isolated SDK-free context data runner")

import lean_graph_original_image_context as controller
import lean_graph_original_image_context_data as data
import lean_graph_original_image_context_io as bound_io

ROOT = Path(os.environ["MERLIN_IMAGE_CONTEXT_PRIMARY"])
REVIEW = (
    ROOT / "out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/runtime-snapshot-review"
)
BASE = ROOT / "out/build/graph-paper-deadline/original-runtime-candidate-v1"
PINS = {
    "metadata": data.FileRef(
        str(REVIEW / "METADATA-REVIEW-v2.json"),
        "8c4a1566fac6ecba764c91e8179b5732d7ccda1a5e9a6f58559d7a130770cd82",
        3017778,
    ),
    "copy": data.FileRef(
        str(BASE / "RESULT.json"), "849060513fb300270eb75fd79518cc187325b4f8472105b34570faeaaf85a7ce", 3531848
    ),
    "probe": data.FileRef(
        str(BASE / "basic-kernel-probe-v1/INTENT.json"),
        "2682ee9260af211714d48a3b41206a0464a5fd18d6495ba071386f87eea67a8b",
        145093,
    ),
    "trace": data.FileRef(
        str(BASE / "basic-kernel-probe-v1/trace_parser.py"),
        "2aaae713ed80a46735434eca16dbb6fd2ed63e4bd8e7df50308f64ac56d0b7ba",
        9118,
    ),
}
DOCUMENTS = {name: bound_io.strict_json(bound_io.read_bound(PINS[name])) for name in ("metadata", "copy", "probe")}
FINITE_REVIEW_PIN = data.FileRef(
    str(REVIEW.parent / "finite-stdlib-diagnostics-v1/REVIEW.json"),
    "200da90f0ca766a78eca5d18a3b4fd315ac733f396017b20324a71d17ebae16b", 392060,
)
FINITE_REVIEW = bound_io.strict_json(bound_io.read_bound(FINITE_REVIEW_PIN))


def reviewed_intent():
    return {
        "schema": "merlin.original-image.reviewed-runtime-intent.v2",
        "runtime": copy.deepcopy(DOCUMENTS["probe"]["runtime"]),
        "finite_allowed_misses": copy.deepcopy(FINITE_REVIEW["proposed_finite_union"]),
        "success_only_misses": copy.deepcopy(FINITE_REVIEW["proposed_success_only_pairs"]),
        "base_intent": PINS["probe"].record(), "finite_review": FINITE_REVIEW_PIN.record(),
    }


def reviewed_selection(intent=None, review=None):
    return data.select(
        DOCUMENTS["metadata"], DOCUMENTS["copy"], reviewed_intent() if intent is None else intent,
        base_probe=DOCUMENTS["probe"], finite_review=FINITE_REVIEW if review is None else review,
    )


def selected(documents=None):
    values = documents or DOCUMENTS
    return data.select(values["metadata"], values["copy"], values["probe"])


def source_ref(path):
    content = path.read_bytes()
    return data.FileRef(str(path), hashlib.sha256(content).hexdigest(), len(content))


class ContextDataTests(unittest.TestCase):
    def test_reviewed_finite_descriptor_exact_sets_and_exit_policy(self):
        with mock.patch("os.open", side_effect=AssertionError("data selection opened SDK")):
            selection = reviewed_selection()
            legacy = selected()
        self.assertEqual(selection.images, legacy.images)
        self.assertEqual(selection.runtime, legacy.runtime)
        self.assertEqual(len(selection.misses), 899)
        self.assertEqual(len(selection.success_only), 181)
        self.assertEqual(len(selection.allowed_misses(0)), 899)
        for rc in (1, 2, -11):
            self.assertEqual(selection.allowed_misses(rc), legacy.allowed_misses(rc))
            self.assertEqual(len(selection.allowed_misses(rc)), 718)
        self.assertEqual(len(legacy.misses), 719)
        self.assertEqual(len(legacy.success_only), 1)
        with self.assertRaises(data.ContextRefusal):
            selection.allowed_misses(False)

    def test_reviewed_descriptor_mutations_refused(self):
        changes = {
            "missing_miss": lambda d: d["finite_allowed_misses"].pop(),
            "unknown_miss": lambda d: d["finite_allowed_misses"][0].__setitem__(1, "unknown_exact_helper"),
            "wildcard": lambda d: d["finite_allowed_misses"][0].__setitem__(1, "l_Lean_*"),
            "wrong_owner": lambda d: d["finite_allowed_misses"][0].__setitem__(0, "/other/library"),
            "duplicate": lambda d: d["finite_allowed_misses"].append(d["finite_allowed_misses"][0]),
            "lost_success_guard": lambda d: d["success_only_misses"].pop(),
            "unknown_success_guard": lambda d: d["success_only_misses"][0].__setitem__(1, "unknown_exact_helper"),
            "new_runtime_pin": lambda d: d["runtime"][0].update(sha256="0" * 64),
            "missing_runtime": lambda d: d["runtime"].pop(),
            "extra_field": lambda d: d.update(approved=True),
            "wrong_base_reference": lambda d: d["base_intent"].update(path="relative"),
            "wrong_review_reference": lambda d: d["finite_review"].update(bytes=True),
        }
        for name, change in changes.items():
            row = reviewed_intent()
            change(row)
            with self.subTest(name=name), self.assertRaises(data.ContextRefusal):
                reviewed_selection(row)

    def test_reviewed_provenance_correspondence_mutations_refused(self):
        changes = {
            "baseline": lambda r: r["baseline_exact_pairs"].pop(),
            "addition": lambda r: r["additional_exact_pairs"].pop(),
            "union": lambda r: r["proposed_finite_union"].pop(),
            "success": lambda r: r["proposed_success_only_pairs"].pop(),
            "owner": lambda r: r["original_runtime_owner"].update(sha256="0" * 64),
            "schema": lambda r: r.update(schema="unknown"),
        }
        for name, change in changes.items():
            row = copy.deepcopy(FINITE_REVIEW)
            change(row)
            with self.subTest(name=name), self.assertRaises(data.ContextRefusal):
                reviewed_selection(review=row)
        with self.assertRaises(data.ContextRefusal):
            data.select(DOCUMENTS["metadata"], DOCUMENTS["copy"], reviewed_intent())

    def test_reviewed_prepare_binds_base_and_review_without_native(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "intent.json"
            path.write_text(json.dumps(reviewed_intent()))
            controls = [source_ref(Path(module.__file__).absolute()) for module in (controller, data, bound_io)]
            read = bound_io.read_bound
            def no_image(ref, forbidden=()):
                self.assertNotIn("/toolchain/", ref.path)
                self.assertFalse(ref.path.startswith("/usr/lib/"))
                return read(ref, forbidden)
            with (
                mock.patch.object(bound_io, "read_bound", side_effect=no_image),
                mock.patch("subprocess.Popen", side_effect=AssertionError("native invocation")),
            ):
                instance = controller.Context.from_review(
                    review={**PINS, "probe": source_ref(path)}, project_sources={},
                    controller_sources=controls, workspace=Path(temporary) / "context",
                )
            self.assertEqual(instance.review["base_intent"], PINS["probe"])
            self.assertEqual(instance.review["finite_review"], FINITE_REVIEW_PIN)
            self.assertEqual(len(instance.definition["selection"]["success_only"]), 181)
            self.assertEqual(instance.definition["review"]["finite_review"], FINITE_REVIEW_PIN.record())

    def test_stale_review_provenance_stops_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            intent = reviewed_intent()
            intent["finite_review"]["sha256"] = "0" * 64
            path = Path(temporary) / "intent.json"
            path.write_text(json.dumps(intent))
            controls = [source_ref(Path(module.__file__).absolute()) for module in (controller, data, bound_io)]
            with self.assertRaises(bound_io.BindingError):
                controller.Context.from_review(
                    review={**PINS, "probe": source_ref(path)}, project_sources={},
                    controller_sources=controls, workspace=Path(temporary) / "context",
                )
            self.assertFalse((Path(temporary) / "context").exists())

    def test_full_retained_stdlib_trace_under_exact_exit_zero_policy(self):
        namespace = {"__name__": "_frozen_finite_trace"}
        exec(compile(bound_io.read_bound(PINS["trace"]), PINS["trace"].path, "exec"), namespace)
        raw = bound_io.read_bound(data.FileRef.parse(FINITE_REVIEW["inputs"]["commands/00001-compile/stderr.txt"]))
        process = bound_io.strict_json(bound_io.read_bound(
            data.FileRef.parse(FINITE_REVIEW["inputs"]["commands/00001-compile/PROCESS.json"])
        ))
        plan = reviewed_selection()
        args = ([ref.record() for ref in plan.runtime], plan.executable, plan.interpreter)
        self.assertEqual(process["returncode"], 0)
        observed = namespace["parse"](raw.decode(), *args, plan.allowed_misses(0), expected_pid=process["pid"])
        self.assertEqual(len(observed["init_files"]), 10)
        self.assertFalse(observed["dynamic_loads_observed"])
        for policy in (plan.allowed_misses(1), selected().allowed_misses(0)):
            with self.assertRaises(namespace["TraceRefusal"]):
                namespace["parse"](raw.decode(), *args, policy, expected_pid=process["pid"])
        exact = FINITE_REVIEW["additional_exact_pairs"][0][1]
        self.assertIn(exact, raw.decode())
        for bad in (
            raw.decode().replace(exact, exact + "_UNREVIEWED", 1),
            raw.decode().replace("[0]", "[1]", 1),
            raw.decode() + "unknown runtime error\n",
        ):
            with self.assertRaises(namespace["TraceRefusal"]):
                namespace["parse"](bad, *args, plan.allowed_misses(0), expected_pid=process["pid"])

    def test_literal_header_nested_comments_and_duplicates(self):
        raw = b'''/- import Fake /- nested -/ end -/
import Init -- ignored import Hidden
import /- module comment -/ Lean.Elab.Tactic.Omega
import Merlin.Example
/-! Body documentation -/
private def message := "import Never"
'''
        self.assertEqual(data.literal_imports(raw), ("Init", "Lean.Elab.Tactic.Omega", "Merlin.Example"))

    def test_literal_header_prelude_and_body_commands(self):
        cases = {
            b"prelude\nimport Init\ntheorem t : True := True.intro": ("Init",),
            b"prelude\n": (),
            b"-- no explicit imports\ntheorem t : True := True.intro": ("Init",),
            b"import Merlin.Example\n#print axioms example": ("Init", "Merlin.Example"),
            b"import Init\n@[simp] theorem t : True := True.intro": ("Init",),
            b"prelude\r\nimport Init\r\n": ("Init",),
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(data.literal_imports(raw), expected)

    def test_literal_header_unsupported_and_malformed_refused(self):
        for raw in (
            b"import", b"import Foo Bar\n", b"import Foo;\n", b"import Foo/Other\n",
            b"import Foo..Bar\n", b"import \xc2\xabFoo\xc2\xbb\n", b"/- unclosed",
            b"import Foo\n/- /- nested -/", b"prelude\nprelude\n", b"import Init\nprelude\n",
            b"public import Foo\n", b"private import Foo\n", b"meta import Foo\n",
            b"module\nimport Foo\n", b"import public Foo\n", b"\xef\xbb\xbfimport Init\n",
            b"import Foo\x00\n", b"\xff",
        ):
            with self.subTest(raw=raw), self.assertRaises(data.ContextRefusal):
                data.literal_imports(raw)

    def test_chained_header_modifiers_refused_even_for_implicit_init(self):
        for raw in (
            b"public meta import Init\n", b"private public import Init\n",
            b"meta /- nested /- comment -/ -/ public import Init\n",
            b"public\n-- interposed comment\nmeta prelude\n",
        ):
            with self.subTest(raw=raw), self.assertRaises(data.ContextRefusal):
                data.literal_imports(raw)
        self.assertEqual(data.literal_imports(b"public meta def identity (n : Nat) := n\n"), ("Init",))

    def _mock_inspection(self, raw, report):
        instance = controller.Context()
        instance.selection, instance.sources, instance.objects = selected(), {}, {}
        instance.definition = {"import_policy": "trusted-closed-project-header-and-actual-object-report-v1"}
        project = data.FileRef("/private/source/Merlin/Dependency.lean", "0" * 64, 0)
        instance.sources = {"Merlin.Dependency": project}
        events = []

        def build(module):
            events.append(("build", module))
            instance.objects[module] = {
                ".olean": data.FileRef("/private/compiled/" + module.replace(".", "/") + ".olean", "0" * 64, 0)
            }

        def invoke(label, arguments):
            events.append((label, arguments))
            return report(instance), Path("/private/commands/0")

        source = data.FileRef("/private/source/Query.lean", "0" * 64, len(raw))
        with (
            mock.patch.object(bound_io, "read_bound", return_value=raw),
            mock.patch.object(instance, "_invoke", side_effect=invoke),
            mock.patch.object(instance, "_build", side_effect=build),
            mock.patch("os.open", side_effect=AssertionError("inspection opened SDK/source file")),
        ):
            result = instance._inspect(source)
        return result, events

    def test_header_build_order_and_actual_compiled_only_agreement(self):
        raw = b"import Merlin.Dependency\nimport Lean.Elab.Tactic.Omega\n"
        def report(instance):
            sdk = instance.selection.toolchain + "/lib/lean/"
            return sdk + "Init.olean\n" + sdk + "Lean/Elab/Tactic/Omega.olean\n" + (
                instance.objects["Merlin.Dependency"][".olean"].path + "\n"
            )
        result, events = self._mock_inspection(raw, report)
        self.assertEqual(events[0], ("build", "Merlin.Dependency"))
        self.assertEqual(events[1][0], "object-imports")
        self.assertEqual(events[1][1][0], "--deps")
        self.assertEqual(result["standard_modules"], ["Init", "Lean.Elab.Tactic.Omega"])
        self.assertNotIn("source_imports", result)
        self.assertEqual(result["host_decoded_modules"], ["Init", "Merlin.Dependency", "Lean.Elab.Tactic.Omega"])

    def test_header_object_missing_extra_and_wrong_module_refused(self):
        for modules in ((), ("Init",), ("Init", "Lean"), ("Init", "Lean.Elab.Tactic.Omega", "Lean")):
            def report(instance, modules=modules):
                return "\n".join(
                    instance.selection.toolchain + "/lib/lean/" + name.replace(".", "/") + ".olean"
                    for name in modules
                )
            with self.subTest(modules=modules), self.assertRaises(data.ContextRefusal):
                self._mock_inspection(b"import Lean.Elab.Tactic.Omega\n", report)

    def test_header_unknown_project_missing_family_refuses_before_command(self):
        def no_command(instance):
            self.fail("unbound import reached native command")
        for raw in (b"import Merlin.Missing\n", b"import Lean.NotSelected\n"):
            with self.subTest(raw=raw), self.assertRaises(data.ContextRefusal):
                self._mock_inspection(raw, no_command)

    def test_prelude_empty_agrees_only_with_empty_actual_report(self):
        result, events = self._mock_inspection(b"prelude\n", lambda instance: "")
        self.assertEqual(result["host_decoded_modules"], [])
        self.assertEqual(len(events), 1)
        with self.assertRaises(data.ContextRefusal):
            self._mock_inspection(b"prelude\n", lambda instance: instance.selection.toolchain + "/lib/lean/Init.olean")

    def test_q_failed_source_headers_recomputed_without_runtime_read(self):
        path = ROOT / "out/build/graph-paper-deadline/q-gate-context-v3/CONTEXT.json"
        context = json.loads(path.read_text())
        observed = {}
        for module, row in context["project_sources"].items():
            self.assertNotIn("/toolchain/", row["path"])
            observed[module] = data.literal_imports(bound_io.read_bound(data.FileRef.parse(row)))
        self.assertIn("Lean.Elab.Tactic.Omega", observed["Merlin.ResourceGeometry"])

    def test_original_full_membership_without_image_reads(self):
        with mock.patch("os.open", side_effect=AssertionError("selection opened a file")):
            result = selected()
        self.assertEqual(len(result.images), 4527)
        self.assertEqual(len(result.runtime), 11)
        self.assertEqual(len(result.misses), 719)
        self.assertFalse(any(ref.path.startswith(result.forbidden_roots) for ref in result.images))

    def test_invalid_exact_refs(self):
        base = {"path": "/tmp/input", "sha256": "0" * 64, "bytes": 0}
        for key, value in (
            ("bytes", True),
            ("bytes", -1),
            ("sha256", "A" * 64),
            ("path", "/tmp/../input"),
            ("path", "relative"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(data.ContextRefusal):
                data.FileRef.parse({**base, key: value})
        with self.assertRaises(data.ContextRefusal):
            data.FileRef.parse({**base, "approved": True})

    def test_json_duplicate_nonfinite_refusal(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(text=text), self.assertRaises(data.ContextRefusal):
                bound_io.strict_json(text)

    def test_regular_file_binding_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.txt"
            path.write_bytes(b"original")
            pin = source_ref(path)
            self.assertEqual(bound_io.read_bound(pin), b"original")
            path.write_bytes(b"modified")
            with self.assertRaises(bound_io.BindingError) as error:
                bound_io.read_bound(pin)
            self.assertEqual(error.exception.evidence["observed_sha256"], hashlib.sha256(b"modified").hexdigest())

    def test_symlink_file_and_parent_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            path = real / "input"
            path.write_bytes(b"bound")
            (root / "alias").symlink_to(real, target_is_directory=True)
            (root / "file-alias").symlink_to(path)
            original = source_ref(path)
            for candidate in (root / "alias/input", root / "file-alias"):
                ref = data.FileRef(str(candidate), original.sha256, original.bytes)
                with self.subTest(path=str(candidate)), self.assertRaises(bound_io.BindingError):
                    bound_io.read_bound(ref)

    def test_held_paths_refused_before_open(self):
        for path in ("/home/person/.elan/toolchains/runtime/bin/lean", "/archive/fresh-environment/toolchain/bin/lean"):
            with mock.patch("os.open", side_effect=AssertionError("held path opened")):
                with self.assertRaises(bound_io.BindingError):
                    bound_io.read_bound(data.FileRef(path, "0" * 64, 0))

    def test_refuse_expectation_learning_outside_workspace(self):
        with self.assertRaises(data.ContextRefusal):
            bound_io.output_reference("/installed/runtime", "/tmp/private")

    def test_proof_diagnostics_are_not_generic_exit_errors(self):
        self.assertTrue(controller.diagnostic_failure("/p/Test.lean:4:2: error: type mismatch\n", {"/p/Test.lean"}))
        for text in ("Segmentation fault", "lean: unknown option", "/other/Test.lean:4:2: error: mismatch"):
            self.assertFalse(controller.diagnostic_failure(text, {"/p/Test.lean"}))

    def test_reviewed_runtime_resource_and_mixed_diagnostics_stop(self):
        prefix = "/bound/Query.lean:3:0: error: "
        proof = prefix + "type mismatch\n"
        cases = [
            "(interpreter) unknown declaration",
            "failed to load shared library",
            "maximum recursion depth has been reached",
            "maximum number of heartbeats exceeded",
            "out of memory",
            "unknown error",
            "type mismatch; out of memory",
        ]
        for message in cases:
            for raw in (prefix + message + "\n", proof + prefix + message + "\n", prefix + message + "\n" + proof):
                with self.subTest(raw=raw):
                    self.assertFalse(controller.diagnostic_failure(raw, {"/bound/Query.lean"}))
        self.assertFalse(controller.diagnostic_failure(proof + "  arbitrary continuation\n", {"/bound/Query.lean"}))
        self.assertTrue(controller.diagnostic_failure(proof, {"/bound/Query.lean"}))

    def test_json_cannot_mint_process_result(self):
        instance = controller.Context()
        instance._issued, instance._key, instance.context_id = {}, object(), "selected"
        with mock.patch.object(instance, "_guard", side_effect=AssertionError("forged receipt reached guard")):
            with self.assertRaises(data.ContextRefusal):
                instance.revalidate({"context_id": "selected", "accepted": True})

    def test_prepare_reads_sources_not_runtime_and_does_not_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "Example.lean"
            source.write_text("theorem example_identity (n : Nat) : n = n := rfl\n")
            controls = [source_ref(Path(module.__file__).absolute()) for module in (controller, data, bound_io)]
            real_read = bound_io.read_bound

            def allowed(ref, forbidden=()):
                self.assertNotIn("/toolchain/", ref.path)
                self.assertFalse(ref.path.startswith("/usr/lib/"))
                return real_read(ref, forbidden)

            with (
                mock.patch.object(bound_io, "read_bound", side_effect=allowed),
                mock.patch("subprocess.Popen", side_effect=AssertionError("native process launched")),
            ):
                instance = controller.Context.from_review(
                    review=PINS,
                    project_sources={"Merlin.Example": source_ref(source)},
                    controller_sources=controls,
                    workspace=Path(temporary) / "context",
                )
            self.assertEqual(instance.context_id, data.identity(instance.definition))
            self.assertEqual(len(instance.sources), 1)
            self.assertFalse(instance.objects)

    def test_compiled_only_standard_names_are_not_source_evidence(self):
        instance = controller.Context()
        instance.selection, instance.sources, instance.objects = selected(), {}, {}
        source = instance.selection.toolchain + "/src/lean/Lean/Elab/Tactic/Omega.lean"
        with mock.patch("os.open", side_effect=AssertionError("report cross-check opened standard source")):
            projects, libraries, rows = instance._reported(source + "\n", "source")
        self.assertEqual(projects, set())
        self.assertEqual(libraries, {"Lean.Elab.Tactic.Omega"})
        self.assertEqual(rows[0]["role"], "compiled-only-standard-name-cross-check")
        for raw in (source.replace("Omega.lean", "Unselected.lean"), "/installed/src/lean/Init.lean"):
            with self.assertRaises(data.ContextRefusal):
                instance._reported(raw + "\n", "source")

    def test_query_cannot_bypass_empty_output_policy_by_dependency_name(self):
        instance = controller.Context()
        instance.sources = {"Merlin.Frozen": object(), "QueryAudit": object()}
        instance.compile_audit_roots = {"Merlin.Frozen": ("root",)}
        with mock.patch.object(instance, "_guard", side_effect=AssertionError("collision reached runtime guard")):
            for module in ("Merlin.Frozen", "Query"):
                with self.subTest(module=module), self.assertRaises(data.ContextRefusal):
                    instance.check_query(module, b"#print axioms root", final_root="root")

    def test_exact_source_report_duplicate_is_one_bound_dependency(self):
        instance = controller.Context()
        instance.selection, instance.sources, instance.objects = selected(), {}, {}
        source = instance.selection.toolchain + "/src/lean/Init.lean"
        actual = instance._reported(source + "\n" + source + "\n", "source")
        self.assertEqual(actual, instance._reported(source + "\n", "source"))
        self.assertEqual(actual[1], {"Init"})

    def test_exact_object_report_duplicate_is_one_bound_dependency(self):
        instance = controller.Context()
        instance.selection, instance.sources, instance.objects = selected(), {}, {}
        target = instance.selection.toolchain + "/lib/lean/Init.olean"
        actual = instance._reported(target + "\n" + target + "\n", "object")
        self.assertEqual(actual, instance._reported(target + "\n", "object"))

    def test_duplicate_policy_never_coalesces_alias_wrong_kind_or_unknown(self):
        instance = controller.Context()
        instance.selection, instance.sources, instance.objects = selected(), {}, {}
        source = instance.selection.toolchain + "/src/lean/Init.lean"
        for bad in (
            source.replace("/Init.lean", "//Init.lean"),
            source + ".olean",
            source.replace("Init.lean", "Unknown.lean"),
            "/unbound/Unknown.lean",
        ):
            for raw in (source + "\n" + bad + "\n", bad + "\n" + bad + "\n"):
                with self.subTest(raw=raw), self.assertRaises(data.ContextRefusal):
                    instance._reported(raw, "source")

    def test_original_w_duplicate_report_recomputed_from_retained_data(self):
        work = ROOT / "out/build/graph-verification/loops/shape-after-consumer-v1"
        raw = (work / "commands/00000-source-imports/stdout.txt").read_text()
        context = json.loads((work / "CONTEXT.json").read_text())
        module = "Merlin.GraphLoopsShapeAfterEpoch"
        original = data.FileRef.parse(context["project_sources"][module])
        instance = controller.Context()
        instance.selection, instance.objects = selected(), {}
        path = str(work / "source/Merlin/GraphLoopsShapeAfterEpoch.lean")
        instance.sources = {module: data.FileRef(path, original.sha256, original.bytes)}
        project, standard, imports = instance._reported(raw, "source")
        self.assertEqual(project, {module})
        self.assertEqual(standard, {"Init"})
        self.assertEqual(len(raw.splitlines()), 3)
        self.assertEqual(len(imports), 2)

    def test_retained_runtime_trace_and_foreign_line_refusal(self):
        namespace = {"__name__": "_frozen_test_trace"}
        exec(compile(bound_io.read_bound(PINS["trace"]), PINS["trace"].path, "exec"), namespace)
        plan = selected()
        raw = (BASE / "basic-kernel-probe-v1/proof.stderr").read_text()
        process = json.loads((BASE / "basic-kernel-probe-v1/PROCESS.json").read_text())
        arguments = ([ref.record() for ref in plan.runtime], plan.executable, plan.interpreter, set(plan.misses))
        observed = namespace["parse"](raw, *arguments, expected_pid=process["pid"])
        self.assertEqual(len(observed["init_files"]), 10)
        for changed in (
            raw + "unknown non-loader error\n",
            raw.replace("[0]", "[1]", 1),
            raw + f"{process['pid']}: calling init: /tmp/unknown.so\n",
        ):
            with self.assertRaises(namespace["TraceRefusal"]):
                namespace["parse"](changed, *arguments, expected_pid=process["pid"])

    def test_revalidation_detects_changed_compile_and_import_records(self):
        # Synthetic issued state tests the binding plumbing, not a real proof.
        from types import SimpleNamespace

        for kind in ("compile", "source-import", "object-import"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                artifact = work / "artifact"
                artifact.write_bytes(b"synthetic")
                log = work / (kind + ".txt")
                log.write_bytes(b"original command output")
                ref, record = source_ref(artifact), source_ref(log)
                instance = controller.Context()
                instance.workspace, instance.held = work, False
                instance.selection = SimpleNamespace(forbidden_roots=())
                instance.context_id, instance._key = "synthetic-only", object()
                checked = controller.CheckedQuery(
                    "synthetic-only", "Synthetic", ref, ref, "root", (), (), ref, ref, ref, (record,), instance._key
                )
                instance._issued = {id(checked): checked}
                with mock.patch.object(instance, "_guard", return_value=None):
                    self.assertIs(instance.revalidate(checked), checked)
                    log.write_bytes(b"substituted command output")
                    with self.assertRaises(controller.InfrastructureStop):
                        instance.revalidate(checked)
                self.assertTrue(instance.held)
                self.assertTrue((work / "FIRST-INFRASTRUCTURE-STOP.json").is_file())

    def test_frozen_axiom_parser_exact_roots(self):
        namespace = {"__name__": "_frozen_test_trace"}
        exec(compile(bound_io.read_bound(PINS["trace"]), PINS["trace"].path, "exec"), namespace)
        parser, refusal = namespace["parse_axioms"], namespace["TraceRefusal"]
        self.assertEqual(parser("'correct' does not depend on any axioms\n", ["correct"]), {"correct": []})
        for text in (
            "'wrong' does not depend on any axioms",
            "'correct' depends on axioms: [sorryAx]",
            "'correct' does not depend on any axioms\nextra",
            "",
        ):
            with self.subTest(text=text), self.assertRaises(refusal):
                parser(text, ["correct"])


def mutation_test(change):
    def test(self):
        documents = copy.deepcopy(DOCUMENTS)
        change(documents)
        with mock.patch("os.open", side_effect=AssertionError("selection touched runtime")):
            with self.assertRaises((data.ContextRefusal, KeyError, TypeError)):
                selected(documents)

    return test


MUTATIONS = {
    "missing_image": lambda d: d["copy"]["files"].pop(),
    "extra_image": lambda d: d["copy"]["files"].append(copy.deepcopy(d["copy"]["files"][0])),
    "wrong_size": lambda d: d["copy"]["files"][0]["selected"].update(bytes=1),
    "wrong_hash": lambda d: d["copy"]["files"][0]["selected"].update(sha256="0" * 64),
    "stale_original": lambda d: d["copy"]["files"][0]["original"].update(sha256="0" * 64),
    "wrong_transport": lambda d: d["copy"]["files"][0]["selected"].update(path="/tmp/other/bin/lean"),
    "archived_as_private": lambda d: d["copy"].update(destination=d["copy"]["source_root"]),
    "wrong_archive": lambda d: d["copy"]["files"][0].update(archived="/tmp/other/bin/lean"),
    "missing_runtime": lambda d: d["probe"]["runtime"].pop(),
    "extra_runtime": lambda d: d["probe"]["runtime"].append(copy.deepcopy(d["probe"]["runtime"][0])),
    "wrong_runtime": lambda d: d["probe"]["runtime"][0].update(sha256="0" * 64),
    "unknown_symbol_owner": lambda d: d["probe"]["finite_allowed_misses"].append(["/tmp/unknown.so", "symbol"]),
    "wildcard_symbol": lambda d: d["probe"]["finite_allowed_misses"][0].__setitem__(1, "initialize_*"),
    "metadata_conflict": lambda d: d["metadata"]["conflicting_original_refs"].append({"conflict": True}),
}
for _name, _change in MUTATIONS.items():
    setattr(ContextDataTests, "test_mutation_" + _name, mutation_test(_change))


if __name__ == "__main__":
    unittest.main()
