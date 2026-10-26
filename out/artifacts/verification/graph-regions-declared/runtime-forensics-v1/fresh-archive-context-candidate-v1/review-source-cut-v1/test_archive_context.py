"""Tiny fake manifest and mocked protocol ONLY; never an SDK/native proof test."""
import ast
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE / "source"))
import archive_selection as archive
import lean_graph_original_image_context as controller
import lean_graph_original_image_context_data as data
import lean_graph_original_image_context_io as io

OUTPUT = HERE / "synthetic-context-v1"
TRACE = Path("out/build/graph-paper-deadline/original-runtime-candidate-v1/basic-kernel-probe-v1/trace_parser.py").absolute()


def reference(path):
    raw = path.read_bytes()
    return data.FileRef(str(path), hashlib.sha256(raw).hexdigest(), len(raw))


def documents():
    top = "fake-sdk"
    request = {"schema": "merlin.fresh-sdk-data-intake.v1", "authority": archive.NO_AUTHORITY,
               "new_root": str(OUTPUT / "never-created-sdk"), "fetch": {"url": "https://example.invalid/fake.zip",
               "expected_sha256": "1" * 64, "exact_bytes": 100}, "environment_identity": {"synthetic": True},
               "extract": {"expected_top_directory": top, "member_count_max": 30, "per_member_bytes_max": 100,
                           "total_uncompressed_bytes_max": 1000}}
    names = [top + "/bin/lean", top + "/lib/lean/libleanshared.so"]
    names += [f'{top}/{"src/lean" if s == ".lean" else "lib/lean"}/Init{s}' for s in archive.FAMILIES]
    rows = [{"path": name, "kind": "file", "declared_bytes": 1, "crc32": 1, "source_mode": 0o100644,
             "compression": 0} for name in names]
    roster = {"members": rows, "parent_sha256": "1" * 64}
    manifest = {"schema": "merlin.fresh-sdk-authenticated-member-manifest.v1", "parent_archive": {
                "url": request["fetch"]["url"], "sha256": "1" * 64, "bytes": 100, "identity": {"synthetic": True}},
                "members": [{**r, "measured_bytes": 1, "sha256": "2" * 64, "output_stat": {}} for r in rows],
                "expanded_bytes": len(rows), "new_environment_pins": True, "old_expectations_replaced": False,
                "authority": archive.NO_AUTHORITY, "request_identity": data.identity(request)}
    host = {"schema": "merlin.fresh-archive-separate-host-closure.v1", "archive_manifest": None,
            "refs": [{"path": "/usr/lib/SYNTHETIC-NEVER-OPEN-LOADER", "sha256": "3" * 64, "bytes": 1}],
            "interpreter": "/usr/lib/SYNTHETIC-NEVER-OPEN-LOADER", "finite_allowed_misses": [],
            "success_only_misses": [], "native_authorized": False}
    selection = {"schema": "merlin.fresh-archive-common-selection.v1", "inputs": {},
                 "toolchain": request["new_root"] + "/data/" + top, "members": names, "modules": ["Init"],
                 "sdk_runtime_members": names[:2], "forbidden_roots": ["/NEVER-OPEN-OLD-SDK"], "native_authorized": False}
    return copy.deepcopy(dict(request=request, manifest=manifest, roster=roster, selection=selection, host=host))


def bind_docs(path, docs):
    path.mkdir()
    refs = {"trace": reference(TRACE)}
    for name in ("request", "roster", "manifest", "host", "selection"):
        if name == "host":
            docs[name]["archive_manifest"] = refs["manifest"].record()
        if name == "selection":
            docs[name]["inputs"] = {k: refs[k].record() for k in ("request", "manifest", "roster", "host", "trace")}
        file = path / (name + ".json")
        io.save(file, docs[name])
        refs[name] = reference(file)
    return refs


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir()  # no replacement of previous results

    def fixture(self):
        docs = documents()
        refs = bind_docs(OUTPUT / self._testMethodName, docs)
        return docs, refs

    def test_pure_full_selection(self):
        docs, refs = self.fixture()
        with patch("os.open", side_effect=AssertionError("pure admission must never open images")):
            selected = archive.select(docs, refs)
        self.assertEqual(len(selected.images), 9)
        self.assertEqual(len(selected.runtime), 3)
        self.assertFalse((OUTPUT / "never-created-sdk").exists())

    def test_adversarial_associations(self):
        docs, refs = self.fixture()
        mutations = {
            "parent": lambda d: d["manifest"]["parent_archive"].update(sha256="0" * 64),
            "request": lambda d: d["manifest"].update(request_identity="0" * 64),
            "input": lambda d: d["selection"]["inputs"]["manifest"].update(sha256="0" * 64),
            "host_parent": lambda d: d["host"]["archive_manifest"].update(sha256="0" * 64),
            "missing": lambda d: d["selection"]["members"].pop(),
            "extra": lambda d: d["selection"]["members"].append("fake-sdk/extra"),
            "duplicate": lambda d: d["selection"]["members"].append(d["selection"]["members"][0]),
            "changed_pin": lambda d: d["manifest"]["members"][0].update(sha256="forged"),
            "wrong_bytes": lambda d: d["manifest"]["members"][0].update(measured_bytes=True),
            "missing_module": lambda d: d["selection"].update(modules=["Absent"]),
            "extra_runtime": lambda d: d["selection"]["sdk_runtime_members"].append("fake-sdk/missing.so"),
            "missing_exe": lambda d: d["selection"]["sdk_runtime_members"].pop(0),
            "loader": lambda d: d["host"].update(interpreter="/usr/lib/unbound"),
            "escalation": lambda d: d["selection"].update(native_authorized=True),
            "held": lambda d: d["selection"]["forbidden_roots"].append(d["selection"]["toolchain"]),
        }
        for name, change in mutations.items():
            altered = copy.deepcopy(docs)
            change(altered)
            with self.subTest(name=name), self.assertRaises(data.ContextRefusal):
                archive.select(altered, refs)

    def test_missing_ir_even_with_matching_manifest_roster(self):
        docs, refs = self.fixture()
        index = next(i for i, r in enumerate(docs["roster"]["members"]) if r["path"].endswith(".ir"))
        member = docs["roster"]["members"].pop(index)["path"]
        docs["manifest"]["members"].pop(index)
        docs["manifest"]["expanded_bytes"] -= 1
        docs["selection"]["members"].remove(member)
        with self.assertRaisesRegex(data.ContextRefusal, "complete source/object/IR"):
            archive.select(docs, refs)

    def test_escape_and_prefix_collision(self):
        docs, refs = self.fixture()
        for name in ("fake-sdk/../escape", "fake-sdk/lib", "/fake-sdk/escape", "fake-sdk//bad"):
            changed = copy.deepcopy(docs)
            changed["roster"]["members"][0]["path"] = name
            changed["manifest"]["members"][0]["path"] = name
            with self.subTest(name=name), self.assertRaises(data.ContextRefusal):
                archive.select(changed, refs)

    def context(self):
        docs, refs = self.fixture()
        source = OUTPUT / self._testMethodName / "Baseline.lean"
        source.write_bytes(b"import Init\nnamespace Merlin\ntheorem baseline : True := True.intro\nend Merlin\n")
        controls = tuple(reference(HERE / "source" / f) for f in (
            "lean_graph_original_image_context.py", "lean_graph_original_image_context_data.py",
            "lean_graph_original_image_context_io.py", "archive_selection.py"))
        with patch.object(controller.subprocess, "Popen", side_effect=AssertionError("native forbidden")):
            result = controller.Context.from_review(review=refs, project_sources={"Merlin.Baseline": reference(source)},
                         controller_sources=controls, workspace=OUTPUT / (self._testMethodName + "-context"))
        return result

    def mocked_query(self, ctx, *, bad_axiom=False, empty_types=False):
        ctx._guard = lambda: {"MOCKED_NOT_PROOF": True}
        def build(module):
            target = ctx.objects_dir / (module.replace(".", "/") + ".olean")
            io.write_new(target, b"SYNTHETIC-NOT-A-LEAN-OBJECT")
            ctx.objects[module] = {".olean": reference(target)}
        ctx._build = build
        ctx._inspect = lambda ref: {"MOCKED_NOT_PROOF": ref.record()}
        def invoke(label, arguments, **kwargs):
            path = ctx.commands_dir / label
            path.mkdir()
            if label == "root-audit":
                stdout = "'Merlin.query' depends on axioms: [" + ("fakeAxiom" if bad_axiom else "propext") + "]\n"
            else:
                stdout = "" if empty_types else "Merlin.query : True\n"
            io.write_new(path / "stdout.txt", stdout.encode())
            io.write_new(path / "stderr.txt", b"MOCKED-NOT-A-RUNTIME-TRACE")
            return stdout, path
        ctx._invoke = invoke
        return ctx.check_query("Merlin.Query", b"import Init\nnamespace Merlin\ntheorem query : True := True.intro\nend Merlin\n", final_root="Merlin.query")

    def test_source_prepare_does_not_guard_or_launch(self):
        ctx = self.context()
        self.assertEqual(ctx.definition["schema"], "merlin.graph.fresh-authenticated-archive-context.v1")
        self.assertFalse(ctx.held)
        self.assertFalse((OUTPUT / "never-created-sdk").exists())

    def test_same_live_typed_authority_with_mocked_execution(self):
        ctx = self.context()
        result = self.mocked_query(ctx)
        self.assertIs(ctx.revalidate(result), result)
        for forged in (result.__dict__, replace(result), object()):
            with self.assertRaises(data.ContextRefusal):
                ctx.revalidate(forged)

    def test_changed_query_preserves_failed_buffer(self):
        ctx = self.context()
        result = self.mocked_query(ctx)
        Path(result.query.path).write_bytes(b"changed tiny synthetic query")
        with self.assertRaises(controller.InfrastructureStop):
            ctx.revalidate(result)
        self.assertTrue(ctx.held)
        self.assertEqual((ctx.workspace / "first-failed-read/already-read-buffer.bin").read_bytes(), b"changed tiny synthetic query")

    def test_nonstandard_axiom_holds(self):
        ctx = self.context()
        with self.assertRaises(controller.InfrastructureStop):
            self.mocked_query(ctx, bad_axiom=True)
        self.assertTrue(ctx.held)

    def test_missing_full_type_output_holds(self):
        ctx = self.context()
        with self.assertRaises(controller.InfrastructureStop):
            self.mocked_query(ctx, empty_types=True)
        self.assertTrue(ctx.held)

    def test_query_roots_and_names(self):
        ctx = self.context()
        with patch.object(ctx, "_guard", side_effect=AssertionError("should refuse before image access")):
            with self.assertRaises(data.ContextRefusal):
                ctx.check_query("Merlin.Baseline", b"", final_root="Merlin.query")
            with self.assertRaises(data.ContextRefusal):
                ctx.check_query("Merlin.New", b"", final_root="Merlin.query", support_roots=("Merlin.query",))

    def test_actual_object_header_disagreement(self):
        ctx = self.context()
        ctx._invoke = lambda *a, **k: ("/unbound/Init.olean\n", ctx.commands_dir)
        with self.assertRaises(data.ContextRefusal):
            ctx._inspect(ctx.sources["Merlin.Baseline"])

    def test_changed_metadata_bytes_refused_before_preparation(self):
        docs, refs = self.fixture()
        path = Path(refs["manifest"].path)
        changed = copy.deepcopy(docs["manifest"])
        changed["members"][0]["sha256"] = "f" * 64
        path.write_text(json.dumps(changed))
        with patch.object(controller.subprocess, "Popen", side_effect=AssertionError("native forbidden")):
            with self.assertRaises(io.BindingError):
                controller.Context.from_review(review=refs, project_sources={}, controller_sources=(),
                                               workspace=OUTPUT / "must-not-prepare")
        self.assertFalse((OUTPUT / "must-not-prepare").exists())

    def test_context_selection_mutation_holds_before_image_reads(self):
        ctx = self.context()
        ctx.selection = replace(ctx.selection, executable="/unbound/lean")
        with patch.object(io, "read_bound", side_effect=AssertionError("must fail before image reads")):
            with self.assertRaises(controller.InfrastructureStop):
                ctx._guard()
        self.assertTrue(ctx.held)

    def test_wrong_typed_reference_refused(self):
        docs, refs = self.fixture()
        refs["manifest"] = refs["manifest"].record()
        with self.assertRaises(data.ContextRefusal):
            controller.Context.from_review(review=refs, project_sources={}, controller_sources=(),
                                           workspace=OUTPUT / "wrong-typed")

    def test_same_common_protocol_methods(self):
        baseline = json.loads((HERE / "BASELINE.json").read_text())
        path = next(r["path"] for r in baseline["source_refs"] if Path(r["path"]).name == "lean_graph_original_image_context.py")
        def methods(source):
            tree = ast.parse(source)
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Context")
            return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n, ast.FunctionDef)}
        before, after = methods(Path(path).read_text()), methods((HERE / "source/lean_graph_original_image_context.py").read_text())
        for name in ("check_query", "_inspect", "_reported", "_build", "_invoke", "_stop", "_freeze"):
            self.assertEqual(before[name], after[name], name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
