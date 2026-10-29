"""Python-only graph admission regressions; never launch a proof or native oracle.

Run through the retained isolated runner with explicit primary/producer paths.
The public capture subprocesses select the unchanged original producer. New
corrective certificates are tested separately, not mislabeled as original4499.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

if not os.environ.get("MERLIN_ADMISSION_RUNNER"):
    raise unittest.SkipTest("requires the explicit SDK-free isolated admission runner")

import lean_graph_input_admission as admission
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.utils.exceptions import ParseError

from merlin.common.dense_float_ir import DenseFloatParser
from merlin.transforms import certificates

PRIMARY = Path(os.environ["MERLIN_ADMISSION_PRIMARY"]).resolve()
FIXTURES = PRIMARY / "merlin/tests/fixtures/graph_input_admission"
REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"
ORIGINAL = Path(os.environ.get("MERLIN_ADMISSION_ORIGINAL", "/home/jeremy/Projects/merlin-graph-proof-intake"))
RUNNER = os.environ.get("MERLIN_ADMISSION_RUNNER")
SIMPLE = "builtin.module { func.func @k(%a: i8, %b: i8) -> i8 { func.return %a : i8 } }"
CSE = """builtin.module { func.func @k(%a: i8, %b: i8) -> i8 {
  %x = arith.addi %a, %b : i8
  %y = arith.addi %a, %b : i8
  %z = arith.muli %x, %y : i8
  func.return %z : i8
} }"""
DIRECT = """builtin.module {
func.func private @callee(i8) -> i8
func.func @k(%x: i8) -> i8 { %y = func.call @callee(%x) : (i8) -> i8
func.return %y : i8 }
}"""
INTERNAL = DIRECT.replace(
    "func.func private @callee(i8) -> i8", "func.func @callee(%a: i8) -> i8 { func.return %a : i8 }"
)
SHAPE = """builtin.module { func.func @k(%x: memref<4xi8>) -> memref<4xi8> {
  %v = "memref.cast"(%x) : (memref<4xi8>) -> memref<4xi8>
  func.return %v : memref<4xi8>
} }"""


@contextmanager
def case_directory(name):
    """Retain original paths when recording evidence; ephemeral otherwise."""
    evidence = os.environ.get("MERLIN_ADMISSION_EVIDENCE")
    if evidence:
        path = Path(evidence) / "work" / name
        path.mkdir(parents=True, exist_ok=False)
        yield str(path)
    else:
        with tempfile.TemporaryDirectory(prefix="graph-admission-") as directory:
            yield directory


def raw_parse(text):
    """Actual parser without our guard, to test parsed-input entry points."""
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, func.Func, arith.Arith, scf.Scf, memref.MemRef):
        context.load_dialect(dialect)
    module = DenseFloatParser(context, text).parse_module()
    module.verify()
    return module


def guarded(text, mode="external-only"):
    return admission.parse_admitted(text, raw_parse, calls=mode)


def metadata(value, field="attributes"):
    module = raw_parse(SIMPLE)
    op = next(op for op in module.walk() if type(op) is func.ReturnOp)
    getattr(op, field)["nested"] = builtin.ArrayAttr([builtin.DictionaryAttr({"deep": value})])
    return module


def child(arguments, *, text=None, original=False):
    if RUNNER is None:
        raise RuntimeError("explicit SDK-free runner required for subprocess regressions")
    environment = dict(os.environ)
    if original:
        environment["MERLIN_ADMISSION_PRODUCER"] = str(ORIGINAL)
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", RUNNER, *arguments],
        input=text,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )


class AdmissionTests(unittest.TestCase):
    def refusal(self, code, action):
        with self.assertRaises(ValueError) as result:
            action()
        self.assertEqual(type(result.exception).__name__, "InputAdmissionError")
        self.assertEqual(result.exception.code, code)

    def test_simple_external_and_explicit_internal_controls(self):
        guarded(SIMPLE)
        guarded(DIRECT)
        self.refusal("defined-callee", lambda: guarded(INTERNAL))
        guarded(INTERNAL, "shape-direct")
        with self.assertRaises(ValueError):
            admission.admit_module(raw_parse(INTERNAL), calls="arbitrary")

    def test_signature_and_nested_direct_callee(self):
        module = raw_parse(DIRECT)
        call = next(op for op in module.walk() if type(op) is func.CallOp)
        call.properties["callee"] = builtin.SymbolRefAttr("callee", ["nested"])
        self.refusal("unresolved-or-nested-callee", lambda: admission.admit_module(module, calls="shape-direct"))
        module = raw_parse(DIRECT)
        declaration = module.body.block.first_op
        declaration.properties["function_type"] = builtin.FunctionType.from_lists([builtin.i32], [builtin.i8])
        self.refusal("call-signature", lambda: admission.admit_module(module, calls="shape-direct"))

    def test_quoted_metadata_and_escapes_are_data(self):
        for text in (
            "// {-# ignored\n" + SIMPLE,
            'builtin.module attributes {s = "{-# fake #-}"} {}',
            r'builtin.module attributes {s = "escaped \22 {-# @callee"} {}',
            r'builtin.module attributes {s = "escaped \\"} {}',
        ):
            with self.subTest(text=text):
                guarded(text)

    def test_malformed_not_reported_as_supported_or_refused_domain(self):
        for text in (
            "builtin.module {",
            'builtin.module attributes {s = "unterminated}',
            "builtin.module { func.func @f() { %x = arith.addi %missing, %missing : i8 } }",
        ):
            with self.subTest(text=text), self.assertRaises(ParseError):
                guarded(text)
        # The delimiter is a domain exclusion, NOT a well-formedness assertion.
        self.refusal("file-metadata", lambda: admission.admit_text("builtin.module {} {-# malformed"))

    def test_no_parser_invocation_before_raw_metadata_refusal(self):
        def must_not_parse(_):
            self.fail("raw file metadata reached parser")

        self.refusal("file-metadata", lambda: admission.parse_admitted(SIMPLE + "\n{-# #-}", must_not_parse))

    def test_function_type_even_without_indirect_operation(self):
        module = raw_parse(SIMPLE)
        op = next(op for op in module.walk() if type(op) is func.ReturnOp)
        op.attributes["signature"] = builtin.FunctionType.from_lists([], [])
        self.refusal("function-value-type", lambda: admission.admit_module(module))

    def test_public_integer_malformed_record_is_distinct(self):
        with case_directory("integer-malformed") as directory:
            source, output = Path(directory) / "bad.mlir", Path(directory) / "result"
            source.write_text("builtin.module {")
            result = child(
                [
                    "capture",
                    "lean_graph_state_integer_capture.py",
                    "--reference",
                    str(ORIGINAL),
                    "--source",
                    str(source),
                    "--destination",
                    str(output),
                ],
                original=True,
            )
            self.assertNotEqual(result.returncode, 0)
            record = json.loads((output / "REJECTION.json").read_text())
            self.assertEqual(record["status"], "malformed")
            self.assertEqual(record["kind"], "ParseError")
            evidence = os.environ.get("MERLIN_ADMISSION_EVIDENCE")
            if evidence:
                shutil.copytree(directory, Path(evidence) / "cases" / "integer-malformed")

    def test_independent_index_decoder(self):
        from lean_graph_loops_segment_capture import operation_tree

        for value in (-1, 0, 7):
            module = raw_parse(
                "builtin.module { func.func @k() -> index { "
                f"%c = arith.constant {value} : index func.return %c : index " + "} }"
            )
            tree = operation_tree(module)
            function = tree["regions"][0][0]["operations"][0]
            constant = function["regions"][0][0]["operations"][0]
            self.assertEqual(constant["index_constant"], value)

    def test_opaque_attribute_not_searched_as_a_string(self):
        module = raw_parse("builtin.module attributes {hidden = #test.attr<@callee>} {}")
        self.refusal("opaque-attribute", lambda: admission.admit_module(module))

    def test_primary_and_corrective_helper_exact_bytes(self):
        current = PRIMARY / "src/merlin/verify/lean_graph_input_admission.py"
        producer = Path(certificates.__file__).resolve().parents[1] / "verify/lean_graph_input_admission.py"
        self.assertEqual(current.read_bytes(), producer.read_bytes())
        self.assertEqual(admission.source_reference()["sha256"], hashlib.sha256(current.read_bytes()).hexdigest())

    def test_review_fixture_provenance(self):
        record = json.loads((FIXTURES / "PROVENANCE.json").read_text())
        self.assertEqual(record["review_commit"], "5a7f872db33d378d6b0c4a294e4099e80efd4956")
        for row in record["files"]:
            data = (FIXTURES / row["path"]).read_bytes()
            self.assertEqual((len(data), hashlib.sha256(data).hexdigest()), (row["bytes"], row["sha256"]))

    def test_resources_rejected_in_fresh_process_and_repeated_process(self):
        names = (
            "resource-blob-changed-before.mlir",
            "resource-blob-changed-after.mlir",
            "resource-blob-dropped-after.mlir",
        )
        for _ in range(2):
            for name in names:
                raw = (FIXTURES / name).read_text()
                code = "dense-resource" if "dropped" in name else "file-metadata"
                self.refusal(code, lambda: guarded(raw))
                result = child(["raw"], text=raw)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout.splitlines()[-1])["code"], code)

    def test_process_global_resource_renaming_does_not_authorize(self):
        raw = (FIXTURES / "resource-blob-changed-before.mlir").read_text()
        for _ in range(3):
            module = raw_parse(raw)
            self.refusal("dense-resource", lambda: admission.admit_module(module))
            self.refusal("dense-resource", lambda: certificates.ir_text(module))

    def test_nested_resources_and_symbols_in_attributes_and_properties(self):
        for field in ("attributes", "properties"):
            for value, code in (
                (builtin.SymbolRefAttr("k"), "unmodeled-symbol-reference"),
                (builtin.DenseResourceAttr.from_params("blob", builtin.TensorType(builtin.i8, [2])), "dense-resource"),
            ):
                with self.subTest(field=field, code=code):
                    self.refusal(code, lambda: admission.admit_module(metadata(value, field)))
                    if code == "dense-resource":
                        self.refusal(code, lambda: certificates.ir_text(metadata(value, field)))

    def test_builtin_opaque_attribute_is_not_an_inspected_symbol_tree(self):
        module = metadata(builtin.OpaqueAttr.from_strings("unknown", "@k"))
        self.refusal("opaque-attribute", lambda: admission.admit_module(module))

    def test_direct_call_callee_exception_does_not_cover_extra_metadata(self):
        module = raw_parse(INTERNAL)
        call = next(op for op in module.walk() if type(op) is func.CallOp)
        call.attributes["hidden"] = builtin.ArrayAttr([builtin.SymbolRefAttr("callee")])
        self.refusal("unmodeled-symbol-reference", lambda: admission.admit_module(module, calls="shape-direct"))

    def test_positive_public_exports_and_explicit_memory_adapter(self):
        from lean_graph_loops_reuse_capture import export as reuse
        from lean_graph_loops_shape_capture import export as shape
        from lean_graph_state_scalar_capture import project

        project(raw_parse(SIMPLE), raw_parse(SIMPLE))
        project(raw_parse(DIRECT), raw_parse(DIRECT))
        self.assertEqual(len(shape(raw_parse(INTERNAL))["calls"]), 1)
        self.refusal("defined-callee", lambda: reuse(raw_parse(INTERNAL)))
        self.assertIn("tree", reuse(raw_parse(INTERNAL), calls="shape-direct"))


def reviewer_test(filename, code):
    def test(self):
        text = (FIXTURES / filename).read_text()
        self.refusal(code, lambda: guarded(text))
        # Producer parser has resource-only admission; semantic guard owns calls.
        if code in ("file-metadata", "dense-resource"):
            self.refusal(code, lambda: certificates.parse_ir(text))

    return test


for _fixture in json.loads((FIXTURES / "PROVENANCE.json").read_text())["files"]:
    _name = _fixture["path"]
    _code = (
        "function-value-route"
        if _name.startswith("indirect")
        else "dense-resource"
        if "dropped" in _name or "projection-after" in _name
        else "file-metadata"
    )
    setattr(AdmissionTests, "test_reviewer_" + _name.replace("-", "_").replace(".", "_"), reviewer_test(_name, _code))


EXPORTERS = (
    ("lean_graph_state_scalar_capture", "project"),
    ("lean_graph_state_pipeline_capture", "project_modules"),
    ("lean_graph_state_integer_capture", "validate_calls"),
    ("lean_graph_loops_shape_capture", "export"),
    ("lean_graph_loops_reuse_capture", "export"),
    ("lean_graph_loops_memory_scf_capture", "export"),
    ("lean_graph_loops_segment_capture", "operation_tree"),
)


def exporter_test(module_name, function, case):
    def test(self):
        if case == "indirect":
            module = raw_parse((FIXTURES / "indirect-local-call-before.mlir").read_text())
            code = "function-value-route"
        elif case == "symbol":
            module = metadata(builtin.SymbolRefAttr("k"), "properties")
            code = "unmodeled-symbol-reference"
        else:
            module = raw_parse((FIXTURES / "resource-blob-dropped-after.mlir").read_text())
            code = "dense-resource"
        export = getattr(importlib.import_module(module_name), function)
        if function == "project":

            def action():
                return export(module, module.clone())
        elif function in ("project_modules", "validate_calls"):

            def action():
                return export([module] * 5)
        else:

            def action():
                return export(module)

        self.refusal(code, action)

    return test


for _module, _function in EXPORTERS:
    for _case in ("indirect", "symbol", "resource"):
        setattr(AdmissionTests, f"test_public_{_module}_{_case}", exporter_test(_module, _function, _case))


def capture_test(route, supplied, case):
    def test(self):
        with case_directory(f"{route}-{supplied}-{case}") as directory:
            work = Path(directory)
            before, after, middle = (work / name for name in ("before.mlir", "after.mlir", "middle.mlir"))
            raw = (
                (FIXTURES / "resource-blob-changed-before.mlir").read_text()
                if case == "resource"
                else (FIXTURES / "indirect-local-call-before.mlir").read_text()
            )
            before.write_text(SIMPLE if supplied else raw)
            after.write_text(raw)
            middle.write_text(SIMPLE)
            output = work / "result"
            if route == "scalar":
                args = ["--reference", str(ORIGINAL), "--source", str(before), "--destination", str(output)]
                if supplied:
                    args += ["--target", str(after)]
                script = "lean_graph_state_scalar_capture.py"
            elif route in ("shape", "reuse", "memory_scf"):
                args = [
                    "--source",
                    str(ORIGINAL),
                    "--revision",
                    REVISION,
                    "--before",
                    str(before),
                    "--output",
                    str(output),
                ]
                if route == "memory_scf":
                    args += ["--index-bits", "32"]
                if supplied:
                    args += ["--after", str(after)]
                    if route == "memory_scf":
                        args += ["--middle", str(middle)]
                script = f"lean_graph_loops_{route}_capture.py"
            else:
                args = ["--reference", str(ORIGINAL), "--destination", str(output)]
                if supplied:
                    count = 5 if route == "pipeline" else 2
                    args += ["--buffers", *([str(before)] * (count - 1)), str(after)]
                else:
                    args += ["--source", str(before)]
                script = f"lean_graph_state_{route}_capture.py"
            result = child(["capture", script, *args], original=True)
            evidence = os.environ.get("MERLIN_ADMISSION_EVIDENCE")
            if evidence:
                saved = Path(evidence) / "cases" / f"{route}-{supplied}-{case}"
                shutil.copytree(work, saved)
                (saved / "stdout.txt").write_text(result.stdout)
                (saved / "stderr.txt").write_text(result.stderr)
                (saved / "PROCESS.json").write_text(
                    json.dumps({"returncode": result.returncode, "arguments": args}) + "\n"
                )
            code = "file-metadata" if case == "resource" else "function-value-route"
            if route == "scalar":
                record = json.loads((output / "projection.json").read_text())
                self.assertFalse(record["projection_checked"])
                self.assertFalse(record["public_acceptance_issued"])
                self.assertEqual(record["reason"]["type"], "InputAdmissionError")
                self.assertIn(code, record["reason"]["detail"])
            elif route in ("integer", "pipeline"):
                self.assertNotEqual(result.returncode, 0)
                record = json.loads((output / "REJECTION.json").read_text())
                self.assertEqual(record["kind"], "InputAdmissionError")
                self.assertFalse(record["qualified"])
                self.assertIn(code, record["reason"])
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("InputAdmissionError", result.stderr, result.stderr)
                self.assertIn(code, result.stderr)

    return test


for _route in ("scalar", "pipeline", "integer", "shape", "reuse", "memory_scf"):
    for _supplied in (False, True):
        for _case in ("resource", "indirect"):
            setattr(
                AdmissionTests, f"test_capture_{_route}_{_supplied}_{_case}", capture_test(_route, _supplied, _case)
            )


def positive_capture_test(route, supplied):
    def test(self):
        with case_directory(f"positive-{route}-{supplied}") as directory:
            work = Path(directory)
            before, after, middle = (work / name for name in ("before.mlir", "after.mlir", "middle.mlir"))
            text = SHAPE if route == "shape" else CSE if route in ("scalar", "pipeline") else SIMPLE
            if route == "integer":
                text = (
                    "builtin.module { func.func @k(%x: i8) -> i8 { %z = arith.constant 0 : i8 "
                    "%y = arith.addi %x, %z : i8 func.return %y : i8 } }"
                )
            for path in (before, after, middle):
                path.write_text(text)
            output = work / "result"
            if route == "scalar":
                args = ["--reference", str(ORIGINAL), "--source", str(before), "--destination", str(output)]
                if supplied:
                    args += ["--target", str(after)]
                script, record_name = "lean_graph_state_scalar_capture.py", "projection.json"
            elif route in ("shape", "reuse", "memory_scf"):
                args = [
                    "--source",
                    str(ORIGINAL),
                    "--revision",
                    REVISION,
                    "--before",
                    str(before),
                    "--output",
                    str(output),
                ]
                if route == "memory_scf":
                    args += ["--index-bits", "32"]
                if supplied:
                    args += ["--after", str(after)]
                    if route == "memory_scf":
                        args += ["--middle", str(middle)]
                script = f"lean_graph_loops_{route}_capture.py"
                record_name = None if supplied else "CAPTURE.json"
            else:
                args = ["--reference", str(ORIGINAL), "--destination", str(output)]
                if supplied:
                    count = 5 if route == "pipeline" else 2
                    args += ["--buffers", *([str(before)] * (count - 1)), str(after)]
                else:
                    args += ["--source", str(before)]
                script, record_name = f"lean_graph_state_{route}_capture.py", "CAPTURE.json"
            result = child(["capture", script, *args], original=True)
            evidence = os.environ.get("MERLIN_ADMISSION_EVIDENCE")
            if evidence:
                saved = Path(evidence) / "cases" / f"positive-{route}-{supplied}"
                shutil.copytree(work, saved)
                (saved / "stdout.txt").write_text(result.stdout)
                (saved / "stderr.txt").write_text(result.stderr)
                (saved / "PROCESS.json").write_text(
                    json.dumps({"returncode": result.returncode, "arguments": args}) + "\n"
                )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            record = json.loads((output / record_name if record_name else output).read_text())
            self.assertEqual(record["input_admission_source"], admission.source_reference())
            if route == "scalar":
                self.assertTrue(record["projection_checked"], record)
                self.assertFalse(record["public_acceptance_issued"])
                if not supplied:
                    self.assertTrue(record["genuine_changed_boundary"])
            if not supplied and route in ("integer", "pipeline"):
                self.assertTrue(record["each_stage_changed"][0])
            if not supplied and route == "shape":
                self.assertGreater(record["changed_sites"], 0)

    return test


for _route in ("scalar", "pipeline", "integer", "shape", "reuse", "memory_scf"):
    for _supplied in (False, True):
        setattr(AdmissionTests, f"test_positive_capture_{_route}_{_supplied}", positive_capture_test(_route, _supplied))


if __name__ == "__main__":
    unittest.main()
