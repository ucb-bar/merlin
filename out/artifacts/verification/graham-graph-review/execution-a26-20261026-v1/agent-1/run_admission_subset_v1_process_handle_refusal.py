"""Independent SDK-free admission controls; this does not replay the 81-case suite.

Execute with the selected isolated interpreter and -I -S -B. It imports only
the selected xDSL distribution, exact parser adapter snapshot and current
admission helper. It never imports producer, proof, solver or native tool code.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import sys
import traceback


ROOT = pathlib.Path.cwd().resolve()
OUTPUT = pathlib.Path(__file__).resolve().parent
SITE = pathlib.Path(sys.executable).parent.parent / "lib/python3.10/site-packages"
if not SITE.is_dir():
    candidates = list((pathlib.Path(sys.executable).parent.parent / "lib").glob("python*/site-packages"))
    if len(candidates) != 1:
        raise RuntimeError("exact isolated site directory required")
    SITE = candidates[0]
sys.path.insert(0, str(SITE))
FIXTURES = ROOT / "merlin/tests/fixtures/graph_input_admission"
ADMISSION = ROOT / "src/merlin/verify/lean_graph_input_admission.py"
PARSER = ROOT / "out/artifacts/verification/graph-pipeline-gate/development/unroll-inline-v1/source-snapshot/dependencies-final/pinned4499/src/merlin/common/dense_float_ir.py"
DENIED = []


def audit(event, arguments):
    if event in ("subprocess.Popen", "os.system", "os.posix_spawn", "ctypes.dlopen"):
        DENIED.append({"event": event, "arguments": repr(arguments)})
        raise RuntimeError("SDK-free test forbids process launch or explicit library loading")
    if event == "open" and isinstance(arguments[0], (str, bytes)):
        text = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
        path = pathlib.Path(text)
        if path.suffix in (".olean", ".ilean", ".o", ".a", ".zip", ".zst") or any(
            part in text for part in ("libleanshared", "libz3", "libLLVM", "lean-w8a8", "runtime-forensics")
        ):
            DENIED.append({"event": event, "path": text})
            raise RuntimeError("SDK/object/archive input refused")


sys.addaudithook(audit)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


admission = load("a26_input_admission", ADMISSION)
dense = load("a26_dense_parser", PARSER)
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.utils.exceptions import ParseError


SIMPLE = "builtin.module { func.func @k(%a: i8, %b: i8) -> i8 { func.return %a : i8 } }"
DIRECT = """builtin.module {
func.func private @callee(i8) -> i8
func.func @k(%x: i8) -> i8 { %y = func.call @callee(%x) : (i8) -> i8
func.return %y : i8 }
}"""
INTERNAL = DIRECT.replace("func.func private @callee(i8) -> i8", "func.func @callee(%a: i8) -> i8 { func.return %a : i8 }")


def parse(text):
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, func.Func, arith.Arith, scf.Scf, memref.MemRef):
        context.load_dialect(dialect)
    module = dense.DenseFloatParser(context, text).parse_module()
    module.verify()
    return module


def guarded(text, calls="external-only"):
    return admission.parse_admitted(text, parse, calls=calls)


def require_refusal(code, action):
    try:
        action()
    except admission.InputAdmissionError as exc:
        assert exc.code == code, (exc.code, code)
        return {"code": exc.code, "path": exc.path}
    raise AssertionError("expected domain refusal " + code)


def require_parse_error(text):
    try:
        guarded(text)
    except ParseError as exc:
        return {"exception": type(exc).__name__}
    raise AssertionError("expected malformed-input ParseError")


def mutated_module(value, field="attributes"):
    module = parse(SIMPLE)
    op = next(op for op in module.walk() if type(op) is func.ReturnOp)
    getattr(op, field)["nested"] = builtin.ArrayAttr([builtin.DictionaryAttr({"deep": value})])
    return module


CASES = []


def case(name, action):
    try:
        detail = action()
        CASES.append({"name": name, "status": "PASS", "detail": detail if isinstance(detail, dict) else None})
    except Exception as exc:
        CASES.append({"name": name, "status": "FAIL", "kind": type(exc).__name__, "reason": str(exc), "traceback": traceback.format_exc()})


case("simple_positive", lambda: guarded(SIMPLE))
case("declared_external_positive", lambda: guarded(DIRECT))
case("defined_direct_refused_by_default", lambda: require_refusal("defined-callee", lambda: guarded(INTERNAL)))
case("explicit_shape_direct_positive", lambda: guarded(INTERNAL, "shape-direct"))
for i, raw in enumerate((
    "// {-# ignored\n" + SIMPLE,
    'builtin.module attributes {s = "{-# fake #-}"} {}',
    r'builtin.module attributes {s = "escaped \22 {-# @callee"} {}',
    r'builtin.module attributes {s = "escaped \\"} {}',
)):
    case("quoted_comment_control_" + str(i), lambda raw=raw: guarded(raw))
for i, raw in enumerate(("builtin.module {", 'builtin.module attributes {s = "unterminated}',
                         "builtin.module { func.func @f() { %x = arith.addi %missing, %missing : i8 } }")):
    case("malformed_control_" + str(i), lambda raw=raw: require_parse_error(raw))


def raw_before_parser():
    def forbidden_parser(_):
        raise AssertionError("parser must not run")
    return require_refusal("file-metadata", lambda: admission.parse_admitted(SIMPLE + "\n{-# #-}", forbidden_parser))


case("metadata_refused_before_parser", raw_before_parser)
provenance = json.loads((FIXTURES / "PROVENANCE.json").read_bytes())
for row in provenance["files"]:
    name = row["path"]
    raw = (FIXTURES / name).read_text()
    assert hashlib.sha256(raw.encode()).hexdigest() == row["sha256"]
    code = "function-value-route" if name.startswith("indirect") else "dense-resource" if "dropped" in name or "projection-after" in name else "file-metadata"
    case("review_fixture_" + name, lambda raw=raw, code=code: require_refusal(code, lambda: guarded(raw)))
for field in ("attributes", "properties"):
    case("nested_symbol_" + field, lambda field=field: require_refusal("unmodeled-symbol-reference", lambda: admission.admit_module(mutated_module(builtin.SymbolRefAttr("k"), field))))
    case("nested_resource_" + field, lambda field=field: require_refusal("dense-resource", lambda: admission.admit_module(mutated_module(builtin.DenseResourceAttr.from_params("blob", builtin.TensorType(builtin.i8, [2])), field))))
case("opaque_attribute_refused", lambda: require_refusal("opaque-attribute", lambda: admission.admit_module(mutated_module(builtin.OpaqueAttr.from_strings("unknown", "@k")))))
case("function_value_type_refused", lambda: require_refusal("function-value-type", lambda: admission.admit_module(mutated_module(builtin.FunctionType.from_lists([], [])))))


def extra_direct_symbol():
    module = parse(INTERNAL)
    call = next(op for op in module.walk() if type(op) is func.CallOp)
    call.attributes["hidden"] = builtin.ArrayAttr([builtin.SymbolRefAttr("callee")])
    return require_refusal("unmodeled-symbol-reference", lambda: admission.admit_module(module, calls="shape-direct"))


case("shape_direct_exception_does_not_cover_metadata", extra_direct_symbol)
for index in range(3):
    case("parsed_resource_process_state_" + str(index), lambda: require_refusal("dense-resource", lambda: admission.admit_module(parse((FIXTURES / "resource-blob-changed-before.mlir").read_text()))))


def signature():
    module = parse(DIRECT)
    module.body.block.first_op.properties["function_type"] = builtin.FunctionType.from_lists([builtin.i32], [builtin.i8])
    return require_refusal("call-signature", lambda: admission.admit_module(module))


case("signature_mismatch", signature)


def nested_callee():
    module = parse(DIRECT)
    call = next(op for op in module.walk() if type(op) is func.CallOp)
    call.properties["callee"] = builtin.SymbolRefAttr("callee", ["nested"])
    return require_refusal("unresolved-or-nested-callee", lambda: admission.admit_module(module, calls="shape-direct"))


case("nested_direct_callee_refused", nested_callee)


def selected_source():
    assert admission.source_reference()["sha256"] == hashlib.sha256(ADMISSION.read_bytes()).hexdigest()
    return {"matches": True}


case("source_reference_matches_selected_helper", selected_source)

source_files = {ADMISSION, PARSER, pathlib.Path(__file__).resolve()}
source_files.update(pathlib.Path(m.__file__).resolve() for name, m in sys.modules.items()
                    if name.startswith(("xdsl", "typing_extensions", "ordered_set", "immutabledict"))
                    and getattr(m, "__file__", None) and str(m.__file__).endswith(".py"))
sources = []
for path in sorted(source_files):
    if not path.is_relative_to(ROOT):
        raise RuntimeError("dependency source lies outside approved workspace: " + str(path))
    data = path.read_bytes()
    sources.append({"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
result = {
    "schema": "merlin.a26.independent-admission-controls.v1",
    "python": sys.version, "executable": sys.executable,
    "xdsl_version": importlib.metadata.version("xdsl"), "site": str(SITE),
    "baseline": "c62f81a8a11347ff53b60101fa75280e69d002a6",
    "cases": CASES, "passed": sum(c["status"] == "PASS" for c in CASES),
    "failed": sum(c["status"] == "FAIL" for c in CASES), "skipped": 0,
    "guard_denials": DENIED, "source_files": sources,
    "original_81_test_replay": "UNAVAILABLE: isolated runner and original producer closure are not transferred",
    "scope": "Independent helper/actual-parser controls; no producer transformations, Lean/SMT/native execution, complete graph qualification or human approval.",
}
destination = OUTPUT / ("ADMISSION_SUBSET_XDSL_" + result["xdsl_version"] + ".json")
with destination.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps({"result": str(destination.relative_to(ROOT)), "passed": result["passed"], "failed": result["failed"], "skipped": 0, "guard_denials": DENIED}, indent=2))
raise SystemExit(1 if result["failed"] else 0)
