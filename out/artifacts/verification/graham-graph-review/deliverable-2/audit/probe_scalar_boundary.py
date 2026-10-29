"""Boundary probes of the scalar CSE/DCE/LICM projection (implementation is the SUBJECT).

Graham independent review (agent), 2026-10-26. Each crafted before/after pair
is fed to the implementation's capture CLI (pinned producer checkout, local
xDSL 0.68.0); when it admits the pair, the implementation's own emit_query
produces the Lean query and Lean runs WITHOUT `-o` against the pre-existing
local development build. Raw stdout/stderr, projection JSON and queries are
retained per case. The expectation column is the reviewer's independent
statement of what a sound complete-boundary validator must do. Outcomes are
local observations, not a qualification.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/grammy/blast/merlin-pass-verification")
PRODUCER = REPO / "out/build/upstreams/merlin-graph-proof-intake"
OUT = REPO / "out/artifacts/verification/graham-graph-review/deliverable-2/probes/scalar-boundary"
CAPTURE = REPO / "src/merlin/verify/lean_graph_state_scalar_capture.py"
LEAN = Path("/home/grammy/.elan/toolchains/leanprover--lean4---v4.35.0/bin/lean")
LEAN_LIB = REPO / "out/build/verification-setup/20261025/graph-smoke-MJ63mO/lib/lean"
PYTHON = REPO / ".venv/bin/python"


def module(body, extra="", tail=""):
    return "builtin.module {\n" + extra + "  func.func @k(%a:i8, %b:i8) -> i8 {\n" + body + "  }\n}" + tail


CSE_BODY = "    %x = arith.addi %a, %b : i8\n    %y = arith.addi %a, %b : i8\n    %z = arith.muli %x, %y : i8\n"
CSE_AFTER = "    %x = arith.addi %a, %b : i8\n    %z = arith.muli %x, %x : i8\n"


def resource(blob):
    return '\n{-#\n  dialect_resources: {\n    builtin: {\n      blob1: "0x0100000001' + blob + '"\n    }\n  }\n#-}\n'


TENSOR_FN = (
    "builtin.module {\n  func.func @k() -> tensor<2xi8> {\n"
    "    %c = arith.constant dense_resource<blob1> : tensor<2xi8>\n"
    "    func.return %c : tensor<2xi8>\n  }\n}"
)
TENSOR_DENSE = (
    "builtin.module {\n  func.func @k() -> tensor<2xi8> {\n"
    "    %c = arith.constant dense<[1, VALUE]> : tensor<2xi8>\n"
    "    func.return %c : tensor<2xi8>\n  }\n}"
)
TENSOR_CSE = (
    "builtin.module {\n  func.func @k(%a:i8, %b:i8) -> (i8, tensor<2xi8>) {\n"
    "    %c = arith.constant dense_resource<blob1> : tensor<2xi8>\n"
    "    %x = arith.addi %a, %b : i8\n    %y = arith.addi %a, %b : i8\n    %z = arith.muli %x, %y : i8\n"
    "    func.return %z, %c : i8, tensor<2xi8>\n  }\n}"
)
INDIRECT = """"builtin.module"() ({
  "func.func"() <{sym_name = "callee", function_type = (i8) -> i8}> ({
  ^bb0(%v: i8):
    %w = "arith.subi"(%v, %v) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    "func.return"(%w) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "k", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %f = "func.constant"() <{value = @callee}> : () -> ((i8) -> i8)
BODY    %r = "func.call_indirect"(%f, %z) : ((i8) -> i8, i8) -> i8
    "func.return"(%r) : (i8) -> ()
  }) : () -> ()
}) : () -> ()"""
GEN_CSE = (
    '    %x = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8\n'
    '    %y = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8\n'
    '    %z = "arith.muli"(%x, %y) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8\n'
)
GEN_CSE_AFTER = (
    '    %x = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8\n'
    '    %z = "arith.muli"(%x, %x) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8\n'
)
DIRECT = module(CSE_BODY + "    %r = func.call @callee(%z) : (i8) -> i8\n    func.return %r : i8\n",
                extra="  func.func @callee(%v:i8) -> i8 {\n    func.return %v : i8\n  }\n")
DIRECT_AFTER = module(CSE_AFTER + "    %r = func.call @callee(%z) : (i8) -> i8\n    func.return %r : i8\n",
                      extra="  func.func @callee(%v:i8) -> i8 {\n    func.return %v : i8\n  }\n")
RET = "    func.return %z : i8\n"

CASES = [
    # name, before, after (None = run the producer), expectation
    ("control-cse", module(CSE_BODY + RET), module(CSE_AFTER + RET), "admit+accept"),
    ("resource-blob-changed", TENSOR_FN + resource("02"), TENSOR_FN + resource("03"),
     "refuse: returned tensor contents differ (1,2) vs (1,3)"),
    ("resource-blob-dropped", TENSOR_FN + resource("02"), TENSOR_FN,
     "refuse: after has no resource payload"),
    ("dense-literal-changed", TENSOR_DENSE.replace("VALUE", "2"), TENSOR_DENSE.replace("VALUE", "3"),
     "refuse (control for the resource case)"),
    ("producer-cse-on-resource-module", TENSOR_CSE + resource("02"), None,
     "producer output must retain the resource payload; boundary must not certify its loss"),
    ("indirect-local-call", INDIRECT.replace("BODY", GEN_CSE), INDIRECT.replace("BODY", GEN_CSE_AFTER),
     "refuse: call to a locally defined function (claimed refused)"),
    ("direct-local-call", DIRECT, DIRECT_AFTER, "refuse (implementation states local calls refused)"),
    ("scf-if-region", module(CSE_BODY + "    %t = arith.constant true\n    scf.if %t {\n    }\n" + RET),
     module(CSE_AFTER + "    %t = arith.constant true\n    scf.if %t {\n    }\n" + RET), "refuse whole boundary"),
    ("unregistered-region-op", module(CSE_BODY + '    "x.region"() ({\n    }) : () -> ()\n' + RET),
     module(CSE_AFTER + '    "x.region"() ({\n    }) : () -> ()\n' + RET), "refuse whole boundary"),
    ("module-level-global",
     module(CSE_BODY + RET, extra='  "x.global"() {sym_name = "g"} : () -> ()\n'),
     module(CSE_AFTER + RET, extra='  "x.global"() {sym_name = "g"} : () -> ()\n'), "refuse whole boundary"),
    ("module-attribute-changed", module(CSE_BODY + RET).replace("builtin.module {", "builtin.module attributes {x.a = 1 : i64} {"),
     module(CSE_AFTER + RET).replace("builtin.module {", "builtin.module attributes {x.a = 2 : i64} {"), "refuse"),
    ("function-visibility-changed", module(CSE_BODY + RET),
     module(CSE_AFTER + RET).replace("func.func @k", "func.func private @k"), "refuse"),
    ("opaque-attribute-changed", module(CSE_BODY + '    "x.obs"(%z) {tag = 1 : i64} : (i8) -> ()\n' + RET),
     module(CSE_AFTER + '    "x.obs"(%z) {tag = 2 : i64} : (i8) -> ()\n' + RET), "refuse (different observer)"),
    ("observer-dropped", module(CSE_BODY + '    "x.obs"(%z) : (i8) -> ()\n' + RET), module(CSE_AFTER + RET),
     "refuse"),
    ("flag-introduced", module(CSE_BODY + RET), module(CSE_AFTER.replace("arith.muli %x, %x", "arith.muli %x, %x overflow<nsw>") + RET),
     "refuse (poison-introducing flag)"),
    ("index-constant-changed",
     module("    %l = arith.constant 0 : index\n    \"x.obs\"(%l) : (index) -> ()\n" + CSE_BODY + RET),
     module("    %l = arith.constant 1 : index\n    \"x.obs\"(%l) : (index) -> ()\n" + CSE_AFTER + RET), "refuse"),
    ("location-only", module(CSE_BODY + RET), module(CSE_AFTER.replace(" : i8\n", ' : i8 loc("x.py":1:1)\n', 1) + RET),
     "admit+accept (locations are not semantic)"),
    ("unsigned-for-generic",
     module('    %s = arith.constant 1 : index\n    "scf.for"(%l0, %l0, %s) <{unsignedCmp}> ({\n    ^bb1(%i: index):\n      "scf.yield"() : () -> ()\n    }) : (index, index, index) -> ()\n' + CSE_BODY + RET).replace("{\n    %s", "{\n    %l0 = arith.constant 0 : index\n    %s"),
     module(CSE_AFTER + RET), "refuse; a parse/verify failure is unavailable, not a semantic refusal"),
    ("multi-block-cfg",
     module(CSE_BODY + '    "cf.br"()[^bb1] : () -> ()\n  ^bb1:\n' + RET), module(CSE_AFTER + RET), "refuse"),
    ("extra-function-in-after", module(CSE_BODY + RET),
     module(CSE_AFTER + RET, extra="  func.func private @new(i8) -> i8\n"), "refuse"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(name, before, after, expectation):
    root = OUT / name
    root.mkdir(parents=True)
    (root / "before.mlir").write_text(before)
    command = [str(PYTHON), str(CAPTURE), "--reference", str(PRODUCER), "--source", str(root / "before.mlir"),
               "--destination", str(root / "projection"), "--mode", "cse"]
    if after is not None:
        (root / "after.mlir").write_text(after)
        command += ["--target", str(root / "after.mlir")]
    env = {**os.environ, "PYTHONPATH": str(PRODUCER / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run(command, cwd=PRODUCER, env=env, capture_output=True, text=True, timeout=120)
    (root / "capture.stdout").write_text(done.stdout)
    (root / "capture.stderr").write_text(done.stderr)
    row = {"case": name, "expectation": expectation, "capture_returncode": done.returncode,
           "before_sha256": sha(root / "before.mlir"),
           "after_sha256": sha(root / "after.mlir") if after is not None else None}
    projection_path = root / "projection/projection.json"
    if done.returncode or not projection_path.exists():
        row["outcome"] = "capture_unavailable (not a semantic refusal)"
        row["stderr_tail"] = done.stderr.strip().splitlines()[-1:] if done.stderr else []
        return row
    projection = json.loads(projection_path.read_text())
    if not projection["projection_checked"]:
        row["outcome"] = "projection_refused"
        row["reason"] = projection["reason"]
        return row
    row["genuine_changed_boundary"] = projection["genuine_changed_boundary"]
    produced = root / "projection/after.mlir"
    if after is None:
        row["producer_after_sha256"] = sha(produced)
        row["producer_after_has_resource_section"] = "dialect_resources" in produced.read_text()
    sys.path.insert(0, str(REPO / "src"))
    from merlin.verify.lean_graph_state_scalar import emit_query  # subject, not oracle

    try:
        query, _ = emit_query(projection)
    except ValueError as error:
        row["outcome"] = "query_refused"
        row["reason"] = str(error)
        return row
    (root / "GraphStateScalarInstance.lean").write_text(query)
    lean = subprocess.run([str(LEAN), "GraphStateScalarInstance.lean"], cwd=root,
                          env={**os.environ, "LEAN_PATH": str(LEAN_LIB)}, capture_output=True, text=True, timeout=600)
    (root / "lean.stdout").write_text(lean.stdout)
    (root / "lean.stderr").write_text(lean.stderr)
    row["lean_returncode_no_output_object"] = lean.returncode
    row["outcome"] = "lean_accepted" if lean.returncode == 0 else "lean_rejected"
    row["registry_size"] = len(projection["registry"])
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    rows = [run(*case) for case in CASES]
    report = {"reviewer": "Graham independent review (agent)", "date": "2026-10-26",
              "script_sha256": sha(__file__), "capture_sha256": sha(CAPTURE), "lean_sha256": sha(LEAN),
              "lean_library": str(LEAN_LIB), "parser_environment": "local .venv xDSL 0.68.0 / Python 3.12 "
              "(differs from the original qualification's recorded xDSL files)", "cases": rows}
    (OUT / "PROBES.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(f"{row['case']:34} {row['outcome']:28} {row.get('genuine_changed_boundary', '')} "
              f"{(row.get('reason') or {}).get('detail', '') if isinstance(row.get('reason'), dict) else row.get('reason', '')}"
              f"{row.get('stderr_tail', '')}")


if __name__ == "__main__":
    main()
