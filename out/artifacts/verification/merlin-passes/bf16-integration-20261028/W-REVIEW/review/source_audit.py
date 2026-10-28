"""Independent source/data inventory. No project imports, SDK, Lean or solver execution.

Run from the assigned worktree. Old reports are inspected as historical data only.
This script does not install a policy, write a lock or approve a claim.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
LEAN = ROOT / "merlin/lean"
OLD = Path("/home/jeremy/Projects/merlin/out/artifacts/verification/merlin-passes/frontend/bf16-f32acc")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": sha(data), "bytes": len(data)}


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def assigned(path, name):
    tree = ast.parse(path.read_text())
    return next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def lean_imports(text):
    # Preserve line boundaries while removing nested Lean block and line comments.
    out, depth, i, quoted = [], 0, 0, False
    while i < len(text):
        if depth:
            if text.startswith("/-", i):
                depth += 1; i += 2
            elif text.startswith("-/", i):
                depth -= 1; i += 2
            else:
                if text[i] == "\n": out.append("\n")
                i += 1
        elif not quoted and text.startswith("/-", i):
            depth = 1; i += 2
        elif not quoted and text.startswith("--", i):
            while i < len(text) and text[i] != "\n": i += 1
        else:
            if text[i] == '"' and (i == 0 or text[i-1] != "\\"): quoted = not quoted
            out.append(text[i]); i += 1
    return [word for line in "".join(out).splitlines()
            if line.strip().startswith("import ") for word in line.strip().split()[1:]]


def closure(roots):
    seen, external, edges = {}, set(), {}
    queue = list(roots)
    while queue:
        module = queue.pop()
        if module in seen or module in external: continue
        path = LEAN / (module.replace(".", "/") + ".lean")
        if not path.is_file():
            external.add(module); continue
        seen[module] = ref(path)
        edges[module] = lean_imports(path.read_text())
        queue.extend(edges[module])
    return {"kind": "static_project_source_import_closure_not_object_audit", "roots": sorted(roots),
            "sources": [dict(module=k, **v) for k, v in sorted(seen.items())],
            "edges": edges, "external_unread": sorted(external)}


def main():
    assert ROOT.name == "merlin-bf16-verification", "Run only in assigned worktree"
    host = ROOT / "src/merlin/verify/lean/bf16_f32acc.py"
    publisher = ROOT / "build_tools/scripts/prepare_bf16_review.py"
    roots = ast.literal_eval(assigned(host, "CONTRACT_THEOREMS"))
    for name in ("BOUNDARY", "SEGMENT"):
        _, module, check, qualify = ast.literal_eval(assigned(host, name))
        roots.update({f"{module}.{theorem}": module for theorem in (check, qualify)})
    proof = closure(set(roots.values()) | {"MerlinVerify.Driver.Bf16F32Acc", "Main"})
    write("PROJECT_SOURCE_CLOSURE.json", proof)
    write("THEOREM_ROOTS.json", {"roots": roots, "elaboration": "NOT_RUN_BY_W", "audit": "NOT_RUN_BY_W"})

    old = json.loads((OLD / "review/candidate.json").read_text())
    write("HISTORICAL_COMPARISON.json", {
        "candidate": ref(OLD / "review/candidate.json"),
        "old_patch": ref(OLD / "review/after-approval.patch"),
        "historical_lock": old.get("spec_lock"), "current_lock": ref(LEAN / "SPEC_LOCK"),
        "same_lock": old.get("spec_lock", {}).get("sha256") == sha((LEAN / "SPEC_LOCK").read_bytes()),
        "old_status": old.get("status"), "historical_human_reviewed": old.get("human_reviewed"),
        "statement_roster_equal": set(old["statements"]) == set(roots),
        "fresh_statement_equality": "UNOBSERVED; historical elaboration cannot supply current proof evidence",
        "receipts": [{"boundary": r.get("boundary"), "candidates": r.get("candidates"),
                      "local": r.get("pass_boundary"), "whole_capture_qualification": r.get("whole_capture_qualification")}
                     for r in old["receipts"]]})

    policies = ast.literal_eval(assigned(publisher, "POLICY_FILES"))
    claims = ast.literal_eval(assigned(publisher, "CLAIM_FILES"))
    # Preserve the existing review proposal's scope; add the definitions behind the new
    # complete-output/domain claims. Entire remaining closure is source-bound in admission.
    additions = ["MerlinVerify/Rules/Bf16F32AccSite.lean",
                 "MerlinVerify/Check/Bf16F32AccExact.lean",
                 "MerlinVerify/Check/ParametricModule.lean", "MerlinVerify/Check/PolicyParametric.lean",
                 "MerlinVerify/Check/PolicyModule.lean", "MerlinVerify/Check/PolicyExact.lean",
                 "MerlinVerify/Rules/Bf16F32AccNonvacuity.lean",
                 "MerlinVerify/Rules/Bf16F32AccTensor.lean"]
    frozen = list(dict.fromkeys(policies + claims + additions))
    guard_path = ROOT / "build_tools/scripts/check_lean_guard.py"
    guard = guard_path.read_text()
    current_frozen = ast.literal_eval(assigned(guard_path, "FROZEN"))
    current_pins = ast.literal_eval(assigned(guard_path, "PINNED"))
    new_guard = guard.replace("FROZEN = [\n", "FROZEN = [\n" + "".join(
        f'    "{p}",\n' for p in frozen if p not in current_frozen), 1)
    new_guard = new_guard.replace("PINNED = {\n", "PINNED = {\n" + "".join(
        f'    "{n}": "{m}",\n' for n, m in roots.items() if n not in current_pins), 1)
    contract = "merlin/lean/MerlinVerify/Rules/Bf16F32AccContract.lean"
    spec = "merlin/lean/MerlinVerify/Spec/Bf16F32AccContract.lean"
    original = (ROOT / contract).read_text()
    # Deliberately retain the unreviewed candidate warning in the proposed Spec bytes.
    # Installation/review status must come from an attributable external review, not comments.
    shim = "-- Compatibility import; install only after independent human approval of exact candidate bytes.\nimport MerlinVerify.Spec.Bf16F32AccContract\n"
    patch = "".join(difflib.unified_diff(guard.splitlines(True), new_guard.splitlines(True),
                                      "a/build_tools/scripts/check_lean_guard.py", "b/build_tools/scripts/check_lean_guard.py"))
    patch += "".join(difflib.unified_diff(original.splitlines(True), shim.splitlines(True), "a/"+contract, "b/"+contract))
    patch += "".join(difflib.unified_diff([], original.splitlines(True), "/dev/null", "b/"+spec))
    (OUT / "PROPOSED_AFTER_HUMAN_APPROVAL.patch").write_text(patch)
    write("PROPOSAL.json", {"status": "UNAPPLIED_HUMAN_APPROVAL_ABSENT", "base": "65e44afbf4dbb6c7270e8936c4ef76b579a78492",
        "patch": ref(OUT / "PROPOSED_AFTER_HUMAN_APPROVAL.patch"), "files_to_freeze": frozen,
        "additional_definition_freezes": additions,
        "proposed_spec_sha256": sha(original.encode()), "current_guard": ref(guard_path),
        "theorem_roots": roots,
        "lock_not_written": True,
        "remaining": ["independent attributable human review", "review closure + host admission pins",
                      "fresh authorized elaborated statement and transitive axiom audit",
                      "coordinator installs approved patch and updates lock", "fresh receipts/qualification/replay"]})

    # Pure actual exporter helper: preserve signed zero, reject format confusion. No MLIR import.
    helper_results = []
    for path in [ROOT/"src/merlin/verify/lean/export.py", ROOT/"src/merlin/verify/lean/mlir_export.py"]:
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_float_bits")
        namespace = {"struct": struct, "ExportError": type("ExportError", (Exception,), {})}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec", dont_inherit=True), namespace)
        for fmt in ("bf16", "f16", "f32"):
            positive = namespace["_float_bits"](0.0, fmt)
            negative = namespace["_float_bits"](-0.0, fmt)
            assert positive == 0 and negative != positive
            helper_results.append({"source": ref(path), "format": fmt,
                                   "positive_zero_bits": positive, "negative_zero_bits": negative})
    write("SOURCE_ONLY_PROBES.json", {"kind": "actual_extracted_helper_no_project_imports",
          "checks": helper_results, "native_or_Lean_executed": False,
          "limits": "Does not test parser, decoder, round trip, generated checker or native float fidelity"})

    extra = [host, publisher, guard_path, LEAN/"SPEC_LOCK", LEAN/"lean-toolchain", LEAN/"lakefile.toml",
             LEAN/"lake-manifest.json", ROOT/"src/merlin/verify/lean/export.py",
             ROOT/"src/merlin/verify/lean/mlir_export.py", ROOT/"src/merlin/llvmlower/passes_xdsl.py"]
    write("SOURCE_MANIFEST.json", {"status": "SOURCE_REVIEW_ONLY", "proof_closure": ref(OUT/"PROJECT_SOURCE_CLOSURE.json"),
         "review_inputs": [ref(p) for p in extra], "native_access": False})
    print(json.dumps({"theorem_roots":len(roots), "project_source_modules":len(proof["sources"]),
                      "helper_controls":len(helper_results), "proposal_frozen_files":len(frozen)}))


if __name__ == "__main__":
    main()
