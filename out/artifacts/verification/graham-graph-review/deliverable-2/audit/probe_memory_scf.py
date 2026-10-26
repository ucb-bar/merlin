"""Which layer refuses a wrong final program in the memory-SCF route?

Graham independent review (agent), 2026-10-26. Compiler-valid mutations of the
retained final buffer are fed to the implementation (SUBJECT): capture CLI,
then project_exports. For each we record whether the trusted Python matcher
refuses, and, if it emits a model, the Obligation.lean bytes. A COUNTERFACTUAL
column re-runs project_exports in this process with the matcher's per-operation
equality test (lean_graph_loops_memory_scf_projection.py:137-138) disabled, and
compares the resulting Obligation.lean with the positive one. Identical bytes
mean the Lean query cannot distinguish that final program from the correct one,
so only the Python matcher stands between it and acceptance. The counterfactual
patch exists only in this process's memory; no source file is modified. No Lean
run is possible locally (no compiled GraphLoopsMemorySCF objects).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_memory_scf as A  # noqa: E402
import mlir_subset as M  # noqa: E402
import regen_memory_scf as R  # noqa: E402

OUT = A.D2 / "probes/memory-scf-mutations"
sys.path.insert(0, str(A.REPO / "src"))
from merlin.verify import lean_graph_loops_memory_scf_projection as P  # noqa: E402  (subject)
from merlin.verify.lean_graph_loops_memory_scf_model import obligation  # noqa: E402  (subject)

AFTER = A.BUFFERS["after"].read_text()
MUTATIONS = {
    "untouched-function-constant": ('<{value = 1 : i32}> : () -> i32\n    %result = "arith.addi"(%input',
                                    '<{value = 2 : i32}> : () -> i32\n    %result = "arith.addi"(%input'),
    "callee-body-addi-to-subi": ('%new = "arith.addi"(%old, %value)', '%new = "arith.subi"(%old, %value)'),
    "second-initializer-stores-seed0": ('"memref.store"(%seed1, %buffer0, %9, %iv1)',
                                        '"memref.store"(%seed0, %buffer0, %9, %iv1)'),
    "second-observer-reads-value0": ('"func.call"(%value1, %caller)', '"func.call"(%value0, %caller)'),
    "first-min-upper-operand-one": ('%3 = "arith.minui"(%2, %bound0)', '%3 = "arith.minui"(%2, %one)'),
    "final-free-removed": ('    "memref.dealloc"(%buffer0) : (memref<2x3xi32>) -> ()\n    %result',
                           '    %result'),
}


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def counterfactual_operation(self, first, second, env):
    """Copy of SegmentBinding.operation WITHOUT the line-137 metadata/operand equality test."""
    self.bind(first["results"], second["results"], env)
    if len(first["regions"]) != len(second["regions"]):
        raise ValueError("retained region count changed")
    a_regions, b_regions = [], []
    for a, b in zip(first["regions"], second["regions"], strict=True):
        if len(a) != len(b):
            raise ValueError("retained block count changed")
        pairs = [self.block(x, y, dict(env)) for x, y in zip(a, b, strict=True)]
        a_regions.append([x for x, _ in pairs])
        b_regions.append([y for _, y in pairs])
    common = dict(P._metadata(first), operands=first["operands"], results=first["results"])
    return dict(common, regions=a_regions), dict(common, regions=b_regions)


def project(tree):
    return P.project_exports(tree["before_export"], tree["middle_export"], tree["after_export"], index_bits=64,
                             max_nodes=5000)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    positive_query = (A.REGEN / "Obligation.lean").read_text()
    rows = []
    for name, (old, new) in MUTATIONS.items():
        assert AFTER.count(old) == 1, name
        folder = OUT / name
        folder.mkdir()
        mutated = folder / "after.mlir"
        mutated.write_text(AFTER.replace(old, new))
        row = {"mutation": name, "replaced": old, "with": new, "after_sha256": sha_text(mutated.read_text())}
        rc = R.run_capture(["--before", str(A.BUFFERS["before"]), "--middle", str(A.BUFFERS["middle"]),
                            "--after", str(mutated), "--output", str(folder / "actual-triple.json")], folder)
        row["capture_returncode"] = rc
        if rc:
            row["layer"] = "capture/parse unavailable (not a semantic refusal)"
            rows.append(row)
            continue
        tree = json.loads((folder / "actual-triple.json").read_text())
        try:
            projection = project(tree)
            query = obligation(projection)
            (folder / "Obligation.lean").write_text(query)
            row["python_matcher"] = "accepted"
            row["query_identical_to_positive"] = query == positive_query
            seg = [c["segment"] for c in projection["configurations"]]
            row["reviewer_eval_of_Lean_Structural_min_operands"] = [
                s["minOperands"] == [s["addResult"], s["upperValue"]] for s in seg]
            row["layer"] = "Lean Structural predicate (python emits model)"
        except ValueError as error:
            row["python_matcher"] = f"refused: {error}"
            row["layer"] = "trusted Python matcher"
        original = P.SegmentBinding.operation
        P.SegmentBinding.operation = counterfactual_operation
        try:
            query = obligation(project(tree))
            (folder / "Obligation.counterfactual.lean").write_text(query)
            row["counterfactual_query_identical_to_positive"] = query == positive_query
        except ValueError as error:
            row["counterfactual"] = f"refused elsewhere: {error}"
        finally:
            P.SegmentBinding.operation = original
        mutant = M.read(mutated.read_text())
        row["reviewer_spec_vs_mutant"] = A.first_difference(
            A.canonical(A.expected_after(M.read(A.BUFFERS["middle"].read_text()), 2)), A.canonical(mutant)) or "equal"
        semantics = [A.interpret(mutant, [s % (1 << 32) for s in seeds], 3) for seeds in ((1, 2), (17, 34))]
        reference = [A.interpret(M.read(A.BUFFERS["before"].read_text()), [s % (1 << 32) for s in seeds], 3)
                     for seeds in ((1, 2), (17, 34))]
        row["reviewer_semantics_differs_from_original"] = semantics != reference
        row["reviewer_semantics_sample"] = {"original": repr(reference[1]), "mutant": repr(semantics[1])}
        rows.append(row)
    report = {"reviewer": "Graham independent review (agent)", "date": "2026-10-26",
              "positive_query_sha256": sha_text(positive_query), "rows": rows}
    (OUT / "MUTATIONS.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(f"{row['mutation']:34} {row.get('python_matcher', '')[:70]:72} cf_identical="
              f"{row.get('counterfactual_query_identical_to_positive')} semantics_differs="
              f"{row.get('reviewer_semantics_differs_from_original')}")


if __name__ == "__main__":
    main()
