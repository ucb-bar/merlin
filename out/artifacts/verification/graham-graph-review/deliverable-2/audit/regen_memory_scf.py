"""Regenerate the memory-SCF worked example's model/query locally and bind them to the seal.

Graham independent review (agent), 2026-10-26. The implementation is the
SUBJECT: its capture CLI (pinned producer checkout, local xDSL 0.68.0) exports
the three retained buffers, its project_exports/obligation build model.json
and Obligation.lean, and their hashes are compared with the sealed
EVIDENCE_INDEX entries. The producer is also re-run in production mode on the
retained original to test whether the retained middle/final are its actual
outputs. No Lean run is possible: no compiled GraphLoopsMemorySCF objects
exist locally and building them is out of scope.
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
REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"
OUT = REPO / "out/artifacts/verification/graham-graph-review/deliverable-2/probes/memory-scf"
V = REPO / "out/artifacts/verification"
BUFFERS = {
    "before": V / "graph-buffer-reuse/actual-cases-v3/two_epochs/before.mlir",
    "middle": V / "graph-buffer-reuse/actual-cases-v3/two_epochs/after.mlir",
    "after": V / "graph-buffer-reuse/actual-neighbor-segment-v1/after.mlir",
}
CAPTURE = REPO / "src/merlin/verify/lean_graph_loops_memory_scf_capture.py"
PYTHON = REPO / ".venv/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_capture(args, folder):
    env = {**os.environ, "PYTHONPATH": str(PRODUCER / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([str(PYTHON), str(CAPTURE), "--source", str(PRODUCER), "--revision", REVISION,
                           "--index-bits", "64", *args], env=env, capture_output=True, text=True, timeout=300)
    (folder / "capture.stdout").write_text(done.stdout)
    (folder / "capture.stderr").write_text(done.stderr)
    return done.returncode


def main():
    example = json.loads((V / "graph-memory-scf/memory-scf-mode-v1/EXAMPLE.json").read_text())
    index = json.loads((V / "graph-memory-scf/memory-scf-mode-v1/EVIDENCE_INDEX.json").read_text())
    sealed = {row["path"].split("/memory-scf-mode-v1/", 1)[1]: row["sha256"] for row in index
              if "/memory-scf-mode-v1/" in row["path"]}
    proof = "qualification/pytest/test_actual_complete_common_me0/proof/"
    OUT.mkdir(parents=True, exist_ok=False)
    report = {"reviewer": "Graham independent review (agent)", "date": "2026-10-26",
              "note": "local re-execution of the implementation under local xDSL; not the original qualification",
              "buffers": {k: {"path": str(p), "sha256": sha(p),
                              "matches_EXAMPLE.json": sha(p) == example["originals"][i]["sha256"]}
                          for i, (k, p) in enumerate(BUFFERS.items())}}
    # 1. supplied triple -> exports -> model/query (subject)
    triple = OUT / "supplied"
    triple.mkdir()
    rc = run_capture(["--before", str(BUFFERS["before"]), "--middle", str(BUFFERS["middle"]),
                      "--after", str(BUFFERS["after"]), "--output", str(triple / "actual-triple.json")], triple)
    report["supplied_capture_returncode"] = rc
    if rc == 0:
        sys.path.insert(0, str(REPO / "src"))
        from merlin.verify.lean_graph_loops_memory_scf_model import obligation
        from merlin.verify.lean_graph_loops_memory_scf_projection import project_exports

        tree = json.loads((triple / "actual-triple.json").read_text())
        projection = project_exports(tree["before_export"], tree["middle_export"], tree["after_export"],
                                     index_bits=64, max_nodes=5000)
        (triple / "model.json").write_text(json.dumps(projection, sort_keys=True, indent=2) + "\n")
        (triple / "Obligation.lean").write_text(obligation(projection))
        for name in ("model.json", "Obligation.lean"):
            report[name] = {"sealed_sha256": sealed.get(proof + name), "local_sha256": sha(triple / name),
                            "match": sealed.get(proof + name) == sha(triple / name)}
    # 2. producer re-execution on the retained original
    produced = OUT / "producer"
    rc = run_capture(["--before", str(BUFFERS["before"]), "--output", str(produced / "run")], OUT)
    report["producer_capture_returncode"] = rc
    for name in ("middle", "after"):
        path = produced / "run" / f"{name}.mlir"
        report[f"producer_{name}"] = {"local_sha256": sha(path) if path.exists() else None,
                                      "retained_sha256": sha(BUFFERS[name]),
                                      "byte_identical": path.exists() and sha(path) == sha(BUFFERS[name])}
    (OUT / "REGENERATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "buffers"}, indent=1))


if __name__ == "__main__":
    main()
