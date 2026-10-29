"""Does the four-pass pipeline projection refuse an indirect call to a locally defined function?

Graham independent review (agent), 2026-10-26. Subject: lean_graph_state_pipeline_capture.project_modules
(which reuses the scalar `project`). The probe module from probes/scalar-boundary/indirect-local-call/before.mlir
is supplied as all five stages (an identity chain suffices to observe admission). Run with
PYTHONPATH=<pinned producer>/src. No Lean run.
"""
import json
import sys
from pathlib import Path

REPO = Path("/home/grammy/blast/merlin-pass-verification")
D2 = REPO / "out/artifacts/verification/graham-graph-review/deliverable-2"
sys.path.insert(0, str(REPO / "src/merlin/verify"))
from merlin.transforms.certificates import parse_ir  # pinned producer parser (subject)
from lean_graph_state_pipeline_capture import project_modules  # subject

text = (D2 / "probes/scalar-boundary/indirect-local-call/before.mlir").read_text()
modules = [parse_ir(text) for _ in range(5)]
row = {"probe": "indirect-local-call as five stages"}
try:
    stages, concrete = project_modules(modules)
    registry = [r["descriptor"]["attributes"] for r in stages[0]["registry"]]
    row.update(outcome="projection admitted", registry_size=len(registry),
               registry_op_names=[json.dumps(a)[:160] for a in registry],
               defined_functions=[f["name"] for f in stages[0]["functions"] if not f["declaration"]])
except ValueError as error:
    row.update(outcome="refused", reason=str(error))
out = D2 / "probes/pipeline-indirect-call/RESULT.json"
out.write_text(json.dumps(row, indent=2) + "\n")
print(json.dumps(row, indent=1))
