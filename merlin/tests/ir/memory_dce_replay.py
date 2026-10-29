"""Fresh-process raw-pair replay helper; saved approval is never authority."""

import json
import os
import sys
from pathlib import Path

from merlin.verify import compiler_memory_dce as gate

request = json.loads(Path(sys.argv[1]).read_text())
destination = Path(sys.argv[2]).resolve()
destination.mkdir(exist_ok=False)
initial = len(gate._LIVE)
assert initial == 0
completion, candidate = gate.replay(
    request["report"],
    lean=request["lean"],
    destination=destination / "proof",
    timeout=300,
)
report = gate.validate_completion(completion, candidate)
assert gate.deliver(completion, candidate) == candidate.final
(destination / "FRESH-RESULT.json").write_text(
    json.dumps(
        {
            "pid": os.getpid(),
            "initial_live_entries": initial,
            "source_sha256": gate.digest(candidate.source),
            "final_sha256": gate.digest(candidate.final),
            "fresh_audit_violations": report["audit"]["violations"],
            "roots": list(candidate.roots),
            "report": completion.report,
        },
        indent=2,
    )
    + "\n"
)
