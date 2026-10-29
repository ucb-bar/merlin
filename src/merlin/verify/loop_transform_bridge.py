"""Isolated pinned-producer bridge; no authority from optimizer diagnostics."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"


class AttributionRefusal(ValueError):
    """Valid parsed boundary does not match selected actual transformation."""


def run(request):
    # Script-directory imports deliberately select the submission projection code,
    # while all merlin.* producer imports above belong to pinned reference/src.
    from lean_graph_input_admission import parse_admitted
    from lean_graph_state_pipeline_capture import project_modules

    from merlin.transforms.certificates import ir_text, parse_ir
    from merlin.transforms.loops import segment_loops
    from merlin.transforms.unrolling import unroll_ordered_loops

    reference = Path(request["reference"]).resolve()
    revision = subprocess.run(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if revision != REVISION:
        raise ValueError("producer revision differs from pinned4499")
    buffers = request["buffers"]
    producing = request.get("produce", False)
    if producing and len(buffers) == 1:
        first = parse_admitted(buffers[0], parse_ir)
        first.verify()
        generated = [ir_text(first)]
        unroll_ordered_loops(first, (request["factor"],), index_bits=64)
        generated.append(ir_text(first))
        segment_loops(first, (request["tile"],), index_bits=64)
        generated.append(ir_text(first))
        buffers = generated
    if len(buffers) != 3:
        raise ValueError("three actual buffers required")
    modules = [parse_admitted(text, parse_ir) for text in buffers]
    for module in modules:
        module.verify()
    produced = parse_admitted(buffers[0], parse_ir)
    unroll_ordered_loops(produced, (request["factor"],), index_bits=64)
    produced.verify()
    middle = ir_text(produced)
    segment_loops(produced, (request["tile"],), index_bits=64)
    produced.verify()
    final = ir_text(produced)
    if middle != ir_text(modules[1]) or final != ir_text(modules[2]):
        raise AttributionRefusal("submitted middle/final differ from actual pinned producer output")
    if buffers[0] == buffers[1] or buffers[1] == buffers[2]:
        raise AttributionRefusal("both actual loop boundaries must change")
    stages, controls = project_modules([modules[0], modules[0], modules[0], modules[1], modules[2]])
    sources = []
    for name, loaded in sorted(tuple(sys.modules.items())):
        if name != "merlin" and not name.startswith("merlin."):
            continue
        path = Path(loaded.__file__).resolve()
        if not path.is_relative_to(reference / "src"):
            raise RuntimeError("producer import escaped selected reference")
        data = path.read_bytes()
        relative = str(path.relative_to(reference))
        original = subprocess.run(
            ["git", "-C", str(reference), "show", REVISION + ":" + relative], capture_output=True, check=True
        ).stdout
        if data != original:
            raise RuntimeError("pinned producer source bytes changed")
        sources.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return {
        "actual_buffers": buffers,
        "stages": stages,
        "index_operations": controls,
        "factor": request["factor"],
        "tile": request["tile"],
        "index_bits": 64,
        "producer_revision": revision,
        "producer_sources": sources,
        "identity_prefix": 2,
    }


if __name__ == "__main__":
    try:
        answer = run(json.loads(sys.stdin.read()))
        print(json.dumps({"status": "prepared", "capture": answer}))
    except AttributionRefusal as exc:
        print(json.dumps({"status": "unsupported", "reason": str(exc)}))
        raise SystemExit(1)
    except Exception as exc:
        # Host/parser errors never become successful proof refusals.
        print(json.dumps({"status": "unavailable", "error": type(exc).__name__, "reason": str(exc)}))
        raise SystemExit(1)
