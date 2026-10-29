"""Independent actual producer capture / authoritative buffer verification.

The producer runs with pinned-reference merlin imports in an isolated process.
No receipt, hint, numerical evaluator or gate authority is consumed here.
"""

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"


def produce(request):
    from merlin.transforms import loops, normalization, planning
    from merlin.transforms.certificates import ir_text, parse_ir

    reference = Path(request["reference"]).resolve()
    observed = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    if observed != REVISION:
        raise ValueError("actual producer revision differs from pinned4499")
    source = request["source"]
    module = parse_ir(source)
    module.verify()
    buffers = [source]
    normalized = normalization.normalize_shapes(module)
    module.verify()
    buffers.append(ir_text(module))
    plan = planning.plan_storage(module)
    reused = planning.reuse_storage(module, plan)
    module.verify()
    buffers.append(ir_text(module))
    segmented = loops.segment_loops(module, [request["tile"]], index_bits=64)
    module.verify()
    buffers.append(ir_text(module))
    sources = []
    for name, loaded in sorted(tuple(sys.modules.items())):
        if name != "merlin" and not name.startswith("merlin."):
            continue
        path = Path(loaded.__file__).resolve()
        if not path.is_relative_to(reference / "src"):
            raise ValueError("producer import escaped pinned reference")
        data = path.read_bytes()
        relative = str(path.relative_to(reference))
        original = subprocess.check_output(["git", "-C", str(reference), "show", REVISION + ":" + relative])
        if data != original:
            raise ValueError("loaded producer source differs from exact pinned commit")
        sources.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return {
        "buffers": buffers,
        "producer_revision": observed,
        "producer_sources": sources,
        "APIs": [
            "normalization.normalize_shapes",
            "planning.plan_storage",
            "planning.reuse_storage",
            "loops.segment_loops",
        ],
        "changes": {"normalization": len(normalized), "reuse": len(reused), "segmentation": len(segmented)},
        "index_bits": 64,
        "tile": request["tile"],
        "authority": False,
        "projection_trusted": True,
    }


def verify(request):
    from torch_mlir import ir

    rows = []
    for path in request["files"]:
        try:
            source = Path(path).read_bytes()
            ref = {
                "path": str(Path(path).resolve()),
                "sha256": hashlib.sha256(source).hexdigest(),
                "bytes": len(source),
            }
            with ir.Context() as context:
                context.allow_unregistered_dialects = True
                module = ir.Module.parse(source.decode())
                valid = bool(module.operation.verify())
            rows.append({"path": path, "valid": valid, "parsed_bytes": ref})
        except Exception as error:
            rows.append({"path": path, "valid": False, "error": str(error)})
    return {
        "files": rows,
        "oracle": {
            "python": sys.version,
            "executable": sys.executable,
            "distributions": {
                name: importlib.metadata.version(name)
                for name in importlib.metadata.packages_distributions().get("torch_mlir", [])
            },
            "allow_unregistered_dialects": True,
            "verification": "Module.parse then operation.verify on each complete buffer",
        },
    }


if __name__ == "__main__":
    request = json.loads(sys.stdin.read())
    try:
        result = produce(request) if sys.argv[1] == "produce" else verify(request)
        print(json.dumps({"status": "ok", "result": result}))
    except Exception as error:
        print(json.dumps({"status": "unavailable", "error": type(error).__name__, "reason": str(error)}))
        raise SystemExit(1)
