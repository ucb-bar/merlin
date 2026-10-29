"""Independent component projection / authoritative MLIR verification workers.

Projection never produces a gate completion. It deliberately omits optimizer
attribution so mutated, valid actual IR can challenge the unchanged Lean emitter.
"""

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


def project(request):
    reference = Path(request["reference"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != "4499eebb959c0f92cefd92cc6d80646ba66474ed":
        raise ValueError("independent projection selected wrong producer source")
    sys.path.insert(0, request["projection_sources"])
    from lean_graph_input_admission import parse_admitted
    from lean_graph_state_pipeline_capture import project_modules

    from merlin.transforms.certificates import parse_ir

    modules = [parse_admitted(s, parse_ir) for s in request["buffers"]]
    for module in modules:
        module.verify()
    stages, controls = project_modules([modules[0], modules[0], modules[0], modules[1], modules[2]])
    return {
        "stages": stages,
        "index_operations": controls,
        "factor": request["factor"],
        "tile": request["tile"],
        "index_bits": 64,
    }


def verify(request):
    from torch_mlir import ir

    results = []
    for path in request["files"]:
        try:
            with ir.Context() as context:
                context.allow_unregistered_dialects = True
                module = ir.Module.parse(Path(path).read_text())
                valid = bool(module.operation.verify())
            results.append({"path": path, "valid": valid})
        except Exception as error:
            results.append({"path": path, "valid": False, "error": str(error)})
    return results


if __name__ == "__main__":
    request = json.loads(sys.stdin.read())
    try:
        result = project(request) if sys.argv[1] == "project" else verify(request)
        document = {"status": "ok", "result": result}
        if sys.argv[1] == "verify":
            import torch_mlir

            document["oracle"] = {
                "python": sys.version,
                "executable": sys.executable,
                "torch_mlir_module": str(torch_mlir.__file__),
                "distributions": {
                    name: importlib.metadata.version(name)
                    for name in importlib.metadata.packages_distributions().get("torch_mlir", [])
                },
                "allow_unregistered_dialects": True,
                "verification": "Module.parse followed by operation.verify on every complete buffer",
            }
        print(json.dumps(document))
    except Exception as error:
        print(json.dumps({"status": "refused", "error": type(error).__name__, "reason": str(error)}))
        raise SystemExit(1)
