"""Independent compiler-MLIR validity qualification, separate from kernel proof.

Uses an explicitly selected existing torch-mlir interpreter. No dependency is
installed and this does not supply the unavailable matching Lean-MLIR semantics.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from merlin.verify import lean_graph_regions as base


def compiler_verify(source: Path, destination: Path, *, python: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    original = base.file_ref(source)
    snapshot = destination / "source.mlir"
    snapshot.write_bytes(base.fresh_ref(original).read_bytes())
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    command = [str(python), "-B", str(Path(__file__).resolve()), "--worker", str(snapshot)]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
    (destination / "stdout").write_text(result.stdout)
    (destination / "stderr").write_text(result.stderr)
    (destination / "process.json").write_text(json.dumps({"command": command, "returncode": result.returncode}))
    if result.returncode:
        raise RuntimeError("compiler MLIR oracle unavailable or failed; see process/stderr")
    verdict = json.loads(result.stdout)
    if type(verdict.get("valid")) is not bool:
        raise ValueError("compiler verifier returned no Boolean verdict")
    if base.fresh_ref(original).read_bytes() != snapshot.read_bytes():
        raise ValueError("compiler source original/snapshot content differs")
    receipt = {
        "schema": "merlin.graph-regions.compiler-validity.v1",
        "original": original,
        "snapshot": base.file_ref(snapshot),
        "worker": base.file_ref(Path(__file__)),
        "oracle": verdict,
        "formal_semantics_proved": False,
    }
    (destination / "EVIDENCE.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    from torch_mlir import ir

    result = {
        "python": sys.version,
        "python_executable": base.file_ref(Path(sys.executable)),
        "torch_mlir_version": importlib.metadata.version("torch-mlir"),
        "ir_module": base.file_ref(Path(ir.__file__)),
    }
    try:
        with ir.Context() as context:
            module = ir.Module.parse(args.worker.read_text(), context)
            result["valid"] = bool(module.operation.verify())
    except ir.MLIRError as error:
        result.update(valid=False, diagnostic=str(error))
    result["native_modules"] = {
        name: base.file_ref(Path(mod.__file__))
        for name, mod in sorted(sys.modules.items())
        if name.startswith("torch_mlir.") and getattr(mod, "__file__", None) and str(mod.__file__).endswith(".so")
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
