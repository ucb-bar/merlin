"""Authoritative parser+verifier and existing exporter, in compiler Python."""

import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

from torch_mlir import ir


def main():
    if sys.argv[1:] == ["--version"]:
        print(json.dumps({"python": sys.version, "torch_mlir": importlib.metadata.version("torch-mlir")}))
        return
    source, output, exporter = map(Path, sys.argv[1:])
    text = source.read_text()
    with ir.Context() as context:
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(text)
        if not module.operation.verify():
            raise ValueError("authoritative MLIR verifier refused")
    spec = importlib.util.spec_from_file_location("qualification_exporter", exporter)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    document = implementation.export(text)
    output.write_text(json.dumps(document))
    print(json.dumps({"verified": True, "exported": str(output)}))


if __name__ == "__main__":
    main()
