"""Test-only adapter: run real MLIR passes in the selected compiler interpreter."""

import json
import runpy
import sys
from pathlib import Path

from torch_mlir import ir
from torch_mlir.passmanager import PassManager


def main():
    exporter, source, destination, pipeline = sys.argv[1:]
    export = runpy.run_path(exporter)["export"]
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(Path(source).read_text())
        PassManager.parse(pipeline).run(module.operation)
        text = module.operation.get_asm(print_generic_op_form=True)
    Path(destination).write_text(json.dumps(export(text)))


if __name__ == "__main__":
    main()
