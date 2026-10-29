"""Probe the installed map contract in the compiler environment, independently of Lean.

LLVM f5e175f added the destination block argument. Documentation examples and package versions
alone do not establish that this compiler implements that convention.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import sys
from pathlib import Path


def _fingerprint(path: str) -> dict:
    binary = Path(path).resolve()
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(binary), "sha256": digest}


def probe() -> dict:
    """Require inputs followed by one destination argument, including in shorthand expansion."""
    from torch_mlir import ir, passmanager
    from torch_mlir._mlir_libs import _mlir

    cases = []
    with ir.Context():
        for count in (0, 1, 2):
            arguments = [f"%a{i}: tensor<2xi32>" for i in range(count)] + ["%init: tensor<2xi32>"]
            inputs = (
                (
                    "ins("
                    + ", ".join(f"%a{i}" for i in range(count))
                    + " : "
                    + ", ".join(["tensor<2xi32>"] * count)
                    + ")"
                )
                if count
                else ""
            )
            prefix = (
                f"func.func @probe({', '.join(arguments)}) -> tensor<2xi32> {{ "
                f"%r = linalg.map {inputs} outs(%init : tensor<2xi32>) "
            )
            suffix = " return %r : tensor<2xi32> }"
            block_args = [f"%x{i}: i32" for i in range(count)] + ["%old: i32"]
            source = prefix + f"({', '.join(block_args)}) {{ linalg.yield %old : i32 }}" + suffix
            try:
                module = ir.Module.parse(source)
            except ir.MLIRError as exc:
                raise RuntimeError("unsupported linalg.map contract: expected inputs plus destination") from exc
            op = module.body.operations[0].regions[0].blocks[0].operations[0]
            if len(op.regions[0].blocks[0].arguments) != count + 1:
                raise RuntimeError("unsupported linalg.map contract: expected inputs plus destination")
            # Verify that the installed generalizer preserves the exact region signature and read.
            passmanager.PassManager.parse("builtin.module(func.func(linalg-generalize-named-ops))").run(
                module.operation
            )
            generic = module.body.operations[0].regions[0].blocks[0].operations[0]
            block = generic.regions[0].blocks[0]
            if (
                generic.operation.name != "linalg.generic"
                or len(block.arguments) != count + 1
                or block.operations[0].operands[0] != block.arguments[count]
            ):
                raise RuntimeError("unsupported linalg.map generalization contract")
            # A legacy body with only the input block arguments must be rejected by the verifier.
            if count:
                legacy = prefix + f"({', '.join(block_args[:-1])}) {{ linalg.yield %x0 : i32 }}" + suffix
                try:
                    ir.Module.parse(legacy)
                except ir.MLIRError:
                    pass
                else:
                    raise RuntimeError("unsupported linalg.map contract: legacy body accepted")
            cases.append({"inputs": count, "body_arguments": count + 1, "reads_destination": True})
        short = ir.Module.parse("""func.func @short(%a: tensor<2xi32>, %b: tensor<2xi32>,
            %init: tensor<2xi32>) -> tensor<2xi32> {
          %r = linalg.map { arith.addi } ins(%a, %b : tensor<2xi32>, tensor<2xi32>)
            outs(%init : tensor<2xi32>)
          return %r : tensor<2xi32>
        }""")
        block = short.body.operations[0].regions[0].blocks[0].operations[0].regions[0].blocks[0]
        payload = block.operations[0]
        if (
            len(block.arguments) != 3
            or payload.operation.name != "arith.addi"
            or list(payload.operands) != list(block.arguments)[:2]
        ):
            raise RuntimeError("unsupported linalg.map shorthand contract")
    return {
        "contract": "map-inputs-then-destination-v1",
        "cases": cases,
        "shorthand_arguments": 3,
        "torch_mlir_package_version": importlib.metadata.version("torch-mlir"),
        "compiler_python": _fingerprint(sys.executable),
        "mlir_binding": _fingerprint(_mlir.__file__),
        "mlir_capi": _fingerprint(str(Path(_mlir.__file__).with_name("libTorchMLIRAggregateCAPI.so"))),
    }
