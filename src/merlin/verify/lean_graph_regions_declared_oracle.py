"""Explicit native compiler/parser oracles and their actually loaded host files.

This standalone worker does not import the affected Lean source-bridge profile.
MLIR validity and Kernel structural validity are separate from Lean soundness.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import runpy
import sys
from pathlib import Path


def ref(path):
    path = path.resolve(strict=True)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("host file changed while being read")
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def host():
    modules = {
        name: ref(Path(module.__file__))
        for name, module in sorted(sys.modules.items())
        if getattr(module, "__file__", None)
    }
    mapped = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            mapped[fields[5]] = ref(Path(fields[5]))
    return dict(
        modules=modules,
        mapped=mapped,
        python=ref(Path(sys.executable)),
        python_version=sys.version,
        configuration=ref(Path(sys.executable).parent.parent / "pyvenv.cfg"),
    )


def kernel(document):
    from merlin.compile.scheduling.schedule_pipeline import _verify_kernel
    from merlin.sched.ir import Call, Const, Kernel, TensorArg

    calls = []
    for call in document["body"]:
        calls.append(
            Call(
                call["instr"],
                tuple((key, Const(value["value"])) for key, value in call["args"]),
                assumes=tuple(call["assumes"]),
                awaits=tuple(call["awaits"]),
                produces=call["produces"],
                sets=tuple(call["sets"]),
                stages=tuple(call["stages"]),
                unit=call["unit"],
            )
        )
    actual = Kernel(
        document["name"],
        tuple(TensorArg(row["name"], tuple(row["shape"]), row["dtype"], row["access"]) for row in document["args"]),
        tuple(calls),
        tuple(map(tuple, document["attrs"])),
    )
    _verify_kernel(actual)
    return actual.text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("mlir", "kernel"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--exporter", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    source = ref(args.source)
    if args.mode == "mlir":
        from torch_mlir import ir

        if args.exporter is None:
            raise ValueError("explicit authoritative exporter required")
        namespace = runpy.run_path(str(args.exporter.resolve()), run_name="declared_authoritative_export")
        with ir.Context() as context:
            module = ir.Module.parse(args.source.read_text(), context)
            valid = bool(module.operation.verify())
        document = namespace["export"](args.source.read_text())
        (args.destination / "export.json").write_text(json.dumps(document, sort_keys=True) + "\n")
        detail = dict(version=importlib.metadata.version("torch-mlir"), exporter=ref(args.exporter))
    else:
        representation = kernel(json.loads(args.source.read_text()))
        (args.destination / "kernel.txt").write_text(representation)
        valid, detail = True, dict(check="pinned schedule_pipeline._verify_kernel; structural validity only")
    if source != ref(args.source):
        raise ValueError("original oracle input changed")
    (args.destination / "ORACLE.json").write_text(
        json.dumps(
            dict(source=source, valid=valid, detail=detail, worker=ref(Path(__file__)), host=host()),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
