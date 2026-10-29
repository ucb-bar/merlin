"""Source-only native grammar admission; the ordinary exporter stays unchanged.

Run by the selected compiler Python. Only Module.parse's MLIRError denotes a
known source-language exclusion. Context, import, export and resource failures
propagate as tool failures. Neither result grants boundary proof authority.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

SCHEMA = "merlin.compiler-source-export.v1"


def export_source(raw, native):
    text = raw.decode("utf-8")
    context = native.ir.Context()
    context.allow_unregistered_dialects = True
    with context:
        try:
            native.ir.Module.parse(text)
        except native.ir.MLIRError as error:
            return {
                "schema": SCHEMA,
                "source": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
                "supported": False,
                "reason": str(error)[-1200:] or "native MLIR parser refused the original source assembly",
            }
    # A second parse inside the unmodified exporter is deliberate: exceptions
    # during actual export are not reclassified as source grammar exclusions.
    return native.export(text)


def main(source, destination):
    path = Path(__file__).with_name("mlir_export.py")
    spec = importlib.util.spec_from_file_location("_merlin_source_admission_exporter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("the canonical MLIR exporter cannot be loaded")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    if native.ir is None:
        raise RuntimeError("source admission requires torch_mlir in the selected compiler Python")
    document = export_source(Path(source).read_bytes(), native)
    Path(destination).write_text(json.dumps(document))


if __name__ == "__main__":
    main(*sys.argv[1:])
