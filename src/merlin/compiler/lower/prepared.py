"""Shared prepared contraction/object compilation with selected OOT callbacks.

Providers supply contract selection, geometry, source paths, rewriting and object
construction. This owner validates service options and records actual products;
partial IR and matrix objects do not establish a complete executable or execution.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from merlin.compiler.artifact import Artifact, Compilation


def _selection(request, *, namespace, select):
    values = request.options.parameters
    unit = values.get(f"{namespace}.unit")
    config = values.get(f"{namespace}.config")
    for key, value in (("unit", unit), ("config", config)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{namespace}.{key} must explicitly select the hardware contract")
    provider, contract, edge, alignment, sources = select(unit, config)
    if type(edge) is not int or edge <= 0 or type(alignment) is not int or alignment <= 0:
        raise ValueError("selected contract produced invalid matrix geometry")
    return provider, contract, edge, alignment, sources


def validate(request, *, namespace, select):
    if request.source.kind != "upstream-mlir":
        raise ValueError(f"{namespace} requires explicitly prepared upstream-mlir input")
    options = request.options
    if any(value is not None for value in (options.system, options.platform, options.harts)):
        raise ValueError("matrix lowering does not build platform images or schedule harts")
    values = options.parameters
    keys = {f"{namespace}.{key}" for key in ("unit", "config", "emit", "cc", "cflags")}
    if set(values) - keys:
        raise ValueError(f"unsupported {namespace} parameters: {sorted(set(values) - keys)}")
    emit = values.get(f"{namespace}.emit")
    if emit not in ("prepared-ir", "object"):
        raise ValueError(f"{namespace}.emit must explicitly select prepared-ir or object")
    cc = values.get(f"{namespace}.cc")
    flags = values.get(f"{namespace}.cflags")
    if emit == "object":
        if not isinstance(cc, str) or not cc.strip() or shutil.which(cc) is None:
            raise FileNotFoundError(f"{namespace}.cc must select an available C compiler")
        if not isinstance(flags, list) or not flags or not all(isinstance(f, str) and f for f in flags):
            raise ValueError(f"{namespace}.cflags must explicitly select nonempty target compiler flags")
    elif cc is not None or flags is not None:
        raise ValueError("prepared-ir emission does not consume compiler options")
    selected = _selection(request, namespace=namespace, select=select)
    provider, contract, _edge, _alignment, _sources = selected
    if emit == "object":
        derived = provider.derive_encodings(contract)
        if not derived.ok:
            raise ValueError("selected encoding derivation failed; refusing matrix object emission")

    return selected


def compile(request, *, namespace, select, remaining):
    provider, _contract, edge, alignment, hardware_sources = validate(request, namespace=namespace, select=select)
    output = request.output
    if output.exists():
        raise FileExistsError(f"compilation requires a fresh output directory: {output}")
    values = request.options.parameters
    output.mkdir(parents=True)
    try:
        source = output / "prepared.mlir"
        source.write_text(request.source.text, encoding="utf-8")
        rewrite = provider.rewrite_prepared_file(
            source,
            output,
            select=provider.selector(edge),
            tile_edge=edge,
        )
        signatures = provider.load_signatures(output)
        if not signatures:
            raise ValueError("no contractions admitted by the selected matrix implementation")
        if signatures != rewrite.signatures:
            raise ValueError("rewritten calls and matrix signature sidecar disagree")
        paths = [
            (source, "partial-mlir", "region"),
            (output / provider.SIDECAR_NAME, "matrix-signatures", "abi"),
        ]
        build = None
        if values[f"{namespace}.emit"] == "object":
            build = provider.build_object(
                signatures,
                output,
                unit=values[f"{namespace}.unit"],
                config=values[f"{namespace}.config"],
                cc=values[f"{namespace}.cc"],
                cflags=values[f"{namespace}.cflags"],
            )
            if build.signatures != signatures:
                raise ValueError("built matrix object signatures differ from rewritten calls")
            if build.tile_edge != edge or build.alignment_bytes != alignment:
                raise ValueError("object builder geometry differs from the rewritten matrix calls")
            paths.extend(
                (
                    (build.object_path, "object", "matrix-region"),
                    (build.source_path, "c-source", "matrix-region"),
                )
            )
            encodings = output / "encodings.json"
            if encodings.is_file():
                paths.append((encodings, "instruction-derivation", "provenance"))
        report = {
            "scope": "prepared_contractions_and_matrix_object" if build else "prepared_contractions",
            "unit": values[f"{namespace}.unit"],
            "config": values[f"{namespace}.config"],
            "tile_edge": edge,
            "alignment_bytes": alignment,
            "routed": rewrite.count,
            "rewrite": rewrite.to_dict(),
            "hardware_sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in hardware_sources},
            "matrix_build": build.to_dict() if build else None,
            "remaining": list(remaining),
            "execution": "not_run",
            "qualification": "not_established",
            "compiler": shutil.which(values[f"{namespace}.cc"]) if build else None,
            "compiler_flags": values.get(f"{namespace}.cflags"),
            "compiler_sha256": hashlib.sha256(Path(shutil.which(values[f"{namespace}.cc"])).read_bytes()).hexdigest()
            if build
            else None,
        }
        receipt = output / "matrix-build.json"
        receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        paths.append((receipt, "matrix-build-receipt", "provenance"))
        artifacts = tuple(Artifact.from_file(path, kind=kind, role=role) for path, kind, role in paths)
        return Compilation(output, namespace, request.source.sha256, artifacts, metadata=report)
    except Exception as exc:
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "type": type(exc).__name__,
                    "reason": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
