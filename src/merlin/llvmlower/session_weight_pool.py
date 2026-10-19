"""Explicit immutable storage for programs with different constant layouts.

Offsets address complete program-local blobs. Equal complete blobs may share
storage; equal tensor shapes, names or checkpoint provenance grant no sharing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import c_runtime


def plan_program_weight_pool(programs: list[dict]) -> dict:
    """Plan aligned, disjoint spans using each generated argument table's blob."""
    alignment = c_runtime.GENERATED_WEIGHT_ALIGNMENT
    names, spans, records, end = set(), {}, [], 0
    for row in programs:
        name, size, digest = row.get("name"), row.get("weights_bytes"), row.get("weights_sha256")
        if (not isinstance(name, str) or name in names or type(size) is not int or size < 0
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or (size == 0 and digest != hashlib.sha256(b"").hexdigest())):
            raise ValueError("program weight pool has an invalid name or blob identity")
        names.add(name)
        key = (size, digest)
        if not size:
            offset = 0
        elif key in spans:
            offset = spans[key]
        else:
            offset = (end + alignment - 1) // alignment * alignment
            spans[key] = offset
            end = offset + size
        records.append({"name": name, "offset": offset, "bytes": size, "sha256": digest})
    return {"schema": "merlin.program_weight_pool.v1", "alignment": alignment,
            "bytes": end, "programs": records}


def materialize_program_weight_pool(session, manifest: dict, out_dir: str | Path, *,
                                    entry_projections=None, weights_sources=None) -> dict:
    """Rebuild every program's argument-table bytes before concatenating spans.

    Existing blobs are only storage candidates: c_runtime still checks their
    complete bytes against the freshly derived capture and selected projection.
    Rechecking each program also covers deduplicated and empty spans.
    """
    rows = manifest.get("programs", [])
    if (len(rows) != len(session.programs)
            or tuple(row.get("name") for row in rows) != session.program_names):
        raise ValueError("program weight pool differs from the session schedule")
    plan = plan_program_weight_pool(rows)
    if manifest.get("program_weight_pool") != plan:
        raise ValueError("selected program weight pool differs from generated argument tables")
    projections, sources = dict(entry_projections or {}), dict(weights_sources or {})
    if not set(projections) <= set(session.program_names) or not set(sources) <= set(session.program_names):
        raise ValueError("program weight pool selects an unknown program")
    out = Path(out_dir).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError("program weight pool needs a fresh workspace")
    out.mkdir(parents=True)
    blobs = []
    for index, (program, row) in enumerate(zip(session.programs, rows, strict=True)):
        work = out / f"program_{index}"
        options = {}
        if program.name in projections:
            from .session_projection import generation_options

            options, _ = generation_options(projections[program.name], entrypoint=row["entrypoint"], out=work)
        if (row.get("entry_projection") is not None) != bool(options):
            raise ValueError("program weight pool needs its selected entry projection")
        if program.name in sources:
            source = Path(sources[program.name])
            if source.is_symlink() or not source.is_file():
                raise ValueError("program weight storage source must be a plain file")
            options["weights_source"] = source
        info = c_runtime.generate(program.bundle, work, program.bundle / "inputs.npz",
                                  ciface_name=row["entrypoint"], invoke_name=row["invoke"],
                                  materialize_weights=True, **options)
        blob = work / "weights.bin"
        if (blob.is_symlink() or not blob.is_file() or blob.stat().st_size != row["weights_bytes"]
                or any(info.get(key) != row.get(key) for key in
                       ("weights_bytes", "weights_sha256", "entry_projection"))):
            raise ValueError("materialized program weights differ from generated argument tables")
        with blob.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["weights_sha256"]:
                raise ValueError("materialized program weight bytes changed")
        blobs.append(blob)
    target, seen = out / "weights.bin", set()
    with target.open("xb") as destination:
        for blob, span in zip(blobs, plan["programs"], strict=True):
            key = (span["bytes"], span["sha256"])
            if not span["bytes"] or key in seen:
                continue
            seen.add(key)
            destination.write(bytes(span["offset"] - destination.tell()))
            digest, count = hashlib.sha256(), 0
            with blob.open("rb") as stream:
                while chunk := stream.read(1 << 20):
                    destination.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
            if count != span["bytes"] or digest.hexdigest() != span["sha256"]:
                raise ValueError("program weight bytes changed during pool assembly")
    if target.stat().st_size != plan["bytes"]:
        raise ValueError("program weight pool extent changed")
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"program": None, "path": str(target), "bytes": plan["bytes"], "sha256": digest,
              "storage_source": None, "program_weight_pool": plan,
              "program_storage_sources": {name: str(Path(path).resolve()) for name, path in sources.items()}}
    (out / "program_weight_pool.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
