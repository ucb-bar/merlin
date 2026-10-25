"""Explicit installed core-import closure for the standalone Lean route.

The existing .ilean import metadata is an audited tooling input, not a theorem.
Every discovered source/object/metadata byte and bundled shared runtime is bound;
no packages are fetched and no compiled project objects are reused.
"""

from __future__ import annotations

import json
from pathlib import Path


def inventory(lean: Path, destination: Path, file_ref) -> dict:
    root = lean.parent.parent
    library = root / "lib/lean"
    seen, pending, files = set(), ["Init", "Init.Data.List.Lemmas"], {}
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        if not name.startswith("Init") or "/" in name:
            raise ValueError("unexpected standalone-core import")
        seen.add(name)
        relative = Path(*name.split("."))
        metadata = library / relative.with_suffix(".ilean")
        imported = json.loads(metadata.read_text())["directImports"]
        pending.extend(row[0] for row in imported)
        for suffix in (".ilean", ".olean", ".olean.private", ".olean.server"):
            path = library / relative.with_suffix(suffix)
            files[str(path.relative_to(root))] = file_ref(path)
        source = root / "src/lean" / relative.with_suffix(".lean")
        files[str(source.relative_to(root))] = file_ref(source)
    for directory in (root / "lib", library, root / "lib/glibc"):
        for path in sorted(directory.glob("*.so*")):
            if path.is_file():
                files[str(path.relative_to(root))] = file_ref(path)
    result = {"module_count": len(seen), "modules": sorted(seen), "files": files}
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return file_ref(destination)


def validate(reference, fresh_ref) -> dict:
    manifest = json.loads(fresh_ref(reference).read_text())
    if type(manifest.get("files")) is not dict or not manifest["files"]:
        raise ValueError("missing core/toolchain dependency inventory")
    for record in manifest["files"].values():
        fresh_ref(record)
    return manifest
