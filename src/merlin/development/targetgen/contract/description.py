"""Render capsule descriptions and verify their generated-copy correspondence."""

from __future__ import annotations

import os
from pathlib import Path

from merlin.common.digest import sha256_file


def render_description(entry: dict, cap: dict) -> bytes:
    """Render the existing capsule summary from its entry and declaration."""
    name = cap.get("name") or entry.get("name", "")
    prose = (entry.get("comment") or "").strip()
    if not prose:
        op = (cap.get("operation") or {}).get("op") or entry.get("op") or "unknown"
        ops = ", ".join(
            f"{i.get('name')}{list(i.get('shape') or [])}:{i.get('dtype')}"
            for i in (cap.get("inputs") or [])
            if i.get("name")
        )
        src = cap.get("source_reference") or entry.get("source_reference") or ""
        prose = f"{name}: {op}" + (f" over {ops}" if ops else "")
        prose += f", authored from {src}." if src else "."
    line = " ".join(
        f"{k}={v}"
        for k, v in (
            ("kind", cap.get("kind") or entry.get("kind")),
            ("label", cap.get("label") or entry.get("label")),
            ("op", (cap.get("operation") or {}).get("op") or entry.get("op")),
            ("modes", (cap.get("expected") or {}).get("modes", {})),
        )
        if v is not None
    )
    return f"# {name}\n\n{prose}\n\n{line}\n".encode("utf-8")


def write_description(entry: dict, cap: dict, directory: Path) -> bool:
    """Write a missing output description; preserve authored text already present."""
    path = directory / "README.md"
    if path.exists():
        return False
    path.write_bytes(render_description(entry, cap))
    return True


def verify_description_projection(source: Path, projected: Path) -> None:
    """Require an exact capsule copy plus its one declaration-derived description."""
    import yaml

    original = source / "README.md"
    description = projected / "README.md"
    if original.exists() or original.is_symlink() or description.is_symlink() or not description.is_file():
        raise ValueError("description projection requires one newly generated ordinary README")
    capsule = yaml.safe_load((source / "capsule.yaml").read_bytes())
    if description.read_bytes() != render_description({}, capsule):
        raise ValueError("description projection differs from its original capsule declaration")
    expected = {p.relative_to(source): p for p in source.rglob("*")}
    actual = {p.relative_to(projected): p for p in projected.rglob("*") if p != description}
    if expected.keys() != actual.keys():
        raise ValueError("description projection changed capsule membership")
    for relative, before in expected.items():
        after = actual[relative]
        if before.is_symlink() != after.is_symlink():
            raise ValueError("description projection changed an original link")
        if before.is_symlink() and os.readlink(before) != os.readlink(after):
            raise ValueError("description projection changed an original link destination")
        if before.is_dir() != after.is_dir() or before.is_file() != after.is_file():
            raise ValueError("description projection changed an original member type")
        if before.is_symlink() and before.is_dir():
            raise ValueError("description projection contains a directory link")
        if before.is_file():
            if sha256_file(before) != sha256_file(after):
                raise ValueError("description projection changed original capsule bytes")
        elif not before.is_dir():
            raise ValueError("description projection contains a nonregular member")
