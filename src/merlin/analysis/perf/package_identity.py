"""What package bytes a whole-model result is a function of: the package and its PROGRAM.

* :func:`package_digest` -- the digest every owner names a candidate package by
  (:func:`merlin.common.tree_hash.hash_tree`), never a second one;
* :func:`program_digest` -- the bytes that can change the PROGRAM: documentation and developer tools
  are excluded, so two packages differing only there are the same measurement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path


class PackageIdentityError(RuntimeError):
    """The package has no hashable content."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Paths that never reach a program: documentation, notes, and the package's own developer tools.
NON_PROGRAM_PARTS = ("docs", "devtools", "__pycache__")
NON_PROGRAM_SUFFIXES = (".md", ".pyc", ".rst")  # never .txt: CMakeLists.txt builds a program


def package_digest(package: Path) -> str:
    """The digest the rest of the loop names candidates by (``hash_tree``), not a second one."""
    from merlin.common.tree_hash import hash_tree

    record = hash_tree(Path(package))
    digest = str(record.get("sha256") or "")
    if len(digest) != 64:
        raise PackageIdentityError(f"{package} has no hashable content")
    return digest


def program_digest(package: Path) -> str:
    """The digest of the bytes that can change the PROGRAM: the manifest, the tool, and the files under
    the manifest's declared ``components``, minus documentation and developer tools.

    Two requests with the same program digest are the same measurement.  A package that declares no
    components is hashed whole (less the non-program paths), never guessed narrower.
    """
    import yaml

    package = Path(package)
    try:
        manifest = yaml.safe_load((package / "manifest.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        manifest = {}
    declared: list[str] = []
    components = manifest.get("components") if isinstance(manifest, Mapping) else None
    for paths in components.values() if isinstance(components, Mapping) else ():
        declared.extend(str(p) for p in paths or () if isinstance(p, str))
    entrypoints = manifest.get("entrypoints") if isinstance(manifest, Mapping) else None
    tool = entrypoints.get("tool") if isinstance(entrypoints, Mapping) else None
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        if any(part in NON_PROGRAM_PARTS for part in relative.split("/")) or relative.endswith(NON_PROGRAM_SUFFIXES):
            continue
        if declared and relative not in ("manifest.yaml", str(tool or "")):
            if not any(relative == d.rstrip("/") or relative.startswith(d.rstrip("/") + "/") for d in declared):
                continue
        digest.update(relative.encode("utf-8") + b"\0" + _sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


__all__ = ["NON_PROGRAM_PARTS", "NON_PROGRAM_SUFFIXES", "PackageIdentityError", "package_digest", "program_digest"]
