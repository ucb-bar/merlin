#!/usr/bin/env python3
"""Materialize the exact atlas-kernels kernel information treatment.

The configured checkout is used only as a git object database. Files are exported from the declared
commit, so a dirty checkout or a moving branch cannot change an experiment bundle. This is the Atlas
counterpart of examples/radiance/phase1/kernel_library/materialize.py, with path exclusions for local
source-access rules. Exporting evidence does not enable selection or admit executable kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import yaml

from merlin.common.paths import artifacts_dir

HERE = Path(__file__).resolve()
SELECTION = HERE.parent / "selection.yaml"


def _git(repo: Path, *args: str, stdout=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=stdout,
        stderr=subprocess.PIPE,
        text=stdout is None,
    )


def _restricted(path: str | Path) -> bool:
    return any(word in str(path).lower() for word in ("hammer", "vlsi"))


def _payload_digest(root: Path) -> tuple[str, int, int]:
    if _restricted(root):
        raise ValueError("restricted artifact path")
    digest = hashlib.sha256()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"artifact must be a regular directory: {root}")
    members = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        # Filter before descending or inspecting members, including symlink destinations.
        blocked = [name for name in dirs + files if _restricted(name)]
        dirs[:] = [name for name in dirs if not _restricted(name)]
        if blocked:
            raise ValueError("artifact contains a restricted member")
        members.extend(Path(directory) / name for name in dirs + files)
    members.sort()
    for path in members:
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"unsupported artifact member: {path}")
    files = [p for p in members if p.is_file()]
    total = 0
    for path in files:
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(data).digest())
        total += len(data)
    return digest.hexdigest(), len(files), total


def _safe_extract(data: bytes, dest: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if _restricted(name) or name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member: {member.name!r}")
        archive.extractall(dest, filter="data")


def materialize(checkout: Path, output: Path) -> dict:
    if any(_restricted(path) for path in (checkout, output, SELECTION)):
        raise ValueError("restricted materialization path")
    selection = yaml.safe_load(SELECTION.read_text(encoding="utf-8")) or {}
    commit = str(selection["source_commit"])
    paths = [
        *selection.get("shared_paths", ()),
        *selection.get("qualified_families", ()),
        *(row["path"] for row in selection.get("experimental_families", ())),
    ]
    if not paths or any(_restricted(path) for path in paths):
        raise ValueError("selection needs unrestricted export paths")
    _git(checkout, "cat-file", "-e", f"{commit}^{{commit}}")
    archive = subprocess.run(
        ["git", "-C", str(checkout), "archive", "--format=tar", commit, "--", *paths,
         ":(exclude,icase)*hammer*", ":(exclude,icase)*vlsi*"],
        check=True,
        capture_output=True,
    ).stdout

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as td:
        stage = Path(td) / output.name
        stage.mkdir()
        _safe_extract(archive, stage)
        shutil.copy2(SELECTION, stage / "selection.yaml")
        payload_sha, n_files, n_bytes = _payload_digest(stage)
        manifest = {
            "schema": "atlas_kernel_library_materialization_v1",
            "source_repo": selection.get("source_pr"),
            "source_pin": selection["source_pin"],
            "source_commit": commit,
            "selection_sha256": hashlib.sha256(SELECTION.read_bytes()).hexdigest(),
            "payload_sha256": payload_sha,
            "n_files": n_files,
            "n_bytes": n_bytes,
            "qualified_families": len(selection.get("qualified_families", ())),
            "experimental_families": len(selection.get("experimental_families", ())),
        }
        (stage / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        (stage / "README.md").write_text(
            "# atlas-kernels kernel-library treatment\n\n"
            f"Exact export of `{commit}`. Qualified and experimental families are separated in "
            "`selection.yaml`. This tree is read-only authoring evidence: derive generalized compiler "
            "rules; do not copy, link, or call it from the submitted package.\n",
            encoding="utf-8",
        )

        if output.exists() or output.is_symlink():
            # Compare the complete generated tree, including its manifest and README.
            # A copied manifest alone cannot attest to an unchanged export.
            if _payload_digest(output) == _payload_digest(stage):
                return manifest
            raise FileExistsError(f"{output} exists with different content; refuse to overwrite a pinned artifact")
        os.replace(stage, output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkout",
        default=os.environ.get("MERLIN_ATLAS_KERNELS", ""),
        help="atlas-kernels checkout used as a git object database",
    )
    parser.add_argument("--output", default=str(artifacts_dir() / "targets/atlas/kernel_library_v1"))
    args = parser.parse_args(argv)
    if not args.checkout:
        parser.error("--checkout or MERLIN_ATLAS_KERNELS is required")
    result = materialize(Path(args.checkout).resolve(), Path(args.output).absolute())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
