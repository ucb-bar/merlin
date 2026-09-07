#!/usr/bin/env python3
"""Materialize the exact Radiance PR#1 kernel information treatment.

The configured checkout is used only as a git object database. Files are exported from the declared
commit, so a dirty checkout or a moving branch cannot change an experiment bundle.
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


HERE = Path(__file__).resolve()
TARGET_DIR = HERE.parent.parent
REPO = TARGET_DIR.parents[4]
SELECTION = TARGET_DIR / "contracts" / "kernel_library_pr1_v1.yaml"


def _git(repo: Path, *args: str, stdout=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, stdout=stdout,
        stderr=subprocess.PIPE, text=stdout is None,
    )


def _payload_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
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
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member: {member.name!r}")
        archive.extractall(dest, filter="data")


def materialize(checkout: Path, output: Path) -> dict:
    selection = yaml.safe_load(SELECTION.read_text(encoding="utf-8")) or {}
    commit = str(selection["source_commit"])
    paths = [
        *selection.get("shared_paths", ()),
        *selection.get("qualified_families", ()),
        *(row["path"] for row in selection.get("experimental_families", ())),
    ]
    _git(checkout, "cat-file", "-e", f"{commit}^{{commit}}")
    archive = subprocess.run(
        ["git", "-C", str(checkout), "archive", "--format=tar", commit, "--", *paths],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as td:
        stage = Path(td) / output.name
        stage.mkdir()
        _safe_extract(archive, stage)
        shutil.copy2(SELECTION, stage / "selection.yaml")
        payload_sha, n_files, n_bytes = _payload_digest(stage)
        manifest = {
            "schema": "radiance_kernel_library_materialization_v1",
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
        (stage / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        (stage / "README.md").write_text(
            "# Radiance PR#1 kernel-library treatment\n\n"
            f"Exact export of `{commit}`. Qualified and experimental families are separated in "
            "`selection.yaml`. This tree is read-only authoring evidence: derive generalized compiler "
            "rules; do not copy, link, or call it from the submitted package.\n",
            encoding="utf-8",
        )

        if output.exists():
            old_manifest = output / "manifest.yaml"
            old = yaml.safe_load(old_manifest.read_text()) if old_manifest.is_file() else {}
            if old == manifest:
                return manifest
            raise FileExistsError(
                f"{output} exists with different content; refuse to overwrite a pinned artifact")
        os.replace(stage, output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkout", default=os.environ.get("MERLIN_RADIANCE_KERNELS", ""),
        help="radiance-kernels checkout used as a git object database")
    parser.add_argument("--output", default=str(REPO / "out/artifacts/targets/radiance/kernel_library_pr1_v1"))
    args = parser.parse_args(argv)
    if not args.checkout:
        parser.error("--checkout or MERLIN_RADIANCE_KERNELS is required")
    result = materialize(Path(args.checkout).resolve(), Path(args.output).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
