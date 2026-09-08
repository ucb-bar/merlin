#!/usr/bin/env python3
"""Verify the immutable compiler and artifact-local target snapshots."""

from __future__ import annotations

import hashlib
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
EXPECTED = {
    "compiler": ("7037327f17379f352a172422d2d8246651b9739a31bd56c1b7095af4462e392c", 42),
    "runtime_target": ("aa7d86fb190c44019d25c8fe5d7fe25290ce047d64c2f84acc44c748520fb22b", 36),
}
SKIP = {"build", "__pycache__", ".git"}


def hash_tree(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or SKIP.intersection(path.parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        files += 1
    return digest.hexdigest(), files


def main() -> int:
    failed = False
    for relative, expected in EXPECTED.items():
        actual = hash_tree(BUNDLE / relative)
        status = "PASS" if actual == expected else "FAIL"
        print(f"{status} {relative}: sha256={actual[0]} files={actual[1]}")
        failed |= actual != expected
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
