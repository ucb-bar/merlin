"""Explicit installed-test inputs; identity observations, not toolchain authority.

The package roster pins selected source bytes. It does not prove import closure,
source-to-bytecode equivalence, dependency provenance, or runtime correctness.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def python_entry(path):
    """Preserve a selected interpreter prefix while binding its symlink chain."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("interpreter entry must be absolute")
    links, pending, current = [], list(path.parts[1:]), Path(path.anchor)
    while pending:
        current /= pending.pop(0)
        if current.is_symlink():
            if len(links) >= 40:
                raise ValueError("interpreter symlink chain is unavailable")
            target = os.readlink(current)
            links.append({"path": str(current), "target": target})
            replacement = Path(target)
            replacement = replacement if replacement.is_absolute() else current.parent / replacement
            pending = list(replacement.parts[1:]) + pending
            current = Path(replacement.anchor)
    actual = path.resolve(strict=True)
    if not actual.is_file() or not os.access(actual, os.X_OK):
        raise ValueError("interpreter entry must resolve to an executable")
    config = path.parent.parent / "pyvenv.cfg"
    if config.is_symlink() or (config.exists() and not config.is_file()):
        raise ValueError("interpreter prefix metadata must be a regular file")
    return {
        "entry_path": str(path),
        "resolved_path": str(actual),
        "sha256": _digest(actual),
        "symlinks": links,
        "prefix_config": {"path": str(config), "sha256": _digest(config) if config.exists() else None},
    }


def _git(root, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"},
        stderr=subprocess.PIPE,
    )


def source_record(path, commit, package, *, check_committed=False):
    """Reopen one explicitly admitted package, never discover SDK directories."""
    root = Path(path)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ValueError("native source must be an absolute canonical checkout")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("native source needs a full lowercase commit")
    if _git(root, "rev-parse", "--show-toplevel").decode().strip() != str(root):
        raise ValueError("native source must select the exact checkout root")
    if _git(root, "rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError("native source commit changed")
    names = tuple(
        value.decode()
        for value in _git(root, "ls-tree", "-rz", "--name-only", commit, "--", package, "pyproject.toml").split(b"\0")
        if value
    )
    if package + "/__init__.py" not in names or "pyproject.toml" not in names:
        raise ValueError("native source lacks its complete selected package")
    members = {}
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe tracked native source member")
        member = root / relative
        if any(root.joinpath(*relative.parts[:i]).is_symlink() for i in range(1, len(relative.parts) + 1)):
            raise ValueError("native source members must not be symlinks")
        if not member.is_file():
            raise ValueError("native source member is missing")
        data = member.read_bytes()
        if check_committed and data != _git(root, "show", commit + ":" + name):
            raise ValueError("native source bytes differ from selected commit")
        members[name] = hashlib.sha256(data).hexdigest()
    # Include ignored files too: an untracked importable module is not admitted.
    for member in (root / package).rglob("*"):
        name = member.relative_to(root).as_posix()
        if member.is_symlink() or (not member.is_dir() and name not in members):
            raise ValueError("native source contains untracked package files or symlinks")
    return {"path": str(root), "commit": commit, "package": package, "source_files": members}


def verify_sources(selected):
    for source in selected.values():
        expected = source["identity"]
        current = source_record(expected["path"], expected["commit"], expected["package"])
        if current != expected:
            raise ValueError("selected native source changed")


def test_inputs_record(path, names):
    """Pin one caller-selected closed test mapping; no source/runtime authority."""
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("native test inputs need a canonical regular file")
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("native test input mapping exceeds its bounded envelope")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("native test input mapping repeats a key")
            result[key] = value
        return result

    mapping = json.loads(raw, object_pairs_hook=unique)
    if (
        type(mapping) is not dict
        or set(mapping) != set(names)
        or any(type(value) is not str or not value or "\0" in value for value in mapping.values())
    ):
        raise ValueError("native test inputs differ from the complete suite-declared string mapping")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "environment": mapping}
