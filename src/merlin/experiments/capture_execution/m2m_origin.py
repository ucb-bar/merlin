"""Own-repository and staged-copy provenance for selected Model2MLIR bytes.

The Git revision is an origin hint. The normalized package tree selected for
execution remains the byte authority, including after Phase 0 copies it out
of its Git worktree.
"""

from __future__ import annotations

import hashlib
import stat
import subprocess
from pathlib import Path
from typing import Any

from merlin.common import strict_json

ORIGIN_SCHEMA = "merlin.selected_m2m_git_origin.v1"
FROZEN_SCHEMA = "merlin.phase0.frozen_m2m_origin.v1"
_SELECTION_SCHEMA = "merlin.phase0.selected_m2m_runtime.v1"


class M2MOriginError(ValueError):
    """Selected source has no trustworthy own-repository or frozen-copy origin."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value: Any, size: int) -> bool:
    return type(value) is str and len(value) == size and all(letter in "0123456789abcdef" for letter in value)


def _is_tree(value: Any) -> bool:
    return (
        type(value) is dict
        and set(value) == {"members", "bytes", "sha256"}
        and type(value["members"]) is int
        and value["members"] > 0
        and type(value["bytes"]) is int
        and value["bytes"] >= 0
        and _is_sha(value["sha256"], 64)
    )


def git_origin(root: Path, *, clean: bool) -> dict[str, str]:
    """Inspect only the selected repository, never an ancestor or Git env redirect."""
    root = Path(root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise M2MOriginError("selected M2M root must be an ordinary absolute directory")
    env = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CEILING_DIRECTORIES": str(root.parent),
    }

    def run(*arguments: str, strip: bool = True) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *arguments],
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise M2MOriginError("selected M2M root must own its Git repository") from exc
        return result.stdout.strip() if strip else result.stdout

    top = run("rev-parse", "--show-toplevel")
    if Path(top) != root:
        raise M2MOriginError("selected M2M root must own its Git repository")
    commit = run("rev-parse", "--verify", "HEAD^{commit}")
    if not _is_sha(commit, 40):
        raise M2MOriginError("selected M2M origin lacks a pinned commit")
    status_args = (
        ("status", "--porcelain", "--untracked-files=all", "--", "m2m")
        if clean
        else ("status", "--porcelain", "--", "m2m")
    )
    status = run(*status_args, strip=False)
    if clean and status:
        raise M2MOriginError("selected M2M package must have a clean pinned commit")
    return {"commit": commit, "worktree_status": status}


def selected_source_origin(root: Path, package: dict, readonly_package: dict) -> dict:
    """Record the true pre-copy Git origin and normalized selected package."""
    if not _is_tree(package) or not _is_tree(readonly_package):
        raise M2MOriginError("selected M2M package inventory is malformed")
    origin = git_origin(root, clean=True)
    return {
        "schema": ORIGIN_SCHEMA,
        "root": str(root),
        "commit": origin["commit"],
        "package": package,
        "readonly_package": readonly_package,
    }


def frozen_selector(path: Path) -> dict[str, str]:
    """Pin a Phase 0-owned staged-selection receipt for later sealed capture."""
    path = Path(path)
    if (
        not path.is_absolute()
        or path.name != "m2m-runtime.json"
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 2_000_000
    ):
        raise M2MOriginError("frozen M2M origin receipt is absent or indirect")
    return {"schema": FROZEN_SCHEMA, "path": str(path), "sha256": _sha256(path.read_bytes())}


def verify_frozen_selector(selector: Any, root: Path, package: dict) -> str:
    """Rejoin an owner-written selection receipt to the exact frozen bytes."""
    root = Path(root)
    if (
        type(selector) is not dict
        or set(selector) != {"schema", "path", "sha256"}
        or selector["schema"] != FROZEN_SCHEMA
        or not _is_sha(selector["sha256"], 64)
        or type(selector["path"]) is not str
        or not root.is_absolute()
        or not _is_tree(package)
        or (root / ".git").exists()
        or (root / ".git").is_symlink()
    ):
        raise M2MOriginError("invalid frozen M2M origin selector")
    receipt = Path(selector["path"])
    if (
        receipt != root.parent / "m2m-runtime.json"
        or receipt.is_symlink()
        or not receipt.is_file()
        or stat.S_IMODE(receipt.stat().st_mode) & 0o222
    ):
        raise M2MOriginError("frozen M2M origin receipt has no selected owner")
    if receipt.stat().st_size > 2_000_000:
        raise M2MOriginError("frozen M2M origin receipt is oversized")
    return verify_frozen_receipt(selector, receipt.read_bytes(), root, package)


def verify_frozen_receipt(selector: Any, raw: bytes, root: Path, package: dict) -> str:
    """Verify copied receipt bytes independently of the live Phase 0 run path."""
    root = Path(root)
    if (
        type(selector) is not dict
        or set(selector) != {"schema", "path", "sha256"}
        or selector["schema"] != FROZEN_SCHEMA
        or not _is_sha(selector["sha256"], 64)
        or type(selector["path"]) is not str
        or Path(selector["path"]) != root.parent / "m2m-runtime.json"
        or type(raw) is not bytes
        or len(raw) > 2_000_000
        or not _is_tree(package)
    ):
        raise M2MOriginError("invalid frozen M2M origin receipt binding")
    if _sha256(raw) != selector["sha256"]:
        raise M2MOriginError("frozen M2M origin receipt changed")
    try:
        document = strict_json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise M2MOriginError("frozen M2M origin receipt is malformed") from exc
    if type(document) is not dict or document.get("schema") != _SELECTION_SCHEMA:
        raise M2MOriginError("frozen M2M origin receipt is not a staged selection")
    origin = document.get("source_origin")
    if (
        type(origin) is not dict
        or set(origin) != {"schema", "root", "commit", "package", "readonly_package"}
        or origin["schema"] != ORIGIN_SCHEMA
        or type(origin["root"]) is not str
        or not Path(origin["root"]).is_absolute()
        or origin["root"] != document.get("root")
        or not _is_sha(origin["commit"], 40)
        or not _is_tree(origin["package"])
        or not _is_tree(origin["readonly_package"])
        or origin["package"] != document.get("package")
        or document.get("frozen_root") != str(root)
        or document.get("frozen_package") != package
        or origin["readonly_package"] != package
    ):
        raise M2MOriginError("frozen M2M origin does not bind the selected copy")
    return origin["commit"]
