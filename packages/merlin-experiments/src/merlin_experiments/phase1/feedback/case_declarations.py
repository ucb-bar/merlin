"""Find a public test case's own declaration (``capsule.yaml``) by name, for advisory feedback only.

The shape panel (:mod:`.shape_panel`), the per-class comparison line of the performance brief and the
capacity reading (:mod:`.capacity_use`) all classify a case by what it DECLARES -- operation, operand
shapes and dtypes -- never by its name. The declarations live in three places, read in this order:

* the corpus roots this process grades, registered by the self-check when it selects capsules
  (:func:`register`; the core roster) and by the panel (the core and scalability rosters);
* the frozen scalability snapshot this process was handed (:data:`.public_scalability.ENV`);
* the public declarations staged in the author's workspace (``<ws>/<group>/<case>/capsule.yaml``).

Nothing here grades, verifies or selects anything; a case whose declaration is not found is simply
not classified. Bounded reads, no ``re``, no target name.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DECLARATION = "capsule.yaml"
#: Largest declaration read (a declaration with inline literals can be large; this is advisory).
MAX_BYTES = 4 << 20
#: Workspace children never searched for staged declarations (author trees, outputs, channels).
_SKIP = frozenset({"submission", "variants", "selfcheck_out", "qa", ".qa_channel", "build", "__pycache__"})

_ROOTS: list[Path] = []
_TARGET: list[str] = []
_CACHE: dict[Path, dict | None] = {}


def register(root=None, *, target: str | None = None) -> None:
    """Remember a corpus root (cases as ``<root>/<name>/capsule.yaml``) and the target it is graded for."""
    if root is not None:
        path = Path(root)
        if path not in _ROOTS:
            _ROOTS.append(path)
    if isinstance(target, str) and target:
        _TARGET[:] = [target]


def target() -> str | None:
    """The target registered by this process, if any."""
    return _TARGET[0] if _TARGET else None


def scalability_root() -> Path | None:
    """The frozen scalability cases directory this process was handed, or None."""
    from .public_scalability import ENV

    raw = os.environ.get(ENV)
    if not raw:
        return None
    try:
        view = json.loads(raw)
    except ValueError:
        return None
    root = view.get("root") if isinstance(view, dict) else None
    return Path(root) / "cases" if isinstance(root, str) and root else None


def _read(path: Path) -> dict | None:
    if path in _CACHE:
        return _CACHE[path]
    document = None
    try:
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BYTES:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            document = loaded if isinstance(loaded, dict) else None
    except (OSError, ValueError, ImportError):
        document = None
    except Exception:  # noqa: BLE001 -- a malformed declaration is unclassified, never fatal
        document = None
    _CACHE[path] = document
    return document


def _workspace_candidates(workspace: Path, name: str):
    try:
        children = sorted(p for p in workspace.iterdir() if p.is_dir() and p.name not in _SKIP)
    except OSError:
        return
    for child in children:
        if not child.name.startswith("."):
            yield child / name / DECLARATION


def load(name: str, workspace=None) -> dict | None:
    """The declaration of public case ``name``, or None when no known root holds it."""
    if not isinstance(name, str) or not name or "/" in name or name.startswith("."):
        return None
    candidates = [root / name / DECLARATION for root in _ROOTS]
    scalability = scalability_root()
    if scalability is not None:
        candidates.append(scalability / name / DECLARATION)
    for path in candidates:
        document = _read(path)
        if document is not None:
            return document
    if workspace is not None:
        for path in _workspace_candidates(Path(workspace), name):
            document = _read(path)
            if document is not None:
                return document
    return None


def roster(root) -> list[tuple[str, dict]]:
    """``[(name, declaration)]`` of every case directly under ``root``, sorted by name."""
    out = []
    if root is None:
        return out
    try:
        children = sorted(Path(root).iterdir())
    except OSError:
        return out
    for child in children:
        if child.is_dir() and not child.name.startswith(("_", ".")):
            document = _read(child / DECLARATION)
            if document is not None:
                out.append((child.name, document))
    return out


def nested_roster(root) -> list[tuple[str, dict]]:
    """``[(name, declaration)]`` of every case one or two levels under ``root`` (grouped corpora)."""
    seen, out = set(), []
    for name, document in roster(root):
        seen.add(name)
        out.append((name, document))
    try:
        groups = sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith(".")) if root else []
    except OSError:
        groups = []
    for group in groups:
        if (group / DECLARATION).is_file():
            continue
        for name, document in roster(group):
            if name not in seen:
                seen.add(name)
                out.append((name, document))
    return sorted(out, key=lambda item: item[0])


__all__ = ["DECLARATION", "load", "nested_roster", "register", "roster", "scalability_root", "target"]
