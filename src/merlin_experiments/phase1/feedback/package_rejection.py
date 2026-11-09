"""Explain a rejected compiler package to its author, and keep the explanation where the author reads.

A package that fails its integrity scan runs no test case. Background grades publish to
``qa/verdict.json`` only while their snapshot is still the current source, and an author who keeps
editing rarely leaves a rejected snapshot current, so the rejection was archived but never shown.
Measured: an editable module created under a namespace the full-compiler selection does not
contain (``merlin/frontend/`` beside the selected ``merlin/frontends``) gated all 252 test cases of
two grades while the author's verdict showed only the model gate. Self-checks did report the
failure, but under a note about cmake builds, which does not describe a namespace violation.

``namespace_hint`` adds what the author needs for a namespace refusal: the rejected path and the
selection's mutable namespaces. ``publish_notice`` records a namespace refusal of a no-longer-current
snapshot as a ``package_failure_notice`` beside the last results while the rejected path is still in
the current submission; the next published grade replaces the notice.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

OUTSIDE = "outside public namespace"
NOTICE = "package_failure_notice"


def _mutable_namespaces() -> list[str] | None:
    try:
        from ..full_compiler_workspace import selected
    except ImportError:
        return None
    try:
        selection = selected()
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return sorted(f"merlin.{name}" for name in selection.mutable_namespaces) if selection is not None else None


def rejected_path(failure) -> str | None:
    """The editable-library-relative path a namespace refusal names, or None."""
    detail = str((failure or {}).get("detail") or "") if isinstance(failure, dict) else ""
    _, marker, tail = detail.partition(OUTSIDE)
    if not marker:
        return None
    relative = tail.partition(":")[2].strip().split()[0] if ":" in tail else ""
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


def namespace_hint(failure) -> dict:
    """Extra fields for a namespace refusal (empty for any other failure)."""
    relative = rejected_path(failure)
    if relative is None:
        return {}
    hint = {
        "rejected_path": relative,
        "rule": ("Every module under submission/compiler_library must live in one of the selected mutable "
                 "namespaces; a module elsewhere rejects the whole package before any test case runs. Move "
                 "or rename it into an allowed namespace (check the spelling of the package directory)."),
    }
    namespaces = _mutable_namespaces()
    if namespaces:
        hint["allowed_namespaces"] = namespaces
    return hint


def enrich(verdict: dict) -> dict:
    """Add the namespace hint to a verdict's ``package_failure`` in place; returns the verdict."""
    failure = verdict.get("package_failure")
    if isinstance(failure, dict):
        failure.update(namespace_hint(failure))
    return verdict


def publish_notice(workspace, verdict: dict) -> dict:
    """Show a non-current snapshot's package rejection beside the author's last results."""
    failure = verdict.get("package_failure")
    if not isinstance(failure, dict):
        return verdict
    try:
        from ..full_compiler_workspace import DIRECTORY
        from .visible_model_gate import publish

        relative = rejected_path(failure)
        # Only a refusal that can be re-checked against the current source is shown: a stale notice of
        # a failure already fixed would mislead more than it helps.
        if relative is None or not (Path(workspace) / "submission" / DIRECTORY / relative).exists():
            return verdict
        publish(Path(workspace) / "qa" / "verdict.json", {NOTICE: {
            **enrich({"package_failure": dict(failure)})["package_failure"],
            "source_sha256": verdict.get("graded_source_sha256"),
            "graded_at": verdict.get("graded_at"),
            "note": "a newer snapshot is being graded; this earlier snapshot was rejected before any test "
                    "case ran, and the rejected source is still present",
        }})
    except (ImportError, OSError, ValueError):
        pass  # the archive keeps the rejection; a notice is advisory
    return verdict
