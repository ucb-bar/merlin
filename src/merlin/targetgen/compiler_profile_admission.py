"""Original frozen-profile admission of exact compiler API imports.

Filesystem isolation remains mandatory. This scope authorizes direct imports of
the admitted source leaves; it does not expose automatic checker dependencies or
replace the separate closed compiler-library contract. Saved profile dictionaries
and candidate callbacks cannot create this host scope.
"""

from __future__ import annotations

import contextlib
import contextvars
import weakref
from pathlib import Path

import yaml

from merlin.common.access import module_name_for
from merlin.verify.lean_contract import document_digest

from .tool_profile import verify_tool_profile

_ACTIVE = contextvars.ContextVar("original_compiler_api_admission", default=None)
_ISSUED = weakref.WeakSet()
_OPTIONAL = frozenset({"isa_dialect_generation", "lean_proof_authoring"})


class _Admission:
    def __init__(self, *, workspace, bundle_manifest, repo):
        from merlin.common import bundle_snapshot as bwrap

        self.workspace, self.bundle_manifest, self.repo = map(Path, (workspace, bundle_manifest, repo))
        if (
            self.bundle_manifest.is_symlink()
            or self.bundle_manifest.resolve() != self.bundle_manifest
            or not self.bundle_manifest.is_file()
            or self.bundle_manifest.is_relative_to(self.workspace)
        ):
            raise ValueError("compiler imports require the original archived ordinary manifest")
        self.bundle_bytes = self.bundle_manifest.read_bytes()
        self.bundle = yaml.safe_load(self.bundle_bytes)
        self.profile = verify_tool_profile(self.bundle)
        if self.profile is None:
            raise ValueError("compiler imports require the original versioned source profile")
        self.bundle_identity = document_digest(self.bundle)
        self.snapshot_sha256 = bwrap.verify_bundle_snapshot(self.workspace, self.bundle, repo=self.repo)[
            "content_sha256"
        ]
        self.snapshot_marker_sha256 = bwrap.snapshot_marker_digest(self.workspace)
        self.modules = frozenset((*self.profile["common_public_modules"], *_optional_modules(self)))

    def verify(self):
        from merlin.common import bundle_snapshot as bwrap

        if self not in _ISSUED:
            raise ValueError("compiler imports require the original host-issued admission")
        if (
            self.bundle_manifest.read_bytes() != self.bundle_bytes
            or document_digest(self.bundle) != self.bundle_identity
            or verify_tool_profile(self.bundle) != self.profile
        ):
            raise ValueError("compiler import admission changed its original profile")
        marker = bwrap.verify_bundle_snapshot(self.workspace, self.bundle, repo=self.repo)
        if (
            marker["content_sha256"] != self.snapshot_sha256
            or bwrap.snapshot_marker_digest(self.workspace) != self.snapshot_marker_sha256
        ):
            raise ValueError("compiler import admission changed its original source snapshot")
        if self.modules != frozenset((*self.profile["common_public_modules"], *_optional_modules(self))):
            raise ValueError("compiler import admission changed its exact leaf membership")

    def permits(self, module: str) -> bool:
        return self in _ISSUED and module in self.modules


def _optional_modules(owner):
    from merlin.common import bundle_snapshot as bwrap

    from . import tool_registry

    for name in set(owner.profile["treatment_tools"]) & _OPTIONAL:
        for logical in tool_registry.spec(name).bundle_paths:
            module = module_name_for(logical)
            if module is None or not logical.endswith(".py") or logical.endswith("/__init__.py"):
                continue
            tail = logical.removeprefix("merlin/python/")
            matching = [
                row["path"]
                for row in owner.bundle["allowed"]
                if row["path"] == logical or row["path"].endswith("/" + tail)
            ]
            if len(matching) != 1:
                raise ValueError("optional compiler API has no exact original source grant")
            original = Path(matching[0])
            if not original.is_absolute():
                original = owner.repo / original
            [source] = bwrap.snapshot_input_paths(owner.workspace, owner.bundle, [original], repo=owner.repo)
            if source.is_symlink() or not source.is_file():
                raise ValueError("optional compiler API is not an original frozen source leaf")
            yield module


@contextlib.contextmanager
def original_compiler_imports(*, workspace: Path, bundle_manifest: Path, repo: Path):
    """Enter from the original common service, with its frozen grants unchanged."""
    admission = _Admission(workspace=workspace, bundle_manifest=bundle_manifest, repo=repo)
    _ISSUED.add(admission)
    token = _ACTIVE.set(admission)
    try:
        admission.verify()
        yield admission
        admission.verify()
    finally:
        _ACTIVE.reset(token)
        _ISSUED.discard(admission)


def _issued_compiler_imports():
    """Inspect issuer identity only; callers still own fresh input verification."""
    admission = _ACTIVE.get()
    if admission is not None:
        if type(admission) is not _Admission:
            raise ValueError("compiler import scope has no original issuer")
        if admission not in _ISSUED:
            raise ValueError("compiler imports require the original host-issued admission")
    return admission


def active_compiler_imports():
    admission = _issued_compiler_imports()
    if admission is not None:
        admission.verify()
    return admission
