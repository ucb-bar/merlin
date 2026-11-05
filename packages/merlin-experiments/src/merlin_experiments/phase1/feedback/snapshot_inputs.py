"""Immutable public-input owner independent of legacy host proof-source pins."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from merlin.common import bundle_snapshot as B
from merlin.common.paths import module_source_path
from merlin.targetgen.tool_profile import verify_tool_profile
from merlin.verify.lean_contract import document_digest


class SnapshotInputs:
    """Trusted evaluator input selection; grants remain the archived bundle's."""
    def __init__(self, *, workspace, bundle_manifest, repo):
        self.workspace, self.bundle_manifest, self.repo = map(Path, (workspace, bundle_manifest, repo))
        if (self.bundle_manifest.resolve() != self.bundle_manifest or self.bundle_manifest.is_symlink()
                or self.bundle_manifest.is_relative_to(self.workspace)):
            raise ValueError("snapshot input authority must be outside the writable author workspace")
        self.bundle_bytes = self.bundle_manifest.read_bytes()
        self.bundle = yaml.safe_load(self.bundle_bytes)
        self.bundle_identity = document_digest(self.bundle)
        self.profile = verify_tool_profile(self.bundle)
        if self.profile is None:
            raise ValueError("snapshot execution requires an explicit public input profile")
        marker = B.verify_bundle_snapshot(self.workspace, self.bundle, repo=self.repo)
        self.snapshot_digest = marker["content_sha256"]
        self.snapshot_marker_digest = B.snapshot_marker_digest(self.workspace)
        # Record the ACTUAL external checker revision. These are deliberately
        # not presented as matching a historical Lean/tool-profile certificate.
        names = {row["module"].removesuffix(".__init__")
                 for row in self.profile["automatic_checker_sources"] if row.get("module")}
        names.update((__name__, "merlin.targetgen.package_runtime",
                      "merlin_experiments.phase1.feedback.snapshot_execution"))
        self.framework_sources = tuple(self._pin(module_source_path(name)) for name in sorted(names))

    @staticmethod
    def _pin(path):
        path = Path(path).resolve()
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def _verify(self):
        if (self.bundle_manifest.read_bytes() != self.bundle_bytes
                or document_digest(self.bundle) != self.bundle_identity
                or verify_tool_profile(self.bundle) != self.profile):
            raise ValueError("snapshot public input declaration changed")
        marker = B.verify_bundle_snapshot(self.workspace, self.bundle, repo=self.repo)
        if (marker["content_sha256"] != self.snapshot_digest
                or B.snapshot_marker_digest(self.workspace) != self.snapshot_marker_digest
                or tuple(self._pin(row["path"]) for row in self.framework_sources) != self.framework_sources):
            raise ValueError("snapshot inputs or external evaluator revision changed")

    def record(self):
        return {"schema": "merlin.snapshot_external_inputs.v1", "workspace": str(self.workspace),
                "repo": str(self.repo), "bundle_manifest": self._pin(self.bundle_manifest),
                "snapshot_sha256": self.snapshot_digest,
                "snapshot_marker_sha256": self.snapshot_marker_digest,
                "profile_sha256": self.profile["sha256"],
                "framework_sources": list(self.framework_sources)}
