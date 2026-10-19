"""Explicit original-host archive selection for the normal queued stage.

This object carries live reviewed owners, never candidate JSON callbacks or
qualification flags. Ordinary Phase 1, corpus and final gates run unchanged.
"""

from dataclasses import dataclass, fields
from pathlib import Path

from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapability

from .archive_spike_runtime import ArchiveSpikeRuntime
from .contracts import StageGateError
from .ordinary_archive_screen import issue_archive_screen


@dataclass(frozen=True)
class ArchiveScreenSelection:
    review_id: str
    archive: Path
    archive_sha256: str
    binding_path: Path
    observations_path: Path
    screen_path: Path
    runtime: ArchiveSpikeRuntime
    correspondence: ReviewedPassCapability
    expected_rtl_facts_sha256: str
    expected_observed_contract_sha256: str

    def preflight(self):
        """Review actual public archive support without starting ordinary search."""
        from .archive_screen_review import review_archive

        self.correspondence.revalidate()
        return review_archive(
            runtime=self.runtime,
            archive=self.archive,
            archive_sha256=self.archive_sha256,
            binding_path=self.binding_path,
            observations_path=self.observations_path,
            screen_path=self.screen_path,
        )

    def install(self, policy):
        """Rebuild original eligibility/held gates before admitting any job."""
        if (
            type(self) is not ArchiveScreenSelection
            or type(self.runtime) is not ArchiveSpikeRuntime
            or type(self.correspondence) is not ReviewedPassCapability
        ):
            raise StageGateError("normal archive screening requires explicit original live host owners")
        arguments = {field.name: getattr(self, field.name) for field in fields(self)}
        owner = issue_archive_screen(policy, **arguments)
        policy.install_screen(owner)
        return owner


def install(selection, policy):
    """Called only after the original policy/build selection and before its pool."""
    if selection is not None:
        if type(selection) is not ArchiveScreenSelection:
            raise StageGateError("archive-screen JSON cannot replace original host selection")
        return selection.install(policy)
    return None
