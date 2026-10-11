"""Closed operator configuration for the ordinary component search lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import StageGateError, mapping_file, sha256_file, write_json
from .search_coordinator import SearchSettings
from .search_inputs import SearchInputs

_FIELDS = {
    "schema",
    "manifest_sha256",
    "corpus_sha256",
    "baseline_sha256",
    "certificate_sha256",
    "resources",
    "job_resources",
    "settings",
    "source_bytes",
    "source_files",
    "retained_source_bytes",
    "max_pending",
    "max_jobs",
    "author_agents",
}


def prepare_search_configuration(
    policy,
    path,
    *,
    resources,
    job_resources,
    settings,
    source_bytes,
    source_files,
    retained_source_bytes,
    max_pending,
    max_jobs,
    author_agents,
    external_boundary=None,
):
    """Pin an operator's explicit budgets to its live admitted development owners.

    This emits selection data, not runtime, model or timing qualification. Run it
    on the allocated worker: CPU assignments are checked against actual affinity.
    """
    from .evaluation_pool import AssignedResources, JobResources

    inputs = SearchInputs(policy)
    identities = inputs.identities()
    if (
        type(resources) is not AssignedResources
        or type(job_resources) is not JobResources
        or type(settings) is not SearchSettings
    ):
        raise StageGateError("search preparation requires explicit typed resources and stopping settings")
    inputs.validate()
    resources.verify(external_boundary=external_boundary)
    job_resources.verify()
    settings.verify()
    path = Path(path)
    if path.exists() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise StageGateError("search preparation requires a fresh ordinary configuration destination")
    document = {
        "schema": "merlin.component_search_configuration.v1",
        **identities,
        "resources": {**asdict(resources), "cpu_ids": list(resources.cpu_ids)},
        "job_resources": asdict(job_resources),
        "settings": asdict(settings),
        "source_bytes": source_bytes,
        "source_files": source_files,
        "retained_source_bytes": retained_source_bytes,
        "max_pending": max_pending,
        "max_jobs": max_jobs,
        "author_agents": author_agents,
    }
    write_json(path, document)
    selection = read_search_configuration(path)
    inputs.validate()
    selection.verify()
    return selection


@dataclass(frozen=True)
class SearchConfiguration:
    path: Path
    sha256: str
    document: dict
    settings: SearchSettings
    author_agents: int

    def verify(self):
        if (
            self.path.is_symlink()
            or sha256_file(self.path) != self.sha256
            or mapping_file(self.path) != self.document
            or SearchSettings(**self.document["settings"]) != self.settings
            or self.document["author_agents"] != self.author_agents
        ):
            raise StageGateError("search launch configuration changed")

    def install(self, policy, *, edit_authority, root, external_boundary=None):
        from .evaluation_pool import AssignedResources, JobResources
        from .retained_revisions import RetainedRevisionStore
        from .search_coordinator import ComponentSearchCoordinator

        self.verify()
        inputs = SearchInputs(policy)
        expected = inputs.identities()
        if any(self.document[key] != value for key, value in expected.items()):
            raise StageGateError("search selects another frozen corpus, fresh baseline or timing certificate")
        resources = dict(self.document["resources"])
        resources["cpu_ids"] = tuple(resources["cpu_ids"])
        resources, grant = AssignedResources(**resources), JobResources(**self.document["job_resources"])
        resources.verify(external_boundary=external_boundary)
        grant.verify()
        revisions = RetainedRevisionStore(
            root / "revisions",
            edit_authority=edit_authority,
            max_bytes=self.document["source_bytes"],
            max_files=self.document["source_files"],
            max_total_bytes=self.document["retained_source_bytes"],
        )
        arms = inputs.analytical.binding.compiled_arms if inputs.analytical is not None else None
        # The fixed compile observer accepts work only under its originally
        # issued private output owner. Revision selection grants no output path.
        evaluation_root = arms.execution.output / "search-evaluations" if arms is not None else root / "evaluations"
        pool = policy.enable_evaluation_pool(
            revisions=revisions,
            resources=resources,
            job_resources=grant,
            root=evaluation_root,
            beam=self.settings.beam_width,
            max_pending=self.document["max_pending"],
            max_jobs=self.document["max_jobs"],
            **({"external_boundary": external_boundary} if external_boundary is not None else {}),
        )
        try:
            policy.search_coordinator = ComponentSearchCoordinator(pool, settings=self.settings)
        except BaseException:
            pool.close()
            raise
        return pool


def read_search_configuration(path):
    """Decode selections before runtime startup; no callbacks, commands or grant paths."""
    path = Path(path)
    digest = sha256_file(path)
    document = mapping_file(path)
    if set(document) != _FIELDS or document["schema"] != "merlin.component_search_configuration.v1":
        raise StageGateError("search configuration requires its complete closed v1 declaration")
    for field in ("manifest_sha256", "corpus_sha256", "baseline_sha256", "certificate_sha256"):
        from merlin.common.digest import is_sha256

        if not is_sha256(document[field]):
            raise StageGateError("search configuration has an invalid immutable identity")
    for field in ("source_bytes", "source_files", "retained_source_bytes", "max_pending", "max_jobs", "author_agents"):
        if type(document[field]) is not int or document[field] < 1:
            raise StageGateError("search source, job and author budgets must be explicit positive integers")
    if (
        document["author_agents"] > 3
        or not 1 <= document["max_pending"] <= 6
        or not document["max_pending"] <= document["max_jobs"] <= 1024
        or document["retained_source_bytes"] < document["source_bytes"]
    ):
        raise StageGateError("search revisions, queue and authors exceed bounded policy")
    if (
        not isinstance(document["settings"], dict)
        or not isinstance(document["resources"], dict)
        or not isinstance(document["job_resources"], dict)
    ):
        raise StageGateError("search settings and resource grants must be explicit mappings")
    try:
        settings = SearchSettings(**document["settings"])
        settings.verify()
        from .evaluation_pool import AssignedResources, JobResources

        resources = document["resources"]
        if not isinstance(resources.get("cpu_ids"), list) or not resources["cpu_ids"]:
            raise StageGateError("search requires an explicit CPU assignment list")
        AssignedResources(**{**resources, "cpu_ids": tuple(resources["cpu_ids"])})
        JobResources(**document["job_resources"]).verify()
    except (TypeError, ValueError) as error:
        raise StageGateError("search has an unknown setting or malformed resource grant") from error
    if settings.beam_width > document["max_pending"]:
        raise StageGateError("search beam exceeds its pending evaluation grant")
    selected = SearchConfiguration(path, digest, document, settings, document["author_agents"])
    selected.verify()
    return selected
