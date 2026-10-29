"""Typed future request recipe; no provider selection, construction or authority.

Larger closures require independent source/native review. Fv5/minimal7 is not
expanded, reused implicitly or given additional modules by this adapter.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

from .lean_graph_pipeline_gate_profiles import INTEGER, selected
from .lean_graph_pipeline_gate_profile_prepare import prepare
from .lean_graph_pipeline_gate_projection import canonical, project_lean_sources, sha


@dataclass(frozen=True)
class ProjectSource:
    module: str
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class FutureRequest:
    schema: str
    profile: str
    prepared_identity: str
    module: str
    query: str
    query_sha256: str
    exact_final_statement: str
    final_root: str
    support_roots: tuple[str, ...]
    boundary_roots: tuple[tuple[str, tuple[str, ...]], ...]
    project_sources: tuple[ProjectSource, ...]
    domain: str
    relation: str
    premises: tuple[str, ...]
    entry_contract: str
    audit_policy: tuple[str, ...]
    host_admission_requirements: tuple[str, ...]
    query_contract: str = "pure_declarations_separate_root_audit_v1"
    provider_selection: str = "UNSELECTED_HELD_REQUIRES_NEW_EXACT_CONTEXT_REVIEW"

    def describe(self):
        return {**asdict(self), "authority": False, "native_authorization": False,
                "query_status": "UNCOMPILED", "outcome": "UNOBSERVED", "qualification": "UNQUALIFIED"}


def request(profile_name, buffers, settings):
    profile = selected(profile_name)
    prepared = prepare(profile_name, buffers, settings)
    if prepared.domain != profile.domain or prepared.relation != profile.relation or prepared.roots[-1] != profile.final_root:
        raise ValueError("exact profile/domain/root contract differs")
    imports = tuple(word for line in prepared.query.splitlines() if line.startswith("import ") for word in line.split()[1:])
    closure = project_lean_sources(imports)
    source_map = tuple(ProjectSource("Merlin." + Path(r.path).stem, r.path, r.sha256, r.size) for r in closure)
    if tuple(r for r in prepared.sources if r.path.endswith(".lean")) != closure:
        raise ValueError("exact separate query import closure required; implicit unions refused")
    marker = "theorem " + profile.final_root
    if prepared.query.count(marker) != 1 or not all(roots for _, roots in prepared.boundary_roots):
        raise ValueError("one full-entry final statement and every actual boundary root required")
    return FutureRequest("merlin.graph_larger_profile_request.v1", profile.name, prepared.identity,
        "GraphLargerProfile_" + prepared.identity, prepared.query, sha(prepared.query),
        marker + prepared.query.split(marker, 1)[1], profile.final_root, prepared.roots[:-1],
        prepared.boundary_roots, source_map, profile.domain, profile.relation, profile.premises,
        "canonical word inputs on correctly sized entries; missing/wrong arity included" if profile == INTEGER
        else "all Int entry arguments; missing/wrong arity included",
        ("exact final and EVERY listed support/boundary root type and transitive std3 audit",
         "only propext/Classical.choice/Quot.sound; no reflection or new axioms",
         "exact selected project source/import/object closure and runtime pre/post evidence",
         "same typed live CheckedQuery identity revalidated under newly approved provider"),
        ("fresh recomputation from every original actual buffer and registered adjacent roster",
         "SAME live original-to-final completion AND all checked actual adjacent boundary roots/receipts",
         "exact source/query/domain/entry-contract/provider/budget binding at delivery",
         "plan_only=False enforced by reviewed service; no local/saved/copied JSON authority",
         "no promotion from old qualified receipts or current minimal7/two-kind completions"))


def matches(saved, profile_name, buffers, settings):
    return type(saved) is dict and canonical(saved) == canonical(request(profile_name, buffers, settings).describe())


def provider_arguments(*args, **kwargs):
    raise RuntimeError("larger source recipe is not an authorized provider context; no default SDK or module union")
