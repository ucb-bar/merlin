"""Closed schedule source request; never extends the existing two-kind gate."""
from dataclasses import asdict, dataclass
from pathlib import Path

from . import lean_graph_pipeline_gate_schedule as schedule
from .lean_graph_pipeline_gate_projection import canonical, sha


@dataclass(frozen=True)
class FutureScheduleRequest:
    schema: str
    profile: str
    kind: str
    prepared_identity: str
    domain: str
    relation: str
    premises: tuple[str, ...]
    query: str
    query_sha256: str
    final_root: str
    support_roots: tuple[str, ...]
    exact_statement: str
    project_sources: tuple[tuple[str,str,str,int], ...]
    host_obligations: tuple[str, ...]
    provider: str = "UNSELECTED_HELD"

    def describe(self):
        return {**asdict(self), "query_contract": "pure_declarations_separate_root_audit_v1",
                "authority": False, "native_authorization": False,
                "query_status": "UNCOMPILED", "outcome": "UNOBSERVED", "qualification": "UNQUALIFIED"}


def request(profile, before, after, order):
    if type(profile) is not str or profile != schedule.PROFILE:
        raise ValueError("exact schedule profile/domain required; no fallback")
    p=schedule.prepare(before,after,order)
    root="merlin_schedule_transform"
    if p.roots[-1]!=root or p.query.count("theorem "+root)!=1:
        raise ValueError("exact full-entry scheduling statement/root required")
    sources=tuple(("Merlin."+Path(r.path).stem,r.path,r.sha256,r.size) for r in p.sources if r.path.endswith('.lean'))
    return FutureScheduleRequest("merlin.graph_schedule_request.v1",p.profile,p.kind,p.identity,p.domain,p.relation,
        p.premises,p.query,sha(p.query),root,p.roots[:-1],"theorem "+root+p.query.split("theorem "+root,1)[1],sources,
        ("fresh complete module/common registry/current order/dependency/opaque-anchor reconstruction",
         "exact final/support root types and transitive std3 audit; no local-lemma-only acceptance",
         "explicit NEW reviewed provider/source/import/object/runtime profile; no default SDK or two-kind gate extension",
         "same typed live full-entry completion revalidated for actual source/output/order at delivery",
         "plan_only=False in future reviewed service; no saved/copied/diagnostic/certificate authority",
         "unchanged-order example stays identity; scheduling is not panel/double-buffer execution proof"))


def matches(saved,profile,before,after,order):
    return type(saved) is dict and canonical(saved)==canonical(request(profile,before,after,order).describe())


def provider_arguments(*args, **kwargs):
    raise RuntimeError("schedule provider unselected/held; source recipe cannot construct a context")
