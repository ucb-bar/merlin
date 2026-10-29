"""Source-bound closed obligations for actual logical host dispatch fusion."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import xdsl

from merlin.common.paths import module_source_path, repo_root

from .configuration_elision_bridge import canonical, digest, file_ref
from .host_graph_fusion_projection import project_pair
from .lean_graph_regions import _nats, _node, _strings

DOMAIN = "static-pure-integer-pointwise-single-host-region-v1"
RELATION = "equal_complete_public_results_under_typed_total_pure_actions"
MODULE = "ActualHostFusion"
ROOT = MODULE + ".complete"
PREMISES = (
    "trusted xDSL parsing, typed SSA projection and direct private-call expansion",
    "each exact scalar-body key denotes the same total deterministic pure typed tensor action",
    "all public input tensor cells are initialized; generic destinations are unread",
    "logical tensor allocation is unobservable and does not fail",
    "no proof of primitive MLIR fidelity, downstream lowering, native allocation or hardware behavior",
)


def proof_sources():
    base = Path(module_source_path("merlin.verify")).parent / "lean"
    return [
        {"module": "Merlin." + name, **file_ref(base / f"Merlin/{name}.lean")}
        for name in ("StateEffects", "GraphRegions", "GraphRegionsFusion", "HostGraphFusion")
    ]


def source_refs():
    modules = (
        "merlin.verify.host_graph_fusion_intake",
        "merlin.verify.host_graph_fusion_projection",
        "merlin.verify.compiler_host_graph_fusion",
        "merlin.verify.lean_graph_regions_projection",
        "merlin.verify.lean_graph_regions",
        "merlin.verify.configuration_elision_runner",
        "merlin.verify.configuration_elision_gate",
        "merlin.verify.configuration_elision_bridge",
        "merlin.verify.pass_validation",
        "merlin.verify.gate_checks",
        "merlin.verify.gate_worker",
        "merlin.verify.speculative_gate",
        "merlin.verify.gate_errors",
        "merlin.common.owned_process",
        "merlin.common.owned_children",
        "merlin.common.owned_group",
        "merlin.common.paths",
        "merlin.llvmlower.graph_optimizations",
        "merlin.llvmlower.lower",
        "merlin.xdsl_dialects.lowering.outlined_plan_emission",
        "merlin.xdsl_dialects.lowering.global_plan_emission",
        "merlin.xdsl_dialects.lowering.outline",
        "merlin.xdsl_dialects.lowering.dispatch_program",
    )
    paths = [module_source_path(name) for name in modules]
    paths += sorted(Path(xdsl.__file__).parent.rglob("*.py"))
    paths += [sys.executable, Path(module_source_path("merlin.verify")).parent / "lean/Merlin/GraphStateAudit.lean"]
    paths += [row["path"] for row in proof_sources()]
    paths += [repo_root() / "merlin/lean/LEDGER.md"]
    return sorted({row["path"]: row for row in map(file_ref, paths)}.values(), key=lambda row: row["path"])


def _graph(graph):
    return (
        "⟨["
        + ",".join(_node(n) for n in graph["nodes"])
        + "],"
        + ",".join((_nats(graph["inputs"]), _nats(graph["observations"]), _strings(graph["context"])))
        + "⟩"
    )


def emit(projection):
    group = projection["fusion"]
    body = (
        "⟨["
        + ",".join(_node(n) for n in group["body"])
        + "],"
        + ",".join((_nats(group["inputs"]), _nats(group["outputs"])))
        + "⟩"
    )
    start, count = group["start"], group["count"]
    query = f"""import Merlin.HostGraphFusion
set_option maxRecDepth 4096
set_option maxHeartbeats 4000000
namespace {MODULE}
open Merlin Merlin.GraphRegions
def before : Graph := {_graph(projection["before"]["graph"])}
def after : Graph := {_graph(projection["after"]["graph"])}
def body : FusedBody := {body}
theorem accepted : HostGraphFusion.check before after {start} {count} body = true := by decide
theorem complete (meaning : ∀ node, TypedAction node Value typing)
    (initial : StateEffects.State Value) (domain : WellTyped typing initial) :
    before.Valid ∧ HostGraphFusion.execute meaning after {start} {count} body initial =
      before.observations.map (executeNodes meaning before.nodes initial) :=
  HostGraphFusion.check_sound accepted meaning initial domain
end {MODULE}
"""
    return query, (
        "Merlin.HostGraphFusion.check_sound",
        "Merlin.GraphRegions.checkFusion_preserves_observers",
        MODULE + ".accepted",
        ROOT,
    )


@dataclass(frozen=True)
class Candidate:
    source: str
    final: str
    request_json: str
    resolution_json: str
    projection_json: str
    query: str
    roots: tuple[str, ...]
    sources_json: str
    proof_sources_json: str
    domain: str = DOMAIN
    relation: str = RELATION

    @property
    def identity(self):
        return digest(canonical(asdict(self)))

    def describe(self):
        projection = json.loads(self.projection_json)
        return {
            "schema": "merlin.host-graph-fusion.candidate.v1",
            "identity": self.identity,
            "domain": self.domain,
            "relation": self.relation,
            "module": MODULE,
            "source_sha256": digest(self.source),
            "final_sha256": digest(self.final),
            "query_sha256": digest(self.query),
            "roots": list(self.roots),
            "final_root": ROOT,
            "entry": projection["before"]["entry"],
            "fused_stages": projection["fusion"]["count"],
            "premises": list(PREMISES),
            "authority": False,
        }


def prepare(source, final):
    projection = project_pair(source, final)
    query, roots = emit(projection)
    return Candidate(
        source,
        final,
        canonical({"schema": "merlin.host-graph-fusion.request.v1"}),
        canonical({"projection_trusted": True, "premises": list(PREMISES)}),
        canonical(projection),
        query,
        roots,
        canonical(source_refs()),
        canonical(proof_sources()),
    )


def ready(candidate):
    if type(candidate) is not Candidate or candidate.domain != DOMAIN or candidate.relation != RELATION:
        raise ValueError("exact current host graph fusion candidate required")
    if prepare(candidate.source, candidate.final) != candidate:
        raise ValueError("host fusion source, projection, query or proof closure changed")
