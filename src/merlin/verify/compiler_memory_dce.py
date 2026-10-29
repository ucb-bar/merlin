"""Fresh kernel checks for ordinary memory-context dead index constants.

The actual source and target are independently decoded. Generic semantic rules
and closed pair facts preserve complete successful returned-value/world outcomes;
parser/native fidelity and categorized error behavior remain separate boundaries.
Saved reports and copied completions confer no delivery authority.
"""

from __future__ import annotations

import json
import os
import threading
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path

from xdsl.utils.exceptions import ParseError, VerifyException

from merlin.common.paths import module_source_path
from merlin.verify import configuration_elision_runner as runner
from merlin.verify.configuration_elision_bridge import canonical, digest, file_ref
from merlin.verify.configuration_elision_gate import NotEstablished, Unavailable, _exact
from merlin.verify.gate_errors import GateRejected, GateUnavailable, GateUnsupported

from .compiler_memory_stage_intake import prepare as prepare_projection

CHECK = "compiler_memory_dce_refinement"
SELECTOR = "prune_dead_pure_tensor_ops"
DOMAIN = "ordinary-memory-context-dead-index-constants-index64-v1"
RELATION = "equal_complete_graph_return_and_world_outcomes"
MODULE = "ActualMemoryStage"
PROOF_MODULE = "Merlin.GraphLoopsShapeDeadConstants"
ROOTS = tuple(
    MODULE + "." + name
    for name in ("before_valid", "after_valid", "frame_exact", "actual_pruned", "complete", "contextual")
) + (
    "Merlin.GraphLoopsShape.DeadConstants.prune_complete",
    "Merlin.GraphLoopsShape.DeadConstants.checked_complete",
)
_LIVE = weakref.WeakKeyDictionary()


def proof_sources():
    base = module_source_path("merlin.verify").parent / "lean"
    sources, seen = [], set()

    def visit(name):
        if name in seen:
            return
        if not name.startswith("Merlin.") or any(not part.isidentifier() for part in name.split(".")):
            raise Unavailable("invalid canonical memory DCE proof import")
        seen.add(name)
        path = base / (name.replace(".", "/") + ".lean")
        if not path.is_file() or path.is_symlink():
            raise Unavailable("canonical memory DCE proof source unavailable: " + name)
        for line in path.read_text().splitlines():
            words = line.split()
            if words and words[0] == "import":
                for child in words[1:]:
                    if child.startswith("Merlin."):
                        visit(child)
        sources.append({"module": name, **file_ref(path)})

    visit(PROOF_MODULE)
    return sources


def _host_sources():
    modules = (
        "merlin.verify.compiler_memory_dce",
        "merlin.verify.compiler_memory_stage_intake",
        "merlin.verify.compiler_memory_origin",
        "merlin.verify.pass_validation",
        "merlin.verify.gate_worker",
        "merlin.verify.gate_checks",
        "merlin.verify.compiler_transforms",
        "merlin.verify.speculative_gate",
        "merlin.verify.configuration_elision_runner",
        "merlin.verify.configuration_elision_gate",
        "merlin.verify.configuration_elision_bridge",
        "merlin.verify.lean_graph_pipeline_gate_projection",
        "merlin.verify.buffer_reuse_bridge",
        "merlin.verify.lean_graph_input_admission",
        "merlin.verify.lean_graph_loops_memory_scf_after_projection",
        "merlin.verify.lean_graph_loops_shape_after_projection",
        "merlin.verify.lean_graph_loops_shape_projection",
        "merlin.verify.lean_graph_loops_shape_capture",
        "merlin.verify.lean_graph_loops_shape_model",
        "merlin.verify.lean_graph_loops_shape_types",
        "merlin.verify.lean_graph_loops_segment_projection",
        "merlin.verify.lean_graph_loops_unroll",
        "merlin.verify.lean_graph_loops_shape_epoch_model",
        "merlin.llvmlower.lower",
        "merlin.llvmlower.leading_cse",
        "merlin.llvmlower.pipeline",
        "merlin.llvmlower.passes_xdsl",
        "merlin.common.owned_process",
        "merlin.common.owned_children",
        "merlin.common.paths",
        "merlin.verify",
        "merlin",
        "merlin.common",
    )
    paths = {module_source_path(name) for name in modules}
    verify = module_source_path("merlin.verify").parent
    paths.update(verify.glob("lean_graph_loops*.py"))
    paths.add(verify / "lean/Merlin/GraphStateAudit.lean")
    paths.add(module_source_path("merlin.xdsl_dialects._common"))
    paths.update(
        verify.parent / "sched" / name
        for name in ("__init__.py", "ir/__init__.py", "ir/expr.py", "ir/kernel.py", "ir/parse.py")
    )
    return [file_ref(path) for path in sorted(paths)]


@dataclass(frozen=True)
class Candidate:
    source: str
    final: str
    projection_json: str
    resolution_json: str
    request_json: str
    query: str
    roots: tuple[str, ...]
    sources_json: str
    proof_sources_json: str
    proposed_sites: int
    domain: str = DOMAIN
    relation: str = RELATION

    @property
    def identity(self):
        return digest(canonical(asdict(self)))

    def describe(self):
        projection = json.loads(self.projection_json)
        return {
            "schema": "merlin.compiler-memory-dce.candidate.v1",
            "identity": self.identity,
            "domain": self.domain,
            "relation": self.relation,
            "source_sha256": digest(self.source),
            "final_sha256": digest(self.final),
            "query_sha256": digest(self.query),
            "roots": list(self.roots),
            "entries": [row["name"] for row in projection["before"]["functions"]],
            "proposed_rewrite_sites": self.proposed_sites,
            "ordinary_same_mode": True,
            "normalization_mode_used": False,
            "allocator_policy_required": False,
            "source_success_required": False,
            "whole_run_equality_required": False,
            "full_error_or_world_equivalence": False,
            "native_fidelity_proved": False,
            "authority": False,
            "kernel_outcome": "UNOBSERVED",
            "qualification": "UNQUALIFIED",
        }


def prepare(source, final, *, index_bits=64):
    try:
        projected = prepare_projection(source, final, index_bits=index_bits)
    except RecursionError as error:
        raise Unavailable("memory DCE projection exhausted its host stack/work budget") from error
    request = {"schema": "merlin.compiler-memory-dce.request.v1", "index_bits": index_bits}
    model = {
        "ordinary_same_mode": True,
        "deleted_nodes": "intrinsic-unused-index-constants-only",
        "retained_actions": "one-shared-arbitrary-partial-interpretation",
        "region_bodies": "partial-handler-congruence-under-closed-name-deletion-plan",
        "current_producer_plan": "entry-block-intrinsic-dead-index-constants",
        "parser_projection_and_primitive_fidelity": "external-trusted-boundary",
        "source_support": "closed-ordinary-graph-validity-not-frozen-tensor-model",
    }
    proofs = proof_sources()
    return Candidate(
        source=source,
        final=final,
        projection_json=canonical(projected["projection"]),
        resolution_json=canonical(model),
        request_json=canonical(request),
        query=projected["query"],
        roots=ROOTS,
        sources_json=canonical([*_host_sources(), *[file_ref(row["path"]) for row in proofs]]),
        proof_sources_json=canonical(proofs),
        proposed_sites=projected["proposed_rewrite_sites"],
    )


def ready(candidate):
    if type(candidate) is not Candidate or candidate.domain != DOMAIN or candidate.relation != RELATION:
        raise Unavailable("exact canonical memory DCE candidate required")
    for row in json.loads(candidate.sources_json):
        _exact(row)
    if candidate.roots != ROOTS or json.loads(candidate.proof_sources_json) != proof_sources():
        raise Unavailable("canonical memory DCE root/source roster changed")
    try:
        rebuilt = prepare(candidate.source, candidate.final, index_bits=64)
    except (ValueError, TypeError, KeyError, ParseError, VerifyException) as error:
        raise Unavailable("canonical memory DCE candidate no longer reconstructs") from error
    if candidate != rebuilt:
        raise Unavailable("actual memory DCE pair/query/frame/renaming changed")


@dataclass(frozen=True, eq=False)
class Completion:
    identity: str
    report: str

    def __reduce_ex__(self, protocol):
        raise TypeError("live memory DCE completion cannot be serialized")

    def __copy__(self):
        return Completion(self.identity, self.report)

    def __deepcopy__(self, memo):
        return self.__copy__()


def check_candidate(candidate, *, lean, destination, timeout=300):
    ready(candidate)
    report, _ = runner.prove(
        candidate,
        lean=lean,
        destination=destination,
        timeout=timeout,
        validate=ready,
        query_module=MODULE,
        source_plan=proof_sources,
        relation=RELATION,
    )
    report.update(schema="merlin.compiler-memory-dce.checked.v1", domain=DOMAIN)
    path = Path(destination).resolve() / "REPORT.json"
    path.write_text(canonical(report))
    completion = Completion(candidate.identity, str(path))
    _LIVE[completion] = {
        "pid": os.getpid(),
        "thread": threading.get_ident(),
        "candidate": candidate,
        "report": file_ref(path),
    }
    return completion


def validate_completion(completion, candidate):
    if type(completion) is not Completion or completion not in _LIVE:
        raise Unavailable("fresh original live memory DCE completion required")
    issued = _LIVE[completion]
    if (
        issued["pid"] != os.getpid()
        or issued["thread"] != threading.get_ident()
        or issued["candidate"] != candidate
        or completion.identity != candidate.identity
        or completion.report != issued["report"]["path"]
    ):
        raise Unavailable("memory DCE completion differs from original owner or candidate")
    ready(candidate)
    _exact(issued["report"])
    report = json.loads(Path(completion.report).read_text())
    if (
        report.get("schema") != "merlin.compiler-memory-dce.checked.v1"
        or report.get("source_identity") != candidate.identity
        or report.get("candidate") != candidate.describe()
        or report.get("domain") != DOMAIN
        or report.get("relation") != RELATION
        or report.get("sources") != json.loads(candidate.sources_json)
        or report.get("source_request") != json.loads(candidate.request_json)
        or [row.get("sha256") for row in report.get("buffers", [])]
        != [digest(candidate.source), digest(candidate.final)]
        or set(report.get("statements", {})) != set(ROOTS)
    ):
        raise Unavailable("memory DCE report differs from the original checked pair")
    refs = []
    for key in ("buffers", "mapping_records", "imports", "library", "process_evidence"):
        refs.extend(report[key])
    refs.extend(report[key] for key in ("query", "object", "tool", "auditor"))
    for row in refs:
        _exact(row)
    library = Path(report["library_root"])
    if any(path.is_symlink() for path in library.rglob("*")) or {
        str(path.resolve()) for path in library.rglob("*") if path.is_file()
    } != {row["path"] for row in report["library"]}:
        raise Unavailable("private memory DCE object membership changed")
    return report


def deliver(completion, candidate):
    validate_completion(completion, candidate)
    return candidate.final


def replay(report_path, *, lean, destination, timeout=300):
    # A saved document only locates the two raw inputs; it restores no authority.
    saved = json.loads(Path(report_path).read_text())
    if saved.get("schema") != "merlin.compiler-memory-dce.checked.v1" or len(saved.get("buffers", [])) != 2:
        raise Unavailable("descriptive complete memory DCE pair required")
    if saved.get("source_request") != {"schema": "merlin.compiler-memory-dce.request.v1", "index_bits": 64}:
        raise Unavailable("replay model/index domain differs from the fixed canonical request")
    for row in saved["buffers"]:
        _exact(row)
    candidate = prepare(*(Path(row["path"]).read_text() for row in saved["buffers"]), index_bits=64)
    if [digest(candidate.source), digest(candidate.final)] != [row["sha256"] for row in saved["buffers"]]:
        raise Unavailable("actual replay buffers changed while they were independently decoded")
    completion = check_candidate(candidate, lean=lean, destination=destination, timeout=timeout)
    for row in saved["buffers"]:
        _exact(row)
    return completion, candidate


def check(job):
    params = job.get("params")
    if (
        type(params) is not dict
        or set(params) != {"schema", "sdk_bin", "index_bits"}
        or params.get("schema") != "merlin.compiler-memory-dce.job.v1"
        or type(params.get("index_bits")) is not int
        or params["index_bits"] != 64
    ):
        raise GateUnsupported("exact original memory DCE job settings required")
    try:
        before, after = [file_ref(job[key]) for key in ("before", "after")]
        candidate = prepare(Path(before["path"]).read_text(), Path(after["path"]).read_text())
        if digest(candidate.source) != before["sha256"] or digest(candidate.final) != after["sha256"]:
            raise Unavailable("actual memory DCE job buffers changed during independent decode")
        completion = check_candidate(
            candidate,
            lean=Path(params["sdk_bin"]) / "lean",
            destination=Path(job["directory"]) / "memory-dce-proof",
            timeout=300,
        )
        report = validate_completion(completion, candidate)
        for row in (before, after):
            _exact(row)
    except NotEstablished as error:
        raise GateRejected(str(error)) from error
    except (Unavailable, OSError) as error:
        raise GateUnavailable(str(error)) from error
    except (ValueError, TypeError, KeyError, ParseError, VerifyException) as error:
        raise GateUnsupported(str(error)) from error
    return {
        "schema": "merlin.compiler-memory-dce.gate.v1",
        "status": "checked",
        "domain": DOMAIN,
        "relation": RELATION,
        "theorems": list(ROOTS),
        "compiler_enforcement": True,
        "ordinary_same_mode": True,
        "normalization_mode_used": False,
        "before": before,
        "after": after,
        "receipt": completion.report,
        "source_identity": candidate.identity,
        "entry_points": candidate.describe()["entries"],
        "changed_sites": candidate.proposed_sites,
        "assurance": report["audit"]["assurance"],
        "axioms": report["audit"]["axioms_used"],
        "full_error_or_world_equivalence": False,
        "native_fidelity_proved": False,
        "allocator_policy_required": False,
    }
