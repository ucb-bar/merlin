"""Canonical fresh-worker gate for the bounded generic-value UNROLL model.

The kernel proves complete operational entry outcomes under the explicit T7
occurrence-aware projection. Ordinary payloads have arbitrary Value meanings;
this adds no tensor-as-integer, floating arithmetic or LLVM correspondence claim.
"""

from __future__ import annotations

import json
import os
import threading
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common.paths import module_source_path

from . import configuration_elision_runner as runner
from .compiler_loop_generic_intake import prepare_query
from .compiler_loop_generic_profile import OutsideProfile
from .configuration_elision_bridge import canonical, digest, file_ref
from .configuration_elision_gate import NotEstablished, Unavailable, _exact, require_query_success
from .gate_errors import GateRejected, GateUnavailable, GateUnsupported

DOMAIN = "compiler-static-ordered-scf-generic-value-index64-v1"
RELATION = "all_defined_entries_equal_returns_errors_terminal_world_under_occurrence_aware_projection"
CHECK = "compiler_loop_generic_refinement"
MODULE = "CompilerGenericLoopQuery"
ROOT = "Merlin.GraphStatePipelineInstance.complete"
ADJACENT = "Merlin.GraphStatePipelineInstance.complete2"
UNROLL = "ordered_loop_unrolling"
TILE = "ordered_loop_segmentation"
_LIVE = weakref.WeakKeyDictionary()


def adjacent(kind):
    if kind not in (UNROLL, TILE):
        raise ValueError("one bounded generic adjacent pass required")
    return "Merlin.GraphStatePipelineInstance.complete" + ("2" if kind == UNROLL else "3")


def proof_sources(query):
    """Exact current transitive project sources in import order; no objects."""
    parent = Path(__file__).parent / "lean"
    result, seen = [], set()

    def visit(name):
        if name in seen:
            return
        if not name.startswith("Merlin.") or any(not part.isidentifier() for part in name.split(".")):
            raise Unavailable("generic loop query imports must be closed project modules")
        seen.add(name)
        path = parent / (name.replace(".", "/") + ".lean")
        for line in path.read_text().splitlines():
            words = line.split()
            if words and words[0] == "import":
                for child in words[1:]:
                    if child.startswith("Merlin."):
                        visit(child)
        result.append({"module": name, **file_ref(path)})

    imports = [name for line in query.splitlines() if line.startswith("import ") for name in line.split()[1:]]
    if not imports:
        raise Unavailable("generic loop proof import roster is empty")
    for name in imports:
        visit(name)
    return result


def _sources(leaf, proofs):
    names = (
        "merlin.verify.compiler_loop_generic",
        "merlin.verify.compiler_loop_source_profile",
        "merlin.verify.compiler_profile",
        "merlin.verify.configuration_elision_runner",
        "merlin.verify.configuration_elision_gate",
        "merlin.verify.configuration_elision_bridge",
        "merlin.verify.configuration_elision_intake",
        "merlin.verify.configuration_elision_emit",
        "merlin.verify.lean_graph_pipeline_gate_profile_prepare",
        "merlin.verify.lean_graph_state_scalar_capture",
        "merlin.verify.compiler_transforms",
        "merlin.verify.gate_worker",
        "merlin.verify.gate_checks",
        "merlin.verify.pass_validation",
        "merlin.verify.speculative_gate",
        "merlin.verify.gate_errors",
        "merlin.common.paths",
        "merlin.common.owned_process",
        "merlin.common.owned_children",
        "merlin.common.owned_group",
        "merlin.transforms.normalization",
    )
    rows = [*leaf, *(file_ref(module_source_path(name)) for name in names)]
    rows += [{key: value[key] for key in ("path", "sha256", "bytes")} for value in proofs]
    rows.append(file_ref(Path(__file__).parent / "lean/Merlin/GraphStateAudit.lean"))
    return sorted({row["path"]: row for row in rows}.values(), key=lambda row: row["path"])


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
        capture = json.loads(self.projection_json)
        return {
            "schema": "merlin.compiler-generic-loop.candidate.v1",
            "identity": self.identity,
            "domain": self.domain,
            "relation": self.relation,
            "source_sha256": digest(self.source),
            "final_sha256": digest(self.final),
            "query_sha256": digest(self.query),
            "roots": list(self.roots),
            "module": MODULE,
            "kind": json.loads(self.request_json)["kind"],
            "entries": [f["name"] for f in capture["capture"]["functions"][0] if not f["declaration"]],
            "premises": [
                "trusted complete typed/capture/callee/metadata projection and selected signed-index64 correspondence",
                "checked occurrence-aware control roles and fitting reachable selected/generated controls",
                "one arbitrary common partial opaque interpretation with successful result arity",
                "arbitrary data Value carrier; no reverse payload encode/decode or numerical rewrite premise",
            ],
            "authority": False,
            "native_fidelity_proved": False,
        }


def prepare(source, final, *, kind=UNROLL, factor=2, tile=3, index_bits=64):
    if kind not in (UNROLL, TILE):
        raise ValueError("registered generic loop profile supports one singleton UNROLL or TILE")
    leaf = prepare_query(source, final, kind=kind, factor=factor, tile=tile, index_bits=index_bits)
    roots = tuple(leaf["roots"])
    if len(set(roots)) != len(roots) or ROOT not in roots or adjacent(kind) not in roots:
        raise Unavailable("generic complete/adjacent root roster is incomplete")
    query = leaf["query"]
    proofs = proof_sources(query)
    request = {
        "schema": "merlin.compiler-generic-loop.source.v1",
        "kind": kind,
        "factor": factor,
        "index_bits": index_bits,
    }
    if kind == TILE:
        request.update(schema="merlin.compiler-generic-loop.source.v2", tile=tile)
    return Candidate(
        source,
        final,
        canonical(request),
        canonical({"profile": DOMAIN, "projection_correspondence": leaf["projection_correspondence"]}),
        canonical(leaf["capture"]),
        query,
        roots,
        canonical(_sources(leaf["sources"], proofs)),
        canonical(proofs),
    )


def ready(candidate):
    if type(candidate) is not Candidate or candidate.domain != DOMAIN or candidate.relation != RELATION:
        raise Unavailable("exact canonical generic loop candidate required")
    for row in json.loads(candidate.sources_json):
        _exact(row)
    try:
        regenerated = prepare(candidate.source, candidate.final, **_request(candidate.request_json))
    except (ValueError, OSError) as error:
        raise Unavailable("canonical generic loop inputs/preparation changed") from error
    if regenerated != candidate:
        raise Unavailable("canonical generic loop query/capture/root/source closure changed")


def _request(raw):
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("exact generic loop original-source request required")
    keys = {"schema", "kind", "factor", "index_bits"}
    if value.get("kind") == UNROLL:
        schema = "merlin.compiler-generic-loop.source.v1"
    elif value.get("kind") == TILE:
        keys.add("tile")
        schema = "merlin.compiler-generic-loop.source.v2"
    else:
        raise ValueError("one registered generic loop original-source kind required")
    if set(value) != keys or value["schema"] != schema:
        raise ValueError("exact versioned generic loop original-source request required")
    return {key: value[key] for key in keys - {"schema"}}


@dataclass(frozen=True, eq=False)
class Completion:
    identity: str
    report: str

    def __reduce_ex__(self, protocol):
        raise TypeError("live generic loop completion cannot be serialized")

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
        source_plan=lambda: proof_sources(candidate.query),
        relation=RELATION,
        # Symbolic generic obligations fail as unsolved goals, not closed
        # `decide` refutations; they are refusals, never counterexamples.
        require_success=lambda process: require_query_success(process, family="generic loop", unsolved_goals=True),
    )
    report.update(schema="merlin.compiler-generic-loop.checked.v1", domain=DOMAIN)
    path = Path(destination).resolve() / "REPORT.json"
    path.write_text(canonical(report))
    completion = Completion(candidate.identity, str(path))
    _LIVE[completion] = (os.getpid(), threading.get_ident(), candidate, file_ref(path), report, ())
    validate_completion(completion, candidate)
    return completion


def validate_completion(completion, candidate):
    if type(completion) is not Completion or completion not in _LIVE:
        raise Unavailable("saved/constructed/copied generic loop completion has no authority")
    pid, thread, original, report_ref, report, origins = _LIVE[completion]
    if (pid, thread) != (os.getpid(), threading.get_ident()) or original is not candidate:
        raise Unavailable("generic loop completion belongs to another live invocation")
    if completion.identity != candidate.identity or file_ref(completion.report) != report_ref:
        raise Unavailable("generic loop report/candidate identity changed")
    ready(candidate)
    if origins:
        _read_bound_buffers(origins)
    library = Path(report["library_root"])
    if any(path.is_symlink() for path in library.rglob("*")):
        raise Unavailable("generic private proof objects may not be symlinks")
    if {str(path.resolve()) for path in library.rglob("*") if path.is_file()} != {r["path"] for r in report["library"]}:
        raise Unavailable("generic private proof object membership changed")
    for row in [
        *report["buffers"],
        *report["mapping_records"],
        report["query"],
        report["object"],
        report["tool"],
        report["auditor"],
        *report["imports"],
        *report["library"],
        *report["process_evidence"],
    ]:
        _exact(row)
    return json.loads(canonical(report))


def deliver(completion, candidate):
    validate_completion(completion, candidate)
    return candidate.final


def _read_bound_buffers(rows):
    if type(rows) not in (list, tuple) or len(rows) != 2:
        raise Unavailable("exact two bound generic replay buffers required")
    result = []
    try:
        for row in rows:
            if type(row) is not dict or set(row) != {"path", "sha256", "bytes"}:
                raise Unavailable("exact generic replay file reference required")
            _exact(row)
            raw = Path(row["path"]).read_bytes()
            if digest(raw) != row["sha256"] or len(raw) != row["bytes"]:
                raise Unavailable("generic replay decoded bytes differ from bound original")
            result.append(raw.decode("utf-8"))
            _exact(row)
    except (OSError, UnicodeError) as error:
        raise Unavailable("generic replay original bytes unavailable") from error
    return tuple(result)


def replay(report_path, *, lean, destination, timeout=300):
    saved = json.loads(Path(report_path).read_bytes().decode("utf-8"))
    if saved.get("schema") != "merlin.compiler-generic-loop.checked.v1":
        raise Unavailable("generic replay requires exact raw source/target locator rows")
    origins = saved.get("buffers")
    raw = _read_bound_buffers(origins)
    # No saved query, roots, projection, status, verdict or proof objects are
    # input authority: only bound raw bytes/settings locate a canonical check.
    candidate = prepare(*raw, **_request(canonical(saved["source_request"])))
    if (candidate.source, candidate.final) != raw:
        raise Unavailable("generic replay preparation changed original bytes")
    completion = check_candidate(candidate, lean=lean, destination=destination, timeout=timeout)
    if _read_bound_buffers(origins) != raw:
        raise Unavailable("generic replay originals changed during proof")
    record = _LIVE.get(completion)
    if record is None:
        raise Unavailable("generic replay did not obtain an owned live completion")
    _LIVE[completion] = (*record[:5], tuple(origins))
    validate_completion(completion, candidate)
    return completion, candidate


def job_params(*, sdk_bin, kind=UNROLL, factor=2, tile=3, index_bits=64):
    return {
        "sdk_bin": str(sdk_bin),
        "index_bits": index_bits,
        "factor": factor,
        "tile": tile,
        "kinds": [kind],
        "selected": [kind],
        "intermediates": [],
        "allocator": None,
    }


def check(job):
    from .compiler_transforms import _input_snapshot, _recheck_inputs

    params = job.get("params")
    if (
        type(params) is not dict
        or set(params) != {"sdk_bin", "index_bits", "factor", "tile", "kinds", "selected", "intermediates", "allocator"}
        or params["kinds"] not in ([UNROLL], [TILE])
        or params["selected"] != params["kinds"]
        or params["intermediates"] != []
        or params["allocator"] is not None
        or type(params["factor"]) is not int
        or (params["kinds"] == [UNROLL] and params["factor"] != 2)
        or type(params["index_bits"]) is not int
        or params["index_bits"] != 64
        or type(params["tile"]) is not int
        or params["tile"] < 2
        or (params["kinds"] == [TILE] and params["tile"] > 65536)
    ):
        raise GateUnsupported("exact bounded generic singleton UNROLL/TILE compiler selection/tool required")
    try:
        inputs = _input_snapshot(job)
        originals = list(inputs.refs)
        candidate = prepare(
            *inputs.buffers,
            kind=params["kinds"][0],
            factor=params["factor"],
            tile=params["tile"],
            index_bits=params["index_bits"],
        )
        completion = check_candidate(
            candidate, lean=Path(params["sdk_bin"]) / "lean", destination=Path(job["directory"]) / "generic-loop-proof"
        )
        report = validate_completion(completion, candidate)
        delivered = deliver(completion, candidate)
        if delivered != candidate.final:
            raise Unavailable("generic live delivery differs from exact actual target")
        delivered_path = Path(job["directory"]) / "generic-delivered.mlir"
        delivered_path.write_bytes(delivered.encode("utf-8"))
        delivered_ref = file_ref(delivered_path)
        if delivered_ref["sha256"] != digest(candidate.final):
            raise Unavailable("generic delivered bytes differ from exact actual target")
        _recheck_inputs(inputs, job)
    except OutsideProfile as error:
        raise GateUnsupported(str(error)) from error
    except NotEstablished as error:
        raise GateRejected(str(error)) from error
    except (Unavailable, OSError) as error:
        raise GateUnavailable(str(error)) from error
    except (ValueError, TypeError, KeyError) as error:
        # Original-source eligibility is handled independently before producer.
        # A malformed target after that admission is a blocking gate outcome.
        raise GateRejected("generic actual boundary preparation refused: " + str(error)) from error
    return {
        "schema": "merlin.compiler-generic-loop.gate.v1",
        "status": "checked",
        "domain": DOMAIN,
        "relation": RELATION,
        "kind": params["kinds"][0],
        "receipt": completion.report,
        "theorems": list(candidate.roots),
        "axioms": report["audit"]["axioms_used"],
        "entries": candidate.describe()["entries"],
        "premises": candidate.describe()["premises"],
        "compiler_enforcement": True,
        "native_fidelity_proved": False,
        "before": originals[0],
        "after": originals[1],
        "delivered": delivered_ref,
    }
