"""Prepare existing bounded buffer-reuse actual-final proof obligations.

No checked/live receipt or allocator conformance is issued. An untrusted plan
is only a certificate: generated Lean goals compare it with independently decoded
actual source and target terms, checked memory bindings and the full entry roster.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .buffer_reuse_bridge import parse, reconstruct_chain
from .buffer_reuse_emit import factor, shape_validity, theorem_roster
from .lean_graph_loops_memory_scf_after_entries import prepare_entries
from .lean_graph_pipeline_gate_projection import project_lean_sources

ALLOCATOR = "total_private_unobservable"
DOMAIN = "static-rectangular-private-buffer-possible-success-index64-v1"
ROOT = "GraphMemoryAfterEntries.every_definition"
RELATION = "equal_possible_successful_public_observations"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


@dataclass(frozen=True)
class Candidate:
    source: str
    final: str
    hint_json: str
    query: str
    roots: tuple[str, ...]
    entry_roster_json: str
    sources_json: str
    domain: str = DOMAIN
    allocator: str = ALLOCATOR
    relation: str = RELATION
    index_bits: int = 64
    raw_chain_json: str = ""
    query_parts_json: str = ""

    @property
    def identity(self):
        return digest(canonical(asdict(self)))

    def describe(self):
        return {
            "schema": "merlin.buffer-reuse.candidate.v1",
            "identity": self.identity,
            "domain": self.domain,
            "relation": self.relation,
            "allocator": self.allocator,
            "index_bits": self.index_bits,
            "source_sha256": digest(self.source),
            "final_sha256": digest(self.final),
            "hint_sha256": digest(self.hint_json),
            "query_sha256": digest(self.query),
            "final_root": ROOT,
            "support_roots": list(self.roots[:-1]),
            "query_status": "UNCOMPILED",
            "kernel_outcome": "UNOBSERVED",
            "qualification": "UNQUALIFIED",
            "authority": False,
            "full_error_or_world_equivalence": False,
            "native_allocator_fidelity": False,
            "pipeline_receipt_issued": False,
            "producer_attribution": "not_checked_by_intake",
            "raw_chain_sha256": [digest(x) for x in json.loads(self.raw_chain_json)] if self.raw_chain_json else [],
            "generated_modules": [
                {key: row[key] for key in ("module", "imports", "sha256")} for row in json.loads(self.query_parts_json)
            ]
            if self.query_parts_json
            else [],
        }


def prepare(source, final, hint, *, allocator, index_bits=64):
    if allocator != ALLOCATOR:
        raise ValueError("finite/observable/unspecified allocator excluded;8->16-byte counterexample retained")
    if type(index_bits) is not int or index_bits != 64:
        raise ValueError("bounded buffer intake index64 only")
    if any(type(raw) is not str or len(raw.encode()) > 256 * 1024 for raw in (source, final)):
        raise ValueError("bounded complete raw source/final required")
    if type(hint) is not dict or "adapter" not in hint or "pair" not in hint:
        raise ValueError("explicit untrusted adapter/frame certificate required")
    if len(canonical(hint).encode()) > 2 * 1024 * 1024:
        raise ValueError("bounded untrusted certificate required")
    result = prepare_entries(
        hint["adapter"], hint["pair"]["afterFrame"], source, final, parser=parse, index_bits=index_bits, max_nodes=5000
    )
    if any(row["declaration"] for row in result["entry_roster"]["source_headers"]):
        raise ValueError(
            "bounded all-entry buffer domain requires defined functions; external declarations are separate"
        )
    if result["final_root"] != ROOT or ROOT in result["support_roots"]:
        raise RuntimeError("wrong complete-entry obligation")
    # Keep every existing library/support root and print the type of every
    # generated theorem too; imported-module audit alone is not a root inventory.
    query = shape_validity(result["query"], count=len(result["entry_roster"]["source_headers"]), bits=index_bits)
    result = {**result, "query": query, "query_sha256": digest(query)}
    roots = tuple(dict.fromkeys([*result["support_roots"], *theorem_roster(result["query"]), ROOT]))
    roots = (*[root for root in roots if root != ROOT], ROOT)
    if len(set(roots)) != len(roots) or len(result["query"].encode()) > 2 * 1024 * 1024:
        raise ValueError("distinct complete bounded root/query required")
    parts = factor(result["query"], roots)
    imports = [word for line in result["query"].splitlines() if line.startswith("import ") for word in line.split()[1:]]
    sources = [asdict(row) for row in project_lean_sources(imports)]
    parent = Path(__file__).parent
    paths = {
        Path(__file__),
        parent / "buffer_reuse_bridge.py",
        parent / "buffer_reuse_emit.py",
        parent / "buffer_reuse_gate.py",
        parent / "lean/Merlin/GraphStateAudit.lean",
        parent.parent / "common/owned_process.py",
        parent.parent / "common/paths.py",
        parent.parent / "common/__init__.py",
        parent / "__init__.py",
        parent.parent / "__init__.py",
        parent / "lean_graph_input_admission.py",
        parent / "lean_graph_pipeline_gate_projection.py",
    }
    paths.update(parent.glob("lean_graph_loops*.py"))
    # Literal rendering imports the Kernel IR namespace through
    # lean_graph_loops_unroll -> lean_graph_loops. These five project sources
    # are required even when only the literal helpers are called. Bind them
    # before proof checking and again at delivery/replay, not just in packaging.
    paths.update(
        parent.parent / "sched" / name
        for name in ("__init__.py", "ir/__init__.py", "ir/expr.py", "ir/kernel.py", "ir/parse.py")
    )
    paths.update(parent / name for name in ("lean_graph_loops_unroll.py", "lean_graph_loops_segment_projection.py"))
    for path in sorted(paths):
        data = path.read_bytes()
        sources.append({"path": str(path), "sha256": digest(data), "bytes": len(data)})
    return Candidate(
        source,
        final,
        canonical(hint),
        result["query"],
        roots,
        canonical(result["entry_roster"]),
        canonical(sources),
        query_parts_json=canonical(parts),
    )


def revalidate(candidate):
    """Detect stale candidate bytes; this cannot turn preparation into authority."""
    if type(candidate) is not Candidate:
        raise ValueError("exact buffer Candidate required")
    for row in json.loads(candidate.sources_json):
        data = Path(row["path"]).read_bytes()
        if digest(data) != row["sha256"] or len(data) != row.get("bytes", row.get("size")):
            raise ValueError("buffer candidate proof/host source changed")
    if candidate.raw_chain_json:
        fresh = prepare_chain(
            tuple(json.loads(candidate.raw_chain_json)), allocator=candidate.allocator, index_bits=candidate.index_bits
        )
    else:
        fresh = prepare(
            candidate.source,
            candidate.final,
            json.loads(candidate.hint_json),
            allocator=candidate.allocator,
            index_bits=candidate.index_bits,
        )
    if fresh != candidate:
        raise ValueError("stale/forged source/final/query/roots/roster/domain")
    return {"fresh": True, "authority": False, "kernel_outcome": "UNOBSERVED"}


def prepare_chain(buffers, *, allocator, index_bits=64):
    """Fresh raw-chain route: retained plan certificates are unnecessary."""
    if allocator != ALLOCATOR or type(index_bits) is not int or index_bits != 64:
        raise ValueError("explicit total/private/unobservable allocator and index64 required")
    hint = reconstruct_chain(buffers, index_bits=index_bits)
    result = prepare(buffers[0], buffers[-1], hint, allocator=allocator, index_bits=index_bits)
    return replace(result, raw_chain_json=canonical(buffers))
