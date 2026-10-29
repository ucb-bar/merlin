"""Source-only two-kind graph chain, with a fresh common interpretation.

No per-boundary numeric descriptor identity is reused. Complete buffers are
reprojected in one descriptor/type universe. Emitting these obligations neither
checks them nor activates the compiler service.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import sys

from .lean_graph_pipeline_gate_projection import (
    DOMAIN, RELATION, PREMISES, GateUnsupported, canonical, complete_query,
    prepare as prepare_boundary, sha, source_closure,
)

ORDER = ("scalar_redundancy_elimination", "scalar_loop_invariant_motion")


@dataclass(frozen=True)
class ChainPrepared:
    schema: str
    python_version: str
    query_contract: str
    original: str
    buffers: tuple[str, ...]
    changes: tuple[str, ...]
    boundaries: tuple[str, ...]
    domain: str
    relation: str
    premises: tuple[str, ...]
    projections: tuple[str, ...]
    query: str
    roots: tuple[str, ...]
    sources: tuple

    @property
    def identity(self):
        return sha(canonical(asdict(self)))

    def describe(self):
        return {**asdict(self), "identity": self.identity,
                "proof_status": "UNCOMPILED", "qualification": "UNQUALIFIED",
                "compiler_pass_chain": "not_established", "saved_report_is_authority": False}


def _roster(projection):
    return [{key: row[key] for key in ("name", "header", "inputs", "outputs", "declaration")}
            for row in projection["functions"]]


def _query(projections):
    imports, chunks, roots = [], [], []
    for label, projection in zip(("Boundary0", "Boundary1", "Endpoint"), projections, strict=True):
        text, support = complete_query(projection)
        # Rename only serializer-owned declarations, never IR strings. The
        # serializer emits no descriptor text, symbol names or arbitrary code.
        replacements = (("Merlin.GraphStateScalarInstance", f"Merlin.GraphGateChain.{label}.Scalar"),
                        ("Merlin.GraphPipelineGateInstance", f"Merlin.GraphGateChain.{label}.Entries"),
                        ("merlin_transform", f"merlin_chain_{label.lower()}"))
        for old, new in replacements:
            text = text.replace(old, new)
            support = tuple(root.replace(old, new) for root in support)
        for line in text.splitlines():
            if line.startswith("import "):
                if line not in imports:
                    imports.append(line)
            else:
                chunks.append(line)
        roots.extend(root for root in support if root not in roots)
    base = "Merlin.GraphGateChain."
    a, b, e = (base + label + ".Entries" for label in ("Boundary0", "Boundary1", "Endpoint"))
    chunks += [
        f"theorem merlin_chain_registry_join : {a}.registry = {b}.registry := rfl",
        f"theorem merlin_chain_endpoint_registry : {a}.registry = {e}.registry := rfl",
        f"theorem merlin_chain_middle_join : {a}.afterEntries = {b}.beforeEntries := rfl",
        f"theorem merlin_chain_original_join : {e}.beforeEntries = {a}.beforeEntries := rfl",
        f"theorem merlin_chain_final_join : {e}.afterEntries = {b}.afterEntries := rfl",
        "theorem merlin_chain_composed {Value World Error : Type}",
        f"    (model : Merlin.GraphState.Scalar.Model Value World Error {e}.registry) :",
        f"    Merlin.GraphState.GateEntries.PreservesEntries model {e}.beforeEntries {e}.afterEntries := by",
        "  intro identity arguments world",
        f"  exact ({a}.complete model identity arguments world).trans",
        f"    ({b}.complete model identity arguments world)",
        "theorem merlin_chain_transform {Value World Error : Type}",
        f"    (model : Merlin.GraphState.Scalar.Model Value World Error {e}.registry) :",
        f"    Merlin.GraphState.GateEntries.PreservesEntries model {e}.beforeEntries {e}.afterEntries :=",
        f"  {e}.complete model",
    ]
    roots += ["merlin_chain_" + name for name in (
        "registry_join", "endpoint_registry", "middle_join", "original_join", "final_join",
        "composed", "transform")]
    return "\n".join(imports + chunks) + "\n", tuple(roots)


def prepare(original, changes):
    """Reconstruct every field; no saved projection/registry/query is accepted.

    The original is the caller's raw source. Each producer certificate continues
    to describe exactly ONE actual change. The endpoint has no fake certificate.
    """
    from merlin.transforms.certificates import TransformationChange, ir_text, parse_ir
    from .lean_graph_input_admission import admit_text, admit_module
    from .lean_graph_state_scalar_capture import project

    if type(original) is not str or len(original.encode()) > 4 * 1024 * 1024:
        raise GateUnsupported("bounded complete original source required")
    if type(changes) is not tuple or len(changes) != 2 or any(type(c) is not TransformationChange for c in changes):
        raise GateUnsupported("exact two actual changed boundary roster required")
    if tuple(c.kind for c in changes) != ORDER:
        raise GateUnsupported("exact CSE/DCE then LICM order required")
    if any(c.before_ir == c.after_ir for c in changes):
        raise GateUnsupported("each actual boundary must change")
    if changes[0].after_ir != changes[1].before_ir:
        raise GateUnsupported("complete middle buffer adjacency mismatch")
    admit_text(original)
    source = parse_ir(original)
    admit_module(source, calls="external-only")
    if ir_text(source) != changes[0].before_ir:
        raise GateUnsupported("original does not join first complete boundary")
    # This validates exact producer kind/scope/certificate and all module fields.
    boundaries = tuple(prepare_boundary(change) for change in changes)
    from merlin.transforms.validation import validate_change
    for change in changes:
        # Producer replay binds attribution/witnesses only. It is not used as
        # a proof of equality; the independent complete Lean queries remain.
        validate_change(change)
    buffers = (changes[0].before_ir, changes[0].after_ir, changes[1].after_ir)
    modules = tuple(parse_ir(text) for text in buffers)
    for module in modules:
        admit_module(module, calls="external-only")
    projections = (project(modules[0], modules[1], context=modules),
                   project(modules[1], modules[2], context=modules),
                   project(modules[0], modules[2], context=modules))
    for p in projections[1:]:
        if any(p[key] != projections[0][key] for key in ("registry", "types", "module_header")):
            raise GateUnsupported("common descriptor/type/module interpretation differs")
        if _roster(p) != _roster(projections[0]):
            raise GateUnsupported("complete defined/declaration roster or interface differs")
    for left, right, x, y in ((0, 1, "after", "before"), (2, 0, "before", "before"), (2, 1, "after", "after")):
        if any(a[x] != b[y] for a, b in zip(projections[left]["functions"], projections[right]["functions"], strict=True)):
            raise GateUnsupported("reprojected complete program join differs")
    query, roots = _query(projections)
    if len(query.encode()) > 3 * 1024 * 1024:
        raise GateUnsupported("complete chain query budget exceeded")
    return ChainPrepared("merlin.graph_gate_chain_prepared.v1", sys.version,
                         "pure_declarations_separate_root_audit_v1", sha(original), tuple(sha(b) for b in buffers),
                         tuple(sha(canonical(asdict(c))) for c in changes),
                         tuple(p.identity for p in boundaries), DOMAIN, RELATION, PREMISES,
                         tuple(canonical(p) for p in projections), query, roots, source_closure())


def matches(saved, original, changes):
    """Saved data only constrains source preparation; never creates authority."""
    return type(saved) is dict and canonical(saved) == canonical(prepare(original, changes).describe())
