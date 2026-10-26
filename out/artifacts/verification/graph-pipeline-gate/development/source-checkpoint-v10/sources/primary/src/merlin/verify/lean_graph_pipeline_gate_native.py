"""Native boundary held pending independently justified runtime authorization.

There is deliberately no environment flag, receipt field or caller callback
that can release this hold. This source must receive a reviewed new implementation
after coordinator authorization, with exact original tool/import/root bindings.
Old scalar prepare/check_query paths are never called as a fallback.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import time

from .lean_graph_pipeline_gate_projection import GateNotEstablished, GateUnavailable, sha

_ISSUED = OrderedDict()
_SEQUENCE = 0
_STD3 = frozenset({"propext", "Classical.choice", "Quot.sound"})


@dataclass(frozen=True)
class QueryRequest:
    """SDK-free handoff shape; no request object grants completion authority."""
    module: str
    query: bytes
    final_root: str
    support_roots: tuple[str, ...]
    project_sources: tuple[tuple[str, str, str, int], ...]


def request(prepared):
    """Construct the exact request for the reviewed common Context provider.

    The context itself is selected by the trusted host, not this request. It
    must independently validate source/import/object/runtime bindings before
    any native action and again in Context.revalidate(CheckedQuery).
    """
    from pathlib import Path

    if prepared.schema != "merlin.graph_pipeline_prepared.v3" or prepared.query_contract != "pure_declarations_separate_root_audit_v1":
        raise GateUnavailable("explicit pure-declaration/separate-audit query contract required")
    if not prepared.roots or prepared.roots[-1] != "merlin_transform":
        raise GateUnavailable("missing exact complete-program final root")
    if len(set(prepared.roots)) != len(prepared.roots):
        raise GateUnavailable("duplicate expected proof roots")
    sources = []
    for ref in prepared.sources:
        path = Path(ref.path)
        if path.suffix == ".lean":
            # This route's fixed existing roots all live directly under Merlin.
            # No path from a saved report can select an SDK or extra source.
            if path.parent.name != "Merlin" or not path.stem.isidentifier():
                raise GateUnavailable("project source outside the supported root layout")
            sources.append(("Merlin." + path.stem, ref.path, ref.sha256, ref.size))
    if len({name for name, _, _, _ in sources}) != len(sources):
        raise GateUnavailable("ambiguous project module source selection")
    if not {"Merlin.GraphStateScalar", "Merlin.GraphStateGateEntries"} <= {name for name, _, _, _ in sources}:
        raise GateUnavailable("complete scalar/roster source roots missing")
    return QueryRequest("GraphPipelineGateQuery", prepared.query.encode(), "merlin_transform",
                        prepared.roots[:-1], tuple(sources))


@dataclass(frozen=True)
class KernelCompletion:
    prepared_identity: str
    query_sha256: str
    checked_roots: tuple[str, ...]
    axioms: tuple[str, ...]
    toolchain_identity: str
    audit_sha256: str


def check(prepared, *, toolchain=None, deadline=None):
    """No SDK file is opened and no checker/process is launched while held."""
    raise GateUnavailable("selected libleanshared runtime incident UNKNOWN; native gate is held")


def revalidate(receipt, *, toolchain=None):
    """Future implementation must recheck the SAME authorized original context."""
    raise GateUnavailable("selected libleanshared runtime incident UNKNOWN; live admission is held")


def _check_with_context(prepared, *, toolchain, deadline=None):
    """INACTIVE integration candidate, callable in SDK-free mocked tests only.

    Public check/revalidate above remain hard-held. After explicit coordinator
    review/authorization a NEW source version may connect them to these methods.
    This function does not select a context or read old SDK/archived images.
    """
    from .lean_graph_pipeline_gate_provider import bound_context

    controller, data, _ = bound_context(toolchain)
    CheckedQuery = controller.CheckedQuery
    ProofRefused, InfrastructureStop = controller.ProofRefused, controller.InfrastructureStop
    ContextRefusal, FileRef = data.ContextRefusal, data.FileRef

    global _SEQUENCE
    selected = request(prepared)
    expected = {name: FileRef(path, digest, size) for name, path, digest, size in selected.project_sources}
    # This gate selects only its minimal closure. Witnesses and other proof
    # domains need separately reviewed contexts, never an implicit source union.
    if toolchain.originals != expected:
        raise GateUnavailable("gate project source selection differs from common context")
    if deadline is not None and time.monotonic() >= deadline:
        raise GateUnavailable("gate host deadline exhausted before checker invocation")
    _SEQUENCE += 1
    selected = replace(selected, module=selected.module + "_" + prepared.identity + "_" + str(_SEQUENCE))
    try:
        checked = toolchain.check_query(selected.module, selected.query, final_root=selected.final_root,
                                       support_roots=selected.support_roots)
        if (type(checked) is not CheckedQuery or checked.context_id != toolchain.context_id
                or checked.module != selected.module or checked.query.sha256 != sha(selected.query)
                or checked.query.bytes != len(selected.query) or checked.final_root != selected.final_root
                or checked.support_roots != selected.support_roots):
            raise GateUnavailable("common checker returned the wrong complete query/context/root")
        ordered = (selected.final_root, *selected.support_roots)
        if tuple(name for name, _ in checked.axioms) != ordered:
            raise GateUnavailable("common checker did not audit every exact gate root")
        axioms = tuple(sorted({axiom for _, values in checked.axioms for axiom in values}))
        if not set(axioms) <= _STD3:
            raise GateUnavailable("common checker report exceeds the standard-three axiom policy")
        if toolchain.revalidate(checked) is not checked:
            raise GateUnavailable("common checker did not revalidate the same live completion")
        if deadline is not None and time.monotonic() >= deadline:
            raise GateUnavailable("gate host deadline exhausted during checker invocation; no completion issued")
        result = KernelCompletion(prepared.identity, sha(selected.query), prepared.roots, axioms,
                                  checked.context_id, checked.audit.sha256)
        key = (result.prepared_identity, result.toolchain_identity, result.audit_sha256)
        _ISSUED[key] = (toolchain, checked)
        if len(_ISSUED) > 128:
            _ISSUED.popitem(last=False)
        return result
    except ProofRefused as error:
        raise GateNotEstablished(str(error)) from error
    except (InfrastructureStop, ContextRefusal) as error:
        raise GateUnavailable("common context unavailable/held: " + str(error)) from error


def _revalidate_with_context(receipt, *, toolchain=None):
    """INACTIVE paired adapter; no saved document recreates the live handle."""
    from .lean_graph_pipeline_gate_provider import bound_context

    key = (receipt.prepared_identity, receipt.toolchain_identity, receipt.audit_sha256)
    found = _ISSUED.get(key)
    if found is None:
        raise GateUnavailable("missing original live common-context completion")
    context, checked = found
    if toolchain is not None and toolchain is not context:
        raise GateUnavailable("admission selected a different context object")
    controller, data, _ = bound_context(context)
    try:
        return context.revalidate(checked) is checked
    except (controller.InfrastructureStop, data.ContextRefusal) as error:
        raise GateUnavailable("common context no longer admits the original completion: " + str(error)) from error
