"""Inactive endpoint adapter. Public entry points remain unconditionally held.

Only mocked typed providers exercise the private candidate in SDK-free tests.
This is a new query contract; no old per-boundary receipt is promoted.
"""
from collections import OrderedDict
from pathlib import Path
import time

from .lean_graph_pipeline_gate_native import KernelCompletion, QueryRequest
from .lean_graph_pipeline_gate_projection import GateNotEstablished, GateUnavailable, sha

_ISSUED = OrderedDict()
_SEQUENCE = 0
_STD3 = frozenset({"propext", "Classical.choice", "Quot.sound"})


def request(prepared):
    from .lean_graph_pipeline_gate_chain import ChainPrepared
    if (type(prepared) is not ChainPrepared or prepared.schema != "merlin.graph_gate_chain_prepared.v1"
            or prepared.query_contract != "pure_declarations_separate_root_audit_v1"):
        raise GateUnavailable("exact new chain preparation required")
    if not prepared.roots or prepared.roots[-1] != "merlin_chain_transform" or len(set(prepared.roots)) != len(prepared.roots):
        raise GateUnavailable("exact distinct chain endpoint/support roots required")
    sources = tuple(("Merlin." + Path(r.path).stem, r.path, r.sha256, r.size)
                    for r in prepared.sources if r.path.endswith(".lean"))
    from .lean_graph_pipeline_gate_provider import _PROJECT_MODULES
    if {r[0] for r in sources} != _PROJECT_MODULES or len(sources) != len(_PROJECT_MODULES):
        raise GateUnavailable("chain requires exact unchanged minimal project closure")
    return QueryRequest("GraphPipelineGateChainQuery", prepared.query.encode(), prepared.roots[-1], prepared.roots[:-1], sources)


def check(prepared, *, toolchain=None, deadline=None):
    raise GateUnavailable("chain native candidate held; runtime incident UNKNOWN; no activation")


def revalidate(receipt, *, toolchain=None):
    raise GateUnavailable("chain live authority held; no activation")


def _check_with_context(prepared, *, toolchain, deadline):
    from .lean_graph_pipeline_gate_provider import bound_context
    controller, data, _ = bound_context(toolchain)
    selected = request(prepared)
    if toolchain.held is not False or toolchain.originals != {
            name: data.FileRef(path, digest, size) for name, path, digest, size in selected.project_sources}:
        raise GateUnavailable("held or different exact chain source context")
    if type(deadline) not in (int, float) or not time.monotonic() < deadline < float("inf"):
        raise GateUnavailable("finite remaining chain deadline required")
    global _SEQUENCE
    _SEQUENCE += 1
    module = selected.module + "_" + prepared.identity + "_" + str(_SEQUENCE)
    try:
        checked = toolchain.check_query(module, selected.query, final_root=selected.final_root, support_roots=selected.support_roots)
        if (type(checked) is not controller.CheckedQuery or checked.context_id != toolchain.context_id
                or checked.module != module or checked.query.sha256 != sha(selected.query)
                or checked.query.bytes != len(selected.query) or checked.final_root != selected.final_root
                or checked.support_roots != selected.support_roots
                or tuple(n for n, _ in checked.axioms) != (selected.final_root, *selected.support_roots)):
            raise GateUnavailable("wrong exact chain query/root/type/context report")
        axioms = tuple(sorted({a for _, values in checked.axioms for a in values}))
        if not set(axioms) <= _STD3 or toolchain.revalidate(checked) is not checked:
            raise GateUnavailable("chain std3 or same-live-query revalidation failed")
        if time.monotonic() >= deadline:
            raise GateUnavailable("chain deadline exhausted; no completion")
        result = KernelCompletion(prepared.identity, sha(selected.query), prepared.roots,
                                  axioms, checked.context_id, checked.audit.sha256)
        _ISSUED[(result.prepared_identity, result.toolchain_identity, result.audit_sha256)] = (toolchain, checked)
        if len(_ISSUED) > 128:
            _ISSUED.popitem(last=False)
        return result
    except controller.ProofRefused as error:
        raise GateNotEstablished(str(error)) from error
    except (controller.InfrastructureStop, data.ContextRefusal) as error:
        raise GateUnavailable("chain context held/unavailable: " + str(error)) from error


def _revalidate_with_context(receipt, *, toolchain=None):
    from .lean_graph_pipeline_gate_provider import bound_context
    found = _ISSUED.get((receipt.prepared_identity, receipt.toolchain_identity, receipt.audit_sha256))
    if found is None or toolchain is not found[0]:
        raise GateUnavailable("missing SAME original chain completion/context")
    context, checked = found
    controller, data, _ = bound_context(context)
    if context.held is not False:
        raise GateUnavailable("chain context held")
    try:
        return context.revalidate(checked) is checked
    except (controller.InfrastructureStop, data.ContextRefusal) as error:
        raise GateUnavailable(str(error)) from error
