"""Separate INACTIVE live delivery candidate: endpoint AND boundary roster.

No modification to the reviewed v1 service or its mock-only route identity.
This module requires separate reviewed service integration and native activation.
"""
from collections import OrderedDict
from dataclasses import dataclass, asdict
import time
import weakref

from . import lean_graph_pipeline_gate_chain as chain
from . import lean_graph_pipeline_gate_chain_native as native
from . import lean_graph_pipeline_gate as gate
from . import lean_graph_pipeline_gate_service as service
from .lean_graph_pipeline_gate_projection import GateUnavailable, canonical, sha


@dataclass(frozen=True, eq=False)
class EndpointCompletion:
    prepared_identity: str
    query_sha256: str
    roots: tuple[str, ...]
    toolchain_identity: str
    audit_sha256: str


_LIVE = weakref.WeakKeyDictionary()
_DELIVERIES = OrderedDict()


def apply_service(selected_service, request, *, work_root, timeout_s):
    """Inactive additive route around the unchanged reviewed service.

    Only the original service's live roster is consumed, never serialized
    receipt fields. Its exact source cut and per-boundary identity are preserved.
    This wrapper is not selected by any public service or compiler entry point.
    """
    from merlin.targetgen.common_transforms import CommonTransformService
    if type(selected_service) is not CommonTransformService:
        raise GateUnavailable("exact trusted service implementation required")
    if type(timeout_s) is not int or not 1 <= timeout_s <= 3600:
        raise GateUnavailable("explicit bounded chain service budget required")
    binding = selected_service.graph_binding
    context = service.validate_binding(binding, compiler_profile_sha256=selected_service.profile["sha256"], work_root=work_root)
    deadline = time.monotonic() + min(timeout_s, binding.timeout_s)
    before = canonical(request)
    original = request.get("source_ir")
    document = selected_service.apply(request, work_root=work_root, timeout_s=timeout_s)
    if canonical(request) != before or not selected_service.compiler_eligible(document, source_ir=original):
        raise GateUnavailable("service request changed or no original live boundary delivery")
    found = service._DELIVERIES.get(id(document))
    if found is None or found[0] is not document or found[2] is not binding or found[3] != original:
        raise GateUnavailable("service live roster unavailable; saved receipts cannot substitute")
    changes, receipts = found[4], found[5]
    prepared = chain.prepare(original, changes)
    endpoint = verify(original, changes, toolchain=context, deadline=deadline)
    # Separate new document and ledger: never mutate the v1 service delivery.
    delivery = {**document, "route": "inactive_graph_chain_candidate_v1", "plan_only": False,
                "chain_prepared_identity": prepared.identity,
                "endpoint": {**asdict(endpoint), "qualification": "UNQUALIFIED"},
                "compiler_pass_chain": "not_established"}
    record_delivery(binding, delivery, original, changes, receipts, endpoint, deadline=deadline)
    return delivery


def verify(original, changes, *, toolchain, deadline):
    prepared = chain.prepare(original, changes)
    result = native.check(prepared, toolchain=toolchain, deadline=deadline)
    if (type(result) is not native.KernelCompletion or result.prepared_identity != prepared.identity
            or result.query_sha256 != sha(prepared.query) or result.checked_roots != prepared.roots
            or type(result.axioms) is not tuple or not set(result.axioms) <= gate._STD3
            or not gate._digest(result.toolchain_identity) or not gate._digest(result.audit_sha256)):
        raise GateUnavailable("endpoint lacks exact typed query/root/std3 completion")
    if chain.prepare(original, changes) != prepared:
        raise GateUnavailable("endpoint/source changed during checking")
    receipt = EndpointCompletion(prepared.identity, sha(prepared.query), prepared.roots,
                                 result.toolchain_identity, result.audit_sha256)
    if native.revalidate(receipt, toolchain=toolchain) is not True:
        raise GateUnavailable("endpoint same-live revalidation required")
    _LIVE[receipt] = (asdict(receipt), toolchain)
    return receipt


def accepted(receipt, original, changes, *, toolchain):
    if type(receipt) is not EndpointCompletion or receipt not in _LIVE:
        return False
    try:
        saved, context = _LIVE[receipt]
        prepared = chain.prepare(original, changes)
        return (context is toolchain and saved == asdict(receipt)
                and receipt.prepared_identity == prepared.identity and receipt.query_sha256 == sha(prepared.query)
                and receipt.roots == prepared.roots and native.revalidate(receipt, toolchain=toolchain) is True)
    except (ValueError, RuntimeError, OSError, TypeError, AttributeError, KeyError):
        return False


def record_delivery(binding, document, original, changes, receipts, endpoint, *, deadline):
    """Candidate hook AFTER actual plan_only=False service completes both stages.

    Whole-chain authority requires fresh endpoint checking, exact two live
    boundary receipts, full original/middle/final joins and the same source map.
    This does not issue qualification or release the public runtime hold.
    """
    context = service.validate_binding(binding, compiler_profile_sha256=binding.compiler_profile_sha256)
    if type(document) is not dict or type(receipts) is not tuple or len(receipts) != 2:
        raise GateUnavailable("complete live chain delivery and two receipts required")
    if type(deadline) not in (int, float) or not time.monotonic() < deadline < float("inf"):
        raise GateUnavailable("remaining finite delivery deadline required")
    prepared = chain.prepare(original, changes)
    if (document.get("source_sha256") != sha(original) or document.get("transformed_ir") != changes[-1].after_ir
            or document.get("plan_only") is not False
            or document.get("chain_prepared_identity") != prepared.identity
            or document.get("compiler_pass_chain") != "not_established"):
        raise GateUnavailable("delivery lacks exact checked-path endpoint binding or overclaims qualification")
    if not accepted(endpoint, original, changes, toolchain=context):
        raise GateUnavailable("fresh complete original-to-final completion required")
    if not all(gate.accepted(r, c, toolchain=context) for r, c in zip(receipts, changes, strict=True)):
        raise GateUnavailable("every actual adjacent boundary must also have a live completion")
    if time.monotonic() >= deadline:
        raise GateUnavailable("delivery deadline expired during revalidation")
    _DELIVERIES[id(document)] = (document, canonical(document), binding, original, changes, receipts, endpoint)
    if len(_DELIVERIES) > 128:
        _DELIVERIES.popitem(last=False)


def eligible_delivery(binding, document, original):
    found = _DELIVERIES.get(id(document))
    if found is None or found[0] is not document or found[2] is not binding or found[3] != original:
        return False
    try:
        context = service.validate_binding(binding, compiler_profile_sha256=binding.compiler_profile_sha256)
        return (canonical(document) == found[1]
                and accepted(found[6], original, found[4], toolchain=context)
                and all(gate.accepted(r, c, toolchain=context) for r, c in zip(found[5], found[4], strict=True)))
    except (ValueError, RuntimeError, OSError, TypeError, AttributeError, KeyError):
        return False
