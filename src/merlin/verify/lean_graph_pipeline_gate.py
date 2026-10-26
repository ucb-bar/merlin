"""Live whole-program gate authority, separate from local proof diagnostics.

Only this trusted host's fresh checker invocation may issue a completion. Saved
JSON, status strings, dataclass reconstruction and local legacy receipts cannot.
Python module integrity and parser/projection correspondence remain trusted.
"""
from __future__ import annotations

import weakref
from dataclasses import asdict, dataclass

from . import lean_graph_pipeline_gate_native as native
from .lean_graph_pipeline_gate_projection import GateNotEstablished, GateUnavailable, GateUnsupported, canonical, prepare, sha


@dataclass(frozen=True, eq=False)
class WholeProgramCompletion:
    schema: str
    prepared_identity: str
    source: str
    target: str
    domain: str
    relation: str
    premises: tuple[str, ...]
    query_sha256: str
    roots: tuple[str, ...]
    toolchain_identity: str
    audit_sha256: str

    def to_dict(self):
        # Descriptive serialization deliberately has no loader/issuer API.
        return {**asdict(self), "status": "conditional_model_relation",
                "native_fidelity": "not_established", "review_approval": False,
                "source_success_in_general": False, "poison_freedom": False}


@dataclass(frozen=True)
class GateReport:
    status: str
    reason: str

    def to_dict(self):
        return asdict(self)


_LIVE = weakref.WeakValueDictionary()
_STD3 = frozenset({"propext", "Classical.choice", "Quot.sound"})


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def verify(change, *, toolchain=None, deadline=None):
    return _verify(change, toolchain=toolchain, deadline=deadline)


def _verify(change, *, toolchain=None, deadline=None, expected_identity=None):
    try:
        prepared = prepare(change)
        if expected_identity is not None and prepared.identity != expected_identity:
            raise GateUnavailable("fresh replay differs from the exact original prepared source")
        completed = native.check(prepared, toolchain=toolchain, deadline=deadline)
        if type(completed) is not native.KernelCompletion:
            raise GateUnavailable("checker returned no typed live completion")
        if (completed.prepared_identity != prepared.identity or completed.query_sha256 != sha(prepared.query)
                or completed.checked_roots != prepared.roots
                or type(completed.axioms) is not tuple or not set(completed.axioms) <= _STD3
                or not _digest(completed.toolchain_identity) or not _digest(completed.audit_sha256)):
            raise GateUnavailable("checker completion lacks exact query/root/std3/context binding")
        # Fresh re-projection also reopens source identities. No successful query
        # can authorize intervening source, mapper, parser, or program changes.
        if prepare(change) != prepared:
            raise GateUnavailable("boundary or trusted source changed during checking")
        receipt = WholeProgramCompletion(
            "merlin.live_whole_program_completion.v1", prepared.identity,
            prepared.source, prepared.target, prepared.domain, prepared.relation,
            prepared.premises, sha(prepared.query), prepared.roots,
            completed.toolchain_identity, completed.audit_sha256)
        _LIVE[id(receipt)] = receipt
        return receipt
    except GateUnsupported as error:
        return GateReport("unsupported", str(error))
    except GateNotEstablished as error:
        return GateReport("not_established", str(error))
    except (GateUnavailable, OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as error:
        return GateReport("unavailable", str(error))


def replay(change, original_prepared, *, toolchain=None, deadline=None):
    """Saved metadata constrains fresh work; it never seeds the live ledger.

    Never open paths from the untrusted document. Reconstruct the entire current
    request/source closure, compare closed canonical data, then run the checker.
    """
    try:
        prepared = prepare(change)
        if type(original_prepared) is not dict or canonical(original_prepared) != canonical(asdict(prepared)):
            raise GateUnavailable("saved original differs from fresh complete preparation")
        return _verify(change, toolchain=toolchain, deadline=deadline, expected_identity=prepared.identity)
    except GateUnsupported as error:
        return GateReport("unsupported", str(error))
    except (GateUnavailable, OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as error:
        return GateReport("unavailable", str(error))


def accepted(receipt, change, *, toolchain=None):
    if type(receipt) is not WholeProgramCompletion or _LIVE.get(id(receipt)) is not receipt:
        return False
    try:
        prepared = prepare(change)
        if (receipt.schema != "merlin.live_whole_program_completion.v1"
                or receipt.prepared_identity != prepared.identity or receipt.source != prepared.source
                or receipt.target != prepared.target or receipt.domain != prepared.domain
                or receipt.relation != prepared.relation or receipt.premises != prepared.premises
                or receipt.query_sha256 != sha(prepared.query) or receipt.roots != prepared.roots):
            return False
        # No old completion can bypass a newly imposed native dependency hold.
        # This revalidation must use the same selected identity, never refresh it.
        return native.revalidate(receipt, toolchain=toolchain) is True
    except (GateUnsupported, GateUnavailable, OSError, ValueError, TypeError, KeyError, AttributeError, ImportError):
        return False
