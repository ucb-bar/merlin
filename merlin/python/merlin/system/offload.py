"""Which contractions a DEVICE could take — derived from that device, not from constants.

The existing offload path answers this with two module-level literals: a dtype triple
``("i8","i8","i32")`` and an op table ``{"linalg.matmul": 2, "linalg.batch_matmul": 3}``. They are
correct for the one device that path was built for, and they are *its* facts: the triple is literally
that device's first declared ``accumulate`` rule, and the ranks are what its ``contraction`` semantic
capability declares. Written as literals they belong to nobody, so a second device either inherits
another device's datapath or needs a second copy of the pass.

Here the same question is asked of a named device and answered from what that device already
declares. Adding a device adds no code.

**Fail closed.** A device whose datapath cannot be derived offloads NOTHING. The alternative --
assuming a triple -- produces a rewrite that compiles and silently computes in the wrong precision,
which is far worse than declining. ``why_not`` says which of the three gates rejected a contraction
so a caller can report it rather than guess.
"""
from __future__ import annotations

from typing import Any

__all__ = ["device_contraction_ranks", "device_dtype_triples", "offloadable_contractions",
           "why_not"]


def _mlir(token: str) -> str | None:
    """A dtype token in MLIR spelling, or None when the registry does not know it (never guessed)."""
    try:
        from merlin.targetgen.corpus_spec import dtype_info
        return dtype_info(str(token))[1]
    except Exception:            # noqa: BLE001 -- an unknown token is a real answer: skip that rule
        return None


def _units(device_name: str) -> list[dict]:
    """The device's compute units as declared/derived in its capability manifest."""
    try:
        from merlin.targetgen.target_experiment import load_capability_manifest
        man = load_capability_manifest(device_name)
        return list((man.contract or {}).get("compute_units") or [])
    except Exception:            # noqa: BLE001
        return []


def _triples_from_facts(device_name: str) -> tuple[tuple[str, str, str], ...]:
    """The datapath triple read straight off the RTL facts' ``datapaths`` block.

    The accumulate matrix in a manifest is a PROJECTION of these facts, and a hand-written contract
    may simply not carry one -- measured: one target's curated contract declares ``accumulate: []``
    while its facts plainly record ``input i8`` and ``accumulator i32``. Reading the facts directly
    means a device is not treated as having no datapath merely because nobody transcribed one.

    Operand and weight share the ``input`` datapath: a contraction datapath feeds both sides of the
    multiply from the same storage class. Where that is untrue the manifest's accumulate matrix says
    so and takes precedence (it is consulted first).
    """
    try:
        from merlin.targetgen.rtl import facts as _f
        body = (_f.load_facts(device_name) or {}).get("facts") or {}
    except Exception:            # noqa: BLE001 -- ungrounded facts are a real answer
        return ()
    by_name = {str(d.get("name")): str(d.get("dtype")) for d in (body.get("datapaths") or ())
               if d.get("name") and d.get("dtype")}
    inp, acc = by_name.get("input"), by_name.get("accumulator")
    if not inp or not acc:
        return ()
    lhs, a = _mlir(inp), _mlir(acc)
    return ((lhs, lhs, a),) if lhs and a else ()


def device_dtype_triples(device_name: str) -> tuple[tuple[str, str, str], ...]:
    """``(lhs, rhs, acc)`` in MLIR spelling for every datapath this device declares.

    Two sources, in order. The unit's ``accumulate`` matrix is the richer one -- it can express
    several datapaths and mixed operand/weight formats -- so it wins where present. Where a contract
    carries none, the RTL facts' own ``datapaths`` block still grounds one, and using it is what keeps
    "this contract never transcribed an accumulate matrix" from reading as "this device has no
    datapath". A rule naming a dtype the registry cannot spell is skipped rather than approximated:
    a triple is a precision claim, and a wrong one is silent.
    """
    out: list[tuple[str, str, str]] = []
    for unit in _units(device_name):
        for rule in (unit.get("accumulate") or ()):
            lhs, rhs, acc = (_mlir(rule.get("in") or ""), _mlir(rule.get("weight") or ""),
                             _mlir(rule.get("acc") or ""))
            if lhs and rhs and acc and (lhs, rhs, acc) not in out:
                out.append((lhs, rhs, acc))
    if out:
        return tuple(out)
    return _triples_from_facts(device_name)


def device_contraction_ranks(device_name: str) -> tuple[int, ...] | None:
    """Legal output ranks for a contraction on this device; ``None`` means unconstrained.

    ``None`` and ``()`` mean the same thing in the capability model (unconstrained) and that reading
    is kept here: a device that never narrowed its ranks has not thereby forbidden every rank.
    """
    ranks: set[int] = set()
    saw = False
    for unit in _units(device_name):
        for cap in (unit.get("semantic_capabilities") or ()):
            if str(cap.get("family")) != "contraction":
                continue
            saw = True
            for r in (cap.get("ranks") or ()):
                ranks.add(int(r))
    return tuple(sorted(ranks)) if (saw and ranks) else None


def why_not(shape, *, triples, ranks) -> str | None:
    """Why this contraction is not offloadable here, or None when it is. Reported, never inferred."""
    if not triples:
        return "device declares no derivable datapath (no accumulate rule this registry can spell)"
    if tuple(shape.dtypes) not in triples:
        return (f"dtypes {tuple(shape.dtypes)} not among the device's datapaths "
                f"{sorted(triples)}")
    if len(shape.reduction) != 1:
        return f"{len(shape.reduction)} reduction dims; a contraction datapath takes exactly one"
    if ranks is not None and len(shape.parallel) not in ranks:
        return f"output rank {len(shape.parallel)} not among the device's legal ranks {list(ranks)}"
    return None


def offloadable_contractions(module, device_name: str, *,
                             require_zero_init: bool = True) -> list[tuple[Any, Any]]:
    """``[(op, shape)]`` for every contraction ``device_name`` COULD take. No decision is made.

    "Could" means legal on this device's declared datapath and shape envelope -- never profitable.
    Whether a legal contraction is worth moving is a placement decision, and keeping the two apart is
    why this enumerates instead of rewriting.

    ``require_zero_init`` keeps the correctness condition the offload ABI depends on: ``linalg.matmul``
    computes ``C_init + A@B`` while a device kernel that OVERWRITES its output computes ``A@B``, and
    those agree only when ``C_init`` is zero. A contraction accumulating onto a live init is declined
    rather than silently losing the addend.
    """
    from merlin.kernels.shapes import observe_contractions

    triples = device_dtype_triples(device_name)
    ranks = device_contraction_ranks(device_name)
    if not triples:
        return []                       # fail closed: an underivable datapath offloads nothing

    picked: list[tuple[Any, Any]] = []
    for op, shape in observe_contractions(module):
        if why_not(shape, triples=triples, ranks=ranks) is not None:
            continue
        if require_zero_init:
            from merlin.llvmlower.passes_opu import zero_initialised
            if not zero_initialised(op):
                continue
        picked.append((op, shape))
    return picked
