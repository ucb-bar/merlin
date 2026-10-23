"""Bounded exception, simulator and output diagnostics for trusted capsule grading."""

from __future__ import annotations

import traceback as _traceback
from pathlib import Path


def _candidate_exception_diagnostic(exc: Exception, *, stage: str) -> dict:
    """Keep a bounded, source-free stack when a model run fails before a receipt.

    The public result already exposes the exception class and message. Frame
    names and line numbers make failures such as RecursionError actionable
    without copying source lines, arguments, or private artifact paths.
    """
    frames = [
        {"module": Path(frame.f_code.co_filename).name, "function": frame.f_code.co_name, "line": line}
        for frame, line in _traceback.walk_tb(exc.__traceback__)
    ]
    return {
        "type": type(exc).__name__,
        "detail": str(exc)[:2000],
        "stage": stage,
        "stack_depth": len(frames),
        "stack": frames if len(frames) <= 20 else frames[:4] + frames[-16:],
    }


def _clip(msg: str, budget: int) -> str:
    """Clip a long exception to ``budget`` chars KEEPING BOTH ENDS -- the head names the failure, the tail
    usually carries the operand/shape specifics.

    A bare ``msg[-budget:]`` throws the head away, and the head is where the diagnosis lives. Measured:
    the muon operand-binding error is one long sentence whose useful half is first ("could not derive
    harness operands: <which of three cases>") and whose second half is fixed advisory prose. Tail-only
    clipping rendered five radiance capsules as `cyclotron crash: es (outputs: ['Y0'])` -- the real cause
    truncated to the last two letters of "shapes" -- and they read as unexplained infra failures across
    many runs while the message that would have named them was being discarded every time.
    """
    msg = str(msg)
    if len(msg) <= budget:
        return msg
    head = budget * 2 // 3  # bias to the head: that is the diagnosis
    tail = budget - head - 5
    return f"{msg[:head]} […] {msg[-tail:]}" if tail > 0 else msg[:budget]


def _readback_refusal(exc: BaseException) -> BaseException | None:
    """The ``ReadbackIntegrityError`` in ``exc``'s cause/context chain, if any.

    The refusal is raised deep in a target's oracle and re-wrapped on the way out (the muon oracle
    raises ``MuonError(str(exc)) from exc``; ``oot_runner`` wraps again). Only the TYPE survives that
    intact, so the chain is walked structurally -- matching the message would break the first time
    the wording changed, which is exactly the class of defect this whole path exists to stop.
    """
    try:
        from merlin.common.readback_integrity import ReadbackIntegrityError
    except ImportError:  # pragma: no cover - module always present in this tree
        return None
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ReadbackIntegrityError):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None


def _engine_token(oracle_kind: str | None) -> str | None:
    """The elaborated-RTL ENGINE named inside an oracle kind, or None.

    An adapter reports a kind like ``rtl_gsim`` / ``rtl_verilator`` / ``rtl_gsim_<something>``; the
    engine is one token inside it. Read STRUCTURALLY (split on the separator, test membership against
    the policy's own declared engine vocabulary) rather than by pattern — a pattern that is slightly too
    narrow silently reports "no engine" for a conformant spelling, which here would mean a cert whose
    engine is unrecorded, i.e. the exact thing this field exists to prevent.
    """
    tokens = {t for part in str(oracle_kind or "").split("_") for t in part.split("-")}
    from . import rtl_engine_policy as _pol

    for engine in _pol.ENGINE_PRIORITY:
        if engine in tokens:
            return engine
    return None


def _record_console(paths, name: str, console) -> tuple[str | None, int | None]:
    """Write a sim console to the capsule's artifacts dir and return ``(filename, bytes)``.

    One writer for all three tier paths, which each open-coded the same two lines and none of which
    recorded the SIZE. Size is not decoration: a console is how an RTL verdict is read, and "the marker
    is absent" and "the console was truncated before the marker" are the same bytes to a reader who
    cannot tell how much there was.
    """
    if console is None:
        return None, None
    if type(console) is bytes:
        fname = f"{name}_console.bin"
        (paths.artifacts_dir / fname).write_bytes(console)
        return fname, len(console)
    text = str(console)
    fname = f"{name}_console.log"
    (paths.artifacts_dir / fname).write_text(text, encoding="utf-8")
    return fname, len(text.encode("utf-8"))


def _oracle_kind(oracle_meta) -> str | None:
    """A filesystem-safe name for the engine an adapter reports, or ``None`` when it reports none.

    Adapters return either a rich ``{"kind": ..., "derived_from_rtl": ...}`` dict or a bare provenance
    string. Both carry the engine's identity; the contract's static ``tier_sim`` map does not, because a
    backend may substitute a different RTL-derived engine at runtime. ``None`` means "the oracle did not
    say", and the caller keeps the declared name -- never a guess.
    """
    kind = oracle_meta.get("kind") if isinstance(oracle_meta, dict) else oracle_meta
    if not isinstance(kind, str) or not kind.strip():
        return None
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in kind.strip())
    return safe or None


def _absent_outputs(nrep: dict) -> list[str]:
    """Outputs the kernel DECLARED (present in the interface/golden) but never wrote — or wrote at the
    wrong length — extracted from :func:`capsule_golden.compare`'s ``per_output``. These structural
    failures are the ones that read as ``mismatch_count > 0`` while ``max_abs_error == 0`` and
    ``first_mismatch is None`` (there is no value to diff — the bytes were never produced), a
    self-contradictory signal unless the missing output is named. Reveals no golden VALUE: only the
    identity of a declared output, which the agent already holds in the interface it was handed."""
    po = nrep.get("per_output") or {}
    return [
        nm
        for nm, d in po.items()
        if isinstance(d, dict)
        and d.get("status") == "fail"
        and str(d.get("reason", "")).startswith(("missing", "length"))
    ]


def _unwritten_output_detail(nrep: dict, sim_name: str) -> str | None:
    """A precise detail when a declared output was READ BACK but holds one constant value, else None.

    Sibling of :func:`_absent_output_detail`, for the case that one misses: there the output key is
    absent from the readback; here it is present and uniformly the buffer's untouched fill, so the
    generic "does not compute the declared operation" is technically true and useless. Measured cost of
    not having this: twelve capsules reported ``functional_mismatch`` for six consecutive rounds with
    ``observed: 0.0`` and ``max_rel_error: 1.0``, while the mismatch COUNT was a function of the
    golden's zero distribution rather than of the kernel -- so the number could not move no matter what
    was emitted, and the agent spent six rounds tuning numerics against an unwritable signal.

    Reveals no golden value: only that the observed side is constant, which the agent can compute from
    its own readback."""
    names = nrep.get("outputs_never_written") or []
    if not names:
        return None
    po = nrep.get("per_output") or {}
    got = {n: (po.get(n) or {}).get("observed_constant") for n in names}
    shown = ", ".join(f"{n} (all {got[n]})" for n in names)
    return (
        f"on {sim_name}, output(s) {shown} came back as a SINGLE CONSTANT value while the expected "
        f"result varies — the store never landed, so what was compared is the buffer's untouched "
        f"fill, not a computed result. This is a WRITEBACK failure, not a numeric one: the mismatch "
        f"count here is set by the reference's own value distribution and will NOT move if you only "
        f"change arithmetic. Decode your emitted artifact and confirm the result is actually stored "
        f"to this output's base address (and that any DMA/commit is awaited before readback)."
    )


def _range_check_bounds(msg: str):
    """(index, size) from a C++ ``_M_range_check`` abort, or (None, None).

    libstdc++ spells it ``__n (which is 1024) >= this->size() (which is 1024)``. Parsed STRUCTURALLY --
    no regex in library code (build_tools/scripts/check_no_regex.py) -- by splitting on the literal
    ``(which is `` marker, which is what libstdc++ emits. Returns None rather than guessing when the
    shape does not match, so a changed message degrades to the generic crash text instead of inventing
    a bound.
    """
    if "_M_range_check" not in msg and "out_of_range" not in msg:
        return None, None
    marker = "(which is "
    parts = msg.split(marker)
    if len(parts) < 3:
        return None, None
    vals = []
    for chunk in parts[1:3]:
        head = chunk.partition(")")[0].strip()
        if not head.isdigit():
            return None, None
        vals.append(int(head))
    return vals[0], vals[1]
