"""Can this lever reach the IR at all? — a preflight the search runs before spending board time.

A transform-dialect lever finds its work with ``transform.structured.match ops{["linalg.matmul"]}``.
When the module contains none of the ops it names, the match yields an EMPTY handle and every op
downstream of it becomes a vacuous no-op: the lever builds, runs, gates clean, and changes nothing.
It reports as applied.

That is not hypothetical. ``passes_quant_int`` rewrote every ``linalg.matmul`` into a
``linalg.generic`` (measured: 15 -> 0 on ``small_llama_int8``), while ``impr_features`` matches
``linalg.matmul``/``linalg.batch_matmul`` in 39 places -- so the entire register-blocking and
accumulator-resident family was inert on the int8 datapath. An 87-fork beam search over exactly
those levers emitted 20 distinct binaries, flagged 34 nodes inert, and could not improve on the two
levers that happen to work at the ``linalg.generic`` level. Every one of those forks cost board time.

The check is deliberately STRUCTURAL and target-agnostic: it reads the op names out of the lever's
own schedule text and counts those ops in the prepared module. It knows nothing about which ops
matter, so it keeps working when a new lever or a new dialect appears.
"""
from __future__ import annotations

_MATCH_TOKEN = "transform.structured.match"
_OPS_OPEN = "ops{["
_OPS_CLOSE = "]}"


def _match_handles(schedule_text: str) -> dict[str, tuple[str, ...]]:
    """``{result handle -> op names}`` for every ``transform.structured.match ops{[...]}`` line."""
    out: dict[str, tuple[str, ...]] = {}
    for line in (schedule_text or "").splitlines():
        if _MATCH_TOKEN not in line or _OPS_OPEN not in line:
            continue
        lhs, _, _rhs = line.partition("=")
        handle = lhs.strip().split()[0].strip() if "%" in lhs else ""
        open_at = line.find(_OPS_OPEN)
        close_at = line.find(_OPS_CLOSE, open_at)
        if close_at < 0:
            continue
        names = tuple(sorted({t.strip().strip('"').strip()
                              for t in line[open_at + len(_OPS_OPEN):close_at].split(",")
                              if t.strip().strip('"').strip()}))
        if handle and names:
            out[handle] = names
    return out


def _payload_handles(schedule_text: str) -> set[str]:
    """Handles a STRUCTURED transform actually consumes -- i.e. the lever's real work targets.

    A schedule also matches container ops it never transforms: ``%f = ...match ops{["func.func"]}``
    exists only to receive ``transform.apply_patterns to %f``. Counting those as work targets is
    what made the first version of this check report ``applicable`` for the exact case it was
    written to catch -- ``func.func`` is always present, so a lever whose every payload op had
    vanished still looked fine. Payload-ness is read off the schedule's own dataflow rather than
    from a list of container op names, which would go stale.
    """
    used: set[str] = set()
    for line in (schedule_text or "").splitlines():
        stripped = line.strip()
        if "transform.structured." not in stripped or _MATCH_TOKEN in stripped:
            continue
        _lhs, _, rhs = stripped.partition("=")
        body = rhs if _ else stripped
        for tok in body.replace(",", " ").replace("(", " ").replace(")", " ").split():
            if tok.startswith("%"):
                used.add(tok.split(":")[0].strip())
    return used


def matched_op_names(schedule_text: str) -> tuple[str, ...]:
    """The op names this schedule actually TRANSFORMS, parsed structurally (no regex).

    De-duplicated and sorted, and restricted to handles a structured transform consumes, so a
    container match (``func.func`` for ``apply_patterns``) does not count as a work target. A
    schedule that matches by interface or attribute rather than by name yields an empty tuple, which
    this module treats as "cannot be judged" -- never as "inapplicable", because refusing a lever we
    cannot analyse would be worse than running it.
    """
    handles = _match_handles(schedule_text)
    payload = _payload_handles(schedule_text)
    names: set[str] = set()
    for handle, ops in handles.items():
        if handle in payload:
            names.update(ops)
    return tuple(sorted(names))


def all_matched_op_names(schedule_text: str) -> tuple[str, ...]:
    """Every op name the schedule matches, including containers it only reads. Diagnostic only."""
    names: set[str] = set()
    for ops in _match_handles(schedule_text).values():
        names.update(ops)
    return tuple(sorted(names))


def module_op_counts(module) -> dict[str, int]:
    """How many of each op the prepared module actually contains."""
    counts: dict[str, int] = {}
    for op in module.walk():
        counts[op.name] = counts.get(op.name, 0) + 1
    return counts


def applicability(schedule_text: str, op_counts: dict[str, int]) -> dict:
    """``{status, needs, present, reason}`` for one lever against one prepared module.

    ``status`` is ``applicable`` | ``inapplicable`` | ``unknown``. ``unknown`` means the schedule
    does not match by op name, so this check has nothing to say -- reported honestly rather than
    guessed, on the same fail-open terms the rest of the search uses for what it cannot measure.
    """
    needs = matched_op_names(schedule_text)
    if not needs:
        return {"status": "unknown", "needs": (), "present": {},
                "reason": "schedule does not match by op name; applicability not decidable here"}
    present = {n: int(op_counts.get(n, 0)) for n in needs}
    if any(v > 0 for v in present.values()):
        return {"status": "applicable", "needs": needs, "present": present, "reason": ""}
    return {"status": "inapplicable", "needs": needs, "present": present,
            "reason": (f"the module contains none of {list(needs)}, so every "
                       f"transform.structured.match on them yields an empty handle and the lever "
                       f"is a no-op; it would still build, gate clean and report as applied")}
