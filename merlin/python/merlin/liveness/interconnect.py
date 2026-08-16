"""(A) Dynamic transaction-level liveness model.

Replay a decoded memory-movement stream (a RoCC ``instruction_trace`` from
:mod:`merlin.targetgen.rocc_decode`) through the target's finite on-chip resources — sized from
introspected capacities (:class:`~merlin.liveness.facts.SiliconFacts`) — and surface the *silicon*
hazards a functional oracle cannot see:

* **scratchpad overflow / back-pressure** — the resident tile footprint walks past the introspected
  scratchpad depth, so loads alias, wrap, or stall behind a full DMA on real silicon;
* **unmapped / provenance-wrong DRAM movement** — a load/store targets an address outside the derived
  DRAM window, or bakes a literal address under a pointer-argument harness (it cannot match the buffer
  the runtime allocated);
* **visibility / missing drain** — the stream does not quiesce (close with a ``FENCE``), so the final
  stores are not guaranteed visible at program halt.

It is a *screening* model: conservative, transaction-level (not cycle-accurate), and honest about what
it cannot derive (``UNKNOWN`` findings). Every capacity/address bound is DERIVED from the target's facts,
never a per-target literal.
"""
from __future__ import annotations

from typing import Any

from .facts import SiliconFacts
from .report import Finding, Severity

# Instruction classes are the decoder's semantic roles (from rocc_decode), not target literals.
_DRAIN = {"FENCE"}
_MOVEMENT = {"MVIN", "MVOUT"}


def _rows(dec: dict) -> int:
    r = dec.get("rows")
    return int(r) if isinstance(r, int) and r > 0 else 1


def simulate(
    trace: dict,
    facts: SiliconFacts,
    *,
    address_model: str | None = None,
    dram_bytes: int | None = None,
) -> tuple[list[Finding], dict[str, Any]]:
    """Replay ``trace`` against ``facts``. Returns ``(findings, resource_peaks)``.

    ``address_model`` is the harness's DRAM-addressing convention (``"pointer_args"`` | ``"fixed_preload"``
    | None) — under ``pointer_args`` a baked literal DRAM address is a provenance fault. ``dram_bytes`` is
    the DRAM window size when the caller can supply it (from the board/manifest); absent, only the lower
    bound + provenance are enforced and the upper bound is surfaced as ``UNKNOWN``.
    """
    findings: list[Finding] = []
    peaks: dict[str, Any] = {}
    ins = trace.get("instructions", [])
    classes = [i.get("class") for i in ins]

    # ---- 1. scratchpad footprint / back-pressure (dynamic occupancy) --------------------------------
    cap = facts.scratchpad_rows
    live: set[int] = set()
    max_top = 0
    for i in ins:
        if i.get("class") != "MVIN":
            continue
        dec = i.get("decoded", {}) or {}
        base = dec.get("spad_addr")
        if not isinstance(base, int):
            continue
        top = base + _rows(dec)
        max_top = max(max_top, top)
        live.update(range(base, top))
    peaks["scratchpad_rows_touched"] = len(live)
    peaks["scratchpad_max_row"] = max_top
    peaks["scratchpad_rows_capacity"] = cap
    if cap is None:
        findings.append(Finding(
            "scratchpad-capacity", Severity.UNKNOWN,
            "scratchpad row capacity not derivable — cannot bound the resident footprint",
            derived_from=facts.provenance))
    elif max_top > cap:
        bad = [i["index"] for i in ins
               if i.get("class") == "MVIN"
               and isinstance((i.get("decoded") or {}).get("spad_addr"), int)
               and (i["decoded"]["spad_addr"] + _rows(i["decoded"])) > cap][:8]
        findings.append(Finding(
            "scratchpad-overflow", Severity.STALL,
            f"resident scratchpad footprint reaches row {max_top} but the target has only {cap} rows "
            f"— loads alias/wrap or stall behind a full DMA on silicon",
            where=f"MVIN #{bad[0]}" if bad else None,
            derived_from=facts.provenance,
            evidence={"max_row": max_top, "capacity_rows": cap, "instruction_indices": bad},
            fix_hint="tile K/N smaller or evict resident tiles; the functional oracle hides this (magic memory)"))

    # ---- 2. DRAM address-map legality of every movement transaction ---------------------------------
    base = facts.dram_base
    hi = (base + dram_bytes) if (isinstance(base, int) and isinstance(dram_bytes, int)) else None
    n_unmapped = n_const = n_unknown = 0
    for i in ins:
        if i.get("class") not in _MOVEMENT:
            continue
        dram = (i.get("decoded") or {}).get("dram")
        if not isinstance(dram, dict):
            continue
        kind = dram.get("kind")
        if kind == "const":
            addr = dram.get("raw")
            if address_model == "pointer_args":
                n_const += 1
                findings.append(Finding(
                    "dram-provenance", Severity.FAULT,
                    f"{i.get('class')} #{i.get('index')} bakes a literal DRAM address ({addr}); the harness "
                    f"passes each operand as a pointer argument, so a baked literal cannot match the buffer "
                    f"the runtime allocated",
                    where=f"#{i.get('index')}", derived_from="harness address_model=pointer_args",
                    fix_hint="derive the DRAM address from the matching kernel argument (ptrtoint of the arg)"))
            elif isinstance(addr, int) and isinstance(base, int):
                if addr < base or (hi is not None and addr >= hi):
                    n_unmapped += 1
                    findings.append(Finding(
                        "dram-unmapped", Severity.FAULT,
                        f"{i.get('class')} #{i.get('index')} targets DRAM address {addr:#x} outside the "
                        f"mapped window [{base:#x}..{('%#x' % hi) if hi else '?'}) — faults on silicon",
                        where=f"#{i.get('index')}", derived_from="dram_facts memory-map green card",
                        evidence={"addr": addr, "dram_base": base, "dram_hi": hi}))
        elif kind == "unknown":
            n_unknown += 1
            findings.append(Finding(
                "dram-provenance-unknown", Severity.UNKNOWN,
                f"{i.get('class')} #{i.get('index')} has an unresolved DRAM operand (decoder could not "
                f"derive its provenance) — cannot verify it is mapped",
                where=f"#{i.get('index')}", derived_from="rocc_decode operand resolution"))
        # kind == "argbase": arg-relative → mapped by construction (runtime allocates the buffer).
    peaks["dram_movements"] = sum(1 for c in classes if c in _MOVEMENT)
    peaks["dram_unmapped"] = n_unmapped
    peaks["dram_baked_const"] = n_const
    peaks["dram_unknown_provenance"] = n_unknown
    if hi is None and any(c in _MOVEMENT for c in classes) and isinstance(base, int):
        findings.append(Finding(
            "dram-window-unknown", Severity.UNKNOWN,
            "DRAM window size not supplied — enforced the lower bound + provenance only; upper bound unchecked",
            derived_from="dram_facts (base only)"))

    # ---- 3. visibility / drain: the stream must quiesce so final stores are visible at halt ---------
    has_store = "MVOUT" in classes
    if has_store and ins:
        # find the last non-drain, non-config movement/compute and confirm a FENCE follows it.
        last_work = max((k for k, c in enumerate(classes)
                         if c in (_MOVEMENT | {"COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"})), default=None)
        drains_after = any(classes[k] in _DRAIN for k in range(last_work + 1, len(classes))) if last_work is not None else False
        if not drains_after:
            findings.append(Finding(
                "visibility-no-drain", Severity.STALL,
                "the stream does not close with a FENCE after its last store/compute — final results are "
                "not guaranteed visible at program halt on silicon (the functional oracle commits eagerly)",
                derived_from="ordering invariant (quiesce-before-halt)",
                fix_hint="emit a closing FENCE to drain outstanding movement before halt"))
    peaks["closes_with_fence"] = bool(classes and classes[-1] in _DRAIN)

    return findings, peaks
