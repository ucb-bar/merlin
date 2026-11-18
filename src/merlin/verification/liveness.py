"""The single derivation seam for the liveness oracle: normalize a target's CIRCT/mlc-introspected
facts (+ its memory-map green card) into a :class:`SiliconFacts` the models consume.

FAIL-CLOSED, per the repo's cardinal rule: every field is derived from the target's OWN sources
(``rtl.facts.load_facts`` — mesh geometry, on-chip memory capacities, the legal funct/opcode set — and
``dram_facts.dram_base_for`` — the DRAM base from the shipped memory-map markdown). A fact that cannot be
derived is left ``None`` (surfaced via :meth:`SiliconFacts.unknowns`), NEVER a target-shaped default —
an assumed 16×16 mesh or 256 KB scratchpad would silently mis-model a different target.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import sys
from pathlib import Path
from typing import Any

from merlin.common.target_data import interface as _facts_interface
from merlin.target.rtl.facts import load_facts

# ---- facts -------------------------------------------------------------------------------------


@dataclasses.dataclass
class SiliconFacts:
    """Normalized, provenance-tagged silicon facts for one target. Any ``None`` = not derivable."""

    target: str
    mesh_rows: int | None = None
    mesh_cols: int | None = None
    scratchpad_bytes: int | None = None
    scratchpad_rows: int | None = None  # SRAM depth in rows (MVIN spad_addr is a row index)
    accumulator_bytes: int | None = None
    accumulator_rows: int | None = None
    accumulator_row_bytes: int | None = None  # derived acc row width (cols * acc elem bytes)
    acc_ctrl_mask: int | None = None  # derived high control bits carried in an acc address
    legal_funct: list[int] | None = None
    custom_opcode: int | None = None
    funct3: int | None = None
    dram_base: int | None = None
    provenance: str = "UNKNOWN"
    resources: dict[str, dict[str, dict[str, Any]]] = dataclasses.field(default_factory=dict)

    def unknowns(self) -> list[str]:
        """Names of the facts that could not be derived (each blocks the checks that need it)."""
        return [
            name
            for name in (
                "mesh_rows",
                "mesh_cols",
                "scratchpad_rows",
                "accumulator_rows",
                "legal_funct",
                "dram_base",
            )
            if getattr(self, name) is None
        ]


def _memory(f: dict, name: str) -> dict | None:
    return next((m for m in f.get("memories", []) if m.get("name") == name), None)


def _dtype_bits(dtype: str | None) -> int | None:
    """Bit width of a datapath dtype token (``i8``->8, ``i32``->32, ``bf16``->16, ``fp8_e4m3``->8).

    RESOLVED THROUGH THE FORMAT REGISTRY FIRST, and only then by the token's own digits. A registered
    format's name is not a width with letters around it: ``fp8_e4m3`` carries three separate digit runs
    and the digit reading returns 843, which then sizes a row 105x too wide and reports every capacity
    check as a fit. The digit fallback stays for the plain machine spellings a registry does not carry
    (an ``i32`` accumulator is a machine type, not a way of encoding a quantized value), and a token
    neither path recognises is UNKNOWN rather than a number.
    """
    if not isinstance(dtype, str):
        return None
    try:
        from merlin.common import target_data as qf

        return qf.get(dtype).element_bits
    except Exception:  # noqa: BLE001 -- not a registered format: fall through to the machine spelling
        pass
    digits = "".join(c for c in dtype if c.isdigit())
    return int(digits) if digits else None


def _datapath_bits(f: dict, name: str) -> int | None:
    dp = next((d for d in f.get("datapaths", []) if d.get("name") == name), None)
    if not dp:
        return None
        # A derived datapath states its width as a NUMBER beside the dtype, because the width is what was
        # measured and the dtype is the name the design put on it. Prefer the measurement.
    bits = dp.get("elem_bits")
    return int(bits) if isinstance(bits, int) and bits > 0 else _dtype_bits(dp.get("dtype"))


def _rows_from_bytes(total_bytes: int | None, cols: int | None, elem_bits: int | None) -> int | None:
    """Logical addressable row count = total_bytes / row_width, row_width = ceil(cols * elem_bits / 8).
    All three inputs are derived facts; returns None if any is missing (the capacity check then skips)."""
    if not total_bytes or not cols or not elem_bits:
        return None
    row_bytes = (cols * elem_bits + 7) // 8
    return total_bytes // row_bytes if row_bytes else None


def silicon_facts(target: str, *, fact_record: dict[str, Any] | None = None) -> SiliconFacts:
    """Derive :class:`SiliconFacts` for ``target``. Never raises for a missing/degraded fact source —
    it fails closed to ``None`` fields (the consuming checks skip and surface ``UNKNOWN``)."""
    out = SiliconFacts(target=target)
    try:
        rec = load_facts(target) if fact_record is None else fact_record
    except Exception:  # noqa: BLE001 — facts not derivable → everything stays UNKNOWN
        out.provenance = "UNKNOWN (RTL facts not derivable)"
        _fill_dram_base(out)
        return out

    f = rec.get("facts", rec)
    # Keep the selected resource descriptions, including resources outside the
    # scratchpad/accumulator transaction model below. No capacities are guessed.
    out.resources = {
        kind: {row["name"]: dict(row) for row in f.get(kind, ()) if isinstance(row, dict) and row.get("name")}
        for kind in ("arrays", "memories", "datapaths")
    }
    mesh = next((a for a in f.get("arrays", []) if a.get("name") == "mesh"), None)
    if mesh:
        out.mesh_rows = mesh.get("rows")
        out.mesh_cols = mesh.get("cols")

        # Element widths from the datapath dtypes (i8→8, i32→32, …) — used to size a logical row. The
        # scratchpad holds input-dtype elements; the accumulator holds accumulator-dtype elements.
    in_bits = _datapath_bits(f, "input")
    acc_bits = _datapath_bits(f, "accumulator")

    sp = _memory(f, "scratchpad")
    if sp:
        out.scratchpad_bytes = sp.get("bytes")
        # scratchpad_rows is the LOGICAL addressable row count an MVIN spad_addr indexes — NOT the physical
        # SRAM depth (banked/wider, would under-count and false-flag passing kernels). rows = bytes /
        # ceil(cols * input_elem_bits / 8), same bound rtl_checks._check_spad_capacity derives.
        out.scratchpad_rows = _rows_from_bytes(out.scratchpad_bytes, out.mesh_cols, in_bits)

    acc = _memory(f, "accumulator")
    if acc:
        out.accumulator_bytes = acc.get("bytes")
        if out.mesh_cols and acc_bits:
            out.accumulator_row_bytes = (out.mesh_cols * acc_bits + 7) // 8
            # Total addressable accumulator rows = bytes / row_bytes (spans all banks) — the introspected
            # ``depth`` is per-bank and would under-count a multi-bank accumulator.
        out.accumulator_rows = _rows_from_bytes(out.accumulator_bytes, out.mesh_cols, acc_bits)

    ft = _facts_interface(f, "funct_decode_table")
    if ft:
        out.legal_funct = ft.get("legal_funct")
        out.custom_opcode = ft.get("custom_opcode")
        out.funct3 = ft.get("funct3")

    gen = rec.get("generator", {}) if isinstance(rec, dict) else {}
    out.provenance = f"circt_introspect facts.json ({gen.get('version', '?')})"
    _fill_acc_ctrl_mask(out)
    _fill_dram_base(out)
    return out


def _fill_acc_ctrl_mask(out: SiliconFacts) -> None:
    """The control bits an accumulator address carries in its high bits (full-C / i8-readout / accumulate),
    OR'd into one mask so a consumer can strip them to recover the raw row index. Derived from the target's
    RoCC ISA constants (``readout_bits``), never a literal; ``None`` when not derivable (acc check skips)."""
    try:
        # `targetgen.rocc_decode` was folded into the `targetgen.rocc` package; the old top-level
        # alias is gone, so this import raised ModuleNotFoundError and the except below swallowed
        # it -- the mask came back None and the accumulator check silently SKIPPED. Measured: an
        # arm-4 run was aborted in round 0 on exactly this class of stale import.
        from merlin.target.isa.rocc.decode import isa_constants

        isa = isa_constants(out.target)
        bits = [isa.get(k) for k in ("FULL_C_BIT", "ACC_I8", "ACC_ACCUM")]
        derived = [b for b in bits if isinstance(b, int)]
        if derived:
            mask = 0
            for b in derived:
                mask |= b
            out.acc_ctrl_mask = mask
    except Exception:  # noqa: BLE001 — no derivable ISA constants → mask stays None (acc check skips)
        pass


def _fill_dram_base(out: SiliconFacts) -> None:
    """DRAM base from the target's shipped memory-map green card (0 for a 0-based RoCC target). Optional
    — a target without a parsed map leaves it ``None`` (the address-map check then bounds only what it
    can and surfaces the gap)."""
    try:
        from merlin.target.isa.machine_model import dram_base_for

        base = dram_base_for(out.target)
        if isinstance(base, int):
            out.dram_base = base
    except Exception:  # noqa: BLE001 — no derivable memory map → dram_base stays None
        pass


# ---- report ------------------------------------------------------------------------------------
# Shared result types for the liveness oracle: a severity-ranked :class:`Finding` and the aggregate
# :class:`LivenessReport`. Kept dependency-free so both the static linter and the dynamic model emit the
# same shape and a caller can merge them.


class Severity(str, enum.Enum):
    """Ordered by how badly it breaks a silicon run. ``UNKNOWN`` is not benign — it means a precondition
    could NOT be derived (fail-closed), so it ranks above ``WARN``: an underived check is a gap the caller
    must see, not a pass."""

    FAULT = "fault"  # will fault / cannot run or build on silicon
    STALL = "stall"  # may hang: back-pressure, missing drain, capacity deadlock
    UNKNOWN = "unknown"  # a precondition could not be derived (surfaced, never dropped)
    WARN = "warn"  # risky but likely benign
    INFO = "info"  # informational (peaks, headroom)

    @property
    def rank(self) -> int:
        return {"fault": 4, "stall": 3, "unknown": 2, "warn": 1, "info": 0}[self.value]

        # Verdict is the worst severity present, mapped to a single word.


_VERDICT = {4: "fault", 3: "stall", 2: "unknown", 1: "warn", 0: "ok"}


@dataclasses.dataclass
class Finding:
    """One precondition/liveness observation. ``rule`` is a stable slug; ``derived_from`` records the
    provenance of the fact the rule used (so a reader can see the check was DERIVED, not assumed)."""

    rule: str
    severity: Severity
    message: str
    where: str | None = None  # pc / addr / instruction index / symbol / tensor
    derived_from: str | None = None  # provenance of the fact used (fail-closed audit trail)
    evidence: dict | None = None
    fix_hint: str | None = None

    def to_dict(self) -> dict:
        d = {
            "rule": self.rule,
            "severity": self.severity.value,
            "message": self.message,
        }
        for k in ("where", "derived_from", "fix_hint"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.evidence:
            d["evidence"] = self.evidence
        return d


@dataclasses.dataclass
class LivenessReport:
    """Aggregate of every finding for one program against one target, plus dynamic resource peaks."""

    target: str
    program: str | None
    findings: list[Finding] = dataclasses.field(default_factory=list)
    resource_peaks: dict = dataclasses.field(default_factory=dict)

    def add(self, f: Finding | None) -> None:
        if f is not None:
            self.findings.append(f)

    def extend(self, fs) -> None:
        for f in fs:
            self.add(f)

    @property
    def worst_rank(self) -> int:
        return max((f.severity.rank for f in self.findings), default=0)

    @property
    def verdict(self) -> str:
        """``fault`` | ``stall`` | ``unknown`` | ``warn`` | ``ok`` — the worst severity present."""
        return _VERDICT[self.worst_rank]

    def by_severity(self, sev: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == sev]

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "program": self.program,
            "verdict": self.verdict,
            "findings": [f.to_dict() for f in self.findings],
            "resource_peaks": self.resource_peaks,
            "counts": {sev.value: len(self.by_severity(sev)) for sev in Severity if self.by_severity(sev)},
        }


# ---- interconnect ------------------------------------------------------------------------------
# (A) Dynamic transaction-level liveness model.
#
# Replay a decoded memory-movement stream (a RoCC ``instruction_trace`` from
# :mod:`merlin.development.targetgen.rocc_decode`) through the target's finite on-chip resources — sized from
# introspected capacities (:class:`~merlin.verification.liveness.SiliconFacts`) — and surface the *silicon*
# hazards a functional oracle cannot see:
#
# * **scratchpad overflow / back-pressure** — the resident tile footprint walks past the introspected
#   scratchpad depth, so loads alias, wrap, or stall behind a full DMA on real silicon;
# * **unmapped / provenance-wrong DRAM movement** — a load/store targets an address outside the derived
#   DRAM window, or bakes a literal address under a pointer-argument harness (it cannot match the buffer
#   the runtime allocated);
# * **visibility / missing drain** — the stream does not quiesce (close with a ``FENCE``), so the final
#   stores are not guaranteed visible at program halt.
#
# It is a *screening* model: conservative, transaction-level (not cycle-accurate), and honest about what
# it cannot derive (``UNKNOWN`` findings). Every capacity/address bound is DERIVED from the target's facts,
# never a per-target literal.

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
    dram_window_why: str | None = None,
) -> tuple[list[Finding], dict[str, Any]]:
    """Replay ``trace`` against ``facts``. Returns ``(findings, resource_peaks)``.

    ``address_model`` is the harness's DRAM-addressing convention (``"pointer_args"`` | ``"fixed_preload"``
    | None) — under ``pointer_args`` a baked literal DRAM address is a provenance fault. ``dram_bytes`` is
    the DRAM window size when the caller can supply it (derived from the target's own memory map by
    ``merlin.target.isa.machine_model.dram_window_for``); absent, only the lower bound + provenance are
    enforced and the upper bound is surfaced as ``UNKNOWN``.

    ``dram_window_why`` is that derivation's PROVENANCE — why the size is what it is, or why it could not
    be derived. It rides here so the ``dram-window-unknown`` finding can name the actual reason (the
    target ships no memory map / its card gives no upper bound) instead of the generic "not supplied",
    which read identically whether the gap was in the target's data or in this tooling. Absent, the
    finding keeps its historical wording, so every existing caller is byte-identically unaffected.
    """
    findings: list[Finding] = []
    peaks: dict[str, Any] = {}
    ins = trace.get("instructions", [])
    classes = [i.get("class") for i in ins]

    # ---- 1. scratchpad footprint / back-pressure (dynamic occupancy) --------------------------------
    # AN MVIN DESTINATION IS NOT ALWAYS A SCRATCHPAD ROW. The same `spad_addr` field carries the high
    # control bits that say the destination is the ACCUMULATOR, and the accumulator budget below already
    # masks them off (`a & ~mask`) while this loop used the address raw. So an accumulator-destined MVIN
    # was read as scratchpad row 0x80000010 == 2147483664 and reported as
    #   "resident scratchpad footprint reaches row 2147483664 but the target has only 16384 rows"
    # -- a STALL verdict, with a fix_hint telling the author to tile smaller, for a correctly lowered
    # program. MEASURED across 3446 reports on disk: 229 scratchpad-overflow findings and 313 `stall`
    # verdicts, most of them this shape. It also under-counted the accumulator, because that budget
    # collects only MVOUT `acc_addr` and PRELOAD `c_addr`.
    #
    # Which bit means what is NOT assumed: the only thing used is the DERIVED mask. Any address with a
    # control bit set is not a plain row, so it is folded into the accumulator side (masked) instead,
    # and a row that cannot be interpreted at all is reported UNKNOWN rather than silently counted.
    cap = facts.scratchpad_rows
    _spad_mask = facts.acc_ctrl_mask
    live: set[int] = set()
    max_top = 0
    mvin_acc_rows: list[int] = []
    _uninterpretable = 0
    for i in ins:
        if i.get("class") != "MVIN":
            continue
        dec = i.get("decoded", {}) or {}
        base = dec.get("spad_addr")
        if not isinstance(base, int):
            continue
        if _spad_mask is None:
            if base > (cap or 0):
                _uninterpretable += 1
                continue
        elif base & _spad_mask:
            mvin_acc_rows.append(base & ~_spad_mask)
            continue
        top = base + _rows(dec)
        max_top = max(max_top, top)
        live.update(range(base, top))
    if _uninterpretable:
        findings.append(
            Finding(
                "scratchpad-address-uninterpretable",
                Severity.UNKNOWN,
                f"{_uninterpretable} MVIN destination(s) carry high bits above the {cap}-row scratchpad and "
                "the address control-bit mask is not derivable — cannot tell an accumulator destination "
                "from an out-of-range row, so the resident footprint is not bounded here",
                derived_from=facts.provenance,
            )
        )
    peaks["scratchpad_rows_touched"] = len(live)
    peaks["scratchpad_max_row"] = max_top
    peaks["scratchpad_rows_capacity"] = cap
    if cap is None:
        findings.append(
            Finding(
                "scratchpad-capacity",
                Severity.UNKNOWN,
                "scratchpad row capacity not derivable — cannot bound the resident footprint",
                derived_from=facts.provenance,
            )
        )
    elif max_top > cap:
        bad = [
            i["index"]
            for i in ins
            if i.get("class") == "MVIN"
            and isinstance((i.get("decoded") or {}).get("spad_addr"), int)
            and (i["decoded"]["spad_addr"] + _rows(i["decoded"])) > cap
        ][:8]
        findings.append(
            Finding(
                "scratchpad-overflow",
                Severity.STALL,
                f"resident scratchpad footprint reaches row {max_top} but the target has only {cap} rows "
                f"— loads alias/wrap or stall behind a full DMA on silicon",
                where=f"MVIN #{bad[0]}" if bad else None,
                derived_from=facts.provenance,
                evidence={"max_row": max_top, "capacity_rows": cap, "instruction_indices": bad},
                fix_hint="tile K/N smaller or evict resident tiles; the functional oracle hides this (magic memory)",
            )
        )

        # ---- 1b. accumulator footprint (mask derived control bits off the acc-row address) --------------
    acc_cap = facts.accumulator_rows
    mask = facts.acc_ctrl_mask
    # acc-row addresses live on MVOUT (acc_addr) and PRELOAD (c_addr), with control bits in the high bits.
    acc_addrs = []
    for i in ins:
        dec = i.get("decoded", {}) or {}
        for key in ("acc_addr", "c_addr"):
            a = dec.get(key)
            if isinstance(a, int):
                acc_addrs.append(a)
                # Accumulator-destined MVINs, already masked above. They belong in THIS budget, not the
                # scratchpad's, and were previously in neither.
    acc_addrs.extend(mvin_acc_rows)
    if acc_addrs:
        if acc_cap is None or mask is None:
            peaks["accumulator_rows_capacity"] = acc_cap
            findings.append(
                Finding(
                    "accumulator-capacity",
                    Severity.UNKNOWN,
                    "accumulator row capacity or the address control-bit mask is not derivable — cannot bound "
                    "the accumulator footprint",
                    derived_from=facts.provenance,
                )
            )
        else:
            acc_rows = [a & ~mask for a in acc_addrs]
            max_acc = max(acc_rows)
            peaks["accumulator_max_row"] = max_acc
            peaks["accumulator_rows_capacity"] = acc_cap
            if max_acc >= acc_cap:
                findings.append(
                    Finding(
                        "accumulator-overflow",
                        Severity.STALL,
                        f"accumulator row {max_acc} (control bits masked) reaches/exceeds the {acc_cap}-row "
                        f"accumulator — output tiles alias/wrap on silicon",
                        derived_from=facts.provenance,
                        evidence={"max_acc_row": max_acc, "capacity_rows": acc_cap, "ctrl_mask": f"{mask:#x}"},
                        fix_hint="commit/evict accumulator tiles sooner or tile N smaller",
                    )
                )

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
                findings.append(
                    Finding(
                        "dram-provenance",
                        Severity.FAULT,
                        f"{i.get('class')} #{i.get('index')} bakes a literal DRAM address ({addr}); the harness "
                        f"passes each operand as a pointer argument, so a baked literal cannot match the buffer "
                        f"the runtime allocated",
                        where=f"#{i.get('index')}",
                        derived_from="harness address_model=pointer_args",
                        fix_hint="derive the DRAM address from the matching kernel argument (ptrtoint of the arg)",
                    )
                )
            elif isinstance(addr, int) and isinstance(base, int):
                if addr < base or (hi is not None and addr >= hi):
                    n_unmapped += 1
                    findings.append(
                        Finding(
                            "dram-unmapped",
                            Severity.FAULT,
                            f"{i.get('class')} #{i.get('index')} targets DRAM address {addr:#x} outside the "
                            f"mapped window [{base:#x}..{('%#x' % hi) if hi else '?'}) — faults on silicon",
                            where=f"#{i.get('index')}",
                            derived_from="dram_facts memory-map green card",
                            evidence={"addr": addr, "dram_base": base, "dram_hi": hi},
                        )
                    )
        elif kind == "unknown":
            n_unknown += 1
            findings.append(
                Finding(
                    "dram-provenance-unknown",
                    Severity.UNKNOWN,
                    f"{i.get('class')} #{i.get('index')} has an unresolved DRAM operand (decoder could not "
                    f"derive its provenance) — cannot verify it is mapped",
                    where=f"#{i.get('index')}",
                    derived_from="rocc_decode operand resolution",
                )
            )
            # kind == "argbase": arg-relative → mapped by construction (runtime allocates the buffer).
    peaks["dram_movements"] = sum(1 for c in classes if c in _MOVEMENT)
    peaks["dram_unmapped"] = n_unmapped
    peaks["dram_baked_const"] = n_const
    peaks["dram_unknown_provenance"] = n_unknown
    peaks["dram_window_bytes"] = dram_bytes if isinstance(dram_bytes, int) else None
    if hi is None and any(c in _MOVEMENT for c in classes) and isinstance(base, int):
        # WHY the upper bound is missing is the actionable half. "not supplied" was the same string for
        # a target whose green card ships `start ~ end` and was simply never asked (a tooling gap this
        # repo owns) and for a target that ships no memory map at all (a target-data gap only new facts
        # can close). The caller's derivation knows which; carry its sentence verbatim.
        findings.append(
            Finding(
                "dram-window-unknown",
                Severity.UNKNOWN,
                (
                    f"DRAM window upper bound not derivable: {dram_window_why} — enforced the lower bound + "
                    f"provenance only; upper bound unchecked"
                )
                if dram_window_why
                else "DRAM window size not supplied — enforced the lower bound + provenance only; upper bound unchecked",
                derived_from=f"dram_facts.dram_window_for: {dram_window_why}"
                if dram_window_why
                else "dram_facts (base only)",
            )
        )

        # ---- 3. visibility / drain: the stream must quiesce so final stores are visible at halt ---------
    has_store = "MVOUT" in classes
    if has_store and ins:
        # find the last non-drain, non-config movement/compute and confirm a FENCE follows it.
        last_work = max(
            (k for k, c in enumerate(classes) if c in (_MOVEMENT | {"COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"})),
            default=None,
        )
        drains_after = (
            any(classes[k] in _DRAIN for k in range(last_work + 1, len(classes))) if last_work is not None else False
        )
        if not drains_after:
            findings.append(
                Finding(
                    "visibility-no-drain",
                    Severity.WARN,
                    "the stream does not close with a FENCE after its last store/compute — final results are "
                    "not guaranteed visible at program halt on silicon (advisory: a host-assisted harness may "
                    "drain externally; on a hostless substrate this is a real visibility hazard)",
                    derived_from="ordering invariant (quiesce-before-halt)",
                    fix_hint="emit a closing FENCE to drain outstanding movement before halt",
                )
            )
    peaks["closes_with_fence"] = bool(classes and classes[-1] in _DRAIN)

    return findings, peaks


# ---- preconditions -----------------------------------------------------------------------------
# (B) Static silicon-precondition linter.
#
# Each rule encodes a real "ran in the functional oracle, hung/faulted on real silicon" bug as a check
# against a DERIVED fact (never a per-target literal); where the needed fact cannot be derived it fails
# closed to an ``UNKNOWN`` finding rather than passing silently.
#
# Rules implemented here (all HW-agnostic):
#
# * ``funct-legality`` — every accelerator instruction's ``funct`` lies in the RTL-derived legal set;
#   an out-of-range funct is an encoding the silicon decoder rejects.
# * ``untranscodable-op`` — for a self-hosted (fixed-format) target, the kernel's rv32 ``.text`` is
#   actually encodable by the target's derived transcoder. This catches the fork-free-SIMT ``fence`` wall:
#   a base-ISA ``fence`` (MISC_MEM, opcode ``0x0f``) is untranscodable, so a kernel containing one cannot
#   be built/run and its final-store visibility cannot be guaranteed.
# * ``host-assist`` — a program destined for a hostless substrate must not depend on the HTIF debug
#   channel (``.htif`` section / ``tohost``): an HTIF image on bare silicon spins forever inside its FIRST
#   print, before any work. (The caller supplies the two derived booleans — image audit + substrate — so
#   this stays target-name-free.)


def funct_legality(trace: dict, facts: SiliconFacts) -> list[Finding]:
    """Every emitted accelerator ``funct`` must be in the RTL-derived legal set."""
    legal = facts.legal_funct
    ins = trace.get("instructions", [])
    if legal is None:
        if any(isinstance(i.get("funct"), int) for i in ins):
            return [
                Finding(
                    "funct-legality",
                    Severity.UNKNOWN,
                    "legal funct set not derivable — cannot verify the emitted opcodes are decodable",
                    derived_from=facts.provenance,
                )
            ]
        return []
    legal_set = set(legal)
    bad = [i for i in ins if isinstance(i.get("funct"), int) and i["funct"] not in legal_set]
    if not bad:
        return []
    functs = sorted({i["funct"] for i in bad})
    return [
        Finding(
            "funct-legality",
            Severity.FAULT,
            f"{len(bad)} instruction(s) use funct {functs} outside the RTL legal set "
            f"[{min(legal_set)}..{max(legal_set)}] — the silicon decoder would reject them",
            where=f"#{bad[0].get('index')}",
            derived_from=facts.provenance,
            evidence={"illegal_functs": functs, "indices": [i.get("index") for i in bad][:8]},
        )
    ]


def untranscodable_op(rv32_text: bytes, target: str) -> list[Finding]:
    """For a fixed-format self-hosted target, confirm the kernel ``.text`` is encodable by the target's
    derived transcoder. A ``TranscodeError`` (e.g. a base-ISA ``fence`` the transcoder cannot encode) is a
    hard precondition failure — the kernel cannot be built/run and its stores' visibility is unguaranteed.

    Not applicable to a RoCC inline-asm target (its transcoder rejects the *model*, not the program) —
    reported as an informational skip, never a fault."""
    try:
        from merlin.target.isa.isa_transcode import FixedFormatTranscoder, TranscodeError
        from merlin.target.isa.machine_model import isa_model_for_target
    except Exception:  # noqa: BLE001 — transcode toolchain unavailable
        return [
            Finding(
                "untranscodable-op",
                Severity.UNKNOWN,
                "ISA transcoder unavailable — cannot verify instruction encodability",
                derived_from="isa_transcode import",
            )
        ]
    try:
        model = isa_model_for_target(target)
    except Exception as e:  # noqa: BLE001 — no derivable ISA model → fail closed
        return [
            Finding(
                "untranscodable-op",
                Severity.UNKNOWN,
                f"ISA model for the target not derivable ({e}) — cannot verify encodability",
                derived_from="isa_model_for_target",
            )
        ]
    try:
        transcoder = FixedFormatTranscoder(model)
    except TranscodeError:
        # Not a fixed-format substrate (e.g. a RoCC inline-asm target) — this rule does not apply.
        return [
            Finding(
                "untranscodable-op",
                Severity.INFO,
                "target is not a fixed-format transcode substrate — rule not applicable",
                derived_from="isa_transcode.FixedFormatTranscoder",
            )
        ]
    try:
        transcoder.transcode_text(rv32_text)
    except TranscodeError as e:
        return [
            Finding(
                "untranscodable-op",
                Severity.FAULT,
                f"the kernel .text contains an instruction the target cannot encode: {e}",
                derived_from="isa_transcode (target-derived opcode table)",
                fix_hint="remove the untranscodable op (e.g. a base-ISA fence on a fork-free SIMT path) or "
                "route through a delivery path that supports it",
            )
        ]
    return []


def vlen_match(*, declared_vlen: int | None, hw_vlen: int | None) -> list[Finding]:
    """A vector program's declared VLEN must match the chip's VLEN.

    ``hw_vlen`` is the target's hardware vector length (derived by the caller from facts/manifest; ``None``
    when the target is not a vector machine — the rule then does not apply and returns nothing).
    ``declared_vlen`` is what the build declared (``VECTOR_MAX_LEN`` / ``-march=…_zvlNNNb``). A mismatch
    silently corrupts state on silicon: an under-declared VLEN overruns per-thread vector storage, and an
    over-declared one doubles LMUL and mis-targets ``-march``. HW-agnostic — the rule carries no VLEN
    literal; both values are supplied."""
    if hw_vlen is None:
        return []  # not a vector target → rule not applicable
    if declared_vlen is None:
        return [
            Finding(
                "vlen-match",
                Severity.UNKNOWN,
                f"target VLEN is {hw_vlen} but the program's declared VLEN was not supplied — cannot verify they match",
                derived_from="facts/manifest (hw VLEN)",
            )
        ]
    if declared_vlen != hw_vlen:
        return [
            Finding(
                "vlen-match",
                Severity.FAULT,
                f"declared VLEN {declared_vlen} != chip VLEN {hw_vlen} — vector state corrupts on silicon "
                f"(under-declared overruns per-thread vector storage; over-declared doubles LMUL / mis-targets march)",
                derived_from="facts/manifest (hw VLEN) vs build declaration",
                fix_hint=f"declare VLEN={hw_vlen} (VECTOR_MAX_LEN and -march=…_zvl{hw_vlen}b) to match the chip",
            )
        ]
    return []


def medany_span(*, uses_medany: bool | None, image_span_bytes: int | None) -> list[Finding]:
    """Under ``-mcmodel=medany`` all addressing is PC-relative within ±2 GB; a large image (e.g. a big
    embedded weights blob) can push symbols out of that window → correct arithmetic on wrong data.

    ``uses_medany`` = whether the target's compile path builds with medany (caller-derived from the target
    compile facts; ``None`` → rule not applicable). ``image_span_bytes`` = the linked image's symbol span
    (caller-derived from the ELF). HW-agnostic — the only constant here is the RISC-V ISA's fixed ±2 GB
    medany window, which is a property of the code model, not of any one target."""
    if not uses_medany or image_span_bytes is None:
        return []  # not a medany target, or no image span supplied → rule not applicable
    window = 1 << 31  # RISC-V medany PC-relative reach (±2 GB); an ISA constant, not a target literal.
    if image_span_bytes >= window:
        return [
            Finding(
                "medany-span",
                Severity.FAULT,
                f"image span {image_span_bytes} bytes ({image_span_bytes / (1 << 30):.2f} GB) meets/exceeds the "
                f"medany ±2 GB PC-relative window — some symbol references cannot be reached and mis-address",
                derived_from="target compile facts (medany) + ELF image span",
                fix_hint="place the large blob AFTER heap/stack in the linker script, or drop below the 2 GB window",
            )
        ]
    if image_span_bytes >= window // 2:
        return [
            Finding(
                "medany-span",
                Severity.WARN,
                f"image span {image_span_bytes / (1 << 30):.2f} GB is over half the medany ±2 GB window — a "
                f"further-placed symbol or a bigger blob could push references out of reach",
                derived_from="target compile facts (medany) + ELF image span",
            )
        ]
    return []


def host_assist(*, hostless: bool | None, has_htif: bool | None, has_tohost: bool | None = None) -> list[Finding]:
    """A program bound for a hostless substrate must not depend on the HTIF debug channel.

    ``hostless`` = the delivery substrate has no fesvr host (derived by the caller from the board/substrate);
    ``has_htif`` / ``has_tohost`` = the image audit (a ``.htif`` section / a ``tohost`` symbol). All three
    are booleans the caller derives, so this rule carries no target name and no ELF parsing of its own."""
    if hostless is None or (has_htif is None and has_tohost is None):
        return [
            Finding(
                "host-assist",
                Severity.UNKNOWN,
                "substrate host-assist and/or image HTIF audit not supplied — cannot check the "
                "HTIF-first-print-hang precondition",
                derived_from="caller (substrate + elf_audit)",
            )
        ]
    depends_on_htif = bool(has_htif) or bool(has_tohost)
    if hostless and depends_on_htif:
        which = ".htif section" if has_htif else "tohost symbol"
        return [
            Finding(
                "host-assist",
                Severity.FAULT,
                f"image depends on the HTIF host channel ({which}) but the delivery substrate has no host — "
                f"it will spin forever inside its FIRST print on bare silicon",
                derived_from="substrate host-assist boundary + image audit",
                fix_hint="link the board's own UART console (console_init before the first character), not HTIF",
            )
        ]
    return []


# ---- oracle ------------------------------------------------------------------------------------
# The unifying entry point: assess a program's silicon liveness against a target, merging the static
# precondition linter (B) and the dynamic transaction-level model (A) into one :class:`LivenessReport`.
#
# A :class:`Program` is a thin container of whatever representations the caller has — a decoded RoCC
# ``instruction_trace``, the kernel's rv32 ``.text``, and the image-audit / substrate booleans. Each check
# consumes what it needs and fails closed (``UNKNOWN``) on what is absent, so a partial program still yields
# a useful, honest report.


@dataclasses.dataclass
class Program:
    """The representations available for one program under assessment (all optional)."""

    name: str | None = None
    trace: dict | None = None  # decoded RoCC instruction_trace (rocc_decode.to_dict)
    rv32_text: bytes | None = None  # self-hosted kernel .text bytes (for transcodability)
    has_htif: bool | None = None  # image audit: a .htif section present?
    has_tohost: bool | None = None  # image audit: a tohost symbol present?
    hostless: bool | None = None  # delivery substrate has no fesvr host?
    address_model: str | None = None  # harness DRAM addressing convention (pointer_args/…)
    dram_bytes: int | None = None  # DRAM window size, when the caller can supply it
    dram_window_why: str | None = None  # provenance of that window (or of why it is not derivable)
    # delivery/build context (not RTL-derivable — the caller who chose the substrate/compile path supplies it):
    declared_vlen: int | None = None  # VLEN the build declared (VECTOR_MAX_LEN / -march zvl)
    hw_vlen: int | None = None  # the target board's hardware VLEN (bits); None = non-vector
    uses_medany: bool | None = None  # does the chosen compile path build with -mcmodel=medany?
    image_span_bytes: int | None = None  # linked image symbol span (for the medany window check)


def assess(program: Program, target: str) -> LivenessReport:
    """Run every applicable liveness/precondition check for ``program`` against ``target``."""
    facts = silicon_facts(target)
    report = LivenessReport(target=target, program=program.name)

    if program.trace is not None:
        report.extend(funct_legality(program.trace, facts))
        findings, peaks = simulate(
            program.trace,
            facts,
            address_model=program.address_model,
            dram_bytes=program.dram_bytes,
            dram_window_why=program.dram_window_why,
        )
        report.extend(findings)
        report.resource_peaks = peaks

    if program.rv32_text is not None:
        report.extend(untranscodable_op(program.rv32_text, target))

    if program.hostless is not None or program.has_htif is not None or program.has_tohost is not None:
        report.extend(host_assist(hostless=program.hostless, has_htif=program.has_htif, has_tohost=program.has_tohost))

        # vector-length + medany rules are delivery/build-context checks (VLEN and medany are substrate/compile
        # properties, not RTL facts), so both sides come from the caller-supplied Program. Each is a no-op unless
        # both its inputs are present (fail-closed / not-applicable), so they are always safe to call.
    report.extend(vlen_match(declared_vlen=program.declared_vlen, hw_vlen=program.hw_vlen))
    report.extend(medany_span(uses_medany=program.uses_medany, image_span_bytes=program.image_span_bytes))

    return report


def persist(report: LivenessReport, *, root: Path | None = None) -> Path:
    """Write ``report`` as JSON under ``out/artifacts/liveness/<target>/<program>.liveness.json`` and
    return the path. Concern-first layout (``liveness/`` owns the subtree, target at the folder level),
    per the generated-output convention."""
    if root is None:
        from merlin.common.paths import artifacts_dir

        root = artifacts_dir() / "liveness" / report.target
    root.mkdir(parents=True, exist_ok=True)
    name = (report.program or "program").replace("/", "_")
    out = root / f"{name}.liveness.json"
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return out


# ---- cli ---------------------------------------------------------------------------------------
# ``merlin-liveness`` CLI — screen a program for silicon liveness/precondition hazards.
#
# Answers "would this stall or fault on real silicon?" from the target's OWN derived facts (never a
# per-target literal), failing closed on what cannot be derived. Consumes a decoded RoCC
# ``instruction_trace`` (from ``merlin-targetgen`` / ``rocc_decode``) and/or image-audit flags.
#
# Usage:
#   merlin-liveness --target <t> --trace path/to/instruction_trace.json --address-model pointer_args
#   merlin-liveness --target <t> --trace <t.json> --dram-bytes 0x40000000 --persist
#   merlin-liveness --target <t> --trace <t.json> --hostless --has-htif   # host-assist precondition
#
# Exit code: 2 if any FAULT finding (unrunnable on silicon); 1 if --strict and any STALL; else 0.


def _int(s: str | None) -> int | None:
    if s is None:
        return None
    return int(s, 0)  # accepts 0x… and decimal


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="merlin-liveness",
        description="HW-agnostic silicon liveness/precondition screening (would it stall/fault on silicon?).",
    )
    ap.add_argument("--target", required=True, help="target name (facts are derived from its RTL/manifest)")
    ap.add_argument("--trace", type=Path, help="decoded RoCC instruction_trace.json")
    ap.add_argument("--name", default=None, help="program label (default: derived from --trace path)")
    ap.add_argument("--address-model", default=None, help="harness DRAM addressing convention (e.g. pointer_args)")
    ap.add_argument(
        "--dram-bytes", default=None, help="DRAM window size (int, accepts 0x…); enables the upper-bound address check"
    )
    ap.add_argument(
        "--hostless",
        action="store_true",
        help="the delivery substrate has no fesvr host (enables the HTIF-hang precondition)",
    )
    ap.add_argument("--has-htif", dest="has_htif", action="store_true", help="image audit: a .htif section is present")
    ap.add_argument(
        "--has-tohost", dest="has_tohost", action="store_true", help="image audit: a tohost symbol is present"
    )
    ap.add_argument("--declared-vlen", default=None, help="VLEN the build declared (VECTOR_MAX_LEN / -march …_zvlNNNb)")
    ap.add_argument(
        "--hw-vlen", default=None, help="the target board's hardware VLEN in bits (enables the vlen-match check)"
    )
    ap.add_argument("--uses-medany", action="store_true", help="the chosen compile path builds with -mcmodel=medany")
    ap.add_argument(
        "--image-span-bytes",
        default=None,
        help="linked image symbol span in bytes (enables the medany ±2GB-window check)",
    )
    ap.add_argument("--strict", action="store_true", help="exit nonzero on STALL as well as FAULT")
    ap.add_argument("--persist", action="store_true", help="write the report under out/artifacts/liveness/<target>/")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = ap.parse_args(argv)

    trace = None
    if args.trace:
        trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    name = args.name or (Path(args.trace).parent.name if args.trace else "program")

    # host-assist audit is only meaningful when the substrate is stated.
    hostless = True if args.hostless else None
    has_htif = True if args.has_htif else (None if hostless is None else False)
    has_tohost = True if args.has_tohost else (None if hostless is None else False)

    prog = Program(
        name=name,
        trace=trace,
        address_model=args.address_model,
        dram_bytes=_int(args.dram_bytes),
        hostless=hostless,
        has_htif=has_htif,
        has_tohost=has_tohost,
        declared_vlen=_int(args.declared_vlen),
        hw_vlen=_int(args.hw_vlen),
        uses_medany=(True if args.uses_medany else None),
        image_span_bytes=_int(args.image_span_bytes),
    )
    report = assess(prog, args.target)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"merlin-liveness: {name} vs {args.target} → verdict={report.verdict.upper()}")
        if report.resource_peaks:
            print(f"  peaks: {report.resource_peaks}")
        for f in report.findings:
            where = f" @ {f.where}" if f.where else ""
            print(f"  [{f.severity.value:7s}] {f.rule}{where}: {f.message}")
        if not report.findings:
            print("  (no findings)")

    if args.persist:
        out = persist(report)
        print(f"  wrote {out}")

    if any(f.severity == Severity.FAULT for f in report.findings):
        return 2
    if args.strict and any(f.severity == Severity.STALL for f in report.findings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
