"""The single derivation seam for the liveness oracle: normalize a target's CIRCT/mlc-introspected
facts (+ its memory-map green card) into a :class:`SiliconFacts` the models consume.

FAIL-CLOSED, per the repo's cardinal rule: every field is derived from the target's OWN sources
(``rtl.facts.load_facts`` — mesh geometry, on-chip memory capacities, the legal funct/opcode set — and
``dram_facts.dram_base_for`` — the DRAM base from the shipped memory-map markdown). A fact that cannot be
derived is left ``None`` (surfaced via :meth:`SiliconFacts.unknowns`), NEVER a target-shaped default —
an assumed 16×16 mesh or 256 KB scratchpad would silently mis-model a different target.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from merlin.targetgen.rtl.facts import load_facts


@dataclasses.dataclass
class SiliconFacts:
    """Normalized, provenance-tagged silicon facts for one target. Any ``None`` = not derivable."""

    target: str
    mesh_rows: int | None = None
    mesh_cols: int | None = None
    scratchpad_bytes: int | None = None
    scratchpad_rows: int | None = None       # SRAM depth in rows (MVIN spad_addr is a row index)
    accumulator_bytes: int | None = None
    accumulator_rows: int | None = None
    accumulator_row_bytes: int | None = None  # derived acc row width (cols * acc elem bytes)
    acc_ctrl_mask: int | None = None           # derived high control bits carried in an acc address
    legal_funct: list[int] | None = None
    custom_opcode: int | None = None
    funct3: int | None = None
    dram_base: int | None = None
    provenance: str = "UNKNOWN"

    def unknowns(self) -> list[str]:
        """Names of the facts that could not be derived (each blocks the checks that need it)."""
        return [
            name
            for name in (
                "mesh_rows", "mesh_cols", "scratchpad_rows", "accumulator_rows",
                "legal_funct", "dram_base",
            )
            if getattr(self, name) is None
        ]


def _memory(f: dict, name: str) -> dict | None:
    return next((m for m in f.get("memories", []) if m.get("name") == name), None)


def _dtype_bits(dtype: str | None) -> int | None:
    """Bit width of a datapath dtype string (``i8``→8, ``i32``→32, ``bf16``→16, ``fp8``→8). Structural —
    the trailing digits are the width; no regex, no per-target literal."""
    if not isinstance(dtype, str):
        return None
    digits = "".join(c for c in dtype if c.isdigit())
    return int(digits) if digits else None


def _datapath_bits(f: dict, name: str) -> int | None:
    dp = next((d for d in f.get("datapaths", []) if d.get("name") == name), None)
    return _dtype_bits(dp.get("dtype")) if dp else None


def _rows_from_bytes(total_bytes: int | None, cols: int | None, elem_bits: int | None) -> int | None:
    """Logical addressable row count = total_bytes / row_width, row_width = ceil(cols * elem_bits / 8).
    All three inputs are derived facts; returns None if any is missing (the capacity check then skips)."""
    if not total_bytes or not cols or not elem_bits:
        return None
    row_bytes = (cols * elem_bits + 7) // 8
    return total_bytes // row_bytes if row_bytes else None


def silicon_facts(target: str) -> SiliconFacts:
    """Derive :class:`SiliconFacts` for ``target``. Never raises for a missing/degraded fact source —
    it fails closed to ``None`` fields (the consuming checks skip and surface ``UNKNOWN``)."""
    out = SiliconFacts(target=target)
    try:
        rec = load_facts(target)
    except Exception:  # noqa: BLE001 — facts not derivable → everything stays UNKNOWN
        out.provenance = "UNKNOWN (RTL facts not derivable)"
        _fill_dram_base(out)
        return out

    f = rec.get("facts", rec)
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

    ft = next((i for i in f.get("interfaces", []) if i.get("name") == "funct_decode_table"), None)
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
        from merlin.targetgen.rocc.decode import isa_constants

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
        from merlin.targetgen.dram_facts import dram_base_for

        base = dram_base_for(out.target)
        if isinstance(base, int):
            out.dram_base = base
    except Exception:  # noqa: BLE001 — no derivable memory map → dram_base stays None
        pass
