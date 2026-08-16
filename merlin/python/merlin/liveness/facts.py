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

    sp = _memory(f, "scratchpad")
    if sp:
        out.scratchpad_bytes = sp.get("bytes")
        # Prefer the introspected SRAM depth (row count) — spad_addr is a row index into it. Fall back
        # to bytes / row-width only when depth is absent (row width = mesh cols * 1 byte for i8 tiles).
        if isinstance(sp.get("depth"), int):
            out.scratchpad_rows = sp["depth"]
        elif out.scratchpad_bytes and out.mesh_cols:
            out.scratchpad_rows = out.scratchpad_bytes // out.mesh_cols

    acc = _memory(f, "accumulator")
    if acc:
        out.accumulator_bytes = acc.get("bytes")
        if isinstance(acc.get("depth"), int):
            out.accumulator_rows = acc["depth"]

    ft = next((i for i in f.get("interfaces", []) if i.get("name") == "funct_decode_table"), None)
    if ft:
        out.legal_funct = ft.get("legal_funct")
        out.custom_opcode = ft.get("custom_opcode")
        out.funct3 = ft.get("funct3")

    gen = rec.get("generator", {}) if isinstance(rec, dict) else {}
    out.provenance = f"circt_introspect facts.json ({gen.get('version', '?')})"
    _fill_dram_base(out)
    return out


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
