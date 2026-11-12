"""Structural memory geometry from HW and FIRRTL module ports.

These readers only interpret the selected text supplied by their caller. Source
selection and provenance remain with the CIRCT introspection orchestrator.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from merlin.development.targetgen.rtl.ports import banked_store_ports


def _int_width(typ: str) -> int | None:
    """Bit width of an ``iN`` HW/MLIR integer type token (``i9`` -> 9), else None."""
    typ = typ.strip()
    return int(typ[1:]) if typ.startswith("i") and typ[1:].isdigit() else None


def _paren_span(line: str, open_idx: int) -> int:
    """Index of the ``)`` that closes the ``(`` at ``open_idx`` (balanced), or -1."""
    depth = 0
    for j in range(open_idx, len(line)):
        if line[j] == "(":
            depth += 1
        elif line[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    return -1


def _module_port_sig(hw_text: str, module: str) -> str | None:
    """The port-list text of ``hw.module ... @<module>( ... )`` — the ports between the balanced
    parens after the module name. None if the module is absent."""
    marker = f"@{module}("
    for line in hw_text.splitlines():
        if "hw.module" not in line or marker not in line:
            continue
        open_idx = line.find(marker) + len(marker) - 1  # the '(' itself
        close = _paren_span(line, open_idx)
        if close != -1:
            return line[open_idx + 1 : close]
    return None


def extract_accumulator(hw_text: str, *, layout: dict[str, str]) -> dict[str, Any] | None:
    """Accumulator capacity from a declared HW-module port layout, scaled by bank count.

    Per bank: depth = 2**addr_width (the declared address port); a row is ``lanes`` words of the
    port-derived bit width; the byte mask confirms bytes/row. ``banks`` counts module instantiations.
    Total bytes = banks * per-bank.

    The signature is read *structurally* — the module's port list is enumerated and each ``name : type``
    port is matched by exact identity and its integer type width parsed — not pattern-matched, so a
    port rename or reorder cannot silently mis-derive a capacity."""
    module = layout["module"]
    sig = _module_port_sig(hw_text, module)
    if sig is None:
        return None
    addr_w = None
    lane_bits_seen: list[int] = []
    mask_bits = 0
    for decl in (d.strip() for d in sig.split(",")):  # ports have no nested parens in their types
        if " : " not in decl:
            continue
        lhs, typ = decl.rsplit(" : ", 1)
        name = lhs.split()[-1].lstrip("%")  # drop the `in`/`out` dir + `%`
        if name == layout["address_port"]:
            addr_w = _int_width(typ)
        elif name.startswith(layout["data_prefix"]) and name.endswith(layout["data_suffix"]):
            w = _int_width(typ)
            if w is not None:
                lane_bits_seen.append(w)
        elif name.startswith(layout["mask_prefix"]):
            mask_bits += 1
    if addr_w is None or not lane_bits_seen or len(set(lane_bits_seen)) != 1:
        return None
    depth = 1 << addr_w
    lane_bits = lane_bits_seen[0]
    n_lanes = len(lane_bits_seen)
    row_bytes = mask_bits or (n_lanes * (lane_bits // 8))
    banks = sum(1 for ln in hw_text.splitlines() if "hw.instance" in ln and f"@{module}" in ln) or 1
    per_bank = depth * row_bytes
    return {
        "name": layout["memory_name"],
        "banks": banks,
        "addr_width": addr_w,
        "depth": depth,
        "lanes": n_lanes,
        "lane_bits": lane_bits,
        "row_bytes": row_bytes,
        "bytes": banks * per_bank,
        "bytes_per_bank": per_bank,
        "elem_bits": lane_bits,
        "evidence": (
            f"{banks}x @{module} instance; per-bank {layout['address_port']}:i{addr_w} "
            f"(depth={depth}); {n_lanes}x {layout['data_prefix']}*{layout['data_suffix']}:i{lane_bits}; "
            f"{mask_bits} byte-mask bits -> {row_bytes} B/row; total={banks * per_bank} B"
        ),
    }


def memories_from_port_geometry(fir_paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    """Memory facts derived from BANKED WRITE PORTS, in the same shape a census emits.

    ⚠️ THIS IS THE CENSUS'S BLIND SPOT, not a second opinion about the same memory. A ``cmem``/``smem``
    census can only see an SRAM DECLARED inside a module the elaboration contains; a store instantiated
    at a tile level a standalone repo cannot elaborate produces NO memories at all, and every
    memory-regime question about that target then reports ``0 / 0`` cells -- unanswerable, while the
    module below the store declares its geometry on its own ports in full.

    ``source`` is ``firrtl_port_geometry`` so a reader can tell these apart from ``firrtl_census``
    entries: the two are read from different evidence and a port-derived row is a WRITE-GRANULARITY
    decomposition (the byte enable), not the SRAM's declared element type.

    Ports that agree on geometry are folded into ONE store, because the same store seen from two ports
    is far likelier than two distinct stores that happen to match -- and the other reading would publish
    twice the capacity the device has, which is the direction in which a schedule silently overcommits.
    """
    merged: dict[tuple, dict[str, Any]] = {}
    for path in fir_paths:
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # an unreadable elaboration contributes nothing; it lies about nothing
        for rec in banked_store_ports(text):
            key = (rec["row_bits"], rec["row_addr_bits"], rec["bank_id_bits"], rec["banks"])
            seen = merged.setdefault(key, {"records": [], "files": []})
            seen["records"].append(rec)
            if p.name not in seen["files"]:
                seen["files"].append(p.name)
    out: list[dict[str, Any]] = []
    for _key, seen in sorted(merged.items(), key=lambda kv: -(kv[0][0] or 0)):
        recs = sorted(seen["records"], key=lambda r: (r["module"], r["field"]))
        rec = recs[0]
        sites = [f"{r['module']}.{r['port']}.{r['field']}" for r in recs]
        mem: dict[str, Any] = {
            "name": f"{rec['module']}.{rec['field']}".casefold(),
            "banks": rec["banks"],
            "depth": rec["rows_per_bank"],
            # The byte enable proves the row is written in BYTE LANES. That is the store's write
            # granularity and it is stated as such; it is NOT a claim about the datapath element the
            # values in it belong to, which a port does not declare (hence no `datapaths` entry).
            "row_elems": rec["row_bytes"],
            "elem_bits": 8,
            "row_bits_rtl": rec["row_bits"],
            "bytes": rec["bytes"],
            "source": "firrtl_port_geometry",
            "modules": sorted({r["module"] for r in recs}),
            "ports": sites,
            "row_element_note": (
                "row_elems x elem_bits is the WRITE GRANULARITY the byte enable "
                "declares (one lane per byte of the row), not a datapath element "
                "width -- a port does not declare what type the stored value has"
            ),
            "evidence": f"{rec['evidence']} [{', '.join(seen['files'])}]",
        }
        if not rec["banks_exact"]:
            # FAIL CLOSED. The geometry bounds the bank count and the bound is what gets published; a
            # capacity is withheld rather than computed from a guessed bank count, because a capacity
            # that is wrong by the bank factor reads as a fit and aborts three layers away.
            mem["banks_min"], mem["banks_max"] = rec["banks_min"], rec["banks_max"]
            mem["banks_unknown"] = rec.get("banks_unknown", "the bank count is bounded, not pinned")
            mem["bytes_unknown"] = (
                "the bank count is not pinned by this elaboration, so a byte "
                "capacity would be a guess multiplied by the bank factor"
            )
        if len(sites) > 1:
            mem["ports_note"] = (
                f"{len(sites)} ports declare this geometry ({sites}); folded into one "
                "store, since two distinct stores of identical geometry would publish "
                "twice the capacity the device has"
            )
        out.append(mem)
    return out
