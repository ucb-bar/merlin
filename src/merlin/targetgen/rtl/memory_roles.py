"""Name scoped-census memories by the target's declared roles, keeping the physical inventory.

The census names a store after the Chisel file and declaration that emitted it (``scratchpad.mem``);
consumers address stores by the role the target declares for them (``scratchpad``,
``accumulator``). Both names describe one piece of storage, so a correctly scoped census must not
rename a role or report one store twice:

* a census row whose geometry equals a declared role-probe store (same banks, depth, row
  elements, element bits and bytes, declared at the probe's source site) takes the role's name and
  the address-space view tag, keeping its census name and evidence;
* a census row that physically backs a declared address-space view (equal depth, row bits and
  total bytes, one copy per view bank) moves to ``census.physical_memories`` with ``backs`` naming
  that view, so the role view is not double-counted;
* every other census row stays a memory under its census name.

Nothing is dropped: the complete physical inventory is kept under ``census.physical_memories``.
When the census is empty (an unscoped elaboration), the declared role-probe rows remain the
memories, exactly as before.
"""

from __future__ import annotations

from typing import Any

_GEOMETRY = ("banks", "depth", "row_elems", "elem_bits", "bytes")


def _site(row: dict) -> str | None:
    evidence = row.get("evidence")
    if not isinstance(evidence, str) or " @ " not in evidence:
        return None
    return evidence.rpartition(" @ ")[2].split()[0] if evidence.rpartition(" @ ")[2].split() else None


def label_roles(census: list[dict], probes: list[dict]) -> list[dict]:
    """Census memories with declared role-probe names applied (or the probes when no census)."""
    declared = [row for row in probes or () if row.get("bytes") is not None]
    if not census:
        return [{**row, "source": "selected_firrtl_role_probe", "view_kind": "address_space"} for row in declared]
    out = []
    for row in census:
        matches = [
            probe
            for probe in declared
            if all(row.get(key) == probe.get(key) for key in _GEOMETRY)
            and _site(probe) is not None
            and _site(probe) in (row.get("evidence") or "")
        ]
        if len(matches) == 1:
            out.append(
                {
                    **row,
                    "name": matches[0]["name"],
                    "census_name": row["name"],
                    "view_kind": "address_space",
                    "role_basis": "census geometry equals the declared role-probe store at its source site",
                }
            )
        else:
            out.append(row)
    return out


def separate_backings(memories: list[dict], views: list[dict], facts: dict[str, Any]) -> list[dict]:
    """Move census rows that physically back a declared view into ``census.physical_memories``."""
    kept, physical = [], []
    for row in memories:
        if row.get("source") != "firrtl_census":
            kept.append(row)
            continue
        physical.append(row)
        copies = row.get("copies", 1) * row.get("banks", 1)
        backs = [
            view
            for view in views
            if view.get("view_kind") == "address_space"
            and view.get("source") != "firrtl_census"
            and view.get("depth") == row.get("depth")
            and view.get("row_bytes") is not None
            and view["row_bytes"] * 8 == row.get("row_bits_rtl")
            and view.get("banks") == copies
            and view.get("bytes") == copies * (row.get("bytes") or 0)
        ]
        if len(backs) == 1:
            physical[-1] = {**row, "backs": backs[0]["name"], "backing_basis": "equal depth, row bits and total bytes"}
        else:
            kept.append(row)
    census = facts.setdefault("census", {})
    if census.get("unit_root") is not None:  # an unscoped census has no inventory to state
        census["physical_memories"] = physical
    return kept
