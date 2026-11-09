"""Write the seed preparation products: SEED_BASELINE.{json,md} and SEED_STRUCTURE.md."""

from __future__ import annotations

import json
from pathlib import Path

from .structure import FAMILIES, FamilyComparison


def _num(value) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _widths(value) -> str:
    if not value:
        return "-"
    if isinstance(value, dict):
        return ",".join(f"{k}:{v}" for k, v in sorted(value.items()))
    return ",".join(str(v) for v in value)


def ranked_gaps(rows: list[dict], per_family: int) -> dict[str, list[dict]]:
    """Largest absolute cycle gaps (model cycles minus roofline bound) per family."""
    out: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("headroom_cycles") is None:
            continue
        out.setdefault(row.get("family") or "other", []).append(row)
    return {fam: sorted(items, key=lambda r: -r["headroom_cycles"])[:per_family]
            for fam, items in sorted(out.items())}


def write_baseline(out: Path, payload: dict, per_family: int = 8) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "SEED_BASELINE.json").write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n",
                                            encoding="utf-8")
    rows = payload["baseline"]["cases"]
    lines = [
        "# Seed baseline performance map",
        "",
        f"Seed compiler: `{payload['seed']['path']}` (tree sha256 `{payload['seed']['sha256'][:16]}`).",
        f"Export: `{payload['export']['path']}`. Timing: {payload['baseline']['method']}.",
        "Operator-side preparation result: Spike cycle-model numbers only; no physical measurements.",
        f"Scope: {payload['baseline']['scope']}.",
        "",
        "## Largest cycle gaps per family",
        "",
    ]
    for family, items in ranked_gaps(rows, per_family).items():
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| case | model cycles | roofline | util % | headroom | bound |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for r in items:
            lines.append(f"| {r['case']} | {_num(r.get('model_cycles'))} | {_num(r.get('roofline_cycles'))} | "
                         f"{_num(r.get('utilization_pct'))} | {_num(r.get('headroom_cycles'))} | "
                         f"{r.get('bound_by') or '-'} |")
        lines.append("")
    lines += [
        "## All cases",
        "",
        "| case | family | status | model cycles | roofline | util % | headroom | fences/out tile | "
        "loads | computes | transfer widths | capacity fallback |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r.get("family") or "", r["case"])):
        lines.append(
            f"| {r['case']} | {r.get('family') or '-'} | {r['status']} | {_num(r.get('model_cycles'))} | "
            f"{_num(r.get('roofline_cycles'))} | {_num(r.get('utilization_pct'))} | "
            f"{_num(r.get('headroom_cycles'))} | {_num(r.get('fences_per_output_tile'))} | "
            f"{_num(r.get('loads'))} | {_num(r.get('computes'))} | {_widths(r.get('transfer_widths'))} | "
            f"{_num(r.get('capacity_fallback'))} |")
    lines.append("")
    promo = payload.get("promotion") or {}
    lines += ["## Variant promotion", ""]
    if not promo:
        lines.append("Not run.")
    else:
        lines.append(f"Variants examined: {promo.get('examined', 0)}; promoted tree: "
                     f"`{promo.get('promoted_tree') or 'none'}`.")
        lines.append("")
        for title, key in (("Promoted", "promoted"), ("Candidates", "candidates")):
            entries = promo.get(key) or []
            lines.append(f"### {title} ({len(entries)})")
            lines.append("")
            for e in entries:
                cases = "; ".join(f"{c['case']} {_num(c.get('before'))}->{_num(c.get('after'))}"
                                  + ("" if c.get("numerics") in (None, "pass") else f" numerics {c['numerics']}")
                                  for c in e.get("cases", []))
                lines.append(f"- **{e['name']}** ({e.get('reason', '')}): files {', '.join(e.get('files', []))}"
                             + (f"; {cases}" if cases else ""))
            lines.append("")
        if promo.get("discovered"):
            lines.append("Variant changes discovered: " + ", ".join(
                f"{k} {v}" for k, v in sorted(promo["discovered"].items())) + ".")
            lines.append("")
        if promo.get("not_examined"):
            lines.append(f"Applicable but not examined in this bounded run ({len(promo['not_examined'])}): "
                         + ", ".join(promo["not_examined"]))
            lines.append("")
        skipped = promo.get("not_applicable") or []
        if skipped:
            lines.append(f"Not applicable ({len(skipped)}): " + ", ".join(
                f"{s['name']} ({s['status']})" for s in skipped))
            lines.append("")
    (out / "SEED_BASELINE.md").write_text("\n".join(lines), encoding="utf-8")


def structure_lines(rows: list[FamilyComparison], effects: dict[str, list[str]]) -> list[str]:
    """Short actionable lines, one block per family."""
    lines: list[str] = []
    for row in rows:
        own = ", ".join(f"{name} ({n} lines)" for name, n in sorted(row.seed_author.items()))
        if row.seed_library:
            uses = "uses library " + ", ".join(m.rpartition(".")[2] for m in row.seed_library)
        else:
            uses = "uses own implementation only"
        if own:
            uses += f"; own modules {own}"
        lines.append(f"- **{row.family}**: {uses}.")
        for module, labels in sorted(row.missing_from_seed.items()):
            note = f"  - library `{module.rpartition('.')[2]}` used by {', '.join(labels)}, not by the seed"
            extra = effects.get(module)
            if extra:
                note += "; " + "; ".join(extra)
            lines.append(note + ".")
        for label, effect_lines in effects.items():
            if label.startswith(f"family:{row.family}:"):
                lines.extend(f"  - {e}" for e in effect_lines)
    return lines


def write_structure(out: Path, seed_label: str, compared: list[str], rows: list[FamilyComparison],
                    effects: dict[str, list[str]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    head = [
        "# Seed structure by family",
        "",
        f"Seed `{seed_label}` compared with {', '.join(compared) or 'no other compilers'}. Routes are the "
        "library modules each compiler's author modules import (static analysis). Cycle effects are Spike "
        "cycle-model numbers on shared cases where measured; UNKNOWN otherwise.",
        "",
    ]
    order = {f: i for i, f in enumerate(FAMILIES)}
    rows = sorted(rows, key=lambda r: order.get(r.family, len(order)))
    (out / "SEED_STRUCTURE.md").write_text("\n".join(head + structure_lines(rows, effects)) + "\n",
                                           encoding="utf-8")
