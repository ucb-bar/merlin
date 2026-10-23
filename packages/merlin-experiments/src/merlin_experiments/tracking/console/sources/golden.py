"""Operator-only comparison of whole-model results against the reference ledger, refusing what is not comparable.

References come from ``merlin/contract/perf_reference_targets.yaml`` through
:mod:`merlin.perf.target_reference`; comparability is :func:`merlin.perf.program_identity.comparability`.
A candidate here is a whole-model capsule of the run. Its cycles are a Spike-model estimate or a gSIM
count, never a measurement on a registered device, and it states no program-identity facts -- so a
comparison is shown side by side with the reason it is refused, and a ratio exists only when both the
program identity and the device agree.
"""

from __future__ import annotations

from typing import Any

from ..io import clean, obj, positive

LEDGER = "perf_reference_targets.yaml"


def _ledger() -> dict[str, Any]:
    from merlin.perf.target_reference import load_ledger

    return dict(load_ledger())


def references(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for name, entry in obj(ledger.get("references")).items():
        entry = obj(entry)
        out.append(
            {
                "name": clean(name, 80),
                "model": clean(entry.get("model"), 40) or None,
                "program_identity": clean(entry.get("program_identity"), 60) or None,
                "device": clean(entry.get("device"), 60) or None,
                "status": clean(entry.get("status"), 24) or None,
                "cycles": positive(obj(entry.get("measured")).get("whole_model_cycles")),
                "source": f"{LEDGER}#{name}",
            }
        )
    return sorted(out, key=lambda r: (r["model"] or "~", r["cycles"] or float("inf")))


def _candidates(ctx) -> list[dict[str, Any]]:
    catalog = [c for c in obj(ctx.shared.get("capsules")).get("capsules") or [] if isinstance(c, dict)]
    perf = {r["capsule"]: r for r in obj(ctx.shared.get("perf")).get("rows") or [] if isinstance(r, dict)}
    out = []
    for row in catalog:
        origin = obj(row.get("origin"))
        if row.get("kind") != "model" and origin.get("full_model") is not True:
            continue
        measured = perf.get(row["name"]) or {}
        cycles, kind = measured.get("gsim_cycles"), "gSIM RTL-simulation count"
        if cycles is None:
            cycles, kind = measured.get("spike_model_cycles"), "Spike timing-model estimate"
        out.append({"subject": row["name"], "model": origin.get("model"), "cycles": cycles, "kind": kind})
    return out


def read(ctx) -> dict[str, Any]:
    if not ctx.operator_private:
        return {"references": [], "comparisons": [], "notes": ["golden references are operator-only"]}
    from merlin.perf.program_identity import comparability, declared_identities, identity_of_reference, resolve

    ledger = _ledger()
    refs = references(ledger)
    roster = declared_identities(ledger)
    by_name = obj(ledger.get("references"))
    comparisons = []
    for candidate in _candidates(ctx):
        # The run records no program-identity facts for its candidates: resolution says so itself.
        identity = resolve({}, roster)
        matched = [r for r in refs if candidate["model"] and r["model"] == candidate["model"] and r["cycles"]]
        if not matched:
            comparisons.append(
                {
                    "subject": candidate["subject"],
                    "candidate_cycles": candidate["cycles"],
                    "reference": None,
                    "reference_cycles": None,
                    "ratio": None,
                    "comparable": False,
                    "reason": "the candidate's source model is not recorded"
                    if not candidate["model"]
                    else f"the ledger holds no measured {candidate['model']} reference",
                }
            )
            continue
        for ref in matched:
            verdict = comparability(identity, identity_of_reference(obj(by_name.get(ref["name"]))))
            reasons = [] if verdict.ok else [str(verdict.get("reason"))]
            reasons.append(f"candidate cycles are a {candidate['kind']}, not a measurement on {ref['device']}")
            comparisons.append(
                {
                    "subject": candidate["subject"],
                    "candidate_cycles": candidate["cycles"],
                    "reference": ref["name"],
                    "reference_cycles": ref["cycles"],
                    "ratio": None,
                    "comparable": False,
                    "reason": clean("; ".join(reasons), 400),
                }
            )
    return {
        "references": refs,
        "comparisons": comparisons,
        "notes": [f"from merlin/contract/{LEDGER}; a ratio is shown only when program identity and device agree"],
    }


def inspect(ctx, key: str) -> dict[str, Any]:
    if not ctx.operator_private:
        return {
            "title": "golden reference",
            "sections": [],
            "paths": [],
            "notice": "golden references are operator-only",
        }
    entry = obj(obj(_ledger().get("references")).get(key))
    if not entry:
        return {"title": f"reference {clean(key, 80)}", "sections": [], "paths": [], "notice": "not in the ledger"}
    lines = [f"{k}: {clean(v, 300)}" for k, v in entry.items() if not isinstance(v, dict | list)]
    sections = [{"title": "Entry", "lines": lines}]
    for block in ("measured", "provenance"):
        if isinstance(entry.get(block), dict):
            sections.append(
                {"title": block.capitalize(), "lines": [f"{k}: {clean(v, 300)}" for k, v in entry[block].items()]}
            )
    caveats = entry.get("caveats")
    if isinstance(caveats, list):
        sections.append({"title": "Caveats", "lines": [clean(c, 600) for c in caveats]})
    return {"title": f"reference {key}", "sections": sections, "paths": [f"merlin/contract/{LEDGER}"]}


__all__ = ["inspect", "read", "references"]
