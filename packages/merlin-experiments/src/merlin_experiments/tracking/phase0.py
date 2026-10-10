"""Read a Phase 0 derivation or generation run into the corpus-and-coverage view's summary.

A Phase 0 directory is one (or a mix) of:

* a DERIVATION root (``merlin-experiment corpus derive``): ``requirements.yaml``, ``synthesis.yaml``,
  ``derivation.json`` -- what the corpus must cover and the capsules planned for it;
* a GENERATION run: ``capsules/MANIFEST.yaml`` and ``capsules/<category>/<name>/capsule.yaml`` (the
  written corpus and its cohorts) and ``coverage/{generation,phase1-capsule-coverage,
  phase2-capsule-coverage}.json`` (what the written cohorts were measured to cover);
* a bare corpus: ``<category>/<name>/capsule.yaml``.

Every one of these is found by its file names (in the directory, its ``phase0/`` child, or one level
down), read as written, and summarized: nothing is re-derived, re-generated or re-measured.  The one
classification this module applies itself is the shape stratum of a capsule that records ``M/K/N`` but
no stratum, and it uses the owner's classifier (:func:`merlin.capture.shape_taxonomy.classify_geometry`)
and labels the value as classified.

HIDDEN MATERIAL IS COUNTED, NOT NAMED.  Capsules labelled ``hidden`` (or under ``hidden/``), the
held-out layer guard's digest, and the revealed members of a holdout are reduced to counts unless the
caller passes ``operator_private=True``; the summary records which mode it was built in.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import records as R

SCHEMA = "merlin_experiment_tracking_phase0_v1"
HIDDEN = "hidden"
#: Requirement axes drawn as required-vs-covered tables: (axis, where the rows are, the key field).
AXES = (
    ("cells", ("cells",), "cell"),
    ("shape_geometry", ("shape_geometry", "required"), "class"),
    ("conv_geometry", ("conv_geometry", "required"), "signature"),
    ("epilogue", ("epilogue", "required"), "stage"),
    ("groups", ("groups", "required"), "signature"),
    ("carried_state", ("carried_state", "required"), "stage"),
    ("scope", ("scope", "required"), "signature"),
)
_TIERS_ORDER = R.tier_key


def _at(document: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        document = document.get(key) if isinstance(document, Mapping) else None
    return document


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# --------------------------------------------------------------------------- discovery
def _candidates(root: Path) -> list[Path]:
    root = Path(root)
    out = [root]
    for base in (root, root / "phase0"):
        if base.is_dir():
            out.append(base)
            out += sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))
    seen, unique = set(), []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def discover(root: Path) -> dict[str, Any]:
    """Where the derivations, the written corpus and the coverage outputs are under ``root``."""
    places = _candidates(root)
    derivations = [p for p in places if (p / "requirements.yaml").is_file()]
    corpus = next((p / "capsules" for p in places if (p / "capsules" / "MANIFEST.yaml").is_file()), None)
    if corpus is None:
        corpus = next((p for p in places if (p / "MANIFEST.yaml").is_file()), None)
    if corpus is None:
        corpus = next((p for p in places if any(p.glob("*/*/capsule.yaml"))), None)
    coverage = next((p / "coverage" for p in places if (p / "coverage" / "generation.json").is_file()), None)
    if coverage is None:
        coverage = next((p / "coverage" for p in places if (p / "coverage").is_dir()), None)
    return {"derivations": derivations, "corpus": corpus, "coverage": coverage}


# --------------------------------------------------------------------------- capsules
def _epilogue(value: Any) -> str:
    if isinstance(value, list | tuple):
        return "+".join(str(v) for v in value) if value else "none"
    return str(value) if value else "none"


def _dtype(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    return {"int8": "i8", "int32": "i32", "float32": "f32", "fp32": "f32"}.get(text, text)


def _window(a: Mapping[str, Any]) -> str | None:
    kh, kw = a.get("kh"), a.get("kw")
    if kh is None or kw is None:
        return None

    def pair(value: Any, default: int) -> str:
        if isinstance(value, list | tuple) and value:
            return "x".join(str(v) for v in value[:2]) if len(value) >= 2 else f"{value[0]}x{value[0]}"
        return f"{value if value is not None else default}x{value if value is not None else default}"

    pad = a.get("padding")
    pad_text = pair(pad[:2] if isinstance(pad, list) and len(pad) == 4 else pad, 0)
    return f"k{kh}x{kw}/s{pair(a.get('stride'), 1)}/d{pair(a.get('dilation'), 1)}/pad{pad_text}"


def _stratum(m: Any, k: Any, n: Any) -> str | None:
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (m, k, n)):
        return None
    from merlin.capture.shape_taxonomy import classify_geometry

    return classify_geometry(m, n, k)


def _family_of(op: Any, recorded: Any) -> str | None:
    if recorded:
        return str(recorded)
    if not op:
        return None
    from merlin.targetgen import semantic_families

    return semantic_families.from_op(str(op))


def written_row(path: Path, cap: Mapping[str, Any]) -> dict[str, Any]:
    """One written capsule (``capsule.yaml``) as an inventory row."""
    operation = cap.get("operation") if isinstance(cap.get("operation"), Mapping) else {}
    attributes = operation.get("attributes") if isinstance(operation.get("attributes"), Mapping) else {}
    performance = cap.get("performance") if isinstance(cap.get("performance"), Mapping) else {}
    form = performance.get("form") if isinstance(performance.get("form"), Mapping) else {}
    semantic = cap.get("semantic") if isinstance(cap.get("semantic"), Mapping) else {}
    inputs = [i for i in cap.get("inputs") or () if isinstance(i, Mapping)]
    tiers = [str(t) for t in cap.get("required_oracle_tiers") or ()]
    tier = cap.get("max_oracle_tier") or (max(tiers, key=_TIERS_ORDER) if tiers else None)
    m, k, n = (attributes.get(x) for x in ("M", "K", "N"))
    if m is None and len(inputs) >= 2:
        lhs = next((i for i in inputs if i.get("role") == "input"), None)
        weight = next((i for i in inputs if i.get("role") == "weight"), None)
        if lhs and weight and len(lhs.get("shape") or ()) == 2 and len(weight.get("shape") or ()) == 2:
            m, k = lhs["shape"]
            n = weight["shape"][1]
    geometry = form.get("geometry") or performance.get("shape_geometry")
    if isinstance(geometry, Mapping):
        geometry = geometry.get("class") or geometry.get("stratum")
    classified = None if geometry else _stratum(m, k, n)
    return {
        "name": str(cap.get("name") or path.parent.name),
        "category": path.parent.parent.name,
        "label": cap.get("label"),
        "source": "written",
        "kind": cap.get("kind"),
        "family": _family_of(operation.get("op"), semantic.get("semantic_family")),
        "op": operation.get("op"),
        "form": form.get("label"),
        "stratum": geometry or classified,
        "stratum_classified": classified is not None,
        "epilogue": _epilogue(attributes.get("epilogue")),
        "dtype": _dtype((cap.get("numeric_policy") or {}).get("dtype") or (inputs[0].get("dtype") if inputs else None)),
        "tier": tier,
        "perf_family": performance.get("family"),
        "axis": semantic.get("generalization_axis"),
        "window": _window(attributes),
        "M": _int(m),
        "K": _int(k),
        "N": _int(n),
    }


def planned_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    """One planned capsule (``synthesis.yaml`` ``capsules[]``) as an inventory row."""
    model_form = entry.get("model_form") if isinstance(entry.get("model_form"), Mapping) else {}
    form = model_form.get("form") if isinstance(model_form.get("form"), Mapping) else {}
    generalization = entry.get("generalization") if isinstance(entry.get("generalization"), Mapping) else {}
    form_label = None
    if form:
        parts = [form.get("op"), _epilogue(form.get("epilogue")), form.get("scale_class"), f"bias:{form.get('bias')}"]
        form_label = "/".join(str(p) for p in parts if p)
    m, k, n = entry.get("M"), entry.get("K"), entry.get("N")
    classified = _stratum(m, k, n)
    return {
        "name": str(entry.get("name") or "?"),
        "category": entry.get("cat"),
        "label": entry.get("label"),
        "source": "planned",
        "kind": entry.get("kind"),
        "family": _family_of(entry.get("op"), None),
        "op": entry.get("op"),
        "form": form_label,
        "stratum": classified,
        "stratum_classified": classified is not None,
        "epilogue": _epilogue(entry.get("epilogue")),
        "dtype": _dtype(entry.get("operand_dtype")),
        "tier": None,
        "perf_family": None,
        "axis": generalization.get("generalization_axis"),
        "window": _window(entry),
        "M": _int(m),
        "K": _int(k),
        "N": _int(n),
        "model": entry.get("model"),
    }


def is_hidden(row: Mapping[str, Any]) -> bool:
    return row.get("label") == HIDDEN or row.get("category") == HIDDEN


INVENTORY_AXES = ("family", "op", "form", "stratum", "epilogue", "dtype", "tier", "category", "label", "perf_family")


def tally(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts per inventory axis, the family -> op tree, conv windows and shapes."""
    by = {axis: Counter(str(r.get(axis) or "not recorded") for r in rows) for axis in INVENTORY_AXES}
    tree: dict[str, dict[str, int]] = {}
    for r in rows:
        family, op = str(r.get("family") or "not recorded"), str(r.get("op") or "not recorded")
        tree.setdefault(family, {})
        tree[family][op] = tree[family].get(op, 0) + 1
    return {
        "n": len(rows),
        "by": {axis: dict(counter.most_common()) for axis, counter in by.items()},
        "tree": tree,
        "windows": dict(Counter(r["window"] for r in rows if r.get("window")).most_common()),
        "shapes": [
            {"name": r["name"], "M": r["M"], "K": r["K"], "N": r["N"], "category": r.get("category")}
            for r in rows
            if r.get("M") and r.get("K") and r.get("N")
        ],
    }


def read_corpus(root: Path | None, inventory: R.Inventory, *, operator_private: bool) -> dict[str, Any] | None:
    """The written corpus: its capsules (public tallied, hidden counted) and its MANIFEST cohorts."""
    if root is None:
        return None
    root = Path(root)
    manifest = inventory.yaml("capsules/MANIFEST.yaml", root / "MANIFEST.yaml")
    rows, unreadable = [], 0
    from merlin.common.yaml import safe_load_text

    for path in sorted(root.glob("*/*/capsule.yaml")):
        try:
            cap = safe_load_text(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- an unreadable capsule is counted, never fatal
            unreadable += 1
            continue
        if isinstance(cap, Mapping):
            rows.append(written_row(path, cap))
        else:
            unreadable += 1
    inventory.note(
        "capsules/<category>/<name>/capsule.yaml",
        root,
        "read" if rows else "absent",
        f"{len(rows)} capsules" + (f", {unreadable} unreadable" if unreadable else ""),
    )
    public = [r for r in rows if not is_hidden(r)]
    hidden = [r for r in rows if is_hidden(r)]
    return {
        "root": str(root),
        "manifest": _manifest(manifest, operator_private=operator_private),
        "public": tally(public),
        "rows": public,
        "hidden": {
            "n": len(hidden),
            "by_family": dict(Counter(str(r.get("family") or "not recorded") for r in hidden).most_common()),
            "tally": tally(hidden) if operator_private else None,
            "rows": hidden if operator_private else None,
        },
        "unreadable": unreadable,
    }


def _manifest(doc: Mapping[str, Any] | None, *, operator_private: bool) -> dict[str, Any] | None:
    if doc is None:
        return None
    corpora = doc.get("phase_corpora") if isinstance(doc.get("phase_corpora"), Mapping) else {}
    cohorts: dict[str, dict[str, Any]] = {}
    for target, entry in corpora.items():
        if not isinstance(entry, Mapping):
            continue
        for phase in ("phase1", "phase2", "diagnostic"):
            cohort = entry.get(phase) if isinstance(entry.get(phase), Mapping) else None
            if cohort is None:
                continue
            members = [str(m) for m in cohort.get("generated_members") or ()]
            cohorts[f"{phase}" if len(corpora) == 1 else f"{target}:{phase}"] = {
                "purpose": cohort.get("purpose"),
                "members": members,
                "by_category": dict(Counter(m.partition("/")[0] for m in members).most_common()),
                "selection_sha256": cohort.get("selection_sha256"),
            }
    perf: dict[str, Any] = {}
    for target, entry in (doc.get("performance_generation") or {}).items():
        if isinstance(entry, Mapping):
            counts = entry.get("counts") if isinstance(entry.get("counts"), Mapping) else {}
            perf[str(target)] = {
                "families": [f.get("family") for f in entry.get("families") or () if isinstance(f, Mapping)],
                "counts": {k: v for k, v in counts.items() if k != "by_family"},
                "by_family": counts.get("by_family") or {},
                "skipped": len(entry.get("skipped_inapplicable") or ()),
                "blocked": len(entry.get("blocked_unimplemented") or ()),
                "errors": len(entry.get("errors") or ()),
            }
    held = doc.get("held_out") if isinstance(doc.get("held_out"), Mapping) else None
    return {
        "generated": len(doc.get("generated") or ()),
        "hand_authored": len(doc.get("hand_authored") or ()),
        "refused_generated": len(doc.get("refused_generated") or ()),
        "held_out": {k: v for k, v in (held or {}).items() if isinstance(v, int)} if held else None,
        "cohorts": cohorts,
        "performance": perf,
        "evidence_status": (doc.get("phase0_evidence") or {}).get("status")
        if isinstance(doc.get("phase0_evidence"), Mapping)
        else None,
    }


# --------------------------------------------------------------------------- requirements
def _axis_rows(requirements: Mapping[str, Any], path: tuple[str, ...], key: str) -> list[dict[str, Any]] | None:
    rows = _at(requirements, path)
    if rows is None:
        return None
    out = []
    for row in rows if isinstance(rows, list) else ():
        if not isinstance(row, Mapping) or row.get(key) is None:
            continue
        weight = row.get("n_regions") or row.get("occurrences") or row.get("groups")
        detail = {
            k: row.get(k)
            for k in ("family", "dtype", "M", "K", "N", "mac_fraction", "observed_in", "sources", "evidenced_by")
            if row.get(k) not in (None, [], {})
        }
        out.append({"key": str(row[key]), "weight": _int(weight), "detail": detail})
    return out


def read_requirements(path: Path, inventory: R.Inventory, *, operator_private: bool) -> dict[str, Any] | None:
    doc = inventory.yaml(f"{path.parent.name}/requirements.yaml", path)
    if doc is None:
        return None
    scope = doc.get("scope") if isinstance(doc.get("scope"), Mapping) else {}
    typed = scope.get("typed_required_instances") if isinstance(scope.get("typed_required_instances"), Mapping) else {}
    instances = [i for i in typed.get("instances") or () if isinstance(i, Mapping)]
    matrix: dict[str, dict[str, int]] = {}
    families: dict[str, dict[str, int]] = {}
    for inst in instances:
        app, sig = str(inst.get("application") or "?"), str(inst.get("signature") or "?")
        matrix.setdefault(app, {})
        matrix[app][sig] = matrix[app].get(sig, 0) + 1
        for region in inst.get("regions") or ():
            if isinstance(region, Mapping):
                fam = str(region.get("semantic_family") or "?")
                families.setdefault(app, {})
                families[app][fam] = families[app].get(fam, 0) + 1
    performance = scope.get("performance") if isinstance(scope.get("performance"), Mapping) else {}
    forms = performance.get("forms") if isinstance(performance.get("forms"), Mapping) else {}
    classes = [
        {
            "class_id": c.get("class_id"),
            "label": c.get("label"),
            "placement": (c.get("key") or {}).get("placement") if isinstance(c.get("key"), Mapping) else None,
            "max_share": c.get("max_share"),
            "members": len(c.get("members") or ()),
            "occurrences": c.get("occurrences"),
            "share_by_application": dict(c.get("share_by_application") or {}),
            "unpriced": list(c.get("unpriced_applications") or ()),
        }
        for c in forms.get("classes") or ()
        if isinstance(c, Mapping)
    ]
    guard = doc.get("heldout_layer_guard") if isinstance(doc.get("heldout_layer_guard"), Mapping) else None
    if guard is not None and not operator_private:
        guard = {k: v for k, v in guard.items() if not str(k).endswith("sha256")}
    demands = doc.get("application_demands") if isinstance(doc.get("application_demands"), Mapping) else {}
    return {
        "target": doc.get("target"),
        "tile_edge": _at(doc, ("boundaries", "tile_edge")),
        "axes": {axis: _axis_rows(doc, where, key) for axis, where, key in AXES},
        "typed": {
            "status": typed.get("status"),
            "n": len(instances),
            "by_application_signature": matrix,
            "by_application_family": families,
        }
        if typed
        else None,
        "performance": {
            "status": performance.get("status"),
            "required": len(performance.get("required") or ()),
            "excluded": len(performance.get("excluded") or ()),
            "unresolved": len(performance.get("unresolved") or ()),
            "classes": classes,
            "roster": list(forms.get("performance_scale_roster") or ())
            if isinstance(forms.get("performance_scale_roster"), list)
            else forms.get("performance_scale_roster"),
        }
        if performance
        else None,
        "heldout_guard": guard,
        "families_observed": dict(_at(doc, ("diagnostics", "families_observed")) or {}),
        "demands": {k: demands.get(k) for k in ("n_operations", "n_signatures") if demands.get(k) is not None},
        "host_lane": [
            {"family": r.get("family"), "dtype": r.get("dtype"), "n_regions": r.get("n_regions")}
            for r in _at(doc, ("host_lane", "required")) or ()
            if isinstance(r, Mapping)
        ],
        "conv_windows": [
            {"signature": r.get("signature"), "n_regions": _int(r.get("n_regions")), "sources": r.get("sources")}
            for r in _at(doc, ("conv_geometry", "required")) or ()
            if isinstance(r, Mapping)
        ],
    }


def read_derivation(root: Path, inventory: R.Inventory, *, operator_private: bool) -> dict[str, Any]:
    root = Path(root)
    derivation = inventory.json(f"{root.name}/derivation.json", root / "derivation.json")
    synthesis = inventory.yaml(f"{root.name}/synthesis.yaml", root / "synthesis.yaml")
    planned = [planned_row(c) for c in (synthesis or {}).get("capsules") or () if isinstance(c, Mapping)]
    public = [r for r in planned if not is_hidden(r)]
    hidden = [r for r in planned if is_hidden(r)]
    return {
        "name": root.name,
        "path": str(root),
        "status": (derivation or {}).get("status"),
        "qualification": (derivation or {}).get("qualification"),
        "blockers": list((derivation or {}).get("blockers") or ()),
        "requirements": read_requirements(root / "requirements.yaml", inventory, operator_private=operator_private),
        "planned": tally(public) if synthesis is not None else None,
        "planned_rows": public,
        "planned_hidden": len(hidden),
    }


# --------------------------------------------------------------------------- coverage
def _gap(axis: Any) -> dict[str, Any] | None:
    if not isinstance(axis, Mapping):
        return None
    return {
        "status": axis.get("status"),
        "reason": axis.get("reason"),
        "n_required": axis.get("n_required"),
        "n_covered": axis.get("n_covered"),
        "uncovered": [str(u) for u in axis.get("uncovered") or ()],
        "covered_by": {str(k): len(v) for k, v in (axis.get("covered_by") or {}).items() if isinstance(v, list)},
    }


def _cohort(doc: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if doc is None:
        return None
    conformance = doc.get("conformance") if isinstance(doc.get("conformance"), Mapping) else {}
    cohort = doc.get("cohort") if isinstance(doc.get("cohort"), Mapping) else {}
    applications = {}
    for app, entry in (doc.get("applications") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        ops = [o for o in entry.get("operations") or () if isinstance(o, Mapping)]
        applications[str(app)] = {
            "covered": sum(1 for o in ops if o.get("status") == "covered"),
            "missing": sum(1 for o in ops if o.get("status") == "missing"),
            "operations": len(ops),
            "by_role": dict(Counter(f"{o.get('role')}:{o.get('status')}" for o in ops)),
        }
    form = doc.get("form_perf_coverage") if isinstance(doc.get("form_perf_coverage"), Mapping) else None
    return {
        "status": doc.get("status"),
        "n_capsules": cohort.get("n_capsules"),
        "cells": {
            "n_required": conformance.get("n_required"),
            "n_covered": conformance.get("n_covered"),
            "uncovered": list(conformance.get("uncovered") or ()),
            "covered": list(conformance.get("corpus_cells") or ()),
        }
        if conformance
        else None,
        "axes": {axis: _gap(conformance.get(axis)) for axis, _, _ in AXES if axis != "cells"},
        "applications": applications,
        "blockers": [
            {k: b.get(k) for k in ("component", "application", "obligation", "reason")}
            for b in doc.get("blockers") or ()
            if isinstance(b, Mapping)
        ],
        "form": {
            "status": form.get("status"),
            "threshold": (form.get("threshold") or {}).get("min_predicted_cycle_share")
            if isinstance(form.get("threshold"), Mapping)
            else None,
            "missing": list(form.get("missing") or ()),
            "windows_without_form": list(form.get("source_windows_without_form") or ()),
            "classes": [
                {
                    k: c.get(k)
                    for k in ("class_id", "label", "max_share", "required", "required_reason", "status", "capsules")
                }
                | {"strata": c.get("geometry_strata")}
                for c in form.get("classes") or ()
                if isinstance(c, Mapping)
            ],
        }
        if form
        else None,
    }


def read_coverage(root: Path | None, inventory: R.Inventory, *, operator_private: bool) -> dict[str, Any] | None:
    if root is None:
        return None
    root = Path(root)
    generation = inventory.json("coverage/generation.json", root / "generation.json")
    phase1 = inventory.json("coverage/phase1-capsule-coverage.json", root / "phase1-capsule-coverage.json")
    phase2 = inventory.json("coverage/phase2-capsule-coverage.json", root / "phase2-capsule-coverage.json")
    gen = None
    if generation is not None:
        disjoint = (
            generation.get("hidden_disjointness") if isinstance(generation.get("hidden_disjointness"), Mapping) else {}
        )
        overlapping = disjoint.get("overlapping_hidden_capsules") or []
        gen = {
            "status": generation.get("evidence_status"),
            "mode": generation.get("mode"),
            "capsules_written": _int(generation.get("capsules_written")),
            "omitted": len(generation.get("omitted") or ()),
            "omitted_rows": [
                {k: o.get(k) for k in ("capsule", "status", "reason")}
                for o in generation.get("omitted") or ()
                if isinstance(o, Mapping) and not str(o.get("capsule") or "").startswith(HIDDEN + "/")
            ],
            "unbuilt_roster": len(generation.get("unbuilt_roster") or ()),
            "failures": [
                {k: f.get(k) for k in ("capsule", "reason")}
                for f in generation.get("failures") or ()
                if isinstance(f, Mapping) and (operator_private or not str(f.get("capsule") or "").startswith(HIDDEN))
            ],
            "performance_errors": generation.get("performance_materialization_errors"),
            "qualification": generation.get("qualification"),
            "cohort_coverage": {
                str(k): {"status": v.get("status"), "n_capsules": v.get("n_capsules")}
                for k, v in (generation.get("cohort_coverage") or {}).items()
                if isinstance(v, Mapping)
            },
            "hidden_disjointness": {
                "status": disjoint.get("status"),
                "hidden_capsules": len(disjoint.get("hidden_capsules") or ())
                if isinstance(disjoint.get("hidden_capsules"), list)
                else disjoint.get("hidden_capsules"),
                "overlapping": overlapping if operator_private else len(overlapping),
            }
            if disjoint
            else None,
        }
    return {"root": str(root), "generation": gen, "phase1": _cohort(phase1), "phase2": _cohort(phase2)}


# --------------------------------------------------------------------------- the summary
def summary(root: Path, *, operator_private: bool = False, now: float | None = None) -> dict[str, Any]:
    """The Phase 0 view of ``root`` (a derivation root, a generation run, or a corpus)."""
    import time

    root = Path(root).expanduser()
    if not root.is_dir():
        from ..spec import SpecError

        raise SpecError(f"not a Phase 0 directory: {root}")
    inventory = R.Inventory()
    found = discover(root)
    derivations = [read_derivation(p, inventory, operator_private=operator_private) for p in found["derivations"]]
    if not found["derivations"]:
        inventory.note("requirements.yaml", root, "absent")
    corpus = read_corpus(found["corpus"], inventory, operator_private=operator_private)
    if found["corpus"] is None:
        inventory.note("capsules/MANIFEST.yaml", root, "absent")
    coverage = read_coverage(found["coverage"], inventory, operator_private=operator_private)
    if found["coverage"] is None:
        inventory.note("coverage/generation.json", root, "absent")
    target = next((d["requirements"]["target"] for d in derivations if d.get("requirements")), None)
    return {
        "schema": SCHEMA,
        "kind": "phase0",
        "root": str(root.resolve()),
        "run_id": root.resolve().name,
        "target": target,
        "generated": time.time() if now is None else now,
        "operator_private": operator_private,
        "derivations": derivations,
        "corpus": corpus,
        "coverage": coverage,
        "inventory": inventory.rows,
    }


__all__ = ["SCHEMA", "discover", "is_hidden", "planned_row", "summary", "tally", "written_row"]
