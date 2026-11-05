"""Performance work starts per passing case, bundled by kind; correctness and cycle evidence per family.

Measured on one run: the author split its helpers by compiler layer and its first measured speed-up
came after about three hours, although many dense products had passed long before. The feedback ranked
only cases with cycles in the request at hand, so nothing said "this case passes: its performance
work can start now", which passing cases share a pipeline and so one change and one check serve them
all, or which library route builds a failing case's first correct version so that it is already fast.
This module states, from the newest completed grade (``qa/latest_completed_grade.json``, rows merged
across grades), current-source completed selfchecks and the response being built:

* **readiness per case**: a case with a passing numerical result (L2) is ready for performance work
  now, whatever its siblings do; a case without a result is unchecked, not failing;
* **performance bundles**: passing cases that share an operation family (each case's own declaration,
  :func:`..operation_optimizations.families_of`; the staged ``OPTIMIZATION_BY_OPERATION.json`` names
  its panel, e.g. ``dense`` or ``conv``) and a shape class (:func:`.shape_classes.classes`, against the
  tile edge the target's RTL facts state -- the classes ``--panel FAMILY`` and
  :mod:`merlin.transforms.shape_policy` report). One representative -- a large or connected case with
  the largest headroom above its structural bound, else the largest declared work -- is the case to
  optimize; its siblings are the held-out check set for the A/B (``--panel``);
* **finish-correctness work per family**: the failing or unchecked cases and the construction route --
  the family's recipes from the staged selection, the one composing most of the family's catalogued
  library mechanisms first (counted in the recipe's own source) -- so the first correct version is built
  on the performant pipeline rather than a per-tile emitter;
* **an informational family line**: correct / total, large, timed (complete cycle evidence: a gSIM
  measurement or a complete timing-model estimate). It gates and delays nothing.

:mod:`.next_steps` ranks the bundle and correctness steps by value (large and connected cases with
headroom first), never one kind behind the other; :func:`preferred_cases` gives the idle timing lanes
the bundle representatives as an ordering preference.

Advisory only: no gate, quota, threshold or skip rule; it changes no status or count. Bounded reads,
no ``re``, no target name.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import reconcile

SCHEMA = "merlin.family_convergence.v1"
KEY = "family_convergence"
#: Cases named per line; records keep up to :data:`KEEP`.
NAMED = 3
KEEP = 12
#: Characters of a recipe's ``use`` text repeated on a line.
USE_CHARS = 110
#: Bundle lines in the brief (largest value first; the records keep every bundle).
BUNDLES = 12
PANEL_COMMAND = "python agent_selfcheck.py --panel {panel}"
_MAX_GRADE_BYTES = 96 << 20
_MAX_INVENTORY_BYTES = 16 << 20
_GRADE_CACHE: dict = {}
_WINDOW_FIELDS = ("stride", "dilation", "padding")


def _read_json(path: Path, limit: int):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            return None
        from .result_details import hydrate

        return hydrate(json.loads(path.read_text(encoding="utf-8")), document_path=path)
    except (OSError, ValueError):
        return None


def _status(value):
    return value.get("status") if isinstance(value, dict) else value


def verdict(row) -> str:
    """``pass``, ``fail`` or ``unchecked`` for one result row (numerics: the L2 tier, else the row's flags)."""
    if not isinstance(row, dict):
        return "unchecked"
    tiers = row.get("tiers") if isinstance(row.get("tiers"), dict) else {}
    l2 = _status(tiers.get("L2"))
    if l2 in ("pass", "fail"):
        return l2
    if row.get("status") == "fail":
        return "fail"
    if row.get("numerical_execution_passed") is True or (not tiers and row.get("pass") is True):
        return "pass"
    return "unchecked"


# --- inputs ---------------------------------------------------------------------------------------------


def grade_rows(workspace) -> dict:
    """``{case: row}`` of the newest completed grade (rows merged across grades); cached by file identity."""
    path = Path(workspace) / "qa" / "latest_completed_grade.json"
    try:
        stat = path.stat()
        identity = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return {}
    cached = _GRADE_CACHE.get(str(path))
    if cached is not None and cached[0] == identity:
        rows = {name: dict(row) for name, row in cached[1].items()}
    else:
        document = _read_json(path, _MAX_GRADE_BYTES)
        rows = {}
        for row in (document or {}).get("per_capsule") or ():
            name = row.get("capsule") if isinstance(row, dict) else None
            if isinstance(name, str) and name:
                rows[name] = _slim(row)
        _GRADE_CACHE.clear()
        _GRADE_CACHE[str(path)] = (identity, rows)
        rows = {name: dict(row) for name, row in rows.items()}
    _refresh_currency(workspace, rows)
    return rows


def _refresh_currency(workspace, rows):
    if any(isinstance(row.get("graded"), dict) for row in rows.values()):
        reach = reconcile.current_reach(workspace)
        source = None
        if reach is None:
            from merlin.common.tree_hash import hash_tree

            try:
                source = hash_tree(Path(workspace) / "submission")["sha256"]
            except OSError:
                pass
        reconcile.label(rows.values(), reach=reach, source=source)
        if not reach and not source:
            for row in rows.values():
                if isinstance(row.get("graded"), dict):
                    row["graded"] = {**row["graded"], "currency": reconcile.OLDER}


def _slim(row: dict) -> dict:
    """The fields this module reads (a merged grade holds every case's full row)."""
    keep = {key: row[key] for key in ("capsule", "status", "pass", "tiers", "barrier_tier", "barrier_cycles",
                                     "numerical_execution_passed")
            if key in row}
    if isinstance(row.get("graded"), dict):
        keep["graded"] = dict(row["graded"])
    performance = row.get("performance")
    if isinstance(performance, dict):
        keep["performance"] = {
            key: performance[key]
            for key in ("status", "complete_event_accounting", "estimated_whole_soc_cycles", "structural_roofline",
                        "scoped_resource_feedback")
            if key in performance
        }
    return keep


def _completed_at(row):
    origin = row.get("graded") or {}
    stamp = origin.get("graded_at_unix_ns")
    if type(stamp) is int:
        return stamp
    try:
        return int(datetime.fromisoformat(origin["graded_at"]).timestamp() * 1_000_000_000)
    except (KeyError, TypeError, ValueError, OverflowError):
        return 0


def _selfcheck_digest(workspace):
    from merlin.common import stat_memo

    from ..brokers.selfcheck import _DIGEST_IGNORES, _submission_digest

    root = Path(workspace) / "submission"
    return stat_memo.cached("family-selfcheck-source", str(root.absolute()),
        lambda: stat_memo.tree_signature(root, skip_names=_DIGEST_IGNORES), lambda: _submission_digest(root))


def _selfcheck_index(path):
    """Cache only the response header and compact row pointers, never its large feedback blocks."""
    from merlin.common import stat_memo

    from .idle_performance import _load

    def read():
        document = _load(path)
        if not isinstance(document, dict):
            return None
        out = {key: document.get(key) for key in ("selfcheck_request_id", "submission_sha256",
               "selfcheck_submission", "graded_at_unix_ns")}
        out["variant"] = bool(document.get("variant"))
        out["rows"] = [{"capsule": row.get("capsule"), "detail": dict(row["detail"])}
                       if isinstance(row.get("detail"), dict) else _slim(row)
                       for row in document.get("cases") or document.get("per_capsule") or ()
                       if isinstance(row, dict)]
        return out

    return stat_memo.cached("family-selfcheck-index", str(path.absolute()),
                            lambda: stat_memo.file_signature(path), read)


def _selfcheck_inputs(workspace, digest):
    """Index each response once; detail dependencies use the existing public row-reader boundary."""
    root = Path(workspace).resolve()
    channel = root / ".qa_channel"
    for path in sorted(channel.glob("resp_*.json")):
        index = _selfcheck_index(path)
        rid = path.stem.removeprefix("resp_")
        if (not isinstance(index, dict) or index.get("selfcheck_request_id") != rid
                or index.get("submission_sha256") != digest or index.get("variant")
                or index.get("selfcheck_submission") != "submission"
                or type(index.get("graded_at_unix_ns")) is not int):
            continue
        paths = [path, channel / f"req_{rid}.json", channel / f"done_{rid}"]
        for row in index["rows"]:
            pointer = row.get("detail")
            if not pointer:
                continue
            relative, sha = pointer.get("path"), pointer.get("sha256")
            if not isinstance(relative, str) or not isinstance(sha, str) or len(sha) != 64:
                paths = None
                break
            detail = root / relative
            if (detail.is_symlink() or not detail.resolve().is_relative_to(root / "qa" / "cases")
                    or detail.name != f"{sha}.json"):
                paths = None
                break
            paths.append(detail)
        yield path, index, paths


def _selfcheck_signature(paths):
    from merlin.common import stat_memo

    if paths is None:
        return None
    rows = []
    for path in paths:
        signature = stat_memo.file_signature(path)
        if signature is None:
            return None  # absent request/DONE/detail is retryable and never cached as completed evidence
        rows.extend(signature)
    return tuple(rows)


def selfcheck_identity(workspace):
    """Cheap source/completion identity for readiness consumers; repaired details invalidate it too."""
    digest = _selfcheck_digest(workspace)
    return digest, tuple((str(path), _selfcheck_signature(paths))
                         for path, _, paths in _selfcheck_inputs(workspace, digest))


def _selfcheck_projection(path, index):
    from .idle_performance import _load
    from .result_details import load_row

    rid = index["selfcheck_request_id"]
    request = _load(path.with_name(f"req_{rid}.json"))
    if (not isinstance(request, dict) or request.get("action")
            or request.get("compile_only") or request.get("candidate_diagnostics")
            or request.get("caller_layout") or request.get("tuning_overrides")
            or request.get("submission", "submission") != "submission"):
        return {}
    rows = {}
    for compact in index["rows"]:
        row = load_row(compact, document_path=path)
        if not isinstance(row, dict) or compact.get("detail") and row is compact:
            continue
        name = row.get("capsule")
        if not isinstance(name, str) or not name or verdict(row) == "unchecked":
            continue
        row = _slim(row)
        row["graded"] = dict(currency=reconcile.CURRENT, graded_source_sha256=index["submission_sha256"],
                             graded_at_unix_ns=index["graded_at_unix_ns"], selfcheck_request_id=rid)
        rows[name] = row
    return rows


def selfcheck_rows(workspace) -> dict:
    """Completed baseline numerical results for the exact source on disk; no grade or certificate credit."""
    from merlin.common import stat_memo

    if not any((Path(workspace) / ".qa_channel").glob("resp_*.json")):
        return {}
    digest = _selfcheck_digest(workspace)
    rows = {}
    for path, index, paths in _selfcheck_inputs(workspace, digest):
        if not path.with_name(f"done_{index['selfcheck_request_id']}").is_file():
            continue
        observed = stat_memo.cached("family-selfcheck-projection", (str(path.absolute()), digest),
            lambda paths=paths: _selfcheck_signature(paths),
            lambda path=path, index=index: _selfcheck_projection(path, index))
        for name, row in observed.items():
            if name not in rows or _completed_at(row) > _completed_at(rows[name]):
                rows[name] = row
    return rows


def readiness_rows(workspace) -> dict:
    """Existing current-grade/selfcheck precedence, shared by summaries and idle readiness readers."""
    graded = grade_rows(workspace)
    for name, row in selfcheck_rows(workspace).items():
        held = graded.get(name)
        if (held is None or reconcile.is_older(held) or verdict(held) == "unchecked"
                or _completed_at(row) >= _completed_at(held)):
            graded[name] = row
    return graded


def inventory(workspace) -> dict:
    """``{case: inventory row}`` from the staged ``PUBLIC_CAPABILITIES.json``; ``{}`` without one."""
    record = _read_json(Path(workspace) / "PUBLIC_CAPABILITIES.json", _MAX_INVENTORY_BYTES)
    members = record.get("members") if isinstance(record, dict) else None
    return members if isinstance(members, dict) else {}


def _declaration_of(member: dict) -> dict:
    """A declaration-shaped view of one inventory row: operation, window attributes, semantic family, inputs."""
    layout = member.get("layout_requirements") if isinstance(member.get("layout_requirements"), dict) else {}
    op = member.get("declared_operation")
    out = {
        "operation": {"op": op, "attributes": {key: layout[key] for key in _WINDOW_FIELDS if key in layout}},
        "inputs": [t for t in member.get("input_tensors") or () if isinstance(t, dict)],
    }
    if isinstance(member.get("family"), str) and member["family"]:
        out["semantic"] = {"semantic_family": member["family"]}
    if op == "model":
        out["kind"] = "model"
    return out


def describe(name: str, member, workspace) -> tuple[str | None, dict | None]:
    """``(operation family, shape description)`` from the case's declaration, else its inventory row."""
    from .. import operation_optimizations
    from . import case_declarations, shape_classes

    declaration = case_declarations.load(name, workspace)
    if declaration is None and isinstance(member, dict) and member.get("declared_operation"):
        declaration = _declaration_of(member)
    found = operation_optimizations.families_of(declaration)
    try:
        shape = shape_classes.describe(declaration) if declaration is not None else None
    except Exception:  # noqa: BLE001 -- an undescribable case simply has no shape class
        shape = None
    return (found[0] if found else None), shape


def family_of(name: str, member, workspace) -> str | None:
    return describe(name, member, workspace)[0]


def measured_cases(workspace) -> set[str]:
    """Cases with a gSIM measurement row in the idle measurement lane's responses."""
    try:
        from .idle_gsim import measured_executables

        return {case for case, _ in measured_executables(Path(workspace) / ".qa_channel")}
    except Exception:  # noqa: BLE001 -- no measurement lane means no measured rows
        return set()


def _edge():
    from . import case_declarations, shape_classes

    return shape_classes.tile_edge(case_declarations.target())[0]


# --- evidence -------------------------------------------------------------------------------------------


def _evidence_entry(name: str, row: dict) -> dict | None:
    """A headroom entry (the performance brief's fields) from one row with complete cycle evidence."""
    from .performance_brief import _evidence
    from .performance_headroom import _bound, _dominant_resource

    found = _evidence(row)
    if found is None:
        return None
    cycles, roofline, label = found
    bound, limiter = _bound(roofline or {})
    entry = {"test_case": name, "cycles": cycles, "evidence": label}
    if bound is not None and cycles >= bound:
        entry.update(roofline_bound_cycles=round(bound, 1), bound_by=limiter, gap_ratio=round(cycles / bound, 2),
                     headroom_cycles=round(cycles - bound, 1))
    dominant = _dominant_resource(row.get("performance") or {})
    if dominant is not None:
        entry["dominant_modeled_resource"] = dominant
    return entry


def _brief_entries(brief) -> dict:
    """The brief's own (richer) entries by case: fence and capacity readings included."""
    out = {}
    if isinstance(brief, dict):
        for key in ("largest_headroom", "diagnostic_controls"):
            for entry in brief.get(key) or ():
                if isinstance(entry, dict) and isinstance(entry.get("test_case"), str):
                    out.setdefault(entry["test_case"], entry)
    return out


_TOP_FIELDS = ("test_case", "cycles", "evidence", "roofline_bound_cycles", "bound_by", "gap_ratio",
               "headroom_cycles", "dominant_modeled_resource", "fence_overlap", "capacity_use", "suggested_mechanisms")


# --- construction routes ---------------------------------------------------------------------------------


def construction_routes(entry: dict) -> list[dict]:
    """The family's recipes, the one composing most of its catalogued mechanisms first (then catalogue order)."""
    from .. import operation_optimizations

    modules = {
        str(m.get("api")).partition(":")[0]
        for m in entry.get("mechanisms") or ()
        if isinstance(m, dict) and m.get("kind") == "catalog" and m.get("api")
    }
    ranked = []
    for index, mechanism in enumerate(entry.get("mechanisms") or ()):
        if not isinstance(mechanism, dict) or mechanism.get("kind") != "recipe":
            continue
        skill, _, script = str(mechanism.get("id")).partition("/")
        try:
            text = (operation_optimizations._SKILLS / skill / "scripts" / script).read_text(encoding="utf-8")
        except OSError:
            text = ""
        composed = sorted(module.rpartition(".")[2] for module in modules if module and module in text)
        ranked.append((-len(composed), index, {"recipe": mechanism["id"], "use": mechanism.get("use"),
                                               "composes": composed}))
    return [item for *_, item in sorted(ranked, key=lambda r: r[:2])]


def _selection_entries(workspace) -> dict:
    from .. import operation_optimizations

    selection = operation_optimizations.load_selection(workspace)
    return {e.get("id"): e for e in (selection or {}).get("entries") or () if isinstance(e, dict)}


# --- the roll-up ----------------------------------------------------------------------------------------


def _response_rows(rows) -> dict:
    out = {}
    for row in rows or ():
        name = (row.get("capsule") or row.get("name")) if isinstance(row, dict) else None
        if isinstance(name, str) and name:
            out[name] = row
    return out


class _Case:
    __slots__ = ("name", "family", "shape", "klass", "status", "role", "primary", "entry", "timed", "older",
                 "older_status")


def _cases(workspace, rows, brief, edge):
    from ..workload_roles import PRIMARY, workspace_roles
    from . import shape_classes

    graded = readiness_rows(workspace)
    current = {name: dict(row) for name, row in _response_rows(rows).items()}
    _refresh_currency(workspace, current)
    members = inventory(workspace)
    try:
        roles = workspace_roles(workspace) or {}
    except (OSError, ValueError):
        roles = {}
    measured = measured_cases(workspace)
    richer = _brief_entries(brief)
    out = []
    for name in sorted(set(members) | set(graded) | set(current)):
        case = _Case()
        case.name = name
        case.family, case.shape = describe(name, members.get(name), workspace)
        if case.family is None:
            continue
        classes = shape_classes.classes(case.shape, edge) if case.shape else ()
        case.klass = classes[0] if classes else None
        response = current.get(name)
        status = verdict(response) if not reconcile.is_older(response) else "unchecked"
        case.status = verdict(graded.get(name)) if status == "unchecked" else status
        case.older = status == "unchecked" and reconcile.is_older(graded.get(name))
        case.older_status = case.status if case.older else None
        if case.older:
            case.status = "unchecked"  # an old failure is not a failure of the edited compiler either
        case.role = roles.get(name) or "unclassified"
        case.primary = case.role in PRIMARY
        case.entry = richer.get(name)
        for source in (current.get(name), graded.get(name)):
            if case.entry is None and isinstance(source, dict) and not reconcile.is_older(source):
                case.entry = _evidence_entry(name, source)
        case.timed = case.status == "pass" and (case.entry is not None or name in measured)
        out.append(case)
    return out


def build(workspace=None, *, rows=(), brief=None, edge=None) -> dict | None:
    """``{schema, scope, families, bundles, lines}`` for the workspace's cases; None when none has a family."""
    from ..author_profiles import modern_advice

    if not modern_advice("performance_diagnostics"):
        return None
    workspace = Path(workspace) if workspace is not None else Path.cwd()
    from .result_details import load_row

    rows = [load_row(row, workspace=workspace) for row in rows or ()]
    edge = _edge() if edge is None else edge
    cases = _cases(workspace, rows, brief, edge)
    if not cases:
        return None
    entries = _selection_entries(workspace) if modern_advice("catalog_advice") else {}
    groups: dict[str, list] = {}
    for case in cases:
        groups.setdefault(case.family, []).append(case)
    families = sorted((_family(f, members, entries.get(f)) for f, members in groups.items()),
                      key=lambda r: (-r["large"], -r["cases"], r["family"]))
    bundles = _bundles(cases, entries)
    ready_untimed = sorted(c.name for c in cases if c.status == "pass" and c.primary and not c.timed)
    return {
        "schema": SCHEMA,
        "advisory_only": True,
        "scope": ("cases per operation family (each case's declaration); correct = current-source numerics (L2) "
                  "in completed grades or baseline selfchecks, overlaid with this response's rows; "
                  "older results remain separate; large = large or connected workload roles; "
                  "timed = complete cycle evidence (gSIM measurement or complete timing-model estimate)"),
        "how_to_use": ("A passing case is ready for performance work now: each bundle of passing cases sharing a "
                       "family and shape class names one representative to optimize and its siblings as the "
                       "held-out check (--panel). Failing cases keep their correctness step beside it; neither "
                       "waits for the other, and no family has to finish first."),
        "families": families,
        "bundles": bundles,
        "bundle_count": len(bundles),
        "ready_untimed": ready_untimed,
        "family_line": family_line(families),
        "lines": [bundle_line(b) for b in bundles[:BUNDLES]],
    }


def _family(family, cases, entry) -> dict:
    passing = [c for c in cases if c.status == "pass"]
    failing = sorted(c.name for c in cases if c.status == "fail")
    unchecked = sorted(c.name for c in cases if c.status == "unchecked")
    # Large and connected failing cases lead the correctness list (the value order of the steps).
    left = sorted((c for c in cases if c.status != "pass"), key=lambda c: (not c.primary, c.status != "fail", c.name))
    record = {
        "family": family,
        "label": (entry or {}).get("panel_family") or family,
        "cases": len(cases),
        "correct": len(passing),
        "correct_older_version": sum(1 for c in cases if c.older_status == "pass"),
        "all_correct": len(passing) == len(cases),
        "failing": failing[:KEEP],
        "failing_count": len(failing),
        "unchecked": unchecked[:KEEP],
        "unchecked_count": len(unchecked),
        "left": [c.name for c in left][:KEEP],
        "left_large": sum(1 for c in left if c.primary),
        "large": sum(1 for c in cases if c.primary),
        "large_correct": sum(1 for c in passing if c.primary),
        "timed": sum(1 for c in cases if c.timed),
        "large_timed": sum(1 for c in cases if c.primary and c.timed),
    }
    if entry and entry.get("panel_family"):
        record["panel"] = entry["panel_family"]
    routes = construction_routes(entry) if isinstance(entry, dict) else []
    if routes:
        record["construction"] = routes[:2]
    return record


def _work(case) -> int:
    return (case.shape or {}).get("work") or 0


def _bundles(cases, entries) -> list[dict]:
    groups: dict[tuple, list] = {}
    for case in cases:
        if case.status == "pass":
            groups.setdefault((case.family, case.klass or ""), []).append(case)
    out = []
    for (family, klass), members in groups.items():
        # The representative: large and connected roles first, then headroom above the bound, then declared work.
        def value(c):
            return (c.primary, (c.entry or {}).get("headroom_cycles") or 0, _work(c), c.name)

        members.sort(key=value, reverse=True)
        rep = members[0]
        panel = (entries.get(family) or {}).get("panel_family")
        label = (entries.get(family) or {}).get("panel_family") or family
        bundle = {
            "family": family,
            "label": f"{label} {klass}" if klass else label,
            "shape_class": klass or None,
            "passing": len(members),
            "representative": rep.name,
            "role": rep.role,
            "primary": rep.primary,
            "siblings": [c.name for c in members[1:]][:KEEP],
            "large": sum(1 for c in members if c.primary),
            "timed": sum(1 for c in members if c.timed),
            "headroom_cycles": (rep.entry or {}).get("headroom_cycles"),
            # The other performance targets: large or connected passing members with headroom.
            "targets": [c.name for c in members[1:] if c.primary and ((c.entry or {}).get("headroom_cycles") or 0) > 0
                        ][:KEEP],
        }
        if rep.entry is not None and (rep.entry.get("headroom_cycles") or 0) > 0:
            bundle["top"] = {key: rep.entry[key] for key in _TOP_FIELDS if key in rep.entry}
        elif not rep.timed:
            bundle["untimed"] = True
        if panel:
            bundle["panel"] = PANEL_COMMAND.format(panel=panel)
        out.append(bundle)
    out.sort(key=lambda b: (not b["primary"], -(b["headroom_cycles"] or 0), -b["passing"], b["label"]))
    return out


# --- lines ----------------------------------------------------------------------------------------------


def _names(names: list[str], total: int) -> str:
    shown = ", ".join(names[:NAMED])
    return shown + (f" (+{total - NAMED})" if total > NAMED else "")


def family_line(families) -> str:
    """One informational line: correct / total and timed per family (it gates nothing)."""
    parts = []
    for record in families:
        text = f"{record['label']} {record['correct']}/{record['cases']} ({record['timed']} timed)"
        if record.get("failing_count") or record.get("unchecked_count"):
            text += f" [{record.get('failing_count', 0)} failed, {record.get('unchecked_count', 0)} unchecked]"
        if record.get("correct_older_version"):
            text += f" ({record['correct_older_version']} passed on an older version)"
        parts.append(text)
    return "families (correct/total; informational, no family waits for another): " + ", ".join(parts)


def bundle_line(bundle: dict, step: dict | None = None) -> str:
    """``dense tall (6 passing: REP, +5 siblings) -> next; compare with --panel dense``."""
    text = f"{bundle['label']} ({bundle['passing']} passing: {bundle['representative']}"
    text += f", +{bundle['passing'] - 1} siblings)" if bundle["passing"] > 1 else ")"
    top = bundle.get("top")
    if step is not None:
        text += f" -> {step.get('choice')}"
    elif top:
        text += f" -> {round(top['headroom_cycles'])} cycles above its {top.get('bound_by')} bound"
    elif bundle.get("untimed"):
        text += " -> time it first"
    if bundle.get("panel") and bundle["passing"] > 1:
        text += f"; compare with {bundle['panel'].replace('python agent_selfcheck.py ', '')}"
    return text


def correctness_text(record: dict) -> str:
    """``conv: 30/34 correct -> finish correctness: a, b, c (+1); build on conv_window_recipe.py (...)``."""
    missing = record["cases"] - record["correct"]
    older = record.get("correct_older_version")
    text = f"{record['label']}: {record['correct']}/{record['cases']} correct"
    text += f" ({older} passed on an older version)" if older else ""
    text += " -> finish correctness: "
    text += _names(record["left"], missing)
    if not record["failing_count"]:
        text += " (no result yet)"
    routes = record.get("construction") or ()
    if routes:
        first = routes[0]
        use = str(first.get("use") or "")
        use = use if len(use) <= USE_CHARS else use[: USE_CHARS - 3].rstrip() + "..."
        text += f"; build on {first['recipe'].rpartition('/')[2]}" + (f" ({use})" if use else "")
        if len(routes) > 1:
            text += f", then {routes[1]['recipe'].rpartition('/')[2]}"
    return text


# --- consumers ------------------------------------------------------------------------------------------


def preferred_cases(workspace, names=None) -> set[str]:
    """Passing large or connected cases without cycle evidence and bundle representatives (of ``names``,
    else all): the idle timing lanes take them first, so evidence follows each case as it passes. An
    ordering preference only; never raises."""
    try:
        found = build(workspace)
    except Exception:  # noqa: BLE001 -- ordering preference only
        return set()
    if not found:
        return set()
    wanted = set(names) if names is not None else None
    reps = {b["representative"] for b in found["bundles"]} | set(found.get("ready_untimed") or ())
    return reps if wanted is None else reps & wanted


def attach(brief, rows, *, workspace=None) -> dict | None:
    """Put :data:`KEY` on the brief (when there is one) and return the roll-up; never raises."""
    try:
        found = build(workspace, rows=rows, brief=brief)
    except Exception:  # noqa: BLE001 -- advisory only
        return None
    if found is not None and isinstance(brief, dict):
        brief[KEY] = found
    return found


__all__ = ["KEY", "SCHEMA", "attach", "build", "bundle_line", "construction_routes",
           "correctness_text", "describe", "family_line", "family_of", "preferred_cases", "verdict"]
