"""Which library passes the author's compiler uses, how often the author touched each, and what A/Bs measured.

One row per library pass the workspace offers -- the scheduling and composition capabilities of the
selected release (``shared-library.json``'s ``scheduling_capabilities``) and the catalogue mechanisms
``OPTIMIZATION_SUGGESTIONS.json`` finds available for the target -- plus any other library module the
author's files import. The definitions are the author skill's own ``library_inventory.py``
(``scheduling-api``), loaded by path, so the console and the author's inventory agree. Per row:

* **in submission** -- the files under ``submission/`` (``compiler_library/`` excluded) that import the
  pass's module or symbol, read with ``ast`` and cached by stat identity;
* **tool calls** -- the author's commands and file edits that mention the pass (its module, symbol or
  name; :class:`.insight_common.Needles`), counted from every agent stream by
  :class:`.insight_common.AgentActivity`. A mention is not use; it is how much attention the pass got;
* **evidence** -- files under ``submission/docs/`` and the workspace's ``evidence/`` (bounded, without
  archived library copies) that mention it;
* **A/B effect** -- for each variant self-check (``.qa_channel/req_*`` naming ``variants/NAME``), the
  variant's cycles over the ``submission/`` cycles of the same case and evidence kind recorded closest
  before it (``qa/perf_history.jsonl`` or the derived history, :mod:`.case_history`), geometric mean over
  the cases both have. A variant is attributed to the passes its own files import and today's
  ``submission/`` files do not (the reverse is not attributed: ``submission/`` may have adopted a pass
  after the variant ran). Knob experiments of ``qa/idle_tuning.json`` labelled with a catalogue
  mechanism count for that mechanism. gSIM and model ratios are never mixed.

Underuse is the point: the summary names the available passes no command or edit ever mentioned.
Display telemetry; no verdict. Case names only with ``operator_private``.

Panel fields beyond the shared shape: ``rows`` (per pass: key, kind, name, modules, symbols, facet,
requires, imported_by (a count), imported_files (operator only), touches, last_touched, evidence,
effects), ``counts`` {passes, imported, touched, untouched, variants, attributed}.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from ..io import Reader, clean, epoch, is_dir, num, obj, scan_dir
from .case_history import definitions, history_for, load_script
from .insight_common import Needles, activity, ago, line, panel, row, table

SHARED = "shared-library.json"
SUGGESTIONS = "OPTIMIZATION_SUGGESTIONS.json"
_SCRIPT = ("phase1", "author_skills", "scheduling-api", "scripts", "library_inventory.py")
MAX_REQUESTS = 65536
MAX_EVIDENCE_FILES = 400
EVIDENCE_FILE_BYTES = 1 << 20
EVIDENCE_SUFFIXES = (".json", ".md", ".txt")
#: Workspace evidence subtrees never scanned (archived library copies and build outputs).
EVIDENCE_SKIP = frozenset({"compiler_library", "archive_variants", "__pycache__", "build", ".git"})
RESPONSE_LIMIT = 48 << 20
#: Share of all known passes and modules above which a file is an inventory rather than evidence.
INVENTORY_SHARE = 0.1
_INVENTORY: list = []
_NEEDLES: dict[str, tuple[tuple, Any]] = {}


def inventory():
    """The scheduling-api skill's stdlib-only inventory script, loaded once by path."""
    if not _INVENTORY:
        import merlin_experiments

        _INVENTORY.append(load_script(Path(merlin_experiments.__file__).parent.joinpath(*_SCRIPT)))
    return _INVENTORY[0]


def _documents(ctx) -> tuple[dict | None, dict | None]:
    workspace = ctx.experiment.get("workspace")
    if not workspace:
        return None, None
    root = Path(workspace)
    return (
        ctx.reader.document(root / SHARED, limit=16 << 20, stale_ok=True),
        ctx.reader.document(root / SUGGESTIONS, limit=16 << 20, stale_ok=True),
    )


def offered(shared: dict | None, suggestions: dict | None) -> list[dict]:
    """The pass rows (inventory script) with a stable ``key``; empty without a library manifest."""
    catalog = obj(shared).get("scheduling_capabilities")
    rows = inventory().passes(catalog if isinstance(catalog, dict) else None, suggestions)
    for item in rows:
        item["key"] = f"{item['kind']}:{item['name']}"
    return rows


def needles_for(rows: list[dict], modules=()) -> Needles:
    """Identifiers and ``parent.leaf`` pairs that name each pass (module leaves, symbols, its name) and
    each other public library module (key ``module:<name>``)."""
    words: dict[str, set[str]] = {}
    pairs: dict[tuple[str, str], set[str]] = {}

    def module_needles(module: str, key: str) -> None:
        parent, _, leaf = module.rpartition(".")
        if "_" in leaf:
            words.setdefault(leaf, set()).add(key)
        elif parent:
            pairs.setdefault((parent.rpartition(".")[2], leaf), set()).add(key)

    for item in rows:
        names = [item["name"]] + [s for s in item["symbols"] if "_" in s or (s[:1].isupper() and len(s) > 5)]
        for name in names:
            words.setdefault(name, set()).add(item["key"])
        for module in item["modules"]:
            module_needles(module, item["key"])
    for module in modules:
        if isinstance(module, str) and module.startswith("merlin."):
            module_needles(module, f"module:{module}")
    return Needles(words, pairs)


def touch_needles(ctx) -> Needles | None:
    """The experiment's needles, rebuilt only when its library manifests change; None without them."""
    shared, suggestions = _documents(ctx)
    if not isinstance(shared, dict) and not isinstance(suggestions, dict):
        return None
    workspace = str(ctx.experiment.get("workspace"))
    stamp = (id(shared), id(suggestions))
    held = _NEEDLES.get(workspace)
    if held is None or held[0] != stamp:
        rows = offered(shared, suggestions)
        modules = obj(shared).get("public_modules") or ()
        held = _NEEDLES[workspace] = (stamp, needles_for(rows, modules) if rows or modules else None)
    return held[1]


# -- what the author's files import ----------------------------------------------------------------------
def _parse_imports(data: bytes) -> tuple[frozenset, frozenset]:
    text = data.decode("utf-8", errors="replace")
    script = inventory()
    return frozenset(script.imports_of(text)), frozenset(script.imports_of(text, direct=True))


def tree_imports(reader: Reader, tree: Path) -> tuple[dict[str, list[str]], set[str]] | None:
    """``({name: [file]}, direct module names)`` of one author tree, or None when it is absent."""
    if not is_dir(tree):
        return None
    names: dict[str, list[str]] = {}
    direct: set[str] = set()
    for path in inventory().python_files(tree):
        parsed = reader.parsed(path, _parse_imports, limit=2 << 20)
        if not parsed:
            continue
        rel = path.relative_to(tree).as_posix()
        for name in parsed[0]:
            names.setdefault(name, []).append(rel)
        direct.update(parsed[1])
    return names, direct


def _users(item: dict, imports: dict[str, list[str]]) -> list[str]:
    return inventory().users(item["modules"], item["symbols"], imports)


# -- evidence files ----------------------------------------------------------------------------------------
def evidence_mentions(ctx, needles: Needles | None) -> dict[str, list[str]]:
    """``{pass key: [file]}`` over ``submission/docs`` and ``evidence/`` (bounded, two levels). A file
    that names more than :data:`INVENTORY_SHARE` of the known passes and modules (a copied catalogue, a
    discovery dump, the library inventory) is left out."""
    workspace = ctx.experiment.get("workspace")
    if not workspace or needles is None:
        return {}
    known = {key for keys in (*needles.words.values(), *needles.pairs.values()) for key in keys}
    inventory_size = max(4, int(len(known) * INVENTORY_SHARE))
    root = Path(workspace)
    found: dict[str, list[str]] = {}
    seen = 0
    queue = [(root / "submission" / "docs", 0), (root / "evidence", 0)]
    while queue and seen < MAX_EVIDENCE_FILES:
        folder, depth = queue.pop(0)
        for entry in scan_dir(folder, 2048):
            if entry.name in EVIDENCE_SKIP or entry.name.startswith("."):
                continue
            if entry.is_dir(follow_symlinks=False):
                if depth < 1:
                    queue.append((Path(entry.path), depth + 1))
                continue
            if not entry.name.endswith(EVIDENCE_SUFFIXES) or seen >= MAX_EVIDENCE_FILES:
                continue
            seen += 1

            def parse(data: bytes) -> frozenset:
                return frozenset(needles.touched(data.decode("utf-8", errors="replace")))

            keys = ctx.reader.parsed(Path(entry.path), parse, limit=EVIDENCE_FILE_BYTES) or ()
            if len(keys) > inventory_size:
                continue  # a file naming a large share of the library is an inventory, not evidence of use
            for key in keys:
                found.setdefault(key, []).append(os.path.relpath(entry.path, root))
    return found


# -- variant A/Bs ------------------------------------------------------------------------------------------
def _response_digest(document: Any) -> dict | None:
    """``{t, rows: [(case, kind, cycles)]}`` of one variant response."""
    if not isinstance(document, dict):
        return None
    ps = definitions()
    out = []
    for item in ps.case_rows(document):
        measured = ps.measured_cycles(document, item)
        modeled = ps.modeled_cycles(item.get("performance")) if ps.passes(item) else None
        if measured is not None:
            out.append((ps.case_name(item), "measured", measured))
        elif modeled is not None:
            out.append((ps.case_name(item), "modeled", modeled))
    return {"t": epoch(document.get("graded_at_unix_ns")), "rows": out}


def variant_checks(ctx) -> list[dict]:
    """``[{variant, request, t, rows}]`` of every completed variant self-check, oldest first."""
    workspace = ctx.experiment.get("workspace")
    channel = Path(workspace) / ".qa_channel" if workspace else None
    if channel is None or not is_dir(channel):
        return []
    names = {entry.name for entry in scan_dir(channel, MAX_REQUESTS)}
    checks = []
    for name in sorted(names):
        if not (name.startswith("req_") and name.endswith(".json")):
            continue
        rid = name[len("req_") : -len(".json")]
        if f"resp_{rid}.json" not in names:
            continue
        request = ctx.reader.document(channel / name, limit=1 << 20)
        label = obj(request).get("submission")
        if not isinstance(label, str) or not label.startswith("variants/"):
            continue
        value = ctx.reader.extract(
            channel / f"resp_{rid}.json", _response_digest, key="library_ab", limit=RESPONSE_LIMIT
        )
        if value is Reader.WAIT or not isinstance(value, dict) or not value["rows"]:
            continue
        checks.append({"variant": label.partition("/")[2], "request": rid, "t": value["t"], "rows": value["rows"]})
    checks.sort(key=lambda c: c["t"] or 0)
    return checks


def _baseline(cases: dict, case: str, kind: str, t: float | None) -> float | None:
    observed = obj(cases.get(case)).get(kind) or []
    if not observed:
        return None
    before = [o for o in observed if t is None or o["t"] <= t]
    return (before[-1] if before else observed[0])["cycles"]


def effect(check: dict, cases: dict) -> dict | None:
    """``{ratio, kind, cases: [..]}``: geometric mean of variant/submission cycles per evidence kind
    (gSIM first), or None without a comparable case."""
    by_kind: dict[str, list[tuple[str, float]]] = {}
    for case, kind, cycles in check["rows"]:
        base = _baseline(cases, case, kind, check["t"])
        if base and cycles:
            by_kind.setdefault(kind, []).append((case, cycles / base))
    for kind in ("measured", "modeled"):
        pairs = by_kind.get(kind)
        if pairs:
            ratio = math.exp(sum(math.log(r) for _, r in pairs) / len(pairs))
            return {"ratio": ratio, "kind": kind, "cases": [c for c, _ in pairs]}
    return None


def idle_tuning_effects(ctx) -> list[dict]:
    """Knob experiments of ``qa/idle_tuning.json`` labelled with catalogue mechanisms: ``{ratio, kind,
    variant, mechanisms, cases, t}`` (variant experiments are self-check requests, read above)."""
    workspace = ctx.experiment.get("workspace")
    ledger = ctx.reader.document(Path(workspace) / "qa" / "idle_tuning.json", limit=8 << 20) if workspace else None
    groups: dict[str, dict] = {}
    for record in obj(ledger).get("experiments") or ():
        record = obj(record)
        candidate = obj(record.get("candidate"))
        delta = num(record.get("delta_pct"))
        if candidate.get("kind") != "knob" or delta is None or delta <= -100:
            continue
        label = str(record.get("mechanism") or "")
        group = groups.setdefault(
            str(candidate.get("key")),
            {
                "kind": "modeled",
                "variant": f"tuning {clean(candidate.get('key'), 60)}",
                "logs": [],
                "cases": [],
                "mechanisms": {part.strip() for part in label.split(",") if part.strip()},
                "t": epoch(obj(ledger).get("updated_at_unix_ns")),
            },
        )
        group["logs"].append(math.log(1 + delta / 100))
        group["cases"].append(str(record.get("test_case")))
    return [{**g, "ratio": math.exp(sum(g["logs"]) / len(g["logs"]))} for g in groups.values() if g["logs"]]


# -- the panel --------------------------------------------------------------------------------------------
def _effect_text(found: list[dict]) -> str | None:
    if not found:
        return None
    best = min(found, key=lambda e: (e["kind"] != "measured", e["ratio"]))
    label = "gSIM" if best["kind"] == "measured" else "model"
    return f"×{best['ratio']:.2f} {label} ({len(found)} A/B)"


def read(ctx) -> dict[str, Any]:
    shared, suggestions = _documents(ctx)
    rows = offered(shared, suggestions)
    workspace = ctx.experiment.get("workspace")
    tracker = activity(ctx)
    commands = tracker.commands
    notes = []
    if not rows:
        notes.append(f"no {SHARED} scheduling capabilities or {SUGGESTIONS} in the workspace")
    submission = tree_imports(ctx.reader, Path(workspace) / "submission") if workspace else None
    imports, direct = submission or ({}, set())
    covered = {m for item in rows for m in item["modules"]}
    public = {m for m in obj(shared).get("public_modules") or () if isinstance(m, str)}
    needles = tracker.needles
    # Other library modules: those the author's files import, and public ones its tool calls mention.
    mentioned = {key[len("module:") :] for key in tracker.library if key.startswith("module:")}
    for module in sorted((direct & public if public else direct) | (mentioned & public)):
        if module not in covered:
            rows.append(
                {
                    "kind": "module",
                    "name": module.rpartition(".")[2],
                    "modules": [module],
                    "symbols": [],
                    "facet": None,
                    "status": None,
                    "requires": [],
                    "key": f"module:{module}",
                }
            )
    evidence = evidence_mentions(ctx, needles)
    history = history_for(ctx)
    cases = history.cases()
    checks = variant_checks(ctx)
    variant_imports: dict[str, tuple | None] = {}
    attributed = 0
    effects: dict[str, list[dict]] = {}
    for check in checks:
        name = check["variant"]
        if name not in variant_imports:
            variant_imports[name] = tree_imports(ctx.reader, Path(workspace) / "variants" / name)
        found = effect(check, cases)
        if found is None or variant_imports[name] is None or submission is None:
            continue
        theirs = variant_imports[name][0]
        hit = False
        for item in rows:
            # Only a pass the variant adds: today's submission/ is not the tree the variant was compared
            # with, so a pass submission/ adopted later says nothing about that variant.
            if _users(item, theirs) and not _users(item, imports):
                effects.setdefault(item["key"], []).append(
                    {**found, "variant": f"variants/{name}", "request": check["request"], "t": check["t"]}
                )
                hit = True
        attributed += hit
    for found in idle_tuning_effects(ctx):
        for item in rows:
            if item["kind"] == "mechanism" and item["name"] in found["mechanisms"]:
                effects.setdefault(item["key"], []).append(found)
    table_rows, keyed = [], {}
    for item in rows:
        users = _users(item, imports)
        touched = tracker.library.get(item["key"]) if needles is not None else None
        found = effects.get(item["key"]) or []
        record = {
            "key": item["key"],
            "kind": item["kind"],
            "name": item["name"],
            "modules": item["modules"],
            "symbols": item["symbols"],
            "facet": item["facet"],
            "requires": item["requires"],
            "imported_by": len(users),
            "imported_files": users if ctx.operator_private else [],
            "touches": touched[0] if touched else (0 if needles is not None else None),
            "last_touched": touched[1] if touched else None,
            "evidence": len(evidence.get(item["key"]) or ()),
            "effects": [{k: e[k] for k in ("ratio", "kind", "variant", "t")} for e in found],
        }
        keyed[item["key"]] = {
            **record,
            "evidence_files": evidence.get(item["key"]) or [],
            "effect_rows": found,
            "users": users,
        }
        untouched = item["kind"] != "module" and not users and record["touches"] == 0
        level = "warn" if untouched and commands >= 50 else ("ok" if users else None)
        table_rows.append(
            (
                record,
                row(
                    [
                        item["name"],
                        ", ".join(m.removeprefix("merlin.") for m in item["modules"][:2]) or None,
                        (f"{len(users)} file" + ("s" if len(users) != 1 else "")) if users else "no",
                        record["touches"],
                        record["evidence"] or None,
                        _effect_text(found),
                    ],
                    level,
                    key=item["key"],
                ),
            )
        )
    ctx.state["rows"] = keyed
    # Sent least touched first (passes before other modules), so underuse leads the table.
    table_rows.sort(
        key=lambda pair: (
            pair[0]["kind"] == "module",
            pair[0]["imported_by"] > 0,
            pair[0]["touches"] or 0,
            pair[0]["name"],
        )
    )
    passes = [r for r, _ in table_rows if r["kind"] != "module"]
    imported = [r for r in passes if r["imported_by"]]
    untouched = [r for r in passes if r["touches"] == 0 and not r["imported_by"]]
    order_least = list(range(len(table_rows)))
    order_most = sorted(
        range(len(table_rows)), key=lambda i: (-(table_rows[i][0]["touches"] or 0), table_rows[i][0]["name"])
    )
    with_effect = [i for i in range(len(table_rows)) if table_rows[i][0]["effects"]]
    order_effect = sorted(with_effect, key=lambda i: min(e["ratio"] for e in table_rows[i][0]["effects"]))
    touches_total = sum(entry[0] for entry in tracker.library.values()) if needles is not None else None
    best = min(
        (e for r, _ in table_rows for e in r["effects"] if e["kind"] == "measured"),
        key=lambda e: e["ratio"],
        default=None,
    )
    summary = [
        line(
            "passes imported by submission/",
            f"{len(imported)} / {len(passes)}" if passes else None,
            "warn" if passes and not imported and commands >= 50 else None,
        ),
        line(
            "passes never touched",
            (f"{len(untouched)}: " + ", ".join(r["name"] for r in untouched[:6]) + (" …" if len(untouched) > 6 else ""))
            if passes and needles
            else None,
            "warn" if untouched and commands >= 50 else ("ok" if passes and needles else None),
            key="untouched",
        ),
        line("library mentions / commands", f"{touches_total} / {commands}" if touches_total is not None else None),
        line("variant A/Bs attributed", f"{attributed} of {len(checks)}" if checks else None),
        line("best measured pass effect", f"×{best['ratio']:.2f} via {best['variant']}" if best else None),
    ]
    if tracker.catching_up:
        notes.append("agent streams are still being read; tool-call counts will grow")
    if history.pending:
        notes.append(f"{history.pending} records wait for the next refresh")
    notes.append("tool calls are commands or edits that mention a pass; A/B effect is variant/submission cycles")
    return panel(
        "LIBRARY",
        summary,
        table_=table(
            ["pass", "module", "in submission", "tool calls", "evidence", "A/B effect"], [r for _, r in table_rows]
        ),
        notes=notes,
        sorts=[
            {"label": "least touched", "order": order_least},
            {"label": "most touched", "order": order_most},
            {"label": "A/B effect", "order": order_effect},
        ],
        rows=[r for r, _ in table_rows],
        counts={
            "passes": len(passes),
            "imported": len(imported),
            "touched": len(passes) - len(untouched),
            "untouched": len(untouched),
            "variants": len(checks),
            "attributed": attributed,
        },
    )


def inspect(ctx, key: str) -> dict[str, Any]:
    keyed = obj(ctx.state.get("rows"))
    if key == "untouched":
        names = [r["name"] for r in keyed.values() if r["kind"] != "module" and r["touches"] == 0 and not r["users"]]
        return {"title": "library passes never touched", "sections": [{"title": "passes", "lines": names}], "paths": []}
    item = keyed.get(key)
    if not item:
        return {"title": f"library pass {clean(key, 80)}", "sections": [], "paths": [], "notice": "no such pass"}
    private = ctx.operator_private
    sections = [
        {
            "title": "pass",
            "lines": [
                f"{item['kind']} {item['name']}" + (f" (facet {item['facet']})" if item["facet"] else ""),
                *(f"module {m}" for m in item["modules"]),
                *(f"symbol {s}" for s in item["symbols"][:8]),
                *(f"requires {clean(r, 140)}" for r in item["requires"][:4]),
            ],
        },
        {
            "title": "in submission/",
            "lines": (item["users"] if private else [f"{len(item['users'])} file(s)"])
            or ["not imported by the author's files"],
        },
        {
            "title": "author tool calls",
            "lines": [
                f"{item['touches']} commands or edits mention it" if item["touches"] is not None else "not counted",
                f"last {ago(ctx.now, item['last_touched'])}" if item["last_touched"] else "never",
            ],
        },
        {
            "title": "evidence files",
            "lines": (item["evidence_files"][:20] if private else [f"{len(item['evidence_files'])} file(s)"])
            or ["none"],
        },
        {
            "title": "variant A/Bs",
            "lines": [
                f"{e['variant']}: ×{e['ratio']:.3f} "
                f"{'gSIM' if e['kind'] == 'measured' else 'model'} over {len(e['cases'])} case(s)"
                + (f" ({', '.join(e['cases'][:4])})" if private else "")
                for e in item["effect_rows"]
            ]
            or ["none"],
        },
    ]
    return {"title": f"library pass {item['name']}", "sections": sections, "paths": []}


__all__ = ["effect", "inspect", "needles_for", "offered", "read", "touch_needles", "tree_imports", "variant_checks"]
