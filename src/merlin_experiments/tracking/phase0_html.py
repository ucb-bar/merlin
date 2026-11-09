"""Render the Phase 0 corpus-and-coverage summary (:mod:`.phase0`) as one self-contained page."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import charts as C
from . import phase0 as P0
from .html import NR, bars, details, esc, inventory_section, mono, page, share, table, text_or_nr, tile

_AXIS_TITLES = {
    "cells": "Family / dtype / alignment cells",
    "shape_geometry": "Shape geometry strata",
    "conv_geometry": "Convolution windows",
    "epilogue": "Epilogue stages",
    "groups": "Compute groups (op | epilogue combination | dtype)",
    "carried_state": "Carried configuration state",
    "scope": "Adjacency scope (operation chains)",
}


def mode_banner(private: bool) -> str:
    if private:
        return (
            '<span class="mode">OPERATOR-PRIVATE mode: hidden capsules, held-out digests and private '
            "references are shown by name. Do not share this page with an agent under test.</span>"
        )
    return (
        '<span class="mode">Public mode: hidden cohort, held-out layers and private references appear as '
        "counts only (pass --operator-private to name them).</span>"
    )


def matrix(rows: Sequence[str], cols: Sequence[str], cell) -> str:
    """An HTML matrix: ``cell(row, col) -> (css class, text, tip)``."""
    if not rows or not cols:
        return f"<p>{NR}</p>"
    head = "<tr><th></th>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr>"
    body = []
    for r in rows:
        cells = []
        for c in cols:
            cls, text, tip = cell(r, c)
            cells.append(f'<td class="{cls}" data-tip="{esc(tip)}">{text}</td>')
        body.append(f"<tr><th>{esc(r)}</th>{''.join(cells)}</tr>")
    return f'<div class="scroll"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'


# --------------------------------------------------------------------------- inventory
def _axis_tables(t: Mapping[str, Any]) -> str:
    blocks = []
    for axis in P0.INVENTORY_AXES:
        counts = t["by"].get(axis) or {}
        rows = [[esc(k), f"{v:,}"] for k, v in list(counts.items())[:25]]
        more = f'<p class="sub">{len(counts) - 25} more values</p>' if len(counts) > 25 else ""
        blocks.append(
            f"<div><h3>By {esc(axis.replace('_', ' '))}</h3>{table([axis, 'capsules'], rows, (1,))}{more}</div>"
        )
    return '<div class="cols">' + "".join(blocks) + "</div>"


def _rows_table(rows: Sequence[Mapping[str, Any]], label: str) -> str:
    body = [
        [
            mono(r["name"]),
            text_or_nr(r.get("category")),
            text_or_nr(r.get("family")),
            text_or_nr(r.get("op")),
            text_or_nr(r.get("form")),
            text_or_nr(r.get("stratum")) + (" *" if r.get("stratum_classified") else ""),
            text_or_nr(r.get("epilogue")),
            text_or_nr(r.get("dtype")),
            text_or_nr(r.get("tier")),
            text_or_nr(r.get("window")),
            esc(f"{r['M']}x{r['K']}x{r['N']}") if r.get("M") and r.get("K") and r.get("N") else NR,
        ]
        for r in rows
    ]
    headers = ["capsule", "category", "family", "op", "form", "stratum", "epilogue", "dtype", "tier", "window", "MxKxN"]
    note = '<p class="sub">* stratum classified from M/K/N by the shape taxonomy (the capsule records none).</p>'
    return details(f"{label} ({len(rows)} capsules)", table(headers, body) + note)


def inventory_block(t: Mapping[str, Any] | None, rows: Sequence[Mapping[str, Any]], label: str) -> str:
    if not t or not t.get("n"):
        return f"<p>{esc(label)}: {NR}.</p>"
    return (
        f"<p>{t['n']:,} capsules.</p>"
        + C.treemap(t["tree"], label=f"{label}: family, then op")
        + _axis_tables(t)
        + _rows_table(rows, label)
    )


# --------------------------------------------------------------------------- requirements vs covered
def _covered(cohort: Mapping[str, Any] | None, axis: str, key: str) -> tuple[str, str]:
    """``(css class, text)`` of one requirement in one cohort's measured coverage."""
    if not cohort:
        return "cell-other", "not recorded"
    if axis == "cells":
        cells = cohort.get("cells")
        if not cells:
            return "cell-other", "not recorded"
        if key in cells["uncovered"]:
            return "gap", "&#10005; gap"
        return ("hit", "&#10003;") if key in cells["covered"] else ("cell-other", "not recorded")
    gap = (cohort.get("axes") or {}).get(axis)
    if not gap:
        return "cell-other", "not recorded"
    if gap.get("status") not in (None, "ok"):
        return "cell-other", esc(f"{gap.get('status')}: {gap.get('reason') or ''}".strip(": "))
    if key in gap["uncovered"]:
        return "gap", "&#10005; gap"
    n = gap["covered_by"].get(key)
    return "hit", f"&#10003; {n} capsules" if n else "&#10003;"


def _axis_table(axis: str, rows: list[Mapping[str, Any]] | None, coverage: Mapping[str, Any] | None) -> str:
    title = _AXIS_TITLES.get(axis, axis)
    if rows is None:
        return f"<h3>{esc(title)}</h3><p>Requirement: {NR}.</p>"
    if not rows:
        return f"<h3>{esc(title)}</h3><p>Nothing required on this axis.</p>"
    p1, p2 = (coverage or {}).get("phase1"), (coverage or {}).get("phase2")
    body = []
    gaps = 0
    for r in rows:
        c1, t1 = _covered(p1, axis, r["key"])
        c2, t2 = _covered(p2, axis, r["key"])
        gaps += (c1 == "gap") + (c2 == "gap")
        detail = ", ".join(f"{k}={v}" for k, v in r["detail"].items() if not isinstance(v, list))
        sources = next((v for k, v in r["detail"].items() if isinstance(v, list)), [])
        body.append(
            f"<tr><td>{mono(r['key'])}</td><td class='n'>{text_or_nr(r.get('weight'))}</td>"
            f"<td>{esc(detail)}</td><td>{esc(', '.join(map(str, sources)))}</td>"
            f'<td class="{c1}">{t1}</td><td class="{c2}">{t2}</td></tr>'
        )
    head = (
        "<tr><th>required</th><th class='n'>regions / weight</th><th>detail</th><th>observed in</th>"
        "<th>Phase 1 cohort</th><th>Phase 2 cohort</th></tr>"
    )
    sub = f'<p class="sub">{len(rows)} required; {gaps} gap(s) recorded across both cohorts.</p>'
    return (
        f"<h3>{esc(title)}</h3>{sub}"
        f'<div class="scroll"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'
    )


def _cells_matrix(rows: list[Mapping[str, Any]] | None, coverage: Mapping[str, Any] | None) -> str:
    if not rows:
        return ""
    keyed = {r["key"]: r for r in rows}
    families = sorted({k.split("/")[0] for k in keyed})
    columns = sorted({"/".join(k.split("/")[1:]) for k in keyed})
    p1 = (coverage or {}).get("phase1")

    def cell(family: str, col: str) -> tuple[str, str, str]:
        key = f"{family}/{col}"
        if key not in keyed:
            return "", "", f"{key}: not required"
        cls, text = _covered(p1, "cells", key)
        weight = keyed[key].get("weight")
        return cls or "cell-other", f"{text}<br><span class='sub'>{weight or ''} regions</span>", f"{key}: required"

    return "<h3>Required cells, family &times; dtype/alignment (Phase 1 cohort)</h3>" + matrix(families, columns, cell)


def _typed(typed: Mapping[str, Any] | None, coverage: Mapping[str, Any] | None) -> str:
    if not typed:
        return f"<h3>Typed required instances</h3><p>scope.typed_required_instances: {NR}.</p>"
    by_sig = typed["by_application_signature"]
    apps = sorted(by_sig)
    sigs = sorted(
        {s for row in by_sig.values() for s in row}, key=lambda s: (-sum(r.get(s, 0) for r in by_sig.values()), s)
    )
    families = sorted({f for row in typed["by_application_family"].values() for f in row})

    def sig_cell(app: str, sig: str) -> tuple[str, str, str]:
        n = by_sig.get(app, {}).get(sig, 0)
        return ("cell-other" if n else ""), (str(n) if n else ""), f"{app}: {sig}: {n} instances"

    def fam_cell(app: str, fam: str) -> tuple[str, str, str]:
        n = typed["by_application_family"].get(app, {}).get(fam, 0)
        return ("cell-other" if n else ""), (str(n) if n else ""), f"{app}: {n} {fam} regions"

    measured = []
    for phase in ("phase1", "phase2"):
        cohort = (coverage or {}).get(phase)
        apps_cov = (cohort or {}).get("applications") or {}
        for app, row in sorted(apps_cov.items()):
            cls = "gap" if row["missing"] else "hit"
            measured.append(
                [
                    esc(phase),
                    esc(app),
                    str(row["operations"]),
                    f'<span class="{cls}">{row["missing"]}</span>',
                    str(row["covered"]),
                ]
            )
    coverage_table = (
        "<h3>Application operations the cohorts were measured to witness</h3>"
        + table(["cohort", "application", "operations", "missing", "covered"], measured, (2, 3, 4))
        if measured
        else f"<p>Per-operation cohort witness: {NR} (no capsule-coverage output beside this run).</p>"
    )
    return (
        f"<h3>Typed required instances ({typed['n']}, status {text_or_nr(typed.get('status'))})</h3>"
        + matrix(apps, sigs[:14], sig_cell)
        + (f'<p class="sub">{len(sigs) - 14} rarer signatures not shown.</p>' if len(sigs) > 14 else "")
        + "<h3>Regions in those instances, application &times; semantic family</h3>"
        + matrix(apps, families, fam_cell)
        + coverage_table
    )


def requirements_section(d: Mapping[str, Any], coverage: Mapping[str, Any] | None, index: int) -> str:
    req = d.get("requirements")
    out = f'<section id="req{index}"><h2>Requirements vs covered &middot; {esc(d["name"])}</h2>'
    out += (
        f'<p class="sub">{mono(d["path"])} &middot; derivation status {text_or_nr(d.get("status"))}: '
        f"{text_or_nr(d.get('qualification'))}</p>"
    )
    if not req:
        return out + f"<p>requirements.yaml: {NR}.</p></section>"
    tiles = [
        tile("Target", text_or_nr(req.get("target")), f"tile edge {text_or_nr(req.get('tile_edge'))}"),
        tile(
            "Application operations",
            text_or_nr((req.get("demands") or {}).get("n_operations")),
            f"{text_or_nr((req.get('demands') or {}).get('n_signatures'))} signatures",
        ),
        tile("Typed required instances", text_or_nr((req.get("typed") or {}).get("n")), ""),
        tile("Planned capsules", text_or_nr((d.get("planned") or {}).get("n")), "synthesis.yaml"),
    ]
    perf = req.get("performance")
    if perf:
        tiles.append(
            tile(
                "Phase 2 scope",
                esc(perf.get("status") or "?"),
                f"{perf['required']} required, {perf['excluded']} excluded, {perf['unresolved']} unresolved",
            )
        )
    out += '<div class="tiles">' + "".join(tiles) + "</div>"
    out += _cells_matrix(req["axes"].get("cells"), coverage)
    for axis, _, _ in P0.AXES:
        out += _axis_table(axis, req["axes"].get(axis), coverage)
    out += _typed(req.get("typed"), coverage)
    families = req.get("families_observed") or {}
    out += "<h3>Semantic families observed in the captures</h3>" + bars(
        [(k, int(v), f"{v} regions") for k, v in sorted(families.items(), key=lambda kv: -int(kv[1]))], "families"
    )
    host = [[esc(h["family"]), esc(h["dtype"]), text_or_nr(h["n_regions"])] for h in req.get("host_lane") or ()]
    out += details(f"Host-lane requirements ({len(host)})", table(["family", "dtype", "regions"], host, (2,)))
    return out + "</section>"


# --------------------------------------------------------------------------- cohorts and forms
def _forms(perf: Mapping[str, Any] | None, cohort: Mapping[str, Any] | None) -> str:
    out = "<h3>Performance forms (scope.performance.forms) and their Phase 2 coverage</h3>"
    form_cov = (cohort or {}).get("form") or {}
    by_id = {c.get("class_id"): c for c in form_cov.get("classes") or ()}
    classes = (perf or {}).get("classes") or []
    if not classes:
        return out + f"<p>Form classes: {NR}.</p>"
    rows = []
    for c in sorted(classes, key=lambda c: -(c.get("max_share") or 0)):
        cov = by_id.get(c["class_id"]) or {}
        status = cov.get("status")
        cls = {"missing": "gap", "covered": "hit"}.get(status or "", "cell-other")
        strata = cov.get("strata") or {}
        rows.append(
            f"<tr><td>{mono(c['label'])}</td><td>{text_or_nr(c.get('placement'))}</td>"
            f"<td class='n'>{share(c.get('max_share'))}</td><td class='n'>{c['members']}</td>"
            f"<td>{esc(', '.join(f'{k} {v:.0%}' for k, v in sorted(c['share_by_application'].items())))}</td>"
            f"<td>{text_or_nr(cov.get('required'))}</td>"
            f'<td class="{cls}">{text_or_nr(status)}</td>'
            f"<td>{esc(', '.join(strata.get('unrepresented') or [])) or '&ndash;'}</td></tr>"
        )
    head = (
        "<tr><th>form</th><th>placement</th><th class='n'>max share</th><th class='n'>members</th>"
        "<th>share by application</th><th>required</th><th>Phase 2 status</th><th>unrepresented strata</th></tr>"
    )
    note = (
        f'<p class="sub">form_perf_coverage: status {text_or_nr(form_cov.get("status"))}, threshold '
        f"{share(form_cov.get('threshold'))} predicted-cycle share; missing {len(form_cov.get('missing') or [])}; "
        f"source windows without a form {len(form_cov.get('windows_without_form') or [])}.</p>"
        if form_cov
        else f'<p class="sub">form_perf_coverage: {NR} (no phase2-capsule-coverage.json).</p>'
    )
    return out + note + f'<div class="scroll"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def cohorts_section(s: Mapping[str, Any]) -> str:
    corpus, coverage = s.get("corpus") or {}, s.get("coverage") or {}
    manifest = corpus.get("manifest")
    out = '<section id="cohorts"><h2>Cohorts: Phase 1 functional, Phase 2 performance, diagnostic</h2>'
    if not manifest:
        out += f"<p>capsules/MANIFEST.yaml: {NR}.</p>"
    else:
        cohorts = manifest.get("cohorts") or {}
        cols = []
        for name, c in cohorts.items():
            cov = coverage.get(name.rpartition(":")[2]) if name.rpartition(":")[2] in ("phase1", "phase2") else None
            status = f"coverage {esc(cov['status'])}" if cov else "coverage not recorded"
            cats = [[esc(k), str(v)] for k, v in c["by_category"].items()]
            cols.append(
                f"<div><h3>{esc(name)} &middot; {len(c['members'])} members</h3>"
                f'<p class="sub">{text_or_nr(c.get("purpose"))}; {status}</p>'
                + table(["category", "members"], cats, (1,))
                + details("Members", "<br>".join(mono(m) for m in c["members"]) or NR)
                + "</div>"
            )
        out += (
            ('<div class="cols">' + "".join(cols) + "</div>")
            if cols
            else (f"<p>MANIFEST phase_corpora: {NR} (a manifest of the older schema).</p>")
        )
        perf_rows = []
        for target, p in (manifest.get("performance") or {}).items():
            for family, counts in sorted((p.get("by_family") or {}).items()):
                perf_rows.append(
                    [
                        esc(target),
                        esc(family),
                        text_or_nr((counts or {}).get("admitted_members")),
                        text_or_nr((counts or {}).get("written_members")),
                    ]
                )
        out += "<h3>Performance families (MANIFEST performance_generation)</h3>" + table(
            ["target", "family", "admitted members", "written members"], perf_rows, (2, 3)
        )
    for phase in ("phase1", "phase2"):
        cohort = coverage.get(phase)
        if not cohort:
            out += f"<p>{phase}-capsule-coverage.json: {NR}.</p>"
            continue
        blockers = [
            [text_or_nr(b.get("component")), text_or_nr(b.get("application")), text_or_nr(b.get("reason"))]
            for b in cohort.get("blockers") or ()
        ]
        out += (
            f"<h3>{esc(phase)} cohort coverage: {esc(cohort.get('status') or '?')}</h3>"
            f"<p>{text_or_nr(cohort.get('n_capsules'))} capsules in the measured cohort.</p>"
            + table(["blocker", "application", "reason"], blockers)
        )
    req = next((d["requirements"] for d in s.get("derivations") or () if d.get("requirements")), None)
    out += _forms((req or {}).get("performance"), coverage.get("phase2"))
    return out + "</section>"


def hidden_section(s: Mapping[str, Any]) -> str:
    private = s.get("operator_private")
    corpus = s.get("corpus") or {}
    hidden = corpus.get("hidden") or {}
    manifest = corpus.get("manifest") or {}
    out = '<section id="hidden"><h2>Hidden and private material</h2>' + mode_banner(bool(private))
    rows = [
        ["hidden capsules in the written corpus", text_or_nr(hidden.get("n")) if corpus else NR],
        ["MANIFEST held_out (counts)", esc(manifest.get("held_out")) if manifest.get("held_out") else NR],
    ]
    for d in s.get("derivations") or ():
        guard = (d.get("requirements") or {}).get("heldout_guard")
        rows.append([f"held-out layer guard ({esc(d['name'])})", esc(guard) if guard else NR])
        rows.append([f"hidden capsules planned ({esc(d['name'])})", str(d.get("planned_hidden") or 0)])
    disjoint = ((s.get("coverage") or {}).get("generation") or {}).get("hidden_disjointness")
    rows.append(["hidden/public disjointness", esc(disjoint) if disjoint else NR])
    out += table(["what", "value"], rows)
    if hidden.get("by_family"):
        out += "<h3>Hidden capsules by family (counts)</h3>" + bars(
            [(k, v, f"{v} hidden capsules") for k, v in hidden["by_family"].items()], "hidden by family"
        )
    if private and hidden.get("rows"):
        out += _rows_table(hidden["rows"], "Hidden capsules (operator-private)")
    return out + "</section>"


def distributions_section(s: Mapping[str, Any]) -> str:
    out = '<section id="distributions"><h2>Convolution windows and shapes</h2>'
    for d in s.get("derivations") or ():
        windows = (d.get("requirements") or {}).get("conv_windows") or []
        out += f"<h3>Required windows ({esc(d['name'])}), by regions observed</h3>" + bars(
            [
                (
                    w["signature"],
                    w["n_regions"] or 0,
                    f"{w['signature']}: {w['n_regions']} regions in {', '.join(w.get('sources') or [])}",
                )
                for w in windows
            ],
            "required conv windows",
        )
    sources = []
    corpus = s.get("corpus") or {}
    if corpus.get("public"):
        sources.append(("written corpus", corpus["public"]))
    for d in s.get("derivations") or ():
        if d.get("planned"):
            sources.append((f"planned ({d['name']})", d["planned"]))
    for name, t in sources:
        out += f"<h3>Windows in the {esc(name)}</h3>" + bars(
            [(k, v, f"{v} capsules") for k, v in t["windows"].items()], f"windows {name}"
        )
        points = [
            {
                "x": r["N"],
                "y": r["M"],
                "series": r.get("category") or "",
                "tip": f"{r['name']}\nM={r['M']} K={r['K']} N={r['N']}",
            }
            for r in t["shapes"]
        ]
        out += f"<h3>Shapes in the {esc(name)} (M against N, log scales)</h3>" + C.scatter(
            points, x_label="N", y_label="M", label=f"shapes {name}", log_x=True, log_y=True
        )
        strata = t["by"].get("stratum") or {}
        out += "<h3>Shape strata</h3>" + bars([(k, v, f"{v} capsules") for k, v in strata.items()], "strata")
    return out + "</section>"


def comparison_section(s: Mapping[str, Any]) -> str:
    ds = s.get("derivations") or []
    if len(ds) < 2:
        return ""
    rows = []
    for d in ds:
        req = d.get("requirements") or {}
        axes = req.get("axes") or {}
        rows.append(
            [
                esc(d["name"]),
                text_or_nr((d.get("planned") or {}).get("n")),
                text_or_nr((req.get("typed") or {}).get("n")),
                *[str(len(axes.get(a) or [])) for a, _, _ in P0.AXES],
                str(len((req.get("performance") or {}).get("classes") or [])),
            ]
        )
    headers = ["derivation", "planned", "typed instances", *[a for a, _, _ in P0.AXES], "form classes"]
    return (
        '<section id="compare"><h2>Derivations side by side</h2>'
        + table(headers, rows, tuple(range(1, 11)))
        + "</section>"
    )


def render(s: Mapping[str, Any], *, status: str = "") -> str:
    corpus = s.get("corpus")
    sections, nav = [], []
    gen = (s.get("coverage") or {}).get("generation")
    tiles = [
        tile("Derivations", str(len(s.get("derivations") or [])), ""),
        tile("Written capsules (public)", text_or_nr((corpus or {}).get("public", {}).get("n")) if corpus else NR, ""),
        tile("Hidden capsules", text_or_nr((corpus or {}).get("hidden", {}).get("n")) if corpus else NR, "count only"),
        tile(
            "Generation",
            text_or_nr((gen or {}).get("status")) if gen else NR,
            f"{text_or_nr(gen.get('capsules_written'))} written, {text_or_nr(gen.get('omitted'))} omitted"
            if gen
            else "coverage/generation.json",
        ),
    ]
    for phase in ("phase1", "phase2"):
        cohort = (s.get("coverage") or {}).get(phase)
        tiles.append(tile(f"{phase} coverage", text_or_nr((cohort or {}).get("status")) if cohort else NR, ""))
    sections.append(
        '<section id="overview"><h2>Overview</h2>'
        + mode_banner(bool(s.get("operator_private")))
        + '<div class="tiles">'
        + "".join(tiles)
        + "</div></section>"
    )
    nav.append(("overview", "Overview"))
    inv = '<section id="inventory"><h2>Capsule inventory</h2>'
    if corpus and corpus.get("public", {}).get("n"):
        inv += f'<p class="sub">Written corpus {mono(corpus["root"])}.</p>' + inventory_block(
            corpus["public"], corpus["rows"], "Written corpus"
        )
    for d in s.get("derivations") or ():
        inv += f"<h3>Planned by {esc(d['name'])} (synthesis.yaml)</h3>" + inventory_block(
            d.get("planned"), d.get("planned_rows") or [], f"Planned ({d['name']})"
        )
    if not corpus and not s.get("derivations"):
        inv += f"<p>Capsules: {NR}.</p>"
    sections.append(inv + "</section>")
    nav.append(("inventory", "Inventory"))
    for i, d in enumerate(s.get("derivations") or ()):
        sections.append(requirements_section(d, s.get("coverage"), i))
        nav.append((f"req{i}", f"Requirements {d['name']}"))
    sections.append(cohorts_section(s))
    nav.append(("cohorts", "Cohorts"))
    sections.append(hidden_section(s))
    nav.append(("hidden", "Hidden"))
    sections.append(distributions_section(s))
    nav.append(("distributions", "Distributions"))
    compare = comparison_section(s)
    if compare:
        sections.append(compare)
        nav.append(("compare", "Compare"))
    if gen:
        failures = [[text_or_nr(f.get("capsule")), text_or_nr(f.get("reason"))] for f in gen.get("failures") or ()]
        omitted = [
            [text_or_nr(o.get("capsule")), text_or_nr(o.get("status")), text_or_nr(o.get("reason"))]
            for o in gen.get("omitted_rows") or ()
        ]
        sections.append(
            '<section id="generation"><h2>Generation record</h2>'
            f"<p>qualification {text_or_nr(gen.get('qualification'))}; mode {text_or_nr(gen.get('mode'))}; "
            f"performance materialization errors {text_or_nr(gen.get('performance_errors'))}; unbuilt roster "
            f"{gen.get('unbuilt_roster')}.</p>"
            + details(f"Failures ({len(failures)})", table(["capsule", "reason"], failures))
            + details(f"Omitted ({len(omitted)})", table(["capsule", "status", "reason"], omitted))
            + "</section>"
        )
        nav.append(("generation", "Generation"))
    sections.append(inventory_section(s.get("inventory") or []))
    nav.append(("records", "Records"))
    mode = "operator-private" if s.get("operator_private") else "public"
    sub = f"Phase 0 corpus and coverage &middot; {esc(mode)} mode &middot; target {text_or_nr(s.get('target'))}"
    if status:
        sub += f" &middot; {status}"
    return page(f"Phase 0 {s['run_id']}", esc(s["run_id"]), sub, nav, "".join(sections), s["generated"])


__all__ = ["matrix", "mode_banner", "render"]
