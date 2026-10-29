"""Private, read-only comparison of compiler choices and their recorded evidence.

The specification supplies explicit input records. This module neither selects
compiler settings nor publishes anything to an author workspace or feedback API.
Record classifications are reading aids, not execution or performance verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

SCHEMA = "merlin.operator.compiler_comparison.v1"
MODES = frozenset({"declared", "lowering_report", "consumption_report", "schedule_recipe",
                   "native_check", "physical_measurement", "imported"})
MAX_BYTES = 4 << 20
MAX_ROWS = 12000


def _load(reference):
    path = Path(reference["path"])
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            return None, "record exceeds display bound"
        if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
            return None, "identity mismatch; values withheld"
        return json.loads(payload), None
    except (OSError, ValueError) as error:
        return None, f"not recorded: {type(error).__name__}"


def _pointer(document, pointer):
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError("record selection requires a JSON pointer")
    value = document
    for token in pointer[1:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list) and len(value) <= 48:
        for index, child in enumerate(value):
            yield from _leaves(child, f"{prefix}[{index}]")
        if not value:
            yield prefix, []
    else:
        yield prefix, value


def comparison(spec, *, operator_private=False):
    """Read explicitly pinned records; missing evidence never becomes zero or use."""
    if not operator_private:
        raise ValueError("compiler comparisons require explicit operator_private access")
    if spec.get("schema") != SCHEMA or spec.get("visibility") != "operator_private":
        raise ValueError("expected an explicitly private comparison specification")
    profiles = spec.get("profiles", [])
    if not profiles or len(profiles) > 32 or len({p["id"] for p in profiles}) != len(profiles):
        raise ValueError("comparison profiles must be nonempty, bounded and uniquely named")
    report = {"schema": SCHEMA, "visibility": "operator_private",
              "title": spec.get("title", "Compiler comparison"), "profiles": [],
              "notes": list(spec.get("notes", [])), "matrix": [], "inventory": []}
    matrix = {}
    for profile in profiles:
        current = {key: profile.get(key) for key in ("id", "label", "role", "notes", "workspace")}
        current["records"] = []
        for document in profile.get("documents", []):
            mode = document["mode"]
            if mode not in MODES:
                raise ValueError("unknown record classification")
            source = dict(document["reference"])
            data, error = _load(source)
            if error is None:
                try:
                    selected = _pointer(data, document.get("pointer", ""))
                except (KeyError, IndexError, TypeError, ValueError):
                    error, selected = "not recorded: selected field absent", None
            else:
                selected = None
            record = {"name": document["name"], "stage": document["stage"], "mode": mode,
                      "source": source, "pointer": document.get("pointer", ""),
                      "status": error or "record_identity_matches", "data": selected}
            current["records"].append(record)
            report["inventory"].append({"profile": profile["id"], **{k: record[k] for k in
                                       ("name", "source", "pointer", "status")}})
            if error or document.get("matrix", True) is False:
                continue
            for field, value in _leaves(selected):
                key = (document["stage"], document["name"], field)
                row = matrix.setdefault(key, {"stage": key[0], "mechanism": key[1],
                                              "field": key[2], "cells": {}})
                if profile["id"] in row["cells"]:
                    raise ValueError("ambiguous repeated comparison field")
                row["cells"][profile["id"]] = {"mode": mode, "value": value, "source": source,
                                               "pointer": record["pointer"]}
                if len(matrix) > MAX_ROWS:
                    raise ValueError("explicit comparison exceeds row bound")
        imports_root = profile.get("imports_root")
        if imports_root:
            from .console.io import Reader
            from .console.sources.library import tree_imports

            imported = tree_imports(Reader(), Path(imports_root))
            if imported is not None:
                names, direct = imported
                for module in sorted(direct):
                    if not module.startswith("merlin."):
                        continue
                    key = ("Construction", "Library imports", module)
                    row = matrix.setdefault(key, {"stage": key[0], "mechanism": key[1],
                                                  "field": key[2], "cells": {}})
                    row["cells"][profile["id"]] = {"mode": "imported", "value": names.get(module, []),
                                                   "source": {"path": imports_root}, "pointer": ""}
            else:
                report["inventory"].append({"profile": profile["id"], "name": "Library imports",
                                            "status": "not recorded", "source": {"path": imports_root}})
        report["profiles"].append(current)
    report["matrix"] = list(matrix.values())
    return report


def render(report):
    """Self-contained operator page: no fetches, tracking or external scripts."""
    escape = lambda value: html.escape(str(value), quote=True)
    compact = lambda value: json.dumps(value, ensure_ascii=True, sort_keys=True)
    profiles = report["profiles"]
    header = "".join(f"<th>{escape(p['label'])}<small>{escape(p['role'])}</small></th>" for p in profiles)
    rows = []
    for row in report["matrix"]:
        cells = []
        for profile in profiles:
            cell = row["cells"].get(profile["id"])
            if cell is None:
                cells.append('<td class="missing">Not recorded</td>')
                continue
            value = compact(cell["value"])
            detail = escape(compact(cell))
            cells.append(f'<td><span class="tag {escape(cell["mode"])}">{escape(cell["mode"])}</span>'
                         f'<details><summary>{escape(value[:220])}{"…" if len(value)>220 else ""}</summary>'
                         f'<pre>{detail}</pre></details></td>')
        rows.append(f'<tr><td><small>{escape(row["stage"])}</small><b>{escape(row["mechanism"])}</b>'
                    f'<code>{escape(row["field"])}</code></td>{"".join(cells)}</tr>')
    records = []
    for profile in profiles:
        body = "".join(f'<details><summary>{escape(r["stage"])} / {escape(r["name"])} · '
                       f'{escape(r["mode"])} · {escape(r["status"])}</summary>'
                       f'<pre>{escape(json.dumps(r,indent=2))}</pre></details>' for r in profile["records"])
        records.append(f'<section><h2>{escape(profile["label"])}</h2>'
                       f'<p>{escape(profile.get("notes") or "")}</p>{body}</section>')
    notes = "".join(f"<li>{escape(note)}</li>" for note in report["notes"])
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='referrer' content='no-referrer'><title>Private compiler comparison</title>"
            "<style>body{font:14px system-ui;background:#101620;color:#e3eaf4;margin:24px}"
            "h1{font-size:26px}p,li{max-width:1100px;line-height:1.6}small,code{display:block;color:#aebed3}"
            ".private{color:#ffcc78;font-weight:700}.scroll{overflow:auto;max-height:70vh}"
            "table{border-collapse:collapse;width:100%}td,th{padding:10px;min-width:220px;max-width:440px;"
            "border:1px solid #354355;vertical-align:top;text-align:left}th{background:#202f42;position:sticky;top:0}"
            "td:first-child{background:#192334}code{font-size:12px;overflow-wrap:anywhere}"
            "details{margin:6px 0}summary{cursor:pointer;overflow-wrap:anywhere}pre{white-space:pre-wrap;"
            "overflow-wrap:anywhere;font-size:12px}section{margin-top:30px}.missing{color:#7d8c9d}"
            ".tag{font-size:11px;padding:2px 6px;border-radius:4px;background:#26354b;color:#b7cff1}"
            ".physical_measurement{background:#174839;color:#a4efce}.declared,.imported{background:#493c25;color:#f5d7a4}"
            "input{padding:10px;width:460px;max-width:85%;background:#1b283a;color:white;border:1px solid #586c89}"
            "</style></head><body><p class='private'>OPERATOR PRIVATE — excluded from author inputs</p>"
            f"<h1>{escape(report['title'])}</h1>"
            "<p>Declared options, library imports, lowering reports, consumed products and measurements are separate. "
            "A matching record hash confirms which record was read; it does not prove that a pass executed. "
            "Missing telemetry means <b>not recorded</b>. This is not an exhaustive pass trace or a performance ranking.</p>"
            f"<ul>{notes}</ul><input id='filter' placeholder='Filter stage, pass, option, schedule or value'>"
            f"<div class='scroll'><table><thead><tr><th>Mechanism / recorded field</th>{header}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>{''.join(records)}"
            "<script>document.getElementById('filter').addEventListener('input',function(){"
            "const q=this.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>{"
            "r.hidden=!r.textContent.toLowerCase().includes(q)})});</script></body></html>")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-private", action="store_true", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    output = args.output.resolve()
    for profile in spec.get("profiles", []):
        for key in ("workspace", "imports_root"):
            root = profile.get(key)
            if root and output.is_relative_to(Path(root).resolve()):
                raise ValueError("private output cannot be inside an observed author/source tree")
        for document in profile.get("documents", []):
            if output == Path(document["reference"]["path"]).resolve():
                raise ValueError("private output would overwrite an observed input")
    if output == args.spec.resolve():
        raise ValueError("private output would overwrite its specification")
    report = comparison(spec, operator_private=args.operator_private)
    output.write_text(render(report))
    print(json.dumps({"output": str(output), "visibility": report["visibility"],
                      "profiles": len(report["profiles"]), "rows": len(report["matrix"])}))


if __name__ == "__main__":
    main()
