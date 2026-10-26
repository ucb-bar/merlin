#!/usr/bin/env python3
"""Generate the author's ``pass-library`` skill from the optimization catalog and the compiler source.

The catalog (``docs/design/joint_optimization_catalog.json``) lists every reusable mechanism with its
facet, required traits, applicability, adapter obligations and evidence. The phase 1 author never read
it as a menu of callable passes, so this generator turns it into an index skill
(``author_skills/pass-library/SKILL.md``) and one reference file per facet
(``author_skills/pass-library/reference/<facet>.md``). Each entry is joined with the compiler source
(the symbol's signature and the first sentence of its docstring, read with ``ast``) and with one
repository test that calls the symbol, found by scanning the test trees.

Busy resources map to facets through ``mechanism_hints.resource_facets``/``bound_facets``, which is the
function the performance brief itself uses. They are not copied here, so the two cannot drift apart.
Mechanisms in the ``fusion`` facet, or whose own id names fusing, are tagged ``fusion``.

Every emitted catalog, source or test string is screened against the no-target-name gate's roster
(targets and substrates) and withheld when it names one. Target facts stay in the author's own
workspace. Run::

    python build_tools/scripts/gen_author_pass_library.py           # regenerate the committed files
    python build_tools/scripts/gen_author_pass_library.py --check   # exit 1 if they are stale
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_no_target_name as _names  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CATALOG = "docs/design/joint_optimization_catalog.json"
OUTPUT = "packages/merlin-experiments/src/merlin_experiments/phase1/author_skills/pass-library"
TEST_ROOTS = ("merlin/tests", "packages")
WITHHELD = "(withheld: names a specific target; derive it from your workspace facts)"
#: Generic timing-model resource names, one per kind the hint mapping distinguishes.
RESOURCE_PROBES = (
    ("local storage busy", "accelerator.storage.<unit>"),
    ("memory system busy", "memory.<level>"),
    ("accelerator load/store lane busy", "accelerator.issue.load"),
    ("accelerator instruction issue busy", "accelerator.issue.<lane>"),
    ("host instruction issue busy", "cpu.issue"),
    ("accelerator execution busy", "accelerator.controller.execute"),
    ("other host work busy", "cpu.<unit>"),
)
BOUND_PROBES = (("`bound_by` read/write requests", "read requests"), ("`bound_by` compute", "compute"))


def _forbidden() -> frozenset[str]:
    names = frozenset(name.lower() for name in (_names.TARGET_NAMES | _names.SUBSTRATE_NAMES))
    if not names:
        raise RuntimeError("target-name roster is empty; refusing to emit unscreened text")
    return names


def names_target(text: str, forbidden: frozenset[str]) -> bool:
    low = str(text).lower()
    return any(_names._contains_identifier(low, name) for name in forbidden)


def _screen(text, forbidden) -> str:
    text = " ".join(str(text or "").split())
    return WITHHELD if names_target(text, forbidden) else text


def _resource_facets() -> list[tuple[str, str, tuple[str, ...]]]:
    from merlin_experiments.phase1.feedback import mechanism_hints

    rows = []
    for label, probe in RESOURCE_PROBES:
        concrete = probe.replace("<unit>", "unit").replace("<level>", "level").replace("<lane>", "lane")
        rows.append((label, f"`{probe}`", mechanism_hints.resource_facets(concrete) or ()))
    for label, bound in BOUND_PROBES:
        rows.append((label, "", mechanism_hints.bound_facets(bound) or ()))
    return rows


def _module_relative(path: str) -> str:
    """``merlin/...`` as staged under ``submission/compiler_library``, from a repository source path."""
    parts = Path(path).parts
    return "/".join(parts[parts.index("src") + 1 :]) if "src" in parts else path


def _first_sentence(doc: str | None) -> str:
    text = " ".join((doc or "").split())
    for end in (". ", ".\n"):
        if end in text:
            return text.split(end, 1)[0] + "."
    return text


def _symbol(root: Path, path: str, symbol: str) -> dict:
    """Signature and docstring of a top-level def/class (or a ``NAME = ...`` binding) in ``path``."""
    source = root / path
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as error:
        return {"status": f"source unreadable: {type(error).__name__}"}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            args = node.args
            names = [a.arg for a in (*args.posonlyargs, *args.args)]
            if args.vararg:
                names.append("*" + args.vararg.arg)
            elif args.kwonlyargs:
                names.append("*")
            names += [a.arg for a in args.kwonlyargs]
            if args.kwarg:
                names.append("**" + args.kwarg.arg)
            return {
                "status": "found",
                "kind": "function",
                "signature": f"{symbol}({', '.join(names)})",
                "doc": _first_sentence(ast.get_docstring(node)),
            }
        if isinstance(node, ast.ClassDef) and node.name == symbol:
            fields = [s.target.id for s in node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
            signature = f"{symbol}({', '.join(fields)})" if fields else f"class {symbol}"
            return {
                "status": "found",
                "kind": "class",
                "signature": signature,
                "doc": _first_sentence(ast.get_docstring(node)),
            }
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == symbol for t in node.targets):
            return {"status": "found", "kind": "binding", "signature": symbol, "doc": ""}
    return {"status": "symbol not found in current source"}


def _test_files(root: Path) -> list[tuple[str, str]]:
    files = []
    for base in TEST_ROOTS:
        for path in sorted((root / base).rglob("test_*.py")):
            relative = path.relative_to(root).as_posix()
            if "/tests/" not in "/" + relative or "/src/" in relative:
                continue
            try:
                files.append((relative, path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    return files


def _imports(text: str, module: str, symbol: str) -> bool:
    parent, _, leaf = module.rpartition(".")
    return (
        f"from {module} import" in text
        or f"import {module}" in text
        or (bool(parent) and f"from {parent} import" in text and leaf in text)
    ) and symbol in text


def _token_at(line: str, symbol: str) -> bool:
    start = 0
    while (index := line.find(symbol, start)) >= 0:
        before = line[index - 1] if index else " "
        after = line[index + len(symbol)] if index + len(symbol) < len(line) else " "
        if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
            return True
        start = index + 1
    return False


def find_test(files, module: str, symbol: str, forbidden) -> dict | None:
    """One test that imports the module and calls the symbol: name-matched file first, then path order."""
    leaf = module.rpartition(".")[2]
    candidates = [
        (path, text) for path, text in files if not names_target(path, forbidden) and _imports(text, module, symbol)
    ]
    candidates.sort(key=lambda item: (not Path(item[0]).name.startswith(f"test_{leaf}"), item[0]))
    for path, text in candidates:
        lines = text.splitlines()
        calls = [
            line
            for line in lines
            if _token_at(line, symbol)
            and f"{symbol}(" in line
            and not line.lstrip().startswith(("from ", "import ", "def ", "class ", "#"))
        ]
        if calls:
            snippet = " ".join(calls[0].split())
            snippet = snippet if len(snippet) <= 160 else snippet[:157] + "..."
            return {"path": path, "call": "" if names_target(snippet, forbidden) else snippet}
    module_tests = sorted(
        (
            path
            for path, text in files
            if not names_target(path, forbidden) and (f"from {module} import" in text or f"import {module}" in text)
        ),
        key=lambda path: (not Path(path).name.startswith(f"test_{leaf}"), path),
    )
    return {"path": module_tests[0], "call": "", "module_only": True} if module_tests else None


def is_fusion(row: dict) -> bool:
    if row.get("facet") == "fusion":
        return True
    words = "".join(c if c.isalnum() else " " for c in str(row.get("id")).lower()).split()
    return any(word.startswith(("fuse", "fusi")) for word in words)


def _entry(row: dict, root: Path, files, forbidden) -> list[str]:
    compiler = row.get("compiler") or {}
    module, symbol = compiler.get("module", ""), compiler.get("symbol", "")
    found = _symbol(root, compiler.get("path", ""), symbol)
    test = find_test(files, module, symbol, forbidden)
    tags = ["fusion"] if is_fusion(row) else []
    lines = [f"### {row['id']}" + (f"  [{', '.join(tags)}]" if tags else ""), ""]
    lines.append(f"- **call**: `{module}:{symbol}` (`{_module_relative(compiler.get('path', ''))}`)")
    if found.get("status") == "found":
        lines.append(f"- **signature**: `{_screen(found['signature'], forbidden)}`")
        if found.get("doc"):
            lines.append(f"- **source says**: {_screen(found['doc'], forbidden)}")
    else:
        lines.append(f"- **source**: {found.get('status')}")
    lines += [
        f"- **purpose**: {_screen(row.get('mechanism'), forbidden)}",
        f"- **applies when**: {_screen(row.get('applicability'), forbidden)}",
        f"- **required traits**: {_screen(', '.join(row.get('required_traits') or ()), forbidden)}",
        f"- **your target supplies**: {_screen(row.get('remaining_adapter_or_feedback'), forbidden)}",
        f"- **evidence to request**: {_screen(row.get('evidence_to_request'), forbidden)}",
        f"- **access**: {_screen(compiler.get('access'), forbidden)}",
    ]
    if test and test.get("module_only"):
        lines.append(f"- **test of its module** (none calls this symbol directly): `{test['path']}`")
    elif test:
        lines.append(f"- **test that calls it**: `{test['path']}`" + (f": `{test['call']}`" if test["call"] else ""))
    else:
        lines.append("- **test that calls it**: none found")
    return lines + [""]


def render(root: Path = ROOT) -> dict[str, str]:
    """``{relative path under the skill: text}`` for the index and every facet reference."""
    forbidden = _forbidden()
    raw = (root / CATALOG).read_bytes()
    catalog = json.loads(raw)
    mechanisms = [row for row in catalog["mechanisms"] if isinstance(row, dict) and row.get("id")]
    facets: dict[str, list[dict]] = {}
    for row in mechanisms:
        facets.setdefault(row["facet"], []).append(row)
    files = _test_files(root)
    provenance = (
        f"Generated by `build_tools/scripts/gen_author_pass_library.py` from `{CATALOG}` "
        f"(sha256 {hashlib.sha256(raw).hexdigest()[:16]}) and the compiler source; do not edit."
    )
    out: dict[str, str] = {}
    for facet, rows in sorted(facets.items()):
        body = [
            f"# Facet: {facet}",
            "",
            provenance,
            "",
            "Paths are under `submission/compiler_library/`. Read the call's docstring and the test line "
            "before using it; the `scheduling-api` skill shows how to read a contract and call a pass.",
            "",
        ]
        for row in rows:
            body += _entry(row, root, files, forbidden)
        out[f"reference/{facet}.md"] = "\n".join(body).rstrip() + "\n"
    resource_rows = ["| busy resource / limit | timing-model name | facets to open, in order |", "|---|---|---|"]
    for label, probe, mapped in _resource_facets():
        resource_rows.append(f"| {label} | {probe} | {', '.join(mapped) or '-'} |")
    facet_rows = ["| facet | mechanisms (fusion-tagged marked *) | reference |", "|---|---|---|"]
    for facet, rows in sorted(facets.items()):
        ids = ", ".join(row["id"] + ("*" if is_fusion(row) else "") for row in rows)
        facet_rows.append(f"| {facet} | {ids} | `reference/{facet}.md` |")
    fusion = sorted(row["id"] for row in mechanisms if is_fusion(row))
    index = [
        "---",
        "name: pass-library",
        "description: Index of the reusable compiler mechanisms (catalog of "
        f"{len(mechanisms)}), each with its library call, applicability, the adapter your target must supply, "
        "the evidence to request and a test that calls it. Open it when a performance brief names a "
        "mechanism (`suggested_mechanisms`), when a unit is busy (`dominant_modeled_resource`) or a "
        "`bound_by` limit holds a case back, or when you look for a library pass instead of writing a "
        "schedule by hand.",
        "---",
        "",
        "# Pass library",
        "",
        provenance,
        "",
        "1. Find the facets for what limits the case (table below; the same mapping the performance brief uses).",
        "2. Open `reference/<facet>.md`; each entry names the call, its signature, when it applies, what your "
        "target supplies and a test line that calls it.",
        "3. `OPTIMIZATION_SUGGESTIONS.json` in your workspace says which mechanisms your target's traits "
        "support; `python optimization_suggest.py --ir <source> --output <file>` checks workload prerequisites.",
        "4. Call it through `scheduling-api`; compare cycles on the same cases before keeping it.",
        "",
        "## Busy resource to facet",
        "",
        *resource_rows,
        "",
        "Within a facet, mechanisms whose entry names the busy storage unit (for example an accumulator) "
        "address that unit's work directly.",
        "",
        "## Facets",
        "",
        *facet_rows,
        "",
        f"Fusion-tagged ({len(fusion)}): {', '.join(fusion)}. The `fusion` skill covers using them.",
    ]
    out["SKILL.md"] = "\n".join(index) + "\n"
    return out


def stale(root: Path = ROOT) -> list[str]:
    """Paths whose committed text differs from a fresh render, plus committed files no longer rendered."""
    expected = render(root)
    folder = root / OUTPUT
    problems = [
        path
        for path, text in expected.items()
        if not (folder / path).is_file() or (folder / path).read_text(encoding="utf-8") != text
    ]
    present = {p.relative_to(folder).as_posix() for p in folder.rglob("*.md")} if folder.is_dir() else set()
    return problems + sorted(present - set(expected))


def write(root: Path = ROOT) -> list[str]:
    expected = render(root)
    folder = root / OUTPUT
    for path, text in expected.items():
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        (folder / path).write_text(text, encoding="utf-8")
    removed = []
    for old in sorted(folder.rglob("*.md")):
        if old.relative_to(folder).as_posix() not in expected:
            old.unlink()
            removed.append(old.relative_to(folder).as_posix())
    return sorted(expected) + [f"removed {path}" for path in removed]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--check" in argv:
        problems = stale()
        if problems:
            print("pass-library skill is stale; run build_tools/scripts/gen_author_pass_library.py:")
            for path in problems:
                print(f"  - {OUTPUT}/{path}")
            return 1
        print("pass-library skill is current")
        return 0
    for line in write():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
