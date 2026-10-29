"""Independent transitive-closure and forbidden-construct scan of Merlin graph Lean sources.

Read-only. Strips Lean comments and string literals with a small tokenizer (no regex),
follows `import Merlin.*` edges from the selected roots, and reports every forbidden
token occurrence with file:line. Usage:

    python -I lean_closure_scan.py <dir-containing-Merlin>[:<more dirs>] <out.json> <query.lean>...

where each query file is a generated instance query whose own imports are followed too.
"""

import hashlib
import json
import sys
from pathlib import Path

FORBIDDEN_WORDS = {
    "sorry", "admit", "axiom", "native_decide", "bv_decide", "implemented_by", "extern",
    "csimp", "unsafe", "opaque", "partial", "ofReduceBool", "ofReduceNat", "reduceBool",
    "decide_native", "nativeDecide", "Lean.ofReduceBool", "Lean.reduceBool", "trustCompiler",
    "debug.skipKernelTC", "skipKernelTC", "macro_rules", "elab", "elab_rules", "run_cmd",
    "run_tac", "run_elab", "initialize", "builtin_initialize", "@[irreducible]", "set_option",
    "unsafeCast", "private", "noncomputable", "Classical", "instance",
}
# Tokens that are reported but are not violations by themselves (context needed).
CONTEXT_ONLY = {"set_option", "private", "noncomputable", "Classical", "instance", "partial",
                "opaque", "initialize"}


def strip(text):
    """Return text with comments and string/char literals blanked, preserving newlines."""
    out = []
    i, n, depth = 0, len(text), 0
    while i < n:
        two = text[i:i + 2]
        if depth == 0 and two == "--":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if two == "/-":
            depth += 1
            out.append("  ")
            i += 2
            continue
        if depth and two == "-/":
            depth -= 1
            out.append("  ")
            i += 2
            continue
        if depth:
            out.append("\n" if text[i] == "\n" else " ")
            i += 1
            continue
        if text[i] == '"':
            out.append(" ")
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    out.append(" ")
                    i += 1
                out.append("\n" if i < n and text[i] == "\n" else " ")
                i += 1
            out.append(" ")
            i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def tokens(line):
    word = []
    for ch in line + " ":
        if ch.isalnum() or ch in "_.'@[]!?":
            word.append(ch)
        else:
            if word:
                yield "".join(word)
                word = []


def imports(text):
    result = []
    for line in strip(text).splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "import":
            result.extend(parts[1:])
    return result


def scan_file(path):
    text = path.read_text()
    hits = []
    for number, line in enumerate(strip(text).splitlines(), 1):
        for token in tokens(line):
            bare = token.strip("@[]!?")
            parts = set(bare.split(".")) | {bare}
            for word in FORBIDDEN_WORDS:
                if word in parts or token == word:
                    hits.append({"line": number, "token": word, "text": text.splitlines()[number - 1].strip()[:160]})
    return hits


def main():
    source_dirs, out = [Path(x) for x in sys.argv[1].split(":")], Path(sys.argv[2])
    queries = [Path(p) for p in sys.argv[3:]]
    seen, order, external = {}, [], set()
    pending = []
    for query in queries:
        pending.extend(imports(query.read_text()))
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        candidates = [d / (module.replace(".", "/") + ".lean") for d in source_dirs]
        found = [c for c in candidates if c.exists()]
        if not (module.startswith("Merlin.") or module.startswith("MerlinVerify.")) or not found:
            external.add(module)
            seen[module] = None
            continue
        path = found[0]
        seen[module] = path
        order.append(module)
        pending.extend(imports(path.read_text()))
    files = []
    for module in sorted(order):
        path = seen[module]
        raw = path.read_bytes()
        hits = scan_file(path)
        files.append({
            "module": module, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "lines": raw.count(b"\n"),
            "violation_hits": [h for h in hits if h["token"] not in CONTEXT_ONLY],
            "context_hits": [h for h in hits if h["token"] in CONTEXT_ONLY and h["token"] != "set_option"],
            "set_option": sorted({h["text"] for h in hits if h["token"] == "set_option"}),
        })
    query_rows = []
    for query in queries:
        raw = query.read_bytes()
        hits = scan_file(query)
        query_rows.append({"path": str(query), "sha256": hashlib.sha256(raw).hexdigest(),
                           "violation_hits": [h for h in hits if h["token"] not in CONTEXT_ONLY],
                           "context_hits": [h for h in hits if h["token"] in CONTEXT_ONLY and h["token"] != "set_option"],
                           "set_option": sorted({h["text"] for h in hits if h["token"] == "set_option"})})
    result = {
        "external_imports": sorted(external),
        "merlin_modules": len(files),
        "violation_total": sum(len(f["violation_hits"]) for f in files) + sum(len(q["violation_hits"]) for q in query_rows),
        "queries": query_rows,
        "files": files,
    }
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("external_imports", "merlin_modules", "violation_total")}))


if __name__ == "__main__":
    main()
