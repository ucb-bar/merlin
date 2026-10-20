"""Conservative package source closure for certificate component identities."""
from __future__ import annotations

from pathlib import Path

_DYNAMIC_IMPORT = frozenset({
    "__import__", "import_module", "load_module", "exec_module",
    "spec_from_file_location", "module_from_spec", "iter_modules",
})

def _manifest_strings(node):
    """Every string anywhere in the manifest, however nested. Structural walk, no pattern matching."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _manifest_strings(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _manifest_strings(v)


def _under(sub: Path, token: str):
    """``token`` resolved to an existing path under the submission root, or None.

    Tokens arrive from the manifest, so they may be doubly rooted (``submission/mlir_oot/opt``) exactly as
    ``oot_runner._resolve_argv`` already documents; that prefix is stripped when, and only when, the
    remainder resolves. Anything absolute, empty, escaping the root, or containing whitespace is not a
    package-relative path and is skipped.
    """
    s = str(token).strip().replace("\\", "/")
    if not s or s.startswith("/") or any(c.isspace() for c in s):
        return None
    for pfx in ("./submission/", "submission/", "./"):
        if s.startswith(pfx) and not (sub / s).exists():
            s = s[len(pfx) :]
    s = s.strip("/")
    if not s or s in (".", ".."):
        return None
    try:
        p = (sub / s).resolve()
        if not p.exists() or p == sub.resolve() or sub.resolve() not in p.parents:
            return None
    except OSError:  # a "path" long enough to be prose, not a path
        return None
    return p


def _files_at(p: Path, token: str = "/"):
    """The file(s) a resolved token stands for. A DIRECTORY counts only when the token was written as a
    path (it contains a separator): a bare single-segment string that happens to match a directory name is
    almost always a module or an identifier, not a data directory, and treating `"xdsl"` as a grant over
    the whole vendored package pulls that package's READMEs into the executable surface -- which then
    makes every other `.md` in the submission look like the same kind of thing as reached code."""
    if p.is_dir():
        return [x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts] if "/" in str(token) else []
    return [p] if "__pycache__" not in p.parts else []


def _command_roots(sub: Path, m: dict):
    """``{command: [entrypoint file, ...]}`` -- the submission files each declared command's argv names.

    ``{tool}`` is substituted from ``entrypoints.tool`` the way the runner substitutes it, so a package
    that references its tool through the placeholder and one that spells the path out both resolve. A
    command whose argv names NO submission file is the refusal case: we do not know what it runs.
    """
    cmds = m.get("commands")
    if not isinstance(cmds, dict) or not cmds:
        return None
    tool = ((m.get("entrypoints") or {}) if isinstance(m.get("entrypoints"), dict) else {}).get("tool")
    out = {}
    for name, spec in cmds.items():
        argv = (spec or {}).get("argv") if isinstance(spec, dict) else None
        if not isinstance(argv, (list, tuple)):
            return None
        roots = []
        for tok in argv:
            tok = str(tok)
            if "{tool}" in tok and tool:
                tok = tok.replace("{tool}", str(tool))
            if "{" in tok:  # an unsubstituted I/O placeholder: not a submission file
                continue
            p = _under(sub, tok)
            if p is not None and p.is_file():
                roots.append(p)
        if not roots:
            return None
        out[str(name)] = roots
    return out


def _py_module_file(sub: Path, dotted: str):
    p = sub.joinpath(*dotted.split("."))
    for cand in (p.with_suffix(".py"), p / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _py_imports(tree, rel_parts) -> set:
    """Dotted module names a parsed file imports, with relative imports resolved against its package."""
    pkg = list(rel_parts[:-1])
    out = set()
    for n in _ast().walk(tree):
        if isinstance(n, _ast().Import):
            for a in n.names:
                out.add(a.name)
        elif isinstance(n, _ast().ImportFrom):
            base = n.module or ""
            if n.level:
                up = pkg[: len(pkg) - (n.level - 1)] if n.level > 1 else pkg
                base = ".".join(up + ([base] if base else []))
            if base:
                out.add(base)
            for a in n.names:
                out.add(f"{base}.{a.name}" if base else a.name)
    return out


def _ast():
    import ast

    return ast


def _scan(f: Path, memo=None):
    """``(tree, string_literals, has_dynamic_import)`` for one Python source file, or None if it will not
    parse. Unparseable means unknown content, and unknown content is the refusal case."""
    if memo is not None and f in memo:
        return memo[f]
    ast = _ast()
    try:
        tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 -- not Python, or broken Python: we cannot say what it reads
        if memo is not None:
            memo[f] = None
        return None
    lits, dyn = set(), False
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            lits.add(n.value)
        elif isinstance(n, ast.Call):
            fn = n.func
            nm = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if nm in _DYNAMIC_IMPORT:
                dyn = True
    out = (tree, lits, dyn)
    if memo is not None:
        memo[f] = out
    return out


def _closure(sub: Path, roots, memo=None):
    """``(files, literals, dynamic_suffixes)`` reachable from ``roots``, or None on any unparseable file.

    Follows intra-submission imports only: a name that resolves to no file under the submission root is a
    stdlib/third-party import the package does not carry, and cannot be edited by the agent.
    """
    seen, lits, dyn_sufs = set(), set(), set()
    stack = list(roots)
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        got = _scan(f, memo)
        if got is None:
            return None
        tree, flits, dyn = got
        lits |= flits
        if dyn:
            # Bounded by what the loader ITSELF names: a suffix-shaped literal in the file that does the
            # dynamic import. Widening on every literal in the closure drags in noise ('.text', '.word')
            # from unrelated code and would make the whole tree "the same kind of thing".
            dyn_sufs |= {s for s in flits if s.startswith(".") and 2 <= len(s) <= 6 and s[1:].isalnum()}
        for mod in _py_imports(tree, f.relative_to(sub).parts):
            parts = mod.split(".")
            for i in range(1, len(parts) + 1):
                g = _py_module_file(sub, ".".join(parts[:i]))
                if g is not None and g not in seen:
                    stack.append(g)
    return seen, lits, dyn_sufs


def _src_suffixes() -> frozenset:
    """Suffixes the runner's own integrity scan READS. A violation in one of these fails the run even
    when nothing imports the file, so such a file is never inert. Taken from the runner, not restated."""
    try:
        from merlin.targetgen.oot_runner import _SRC_SUFFIXES

        return frozenset(_SRC_SUFFIXES)
    except Exception:  # noqa: BLE001 -- runner not importable here: the closure's own suffixes still apply
        return frozenset()
