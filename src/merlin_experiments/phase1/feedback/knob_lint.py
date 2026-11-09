"""Find the choices an author's compiler already switches on, so they can be declared and timed.

Measured on one attempt: the compiler defaulted an internal string policy to its slower value (a
full-device fence after every output tile) and carried the faster alternative behind the same switch,
but never compared the two; nothing pointed at the switch. A choice declared in
``submission/TUNING_KNOBS.json`` (:mod:`merlin.compile.tuning_knobs`) is timed value by value on idle
CPUs (:mod:`.idle_tuning`), so the lint names undeclared ones.

Detection reads the source with :mod:`ast` and never imports or edits it:

* an **option read** ``<mapping>.get("<name>", "<default>")`` with string literals, whose value flows
  into a variable or into a parameter of a function of the same tree that compares it with string
  literals (``==``, ``!=``, ``in``, ``not in``);
* a **parameter** whose name is ``policy``/``mode``/``strategy`` or ends with ``_policy``, ``_mode``
  or ``_strategy``, has a string default and is compared with string literals in its function.

A candidate is reported when at least two distinct literal values are visible, with its location and
the values. Knobs already declared are listed apart. Advisory: the author decides what to declare.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

SCHEMA = "merlin.tuning_knob_candidates.v1"
DECLARATION = "TUNING_KNOBS.json"
SUFFIXES = ("_policy", "_mode", "_strategy")
BARE = frozenset(suffix[1:] for suffix in SUFFIXES)
#: Bounds on what one scan reads.
MAX_FILES = 1000
MAX_FILE_BYTES = 2 << 20
LIMIT = 12
_IGNORED = frozenset({"build", "__pycache__", ".git", "selfcheck_out", "node_modules"})
_CACHE: dict[str, tuple[tuple, dict]] = {}

HOW = (
    "Declare a candidate to have idle CPUs time each value: replace the read with "
    '`knob("NAME", default=DEFAULT, values=(...))` from merlin.compile.tuning_knobs and declare it in '
    "submission/TUNING_KNOBS.json (skill ab-library-pass, step 3); once declared, idle tuning measures "
    "every value on the cases with the most headroom."
)


def _switch_name(name: str) -> bool:
    return name in BARE or name.endswith(SUFFIXES)


def _literals(node) -> set[str]:
    """Non-empty string literals of a constant or a literal collection (an empty string is no value)."""
    items = node.elts if isinstance(node, (ast.Tuple, ast.List, ast.Set)) else [node]
    return {
        item.value for item in items if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value
    }


def _option_read(node) -> tuple[str, str] | None:
    """``(name, default)`` of ``x.get("name", "default")``, else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and len(node.args) == 2
        and not node.keywords
    ):
        name, default = node.args
        if (
            isinstance(name, ast.Constant)
            and isinstance(name.value, str)
            and isinstance(default, ast.Constant)
            and isinstance(default.value, str)
        ):
            return name.value, default.value
    return None


def _callee(node) -> str | None:
    func = node.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


class _Module(ast.NodeVisitor):
    """Per function: literal comparisons by variable, parameters, option reads and where they flow."""

    def __init__(self, path: str):
        self.path, self.stack = path, ["<module>"]
        self.compares: dict[tuple[str, str], set[str]] = {}
        self.functions: dict[str, list[tuple[str, str | None]]] = {}
        self.reads: list[dict] = []

    def visit_FunctionDef(self, node):
        args = [*node.args.posonlyargs, *node.args.args]
        defaults = [None] * (len(args) - len(node.args.defaults)) + list(node.args.defaults)
        pairs = list(zip(args, defaults)) + list(zip(node.args.kwonlyargs, node.args.kw_defaults))
        params = [
            (arg.arg, default.value if isinstance(default, ast.Constant) and isinstance(default.value, str) else None)
            for arg, default in pairs
        ]
        self.functions.setdefault(node.name, params)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Compare(self, node):
        sides = [node.left, *node.comparators]
        if all(isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops):
            names = [side.id for side in sides if isinstance(side, ast.Name)]
            values = set().union(*(_literals(side) for side in sides))
            for name in names:
                if values:
                    self.compares.setdefault((self.stack[-1], name), set()).update(values)
        self.generic_visit(node)

    def visit_Assign(self, node):
        read = _option_read(node.value)
        if read is not None:
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            self._read(node.value, read, variable=targets[0] if targets else None)
        self.generic_visit(node)

    def visit_Call(self, node):
        callee = _callee(node)
        for index, arg in enumerate(node.args):
            read = _option_read(arg)
            if read is not None:
                self._read(arg, read, callee=callee, position=index)
        for keyword in node.keywords:
            read = _option_read(keyword.value)
            if read is not None and keyword.arg:
                self._read(keyword.value, read, callee=callee, keyword=keyword.arg)
        self.generic_visit(node)

    def _read(self, node, read, **flow):
        if any(r["node"] is node for r in self.reads):
            return
        self.reads.append(
            {"node": node, "name": read[0], "default": read[1], "function": self.stack[-1], "line": node.lineno, **flow}
        )


def _library_package() -> str:
    """The harness's own top-level package; a copy of it inside the tree is the shipped library."""
    import merlin

    return merlin.__name__.split(".")[0]


def _files(root: Path) -> list[Path]:
    found, library = [], _library_package()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).parts
        if any(part in _IGNORED or part.startswith(".") or part == library for part in relative[:-1]):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        found.append(path)
        if len(found) >= MAX_FILES:
            break
    return found


def declared(root: Path) -> set[str]:
    try:
        document = json.loads((root / DECLARATION).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    knobs = document.get("knobs") if isinstance(document, dict) else None
    return set(knobs) if isinstance(knobs, dict) else set()


def scan(root) -> dict:
    """Candidates in the source tree ``root`` (module docstring)."""
    root = Path(root)
    modules: list[_Module] = []
    for path in _files(root):
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except (OSError, SyntaxError, ValueError):
            continue
        module = _Module(path.relative_to(root).as_posix())
        module.visit(tree)
        modules.append(module)
    compares: dict[tuple[str, str], set[str]] = {}
    functions: dict[str, list[tuple[str, str | None]]] = {}
    for module in modules:
        for key, values in module.compares.items():
            compares.setdefault(key, set()).update(values)
        for name, params in module.functions.items():
            functions.setdefault(name, params)
    candidates, consumed = {}, set()
    for module in modules:
        for read in module.reads:
            values, seen = {read["default"]}, []
            if read.get("variable"):
                values |= compares.get((read["function"], read["variable"]), set())
            params = functions.get(read.get("callee") or "", [])
            parameter = read.get("keyword")
            if parameter is None and read.get("position") is not None and read["position"] < len(params):
                parameter = params[read["position"]][0]
            if parameter is not None and params:
                default = dict(params).get(parameter)
                values |= compares.get((read["callee"], parameter), set()) | ({default} if default else set())
                seen.append(f"{read['callee']}({parameter})")
                consumed.add((read["callee"], parameter))
            entry = candidates.setdefault(
                read["name"],
                {
                    "name": read["name"],
                    "kind": "option_read",
                    "default": read["default"],
                    "locations": [],
                    "observed_values": set(),
                    "compared_in": [],
                },
            )
            entry["locations"].append(f"{module.path}:{read['line']}")
            entry["observed_values"] |= values
            entry["compared_in"] += [item for item in seen if item not in entry["compared_in"]]
    for module in modules:
        for function, params in module.functions.items():
            for parameter, default in params:
                if not default or not _switch_name(parameter) or (function, parameter) in consumed:
                    continue
                values = {default} | compares.get((function, parameter), set())
                key = f"{function}({parameter})"
                if key not in candidates:
                    candidates[key] = {
                        "name": key,
                        "kind": "parameter",
                        "default": default,
                        "locations": [module.path],
                        "observed_values": values,
                        "compared_in": [],
                    }
    already = declared(root)
    found, known = [], []
    for entry in candidates.values():
        if len(entry["observed_values"]) < 2:
            continue
        entry["observed_values"] = sorted(entry["observed_values"])
        entry["locations"] = entry["locations"][:4]
        (known if entry["name"] in already else found).append(entry)
    found.sort(key=lambda e: (e["kind"] != "option_read", not _switch_name(e["name"]), e["name"]))
    return {
        "schema": SCHEMA,
        "advisory_only": True,
        "root": root.name,
        "files_scanned": len(modules),
        "candidates": found[:LIMIT],
        "candidate_count": len(found),
        "already_declared": sorted(e["name"] for e in known),
        "how_to_declare": HOW,
    }


def cached_scan(root) -> dict:
    """:func:`scan`, reused while no ``.py`` file or the declaration under ``root`` changed."""
    root = Path(root)
    stamp = []
    for path in [*_files(root), root / DECLARATION]:
        try:
            stat = path.stat()
            stamp.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            continue
    key, stamp = str(root.resolve()), tuple(stamp)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    result = scan(root)
    _CACHE[key] = (stamp, result)
    return result


def attach(document: dict, workspace, *, submission: str = "submission") -> dict:
    """Add ``tuning_knob_candidates`` to a ``submission/`` response's ``performance_brief``; never fails it."""
    try:
        brief = document.get("performance_brief")
        if (
            not isinstance(brief, dict)
            or document.get("variant")
            or document.get("selfcheck_submission") not in (None, submission)
        ):
            return document
        root = Path(workspace) / submission
        if not root.is_dir():
            return document
        found = cached_scan(root)
        if found.get("candidates"):
            brief["tuning_knob_candidates"] = found
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return document
    return document


def main(argv=None) -> int:
    """``python -m ...knob_lint TREE``: print the candidates of one source tree."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tree", type=Path)
    print(json.dumps(scan(parser.parse_args(argv).tree), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
