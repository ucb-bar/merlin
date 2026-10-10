#!/usr/bin/env python3
"""Gate: an instrument nothing calls is not an instrument.

A module that computes a quality signal is "done" when it has tests, and nothing has ever required
that production code call it. Measured: the module that reports what fraction of a model reached
the accelerator, the one that records why a capability was refused, and the one that names every
unjustified host placement each had tests and no caller, while the defects they exist to show went
unreported for a year. An unwired check reads exactly like a passing one.

This gate builds the import graph structurally (``ast``; a word search over-counts, because a
module's name appears in comments and docstrings of code that never imports it) and requires every
module under the instrumented packages to have a PRODUCTION importer: code under the library, the
experiments, the targets (``merlin/targets`` and the target workflows under ``examples/``) or the
build tools, excluding the test suite and the module itself. A declared console script or an
executable ``python -m`` command in the shared runtime-rendered task prompt counts as wired.
A literal module selected through the canonical ``module_source_path`` helper is a source
dependency too: isolated workers execute its file under another interpreter. This static
dependency scan establishes no actual execution or grading authority. Tests do not: a test proves
a module works, not that anything uses it.

TWO GRANULARITIES, BECAUSE MODULE GRANULARITY HAS A BLIND SPOT THE SIZE OF A MODULE. An imported
module is "wired" whatever is inside it, so a function nobody calls hides inside one perfectly:
``llvmlower/device_build.py::routing_for_placement`` sat with zero callers and ten test references
inside a module this gate reported as wired, and would have gone on doing so: its own ``__all__`` was
the only production mention of the name in the tree. :func:`unwired_symbols` therefore repeats the
question one level down, over the public top-level definitions of the instrumented packages, with the
SAME rule: a production reference, or it is debt.

That scan is deliberately CONSERVATIVE, because a ledger full of accusations nobody can defend is
worse than no ledger -- an entry you cannot justify can never be removed. Four conditions must all
hold, and each one THROWS AWAY real debt on purpose:

  * References are matched by NAME (``ast.Name`` / ``ast.Attribute`` / ``import`` aliases / string
    constants, so a name dispatched dynamically counts as used). Name matching over-counts
    references, which under-counts debt. That is the direction this gate wants to be wrong in.
  * ``__all__`` is excluded from reference counting -- an export list DECLARES a name, it does not
    USE it, and counting it hid the very case above.
  * A DECORATED definition is never flagged. A decorator is a call that can register the object
    somewhere this gate cannot see; the gate declines to judge rather than guess.
  * The symbol must have at least one reference in a test suite. A public definition nothing
    references at all is ordinary dead code; this ledger is about the narrower, worse case -- work
    with a test behind it and no caller in front of it, which reads exactly like a passing check.

A symbol in a module the module-level ledger already holds is not repeated: that debt is recorded
once, at the coarser granularity, and paying it off resolves both.

Known debt lives in two ledgers beside this file -- ``unwired_ratchet.txt`` (one repo-relative module
path per line) and ``unwired_symbols_ratchet.txt`` (``<path>::<name>``), both keyed by the stable
policy path so a relocated file keeps its identity. Both may only shrink (``check_ratchets_shrink.py``
holds every ``*_ratchet.txt``), and an entry that has since been wired or deleted fails this gate
until it is removed, so neither ledger can rot into an allowlist.

    python build_tools/scripts/check_wiring.py            # exit 1 on a new unwired module or symbol
    python build_tools/scripts/check_wiring.py --write    # regenerate both ledgers (review the diff)
    python build_tools/scripts/check_wiring.py --list     # print every unwired module and symbol
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _source_layout  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = _source_layout.core_package(ROOT).parent
LEDGER = ROOT / "build_tools" / "scripts" / "unwired_ratchet.txt"
SYMBOL_LEDGER = ROOT / "build_tools" / "scripts" / "unwired_symbols_ratchet.txt"
#: Packages whose modules compute quality signals, verdicts or evidence.
INSTRUMENTED = tuple(
    f"{prefix}/{package}"
    for prefix in ("src/merlin", "merlin/python/merlin")
    for package in ("perf", "targetgen", "verify")
) + ("packages/merlin-experiments/src/merlin_experiments/evaluation",)
#: Where a production importer may live.
PRODUCTION = (*_source_layout.SOURCE_SCAN_ROOTS, "merlin/experiments", "merlin/targets", "build_tools")
#: Where target workflows live since the layout consolidation moved them out of ``merlin/targets``.
EXAMPLES = "examples"


def _target_workflow_roots() -> tuple[str, ...]:
    """``examples/<name>`` directories that are a target's workflow: they carry the target's own
    inputs (``target/``) or its experiment definition (``experiment.yaml``).

    These hold the operator commands a target's README documents -- binding a run to an exact RTL
    binary, probing a headline kernel -- so an import there is a production caller, exactly as one
    under ``merlin/targets`` was before the move. Frontend samples, shared shell helpers and
    standalone packages beside them (no ``target/``, no ``experiment.yaml``) do not count.
    """
    base = ROOT / EXAMPLES
    if not base.is_dir():
        return ()
    return tuple(
        directory.relative_to(ROOT).as_posix()
        for directory in sorted(base.iterdir())
        if directory.is_dir() and ((directory / "target").is_dir() or (directory / "experiment.yaml").is_file())
    )


EXCLUDED_PARTS = ("_data", "__pycache__", "tests", "_qa_ws")


def _python_files(relative_root: str) -> list[Path]:
    return [
        ROOT / path
        for path in _source_layout.python_files(ROOT, (relative_root,))
        if not any(part in EXCLUDED_PARTS for part in path.parts)
    ]


def _module_name(path: Path) -> str | None:
    module = _source_layout.module_name(path, ROOT)
    if module is not None:
        return module
    # Preserve the standalone worker-inspection seam, where callers supply a package root outside
    # the checkout rather than changing ROOT (e.g. a frozen worker source tree).
    try:
        relative = path.relative_to(PACKAGE_ROOT)
    except ValueError:
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _tree(path: Path) -> ast.AST | None:
    """The parsed file, or None for one that does not parse. Deliberately NOT cached: holding a few
    thousand trees alive made the collector's full passes cost more than parsing twice."""
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def _source_path_selections(tree: ast.AST) -> set[str]:
    """Literal source dependencies selected by the explicitly imported canonical helper.

    Do not interpret arbitrary strings, similarly named helpers or computed requests.
    Like imports, this records a source reference rather than proving execution.
    """
    selectors: set[tuple[str, ...]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level:
            for alias in node.names:
                if node.module == "merlin.common.paths" and alias.name == "module_source_path":
                    selectors.add((alias.asname or alias.name,))
                elif node.module == "merlin.common" and alias.name == "paths":
                    selectors.add((alias.asname or alias.name, "module_source_path"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "merlin.common.paths":
                    prefix = (alias.asname,) if alias.asname else tuple(alias.name.split("."))
                    selectors.add((*prefix, "module_source_path"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts: list[str] = []
        function = node.func
        while isinstance(function, ast.Attribute):
            parts.append(function.attr)
            function = function.value
        if not isinstance(function, ast.Name) or (function.id, *reversed(parts)) not in selectors:
            continue
        argument = None
        if len(node.args) == 1 and not node.keywords:
            argument = node.args[0]
        elif not node.args and len(node.keywords) == 1 and node.keywords[0].arg == "module":
            argument = node.keywords[0].value
        if (
            isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and all(part.isidentifier() for part in argument.value.split("."))
        ):
            found.add(argument.value)
    return found


def _imports(path: Path) -> set[str]:
    """Imported and fixed source-selected module names, with relative imports resolved."""
    tree = _tree(path)
    if tree is None:
        return set()
    own = _module_name(path)
    package = None
    if own is not None:
        package = own if path.name == "__init__.py" else own.rpartition(".")[0]
    found = _source_path_selections(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
            # A worker that runs under another interpreter cannot import the package, so it puts
            # its own directory on the path and imports a sibling by bare name. That is a
            # production import of the sibling module, and the file beside it is the evidence.
            for alias in node.names:
                if package and "." not in alias.name and (path.parent / f"{alias.name}.py").is_file():
                    found.add(f"{package}.{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                if package is None:
                    continue
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, base] if base else anchor)
            if base:
                found.add(base)
                found.update(f"{base}.{alias.name}" for alias in node.names)
                # Match the bare-import worker seam above for ``from sibling import name``.
                # Only a real adjacent source file establishes that package ownership.
                if not node.level and package and "." not in base and (path.parent / f"{base}.py").is_file():
                    found.add(f"{package}.{base}")
    return found


def _console_script_modules() -> set[str]:
    modules: set[str] = set()
    for pyproject in (ROOT / "pyproject.toml", *sorted((ROOT / "packages").glob("*/pyproject.toml"))):
        if not pyproject.is_file():
            continue
        in_scripts = False
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_scripts = stripped == "[project.scripts]"
                continue
            if in_scripts and "=" in stripped:
                target = stripped.split("=", 1)[1].strip().strip('"')
                modules.add(target.split(":", 1)[0])
    return modules


def _module_entrypoint(path: Path) -> bool:
    """A ``python -m`` target must have both a main function and an executable guard."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    has_main = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main" for node in tree.body
    )
    for node in tree.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if (
            not isinstance(test.left, ast.Name)
            or test.left.id != "__name__"
            or len(test.ops) != 1
            or not isinstance(test.ops[0], ast.Eq)
            or len(test.comparators) != 1
            or not isinstance(test.comparators[0], ast.Constant)
            or test.comparators[0].value != "__main__"
        ):
            continue
        if has_main and any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "main"
            for statement in node.body
            for call in ast.walk(statement)
        ):
            return True
    return False


def _generated_prompt_module_commands(imported_by: dict[str, set[Path]], candidates: dict[str, Path]) -> set[str]:
    """Only commands in the actually rendered, production-called shared prompt are entrypoints.

    The prompt generator is the task command definition. Docs, tests, comments and unrelated
    dormant strings must not make a module appear wired. This does not infer commands from all
    string literals in production code.
    """
    path = ROOT / "src/merlin/targetgen/generate_prompt.py"
    if not path.is_file() or not imported_by.get("merlin.targetgen.generate_prompt"):
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    template = None
    rendered = False
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_TEMPLATE" for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            template = node.value.value
        elif isinstance(node, ast.FunctionDef) and node.name == "render_prompt":
            rendered = any(
                isinstance(result, ast.Return)
                and isinstance(result.value, ast.Call)
                and isinstance(result.value.func, ast.Attribute)
                and result.value.func.attr == "format"
                and isinstance(result.value.func.value, ast.Name)
                and result.value.func.value.id == "_TEMPLATE"
                for result in ast.walk(node)
            )
    if not rendered or template is None:
        return set()
    commands: set[str] = set()
    for suffix in template.split("python -m ")[1:]:
        module = ""
        for char in suffix:
            if not char.isascii() or not (char.isalnum() or char in "_."):
                break
            module += char
        if not module or any(not part.isidentifier() for part in module.split(".")):
            continue
        target = candidates.get(module)
        if target is not None and _module_entrypoint(target):
            commands.add(module)
    return commands


def unwired() -> list[str]:
    candidates = {}
    # Evaluators can live in any separately installed shared-namespace distribution. Relocation
    # must not remove their obligation to have a production caller.
    extension_roots = (
        (package / part).relative_to(ROOT).as_posix()
        for package in _source_layout.source_packages(ROOT)
        if package.name == "merlin" and package.is_relative_to(ROOT / "packages")
        for part in ("perf", "targetgen", "verify")
    )
    for relative_root in (*INSTRUMENTED, *extension_roots):
        for path in _python_files(relative_root):
            if path.name == "__init__.py":
                continue
            name = _module_name(path)
            if name:
                candidates.setdefault(name, path.resolve())
    imported_by: dict[str, set[Path]] = {name: set() for name in candidates}
    for relative_root in (*PRODUCTION, *_target_workflow_roots()):
        for path in _python_files(relative_root):
            for name in _imports(path):
                if name in imported_by and candidates[name] != path.resolve():
                    imported_by[name].add(path)
    scripts = _console_script_modules() | _generated_prompt_module_commands(imported_by, candidates)
    return sorted(
        str(candidates[name].relative_to(ROOT))
        for name, importers in imported_by.items()
        if not importers and name not in scripts
    )


# --------------------------------------------------------------------------- symbol granularity


def _console_script_symbols() -> set[str]:
    """Function names a declared console script calls (``pkg.mod:main`` -> ``main``).

    An entry point has no in-tree caller by construction; the packaging metadata IS its caller."""
    symbols: set[str] = set()
    for pyproject in (ROOT / "pyproject.toml", *sorted((ROOT / "packages").glob("*/pyproject.toml"))):
        if not pyproject.is_file():
            continue
        in_scripts = False
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_scripts = stripped == "[project.scripts]"
                continue
            if in_scripts and "=" in stripped:
                target = stripped.split("=", 1)[1].strip().strip('"')
                _module, _, symbol = target.partition(":")
                if symbol:
                    symbols.add(symbol.split(".", 1)[0])
    return symbols


#: Dispatchers that reach an entry point by NAME PREFIX -- ``dir(module)`` filtered by a constant they
#: publish -- in every module they import. Such a call site names no symbol, so the name match below
#: cannot see it: ``claims/dispatch.resolve`` runs exactly one ``preflight_*`` per analyzer module and
#: no file spells the three that only it reaches. Each dispatcher is declared with the constant it
#: publishes, and the prefix is READ from its source, so renaming either cannot leave a stale copy here.
PREFIX_DISPATCHERS = (
    ("packages/merlin-experiments/src/merlin_experiments/phase2/claims/dispatch.py", "PREFLIGHT_PREFIX"),
)


def _published_constant(tree: ast.AST | None, name: str) -> str | None:
    """The non-empty string a module binds to ``name`` at top level, or None."""
    for node in getattr(tree, "body", ()):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.value.value
        ):
            return node.value.value
    return None


def prefix_dispatched() -> dict[str, set[str]]:
    """``{dotted module: {prefix, ...}}`` for every module a declared prefix dispatcher imports.

    A dispatcher absent from this tree reaches nothing (a fixture tree carries none). One that exists
    but no longer publishes its constant is an error, never an empty answer: the reach it declared
    would otherwise vanish and its entry points would read as unwired, or worse, the other way round.
    """
    reached: dict[str, set[str]] = {}
    for relative, constant in PREFIX_DISPATCHERS:
        path = ROOT / relative
        if not path.is_file():
            continue
        prefix = _published_constant(_tree(path), constant)
        if prefix is None:
            raise SystemExit(
                f"[FAIL] wiring: {relative} no longer publishes {constant}; its dispatch cannot be followed"
            )
        for module in _imports(path):
            reached.setdefault(module, set()).add(prefix)
    return reached


def _export_list_nodes(tree: ast.AST) -> set[int]:
    """Every node under an ``__all__ = [...]`` assignment.

    An export list NAMES a symbol; it does not USE it. Counting it as a reference is what let
    ``device_build.routing_for_placement`` -- zero callers, ten test references -- read as wired.
    """
    skip: set[int] = set()
    for node in getattr(tree, "body", ()):  # an export list is a module-level statement
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        if node.value is not None and any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            skip.update(id(sub) for sub in ast.walk(node.value))
    return skip


def _referenced_names(path: Path) -> set[str]:
    """Every name ``path`` could be USING, matched structurally but broadly.

    Broadly on purpose: an ``ast.Name``, an attribute access, an import alias and a bare string
    constant all count, so a symbol reached through ``getattr(module, "name")`` or a registry table
    keyed by string is treated as referenced. Over-counting references under-counts debt, and that is
    the direction a ledger of accusations should err in.
    """
    tree = _tree(path)
    if tree is None:
        return set()
    skip = _export_list_nodes(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.alias):
            found.add(node.name.rpartition(".")[2])
            if node.asname:
                found.add(node.asname)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
    return found


#: Every ASCII character that cannot occur in an identifier, mapped to a space: one C-level pass turns a
#: source into the words it could possibly reference, so a file that names no candidate is never walked.
_NON_IDENTIFIER = str.maketrans({chr(c): " " for c in range(128) if not (chr(c).isalnum() or chr(c) == "_")})


def _words(path: Path) -> set[str]:
    """A SUPERSET of the names :func:`_referenced_names` can return for ``path`` that are identifiers."""
    try:
        return set(path.read_text(encoding="utf-8").translate(_NON_IDENTIFIER).split())
    except (OSError, UnicodeDecodeError):
        return set()


def _referenced_among(paths, candidates: set[str]) -> set[str]:
    """The ``candidates`` some file in ``paths`` references. A file whose words name none of the
    candidates still open is not walked, which is what keeps the scan inside a pre-commit budget."""
    open_names, found = set(candidates), set()
    for path in paths:
        if not open_names or not (_words(path) & open_names):
            continue
        hits = _referenced_names(path) & open_names
        found |= hits
        open_names -= hits
    return found


def _public_definitions(path: Path) -> list[str]:
    """Public top-level ``def`` / ``async def`` / ``class`` names in ``path``, undecorated only.

    A decorated definition is skipped: the decorator is a call, and a call can register the object in
    a table this gate cannot follow. Declining to judge is the conservative answer.
    """
    tree = _tree(path)
    if tree is None:
        return []
    return [
        node.name
        for node in getattr(tree, "body", ())
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
        and not node.decorator_list
    ]


def _test_files() -> list[Path]:
    """Every test-suite source: the cross-subsystem suite and each distribution's own tests."""
    roots = [ROOT / "merlin" / "tests", *sorted((ROOT / "packages").glob("*/tests"))]
    return [
        path
        for base in roots
        if base.is_dir()
        for path in sorted(base.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def _instrumented_files() -> dict[str, Path]:
    """``{repo-relative path: file}`` for every module under the instrumented packages."""
    extension_roots = (
        (package / part).relative_to(ROOT).as_posix()
        for package in _source_layout.source_packages(ROOT)
        if package.name == "merlin" and package.is_relative_to(ROOT / "packages")
        for part in ("perf", "targetgen", "verify")
    )
    files: dict[str, Path] = {}
    for relative_root in (*INSTRUMENTED, *extension_roots):
        for path in _python_files(relative_root):
            if path.name != "__init__.py":
                files.setdefault(path.resolve().relative_to(ROOT.resolve()).as_posix(), path)
    return files


def unwired_symbols(known_unwired_modules: set[str] | None = None) -> list[str]:
    """``<policy path>::<name>`` for every public definition with tests and no production use.

    ``known_unwired_modules`` are the module-granular findings; a symbol inside one is left out so
    the same debt is not recorded at two granularities.
    """
    if known_unwired_modules is None:
        known_unwired_modules = set(unwired())
    known = {_source_layout.policy_path(path) for path in known_unwired_modules}

    definitions: dict[str, list[str]] = {}
    modules: dict[str, Path] = {}
    for relative, path in _instrumented_files().items():
        policy = _source_layout.policy_path(relative)
        if policy in known:
            continue
        names = _public_definitions(path)
        if names:
            definitions[policy] = names
            modules[policy] = path

    candidates = {name for names in definitions.values() for name in names}
    test_references = _referenced_among(_test_files(), candidates)
    # Every name production code could be using, ANYWHERE -- including the defining module itself. A
    # public helper called only by its own module's wired entry point is reached; it is not debt.
    used = _console_script_symbols()
    production = (path for root in (*PRODUCTION, *_target_workflow_roots()) for path in _python_files(root))
    used |= _referenced_among(production, test_references - used)
    # A definition a prefix dispatcher reaches in its own module is used, by that module only.
    dispatched = prefix_dispatched()
    reached = {
        policy: {name for name in names if any(name.startswith(prefix) for prefix in dispatched.get(module, ()))}
        for policy, names, module in (
            (policy, names, _module_name(modules[policy])) for policy, names in definitions.items()
        )
    }

    return sorted(
        f"{policy}::{name}"
        for policy, names in definitions.items()
        for name in names
        if name not in used and name not in reached[policy] and name in test_references
    )


def _ledger(path: Path | None = None) -> list[str]:
    path = LEDGER if path is None else path
    if not path.is_file():
        return []
    return [
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]


SYMBOL_HEADER = (
    "# Public top-level definitions under the instrumented packages that no production code\n"
    "# references, and at least one test does. One `<policy path>::<name>` per line.\n"
    "#\n"
    "# WHY A SECOND GRANULARITY. An imported module is `wired` whatever is inside it, so the\n"
    "# module-level ledger beside this one cannot see a function nobody calls.\n"
    "#\n"
    "# WHAT IS *NOT* HERE, and why this list is shorter than the debt. The scan is deliberately\n"
    "# conservative -- see check_wiring.py's docstring: references are matched by name (so dynamic\n"
    "# dispatch counts as use), decorated definitions are never flagged (a decorator may register\n"
    "# the object), a definition with no test behind it is left to ordinary dead-code review, and a\n"
    "# symbol inside a module the module-level ledger already holds is recorded once, there.\n"
    "#\n"
    "# May only shrink: call it from the path it was built for, or delete it.\n"
    "# growth-accepted: the symbol granularity is new; this is debt it discovered, not debt added.\n"
)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    found = unwired()
    found_symbols = unwired_symbols(set(found))
    if "--list" in arguments:
        print("\n".join([*found, *found_symbols]))
        return 0
    if "--write" in arguments:
        header = (
            "# Modules under the instrumented packages that no production code imports.\n"
            "# May only shrink: wire the module into a production path, or delete it.\n"
            "# growth-accepted: the gate is new; this is debt it discovered, not debt added.\n"
        )
        LEDGER.write_text(header + "".join(f"{path}\n" for path in found), encoding="utf-8")
        print(f"wrote {LEDGER.relative_to(ROOT)} ({len(found)} entries)")
        SYMBOL_LEDGER.write_text(SYMBOL_HEADER + "".join(f"{key}\n" for key in found_symbols), encoding="utf-8")
        print(f"wrote {SYMBOL_LEDGER.relative_to(ROOT)} ({len(found_symbols)} entries)")
        return 0
    known = set(_ledger())
    stable_found = {_source_layout.policy_path(path) for path in found}
    new = [path for path in found if path not in known and _source_layout.policy_path(path) not in known]
    stale = sorted(known - stable_found - set(found))
    for path in new:
        print(
            f"[FAIL] unwired: {path} has tests at most -- no production code imports it. Call it "
            f"from the path it was built for, or delete it."
        )
    for path in stale:
        print(
            f"[FAIL] stale ledger entry: {path} is wired or gone -- remove it from "
            f"{LEDGER.relative_to(ROOT)} so the ledger keeps shrinking."
        )
    known_symbols = set(_ledger(SYMBOL_LEDGER))
    new_symbols = [key for key in found_symbols if key not in known_symbols]
    stale_symbols = sorted(known_symbols - set(found_symbols))
    for key in new_symbols:
        print(
            f"[FAIL] unwired symbol: {key} is referenced by tests and by no production code. Call it "
            f"from the path it was built for, or delete it."
        )
    for key in stale_symbols:
        print(
            f"[FAIL] stale ledger entry: {key} is wired or gone -- remove it from "
            f"{SYMBOL_LEDGER.relative_to(ROOT)} so the ledger keeps shrinking."
        )
    if new or stale or new_symbols or stale_symbols:
        return 1
    print(
        f"[  ok] wiring: {len(found)} known unwired module(s) and {len(found_symbols)} known unwired "
        f"symbol(s) in the ledgers; no new one."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
