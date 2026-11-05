"""Shared checks that keep an author skill's text honest against the library it describes.

A skill is ``author_skills/<name>/SKILL.md`` plus ``reference/*.md`` and ``scripts/*.py``. Its python
fences are executed (they define helpers the skill's own test then calls on fixtures), and every
inline code span that names a ``merlin.`` symbol is resolved against the core source by ``ast`` --
with the parameters written inside its parentheses checked against the real signature -- so a
renamed function or keyword fails a test instead of misleading the author.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from merlin.common.paths import repo_root

SKILLS = repo_root() / "packages" / "merlin-experiments" / "src" / "merlin_experiments" / "phase1" / "author_skills"
CORE = repo_root() / "src"
#: SKILL.md body budget (front matter excluded): procedural text, long material in reference/.
BODY_LIMIT = 4400
FENCE = "```"


def skill_dir(name: str) -> Path:
    return SKILLS / name


def documents(name: str) -> dict[str, str]:
    """``relative path -> text`` for SKILL.md and every reference markdown file of one skill."""
    root = skill_dir(name)
    paths = [root / "SKILL.md", *sorted((root / "reference").glob("*.md"))]
    return {str(path.relative_to(root)): path.read_text(encoding="utf-8") for path in paths}


def frontmatter(text: str) -> tuple[dict[str, object], str]:
    if not text.startswith("---\n"):
        raise ValueError("skill has no front matter")
    head, _, body = text[4:].partition("\n---\n")
    fields = {}
    for line in head.splitlines():
        key, sep, value = line.partition(": ")
        if not sep:
            raise ValueError(f"front-matter line is not 'key: value': {line!r}")
        fields[key] = json.loads(value) if value.startswith('"') else value
    return fields, body


def _split_fences(text: str) -> tuple[list[tuple[str, str]], str]:
    """``([(language, code)], prose without fences)``."""
    blocks, prose, current, language = [], [], None, ""
    for line in text.splitlines():
        if line.startswith(FENCE):
            if current is None:
                current, language = [], line[len(FENCE):].strip()
            else:
                blocks.append((language, "\n".join(current) + "\n"))
                current = None
            continue
        (prose if current is None else current).append(line)
    if current is not None:
        raise ValueError("unterminated code fence")
    return blocks, "\n".join(prose)


def python_blocks(text: str) -> list[str]:
    return [code for language, code in _split_fences(text)[0] if language == "python"]


def execute_blocks(name: str) -> dict[str, object]:
    """Run every python fence of one skill in one namespace; return what they define."""
    namespace: dict[str, object] = {"__name__": f"author_skill_{name.replace('-', '_')}"}
    for relative, text in documents(name).items():
        for index, code in enumerate(python_blocks(text)):
            exec(compile(code, f"{name}/{relative}#python{index}", "exec"), namespace)  # noqa: S102
    return namespace


def code_spans(text: str) -> list[str]:
    """Inline ``code`` spans of the prose (fenced blocks excluded)."""
    prose = _split_fences(text)[1]
    return [piece for index, piece in enumerate(prose.split("`")) if index % 2 == 1 and piece]


def symbol_references(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """``(dotted symbol, parameter names)`` for each span naming a ``merlin.`` symbol."""
    found = []
    for span in code_spans(text):
        if not span.startswith("merlin.") or " " in span.partition("(")[0]:
            continue
        name, paren, rest = span.partition("(")
        params = ()
        if paren:
            inner = rest.rpartition(")")[0]
            params = tuple(part.strip().partition("=")[0] for part in inner.split(",") if part.strip())
        found.append((name.strip(), params))
    return found


def _module_file(parts: list[str]) -> Path | None:
    base = CORE.joinpath(*parts)
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _definitions(body):
    names = {}
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names[node.target.id] = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names[alias.asname or alias.name] = node
    return names


def _parameters(node) -> set[str] | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)} | {
            a.arg for a in (args.vararg, args.kwarg) if a is not None
        }
    if isinstance(node, ast.ClassDef):
        fields = {item.target.id for item in node.body
                  if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
        init = _definitions(node.body).get("__init__")
        return fields | (_parameters(init) or set())
    return None


def resolve(symbol: str, params: tuple[str, ...] = ()) -> str:
    """Resolve ``merlin.a.b.Name[.attr]`` (or ``module:qualname``) in the core source.

    Returns the reason it does not resolve, or ``''``."""
    parts = symbol.replace(":", ".").split(".")
    for cut in range(len(parts), 0, -1):
        path = _module_file(parts[:cut])
        if path is not None:
            break
    else:
        return f"{symbol}: no module"
    node, attributes = None, parts[cut:]
    scope = ast.parse(path.read_text(encoding="utf-8")).body
    for depth, attribute in enumerate(attributes):
        node = _definitions(scope).get(attribute)
        if node is None:
            return f"{symbol}: {attribute!r} is not defined in {path.relative_to(CORE)}"
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                package = parts[: cut - (0 if path.name == "__init__.py" else 1)]
                module = ".".join(package[: len(package) - node.level + 1] + ([module] if module else []))
            original = next(a.name for a in node.names if (a.asname or a.name) == attribute)
            return resolve(".".join([module, original, *attributes[depth + 1:]]), params)
        scope = getattr(node, "body", [])
    if params and attributes:
        known = _parameters(node)
        if known is None:
            return f"{symbol}: parameters given for a non-callable definition"
        missing = [param for param in params if param not in known]
        if missing:
            return f"{symbol}: no parameter(s) {missing}"
    return ""


GENERATED = "(generated)"


def unresolved_symbols(name: str) -> list[str]:
    """Hand-written documents only: a generated reference quotes the scanned code's own labels."""
    problems = []
    for relative, text in documents(name).items():
        if GENERATED in text.partition("\n")[0]:
            continue
        for symbol, params in symbol_references(text):
            reason = resolve(symbol, params)
            if reason:
                problems.append(f"{relative}: {reason}")
    return problems


def target_words() -> frozenset[str]:
    """The repo's own registry of target and substrate names (the no-target-name gate's roster)."""
    import importlib.util
    import sys

    path = repo_root() / "build_tools" / "scripts" / "check_no_target_name.py"
    spec = importlib.util.spec_from_file_location("_skill_target_gate", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return frozenset(module.TARGET_NAMES | module.SUBSTRATE_NAMES)


def words(text: str) -> set[str]:
    out, current = set(), []
    for char in text.lower() + " ":
        if char.isalnum():
            current.append(char)
        elif current:
            out.add("".join(current))
            current = []
    return out


def structural_problems(name: str) -> list[str]:
    """Front matter, body budget, target names in text, and the script rules (no regex, size)."""
    problems = []
    fields, body = frontmatter((skill_dir(name) / "SKILL.md").read_text(encoding="utf-8"))
    if fields.get("name") != name:
        problems.append(f"front-matter name {fields.get('name')!r} differs from directory {name!r}")
    description = fields.get("description")
    if not isinstance(description, str) or not 80 <= len(description) <= 1024:
        problems.append("description must say precisely when to open the skill (80-1024 chars)")
    if len(body.encode()) > BODY_LIMIT:
        problems.append(f"SKILL.md body is {len(body.encode())} bytes > {BODY_LIMIT}")
    roster = target_words()
    texts = dict(documents(name))
    for script in sorted((skill_dir(name) / "scripts").glob("*.py")):
        source = script.read_text(encoding="utf-8")
        texts[str(script.relative_to(skill_dir(name)))] = source
        tree = ast.parse(source)
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        if "re" in imported:
            problems.append(f"{script.name}: imports re")
        if len(source.splitlines()) > 1500:
            problems.append(f"{script.name}: over 1500 lines")
    for relative, text in texts.items():
        hits = sorted(words(text) & roster)
        if hits:
            problems.append(f"{relative}: names target(s) {hits}")
    return problems
