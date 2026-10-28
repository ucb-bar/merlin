"""Static library-route analysis of a frozen compiler tree, per operation family.

A compiler tree is an author directory (top-level ``*.py`` modules) plus an
optional ``compiler_library/`` copy of the shared library.  The routes a tree
uses are the library modules its author modules import (parsed with ``ast``;
no code is executed).  Each module is assigned an operation family from the
words in its name, read left to right, so ``dense_ready_readout`` is a dense
route and ``spatial_readout`` a convolution route.  Families are generic
operation kinds, never target facts.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

LIBRARY_DIR = "compiler_library"
SHARED = "shared"

# Ordered (family, words) pairs.  A module name's first word found here decides.
FAMILY_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("attention", ("attention", "softmax", "sdpa", "kv", "flash")),
    ("conv", ("conv", "convolution", "spatial", "im2col", "window", "rolling", "tap", "taps")),
    ("decode", ("gemv", "decode", "decoder", "m1")),
    ("pooling/epilogue", ("pool", "maxpool", "max", "epilogue", "readout", "residual",
                          "requant", "requantization", "pointwise", "elementwise", "bias")),
    ("dense", ("dense", "matmul", "gemm", "panel", "contraction", "contractions", "rhs",
               "lhs", "segmented", "periodic", "staged", "longk", "products", "batched")),
)
FAMILIES: tuple[str, ...] = tuple(name for name, _ in FAMILY_WORDS)


def family_of(module_name: str) -> str:
    """Family of a module from the first family word in its last dotted component."""
    words = module_name.rpartition(".")[2].lower().split("_")
    for word in words:
        for family, vocabulary in FAMILY_WORDS:
            if word in vocabulary:
                return family
    return SHARED


@dataclass
class TreeRoutes:
    root: str
    author_modules: dict[str, int] = field(default_factory=dict)  # name -> line count
    author_imports: dict[str, set[str]] = field(default_factory=dict)  # author -> author deps
    library_imports: dict[str, set[str]] = field(default_factory=dict)  # author -> library modules
    unparsed: list[str] = field(default_factory=list)

    def library_by_family(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {name: set() for name in FAMILIES}
        for modules in self.library_imports.values():
            for module in modules:
                family = family_of(module)
                if family != SHARED:
                    out[family].add(module)
        return out

    def author_by_family(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {name: {} for name in FAMILIES}
        for module, lines in self.author_modules.items():
            family = family_of(module)
            if family != SHARED:
                out[family][module] = lines
        return out

    def to_json(self) -> dict:
        return {
            "root": self.root,
            "author_modules": dict(sorted(self.author_modules.items())),
            "library_by_family": {k: sorted(v) for k, v in self.library_by_family().items()},
            "author_by_family": self.author_by_family(),
            "unparsed": self.unparsed,
        }


def _library_modules(tree: Path) -> set[str]:
    library = tree / LIBRARY_DIR
    names: set[str] = set()
    if not library.is_dir():
        return names
    for path in library.rglob("*.py"):
        rel = path.relative_to(library).with_suffix("")
        parts = list(rel.parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            names.add(".".join(parts))
    return names


def _imported_names(node: ast.AST) -> list[str]:
    """Absolute or tree-relative module names a single import statement names."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        base = node.module or ""
        if node.level and not base:
            return [alias.name for alias in node.names]
        names = [base]
        names.extend(f"{base}.{alias.name}" for alias in node.names)
        return names
    return []


def analyze_tree(tree: Path) -> TreeRoutes:
    """Author modules, their author-level imports and the library modules they use."""
    tree = Path(tree)
    library = _library_modules(tree)
    result = TreeRoutes(root=str(tree))
    authors = {p.stem: p for p in sorted(tree.glob("*.py")) if p.is_file()}
    for stem, path in authors.items():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            parsed = ast.parse(text)
        except (OSError, SyntaxError, ValueError):
            result.unparsed.append(path.name)
            continue
        result.author_modules[stem] = text.count("\n") + 1
        deps: set[str] = set()
        libs: set[str] = set()
        for node in ast.walk(parsed):
            for name in _imported_names(node):
                head = name.partition(".")[0]
                if head in authors and head != stem:
                    deps.add(head)
                elif name in library:
                    libs.add(name)
        result.author_imports[stem] = deps
        result.library_imports[stem] = libs
    return result


def reachable_library(routes: TreeRoutes, roots: list[str]) -> set[str]:
    """Library modules reachable from ``roots`` through author-level imports."""
    seen: set[str] = set()
    stack = [r for r in roots if r in routes.author_modules]
    found: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        found |= routes.library_imports.get(current, set())
        stack.extend(routes.author_imports.get(current, set()) - seen)
    return found


@dataclass
class FamilyComparison:
    family: str
    seed_library: list[str]
    seed_author: dict[str, int]
    other_library: dict[str, list[str]]  # compiler label -> library routes it uses
    missing_from_seed: dict[str, list[str]]  # route -> compilers that use it


def compare_families(seed: TreeRoutes, others: dict[str, TreeRoutes]) -> list[FamilyComparison]:
    seed_lib = seed.library_by_family()
    seed_auth = seed.author_by_family()
    other_lib = {label: r.library_by_family() for label, r in others.items()}
    rows: list[FamilyComparison] = []
    for family in FAMILIES:
        missing: dict[str, list[str]] = {}
        for label, by_family in other_lib.items():
            for module in sorted(by_family[family] - seed_lib[family]):
                missing.setdefault(module, []).append(label)
        rows.append(FamilyComparison(
            family=family,
            seed_library=sorted(seed_lib[family]),
            seed_author=seed_auth[family],
            other_library={label: sorted(v[family]) for label, v in other_lib.items()},
            missing_from_seed=missing,
        ))
    return rows
