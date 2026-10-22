"""Explicit, host-reviewed compiler-library admission for independent backends.

This contract records reviewed bytes and a closed set of direct Merlin imports.
It does not authorize a candidate to select its own dependencies, establish
semantic generality, or replace runtime filesystem/network isolation.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import hashlib
import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import lru_cache
from importlib.util import resolve_name
from pathlib import Path, PurePosixPath

from merlin.common.access import MODULE_ACCESS, is_harness_module, module_matches

SCHEMA = "merlin.compiler_library.v1"
_FORBIDDEN_MODULES = ("merlin.runtime.reference", "merlin.runtime.simulator", "merlin_experiments")
_EXCLUDED_PARTS = frozenset({".git", "out", "docs", "tests", "__pycache__"})
_ACTIVE_LIBRARY = contextvars.ContextVar("selected_additional_compiler_library", default=None)
# Exact, target-neutral re-export API. This does not authorize a package prefix.
_PUBLIC_REEXPORTS = {"merlin.sched.ir": frozenset({"expr", "kernel", "parse"})}


def _verify_public_reexports(payload: bytes, module: str, public_modules: tuple[str, ...]) -> None:
    """Admit only a static facade over separately granted implementation leaves."""
    imports, declared = set(), None
    for node in ast.parse(payload).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if (isinstance(node, ast.ImportFrom) and node.level == 1
                and node.module in _PUBLIC_REEXPORTS[module]
                and f"{module}.{node.module}" in public_modules
                and all(alias.name != "*" and alias.asname is None for alias in node.names)):
            imports.update(alias.name for alias in node.names)
            continue
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__all__"
                and isinstance(node.value, (ast.List, ast.Tuple)) and declared is None
                and all(isinstance(item, ast.Constant) and type(item.value) is str for item in node.value.elts)):
            declared = [item.value for item in node.value.elts]
            continue
        raise CompilerLibraryError("public namespace must contain only reviewed static leaf re-exports")
    if not imports or declared is None or len(declared) != len(set(declared)) or set(declared) != imports:
        raise CompilerLibraryError("public namespace exports must exactly name its reviewed leaf imports")


def _parse_syntax_facts(payload: bytes, filename: str) -> tuple:
    """Pure immutable syntax; no import admission or filesystem result is retained."""
    tree = ast.parse(payload, filename=filename)
    exports, imports = set(), []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exports.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            exports.update(alias.asname or alias.name.split(".")[0] for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Assign):
            exports.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            exports.add(node.target.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.append(("import", 0, "", tuple(alias.name for alias in node.names)))
        elif isinstance(node, ast.ImportFrom):
            imports.append(("from", node.level, node.module or "", tuple(alias.name for alias in node.names)))
    return tuple(sorted(exports)), tuple(imports)


@lru_cache(maxsize=512)
def _cached_syntax_facts(payload: bytes, filename: str, parser, syntax_parser) -> tuple:
    # Both original parser objects are part of the key; a replaced verifier or
    # Python syntax parser cannot inherit facts derived by its predecessor.
    return parser(payload, filename)


def _syntax_facts(payload: bytes, filename: str) -> tuple:
    # One reviewed library may exceed 128 members; sequential verification of
    # a larger set would otherwise evict nearly every fact before its next use.
    # At most 512 * 256 KiB = 128 MiB of source bytes are retained (immutable
    # facts add overhead). Larger members retain the uncached parser. No ASTs
    # or admission decisions are retained; callers still reopen/hash all files.
    if len(payload) > 256 << 10:
        return _parse_syntax_facts(payload, filename)
    return _cached_syntax_facts(payload, filename, _parse_syntax_facts, ast.parse)


class CompilerLibraryError(ValueError):
    """The reviewed library cannot be admitted without changing its contract."""


def _safe_member(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or any(part in {".", ".."} or part in _EXCLUDED_PARTS for part in path.parts)
        or "\\" in value
        or any(ord(char) < 32 for char in value)
    ):
        raise CompilerLibraryError(f"unsafe compiler library member: {value!r}")
    return value


def _forbidden(module: str) -> bool:
    return any(module_matches(module, prefix) for prefix in _FORBIDDEN_MODULES) or any(
        module_matches(module, prefix) for access in MODULE_ACCESS for prefix in access.modules
    )


def _read(root: Path, relative: str) -> bytes:
    path = root / _safe_member(relative)
    if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_file():
        raise CompilerLibraryError(f"library member absent or linked: {relative}")
    if not path.resolve().is_relative_to(root):
        raise CompilerLibraryError(f"library member escapes root: {relative}")
    return path.read_bytes()


@dataclass(frozen=True)
class LibraryMember:
    path: str
    sha256: str
    module: str | None = None

    def __post_init__(self) -> None:
        _safe_member(self.path)
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise CompilerLibraryError("library member needs a lowercase SHA-256")
        if self.module is not None and (
            not self.module
            or not all(part.isidentifier() for part in self.module.split("."))
            or _forbidden(self.module)
            or not self.path.endswith(".py")
        ):
            raise CompilerLibraryError("invalid or withheld compiler library module")
        if self.module is not None and self.path not in {
            self.module.replace(".", "/") + ".py",
            self.module.replace(".", "/") + "/__init__.py",
        }:
            raise CompilerLibraryError("library source path must match its import identity")
        if self.module is None and PurePosixPath(self.path).suffix in {
            ".py",
            ".pyc",
            ".pyo",
            ".so",
            ".pyd",
            ".zip",
            ".whl",
        }:
            raise CompilerLibraryError("executable library members require explicit module ownership")


@dataclass(frozen=True)
class CompilerLibraryContract:
    review_id: str
    public_modules: tuple[str, ...]
    members: tuple[LibraryMember, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.review_id, str) or not self.review_id.strip():
            raise CompilerLibraryError("library requires explicit host review attribution")
        if (
            not isinstance(self.members, tuple)
            or not self.members
            or any(type(member) is not LibraryMember for member in self.members)
        ):
            raise CompilerLibraryError("library requires immutable, nonempty reviewed membership")
        paths = [member.path for member in self.members]
        modules = [member.module for member in self.members if member.module is not None]
        if len(set(paths)) != len(paths) or len(set(modules)) != len(modules):
            raise CompilerLibraryError("duplicate compiler library ownership")
        if (
            not isinstance(self.public_modules, tuple)
            or not self.public_modules
            or len(set(self.public_modules)) != len(self.public_modules)
            or any(module not in modules or _forbidden(module) for module in self.public_modules)
        ):
            raise CompilerLibraryError("public APIs must name exact reviewed module members")
        indexed = {member.module: member for member in self.members if member.module is not None}
        if any(PurePosixPath(indexed[module].path).name == "__init__.py" and module not in _PUBLIC_REEXPORTS
               for module in self.public_modules):
            raise CompilerLibraryError("public APIs must be leaf modules, not broad package grants")

    def permits(self, module: str) -> bool:
        """Exact API admission: approving a package does not approve its children."""
        return module in self.public_modules and not _forbidden(module)

    def record(self) -> dict:
        return {
            "schema": SCHEMA,
            "review_id": self.review_id,
            "public_modules": list(self.public_modules),
            "members": [
                {"path": member.path, "sha256": member.sha256, "module": member.module} for member in self.members
            ],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def verify(self, root: Path) -> None:
        """Recheck bytes and direct import closure without importing reviewed code."""
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise CompilerLibraryError("compiler library root is not a directory")
        modules = {member.module: member for member in self.members if member.module is not None}
        package_exports = {}
        for module, member in modules.items():
            if PurePosixPath(member.path).name != "__init__.py":
                continue
            try:
                exports, _ = _syntax_facts(_read(root, member.path), member.path)
            except (SyntaxError, UnicodeError) as exc:
                raise CompilerLibraryError("invalid library namespace initializer") from exc
            package_exports[module] = set(exports)
        for module in modules:
            pieces = module.split(".")
            for length in range(1, len(pieces)):
                parent = modules.get(".".join(pieces[:length]))
                if parent is None or PurePosixPath(parent.path).name != "__init__.py":
                    raise CompilerLibraryError(f"missing library namespace initializer: {module}")
        for member in self.members:
            payload = _read(root, member.path)
            if hashlib.sha256(payload).hexdigest() != member.sha256:
                raise CompilerLibraryError(f"reviewed compiler library bytes changed: {member.path}")
            if member.module is None:
                continue
            try:
                _, imports = _syntax_facts(payload, member.path)
                if member.module in self.public_modules and PurePosixPath(member.path).name == "__init__.py":
                    _verify_public_reexports(payload, member.module, self.public_modules)
            except (SyntaxError, UnicodeError) as exc:
                raise CompilerLibraryError(f"invalid reviewed Python source: {member.path}") from exc
            package = (
                member.module if PurePosixPath(member.path).name == "__init__.py" else member.module.rpartition(".")[0]
            )
            for kind, level, module, names in imports:
                dependencies: list[str] = []
                if kind == "import":
                    dependencies = list(names)
                else:
                    try:
                        base = resolve_name("." * level + module, package) if level else module
                    except (ImportError, ValueError) as exc:
                        raise CompilerLibraryError("invalid relative library import") from exc
                    dependencies = [base]
                    dependencies.extend(f"{base}.{name}" for name in names if f"{base}.{name}" in modules)
                    if base in package_exports and any(
                        f"{base}.{name}" not in modules and name not in package_exports[base] for name in names
                    ):
                        raise CompilerLibraryError(f"unreviewed namespace import from {base}")
                for dependency in dependencies:
                    if not is_harness_module(dependency):
                        continue
                    if _forbidden(dependency) or dependency not in modules:
                        raise CompilerLibraryError(f"unreviewed library dependency: {dependency}")
                    # Python executes namespace initializers before the leaf module.
                    pieces = dependency.split(".")
                    for length in range(1, len(pieces)):
                        if ".".join(pieces[:length]) not in modules:
                            raise CompilerLibraryError(f"missing library namespace initializer: {dependency}")


def freeze_compiler_library(
    root: Path,
    *,
    review_id: str,
    public_modules: tuple[str, ...],
    sources: tuple[tuple[str, str | None], ...],
) -> CompilerLibraryContract:
    """Freeze only explicit reviewed sources; never discover a checkout or infer grants."""
    root = Path(root).resolve(strict=True)
    contract = CompilerLibraryContract(
        review_id,
        public_modules,
        tuple(LibraryMember(path, hashlib.sha256(_read(root, path)).hexdigest(), module) for path, module in sources),
    )
    contract.verify(root)
    return contract


def selected_library_record(contract: CompilerLibraryContract | None, root: Path | None) -> dict | None:
    """Reopen an explicit host selection; this record grants no semantic authority."""
    if contract is None and root is None:
        return None
    if type(contract) is not CompilerLibraryContract or not isinstance(root, Path):
        raise CompilerLibraryError("compiler library requires the exact explicit contract and root together")
    if not root.is_absolute() or root.resolve() != root or any(path.is_symlink() for path in (root, *root.parents)):
        raise CompilerLibraryError("compiler library selection requires a canonical unlinked root")
    contract.verify(root)
    return {"root": str(root), "contract_sha256": contract.sha256, "contract": contract.record()}


@dataclass(frozen=True, eq=False)
class CompilerLibrarySelection:
    """Explicit host selection, separate from a historical compiler profile.

    Membership is reviewed by the caller. This binds immutable bytes and exact
    imports only; runtime isolation and numerical qualification stay independent.
    """

    contract: CompilerLibraryContract
    root: Path
    _binding: bytes = dataclass_field(init=False, repr=False)

    def __post_init__(self):
        if type(self.contract) is not CompilerLibraryContract or not isinstance(self.root, Path):
            raise CompilerLibraryError("additional compiler imports require an explicit contract and root")
        record = selected_library_record(self.contract, self.root)
        object.__setattr__(self, "_binding", json.dumps(record, sort_keys=True, separators=(",", ":")).encode())

    def record(self):
        record = selected_library_record(self.contract, self.root)
        if json.dumps(record, sort_keys=True, separators=(",", ":")).encode() != self._binding:
            raise CompilerLibraryError("additional compiler library selection changed")
        return record


@contextlib.contextmanager
def scoped_compiler_library(selection: CompilerLibrarySelection | None):
    """Trusted call scope; candidate manifests cannot install this selection."""
    if selection is None:
        yield
        return
    if type(selection) is not CompilerLibrarySelection:
        raise CompilerLibraryError("additional compiler imports require an explicit host selection")
    selection.record()
    active = _ACTIVE_LIBRARY.get()
    if active is not None and active is not selection:
        raise CompilerLibraryError("cannot replace a live additional compiler library selection")
    token = _ACTIVE_LIBRARY.set(selection)
    try:
        yield
        selection.record()
    finally:
        _ACTIVE_LIBRARY.reset(token)


def active_compiler_library():
    """Reopen the current host-selected bytes before import or mount admission."""
    selection = _ACTIVE_LIBRARY.get()
    if selection is not None:
        if type(selection) is not CompilerLibrarySelection:
            raise CompilerLibraryError("additional compiler imports lost their selected owner")
        selection.record()
    return selection
