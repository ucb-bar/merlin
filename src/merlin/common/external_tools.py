"""Scoped access to independently versioned upstream tools (model2MLIR, ModeLIR, SpecIR) and the
cross-toolchain prefix; no eager framework imports.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from merlin.common.paths import env

# ---- model2mlir --------------------------------------------------------------------------------
# model2MLIR checkout and shared capture-interpreter selection.
#
# This adapter does not import frameworks or perform capture. Per-workload capture.toml
# selection remains in capture.bundle; LLVM Python selection remains in llvmlower.toolchain.


def root(*, default: str | Path = "/path/to/model2MLIR") -> Path:
    """Keep the historical capture-bundle alias precedence and unconfigured placeholder.

    Either process alias precedes either .env alias. Within each source,
    MERLIN_MODEL2MLIR precedes MERLIN_M2M_DIR. Resolution does not validate existence
    or fall through an explicit invalid path to a different checkout.
    """
    value = (
        os.environ.get("MERLIN_MODEL2MLIR")
        or os.environ.get("MERLIN_M2M_DIR")
        or env("MERLIN_MODEL2MLIR")
        or env("MERLIN_M2M_DIR")
        or default
    )
    return Path(value)


def capture_python_path(*, checkout: Path | None = None) -> Path:
    """Select capture Python without requiring an installed framework environment.

    MERLIN_M2M_PYTHON precedes MERLIN_M2M_VENV, then the checkout's .venv. Each setting
    honors process environment before .env. Selection never falls through an explicit
    invalid path. Keep its spelling: resolving a venv symlink loses its environment.
    This is not a replacement for a workload's explicitly pinned capture.toml environment.
    Lazy availability probes use this function; execution edges use capture_python.
    """
    executable = env("MERLIN_M2M_PYTHON")
    if executable:
        selected = Path(executable).expanduser()
    else:
        venv = env("MERLIN_M2M_VENV")
        base = Path(venv).expanduser() if venv else (checkout if checkout is not None else root()) / ".venv"
        selected = base / "bin" / "python"
    return selected


def capture_python(*, checkout: Path | None = None) -> Path:
    """Select and validate shared capture Python, never the compiler interpreter.

    Unlike capture_python_path, missing/nonexecutable choices raise immediately.
    Validation never falls back to a different checkout or environment.
    """
    selected = capture_python_path(checkout=checkout)
    if not selected.is_file() or not os.access(selected, os.X_OK):
        raise RuntimeError(
            f"model2MLIR capture Python is missing or not executable: {selected}; "
            "configure MERLIN_M2M_PYTHON or MERLIN_M2M_VENV (compiler Python is separate)"
        )
    return selected


#: The software pin (``src/merlin/resources/contracts/software_pins.yaml``) every capture is verified against.
CAPTURE_PIN = "model2mlir_capture"


def capture_pin(checkout: "Path | str | None") -> dict:
    """The capture checkout verified against :data:`CAPTURE_PIN`, as the record a capture carries.

    ``{"pin", "ok", "citation", "verification"}``; ``ok`` is False on ANY drift -- another commit, or
    bytes of a read path that are not the pinned commit's (an uncommitted edit included). No checkout,
    or a verification that could not run, is recorded as unverified (``ok`` False) rather than raised:
    whether that stops a capture is the caller's decision."""
    from merlin.common import provenance as P
    from merlin.common.provenance_claims import software_pins_path

    if checkout is None:
        return {"pin": CAPTURE_PIN, "ok": False, "citation": f"{CAPTURE_PIN} UNKNOWN: no checkout", "verification": {}}
    path = software_pins_path()
    try:
        got = P.verify(CAPTURE_PIN, checkout=Path(checkout), path=path)
        citation = P.citation(got, path=path)
    except Exception as exc:  # noqa: BLE001 -- recorded as unverified, never as pinned
        return {
            "pin": CAPTURE_PIN,
            "ok": False,
            "citation": f"{CAPTURE_PIN} UNKNOWN: {type(exc).__name__}: {exc}",
            "verification": {},
        }
    return {"pin": CAPTURE_PIN, "ok": got.ok, "citation": citation, "verification": got.to_dict()}


# ---- modelir -----------------------------------------------------------------------------------
# Scoped access to the external ModeLIR Python package with explicit import-cache and working-directory lifecycles.


@contextmanager
def modelir_importable(mlc_dir):
    """Temporarily expose a checkout when ``mlc`` is not importable; retain loaded modules.

    Only the path insertion is scoped. Module identities deliberately survive oracle calls;
    this does not isolate competing checkouts or roll back upstream import side effects.
    """
    added = None
    if mlc_dir is not None and importlib.util.find_spec("mlc") is None:
        added = str(mlc_dir)
        sys.path.insert(0, added)
    try:
        yield
    finally:
        if added is not None:
            try:
                sys.path.remove(added)
            except ValueError:
                pass


@contextmanager
def discovery_imports(root: str | Path | None):
    """Expose discovery imports, removing newly loaded ``mlc.*`` only if we inserted the root.

    Existing importable installations and borrowed modules are never evicted. This preserves
    discovery's availability behavior, not arbitrary import side effects or thread isolation.
    """
    d = root
    added = None
    before: frozenset[str] = frozenset()
    if d is not None and importlib.util.find_spec("mlc") is None:
        added = str(d)
        sys.path.insert(0, added)
        before = frozenset(sys.modules)
    try:
        yield
    finally:
        if added is not None:
            try:
                sys.path.remove(added)
            except ValueError:
                pass
            # Drop mlc modules loaded during the block so ``import mlc`` does not keep resolving from the
            # module cache after the path entry is gone (which would flip mlc_available() True process-wide).
            for name in [n for n in sys.modules if n == "mlc" or n.startswith("mlc.")]:
                if name not in before:
                    del sys.modules[name]


#: Serializes :func:`artifact_context`. The working directory and ``sys.path`` are PROCESS-global, but capsules
#: are graded on a ThreadPoolExecutor — so two threads entering this block interleave their chdirs, and
#: the first to leave restores the repo cwd out from under the one still inside. mlc then cannot resolve
#: its own ``runs/...`` artifacts and reports the arc model as ABSENT: a correct submission grades
#: ``incomplete`` on "mlc arc model unavailable", intermittently and only under parallel grading (the
#: worse tail is the loser restoring LAST, leaving every later thread with the wrong cwd). Reentrant so a
#: nested entry on the same thread does not deadlock.
_ARTIFACT_LOCK = threading.RLock()


@contextmanager
def artifact_context(root: str | Path | None = None, *, resolve_root: Callable[[], str | Path | None] | None = None):
    """mlc resolves its ``runs/...`` arc artifacts by paths RELATIVE to its own root, so its cosim +
    discovery entry points run with CWD = the mlc dir. Also CONTEXT-INSERT the mlc dir on ``sys.path`` so
    ``import mlc`` resolves even when the process's ``sys.path`` no longer carries the cwd (``''``) entry —
    mlc is not pip-installed, so without this the import relied on cwd-relative resolution and broke inside
    a process that rewrote ``sys.path`` (e.g. the capsule-bench driver's xdsl setup), silently failing the
    arc oracle preflight. The insert is context-managed (removed on exit), NOT global: a permanent insert
    flips ``mlc_available()`` process-wide and un-skips heavy tests.

    Held under :data:`_ARTIFACT_LOCK`: cwd/``sys.path`` are process state, and the grader runs capsules on
    threads, so concurrent entries must not interleave (see the lock's note). ``resolve_root`` is
    called inside that lock, preserving configured-root lookup ordering for bridge callers.
    Imported modules remain cached. Other code changing cwd without this context is not isolated.
    """
    if root is not None and resolve_root is not None:
        raise ValueError("provide an explicit ModeLIR root or resolve_root, not both")
    with _ARTIFACT_LOCK:
        d = resolve_root() if resolve_root is not None else root
        prev = os.getcwd()
        ds = str(d) if d is not None else None
        inserted = False
        if d is not None:
            os.chdir(d)
            if ds not in sys.path:
                sys.path.insert(0, ds)
                inserted = True
        try:
            yield
        finally:
            os.chdir(prev)
            if inserted:
                try:
                    sys.path.remove(ds)
                except ValueError:
                    pass


# ---- specir ------------------------------------------------------------------------------------
# Scoped SpecIR imports, without owning reference-model semantics or discovery.

_IMPORT_LOCK = threading.RLock()


class SpecIRImportError(ImportError):
    """The requested checkout cannot supply an unambiguous SpecIR module graph."""


def _check_loaded(package: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != "specir" and not name.startswith("specir."):
            continue
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        locations = list(getattr(module, "__path__", ()))
        if origin and origin not in {"built-in", "frozen"}:
            locations.append(origin)
        if not locations or any(not Path(path).resolve().is_relative_to(package) for path in locations):
            raise SpecIRImportError(
                f"loaded {name} does not belong to SpecIR checkout {package.parent}; "
                "select the checkout in a fresh process (modules are never unloaded)"
            )


@contextmanager
def specir_importable(root: str | Path | None):
    """Expose a selected checkout only for this scope; retain normal module identity.

    None uses ordinary installed resolution unchanged. Explicit roots cannot fall back
    to another installation. Cooperating callers serialize path changes, not arbitrary
    upstream side effects or unrelated imports. This is not a Python sandbox.
    """
    with _IMPORT_LOCK:
        before = list(sys.path)
        try:
            package = None
            if root is not None:
                base = Path(root).resolve()
                package = base / "specir"
                if not (package / "__init__.py").is_file():
                    raise SpecIRImportError(f"SpecIR checkout missing at {base}; configure SPECIR_ROOT")
                package = package.resolve()
                _check_loaded(package)
                sys.path.insert(0, str(base))
            yield
            # Only validate successful bodies: never replace their primary exception
            # with a secondary ownership error from partially imported modules.
            if package is not None:
                _check_loaded(package)
        finally:
            sys.path[:] = before


# ---- toolchain_locator -------------------------------------------------------------------------
# Locate a cross-toolchain PREFIX from what a target contract declares about it.
#
# Three callers resolved the same cross-toolchain -- the board adapter, the ExecuTorch session build and
# the paper-package preflight -- and each carried its own copy of the resolver with the vendor's
# extracted-release directory name spelled into it. That naming is a fact about the toolchain, not about
# the code looking for it, so it is now DATA: the target contract that owns the toolchain declares it
# once under ``runtime.toolchain``, keyed by the environment variable that points at the install::
#
#     runtime:
#       toolchain:
#         env: <VARIABLE>          # the variable that locates the install
#         release_dirs: [<glob>]   # the vendor's extracted-release directory naming
#         default_install: <path>  # repo-relative; used only when <VARIABLE> is unset
#
# A prefix is a directory holding the requested ``bin/<tool>`` files. The install itself is tried first,
# then each declared release directory one level under it, then two. Nothing is guessed: with no
# declaration only an explicit prefix is accepted, and a missing tool comes back as ``None`` for the
# caller to report in its own terms.


@dataclass(frozen=True)
class ToolchainDeclaration:
    """One contract's ``runtime.toolchain`` block."""

    target: str  # the target whose contract declares it
    env: str  # the variable that locates the install
    release_dirs: tuple[str, ...] = ()  # extracted-release directory globs under the install
    default_install: str | None = None  # repo-relative (or absolute) install used when unset


def declared(env_var: str) -> ToolchainDeclaration | None:
    """The toolchain a target contract declares for ``env_var``, or None when none does.

    Exactly one contract may claim a variable. Two claims would make the layout ambiguous, so that
    also yields None -- the caller then accepts only an explicit prefix -- rather than a pick between
    them. A registry or contract that cannot be read declares nothing.
    """
    try:
        from merlin.target.registry import identity as target_registry

        names = list(target_registry.all_targets())
    except Exception:  # noqa: BLE001 — no registry, no declaration
        return None
    found: list[ToolchainDeclaration] = []
    for name in names:
        try:
            contract = target_registry.resolve(name).load_contract() or {}
        except Exception:  # noqa: BLE001 — a contract that will not load declares nothing
            continue
        block = (contract.get("runtime") or {}).get("toolchain")
        if not isinstance(block, dict) or block.get("env") != env_var:
            continue
        install = block.get("default_install")
        found.append(
            ToolchainDeclaration(
                target=name,
                env=env_var,
                release_dirs=tuple(str(g) for g in (block.get("release_dirs") or ()) if str(g)),
                default_install=str(install) if install else None,
            )
        )
    return found[0] if len(found) == 1 else None


def default_install(env_var: str) -> Path | None:
    """The install directory declared for ``env_var`` (a relative path is taken from the repo root),
    or None when no contract declares one."""
    decl = declared(env_var)
    if decl is None or not decl.default_install:
        return None
    path = Path(decl.default_install)
    if path.is_absolute():
        return path
    from merlin.common.paths import repo_root

    return repo_root() / path


def candidates(install: str | Path, env_var: str) -> list[Path]:
    """Where a prefix may sit, in search order: the install itself, then each declared release
    directory one level under it, then two."""
    install = Path(install)
    out = [install]
    if install.is_dir():
        decl = declared(env_var)
        globs = decl.release_dirs if decl is not None else ()
        for pattern in globs:
            out.extend(sorted(install.glob(pattern)))
        for pattern in globs:
            out.extend(sorted(install.glob(f"*/{pattern}")))
    return out


def find_prefix(install: str | Path, env_var: str, *, tools: Sequence[str] = ("clang",)) -> Path | None:
    """The first candidate holding every ``bin/<tool>`` in ``tools`` (returned as found, unresolved),
    or None when none does."""
    for cand in candidates(install, env_var):
        if all((cand / "bin" / tool).is_file() for tool in tools):
            return cand
    return None


__all__ = [
    "ToolchainDeclaration",
    "candidates",
    "declared",
    "default_install",
    "find_prefix",
]
