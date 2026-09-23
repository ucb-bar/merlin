"""Locate a cross-toolchain PREFIX from what a target contract declares about it.

Three callers resolved the same cross-toolchain -- the board adapter, the ExecuTorch session build and
the paper-package preflight -- and each carried its own copy of the resolver with the vendor's
extracted-release directory name spelled into it. That naming is a fact about the toolchain, not about
the code looking for it, so it is now DATA: the target contract that owns the toolchain declares it
once under ``runtime.toolchain``, keyed by the environment variable that points at the install::

    runtime:
      toolchain:
        env: <VARIABLE>          # the variable that locates the install
        release_dirs: [<glob>]   # the vendor's extracted-release directory naming
        default_install: <path>  # repo-relative; used only when <VARIABLE> is unset

A prefix is a directory holding the requested ``bin/<tool>`` files. The install itself is tried first,
then each declared release directory one level under it, then two. Nothing is guessed: with no
declaration only an explicit prefix is accepted, and a missing tool comes back as ``None`` for the
caller to report in its own terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ToolchainDeclaration", "declared", "default_install", "candidates", "find_prefix"]


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
        from merlin.targetgen import target_registry

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
