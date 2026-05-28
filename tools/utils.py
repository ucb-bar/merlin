#!/usr/bin/env python3
# tools/utils.py

import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Sequence

# Constants
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS_CONFIG = REPO_ROOT / "config" / "targets.json"


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _shell_quote(text: str) -> str:
    if text == "":
        return "''"
    if all(ch.isalnum() or ch in "._-/:=+" for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def run(
    cmd: Sequence[str],
    *,
    cwd: pathlib.Path | None = None,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> int:
    cmd_str = " ".join(_shell_quote(x) for x in cmd)
    print(f"+ {cmd_str}")
    if dry_run:
        return 0
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    # Flush stdout so Python output appears before subprocess output
    sys.stdout.flush()
    completed = subprocess.run(list(cmd), cwd=str(cwd or REPO_ROOT), env=merged_env, check=False)
    return completed.returncode


def resolve_repo_path(relative: str) -> pathlib.Path:
    return (REPO_ROOT / relative).resolve()


def load_targets_config() -> dict:
    if not TARGETS_CONFIG.exists():
        return {}
    with TARGETS_CONFIG.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_repo_script(relative_script: str, script_args: Sequence[str], dry_run: bool) -> int:
    script = resolve_repo_path(relative_script)
    if not script.exists():
        eprint(f"Script not found: {script}")
        return 2
    return run(["bash", str(script), *script_args], dry_run=dry_run)


def find_toolchain_binary(
    name: str,
    *,
    env_var: str | None = None,
    aliases: Sequence[str] = (),
    fallbacks: Sequence[str] = (),
) -> pathlib.Path:
    """Locate a cross-toolchain binary, preferring portable paths.

    Resolution order:
      1. ``$env_var`` if set and the path exists.
      2. ``shutil.which(name)`` on ``$PATH``.
      3. Each alias under ``shutil.which``.
      4. Each absolute path in ``fallbacks`` if it exists.

    Raises ``FileNotFoundError`` with a helpful message if nothing resolves.

    Example::

        RISCV_OBJDUMP = find_toolchain_binary(
            "riscv64-unknown-elf-objdump",
            env_var="MERLIN_RISCV_OBJDUMP",
            aliases=("riscv64-zephyr-elf-objdump",),
            fallbacks=(),  # no developer-machine-specific absolute paths
        )

    Using this helper everywhere keeps machine-specific paths out of
    `tools/<x>/` packages (see CLAUDE.md "Tool Extension Protocol" — the
    no-overfit rule).
    """
    import shutil

    if env_var:
        env_val = os.environ.get(env_var)
        if env_val and pathlib.Path(env_val).exists():
            return pathlib.Path(env_val)

    for candidate in (name, *aliases):
        path = shutil.which(candidate)
        if path:
            return pathlib.Path(path)

    for fallback in fallbacks:
        if pathlib.Path(fallback).exists():
            return pathlib.Path(fallback)

    tried = [
        f"$PATH (name={name!r}" + (f", aliases={list(aliases)!r}" if aliases else "") + ")",
    ]
    if env_var:
        tried.insert(0, f"${env_var}")
    if fallbacks:
        tried.append(f"fallbacks={list(fallbacks)!r}")
    raise FileNotFoundError(f"could not locate {name!r}; tried (in order): " + " → ".join(tried))
