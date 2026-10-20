"""The Claude CLI home an agent sandbox mounts: staged fresh for each launch, never the operator's.

The operator's ``~/.claude`` is the state of every Claude Code session on the host: prompt history,
file-history backups, plans, todos, shell snapshots, session transcripts, settings and hooks, skills,
plugins and global instructions. ``~/.claude.json`` adds per-project state and user MCP servers. The
sandbox used to bind that tree whole with only ``projects/`` masked, so all the rest was readable (and
writable) by an authoring agent, although ``claude --print`` needs none of it.

A launch instead gets a home staged under a purgeable cache: an empty ``.claude/`` and a minimal
``.claude.json``, mounted where the CLI looks for them. The one operator file a mount may name is the
OAuth credential, bound writable onto the staged home's credential path. Refresh tokens are single-use
and rotate, so a copy or a read-only bind spends the operator's token with nowhere to write its
replacement; the Codex arm measured exactly that and binds its credential the same way. A further CLI
need is added here explicitly, never by mounting more of the operator's home.
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

CREDENTIAL = ".credentials.json"
#: The staged global config. Print mode reads no operator state; this only skips first-run prompts.
_CONFIG = {"hasCompletedOnboarding": True}
_BIND_OPS = ("--bind", "--ro-bind", "--dev-bind", "--bind-try", "--ro-bind-try", "--dev-bind-try")


def operator_state(home: Path | None = None) -> tuple[Path, Path]:
    """The operator's Claude state directory and global config file."""
    home = Path.home() if home is None else Path(home)
    return home / ".claude", home / ".claude.json"


def _forms(path: str | Path) -> set[Path]:
    forms = {Path(os.path.abspath(path))}
    try:
        forms.add(Path(path).resolve())
    except (OSError, RuntimeError):
        pass
    return forms


def is_operator_state(path: str | Path, home: Path | None = None) -> bool:
    """Whether ``path`` is operator Claude state, lexically or through a symlink."""
    directory, config = operator_state(home)
    directories, configs = _forms(directory), _forms(config)
    return any(form in configs or any(form.is_relative_to(state) for state in directories) for form in _forms(path))


def without_operator_state(argv: Sequence[str]) -> list[str]:
    """``argv`` minus every bind or tmpfs whose source or destination is operator Claude state."""
    kept: list[str] = []
    index = 0
    while index < len(argv):
        option = argv[index]
        width = 3 if option in _BIND_OPS else 2 if option == "--tmpfs" else 1
        operands = argv[index + 1 : index + width]
        if width > 1 and len(operands) == width - 1 and any(is_operator_state(path) for path in operands):
            index += width
            continue
        kept.extend(argv[index : index + width])
        index += width
    return kept


def launches_claude(command: str) -> bool:
    """Whether a sandboxed shell command runs the ``claude`` CLI (as any word of it, by basename).

    A miss would leave a launch without credentials, so any word counts: ``env``, ``timeout`` or a
    redirect around the CLI does not hide it.
    """
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    return any(Path(word).name == "claude" for word in words)


@dataclass(frozen=True)
class ClaudeHome:
    """One launch's staged home (host side) and the operator home whose paths it is mounted over."""

    root: Path
    operator_home: Path
    credential: Path | None

    def binds(self) -> list[str]:
        """Mount this home at the operator paths the CLI reads; the credential is the only operator file."""
        directory, config = operator_state(self.operator_home)
        argv = ["--bind", str(self.root / ".claude"), str(directory)]
        argv += ["--bind", str(self.root / ".claude.json"), str(config)]
        if self.credential is not None:
            argv += ["--bind", str(self.credential), str(directory / CREDENTIAL)]
        # An inherited config-dir selection would point the CLI back outside this home.
        return [*argv, "--unsetenv", "CLAUDE_CONFIG_DIR"]


def _write_private(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def stage(name: str, *, workspace: Path, root: Path | None = None) -> ClaudeHome:
    """A fresh, never-reused Claude home for one launch, created under ``root``.

    ``root`` defaults to the purgeable ``claude_home`` cache. It may not overlap the workspace (the
    home would be frozen and graded with the candidate) or the operator's Claude state. A Bedrock
    launch authenticates through AWS, so it is given no OAuth credential.
    """
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"a Claude home name must be one plain path component: {name!r}")
    if root is None:
        from merlin.common.artifacts import cache_dir

        root = cache_dir("claude_home")
    root = Path(root).resolve()
    home = Path.home()
    directory, _config = operator_state(home)
    for other in (Path(workspace).resolve(), directory.resolve()):
        if root.is_relative_to(other) or other.is_relative_to(root):
            raise ValueError(f"a staged Claude home root must not overlap {other}: {root}")
    root.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=root))
    (staged / ".claude").mkdir(mode=0o700)
    _write_private(staged / ".claude.json", json.dumps(_CONFIG) + "\n")
    credential = directory / CREDENTIAL
    if os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1" or not credential.is_file():
        return ClaudeHome(staged, home, None)
    # Only the mount point: the credential's bytes stay in the operator's file, never on this disk.
    _write_private(staged / ".claude" / CREDENTIAL, "")
    return ClaudeHome(staged, home, credential)
