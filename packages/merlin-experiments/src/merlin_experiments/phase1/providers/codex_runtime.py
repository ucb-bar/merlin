"""Explicit client selections shared by ordinary authoring and its sealed canary.

The credential selection is path/stat only. This owner never reads credentials,
searches homes or grants runtime/target qualification.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path


def cached_cli_version(codex_bin: str, cache: dict) -> str | None:
    """Retain the client's verbatim version, including unavailable observations."""
    if codex_bin in cache:
        return cache[codex_bin]
    out = None
    try:
        proc = subprocess.run([codex_bin, "--version"], capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            lines = (proc.stdout or proc.stderr or "").strip().splitlines()
            out = lines[0].strip() if lines else None
    except (OSError, subprocess.SubprocessError):
        pass
    cache[codex_bin] = out or None
    return cache[codex_bin]


def canonical_file(value: str, label: str) -> Path:
    path = Path(value)
    try:
        valid = value and path.is_absolute() and path.resolve() == path and stat.S_ISREG(path.stat().st_mode)
    except OSError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be an explicit canonical regular file") from None
    return path


def overlap(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def validate_options(options) -> None:
    selected = (options.codex_binary, options.codex_auth_source, options.codex_home_root)
    if not any(selected):
        if options.codex_canary:
            raise ValueError("Codex canary requires complete explicit client selections")
        return
    if not all(selected) or options.driver != "codex" or options.sandbox != "bwrap" or options.allow_unsandboxed:
        raise ValueError("explicit client selections require the complete isolated Codex route")
    binary = canonical_file(options.codex_binary, "Codex executable")
    auth = canonical_file(options.codex_auth_source, "client credential")
    try:
        binary_stat, auth_stat = binary.stat(), auth.stat()
    except OSError:
        raise ValueError("selected client files changed during validation") from None
    if (binary_stat.st_dev, binary_stat.st_ino) == (auth_stat.st_dev, auth_stat.st_ino):
        raise ValueError("client executable and credential must be distinct files")
    if not os.access(binary, os.X_OK):
        raise ValueError("selected Codex executable is not executable")
    home = Path(options.codex_home_root)
    if (
        not home.is_absolute()
        or home.resolve() != home
        or home == Path("/")
        or home.is_symlink()
        or any(overlap(home, path) for path in (auth, binary))
    ):
        raise ValueError("isolated client home must be canonical and separate from client inputs")
    if home.exists() and not home.is_dir():
        raise ValueError("client home root must be a directory; each round home must be fresh")


def selection_record(options) -> dict | None:
    validate_options(options)
    if not options.codex_binary:
        return None
    binary = Path(options.codex_binary)
    # Stream only the explicitly selected executable; never read credential bytes.
    digest = hashlib.sha256()
    try:
        with binary.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        raise ValueError("selected client executable is unreadable") from None
    sandbox_binary = canonical_file(str(Path(shutil.which("bwrap") or "").resolve()), "client sandbox executable")
    code_host = binary.with_name("codex-code-mode-host")
    companion = {}
    if code_host.exists():
        code_host = canonical_file(str(code_host), "client code-mode host")
        if overlap(code_host, Path(options.codex_auth_source)) or overlap(code_host, Path(options.codex_home_root)):
            raise ValueError("client companion overlaps private selections")
        companion = {
            "code_mode_host": str(code_host),
            "code_mode_host_sha256": hashlib.sha256(code_host.read_bytes()).hexdigest(),
        }
    return {
        "sandbox_binary": str(sandbox_binary),
        "sandbox_binary_sha256": hashlib.sha256(sandbox_binary.read_bytes()).hexdigest(),
        **companion,
        "schema": "phase1_codex_runtime.v1",
        "binary": str(binary),
        "binary_sha256": digest.hexdigest(),
        "helper_path": str(binary.with_name("codex-linux-sandbox")),
        "helper_search_path": str(binary.parent) + ":" + os.environ.get("PATH", ""),
        "auth_source": options.codex_auth_source,
        "home_root": options.codex_home_root,
    }


@dataclass(frozen=True)
class SelectedCodexRuntime:
    selection: tuple[tuple[str, str], ...]
    read_paths: tuple[str, ...]
    toolchain_mounts: tuple[str, ...]
    python_paths: tuple[str, ...]
    tool_environment: str

    def record(self) -> dict:
        return {
            **dict(self.selection),
            "read_paths": list(self.read_paths),
            "toolchain_mounts": list(self.toolchain_mounts),
            "python_paths": list(self.python_paths),
            "tool_environment": self.tool_environment,
        }

    def verify(self) -> None:
        from types import SimpleNamespace

        selected = dict(self.selection)
        options = SimpleNamespace(
            codex_binary=selected["binary"],
            codex_auth_source=selected["auth_source"],
            codex_home_root=selected["home_root"],
            codex_canary=False,
            driver="codex",
            sandbox="bwrap",
            allow_unsandboxed=False,
            preflight_only=False,
        )
        try:
            current = selection_record(options)
        except ValueError:
            raise ValueError("selected client executable or runtime selection changed") from None
        if current != selected:
            raise ValueError("selected client executable or runtime selection changed")

    def runtime_binds(self, home: Path) -> list[str]:
        self.verify()
        selected = dict(self.selection)
        root = Path(selected["home_root"])
        if home.parent != root or not home.is_absolute() or home.resolve() != home:
            raise ValueError("client round home escaped its selected root")
        alias_root = home / "tmp" / "arg0"
        alias_root.mkdir(parents=True, exist_ok=True)
        if alias_root.resolve() != alias_root or any(alias_root.iterdir()):
            raise ValueError("client helper alias directory must remain empty")
        companions = []
        if "code_mode_host" in selected:
            companions = ["--ro-bind", selected["code_mode_host"], selected["code_mode_host"]]
        return [
            *companions,
            "--ro-bind", selected["sandbox_binary"], "/usr/bin/bwrap",
            "--ro-bind",
            selected["binary"],
            selected["binary"],
            "--ro-bind",
            selected["binary"],
            selected["helper_path"],
            "--setenv",
            "PATH",
            selected["helper_search_path"],
            "--bind",
            str(home),
            str(home),
            "--bind",
            selected["auth_source"],
            str(home / "auth.json"),
            "--setenv",
            "CODEX_HOME",
            str(home),
            "--ro-bind", str(alias_root), str(alias_root),
        ]

    def round_kwargs(self) -> dict:
        self.verify()
        selected = dict(self.selection)
        return {
            "codex_binary": selected["binary"],
            "auth_source": Path(selected["auth_source"]),
            "codex_home_root": Path(selected["home_root"]),
            "candidate_read_paths": self.read_paths,
            "runtime_binds": self.runtime_binds,
            "require_fresh_home": True,
        }


def select(options, context, bundle: dict, workspace: Path, private_run: Path) -> SelectedCodexRuntime | None:
    selected = selection_record(options)
    if selected is None:
        return None
    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.sandbox import toolchain as TC
    from merlin.targetgen.target_experiment import load_target_experiment

    home, auth = Path(selected["home_root"]), Path(selected["auth_source"])
    if overlap(home, workspace) or overlap(home, private_run) or overlap(auth, workspace):
        raise ValueError("client private selections overlap workspace or private run")
    target = load_target_experiment(context.descriptor)
    mounts = tuple(TC.toolchain_binds(target))
    paths_selection = TC.paths_for_bundle(workspace, bundle)
    python_paths = (
        (str(paths_selection.repo / "merlin/python"),)
        if paths_selection.python_import_roots is None
        else paths_selection.python_import_roots
    )
    python_paths = (*python_paths, str(workspace))
    tool_environment = TC.sandbox_env(target, workspace, bundle=bundle)
    _, grants = BW._snapshot_grants(workspace, bundle, context.repo)
    paths = [str(destination) for _, destination, _ in grants]
    index = 0
    while index < len(mounts):
        flag = mounts[index]
        if flag == "--ro-bind" and index + 2 < len(mounts):
            paths.append(mounts[index + 2])
            index += 3
        elif flag in {"--unsetenv", "--tmpfs"} and index + 1 < len(mounts):
            index += 2
        else:
            raise ValueError("unsupported ordinary toolchain mount selection")
    paths.append(selected["binary"])
    if "code_mode_host" in selected:
        paths.append(selected["code_mode_host"])
    # The standalone sandbox command uses the public alias. Session startup
    # falls back to the pinned executable when private alias creation is disabled.
    paths.append(selected["helper_path"])
    read_paths = tuple(dict.fromkeys(paths))
    for value in read_paths:
        path = Path(value)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or path == Path("/")
            or overlap(path, home)
            or overlap(path, auth)
            or overlap(path, private_run)
        ):
            raise ValueError("client read grant overlaps private selections")
    return SelectedCodexRuntime(tuple(sorted(selected.items())), read_paths, mounts, python_paths, tool_environment)
