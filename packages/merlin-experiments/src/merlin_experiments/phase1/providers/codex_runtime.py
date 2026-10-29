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



def failed_round_retry(home_root: Path, run_dir: Path, rnd: int) -> tuple[Path, dict]:
    """Allocate a fresh sibling only from an exact retained failed driver record.

    No old home is reopened by the new author. The unchanged selected root and
    fresh-home checks still govern its runtime permissions. Existing successful,
    foreign, unrecorded or modified homes cannot authorize a retry.
    """
    import json
    import uuid

    root, run = Path(home_root), Path(run_dir)
    base = root / f"{run.name}_r{rnd:02d}"
    if not base.exists() and not base.is_symlink():
        return base, {}
    if root.resolve() != root or base.is_symlink() or not base.is_dir():
        raise ValueError("failed round retry requires canonical independent homes")
    records = list((run / "rounds").glob(f"round_{rnd:02d}*.codex_summary.json"))
    records = [p for p in records if p.name.startswith(f"round_{rnd:02d}.")]
    if not records:
        raise ValueError("fresh authoring refuses an existing home without a failed attempt record")
    summary_path = max(records, key=lambda p: p.stat().st_mtime_ns)
    if summary_path.is_symlink() or summary_path.resolve() != summary_path:
        raise ValueError("failed round retry summary must be an original regular file")
    raw = summary_path.read_bytes()
    summary = json.loads(raw)
    failed = (type(summary.get("exit_code")) is int and summary["exit_code"] != 0)
    if not failed:
        raise ValueError("fresh authoring refuses retry of a successful or incomplete attempt record")
    recorded = summary.get("codex_home") or {}
    prior = Path(recorded.get("codex_home", ""))
    prefix = base.name + "_attempt_"
    if (prior.parent != root or not (prior == base or prior.name.startswith(prefix))
            or prior.resolve() != prior or prior.is_symlink() or not prior.is_dir()):
        raise ValueError("failed round retry summary names a foreign home")
    config = prior / "config.toml"
    if (config.is_symlink() or not config.is_file()
            or hashlib.sha256(config.read_bytes()).hexdigest() != recorded.get("config_sha256")):
        raise ValueError("failed round retry lost its original config identity")
    attempt = uuid.uuid4().hex
    home = root / (prefix + attempt)
    home.mkdir(mode=0o700)  # Atomic, never consume an existing retry directory.
    stat = home.stat()
    return home, {
        "schema": "merlin.codex_failed_round_retry.v1", "attempt_id": attempt,
        "logical_round": rnd, "home": str(home), "previous_home": str(prior),
        "previous_summary": str(summary_path),
        "previous_summary_sha256": hashlib.sha256(raw).hexdigest(),
        "home_device": stat.st_dev, "home_inode": stat.st_ino,
        "retained_compiler": True, "reused_session": False,
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
            "allow_failed_home_retry": True,
        }


def select(
    options, context, bundle: dict, workspace: Path, private_run: Path, *, compiler_library_selection=None
) -> SelectedCodexRuntime | None:
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
    library_root = None
    if compiler_library_selection is not None:
        from merlin.targetgen.compiler_library import CompilerLibrarySelection
        from merlin_experiments.phase2.compiler_libraries import guest_root

        if type(compiler_library_selection) is not CompilerLibrarySelection:
            raise ValueError("client library paths require an original reviewed selection")
        library_root = str(guest_root(compiler_library_selection))
        python_paths = (library_root, *python_paths)
    tool_environment = TC.sandbox_env(target, workspace, bundle=bundle)
    _, grants = BW._snapshot_grants(workspace, bundle, context.repo)
    paths = [str(destination) for _, destination, _ in grants]
    if library_root is not None:
        # This guest namespace contains only the original exact-leaf overlay;
        # it is not a host source root. Outer bwrap answer masks still win.
        paths.append(library_root)
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
