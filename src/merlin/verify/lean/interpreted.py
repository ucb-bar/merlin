"""Bind standalone Lean execution to its interpreter, entry and local proof artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.verify.lean.check import _file_ref, lean_exe, lean_root


def _lake_output(*arguments: str) -> str:
    proc = subprocess.run(
        ["lake", "env", *arguments], cwd=lean_root(), capture_output=True, text=True, check=False, timeout=30
    )
    if proc.returncode or not proc.stdout.strip():
        raise RuntimeError(f"cannot identify Lean runtime: {proc.stderr.strip()[-300:]}")
    return proc.stdout.strip()


def _modules(search: tuple[Path, ...]) -> list[Path]:
    # Pin a conservative superset of local imports in every actual search directory.
    # This also detects a newly introduced shadowing module or a switched symlink.
    return sorted(
        {
            path.resolve()
            for directory in search
            for path in (directory / "MerlinVerify").rglob("*.olean*")
            if path.is_file()
        }
    )


@dataclass(frozen=True)
class Runtime:
    executable: Path
    auditor: Path
    entry: Path
    sysroot: Path
    search: tuple[Path, ...]
    modules: tuple[Path, ...]
    records: tuple[dict[str, Any], ...]

    @property
    def environment(self) -> dict[str, str]:
        return {**os.environ, "LEAN_SYSROOT": str(self.sysroot), "LEAN_PATH": os.pathsep.join(map(str, self.search))}

    @property
    def evidence(self) -> dict[str, Any]:
        return {
            "kind": "lean_interpreter_with_local_proof_artifacts",
            "executable": str(self.executable),
            "auditor": str(self.auditor),
            "entry": str(self.entry),
            "sysroot": str(self.sysroot),
            "search_path": list(map(str, self.search)),
            "files": list(self.records),
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def assert_current(self) -> None:
        if tuple(_modules(self.search)) != self.modules:
            raise ValueError("Lean proof artifact inventory changed during verification")
        if any(_file_ref(Path(record["path"])) != record for record in self.records):
            raise ValueError("Lean interpreter, entry or proof artifact changed during verification")

    def run(self, arguments: list[str], *, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        self.assert_current()
        proc = subprocess.run(
            [str(self.executable), "--run", str(self.entry), *arguments],
            cwd=lean_root(),
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        self.assert_current()
        return proc

    def axioms(self, theorem: str, module: str) -> list[str]:
        """Audit the same pinned artifacts freshly; a public binary cache cannot key these modules."""
        self.assert_current()
        proc = subprocess.run(
            [str(self.auditor), "--axioms", theorem, module],
            cwd=lean_root(),
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assert_current()
        if proc.returncode:
            raise ValueError(f"standalone theorem audit failed: {proc.stderr[-300:]}")
        result = json.loads(proc.stdout)
        if set(result) != {"decl", "axioms"} or result["decl"] != theorem or not isinstance(result["axioms"], list):
            raise ValueError("standalone theorem audit returned a different declaration")
        if any(not isinstance(axiom, str) for axiom in result["axioms"]):
            raise ValueError("standalone theorem axiom names must be strings")
        return result["axioms"]


def capture(entry: Path, *, required_module: str) -> Runtime:
    """Use Lake only to resolve the configured environment; execute the pinned interpreter directly.

    Package/toolchain trust remains the existing receipt ledger's responsibility. The capsule
    adds the actual interpreted entry and all local MerlinVerify artifacts; a public checker
    executable's digest alone cannot identify this execution path.
    """
    sysroot = Path(_lake_output("lean", "--print-prefix")).resolve(strict=True)
    # Lake can list optional package build directories that do not exist yet. Lean
    # skips them; the inventory will detect any local module introduced there later.
    search = tuple(Path(path).resolve() for path in _lake_output("printenv", "LEAN_PATH").split(os.pathsep))
    modules = tuple(_modules(search))
    expected = Path(*required_module.split(".")).with_suffix(".olean")
    if not any((directory / expected).is_file() for directory in search):
        raise ValueError(f"proof artifacts have not been built: {required_module}")
    executable = (sysroot / "bin/lean").resolve(strict=True)
    auditor = lean_exe().with_name("merlin-lean-audit").resolve(strict=True)
    entry = entry.resolve(strict=True)
    files = {
        executable,
        entry,
        lean_exe().resolve(strict=True),
        auditor,
        *(lean_root() / name for name in ("lean-toolchain", "lakefile.toml", "lake-manifest.json", "SPEC_LOCK")),
        *modules,
    }
    runtime = Runtime(
        executable, auditor, entry, sysroot, search, modules, tuple(_file_ref(path) for path in sorted(files))
    )
    runtime.assert_current()
    return runtime
