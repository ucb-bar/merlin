"""Fail-closed identity checks shared by every ExecuTorch baseline precision.

An exported ``.pte`` is interpreted by a runtime built from the ExecuTorch source checkout.  A
Python exporter wheel left over from a different checkout can produce an ABI/semantic mismatch that
ordinary correctness smoke tests do not prove absent.  The paper harness therefore requires the
full commit reported by ``executorch.version.git_version`` to equal ``git rev-parse HEAD`` in the
runtime source tree.  Missing identity is a blocker, not permission to continue.

This check covers FP32 and quantized exports alike.  It intentionally does not claim to identify an
already-built runner binary: those need a build-time sidecar binding runner SHA-256 to this identity.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


#: A full git object name: a length plus a character set. Spelled structurally because the repo forbids
#: regex in library code, and because `^[0-9a-f]{40}$` cannot be checked by eye without counting braces.
#: That matters here more than most places: this predicate is what decides whether a measured result is
#: attributed to a known revision at all, and a validator nobody can read is the wrong tool for a
#: provenance check. `.lower()` is applied by both callers before the test, so uppercase input is
#: normalised rather than rejected -- the same behaviour the pattern had.
_FULL_GIT_SHA_LEN = 40
_HEX_LOWER = "0123456789abcdef"
_PROBE_MARKER = "MERLIN_EXECUTORCH_IDENTITY="


def _is_full_git_sha(value: str) -> bool:
    """True for exactly ``_FULL_GIT_SHA_LEN`` lowercase hex digits, nothing else.

    (A sibling check for 64-hex digests lives in ``executorch_session``; if a third caller appears the
    two should become one helper rather than a third copy.)
    """
    return len(value) == _FULL_GIT_SHA_LEN and all(c in _HEX_LOWER for c in value)


class ExecuTorchIdentityError(RuntimeError):
    """Exporter/source identity is absent, malformed, or mismatched."""


@dataclass(frozen=True)
class ExecuTorchIdentity:
    exporter_version: str
    exporter_git_sha: str
    source_git_sha: str

    @property
    def matches(self) -> bool:
        return self.exporter_git_sha == self.source_git_sha

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "exporter_version": self.exporter_version,
            "exporter_git_sha": self.exporter_git_sha,
            "source_git_sha": self.source_git_sha,
            "matches": self.matches,
        }


def _exporter_probe(python: Path) -> dict[str, str]:
    script = (
        "import json\n"
        "from executorch import version\n"
        "payload = {'version': str(version.__version__), "
        "'git_sha': str(version.git_version or '')}\n"
        f"print({_PROBE_MARKER!r} + json.dumps(payload))\n"
    )
    env = dict(os.environ)
    # Do not let the sibling ``baselines/executorch.py`` shadow the installed package.
    env.pop("PYTHONPATH", None)
    try:
        proc = subprocess.run(
            [str(python), "-c", script], capture_output=True, text=True, timeout=120,
            cwd=str(python.parent), env=env,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ExecuTorchIdentityError(f"cannot probe ExecuTorch exporter at {python}: {error}") \
            from error
    if proc.returncode:
        detail = (proc.stdout + proc.stderr)[-600:].replace("\n", " ")
        raise ExecuTorchIdentityError(
            f"cannot import ExecuTorch exporter at {python}: {detail}")
    marked = [line[len(_PROBE_MARKER):] for line in proc.stdout.splitlines()
              if line.startswith(_PROBE_MARKER)]
    if len(marked) != 1:
        raise ExecuTorchIdentityError(
            "ExecuTorch exporter emitted no unique machine-readable build identity")
    try:
        value = json.loads(marked[0])
    except json.JSONDecodeError as error:
        raise ExecuTorchIdentityError("ExecuTorch exporter build identity is malformed") from error
    if not isinstance(value, dict):
        raise ExecuTorchIdentityError("ExecuTorch exporter build identity is not an object")
    return {"version": str(value.get("version", "")), "git_sha": str(value.get("git_sha", ""))}


def _source_commit(source: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ExecuTorchIdentityError(
            f"cannot resolve ExecuTorch runtime source identity at {source}: {error}") from error
    commit = proc.stdout.strip().lower() if proc.returncode == 0 else ""
    if not _is_full_git_sha(commit):
        detail = proc.stderr[-400:].replace("\n", " ")
        raise ExecuTorchIdentityError(
            f"ExecuTorch runtime source has no full git identity at {source}: {detail}")
    return commit


def inspect_executorch_identity(python: str | Path, source: str | Path) -> ExecuTorchIdentity:
    """Return the exporter/source identity, rejecting missing or abbreviated hashes."""
    # Do not ``resolve`` the venv's ``bin/python`` symlink: following it selects the base interpreter
    # and silently probes the wrong environment.
    python, source = Path(python).absolute(), Path(source).resolve()
    if not python.is_file():
        raise ExecuTorchIdentityError(f"ExecuTorch exporter Python is absent: {python}")
    if not source.is_dir():
        raise ExecuTorchIdentityError(f"ExecuTorch runtime source is absent: {source}")
    exported = _exporter_probe(python)
    exporter_sha = exported["git_sha"].strip().lower()
    if not _is_full_git_sha(exporter_sha):
        raise ExecuTorchIdentityError(
            "installed ExecuTorch exposes no reliable full git identity: "
            f"version={exported['version']!r}, git_version={exported['git_sha']!r}")
    return ExecuTorchIdentity(
        exporter_version=exported["version"], exporter_git_sha=exporter_sha,
        source_git_sha=_source_commit(source),
    )


def require_matching_executorch(python: str | Path, source: str | Path) -> ExecuTorchIdentity:
    """Require one exact ExecuTorch commit for export and runtime-source compilation."""
    identity = inspect_executorch_identity(python, source)
    if not identity.matches:
        raise ExecuTorchIdentityError(
            "ExecuTorch exporter/runtime source identity mismatch: "
            f"exporter={identity.exporter_git_sha} ({identity.exporter_version}), "
            f"runtime_source={identity.source_git_sha}. This blocks FP32 and quantized paper cells.")
    return identity
