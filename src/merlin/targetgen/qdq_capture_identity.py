"""Observe the selected Q/DQ producer and compare installed Torch with a pinned wheel.

These are reproducible source observations, not a sandbox or execution attestation.
No framework is imported until the explicitly selected capture calls this module.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from merlin.common.paths import repo_root


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def producer_identity(checkout: Path) -> dict:
    import m2m

    from merlin.integrations.model2mlir import capture_pin

    root = checkout.resolve()
    if Path(m2m.__file__).resolve() != root / "m2m/__init__.py":
        raise ValueError("imported model2MLIR is not the selected producer checkout")
    pin = capture_pin(root)
    if not pin["ok"]:
        raise ValueError(f"Q/DQ capture producer differs from software pin: {pin['citation']}")
    git = ["git", "-C", str(root)]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    entries = subprocess.check_output([*git, "ls-tree", "-r", "HEAD", "m2m"]).decode().splitlines()
    sources = {}
    for entry in entries:
        metadata, name = entry.split("\t", 1)
        mode, kind, expected = metadata.split()
        path = root / name
        if mode not in {"100644", "100755"} or kind != "blob" or path.is_symlink():
            raise ValueError(f"unsupported producer source member: {name}")
        raw = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if actual != expected:
            raise ValueError(f"producer source differs from its commit: {name}")
        sources[name] = sha(raw)
    # An untracked Python module could shadow a tracked import.
    actual_python = {p.relative_to(root).as_posix() for p in (root / "m2m").rglob("*.py")}
    if actual_python != {name for name in sources if name.endswith(".py")}:
        raise ValueError("producer Python source inventory differs from its commit")
    for name, module in list(sys.modules.items()):
        if name == "m2m" or name.startswith("m2m."):
            path = getattr(module, "__file__", None)
            if path is not None and not Path(path).resolve().is_relative_to(root / "m2m"):
                raise ValueError(f"foreign imported producer module: {name}")
    return {"root": str(root), "commit": commit, "sources": sources}


def torch_identity(wheel: Path) -> dict:
    import torch

    requirements = repo_root() / "build_tools/toolchains/lean-torch-requirements.txt"
    row = next(line for line in requirements.read_text().splitlines() if line.startswith("torch @ "))
    expected = row.rsplit("#sha256=", 1)[1]
    wheel_digest = sha(wheel.read_bytes())
    if wheel_digest != expected:
        raise ValueError("selected Torch wheel differs from the pinned source-oracle wheel")
    distribution = importlib.metadata.distribution("torch")
    if torch.__version__ != "2.10.0+cpu" or distribution.version != torch.__version__:
        raise ValueError("Q/DQ capture requires the pinned PyTorch 2.10.0+cpu")
    if Path(torch.__file__).resolve() != Path(distribution.locate_file("torch/__init__.py")).resolve():
        raise ValueError("imported Torch differs from the selected installed distribution")
    members = {}
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.filename.endswith(".dist-info/RECORD"):
                continue
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts or ".data/" in name:
                raise ValueError(f"unsupported wheel member: {name}")
            expected_member = sha(archive.read(info))
            installed = Path(distribution.locate_file(name))
            if installed.is_symlink() or sha(installed.read_bytes()) != expected_member:
                raise ValueError(f"installed Torch differs from pinned wheel: {name}")
            members[name] = expected_member
    return {
        "version": torch.__version__,
        "git_version": torch.version.git_version,
        "wheel": str(wheel.absolute()),
        "wheel_sha256": wheel_digest,
        "requirements_sha256": sha(requirements.read_bytes()),
        "installed_members": members,
    }


def begin(checkout: Path, wheel: Path) -> dict:
    return {
        "producer": producer_identity(checkout),
        "torch": torch_identity(wheel),
        "python": {"executable": sys.executable, "version": sys.version},
        "checker_sources": checker_sources(),
    }


def checker_sources() -> dict:
    return {
        name: sha(Path(__file__).with_name(name).read_bytes())
        for name in (
            "qdq_capture.py",
            "qdq_capture_identity.py",
            "_m2m_capture_worker.py",
            "_m2m_capture_artifacts.py",
        )
    }


def recheck(identity: dict) -> None:
    """Reopen sources at completion; a saved identity cannot authorize another capture."""
    if identity["producer"] != producer_identity(Path(identity["producer"]["root"])):
        raise ValueError("producer changed during Q/DQ capture")
    if identity["torch"] != torch_identity(Path(identity["torch"]["wheel"])):
        raise ValueError("Torch changed during Q/DQ capture")
    if identity.get("checker_sources") != checker_sources():
        raise ValueError("Q/DQ capture checker sources changed")


def digest(identity: dict) -> str:
    return sha(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
