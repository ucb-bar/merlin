"""Read explicitly authored numerical assumptions at the experiment edge.

This is not hardware discovery or a merge of generated/private recipe sidecars.
Frozen callers supply a profile path resolved by their existing snapshot verifier;
this reader neither creates a seal nor falls back from frozen to live inputs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from merlin.targetgen.corpus_spec import profile_datapath


def numeric_profile_path(declaration: str | None, *, repo: Path) -> Path | None:
    """Resolve a declared file, without target-name or sibling-file discovery."""
    if declaration is None:
        return None
    if not isinstance(declaration, str) or not declaration.strip():
        raise ValueError("numeric_profile must be a non-empty file path")
    path = Path(declaration)
    if ".." in path.parts or not path.name:
        raise ValueError("numeric_profile must not contain parent traversal")
    return path if path.is_absolute() else Path(repo).absolute() / path


def load_declared_numeric_policy(
    experiment, *, repo: Path, frozen_profile: Path | None = None
) -> tuple[dict | None, dict | None]:
    """Return complete declared arithmetic/comparison policy and byte provenance.

    Omission is distinct from an empty declaration: callers decide whether their
    numerical regime permits omission. The hash describes the exact bytes parsed,
    not an assertion that the assumptions have been verified against hardware.
    """
    declared = getattr(experiment, "numeric_profile", None)
    source = numeric_profile_path(declared, repo=repo)
    if source is None:
        if frozen_profile is not None:
            raise ValueError("a frozen numeric profile requires an explicit declaration")
        return None, None
    source = Path(frozen_profile) if frozen_profile is not None else source
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"declared numeric profile is missing, symlinked or not a file: {source}")
    payload = source.read_bytes()
    document = yaml.safe_load(payload)
    if not isinstance(document, dict) or not isinstance(document.get("datapath"), dict):
        raise ValueError("declared numeric profile requires a datapath mapping")
    policy = profile_datapath(document, numeric_only=True)
    if not policy:
        raise ValueError("declared numeric profile has no numerical assumptions")
    return policy, {
        "declaration": declared,
        "path": str(source.absolute()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "source": "frozen-input" if frozen_profile is not None else "authored-input",
        "scope": "declared-numerical-assumptions",
        "hardware_verified": False,
    }
