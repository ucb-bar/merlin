"""Read explicitly authored numerical assumptions at the experiment edge.

This is not hardware discovery or a merge of generated/private recipe sidecars.
Frozen callers supply a profile path resolved by their existing snapshot verifier;
this reader neither creates a seal nor falls back from frozen to live inputs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import yaml

from merlin.targetgen.corpus_spec import profile_datapath
from merlin.targetgen.software_spec import (
    numerical_datapath,
    software_spec_path_for_recipe,
    validate_software_spec,
)


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
    experiment,
    *,
    repo: Path,
    frozen_profile: Path | None = None,
    frozen_software_spec: Path | None = None,
    frozen_resolver: Callable[[Path], Path] | None = None,
) -> tuple[dict | None, dict | None]:
    """Return complete declared arithmetic/comparison policy and byte provenance.

    Omission is distinct from an empty declaration: callers decide whether their
    numerical regime permits omission. The hash describes the exact bytes parsed,
    not an assertion that the assumptions have been verified against hardware.
    """
    declared = getattr(experiment, "numeric_profile", None)
    original = numeric_profile_path(declared, repo=repo)
    if original is None:
        if frozen_profile is not None or frozen_software_spec is not None:
            raise ValueError("a frozen numeric profile requires an explicit declaration")
        return None, None
    source = Path(frozen_profile) if frozen_profile is not None else original
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"declared numeric profile is missing, symlinked or not a file: {source}")
    payload = source.read_bytes()
    document = yaml.safe_load(payload)
    if not isinstance(document, dict) or not isinstance(document.get("datapath"), dict):
        raise ValueError("declared numeric profile requires a datapath mapping")
    policy = profile_datapath(document, numeric_only=True)
    selected_spec = software_spec_path_for_recipe(original, document=document)
    spec_identity = None
    if selected_spec is not None:
        selected = selected_spec
        if frozen_profile is not None:
            if frozen_software_spec is not None:
                selected = Path(frozen_software_spec)
            elif frozen_resolver is not None:
                selected = Path(frozen_resolver(selected_spec))
            else:
                raise ValueError("selected software spec requires an explicitly verified frozen input")
        elif frozen_software_spec is not None:
            raise ValueError("a frozen software spec requires its frozen numeric profile")
        if selected.is_symlink() or not selected.is_file():
            raise ValueError("selected software spec is missing, symlinked or not a file")
        spec_payload = selected.read_bytes()
        spec = validate_software_spec(
            yaml.safe_load(spec_payload),
            target=getattr(experiment, "target", None),
            source=selected,
        )
        axes = numerical_datapath(spec)
        conflicts = sorted(key for key, value in axes.items() if key in policy and policy[key] != value)
        if conflicts:
            raise ValueError(f"numeric profile conflicts with selected software spec: {conflicts}")
        policy.update(axes)
        spec_identity = {
            "declaration": str(selected_spec),
            "path": str(selected.absolute()),
            "schema": spec["schema"],
            "target": spec["target"],
            "status": spec["status"],
            "sha256": hashlib.sha256(spec_payload).hexdigest(),
            "size_bytes": len(spec_payload),
            "source": "frozen-input" if frozen_profile is not None else "authored-input",
        }
    elif frozen_software_spec is not None:
        raise ValueError("frozen software spec is not declared by the numeric profile")
    if not policy:
        raise ValueError("declared numeric profile has no numerical assumptions")
    identity = {
        "declaration": declared,
        "path": str(source.absolute()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "source": "frozen-input" if frozen_profile is not None else "authored-input",
        "scope": "declared-numerical-assumptions",
        "hardware_verified": False,
    }
    if spec_identity is not None:
        identity["software_spec"] = spec_identity
    return policy, identity
