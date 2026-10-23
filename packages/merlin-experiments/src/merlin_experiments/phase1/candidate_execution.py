"""Native selection of an existing frozen compiler-command policy.

Operator inputs select tools and source masks; this owner never freezes a live
runtime on lookup or treats compiler products as grading authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def selection(path: Path) -> tuple[dict, dict]:
    """Read the explicit operator selection and verify its frozen resource owner."""
    from merlin_experiments.phase2.qualification_policy import restore

    path = Path(path)
    if ".." in path.parts:
        raise ValueError("candidate compilation selection must have a canonical spelling")
    path = path.absolute()
    if path.resolve(strict=True) != path or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("candidate compilation selection must be an ordinary canonical file")
    payload = path.read_bytes()
    record = json.loads(payload)
    if not isinstance(record, dict) or set(record) != {"root", "policy", "bwrap", "workspace_root"}:
        raise ValueError("candidate compilation selection has an invalid field roster")
    for key in ("root", "workspace_root"):
        if type(record[key]) is not str:
            raise ValueError("candidate compilation selection has an invalid path")
        owner = Path(record[key])
        if (
            not owner.is_absolute()
            or ".." in owner.parts
            or owner.resolve(strict=True) != owner
            or any(p.is_symlink() for p in (owner, *owner.parents))
            or not owner.is_dir()
        ):
            raise ValueError("candidate compilation selection has an invalid directory owner")
    binary = record["bwrap"]
    if not isinstance(binary, dict) or set(binary) != {"path", "sha256"}:
        raise ValueError("candidate compilation selection lacks its sandbox executable identity")
    if type(binary["path"]) is not str or type(binary["sha256"]) is not str:
        raise ValueError("candidate compilation sandbox identity is malformed")
    executable = Path(binary["path"])
    if (
        not executable.is_absolute()
        or ".." in executable.parts
        or executable.resolve(strict=True) != executable
        or any(p.is_symlink() for p in (executable, *executable.parents))
        or not executable.is_file()
        or hashlib.sha256(executable.read_bytes()).hexdigest() != binary["sha256"]
    ):
        raise ValueError("candidate compilation sandbox executable changed")
    restored = restore(Path(record["root"]), record["policy"])
    identity = {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "policy_sha256": restored.record["sha256"],
        "bwrap_sha256": binary["sha256"],
    }
    return record, identity


def selection_record(context) -> dict | None:
    path = getattr(context, "candidate_policy", None)
    return None if path is None else selection(path)[1]


def verify_selection(context, expected) -> None:
    if selection_record(context) != expected:
        raise RuntimeError("frozen invocation candidate compilation selection changed or is absent")


def functional_codegen_scope(context, *, public_root: Path | None, hidden_root: Path | None) -> dict | None:
    """Identify a selected functional source-product roster, never certify codegen.

    The command-buffer production smoke covers a different ABI. Missing or
    ambiguous declarations retain that legacy gate instead of earning an exemption.
    Callers supply the already verified frozen grading roots, not live inputs.
    """
    identity = selection_record(context)
    if identity is None or public_root is None or hidden_root is None:
        return None
    import yaml

    from merlin.targetgen.core_aten_capsules import is_full_call

    hashes, declared = [], set()
    try:
        for root in (Path(public_root), Path(hidden_root)):
            if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
                return None
            files = sorted(root.rglob("capsule.yaml"))
            if not files:
                return None
            for path in files:
                if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                    return None
                payload = path.read_bytes()
                capsule = yaml.safe_load(payload)
                if not isinstance(capsule, dict) or not is_full_call(capsule):
                    return None
                tiers = capsule.get("required_oracle_tiers")
                if (
                    not isinstance(tiers, list)
                    or not tiers
                    or any(not isinstance(tier, str) or tier not in {"L0", "L1", "L2"} for tier in tiers)
                ):
                    return None
                declared.update(tiers)
                hashes.append(hashlib.sha256(payload).hexdigest())
    except (OSError, ValueError, AttributeError, TypeError, yaml.YAMLError):
        return None
    if "L2" not in declared:
        return None
    return {
        "status": "legacy_not_run",
        "selected_path": "source_product_full_call",
        "candidate_compilation": identity,
        "required_oracle_tiers": sorted(declared),
        "n_capsules": len(hashes),
        "capsule_bytes_sha256": hashlib.sha256(json.dumps(sorted(hashes)).encode()).hexdigest(),
    }


def transport(context):
    """Construct the selected transport; no compiler or probe is executed here."""
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin_experiments.phase1.candidate_products import CandidateProductTransport
    from merlin_experiments.phase2.qualification_policy import restore

    path = getattr(context, "candidate_policy", None)
    if path is None:
        return None
    record, _ = selection(path)
    private = (Path(path),)
    if context.reports.exists():
        private += (context.reports,)
    workspace = Path(record["workspace_root"])
    if any(
        workspace == owner or workspace.is_relative_to(owner) or owner.is_relative_to(workspace) for owner in private
    ):
        raise ValueError("candidate compiler workspace overlaps its private invocation owners")
    return CandidateProductTransport(
        load_target_experiment(context.descriptor),
        inputs=restore(Path(record["root"]), record["policy"]),
        bwrap_binary=Path(record["bwrap"]["path"]),
        private_paths=private,
        workspace_root=workspace,
    )
