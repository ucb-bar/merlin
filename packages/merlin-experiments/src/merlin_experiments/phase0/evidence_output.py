"""Immutable publication of already selected Phase 0 evidence bytes."""

from pathlib import Path


def _materialize_evidence(root: Path, outputs: dict[str, bytes]) -> None:
    """Validate all destinations before adding immutable saved evidence members."""
    for name in outputs:
        member = Path(name)
        if member.is_absolute() or not member.parts or ".." in member.parts:
            raise ValueError(f"invalid evidence output member: {name}")
        if member.parts[0] == "private":
            raise ValueError(f"evidence output may not occupy the private source snapshot: {name}")
        path = root / name
        if path.is_symlink():
            raise ValueError(f"evidence output may not be symlinked: {path}")
        if path.exists() and path.read_bytes() != outputs[name]:
            raise FileExistsError(f"changed evidence output already exists: {path}")
        if any(parent.is_symlink() for parent in path.parents):
            raise ValueError(f"evidence output traverses a symlink: {path}")
    for name, raw in outputs.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        with path.open("xb") as handle:
            handle.write(raw)

