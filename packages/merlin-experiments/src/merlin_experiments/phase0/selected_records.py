"""Stable ordinary JSON inputs; their semantic admission belongs to each consumer."""

from pathlib import Path

from merlin.common import strict_json


def read_mapping(path: Path, *, subject: str) -> dict:
    """Read an explicit, bounded file without following aliases or concurrent changes."""
    path = Path(path).absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents) or not path.is_file():
        raise ValueError(f"{subject} must be an ordinary file")
    before = path.stat()
    if before.st_size > 1024 * 1024:
        raise ValueError(f"{subject} exceeds the one-MiB record budget")
    raw = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError(f"{subject} changed during reading")
    data = strict_json.loads(raw, max_bytes=1024 * 1024)
    if not isinstance(data, dict):
        raise ValueError(f"{subject} must be a mapping")
    return data
