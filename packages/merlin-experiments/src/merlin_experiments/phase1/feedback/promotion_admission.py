"""Current-source admission of automatic promotions; never qualification authority.

Obsolete requests and their original source/ledger records remain untouched. A
deferred request becomes eligible again if its source or proven dependencies
return. Explicit caller requests do not use this policy.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from merlin.targetgen.oracle_schedule import CapsuleState, Verdict, valid_execution_digest


@dataclass(frozen=True)
class Source:
    digest: str
    components: dict[str, str]


def read_note(path: Path):
    """Bounded scheduling metadata read; candidate channel links confer no access."""
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 8192:
            return None
        with os.fdopen(descriptor) as stream:
            descriptor = None
            return json.loads(stream.read(8192))
    except (OSError, ValueError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def write_note(path: Path, document: dict) -> None:
    """Atomically replace metadata without following an author-created link."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", prefix=".promotion-admission-", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def observe_source(workspace: Path) -> Source:
    from . import promotion

    if not (Path(workspace) / "submission").is_dir():
        raise ValueError("promotion source is unavailable")
    digest, components, _ = promotion.submission_digests(workspace)
    return Source(digest, components)


def current_execution(state: dict, name: str, tier: str, digest: str, *, instrument=None) -> str | None:
    """An existing host-recorded loop result explicitly belonging to this source."""
    from . import promotion

    if not valid_execution_digest(instrument):
        return None
    for entry in reversed(list(promotion._slots_ro(state, name, tier).values())):
        if (isinstance(entry, dict) and entry.get("status") == "pass"
                and entry.get("digest") == digest and entry.get("instrument") == instrument
                and valid_execution_digest(entry.get("execution_digest"))):
            return entry["execution_digest"]
    return None


def reason(
    name: str, *, current: Source, original: Source,
    current_instrument: str | None, original_instrument: str | None,
    original_execution: str | None = None, observed_execution: str | None = None,
) -> str | None:
    """None admits work; a string defers it without changing any verdict.

    The dependency union includes newly added/removed components and the
    unattributed bucket. Missing narrower evidence falls through the existing
    scheduler's conservative whole-source comparison.
    """
    if not current.digest or not original.digest:
        return "source identity unavailable"
    if not valid_execution_digest(current_instrument) or not valid_execution_digest(original_instrument):
        return "grading instrument identity unavailable"
    if current_instrument != original_instrument:
        return "grading instrument changed"
    state = CapsuleState(
        name=name, digest=current.digest, components=current.components,
        depends_on=tuple(sorted(current.components.keys() | original.components.keys())) or None,
        execution_digest=observed_execution,
        verdicts={"cert": Verdict("pending", original.digest, original.components, original_execution)},
    )
    invalid = state.invalidated_by("cert")
    return "superseded source: " + ", ".join(str(item) for item in invalid) if invalid else None


def queued_reason(
    workspace: Path, request: dict, *, loop_tier: str, cert_tier: str,
    instrument: str | None, loop_instrument: str | None, current: Source, state: dict,
    snapshots: dict[str, Source],
) -> str | None:
    """Authenticate the protected original snapshot and its outstanding ledger slot."""
    if not request.get("promoted"):
        return None
    from . import promotion
    from .snapshots import promotion_snapshot_path

    digest, identity = request.get("submission_digest"), request.get("identity")
    if not isinstance(digest, str) or not digest or not isinstance(identity, str):
        return "enqueue source identity unavailable"
    token = request.get("submission_snapshot")
    frozen = promotion_snapshot_path(workspace, token)
    if frozen is None:
        return "protected enqueue snapshot unavailable"
    try:
        original = snapshots.get(token)
        if original is None:
            original = observe_source(frozen)
            snapshots[token] = original
    except (OSError, ValueError):
        return "protected enqueue snapshot unavailable"
    if original.digest != digest:
        return "protected enqueue snapshot identity changed"
    names = str(request.get("capsules") or "").split(",")
    if not names or any(not name or name == "all" for name in names):
        return "automatic promotion has no explicit capsule identity"
    for name in names:
        entry = promotion._slots_ro(state, name, cert_tier).get(identity)
        if (not isinstance(entry, dict) or entry.get("status") != "pending"
                or entry.get("digest") != digest or promotion.record_identity(entry) != identity):
            return "protected outstanding promotion identity unavailable"
        refusal = reason(
            name, current=current, original=original, current_instrument=instrument,
            original_instrument=entry.get("instrument"), original_execution=entry.get("execution_digest"),
            observed_execution=current_execution(state, name, loop_tier, current.digest, instrument=loop_instrument),
        )
        if refusal:
            return refusal
    return None
