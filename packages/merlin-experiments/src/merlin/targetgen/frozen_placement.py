"""The host/accelerator split of a whole-program capsule, read from its frozen Phase 0 placement.

Phase 1 grades the author's accelerator compiler only. Every operation Phase 0 placed on the host is
compiled by Merlin's own generic host codegen, so the split between the two must not come from the
candidate, and must not drift from what Phase 0 admitted. This module reads the ``software_screen``
Phase 0 froze into the sealed ``capsule.yaml``, binds it to the capsule's program bytes, and checks
that Merlin's own device groups and the independent eligibility census hold exactly the operations
the frozen placement puts on the accelerator. It grants no execution, numerical or tier evidence.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ACCELERATOR = "accelerator"
HOST = "host"


class FrozenPlacementRefusal(ValueError):
    """The frozen placement is absent, unbound, or disagrees with the split Merlin would build."""


def _program(declaration: Mapping[str, Any], directory: Path) -> Path:
    name = declaration.get("interface_mlir") or declaration.get("linalg_mlir")
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise FrozenPlacementRefusal("frozen capsule names no contained program")
    program = Path(directory) / name
    if program.is_symlink() or not program.is_file():
        raise FrozenPlacementRefusal("frozen capsule program is absent or indirect")
    return program


def read(declaration: Mapping[str, Any], directory: str | Path) -> dict[str, Any]:
    """The frozen per-lane operation counts, bound to the capsule's own program bytes.

    Refuses a screen that is missing, not admitted, bound to other bytes, or that places nothing on
    the accelerator: such a capsule has no author-owned part to split off.
    """
    screen = declaration.get("software_screen")
    if not isinstance(screen, Mapping):
        raise FrozenPlacementRefusal("capsule carries no frozen Phase 0 placement")
    if screen.get("status") != "admitted":
        raise FrozenPlacementRefusal("frozen Phase 0 placement is not admitted")
    program = _program(declaration, Path(directory))
    digest = hashlib.sha256(program.read_bytes()).hexdigest()
    if screen.get("program_sha256") != digest:
        raise FrozenPlacementRefusal("frozen Phase 0 placement is bound to other program bytes")
    decisions = screen.get("decisions")
    placements = screen.get("placements")
    if not isinstance(decisions, list) or not isinstance(placements, Mapping):
        raise FrozenPlacementRefusal("frozen Phase 0 placement has no per-operation decisions")
    lanes: dict[str, Counter] = {ACCELERATOR: Counter(), HOST: Counter()}
    rows: Counter = Counter()
    for row in decisions:
        if not isinstance(row, Mapping):
            raise FrozenPlacementRefusal("frozen Phase 0 placement decision is malformed")
        lane, operation, count = row.get("placement"), row.get("mlir_operation"), row.get("count")
        if row.get("status") != "admitted" or lane not in lanes:
            raise FrozenPlacementRefusal("frozen Phase 0 placement holds an operation without one admitted lane")
        if not isinstance(operation, str) or not operation or type(count) is not int or count < 1:
            raise FrozenPlacementRefusal("frozen Phase 0 placement decision has no operation or count")
        lanes[lane][operation] += count
        rows[lane] += 1
    if dict(sorted(rows.items())) != dict(sorted(placements.items())):
        raise FrozenPlacementRefusal("frozen Phase 0 placement counts disagree with its decisions")
    if not lanes[ACCELERATOR]:
        raise FrozenPlacementRefusal("frozen Phase 0 placement puts no operation on the accelerator")
    return {
        "program_sha256": digest,
        "placements": dict(sorted(placements.items())),
        "accelerator_operations": dict(sorted(lanes[ACCELERATOR].items())),
        "host_operations": dict(sorted(lanes[HOST].items())),
    }


def requires_merlin_host(declaration: Mapping[str, Any]) -> bool:
    """Whether a whole-program capsule's frozen placement gives the author accelerator work.

    Only the declaration is read: whether its bytes are bound is checked by :func:`read` on the
    route this selects, which fails closed.
    """
    if declaration.get("kind") != "model":
        return False
    placements = (declaration.get("software_screen") or {}).get("placements")
    if not isinstance(placements, Mapping):
        return False
    count = placements.get(ACCELERATOR)
    return type(count) is int and count > 0


def operation_name(op) -> str:
    """The operation name an MLIR operation is recorded under in a Phase 0 decision."""
    return op.op_name.data if op.name == "builtin.unregistered" else op.name


def device_operations(groups: Iterable, host_placement) -> dict[str, int]:
    """Per-operation counts of every member of Merlin's non-host groups."""
    counts: Counter = Counter()
    for group in groups:
        if group.placement == host_placement:
            continue
        for member in group.members:
            counts[operation_name(member)] += 1
    return dict(sorted(counts.items()))


def require_split(
    frozen: Mapping[str, Any],
    *,
    source_sha256: str,
    device: Mapping[str, int],
    eligible: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Refuse a build whose device groups or eligibility differ from the frozen accelerator split.

    ``device`` counts the members of Merlin's own device groups of the routed source; ``eligible``,
    when given, counts the independently eligible source operations. Either disagreeing with the
    frozen accelerator operations means work would cross lanes: accelerator work onto Merlin's host,
    or host work into a group the author's compiler is handed.
    """
    if source_sha256 != frozen.get("program_sha256"):
        raise FrozenPlacementRefusal("routed source differs from the program the frozen placement is bound to")
    expected = dict(frozen.get("accelerator_operations") or {})
    if dict(device) != expected:
        raise FrozenPlacementRefusal(
            f"device groups hold {dict(device)} but the frozen Phase 0 placement puts {expected} on the "
            "accelerator; host placement is Merlin-supplied and may not cross lanes"
        )
    if eligible is not None and dict(eligible) != expected:
        raise FrozenPlacementRefusal(
            f"independent eligibility finds {dict(eligible)} but the frozen Phase 0 placement puts "
            f"{expected} on the accelerator"
        )
    return {
        "status": "matched",
        "authority": "frozen Phase 0 software_screen",
        "program_sha256": frozen["program_sha256"],
        "accelerator_operations": expected,
        "host_operations": dict(frozen.get("host_operations") or {}),
        "scope": "host-placed operations are compiled by Merlin's host package; the candidate builds "
        "only the device groups",
    }
