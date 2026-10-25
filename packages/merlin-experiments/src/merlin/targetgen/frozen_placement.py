"""The host/accelerator split of a whole-program capsule, read from its frozen Phase 0 placement.

Phase 0 fixes the required accelerator work; Merlin supplies the remaining host computation.
Merlin's capability-checked groups may also offload optional operations, including a contraction's
readout or layout plumbing and standalone supported groups. They do not change the required accelerator census
or authorize scalar tensor computation in a device kernel. This module grants no execution,
numerical or tier evidence.
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
    source_placements: dict[int, str] = {}
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
        if "source_op_indices" in row:
            indices = _source_indices(row["source_op_indices"])
            if len(indices) != count or source_placements.keys() & indices:
                raise FrozenPlacementRefusal("frozen decisions have duplicate or inconsistent source identities")
            source_placements.update(dict.fromkeys(indices, lane))
    if dict(sorted(rows.items())) != dict(sorted(placements.items())):
        raise FrozenPlacementRefusal("frozen Phase 0 placement counts disagree with its decisions")
    if not lanes[ACCELERATOR]:
        raise FrozenPlacementRefusal("frozen Phase 0 placement puts no operation on the accelerator")
    return {
        "program_sha256": digest,
        "placements": dict(sorted(placements.items())),
        "accelerator_operations": dict(sorted(lanes[ACCELERATOR].items())),
        "host_operations": dict(sorted(lanes[HOST].items())),
        **({"source_placements": source_placements} if source_placements else {}),
    }


def _source_indices(value) -> set[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(index) is not int or index < 0 for index in value)
        or len(set(value)) != len(value)
    ):
        raise FrozenPlacementRefusal("UNKNOWN: missing or ambiguous source operation identities")
    return set(value)


def require_same_region(frozen, composition, evidence) -> dict[str, Any]:
    """Bind exact source members to one independently routed kernel AND region.

    The caller owns the frozen source census, routed sidecar and host def-use
    audit. Candidate counts or volunteered placement claims are not evidence.
    An explicit empty host-consumer list means the host def-use audit ran.
    """
    source = frozen.get("program_sha256")
    if (
        not isinstance(source, str)
        or len(source) != 64
        or evidence.get("source_sha256") != source
        or evidence.get("routed_source_sha256") != source
    ):
        raise FrozenPlacementRefusal("UNKNOWN: routed composition has different source bytes")
    members = _source_indices(composition.get("source_op_indices"))
    producers = _source_indices(composition.get("producer_source_op_indices"))
    if not producers <= members:
        raise FrozenPlacementRefusal("composition producer is outside its source chain")
    placements = frozen.get("source_placements")
    if not isinstance(placements, Mapping) or any(
        placements.get(index, placements.get(str(index))) != ACCELERATOR for index in members
    ):
        raise FrozenPlacementRefusal("UNKNOWN: every chain member needs frozen accelerator placement")
    if evidence.get("host_intermediate_consumers") != []:
        raise FrozenPlacementRefusal("UNKNOWN: host consumes a chain intermediate or lacks a def-use audit")
    routed = evidence.get("routed")
    if not isinstance(routed, list) or not routed:
        raise FrozenPlacementRefusal("UNKNOWN: composition has no routed source bindings")
    owners = []
    for row in routed:
        indices = _source_indices(row.get("source_op_indices"))
        if members & indices:
            owners.append(row)
    if len(owners) != 1 or not members <= _source_indices(owners[0]["source_op_indices"]):
        raise FrozenPlacementRefusal("chain members are split across kernels or regions")
    owner = owners[0]
    if any(not isinstance(owner.get(key), str) or not owner[key] for key in ("symbol", "region_id")):
        raise FrozenPlacementRefusal("UNKNOWN: routed chain lacks a kernel/region identity")
    if sum(row.get("symbol") == owner["symbol"] for row in routed) != 1:
        raise FrozenPlacementRefusal("UNKNOWN: reused kernel symbol lacks per-invocation region attribution")
    return {
        "status": "matched",
        "symbol": owner["symbol"],
        "region_id": owner["region_id"],
        "source_op_indices": sorted(members),
    }


def host_chain_consumers(source_text, frozen, composition) -> list[int]:
    """Derive host uses of chain intermediates from the original source SSA.

    Final chain results may leave the device. A value consumed by another chain
    member is an intermediate; its other uses cannot become host work merely
    because the accelerator's aggregate operation counts still match.
    """
    from merlin.common import mlir_query as mq

    module = mq.parse(source_text)
    functions = [op for op in module.body.block.ops if mq.op_name(op) == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise FrozenPlacementRefusal("UNKNOWN: host chain audit requires one original source block")
    operations = [op for op in functions[0].body.blocks[0].ops if mq.op_name(op) != "func.return"]
    members = _source_indices(composition.get("source_op_indices"))
    if not members <= set(range(len(operations))):
        raise FrozenPlacementRefusal("UNKNOWN: chain identities are outside the original source inventory")
    indices = {op: index for index, op in enumerate(operations)}
    chain = {operations[index] for index in members}
    intermediates = [
        result for op in chain for result in op.results if any(use.operation in chain for use in result.uses)
    ]
    consumers = set()
    pending, visited = list(intermediates), set()
    placements = frozen.get("source_placements") or {}
    while pending:
        value = pending.pop()
        if value in visited:
            continue
        visited.add(value)
        for use in value.uses:
            op = use.operation
            if op in chain or mq.op_name(op) == "func.return":
                continue
            index = indices.get(op)
            lane = placements.get(index, placements.get(str(index)))
            if index is None or lane not in {HOST, ACCELERATOR}:
                raise FrozenPlacementRefusal("UNKNOWN: intermediate consumer has no original frozen lane")
            if lane == HOST:
                consumers.add(index)
            pending.extend(op.results)
    return sorted(consumers)


def requires_merlin_host(declaration: Mapping[str, Any]) -> bool:
    """Whether a whole-program capsule's frozen placement gives the author accelerator work.

    Only the declaration is read: whether its bytes are bound is checked by :func:`read` on the
    route this selects, which fails closed.
    """
    if declaration.get("kind") not in {"model", "model_slice"}:
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
    source_operations: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Require frozen accelerator work while permitting source-bound optional offload and fusion.

    ``device`` comes from Merlin's capability-checked groups, never candidate placement claims.
    Their members must include the mandatory work and belong to the original source inventory.
    Fresh independent eligibility must still equal the frozen accelerator census exactly; the
    caller separately checks the group assignment of each eligible source operation. The ordinary
    device/host-compute and complete-output gates also remain mandatory.
    """
    if source_sha256 != frozen.get("program_sha256"):
        raise FrozenPlacementRefusal("routed source differs from the program the frozen placement is bound to")
    expected = dict(frozen.get("accelerator_operations") or {})
    if any(type(count) is not int or count <= 0 for count in device.values()):
        raise FrozenPlacementRefusal("device groups contain an invalid operation count")
    if Counter(expected) - Counter(device):
        raise FrozenPlacementRefusal(
            f"device groups hold {dict(device)} but the frozen Phase 0 placement puts {expected} on the "
            "accelerator; required accelerator work may not move to the host"
        )
    supporting = dict(sorted((Counter(device) - Counter(expected)).items()))
    if supporting and source_operations is None:
        raise FrozenPlacementRefusal("supporting device operations lack their original source inventory")
    if source_operations is not None and Counter(device) - Counter(source_operations):
        raise FrozenPlacementRefusal("device groups contain operations outside the original source inventory")
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
        "device_operations": dict(sorted(device.items())),
        "supporting_device_operations": supporting,
        "scope": "frozen accelerator operations remain mandatory; Merlin supplies host computation "
        "outside its capability-checked device groups, which may fuse source-bound supporting operations",
    }
