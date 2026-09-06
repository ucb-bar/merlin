"""Exact operand traffic recovered from the compiler's command-buffer IR.

The movement-axis analogue of :mod:`merlin.perf.work_volume`, and the denominator an operational
intensity needs.  :mod:`merlin.perf.dma_volume` answers the same question one layer lower, from ISA
instructions; a roofline built beside ``work_volume`` needs the two axes counted over the SAME
artifact, or the ratio is of two different programs.

The unit is bytes that cross the accelerator's operand boundary, so a lever that moves less data
shows up here and a lever that merely reorders the same traffic does not.

RESIDENCY IS THE POINT, so it is counted once.  A resident handle's bytes are charged where the
program actually pays them -- at the pack -- and NOT again at each command that reads the handle.
Charging them per use would report a resident weight re-fetched on every tile, which is precisely the
traffic residency exists to remove: the lever would then look inert, or worse, harmful.

Unsupported or malformed commands make the whole result UNKNOWN; known traffic is retained only as a
lower bound.  A byte count that silently omits an opcode reads as a program that moved less data,
which on this axis means "faster" -- the same failure `work_volume` refuses on the compute axis.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

__all__ = ["CommandMovement", "ProgramMovement", "movement_from_command_buffer",
           "movement_evidence", "NO_COMMAND_BUFFER_REFUSAL"]

#: Commands that move no operand bytes themselves. ``EVICT`` unbinds a handle; the bytes it drops were
#: charged at the pack and are not moved again by dropping them.
_NO_TRAFFIC = frozenset({"EVICT"})

#: Commands whose declared operands are read across the boundary, by the operand ROLE the ABI gives
#: them. A resident handle among them is skipped -- see the residency note above.
_READS = {
    "MATMUL": ("lhs", "weight", "rhs"),
    "MATMUL_RESIDENT": ("lhs", "weight", "rhs"),
    "BATCHED_MATMUL": ("lhs", "weight", "rhs"),
    "CONV2D": ("input", "weight", "lhs", "rhs"),
    "ATTENTION_QK": ("q", "k", "lhs", "rhs"),
    "ATTENTION_PV": ("p", "v", "lhs", "rhs"),
}


@dataclass(frozen=True)
class CommandMovement:
    index: int
    opcode: str
    bytes_in: int | None
    bytes_out: int | None
    provenance: str
    refusal: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "opcode": self.opcode, "bytes_in": self.bytes_in,
                "bytes_out": self.bytes_out, "provenance": self.provenance, "refusal": self.refusal}


@dataclass(frozen=True)
class ProgramMovement:
    commands: tuple[CommandMovement, ...]
    known_bytes_in: int
    known_bytes_out: int
    is_lower_bound: bool
    refusals: tuple[str, ...]
    artifact_sha256: str = ""
    basis: str = "compiler_command_buffer"
    unit: str = "bytes"

    @property
    def known_bytes(self) -> int:
        return self.known_bytes_in + self.known_bytes_out

    @property
    def exact_bytes(self) -> int | None:
        """Total moved bytes, or ``None`` when any command was refused.

        ``None`` and ``0`` are different answers: the first is "this program's traffic is not
        established", the second is "it moved nothing". Collapsing them makes an unreadable program
        the most memory-efficient one in the corpus.
        """
        return None if self.is_lower_bound else self.known_bytes

    def to_dict(self) -> dict[str, Any]:
        return {"basis": self.basis, "unit": self.unit,
                "known_bytes_in": self.known_bytes_in, "known_bytes_out": self.known_bytes_out,
                "known_bytes": self.known_bytes, "exact_bytes": self.exact_bytes,
                "is_lower_bound": self.is_lower_bound, "artifact_sha256": self.artifact_sha256,
                "refusals": list(self.refusals),
                # THE LIMITATION TRAVELS WITH THE NUMBER. A resident operand is charged once at its
                # pack, so this counts the traffic the command buffer DECLARES, not the traffic the
                # emitted program issues. A backend whose lowering re-loads a resident tile per
                # output tile produces exactly the same number here -- measured 2026-09-06 on
                # gemmini, where the command buffer's RES_PACK / MATMUL_RESIDENT / EVICT sequence is
                # correct while the emitted stream reportedly reloads. Residency is therefore NOT
                # verifiable from this block, and a reader who treats it as issued traffic will
                # conclude residency was achieved whenever it was merely intended.
                "counts": "declared_by_command_buffer",
                "resident_operand_charged": "once_at_pack",
                "cannot_detect": ("a lowering that re-loads a resident operand; compare mvin count "
                                  "against RES_PACK count in the decoded instruction stream"),
                "commands": [command.to_dict() for command in self.commands]}


def _nbytes(tensors: Mapping[str, Any], name: Any) -> int | None:
    """Storage bytes of a declared tensor, or ``None`` when it does not resolve.

    Sized through :func:`merlin.targetgen.capsule_dram.tensor_nbytes`, which ceils over the packed bit
    total -- so a sub-byte format occupies what it actually occupies rather than being rounded up per
    element, and an unknown dtype raises instead of defaulting to four bytes.
    """
    tensor = tensors.get(name) if isinstance(name, str) else None
    if not isinstance(tensor, Mapping):
        return None
    shape, dtype = tensor.get("shape"), tensor.get("dtype")
    if (not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or not shape
            or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in shape)
            or not isinstance(dtype, str) or not dtype):
        return None
    from merlin.targetgen.capsule_dram import tensor_nbytes
    try:
        return int(tensor_nbytes(list(shape), dtype))
    except (KeyError, ValueError):
        return None


def movement_from_command_buffer(command_buffer: Mapping[str, Any]) -> ProgramMovement:
    """Recover exact operand traffic from shared IR semantics, preserving every refusal."""
    try:
        artifact_sha256 = hashlib.sha256(json.dumps(
            command_buffer, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        artifact_sha256 = ""
    tensors = command_buffer.get("tensors") if isinstance(command_buffer, Mapping) else None
    instructions = command_buffer.get("commands") if isinstance(command_buffer, Mapping) else None
    if not isinstance(tensors, Mapping) or not isinstance(instructions, Sequence) \
            or isinstance(instructions, (str, bytes)):
        refusal = "command buffer must carry tensor declarations and a command sequence"
        return ProgramMovement((), 0, 0, True, (refusal,), artifact_sha256=artifact_sha256)

    handles: dict[str, str] = {}
    rows: list[CommandMovement] = []
    refusals: list[str] = []
    total_in = total_out = 0
    for index, raw in enumerate(instructions):
        if not isinstance(raw, Mapping):
            reason = f"command {index} is not a mapping"
            rows.append(CommandMovement(index, "", None, None, f"commands[{index}]", reason))
            refusals.append(reason)
            continue
        opcode = str(raw.get("opcode") or "")
        operands = raw.get("operands") if isinstance(raw.get("operands"), Mapping) else {}
        provenance = f"command_buffer.commands[{index}]({opcode})"
        reason: str | None = None

        if opcode in _NO_TRAFFIC:
            rows.append(CommandMovement(index, opcode, 0, 0, provenance))
            continue
        if opcode == "RES_PACK":
            src, dst = operands.get("src"), operands.get("dst")
            nbytes = _nbytes(tensors, src)
            if isinstance(src, str) and isinstance(dst, str) and nbytes is not None:
                handles[dst] = src
                total_in += nbytes
                rows.append(CommandMovement(index, opcode, nbytes, 0, provenance))
                continue
            reason = "resident-pack source/destination does not resolve to a declared tensor"
        elif opcode in ("COMMIT", "MOVEMENT"):
            # The destination is what crossed the boundary: a commit writes the output tensor back,
            # and a movement's source may be an accumulator handle with no declared bytes of its own.
            dst = operands.get("dst")
            nbytes = _nbytes(tensors, dst)
            if nbytes is not None:
                total_out += nbytes
                rows.append(CommandMovement(index, opcode, 0, nbytes, provenance))
                continue
            reason = f"{opcode.lower()} destination does not resolve to a declared tensor"
        elif opcode in _READS:
            moved = 0
            unresolved: list[str] = []
            for role in _READS[opcode]:
                name = operands.get(role)
                if not isinstance(name, str):
                    continue
                if name in handles:
                    continue          # resident: charged once, at its pack
                nbytes = _nbytes(tensors, name)
                if nbytes is None:
                    unresolved.append(role)
                    continue
                moved += nbytes
            if not unresolved:
                total_in += moved
                rows.append(CommandMovement(index, opcode, moved, 0, provenance))
                continue
            reason = (f"{opcode} operand(s) {sorted(unresolved)} do not resolve to a declared tensor "
                      f"or a resident handle")
        else:
            reason = (f"opcode {opcode!r} has no traffic-counting rule; whether it moves operand "
                      f"bytes is UNKNOWN")
        rows.append(CommandMovement(index, opcode, None, None, provenance, reason))
        refusals.append(reason)

    return ProgramMovement(tuple(rows), total_in, total_out, bool(refusals), tuple(refusals),
                           artifact_sha256=artifact_sha256)


NO_COMMAND_BUFFER_REFUSAL = ("the graded run produced no compiler command buffer, so its operand "
                             "traffic is not established")


def movement_evidence(command_buffer: Any, *, compiler_provenance: str) -> dict[str, Any]:
    """The ``movement_volume`` block a perf consumer reads, beside ``work_volume``'s.

    Shares the artifact digest with :func:`merlin.perf.work_volume.command_buffer_evidence`, so a
    reader can confirm the two axes were counted over the SAME emitted program before dividing one by
    the other. An intensity taken across two programs is not an intensity.
    """
    if not isinstance(command_buffer, Mapping):
        return {"basis": "compiler_command_buffer", "unit": "bytes", "known_bytes_in": 0,
                "known_bytes_out": 0, "known_bytes": 0, "exact_bytes": None,
                "is_lower_bound": True, "artifact_sha256": "",
                "refusals": [NO_COMMAND_BUFFER_REFUSAL], "commands": [],
                "compiler_provenance": compiler_provenance}
    block = movement_from_command_buffer(command_buffer).to_dict()
    block["compiler_provenance"] = compiler_provenance
    return block
