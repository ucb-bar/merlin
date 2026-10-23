"""Which commands of a command buffer must be separated by a ``FENCE``.

A target whose reservation station tracks no DRAM address (``hazard_resolution: explicit`` with the
obligation ``must_fence_dram_read_after_write``) does not order a command that READS a tensor after the
command that WROTE it: the reader's loads can issue before the writer's stores land. Between two
commands of one buffer that hand a tensor through memory, a ``FENCE`` is therefore required.

This checks exactly that, on the command buffer, from operand names alone. On-device handles (a
resident weight, an accumulator) are device state, not memory a later load reads back, so only
declared tensors are tracked. The intra-command hazards (an overlapping load after a load, a pooling
store from the scratchpad) are properties of the backend's instruction stream and are not visible here.
"""

from __future__ import annotations

from typing import Any

#: The contract obligation this check enforces.
OBLIGATION = "must_fence_dram_read_after_write"


def _reads(command: dict[str, Any]) -> list[str]:
    operands = command.get("operands") or {}
    return [value for key, value in operands.items() if key != "dst" and isinstance(value, str)]


def fence_violations(cb: dict[str, Any]) -> list[dict[str, Any]]:
    """Every read of a tensor an earlier command wrote with no ``FENCE`` in between.

    Returns ``[{reader, reader_index, writer, writer_index, tensor}]`` in program order; empty when the
    buffer is ordered. A tensor that only the host supplies (a leaf input) needs no fence.
    """
    declared = set((cb.get("tensors") or {}).keys())
    written_since_fence: dict[str, tuple[int, str]] = {}
    violations: list[dict[str, Any]] = []
    for index, command in enumerate(cb.get("commands") or ()):
        opcode = str(command.get("opcode"))
        if opcode == "FENCE":
            written_since_fence.clear()
            continue
        for name in _reads(command):
            if name in written_since_fence:
                writer_index, writer = written_since_fence[name]
                violations.append(
                    {
                        "reader": opcode,
                        "reader_index": index,
                        "writer": writer,
                        "writer_index": writer_index,
                        "tensor": name,
                    }
                )
        destination = (command.get("operands") or {}).get("dst")
        if isinstance(destination, str) and destination in declared:
            written_since_fence[destination] = (index, opcode)
    return violations


def requires_fences(contract: dict[str, Any] | None) -> bool:
    """Whether a capability contract obliges a fence between memory-chained commands."""
    if not isinstance(contract, dict):
        return False
    memory = contract.get("memory_model") or {}
    return memory.get("hazard_resolution") == "explicit" and OBLIGATION in (contract.get("compiler_obligations") or ())
