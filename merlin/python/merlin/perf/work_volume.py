"""Exact contraction work recovered from the compiler's command-buffer IR.

This is the compute-axis analogue of :mod:`merlin.perf.dma_volume`: work is read from the program the
compiler emitted, never from a benchmark spreadsheet or a target name.  The unit is MACs, so no
convention about whether one MAC is one or two operations is hidden in the number.  Unsupported or
malformed compute commands make the whole result UNKNOWN; known work is retained only as a lower
bound.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

__all__ = ["CommandWork", "ProgramWork", "command_buffer_evidence", "work_from_command_buffer",
           "NO_COMMAND_BUFFER_REFUSAL"]


@dataclass(frozen=True)
class CommandWork:
    index: int
    opcode: str
    macs: int | None
    provenance: str
    refusal: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "opcode": self.opcode, "macs": self.macs,
                "provenance": self.provenance, "refusal": self.refusal}


@dataclass(frozen=True)
class ProgramWork:
    commands: tuple[CommandWork, ...]
    known_macs: int
    is_lower_bound: bool
    refusals: tuple[str, ...]
    artifact_sha256: str = ""
    basis: str = "compiler_command_buffer"
    unit: str = "macs"

    @property
    def exact_macs(self) -> int | None:
        return None if self.is_lower_bound else self.known_macs

    def to_dict(self) -> dict[str, Any]:
        return {"basis": self.basis, "unit": self.unit, "known_macs": self.known_macs,
                "exact_macs": self.exact_macs, "is_lower_bound": self.is_lower_bound,
                "artifact_sha256": self.artifact_sha256,
                "refusals": list(self.refusals),
                "commands": [command.to_dict() for command in self.commands]}


def _shape(tensors: Mapping[str, Any], name: Any) -> tuple[int, ...] | None:
    tensor = tensors.get(name) if isinstance(name, str) else None
    raw = tensor.get("shape") if isinstance(tensor, Mapping) else None
    if (not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in raw)):
        return None
    return tuple(int(value) for value in raw)


def _product(values: Sequence[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def _matmul_shapes(lhs: tuple[int, ...] | None, rhs: tuple[int, ...] | None) -> int | None:
    if lhs is None or rhs is None or len(lhs) != 2 or len(rhs) != 2 or lhs[1] != rhs[0]:
        return None
    return lhs[0] * lhs[1] * rhs[1]


def _conv_work(ifm: tuple[int, ...] | None, weight: tuple[int, ...] | None,
               attrs: Mapping[str, Any]) -> int | None:
    kernel = attrs.get("kernel")
    stride = attrs.get("stride")
    padding = attrs.get("padding")
    dilation = attrs.get("dilation")
    if (ifm is None or len(ifm) != 4
            or not all(isinstance(value, Sequence) and not isinstance(value, (str, bytes))
                       for value in (kernel, stride, padding, dilation))):
        return None
    try:
        kh, kw, ci, co = (int(value) for value in kernel)
        sh, sw = (int(value) for value in stride)
        pt, pl, pb, pr = (int(value) for value in padding)
        dh, dw = (int(value) for value in dilation)
    except (TypeError, ValueError):
        return None
    batch, height, width, channels = ifm
    values = (kh, kw, ci, co, sh, sw, dh, dw)
    if (any(value <= 0 for value in values) or min(pt, pl, pb, pr) < 0 or channels != ci
            or weight != (kh * kw * ci, co)):
        return None
    effective_h = dh * (kh - 1) + 1
    effective_w = dw * (kw - 1) + 1
    numer_h = height + pt + pb - effective_h
    numer_w = width + pl + pr - effective_w
    if numer_h < 0 or numer_w < 0:
        return None
    out_h, out_w = numer_h // sh + 1, numer_w // sw + 1
    return batch * out_h * out_w * kh * kw * ci * co


def work_from_command_buffer(command_buffer: Mapping[str, Any]) -> ProgramWork:
    """Recover exact MAC work from shared IR semantics, preserving every refusal.

    The recognized opcodes are the target-independent command-buffer ABI.  Resident handles are
    followed back to the declared source tensor, so a package cannot alter the counted work by naming
    its handle differently.  An unknown opcode is refused because silently treating it as non-compute
    would make dropping work look like an optimization.
    """
    try:
        artifact_sha256 = hashlib.sha256(json.dumps(
            command_buffer, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        artifact_sha256 = ""
    tensors = command_buffer.get("tensors")
    instructions = command_buffer.get("commands")
    if not isinstance(tensors, Mapping) or not isinstance(instructions, Sequence) \
            or isinstance(instructions, (str, bytes)):
        refusal = "command buffer must carry tensor declarations and a command sequence"
        return ProgramWork((), 0, True, (refusal,), artifact_sha256=artifact_sha256)

    handles: dict[str, str] = {}
    rows: list[CommandWork] = []
    refusals: list[str] = []
    non_compute = {"RES_PACK", "COMMIT", "EVICT", "MOVEMENT"}
    for index, raw in enumerate(instructions):
        if not isinstance(raw, Mapping):
            reason = f"command {index} is not a mapping"
            rows.append(CommandWork(index, "", None, f"commands[{index}]", reason))
            refusals.append(reason)
            continue
        opcode = str(raw.get("opcode") or "")
        operands = raw.get("operands") if isinstance(raw.get("operands"), Mapping) else {}
        attrs = raw.get("attributes") if isinstance(raw.get("attributes"), Mapping) else {}
        provenance = f"command_buffer.commands[{index}]({opcode})"
        if opcode == "RES_PACK":
            src, dst = operands.get("src"), operands.get("dst")
            if isinstance(src, str) and isinstance(dst, str) and _shape(tensors, src) is not None:
                handles[dst] = src
                continue
            reason = "resident-pack source/destination does not resolve to a declared tensor"
        elif opcode == "EVICT":
            handle = operands.get("handle")
            if isinstance(handle, str) and handle in handles:
                del handles[handle]
                continue
            reason = "eviction handle is absent or not currently resident"
        elif opcode in ("MATMUL", "MATMUL_RESIDENT"):
            rhs_name = operands.get("rhs")
            if opcode == "MATMUL_RESIDENT":
                rhs_name = handles.get(rhs_name) if isinstance(rhs_name, str) else None
            macs = _matmul_shapes(_shape(tensors, operands.get("lhs")), _shape(tensors, rhs_name))
            if macs is not None:
                rows.append(CommandWork(index, opcode, macs, provenance))
                continue
            reason = "matmul operands do not resolve to compatible rank-2 tensor shapes"
        elif opcode == "BATCHED_MATMUL":
            # A batch of INDEPENDENT contractions: the work is one 2-D contraction's MACs times the
            # batch, and the batch extents must agree or the two operands describe different batches.
            a, w = _shape(tensors, operands.get("a")), _shape(tensors, operands.get("w"))
            macs = None
            if a is not None and w is not None and len(a) == len(w) == 3 and a[0] == w[0]:
                per_slice = _matmul_shapes(a[1:], w[1:])
                macs = None if per_slice is None else a[0] * per_slice
            if macs is not None:
                rows.append(CommandWork(index, opcode, macs, provenance))
                continue
            reason = "batched-matmul operands do not resolve to two rank-3 shapes over one batch"
        elif opcode == "ATTENTION_QK":
            q, k = _shape(tensors, operands.get("q")), _shape(tensors, operands.get("k"))
            macs = q[0] * q[1] * k[0] if q and k and len(q) == len(k) == 2 and q[1] == k[1] else None
            if macs is not None:
                rows.append(CommandWork(index, opcode, macs, provenance))
                continue
            reason = "attention-QK operands do not resolve to [queries,depth] and [keys,depth]"
        elif opcode == "ATTENTION_PV":
            p, v = _shape(tensors, operands.get("p")), _shape(tensors, operands.get("v"))
            macs = _matmul_shapes(p, v)
            if macs is not None:
                rows.append(CommandWork(index, opcode, macs, provenance))
                continue
            reason = "attention-PV operands do not resolve to compatible rank-2 tensor shapes"
        elif opcode == "CONV2D":
            weight_name = operands.get("weight")
            if isinstance(weight_name, str) and weight_name in handles:
                weight_name = handles[weight_name]
            macs = _conv_work(_shape(tensors, operands.get("ifm")),
                              _shape(tensors, weight_name), attrs)
            if macs is not None:
                rows.append(CommandWork(index, opcode, macs, provenance))
                continue
            reason = "convolution geometry is absent, invalid, or inconsistent with the input"
        elif opcode in non_compute:
            continue
        else:
            reason = f"opcode {opcode!r} has no work-counting rule; whether it computes is UNKNOWN"
        rows.append(CommandWork(index, opcode, None, provenance, reason))
        refusals.append(f"[{index}] {reason}")

    known = sum(row.macs or 0 for row in rows)
    return ProgramWork(tuple(rows), known, bool(refusals), tuple(refusals),
                       artifact_sha256=artifact_sha256)


#: Stated when a graded run produced no command buffer at all. The keys are still emitted carrying
#: this: an ABSENT key reads to a consumer as "this axis does not apply", while an explicit UNKNOWN
#: reads as "the work could not be priced" -- and only the second one is true. A zero would be worse
#: still, because on a performance bench "this program does no work" means "infinitely fast".
NO_COMMAND_BUFFER_REFUSAL = ("the graded run produced no compiler command buffer, so its work "
                             "volume is UNKNOWN")


def _unknown_evidence(refusal: str, compiler_provenance: str) -> tuple[dict[str, Any], dict[str, Any]]:
    work = ProgramWork((), 0, True, (refusal,)).to_dict()
    artifact = {"command_buffer": None, "artifact_sha256": None,
                "compiler_provenance": str(compiler_provenance), "refusal": refusal}
    return work, artifact


def command_buffer_evidence(command_buffer: Any, *, compiler_provenance: str
                            ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the ``(work_volume, command_buffer_artifact)`` pair a perf consumer reads.

    ONE notion of work. The totals are :func:`work_from_command_buffer`'s and nothing else's, and the
    artifact carries the RAW buffer they were counted from under the canonical digest that both
    halves agree on -- so a reader can recount the program itself and refuse the pair if the two
    disagree, instead of trusting a total it cannot check. Two independently computed notions of
    "work" is exactly how the number and the receipt come to disagree silently.

    Never returns a zero total for unpriceable work: an unknown stays ``exact_macs=None`` with the
    counter's own refusals attached, and a buffer that is absent or not serialisable evidence yields
    the same explicit UNKNOWN rather than a missing key.
    """
    if not isinstance(command_buffer, Mapping):
        return _unknown_evidence(NO_COMMAND_BUFFER_REFUSAL, compiler_provenance)
    try:
        canonical = json.dumps(command_buffer, sort_keys=True, separators=(",", ":"))
        # Round-tripped, so the bytes a reader re-digests are the bytes this digest was taken over.
        replayed = json.loads(canonical)
    except (TypeError, ValueError) as exc:
        return _unknown_evidence(
            f"command buffer is not serialisable evidence and cannot be priced: {exc}",
            compiler_provenance)
    work = work_from_command_buffer(command_buffer).to_dict()
    artifact = {"command_buffer": replayed,
                "artifact_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "compiler_provenance": str(compiler_provenance)}
    return work, artifact
