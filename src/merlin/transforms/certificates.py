"""Structural change identities, not proof receipts or candidate assertions."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.ir import Operation
from xdsl.parser import Parser
from xdsl.printer import Printer


def ir_text(op: Operation) -> str:
    """Generic printing retains attributes, properties and all SSA dependencies."""
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(op)
    return stream.getvalue()


def ir_digest(op: Operation | str) -> str:
    return hashlib.sha256((op if isinstance(op, str) else ir_text(op)).encode()).hexdigest()


def parse_ir(text: str) -> builtin.ModuleOp:
    """Parse the common domain, retaining opaque OOT operations conservatively."""
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, func.Func, arith.Arith, scf.Scf, memref.MemRef):
        context.load_dialect(dialect)
    module = Parser(context, text).parse_module()
    module.verify()
    return module


@dataclass(frozen=True)
class IntegerWitness:
    operation_index: int
    operation: str
    rule: str
    width: int
    lhs: int | None
    rhs: int | None
    result: int | None
    retained_operand: int | None = None


@dataclass(frozen=True)
class ShapeWitness:
    operation_index: int
    rule: str
    shape: tuple[int, ...]
    axis: int | None = None


@dataclass(frozen=True)
class LoopWitness:
    operation_index: int
    lower: int
    upper: int
    step: int
    tile_iterations: int
    index_bits: int
    carried_types: tuple[str, ...]
    body_sha256: str


@dataclass(frozen=True)
class BufferInterval:
    allocation_index: int
    initialized_index: int
    deallocation_index: int
    shape: tuple[int, ...]
    element_bits: int
    alignment: int
    size_bytes: int
    slot: int


@dataclass(frozen=True)
class BufferReuseWitness:
    block_index: int
    intervals: tuple[BufferInterval, ...]


@dataclass(frozen=True)
class DependencyWitness:
    block_index: int
    operation_sha256: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    order: tuple[int, ...]
    unknown_effect_indices: tuple[int, ...]


Witness = IntegerWitness | ShapeWitness | LoopWitness | BufferReuseWitness | DependencyWitness

_WITNESS_KINDS = {
    IntegerWitness: "integer",
    ShapeWitness: "shape",
    LoopWitness: "loop",
    BufferReuseWitness: "buffer_reuse",
    DependencyWitness: "dependency",
}

TRANSFORM_SCOPES = {
    "shape_view_normalization": ("identical_memref_cast", "static_dimensions"),
    "exact_integer_simplification": ("scalar_signless_wrapping_integers",),
    "buffer_reuse": ("static_contiguous_initialized_integer_buffers", "sequential_nonoverlapping_lifetimes"),
    "ordered_loop_segmentation": (
        "static_nonnegative_index_bounds",
        "positive_step",
        "exact_iteration_order",
        "forwarded_loop_carried_state",
    ),
    "sequential_dependency_schedule": ("ssa_dependencies", "sequential_effect_order"),
}


@dataclass(frozen=True)
class ChangeCertificate:
    kind: str
    before_ir_sha256: str
    after_ir_sha256: str
    witnesses: tuple[Witness, ...]
    scope: tuple[str, ...]
    schema: str = "merlin.shared_transform.v1"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["witnesses"] = [{"kind": _WITNESS_KINDS[type(w)], **asdict(w)} for w in self.witnesses]
        return result


@dataclass(frozen=True)
class TransformationChange:
    kind: str
    before_ir: str
    after_ir: str
    parameters: Mapping[str, Any]
    certificate: ChangeCertificate

    def validate_identity(self) -> None:
        if self.certificate.kind != self.kind:
            raise ValueError("certificate kind does not match the actual change")
        if ir_digest(self.before_ir) != self.certificate.before_ir_sha256:
            raise ValueError("before IR does not match certificate identity")
        if ir_digest(self.after_ir) != self.certificate.after_ir_sha256:
            raise ValueError("after IR does not match certificate identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "before_ir": self.before_ir,
            "after_ir": self.after_ir,
            "parameters": dict(self.parameters),
            "certificate": self.certificate.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TransformationChange:
        """Decode a transport record, retaining no candidate proof authority."""
        if set(value) != {"kind", "before_ir", "after_ir", "parameters", "certificate"}:
            raise ValueError("invalid transformation change fields")
        certificate = value["certificate"]
        if not isinstance(certificate, Mapping) or set(certificate) != {f.name for f in fields(ChangeCertificate)}:
            raise ValueError("invalid transformation certificate fields")
        if certificate["schema"] != "merlin.shared_transform.v1":
            raise ValueError("unknown transformation certificate schema")
        kinds = {v: k for k, v in _WITNESS_KINDS.items()}
        witnesses = []
        for encoded in certificate["witnesses"]:
            if not isinstance(encoded, Mapping) or encoded.get("kind") not in kinds:
                raise ValueError("unknown transformation witness kind")
            dtype = kinds[encoded["kind"]]
            data = dict(encoded)
            del data["kind"]
            if set(data) != {f.name for f in fields(dtype)}:
                raise ValueError("invalid transformation witness fields")
            if dtype is ShapeWitness:
                data["shape"] = tuple(data["shape"])
            elif dtype is LoopWitness:
                data["carried_types"] = tuple(data["carried_types"])
            elif dtype is BufferReuseWitness:
                intervals = []
                for encoded_interval in data["intervals"]:
                    if set(encoded_interval) != {f.name for f in fields(BufferInterval)}:
                        raise ValueError("invalid buffer interval fields")
                    interval = dict(encoded_interval)
                    interval["shape"] = tuple(interval["shape"])
                    intervals.append(BufferInterval(**interval))
                data["intervals"] = tuple(intervals)
            elif dtype is DependencyWitness:
                data["operation_sha256"] = tuple(data["operation_sha256"])
                data["edges"] = tuple(tuple(edge) for edge in data["edges"])
                data["order"] = tuple(data["order"])
                data["unknown_effect_indices"] = tuple(data["unknown_effect_indices"])
            witnesses.append(dtype(**data))
        decoded = dict(certificate)
        decoded["witnesses"] = tuple(witnesses)
        decoded["scope"] = tuple(decoded["scope"])
        result = cls(
            value["kind"],
            value["before_ir"],
            value["after_ir"],
            dict(value["parameters"]),
            ChangeCertificate(**decoded),
        )
        result.validate_identity()
        return result


def make_change(
    kind: str,
    before: str,
    after: Operation,
    parameters: Mapping[str, Any],
    witnesses: tuple[Witness, ...],
    scope: tuple[str, ...],
) -> TransformationChange:
    after.verify()
    rendered = ir_text(after)
    return TransformationChange(
        kind,
        before,
        rendered,
        dict(parameters),
        ChangeCertificate(kind, ir_digest(before), ir_digest(rendered), witnesses, scope),
    )
