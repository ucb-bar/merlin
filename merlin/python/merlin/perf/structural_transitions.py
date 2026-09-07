"""Source-bound layout-copy transitions without a fictional latency model.

The first supported mechanism is an exact, bit-preserving scalar copy between
static strided layouts in the same address space. Padding is allocated but not
read as logical data. The emitter must allocate a fresh destination and realize
the declared address functions; this module does not claim that emission occurred.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

from merlin.xdsl_dialects.lowering.dispatch_program import DispatchProgram
from merlin.xdsl_dialects.lowering.global_plan import (
    CycleInterval, TransitionAlternative, ValueRepresentation,
)


def _element_bytes(dtype: str) -> int:
    fixed = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8}
    if dtype in fixed:
        return fixed[dtype]
    if dtype.startswith("i") and dtype[1:].isdigit():
        bits = int(dtype[1:])
        if bits > 0 and bits % 8 == 0:
            return bits // 8
    raise ValueError(f"byte-addressable scalar representation unknown for {dtype!r}")


@dataclass(frozen=True)
class StaticStridedLayout:
    shape: tuple[int, ...]
    strides_elements: tuple[int, ...]
    storage_bytes: int
    offset_elements: int = 0

    def validate(self, dtype: str) -> None:
        if (len(self.shape) != len(self.strides_elements)
                or any(type(dim) is not int or dim <= 0 for dim in self.shape)
                or any(type(stride) is not int or stride <= 0 for stride in self.strides_elements)
                or type(self.offset_elements) is not int or self.offset_elements < 0
                or type(self.storage_bytes) is not int or self.storage_bytes <= 0):
            raise ValueError("layout requires positive static extents/strides and a bounded allocation")
        # A conservative injectivity proof, including axis permutations and row
        # padding. Unknown overlapping/interleaved affine layouts fail closed.
        span = 1
        for stride, dim in sorted(zip(self.strides_elements, self.shape)):
            if dim > 1 and stride < span:
                raise ValueError("layout address mapping is not proved injective")
            span += (dim-1) * stride
        end = self.offset_elements + 1 + sum((dim-1)*stride
            for dim, stride in zip(self.shape, self.strides_elements))
        if end * _element_bytes(dtype) > self.storage_bytes:
            raise ValueError("layout touches storage outside its declared allocation")

    def representation(self, *, dtype: str, placement: str) -> ValueRepresentation:
        self.validate(dtype)
        return ValueRepresentation(placement, "static_strided", dtype,
            attributes=tuple(sorted((
                ("shape", ",".join(map(str, self.shape))),
                ("strides_elements", ",".join(map(str, self.strides_elements))),
                ("offset_elements", str(self.offset_elements)),
                ("storage_bytes", str(self.storage_bytes)),
            ))))

    def to_dict(self):
        return {"shape": list(self.shape), "strides_elements": list(self.strides_elements),
                "storage_bytes": self.storage_bytes, "offset_elements": self.offset_elements}


@dataclass(frozen=True)
class StructuralQuantity:
    """One named observation; no implicit composition across unlike quantities."""
    name: str
    amount: int
    unit: str
    basis: str
    provenance: str
    artifact_sha256: str | None = None

    def __post_init__(self):
        if not self.name.strip() or not self.unit.strip() or not self.provenance.strip():
            raise ValueError("a structural quantity requires name, unit and provenance")
        if type(self.amount) is not int or self.amount < 0:
            raise ValueError("structural quantities require nonnegative exact integers")
        if self.unit.lower() in {"cycle", "cycles", "s", "second", "seconds", "ms", "us", "ns"}:
            raise ValueError("timing quantities do not belong to structural cost selection")
        if self.basis not in {"derived", "measured"}:
            raise ValueError("quantity basis must distinguish static derivation from measurement")
        if self.basis == "measured" and (self.artifact_sha256 is None
                or len(self.artifact_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.artifact_sha256)):
            raise ValueError("measured quantity requires its exact evidence artifact hash")

    def to_dict(self):
        return dict(name=self.name, amount=self.amount, unit=self.unit, basis=self.basis,
                    provenance=self.provenance, artifact_sha256=self.artifact_sha256)


@dataclass(frozen=True, kw_only=True)
class StructuralTransitionAlternative(TransitionAlternative):
    """Explicit non-cycle variant; the legacy transition constructor stays strict."""
    source_layout: StaticStridedLayout
    destination_layout: StaticStridedLayout
    quantities: tuple[StructuralQuantity, ...]
    legality_provenance: tuple[str, ...]

    def __post_init__(self):
        if not self.id.strip() or not self.buffer.strip() or self.kind != "static_strided_copy":
            raise ValueError("structural transition requires an identified static strided copy")
        if self.cycles.resolved or self.occupancy or self.demands:
            raise ValueError("structural transitions retain UNKNOWN cycles and explicit named quantities only")
        if not self.materializes:
            raise ValueError("strided copy must allocate and emit actual materialization")
        if self.source.placement != self.destination.placement:
            raise ValueError("cross-address-space transfer requires a separate capability proof")
        if (self.source.dtype != self.destination.dtype
                or self.source.quantization != "none" or self.destination.quantization != "none"
                or self.source.encoding != "plain" or self.destination.encoding != "plain"):
            raise ValueError("layout copy cannot silently cast, quantize or decode values")
        if self.source_layout.shape != self.destination_layout.shape:
            raise ValueError("layout copy must preserve the complete logical domain")
        for representation, layout in ((self.source, self.source_layout),
                                       (self.destination, self.destination_layout)):
            if representation != layout.representation(dtype=representation.dtype,
                                                        placement=representation.placement):
                raise ValueError("representation does not match its proved address function")
        if self.source == self.destination:
            raise ValueError("an unchanged representation does not require a transition")
        if not self.legality_provenance or any(not item.strip() for item in self.legality_provenance):
            raise ValueError("layout legality requires explicit provenance")
        if tuple(sorted(self.metadata)) != self.metadata:
            raise ValueError("transition metadata must be sorted")
        names = [item.name for item in self.quantities]
        if len(names) != len(set(names)):
            raise ValueError("transition repeats a structural quantity")
        expected = {"scalar_load_payload": prod(self.source_layout.shape)*_element_bytes(self.source.dtype),
                    "scalar_store_payload": prod(self.source_layout.shape)*_element_bytes(self.source.dtype),
                    "destination_storage": self.destination_layout.storage_bytes}
        observations = {item.name: item for item in self.quantities}
        for name, amount in expected.items():
            item = observations.get(name)
            if item is None or item.amount != amount or item.unit != "bytes" or item.basis != "derived":
                raise ValueError(f"strided copy requires exact derived {name} bytes")

    def to_dict(self):
        return {**super().to_dict(), "cost_mode": "structural",
                "source_layout": self.source_layout.to_dict(),
                "destination_layout": self.destination_layout.to_dict(),
                "quantities": [item.to_dict() for item in self.quantities],
                "legality_provenance": list(self.legality_provenance),
                "required_emitter_obligation": "fresh disjoint destination and exact logical-index copy",
                "dram_bytes": None, "cache_traffic_bytes": None}


def verify_structural_transition(program: DispatchProgram, transition: StructuralTransitionAlternative,
        *, buffer: str, producer, consumer, source, destination) -> None:
    """Bind the address proof to the exact logical dataflow edge requested by the planner."""
    expected_edge = (buffer, producer.id if producer else None, consumer.id if consumer else None)
    if (transition.buffer, transition.producer, transition.consumer) != expected_edge:
        raise ValueError("structural transition is not bound to the requested dependency")
    if transition.source != source or transition.destination != destination:
        raise ValueError("structural transition does not connect the requested representations")
    spec = program.buffers.get(buffer)
    if spec is None or tuple(spec.shape) != transition.source_layout.shape or spec.dtype != source.dtype:
        raise ValueError("structural transition shape/dtype differs from its immutable logical value")
    transition.__post_init__()


def strided_copy_transition(*, id: str, buffer: str, producer: str | None, consumer: str | None,
        source_layout: StaticStridedLayout, destination_layout: StaticStridedLayout,
        dtype: str, placement: str, provenance: tuple[str, ...]) -> StructuralTransitionAlternative:
    payload = prod(source_layout.shape)*_element_bytes(dtype)
    quantities = tuple(StructuralQuantity(name, amount, "bytes", "derived", evidence)
        for name, amount, evidence in (
            ("scalar_load_payload", payload, "one scalar load per logical coordinate"),
            ("scalar_store_payload", payload, "one scalar store per logical coordinate"),
            ("destination_storage", destination_layout.storage_bytes, "explicit destination allocation extent")))
    return StructuralTransitionAlternative(id, "static_strided_copy", buffer, producer, consumer,
        source_layout.representation(dtype=dtype, placement=placement),
        destination_layout.representation(dtype=dtype, placement=placement),
        CycleInterval.unknown("static address proof does not establish transition latency"),
        source_layout=source_layout, destination_layout=destination_layout,
        quantities=quantities, legality_provenance=provenance)


class StructuralTransitionAdapterView:
    """Opt-in structural hook; cycle planner still calls the original transition method."""
    def __init__(self, adapter):
        self.adapter = adapter

    def boundary_representation(self, *args, **kwargs):
        return self.adapter.boundary_representation(*args, **kwargs)

    def transition(self, program, **edge):
        hook = getattr(self.adapter, "structural_transition", None)
        if hook is None:
            return self.adapter.transition(program, **edge)
        transition = hook(program, **edge)
        if transition is None:
            return None
        if not isinstance(transition, StructuralTransitionAlternative):
            raise ValueError("structural transition hook must return an explicit structural proof")
        verify_structural_transition(program, transition, **edge)
        return transition
