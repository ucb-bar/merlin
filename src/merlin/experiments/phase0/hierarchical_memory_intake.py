"""Issue data-only hierarchical bindings from the same live native memory source."""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common.paths import module_source_path
from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_hierarchy_bindings import HierarchyBindingLimits, hierarchical_memory_bindings
from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits

from .memory_port_intake import IndependentMemoryPortIntake
from .memory_port_intake import verify_record as verify_memory_record
from .rtl_intake import RtlIntakePin, RtlIntakeRefusal, _exclusion_prefix, _json, _outside, _pin, _plain

SCHEMA = "merlin.independent_hierarchical_memory_intake.v1"
_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_READERS = (
    __name__,
    "merlin.targetgen.rtl.hw_hierarchy_bindings",
    "merlin.targetgen.rtl.hw_graph",
    "merlin.targetgen.rtl.hw_memory_ports",
    "merlin.targetgen.rtl.hw_instance_inputs",
    "merlin.targetgen.rtl.hw_observations",
    "merlin.targetgen.rtl.hw_combinational",
    "merlin.targetgen.rtl.ports",
    "merlin.targetgen.contract.mlir_source_admission",
    "xdsl.parser",
)
_UNKNOWN = (
    "historical_elaboration_and_bitstream_correspondence",
    "native_tool_and_reader_runtime_dependency_closure",
    "command_resource_axis_and_physical_execution_correspondence",
)


def _one(pins, role):
    selected = [pin for pin in pins if pin.role == role]
    if len(selected) != 1:
        raise RtlIntakeRefusal("hierarchical intake lost an exact original source member")
    return selected[0]


def _derive(memory, hardware, *, source_bytes, limits):
    if type(source_bytes) is not int or source_bytes <= 0 or type(limits) is not HierarchyBindingLimits:
        raise RtlIntakeRefusal("hierarchical intake requires explicit positive source and metadata limits")
    generic = _one([RtlIntakePin(**row) for row in memory["source_pins"]], "generic-core-hw")
    path = _plain(generic.path)
    if path.stat().st_size > source_bytes:
        raise RtlIntakeRefusal("hierarchical memory source exceeds its explicit preparse byte budget")
    root = hardware["production"]["production"]["core_root"]
    text = path.read_text()
    admit_mlir_source(
        text,
        max_source_bytes=source_bytes,
        max_nesting=64,
        max_integer_bits=max(64, limits.scalar_bits),
        allow_dense=False,
        allow_dense_resource=False,
    )
    return hierarchical_memory_bindings(
        parse_generic_hw(text, reject_dense_literals=True),
        root=root,
        local_limits=MemoryPortLimits(**memory["limits"]),
        limits=limits,
    )


def verify_record(record):
    """Recompute original typed connectivity; exported records grant no authority."""
    if (
        not isinstance(record, dict)
        or set(record)
        != {
            "schema",
            "memory_intake_sha256",
            "hardware_intake_sha256",
            "source_pins",
            "source_bytes",
            "limits",
            "facts",
            "unknowns",
        }
        or record["schema"] != SCHEMA
        or record["unknowns"] != list(_UNKNOWN)
        or not isinstance(record["limits"], dict)
        or set(record["limits"]) != set(HierarchyBindingLimits.__dataclass_fields__)
    ):
        raise RtlIntakeRefusal("hierarchical intake requires its complete closed original record")
    limits = HierarchyBindingLimits(**record["limits"])
    pins = [RtlIntakePin(**row) for row in record["source_pins"]]
    if len({(pin.role, pin.path) for pin in pins}) != len(pins) or {pin.role for pin in pins} != {
        "local-memory-intake",
        "hardware-intake",
        "hierarchy-reader",
    }:
        raise RtlIntakeRefusal("hierarchical intake has missing, duplicate or unsupported source membership")
    for pin in pins:
        pin.verify()
    if {pin.path for pin in pins if pin.role == "hierarchy-reader"} != {
        str(module_source_path(name)) for name in _READERS
    }:
        raise RtlIntakeRefusal("hierarchical intake lost its fixed original reader closure")
    memory_pin, hardware_pin = _one(pins, "local-memory-intake"), _one(pins, "hardware-intake")
    memory = verify_memory_record(json.loads(Path(memory_pin.path).read_bytes()))
    hardware = json.loads(Path(hardware_pin.path).read_bytes())
    if (
        memory_pin.sha256 != record["memory_intake_sha256"]
        or hardware_pin.sha256 != record["hardware_intake_sha256"]
        or memory["hardware_intake_sha256"] != hardware_pin.sha256
    ):
        raise RtlIntakeRefusal("hierarchical intake lost its exact same live source identities")
    for row in hardware["sources"]:
        RtlIntakePin(**row).verify()
    if _derive(memory, hardware, source_bytes=record["source_bytes"], limits=limits) != record["facts"]:
        raise RtlIntakeRefusal("hierarchical facts differ from the complete original source bindings")
    return record


@dataclass(frozen=True, eq=False)
class IndependentHierarchicalMemoryIntake:
    memory: IndependentMemoryPortIntake
    source_pins: tuple[RtlIntakePin, ...]
    receipt_json: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt_json).hexdigest()

    def _identity(self):
        return hashlib.sha256(
            _json(
                {
                    "memory": self.memory.sha256,
                    "sources": [pin.record() for pin in self.source_pins],
                    "receipt": self.sha256,
                }
            )
        ).hexdigest()

    def verify(self):
        if type(self.memory) is not IndependentMemoryPortIntake or _ISSUED.get(self) != self._identity():
            raise RtlIntakeRefusal("hierarchical intake requires live same-HW memory source authority")
        self.memory.verify()
        for pin in self.source_pins:
            pin.verify()
        record = verify_record(json.loads(self.receipt_json))
        if (
            record["memory_intake_sha256"] != self.memory.sha256
            or record["hardware_intake_sha256"] != self.memory.hardware.sha256
        ):
            raise RtlIntakeRefusal("hierarchical intake lost its live original hardware/memory join")

    def record(self):
        self.verify()
        return json.loads(self.receipt_json)

    def public_facts(self):
        record = self.record()
        return {
            "schema": SCHEMA,
            "memory_intake_sha256": self.memory.sha256,
            "hardware_intake_sha256": self.memory.hardware.sha256,
            "facts": record["facts"],
            "unknowns": record["unknowns"],
        }


def issue_independent_hierarchical_memory_intake(*, memory, source_bytes, limits, forbidden_roots, output):
    """Follow the complete root from original production; no role/path selectors."""
    if type(memory) is not IndependentMemoryPortIntake:
        raise RtlIntakeRefusal("hierarchical intake requires independently issued live native memory authority")
    memory.verify()
    if not isinstance(forbidden_roots, tuple) or not forbidden_roots:
        raise RtlIntakeRefusal("hierarchical intake requires explicit protected campaign exclusions")
    if type(limits) is not HierarchyBindingLimits or type(source_bytes) is not int or source_bytes <= 0:
        raise RtlIntakeRefusal("hierarchical intake requires explicit complete-roster source and metadata budgets")
    forbidden = tuple(_exclusion_prefix(path) for path in forbidden_roots)
    pins = [
        _pin("local-memory-intake", _one(memory.source_pins, "memory-intake-receipt").path, forbidden),
        _pin("hardware-intake", _one(memory.hardware.source_pins, "intake-receipt").path, forbidden),
        *[_pin("hierarchy-reader", module_source_path(name), forbidden) for name in _READERS],
    ]
    destination = Path(output).absolute()
    if destination.exists() or ".." in destination.parts or any(path.is_symlink() for path in destination.parents):
        raise RtlIntakeRefusal("hierarchical intake requires a fresh ordinary output owner")
    _outside(destination, forbidden)
    if any(
        Path(pin.path).is_relative_to(destination) for pin in (*pins, *memory.source_pins, *memory.hardware.source_pins)
    ):
        raise RtlIntakeRefusal("hierarchical intake output cannot contain its protected inputs")
    record = {
        "schema": SCHEMA,
        "memory_intake_sha256": memory.sha256,
        "hardware_intake_sha256": memory.hardware.sha256,
        "source_pins": [pin.record() for pin in pins],
        "source_bytes": source_bytes,
        "limits": asdict(limits),
        "facts": _derive(
            json.loads(memory.receipt_json),
            json.loads(memory.hardware.receipt_json),
            source_bytes=source_bytes,
            limits=limits,
        ),
        "unknowns": list(_UNKNOWN),
    }
    destination.mkdir(parents=True, mode=0o700)
    receipt = destination / "intake.json"
    receipt.write_bytes(_json(record))
    authority = IndependentHierarchicalMemoryIntake(
        memory, (*pins, _pin("hierarchy-intake-receipt", receipt, forbidden)), receipt.read_bytes()
    )
    _ISSUED[authority] = authority._identity()
    authority.verify()
    return authority
