"""Issue conditional typed-index domain facts from the same live native source.

Known-bit domain containment never grants definedness or original phase admission.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common.jsonio import strict_json_equal
from merlin.common.paths import module_source_path
from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source
from merlin.targetgen.rtl.hw_address_transitions import AddressTransitionLimits
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_hierarchy_bindings import HierarchyBindingLimits
from merlin.targetgen.rtl.hw_index_ranges import IndexRangeLimits, index_range_observations
from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits

from .address_transition_intake import _READERS as _TRANSITION_READERS
from .address_transition_intake import IndependentAddressTransitionIntake
from .address_transition_intake import verify_record as verify_transition_record
from .rtl_intake import RtlIntakePin, RtlIntakeRefusal, _exclusion_prefix, _json, _outside, _pin, _plain

SCHEMA = "merlin.independent_conditional_index_range_intake.v1"
_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_READERS = tuple(
    dict.fromkeys((__name__, "merlin.targetgen.rtl.hw_index_ranges", "xdsl.dialects.builtin", *_TRANSITION_READERS))
)
_UNKNOWN = (
    "historical_elaboration_and_bitstream_correspondence",
    "native_tool_and_reader_runtime_dependency_closure",
    "original_definedness_state_event_command_resource_and_effect_admission",
)


def _one(pins, role):
    selected = [pin for pin in pins if pin.role == role]
    if len(selected) != 1:
        raise RtlIntakeRefusal("index range intake lost an exact original source member")
    return selected[0]


def _derive(transitions, memory, *, source_bytes, limits):
    if type(source_bytes) is not int or source_bytes <= 0 or type(limits) is not IndexRangeLimits:
        raise RtlIntakeRefusal("index range intake requires explicit positive source and proof budgets")
    generic = _one([RtlIntakePin(**row) for row in memory["source_pins"]], "generic-core-hw")
    path = _plain(generic.path)
    if path.stat().st_size > source_bytes:
        raise RtlIntakeRefusal("index range source exceeds its preparse byte budget")
    text = path.read_text()
    admit_mlir_source(
        text,
        max_source_bytes=source_bytes,
        max_nesting=64,
        max_integer_bits=max(64, limits.scalar_bits, transitions["limits"]["scalar_bits"]),
        allow_dense=False,
        allow_dense_resource=False,
    )
    return index_range_observations(
        parse_generic_hw(text, reject_dense_literals=True),
        root=transitions["facts"]["root"],
        local_limits=MemoryPortLimits(**memory["limits"]),
        hierarchy_limits=HierarchyBindingLimits(**transitions["facts"]["hierarchy_limits"]),
        transition_limits=AddressTransitionLimits(**transitions["limits"]),
        limits=limits,
    )


def verify_record(record):
    """Recompute the exact closed source contract; saved records confer no authority."""
    if (
        not isinstance(record, dict)
        or set(record)
        != {
            "schema",
            "transition_intake_sha256",
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
        or set(record["limits"]) != set(IndexRangeLimits.__dataclass_fields__)
    ):
        raise RtlIntakeRefusal("index range intake requires its complete closed original record")
    limits = IndexRangeLimits(**record["limits"])
    pins = [RtlIntakePin(**row) for row in record["source_pins"]]
    if len({(pin.role, pin.path) for pin in pins}) != len(pins) or {pin.role for pin in pins} != {
        "typed-transition-intake",
        "hardware-intake",
        "index-range-reader",
    }:
        raise RtlIntakeRefusal("index range intake has missing, duplicate or unsupported source membership")
    for pin in pins:
        pin.verify()
    if {pin.path for pin in pins if pin.role == "index-range-reader"} != {
        str(module_source_path(name)) for name in _READERS
    }:
        raise RtlIntakeRefusal("index range intake lost its fixed original reader closure")
    transition_pin, hardware_pin = _one(pins, "typed-transition-intake"), _one(pins, "hardware-intake")
    transitions = verify_transition_record(json.loads(Path(transition_pin.path).read_bytes()))
    hierarchy_pin = _one([RtlIntakePin(**row) for row in transitions["source_pins"]], "hierarchical-memory-intake")
    hierarchy = json.loads(Path(hierarchy_pin.path).read_bytes())
    memory_pin = _one([RtlIntakePin(**row) for row in hierarchy["source_pins"]], "local-memory-intake")
    memory = json.loads(Path(memory_pin.path).read_bytes())
    if (
        transition_pin.sha256 != record["transition_intake_sha256"]
        or hardware_pin.sha256 != record["hardware_intake_sha256"]
        or transitions["hardware_intake_sha256"] != hardware_pin.sha256
    ):
        raise RtlIntakeRefusal("index range intake lost its exact same native source identities")
    if not strict_json_equal(
        _derive(transitions, memory, source_bytes=record["source_bytes"], limits=limits), record["facts"]
    ):
        raise RtlIntakeRefusal("index range facts differ from the complete original native source")
    return record


@dataclass(frozen=True, eq=False)
class IndependentIndexRangeIntake:
    transitions: IndependentAddressTransitionIntake
    source_pins: tuple[RtlIntakePin, ...]
    receipt_json: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt_json).hexdigest()

    def _identity(self):
        return hashlib.sha256(
            _json(
                {
                    "transitions": self.transitions.sha256,
                    "sources": [pin.record() for pin in self.source_pins],
                    "receipt": self.sha256,
                }
            )
        ).hexdigest()

    def verify(self):
        if type(self.transitions) is not IndependentAddressTransitionIntake or _ISSUED.get(self) != self._identity():
            raise RtlIntakeRefusal("index range intake requires live same-HW source authority")
        self.transitions.verify()
        for pin in self.source_pins:
            pin.verify()
        record = verify_record(json.loads(self.receipt_json))
        if (
            record["transition_intake_sha256"] != self.transitions.sha256
            or record["hardware_intake_sha256"] != self.transitions.hierarchy.memory.hardware.sha256
        ):
            raise RtlIntakeRefusal("index range intake lost its live original source join")
        return record

    def require_issued(self):
        self.verify()
        return self

    def record(self):
        return self.verify()

    def public_facts(self):
        record = self.record()
        return {
            "schema": SCHEMA,
            "transition_intake_sha256": record["transition_intake_sha256"],
            "hardware_intake_sha256": record["hardware_intake_sha256"],
            "facts": record["facts"],
            "unknowns": record["unknowns"],
        }


def issue_independent_index_range_intake(*, transitions, source_bytes, limits, forbidden_roots, output):
    """Retain every original address; no path, role, type or depth selector."""
    if type(transitions) is not IndependentAddressTransitionIntake:
        raise RtlIntakeRefusal("index range intake requires live independent original address transitions")
    transitions.verify()
    if not isinstance(forbidden_roots, tuple) or not forbidden_roots:
        raise RtlIntakeRefusal("index range intake requires explicit protected campaign exclusions")
    if type(source_bytes) is not int or source_bytes <= 0 or type(limits) is not IndexRangeLimits:
        raise RtlIntakeRefusal("index range intake requires explicit complete-roster source/proof budgets")
    forbidden = tuple(_exclusion_prefix(path) for path in forbidden_roots)
    pins = [
        _pin("typed-transition-intake", _one(transitions.source_pins, "transition-intake-receipt").path, forbidden),
        _pin(
            "hardware-intake", _one(transitions.hierarchy.memory.hardware.source_pins, "intake-receipt").path, forbidden
        ),
        *[_pin("index-range-reader", module_source_path(name), forbidden) for name in _READERS],
    ]
    destination = Path(output).absolute()
    if destination.exists() or ".." in destination.parts or any(path.is_symlink() for path in destination.parents):
        raise RtlIntakeRefusal("index range output must be a new owned plain directory")
    _outside(destination, forbidden)
    if any(
        Path(pin.path).is_relative_to(destination)
        for pin in (
            *pins,
            *transitions.source_pins,
            *transitions.hierarchy.source_pins,
            *transitions.hierarchy.memory.source_pins,
            *transitions.hierarchy.memory.hardware.source_pins,
        )
    ):
        raise RtlIntakeRefusal("index range output cannot contain its protected sources")
    record = {
        "schema": SCHEMA,
        "transition_intake_sha256": transitions.sha256,
        "hardware_intake_sha256": transitions.hierarchy.memory.hardware.sha256,
        "source_pins": [pin.record() for pin in pins],
        "source_bytes": source_bytes,
        "limits": asdict(limits),
        "facts": _derive(
            json.loads(transitions.receipt_json),
            json.loads(transitions.hierarchy.memory.receipt_json),
            source_bytes=source_bytes,
            limits=limits,
        ),
        "unknowns": list(_UNKNOWN),
    }
    destination.mkdir(parents=True, mode=0o700)
    receipt = destination / "intake.json"
    receipt.write_bytes(_json(record))
    issued = IndependentIndexRangeIntake(
        transitions, (*pins, _pin("index-range-receipt", receipt, forbidden)), _json(record)
    )
    _ISSUED[issued] = issued._identity()
    issued.verify()
    return issued
