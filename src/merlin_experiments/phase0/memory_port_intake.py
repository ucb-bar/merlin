"""Issue local typed memory-port facts from the same live public HW source."""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits, memory_port_observations

from .rtl_intake import (
    IndependentHardwareIntake,
    RtlIntakePin,
    RtlIntakeRefusal,
    _exclusion_prefix,
    _json,
    _outside,
    _pin,
    _plain,
)

SCHEMA = "merlin.independent_memory_port_intake.v1"
_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
_READERS = (
    __name__,
    "merlin.targetgen.rtl.hw_memory_ports",
    "merlin.targetgen.rtl.hw_graph",
    "merlin.targetgen.rtl.hw_combinational",
    "merlin.targetgen.rtl.hw_instance_inputs",
    "merlin.targetgen.rtl.hw_observations",
    "merlin.targetgen.rtl.ports",
    "merlin.targetgen.contract.mlir_source_admission",
    "xdsl.parser",
)
_UNKNOWN = (
    "historical_elaboration_and_bitstream_correspondence",
    "native_tool_and_reader_runtime_dependency_closure",
    "decoder_state_transfer_and_physical_execution_correspondence",
)


def _derive(path, *, source_bytes, limits):
    if type(source_bytes) is not int or source_bytes <= 0 or type(limits) is not MemoryPortLimits:
        raise RtlIntakeRefusal("memory intake requires explicit positive source and metadata budgets")
    if path.stat().st_size > source_bytes:
        raise RtlIntakeRefusal("memory source exceeds its explicit preparse byte budget")
    text = path.read_text()
    admit_mlir_source(
        text,
        max_source_bytes=source_bytes,
        max_nesting=64,
        max_integer_bits=max(64, limits.scalar_bits),
        allow_dense=False,
        allow_dense_resource=False,
    )
    return memory_port_observations(parse_generic_hw(text, reject_dense_literals=True), limits=limits)


def verify_record(record):
    """Recompute exact local declarations/bindings; exported JSON is no authority."""
    if (
        not isinstance(record, dict)
        or set(record)
        != {
            "schema",
            "hardware_intake_sha256",
            "source_pins",
            "invocation",
            "source_bytes",
            "limits",
            "facts",
            "unknowns",
        }
        or record["schema"] != SCHEMA
        or record["unknowns"] != list(_UNKNOWN)
        or not isinstance(record["limits"], dict)
        or set(record["limits"]) != set(MemoryPortLimits.__dataclass_fields__)
    ):
        raise RtlIntakeRefusal("memory intake requires its complete closed original record")
    limits = MemoryPortLimits(**record["limits"])
    pins = [RtlIntakePin(**row) for row in record["source_pins"]]
    if len({(pin.role, pin.path) for pin in pins}) != len(pins):
        raise RtlIntakeRefusal("memory intake has duplicate source membership")
    for pin in pins:
        pin.verify()

    def one(role):
        selected = [pin for pin in pins if pin.role == role]
        if len(selected) != 1:
            raise RtlIntakeRefusal("memory intake lost one exact original source member")
        return selected[0]

    if {pin.path for pin in pins if pin.role == "memory-reader"} != {
        str(module_source_path(name)) for name in _READERS
    }:
        raise RtlIntakeRefusal("memory intake lost the fixed original reader closure")
    core, tool, generic = one("original-core-hw"), one("circt-opt"), one("generic-core-hw")
    invocation = I.require_environment(Path(record["invocation"]), environment=_ENVIRONMENT)
    if (
        invocation["argv"] != [tool.path, "--mlir-print-op-generic", core.path, "-o", generic.path]
        or invocation["stage"] != "independent_memory_port_genericization"
        or invocation["inputs"] != [{"path": core.path, "sha256": core.sha256}]
        or {row["path"] for row in invocation["outputs"]} != {generic.path}
    ):
        raise RtlIntakeRefusal("memory native serialization differs from its selected original source")
    observed = {pin.path: pin.sha256 for pin in pins}
    evidence = [
        {"path": record["invocation"], "sha256": hashlib.sha256(_plain(record["invocation"]).read_bytes()).hexdigest()},
        invocation["executable"],
        invocation["stdout"],
        invocation["stderr"],
        *invocation["outputs"],
    ]
    if any(observed.get(row["path"]) != row["sha256"] for row in evidence):
        raise RtlIntakeRefusal("memory intake lost complete actual native evidence")
    if _derive(Path(generic.path), source_bytes=record["source_bytes"], limits=limits) != record["facts"]:
        raise RtlIntakeRefusal("memory facts differ from the complete original typed memory/port graph")
    return record


@dataclass(frozen=True, eq=False)
class IndependentMemoryPortIntake:
    hardware: IndependentHardwareIntake
    source_pins: tuple[RtlIntakePin, ...]
    receipt_json: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt_json).hexdigest()

    def _identity(self):
        return hashlib.sha256(
            _json(
                {
                    "hardware": self.hardware.sha256,
                    "sources": [pin.record() for pin in self.source_pins],
                    "receipt": self.sha256,
                }
            )
        ).hexdigest()

    def verify(self):
        if type(self.hardware) is not IndependentHardwareIntake or _ISSUED.get(self) != self._identity():
            raise RtlIntakeRefusal("memory intake requires live same-HW independent source authority")
        self.hardware.verify()
        for pin in self.source_pins:
            pin.verify()
        record = verify_record(json.loads(self.receipt_json))
        original = {
            pin.path: pin.sha256
            for pin in self.hardware.source_pins
            if pin.role == "produced-artifact" and Path(pin.path).name == "core.hw.mlir"
        }
        selected = {row["path"]: row["sha256"] for row in record["source_pins"] if row["role"] == "original-core-hw"}
        if selected != original or record["hardware_intake_sha256"] != self.hardware.sha256:
            raise RtlIntakeRefusal("memory intake lost its live original hardware production join")

    def record(self):
        self.verify()
        return json.loads(self.receipt_json)

    def public_facts(self):
        record = self.record()
        return {
            "schema": SCHEMA,
            "hardware_intake_sha256": self.hardware.sha256,
            "facts": record["facts"],
            "unknowns": record["unknowns"],
            "source_sha256": [
                {"role": pin.role, "sha256": pin.sha256}
                for pin in self.source_pins
                if pin.role != "memory-intake-receipt"
            ],
        }


def issue_independent_memory_port_intake(*, hardware, circt_opt, source_bytes, limits, forbidden_roots, output):
    """Replay exact core serialization; no supplied memory/port/role selectors."""
    if type(hardware) is not IndependentHardwareIntake:
        raise RtlIntakeRefusal("memory intake requires live independently issued hardware authority")
    hardware.verify()
    if not isinstance(forbidden_roots, tuple) or not forbidden_roots:
        raise RtlIntakeRefusal("memory intake requires explicit protected campaign exclusions")
    if type(limits) is not MemoryPortLimits or type(source_bytes) is not int or source_bytes <= 0:
        raise RtlIntakeRefusal("memory intake requires explicit complete-roster parse and metadata budgets")
    forbidden = tuple(_exclusion_prefix(path) for path in forbidden_roots)
    tool = _plain(circt_opt)
    _outside(tool, forbidden)
    cores = [
        pin for pin in hardware.source_pins if pin.role == "produced-artifact" and Path(pin.path).name == "core.hw.mlir"
    ]
    if len(cores) != 1:
        raise RtlIntakeRefusal("memory intake requires one exact actual reproduced core HW source")
    if Path(cores[0].path).stat().st_size > source_bytes:
        raise RtlIntakeRefusal("original memory source exceeds its explicit preparse byte budget")
    pins = [_pin("original-core-hw", cores[0].path, forbidden), _pin("circt-opt", tool, forbidden)]
    pins += [_pin("memory-reader", module_source_path(name), forbidden) for name in _READERS]
    destination = Path(output).absolute()
    if destination.exists() or ".." in destination.parts or any(path.is_symlink() for path in destination.parents):
        raise RtlIntakeRefusal("memory intake requires a fresh ordinary output owner")
    _outside(destination, forbidden)
    if any(Path(pin.path).is_relative_to(destination) for pin in (*pins, *hardware.source_pins)):
        raise RtlIntakeRefusal("memory intake output cannot contain its protected inputs")
    destination.mkdir(parents=True, mode=0o700)
    generic = destination / "core.generic.mlir"
    I.run(
        [str(tool), "--mlir-print-op-generic", cores[0].path, "-o", str(generic)],
        directory=destination,
        stage="independent_memory_port_genericization",
        inputs=(Path(cores[0].path),),
        outputs=(generic,),
        capture_output=True,
        check=True,
        env=_ENVIRONMENT,
        timeout=120,
    )
    invocation = next(destination.glob("invocations/*/invocation.json"))
    pins.append(_pin("generic-core-hw", generic, forbidden))
    pins += [
        _pin("memory-native-evidence", path, forbidden)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path != generic
    ]
    record = {
        "schema": SCHEMA,
        "hardware_intake_sha256": hardware.sha256,
        "source_pins": [pin.record() for pin in pins],
        "invocation": str(invocation),
        "source_bytes": source_bytes,
        "limits": asdict(limits),
        "facts": _derive(generic, source_bytes=source_bytes, limits=limits),
        "unknowns": list(_UNKNOWN),
    }
    receipt = destination / "intake.json"
    receipt.write_bytes(_json(record))
    authority = IndependentMemoryPortIntake(
        hardware, (*pins, _pin("memory-intake-receipt", receipt, forbidden)), receipt.read_bytes()
    )
    _ISSUED[authority] = authority._identity()
    authority.verify()
    return authority
