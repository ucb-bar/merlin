"""Issue replayed local packing facts, never semantic or physical allocation."""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_packing import equal_partitions

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

SCHEMA = "merlin.independent_local_packing_intake.v1"
MEMORY_SCHEMA = "merlin.independent_local_packing_intake.v2"
MEMORY_SELECTION_SCHEMA = "merlin.local_packing_selection.v2"
_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
_READERS = (
    __name__,
    "merlin.targetgen.rtl.hw_packing",
    "merlin.targetgen.rtl.hw_graph",
    "merlin.targetgen.rtl.hw_observations",
    "merlin.targetgen.rtl.ports",
    "xdsl.parser",
)
_UNKNOWN = (
    "historical_elaboration_and_bitstream_correspondence",
    "native_tool_and_reader_runtime_dependency_closure",
    "full_packing_and_instruction_domain",
    "semantic_axis_allocation_capacity_and_physical_tail_roles",
)
_MEMORY_UNKNOWN = (*_UNKNOWN, "conditional_memory_binding_definedness_events_and_history")
_MEMORY_READERS = (
    *_READERS,
    "merlin.targetgen.rtl.hw_partition_memory_bindings",
    "merlin.targetgen.rtl.hw_hierarchy_bindings",
    "merlin.targetgen.rtl.hw_memory_ports",
    "merlin.targetgen.rtl.hw_instance_inputs",
    "merlin.targetgen.rtl.hw_combinational",
    "merlin.targetgen.contract.mlir_source_admission",
    "merlin.common.jsonio",
)


def validate_memory_selection(selection):
    """Close metadata budgets only; callers cannot supply roots or mappings."""
    from merlin.targetgen.rtl.hw_hierarchy_bindings import HierarchyBindingLimits
    from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits
    from merlin.targetgen.rtl.hw_partition_memory_bindings import PartitionMemoryLimits

    fields = {
        "memory_limits": MemoryPortLimits,
        "hierarchy_limits": HierarchyBindingLimits,
        "binding_limits": PartitionMemoryLimits,
    }
    if (
        type(selection) is not dict
        or set(selection) != {"schema", "source_bytes", *fields}
        or selection["schema"] != MEMORY_SELECTION_SCHEMA
        or type(selection["source_bytes"]) is not int
        or selection["source_bytes"] <= 0
    ):
        raise RtlIntakeRefusal("conditional memory packing needs its explicit v2 selection and source budget")
    for name, constructor in fields.items():
        if type(selection[name]) is not dict or set(selection[name]) != set(constructor.__dataclass_fields__):
            raise RtlIntakeRefusal("conditional memory packing lost its complete metadata budgets")
        constructor(**selection[name])
    # A caller's later mutable declaration cannot change this observation.
    return {
        "schema": MEMORY_SELECTION_SCHEMA,
        "source_bytes": selection["source_bytes"],
        **{name: asdict(constructor(**selection[name])) for name, constructor in fields.items()},
    }


def _memory_bindings(path, hardware, selection):
    from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source
    from merlin.targetgen.rtl.hw_hierarchy_bindings import HierarchyBindingLimits
    from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits
    from merlin.targetgen.rtl.hw_partition_memory_bindings import PartitionMemoryLimits, partition_memory_bindings

    selected = validate_memory_selection(selection)
    if path.stat().st_size > selected["source_bytes"]:
        raise RtlIntakeRefusal("conditional memory packing source exceeds its preparse byte budget")
    hierarchy = HierarchyBindingLimits(**selected["hierarchy_limits"])
    root = hardware["production"]["production"]["core_root"]
    if type(root) is not str or not root:
        raise RtlIntakeRefusal("conditional memory packing lost its original production root")
    text = path.read_text()
    admit_mlir_source(
        text,
        max_source_bytes=selected["source_bytes"],
        max_nesting=64,
        max_integer_bits=max(64, hierarchy.scalar_bits),
        allow_dense=False,
        allow_dense_resource=False,
    )
    return partition_memory_bindings(
        parse_generic_hw(text, reject_dense_literals=True),
        root=root,
        local_limits=MemoryPortLimits(**selected["memory_limits"]),
        hierarchy_limits=hierarchy,
        limits=PartitionMemoryLimits(**selected["binding_limits"]),
    )


def verify_record(record):
    """Reopen the actual generic serialization and rederive every complete partition.

    This diagnostic replay never recreates live issuer authority or upgrades
    the original structural hardware intake's independent qualification scope.
    """
    memory = isinstance(record, dict) and record.get("schema") == MEMORY_SCHEMA
    fields = {"schema", "hardware_intake_sha256", "source_pins", "invocation", "facts", "unknowns"}
    if memory:
        fields |= {"memory_binding_selection", "memory_bindings"}
    if (
        not isinstance(record, dict)
        or set(record) != fields
        or record["schema"] not in {SCHEMA, MEMORY_SCHEMA}
        or record["unknowns"] != list(_MEMORY_UNKNOWN if memory else _UNKNOWN)
    ):
        raise RtlIntakeRefusal("local packing intake needs its complete closed original record")
    pins = [RtlIntakePin(**row) for row in record["source_pins"]]
    if len({(pin.role, pin.path) for pin in pins}) != len(pins):
        raise RtlIntakeRefusal("local packing intake has duplicate source membership")
    for pin in pins:
        pin.verify()
    if memory and {pin.role for pin in pins} != {
        "original-core-hw",
        "circt-opt",
        "generic-core-hw",
        "packing-reader",
        "packing-native-evidence",
        "hardware-intake",
    }:
        raise RtlIntakeRefusal("conditional memory packing changed its original source membership")

    def one(role):
        rows = [pin for pin in pins if pin.role == role]
        if len(rows) != 1:
            raise RtlIntakeRefusal("local packing replay lost an exact source member")
        return rows[0]

    readers = {pin.path for pin in pins if pin.role == "packing-reader"}
    if readers != {str(module_source_path(name)) for name in (_MEMORY_READERS if memory else _READERS)}:
        raise RtlIntakeRefusal("local packing replay needs the exact fixed reader closure")
    core, tool, generic = one("original-core-hw"), one("circt-opt"), one("generic-core-hw")
    invocation = I.require_environment(Path(record["invocation"]), environment=_ENVIRONMENT)
    if (
        invocation["argv"] != [tool.path, "--mlir-print-op-generic", core.path, "-o", generic.path]
        or invocation["stage"] != "independent_local_packing_genericization"
        or invocation["inputs"] != [{"path": core.path, "sha256": core.sha256}]
        or {row["path"] for row in invocation["outputs"]} != {generic.path}
    ):
        raise RtlIntakeRefusal("local packing serialization differs from its exact original source invocation")
    observed = {pin.path: pin.sha256 for pin in pins}
    if observed.get(record["invocation"]) != hashlib.sha256(_plain(record["invocation"]).read_bytes()).hexdigest():
        raise RtlIntakeRefusal("local packing intake omitted its actual native invocation receipt")
    for member in [invocation["executable"], invocation["stdout"], invocation["stderr"], *invocation["outputs"]]:
        if observed.get(member["path"]) != member["sha256"]:
            raise RtlIntakeRefusal("local packing intake omitted actual native evidence")
    if not memory and equal_partitions(parse_generic_hw(Path(generic.path).read_text())) != record["facts"]:
        raise RtlIntakeRefusal("local packing facts differ from their original typed SSA")
    if memory:
        from merlin.common.jsonio import strict_json_equal

        hardware_pin = one("hardware-intake")
        hardware = json.loads(Path(hardware_pin.path).read_bytes())
        if hardware_pin.sha256 != record["hardware_intake_sha256"]:
            raise RtlIntakeRefusal("conditional memory packing changed its original hardware owner")
        original = {}
        for row in hardware["sources"]:
            pin = RtlIntakePin(**row)
            pin.verify()
            if pin.role == "produced-artifact" and Path(pin.path).name == "core.hw.mlir":
                original[pin.path] = pin.sha256
        if original != {core.path: core.sha256}:
            raise RtlIntakeRefusal("conditional memory packing changed its original source container")
        facts = _memory_bindings(Path(generic.path), hardware, record["memory_binding_selection"])
        if not strict_json_equal(facts, record["memory_bindings"]) or not strict_json_equal(
            facts["local_partitions"], record["facts"]
        ):
            raise RtlIntakeRefusal("conditional memory packing differs from complete original occurrence bindings")
    return record


@dataclass(frozen=True, eq=False)
class IndependentPackingIntake:
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
            raise RtlIntakeRefusal("local packing requires live independently issued source authority")
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
            raise RtlIntakeRefusal("local packing lost its exact live hardware production correspondence")

    def record(self):
        self.verify()
        return json.loads(self.receipt_json)

    def public_facts(self):
        record = self.record()
        return {
            "schema": record["schema"],
            "hardware_intake_sha256": self.hardware.sha256,
            "facts": record["facts"],
            "source_sha256": [
                {"role": pin.role, "sha256": pin.sha256}
                for pin in self.source_pins
                if pin.role != "packing-intake-receipt"
            ],
            "unknowns": record["unknowns"],
            **(
                {
                    "memory_binding_selection": record["memory_binding_selection"],
                    "memory_bindings": record["memory_bindings"],
                }
                if record["schema"] == MEMORY_SCHEMA
                else {}
            ),
        }

    def verify_public_fact_view(self, view):
        from merlin_experiments.phase2.component_experiment import ComponentView, verify_component_view

        if type(view) is not ComponentView:
            raise RtlIntakeRefusal("local packing fact view requires the ordinary typed public component view")
        facts = _json(self.public_facts())
        manifest = verify_component_view(view)
        rows = [row for row in manifest["members"] if row["path"] == "contract/packing_facts.json"]
        if (
            len(rows) != 1
            or rows[0]["role"] != "contract"
            or rows[0]["sha256"] != hashlib.sha256(facts).hexdigest()
            or _plain(view.root / "contract/packing_facts.json").read_bytes() != facts
        ):
            raise RtlIntakeRefusal("public view lost the complete independently observed local packing projection")


def issue_independent_packing_intake(*, hardware, circt_opt, forbidden_roots, output, memory_selection=None):
    """Execute only a fixed generic serialization of the freshly issued core.

    No supplied partitions, role names, software links, widths, module selectors
    or prior packing tables enter this source-derived observer.
    """
    if type(hardware) is not IndependentHardwareIntake:
        raise RtlIntakeRefusal("local packing needs original live independent hardware authority")
    hardware.verify()
    selected = validate_memory_selection(memory_selection) if memory_selection is not None else None
    if not isinstance(forbidden_roots, tuple) or not forbidden_roots:
        raise RtlIntakeRefusal("local packing needs explicit protected campaign exclusions")
    forbidden = tuple(_exclusion_prefix(path) for path in forbidden_roots)
    tool = _plain(circt_opt)
    _outside(tool, forbidden)
    cores = [
        pin for pin in hardware.source_pins if pin.role == "produced-artifact" and Path(pin.path).name == "core.hw.mlir"
    ]
    if len(cores) != 1:
        raise RtlIntakeRefusal("local packing needs one actual reproduced core HW source")
    pins = [_pin("original-core-hw", cores[0].path, forbidden), _pin("circt-opt", tool, forbidden)]
    if selected is not None and Path(cores[0].path).stat().st_size > selected["source_bytes"]:
        raise RtlIntakeRefusal("conditional memory packing source exceeds its preparse byte budget")
    pins += [
        _pin("packing-reader", module_source_path(name), forbidden)
        for name in (_MEMORY_READERS if selected is not None else _READERS)
    ]
    if selected is not None:
        receipts = [pin for pin in hardware.source_pins if pin.role == "intake-receipt"]
        if len(receipts) != 1 or receipts[0].sha256 != hardware.sha256:
            raise RtlIntakeRefusal("conditional memory packing requires its exact live hardware receipt")
        pins.append(_pin("hardware-intake", receipts[0].path, forbidden))
    destination = Path(output).absolute()
    if destination.exists() or ".." in destination.parts or any(p.is_symlink() for p in destination.parents):
        raise RtlIntakeRefusal("local packing needs a fresh ordinary output root")
    _outside(destination, forbidden)
    if any(Path(pin.path).is_relative_to(destination) for pin in (*pins, *hardware.source_pins)):
        raise RtlIntakeRefusal("local packing output may not contain its protected inputs")
    destination.mkdir(parents=True, mode=0o700)
    generic = destination / "core.generic.mlir"
    I.run(
        [str(tool), "--mlir-print-op-generic", cores[0].path, "-o", str(generic)],
        directory=destination,
        stage="independent_local_packing_genericization",
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
        _pin("packing-native-evidence", p, forbidden)
        for p in sorted(destination.rglob("*"))
        if p.is_file() and p != generic
    ]
    memory_facts = (
        _memory_bindings(generic, json.loads(hardware.receipt_json), selected) if selected is not None else None
    )
    record = {
        "schema": MEMORY_SCHEMA if selected is not None else SCHEMA,
        "hardware_intake_sha256": hardware.sha256,
        "source_pins": [pin.record() for pin in pins],
        "invocation": str(invocation),
        "facts": memory_facts["local_partitions"]
        if memory_facts is not None
        else equal_partitions(parse_generic_hw(generic.read_text())),
        "unknowns": list(_MEMORY_UNKNOWN if selected is not None else _UNKNOWN),
    }
    if selected is not None:
        record.update(memory_binding_selection=selected, memory_bindings=memory_facts)
    receipt = destination / "intake.json"
    receipt.write_bytes(_json(record))
    authority = IndependentPackingIntake(
        hardware, (*pins, _pin("packing-intake-receipt", receipt, forbidden)), receipt.read_bytes()
    )
    _ISSUED[authority] = authority._identity()
    authority.verify()
    return authority
