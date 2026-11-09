"""Issue independently replayed local arithmetic facts, never datapath roles."""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.rtl.hw_arithmetic import local_arithmetic
from merlin.targetgen.rtl.hw_graph import parse_generic_hw

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

SCHEMA = "merlin.independent_local_arithmetic_intake.v1"
_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
_READERS = (
    __name__,
    "merlin.targetgen.rtl.hw_arithmetic",
    "merlin.targetgen.rtl.hw_graph",
    "merlin.targetgen.rtl.hw_observations",
    "merlin.targetgen.rtl.ports",
    "xdsl.parser",
)
_UNKNOWN = (
    "historical_elaboration_and_bitstream_correspondence",
    "native_tool_and_reader_runtime_dependency_closure",
    "full_arithmetic_and_instruction_domain",
    "complete_contraction_and_physical_resource_roles",
)


def verify_record(record):
    """Reopen the actual generic serialization and rederive every expression.

    This diagnostic replay never recreates live issuer authority or upgrades
    the original structural hardware intake's independent qualification scope.
    """
    if (
        not isinstance(record, dict)
        or set(record) != {"schema", "hardware_intake_sha256", "source_pins", "invocation", "facts", "unknowns"}
        or record["schema"] != SCHEMA
        or record["unknowns"] != list(_UNKNOWN)
    ):
        raise RtlIntakeRefusal("local arithmetic intake needs its complete closed original record")
    pins = [RtlIntakePin(**row) for row in record["source_pins"]]
    if len({(pin.role, pin.path) for pin in pins}) != len(pins):
        raise RtlIntakeRefusal("local arithmetic intake has duplicate source membership")
    for pin in pins:
        pin.verify()

    def one(role):
        rows = [pin for pin in pins if pin.role == role]
        if len(rows) != 1:
            raise RtlIntakeRefusal("local arithmetic replay lost an exact source member")
        return rows[0]

    readers = {pin.path for pin in pins if pin.role == "arithmetic-reader"}
    if readers != {str(module_source_path(name)) for name in _READERS}:
        raise RtlIntakeRefusal("local arithmetic replay needs the exact fixed reader closure")
    core, tool, generic = one("original-core-hw"), one("circt-opt"), one("generic-core-hw")
    invocation = I.require_environment(Path(record["invocation"]), environment=_ENVIRONMENT)
    if (
        invocation["argv"] != [tool.path, "--mlir-print-op-generic", core.path, "-o", generic.path]
        or invocation["stage"] != "independent_local_arithmetic_genericization"
        or invocation["inputs"] != [{"path": core.path, "sha256": core.sha256}]
        or {row["path"] for row in invocation["outputs"]} != {generic.path}
    ):
        raise RtlIntakeRefusal("local arithmetic serialization differs from its exact original source invocation")
    observed = {pin.path: pin.sha256 for pin in pins}
    if observed.get(record["invocation"]) != hashlib.sha256(_plain(record["invocation"]).read_bytes()).hexdigest():
        raise RtlIntakeRefusal("local arithmetic intake omitted its actual native invocation receipt")
    for member in [invocation["executable"], invocation["stdout"], invocation["stderr"], *invocation["outputs"]]:
        if observed.get(member["path"]) != member["sha256"]:
            raise RtlIntakeRefusal("local arithmetic intake omitted actual native evidence")
    if local_arithmetic(parse_generic_hw(Path(generic.path).read_text())) != record["facts"]:
        raise RtlIntakeRefusal("local arithmetic facts differ from their original typed SSA")
    return record


@dataclass(frozen=True, eq=False)
class IndependentArithmeticIntake:
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
            raise RtlIntakeRefusal("local arithmetic requires live independently issued source authority")
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
            raise RtlIntakeRefusal("local arithmetic lost its exact live hardware production correspondence")

    def record(self):
        self.verify()
        return json.loads(self.receipt_json)

    def public_facts(self):
        record = self.record()
        return {
            "schema": SCHEMA,
            "hardware_intake_sha256": self.hardware.sha256,
            "facts": record["facts"],
            "source_sha256": [
                {"role": pin.role, "sha256": pin.sha256}
                for pin in self.source_pins
                if pin.role != "arithmetic-intake-receipt"
            ],
            "unknowns": record["unknowns"],
        }

    def verify_public_fact_view(self, view):
        from merlin_experiments.phase2.component_experiment import ComponentView, verify_component_view

        if type(view) is not ComponentView:
            raise RtlIntakeRefusal("local arithmetic fact view requires the ordinary typed public component view")
        facts = _json(self.public_facts())
        manifest = verify_component_view(view)
        rows = [row for row in manifest["members"] if row["path"] == "contract/arithmetic_facts.json"]
        if (
            len(rows) != 1
            or rows[0]["role"] != "contract"
            or rows[0]["sha256"] != hashlib.sha256(facts).hexdigest()
            or _plain(view.root / "contract/arithmetic_facts.json").read_bytes() != facts
        ):
            raise RtlIntakeRefusal("public view lost the complete independently observed local arithmetic projection")


def issue_independent_arithmetic_intake(*, hardware, circt_opt, forbidden_roots, output):
    """Execute only a fixed generic serialization of the freshly issued core.

    No supplied expressions, role names, software links, widths, module selectors
    or prior arithmetic tables enter this source-derived observer.
    """
    if type(hardware) is not IndependentHardwareIntake:
        raise RtlIntakeRefusal("local arithmetic needs original live independent hardware authority")
    hardware.verify()
    if not isinstance(forbidden_roots, tuple) or not forbidden_roots:
        raise RtlIntakeRefusal("local arithmetic needs explicit protected campaign exclusions")
    forbidden = tuple(_exclusion_prefix(path) for path in forbidden_roots)
    tool = _plain(circt_opt)
    _outside(tool, forbidden)
    cores = [
        pin for pin in hardware.source_pins if pin.role == "produced-artifact" and Path(pin.path).name == "core.hw.mlir"
    ]
    if len(cores) != 1:
        raise RtlIntakeRefusal("local arithmetic needs one actual reproduced core HW source")
    pins = [_pin("original-core-hw", cores[0].path, forbidden), _pin("circt-opt", tool, forbidden)]
    pins += [_pin("arithmetic-reader", module_source_path(name), forbidden) for name in _READERS]
    destination = Path(output).absolute()
    if destination.exists() or ".." in destination.parts or any(p.is_symlink() for p in destination.parents):
        raise RtlIntakeRefusal("local arithmetic needs a fresh ordinary output root")
    _outside(destination, forbidden)
    if any(Path(pin.path).is_relative_to(destination) for pin in (*pins, *hardware.source_pins)):
        raise RtlIntakeRefusal("local arithmetic output may not contain its protected inputs")
    destination.mkdir(parents=True, mode=0o700)
    generic = destination / "core.generic.mlir"
    I.run(
        [str(tool), "--mlir-print-op-generic", cores[0].path, "-o", str(generic)],
        directory=destination,
        stage="independent_local_arithmetic_genericization",
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
        _pin("arithmetic-native-evidence", p, forbidden)
        for p in sorted(destination.rglob("*"))
        if p.is_file() and p != generic
    ]
    record = {
        "schema": SCHEMA,
        "hardware_intake_sha256": hardware.sha256,
        "source_pins": [pin.record() for pin in pins],
        "invocation": str(invocation),
        "facts": local_arithmetic(parse_generic_hw(generic.read_text())),
        "unknowns": list(_UNKNOWN),
    }
    receipt = destination / "intake.json"
    receipt.write_bytes(_json(record))
    authority = IndependentArithmeticIntake(
        hardware, (*pins, _pin("arithmetic-intake-receipt", receipt, forbidden)), receipt.read_bytes()
    )
    _ISSUED[authority] = authority._identity()
    authority.verify()
    return authority
