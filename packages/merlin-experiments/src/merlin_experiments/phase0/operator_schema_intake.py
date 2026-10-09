"""Live independent public-source/operator-schema observation authority.

The protected source selection and original minimal-SW example membership are
reopened before actual native observation. Captured/public/registered equality
does not prove installed-library build provenance or physical alias behavior.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.frontend_operator_effects import original_operator_effects
from merlin.targetgen.frontend_use_def import original_use_def_semantics

from .command_intake import _tracked_source
from .rtl_intake import RtlIntakePin, RtlIntakeRefusal, _exclusion_prefix, _json, _outside, _pin, _plain
from .software_intake import IndependentSoftwareIntake

SCHEMA = "merlin.independent_operator_schema_intake.v1"
SELECTION_SCHEMA = "merlin.independent_operator_schema_selection.v1"
TENSOR_SCHEMA = "merlin.independent_operator_schema_intake.v2"
TENSOR_SELECTION_SCHEMA = "merlin.independent_operator_schema_selection.v2"
_ISSUED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_UNKNOWN = (
    "installed_framework_build_source_correspondence",
    "native_runtime_dependency_closure",
    "non_schema_effects_and_whole_effect_domain",
    "conditional_alias_outcomes",
    "physical_alias_ownership_lifetime_completion",
    "hardware_resource_roles_and_axis_mapping",
)


def _selection(raw):
    selected = yaml.safe_load(raw)
    fields = {"schema", "status", "software_intake_sha256", "namespace", "python", "canonical_source"}
    tensor = isinstance(selected, dict) and selected.get("schema") == TENSOR_SELECTION_SCHEMA
    if tensor:
        fields.add("tensor_arguments")
    if (
        not isinstance(selected, dict)
        or set(selected) != fields
        or selected["schema"] not in {SELECTION_SCHEMA, TENSOR_SELECTION_SCHEMA}
        or selected["status"] != "reviewed"
        or not isinstance(selected["namespace"], str)
        or not selected["namespace"].isidentifier()
        or not isinstance(selected["canonical_source"], dict)
        or set(selected["canonical_source"]) != {"checkout", "commit", "path"}
    ):
        raise RtlIntakeRefusal("operator schemas need a closed protected public source selection")
    if tensor and (
        selected["namespace"] != "aten"
        or not isinstance(selected["tensor_arguments"], dict)
        or set(selected["tensor_arguments"]) != {"compiler"}
        or not isinstance(selected["tensor_arguments"]["compiler"], str)
    ):
        raise RtlIntakeRefusal("Tensor argument conversion needs the explicit supported public API/compiler selection")
    return selected


def verify_record(record):
    """Reopen diagnostics only; saved JSON cannot recreate live issuer authority."""
    tensor = record.get("schema") == TENSOR_SCHEMA
    fields = {
        "schema",
        "software_intake_sha256",
        "selection_path",
        "observer_path",
        "tracked_source",
        "source_pins",
        "members",
        "unknowns",
    }
    if tensor:
        fields.add("tensor_argument_getter")
    if (
        set(record) != fields
        or record.get("schema") not in {SCHEMA, TENSOR_SCHEMA}
        or record["unknowns"] != list(_UNKNOWN)
    ):
        raise RtlIntakeRefusal("operator schema intake has the wrong record schema")
    for row in record["source_pins"]:
        RtlIntakePin(**row).verify()
    selected = _selection(Path(record["selection_path"]).read_bytes())
    if tensor != (selected["schema"] == TENSOR_SELECTION_SCHEMA):
        raise RtlIntakeRefusal("operator schema receipt version differs from its original public selection")
    source = selected["canonical_source"]
    git = _tracked_source(Path(source["checkout"]), Path(source["path"]), source["commit"])
    if git != record["tracked_source"] or selected["software_intake_sha256"] != record["software_intake_sha256"]:
        raise RtlIntakeRefusal("operator schema source/software correspondence changed")
    observer = module_source_path("merlin.targetgen.torch_schema_observer")
    if record["observer_path"] != str(observer):
        raise RtlIntakeRefusal("operator schema replay requires its actual selected fixed reader")
    if tensor:
        from .tensor_argument_intake import verify_getter

        getter = verify_getter(record["tensor_argument_getter"])
        if (
            getter["checkout"] != source["checkout"]
            or getter["commit"] != source["commit"]
            or getter["python"] != selected["python"]
            or getter["compiler"] != str(Path(selected["tensor_arguments"]["compiler"]).resolve(strict=True))
        ):
            raise RtlIntakeRefusal("Tensor argument APIs differ from selected schema runtime/public sources")
    for member in record["members"]:
        member_fields = {
            "graph_path",
            "request",
            "observation",
            "invocation",
            "public_semantics",
            "witnesses",
            "unknowns",
        }
        if tensor:
            member_fields |= {"tensor_arguments", "tensor_bindings"}
        if set(member) != member_fields:
            raise RtlIntakeRefusal("operator schema members need the complete closed original observation record")
        actual = I.verify(Path(member["invocation"]))
        # The fixed observer reads only the exact protected request and source.
        expected = [selected["python"], "-I", str(observer), member["request"], source["path"]]
        if actual["argv"] != expected or actual["stage"] != "native_operator_schema_observation":
            raise RtlIntakeRefusal("operator schema invocation does not bind its fixed observer and selected sources")
        I.require_environment(Path(member["invocation"]), environment={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
        observation = json.loads(Path(member["observation"]).read_bytes())
        if Path(member["observation"]).read_bytes() != Path(actual["stdout"]["path"]).read_bytes():
            raise RtlIntakeRefusal("operator schema rows differ from the actual native output")
        trace = json.loads(Path(member["graph_path"]).read_bytes())
        relation = original_use_def_semantics(trace)
        request = {
            "namespace": selected["namespace"],
            "captured_schemas": trace["graphs"]["original"].get("operator_schemas", {}),
            "operations": list(relation.operations),
        }
        if json.loads(Path(member["request"]).read_bytes()) != request:
            raise RtlIntakeRefusal("operator schema request differs from complete original source calls")
        tensor_arguments = None
        if tensor:
            from .tensor_argument_intake import verify_arguments

            tensor_arguments = verify_arguments(
                trace=trace,
                schema_observation=observation,
                getter=getter,
                member=member["tensor_arguments"],
            )
        effects = original_operator_effects(trace, observation, tensor_arguments=tensor_arguments)
        if (
            effects.public_semantics() != member["public_semantics"]
            or effects.witnesses() != member["witnesses"]
            or effects.unknowns() != member["unknowns"]
            or (tensor and effects.tensor_bindings() != member["tensor_bindings"])
        ):
            raise RtlIntakeRefusal("operator effect relations differ from exact original argument/result replay")
    return record


@dataclass(frozen=True, eq=False)
class IndependentOperatorSchemaIntake:
    software: IndependentSoftwareIntake
    source_pins: tuple[RtlIntakePin, ...]
    receipt_json: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt_json).hexdigest()

    def _identity(self):
        return hashlib.sha256(
            _json(
                {
                    "receipt_sha256": self.sha256,
                    "software_intake_sha256": self.software.sha256,
                    "source_pins": [pin.record() for pin in self.source_pins],
                }
            )
        ).hexdigest()

    def verify(self):
        if type(self.software) is not IndependentSoftwareIntake or _ISSUED.get(self) != self._identity():
            raise RtlIntakeRefusal("operator schemas require actual live independent issuance")
        self.software.verify()
        for pin in self.source_pins:
            pin.verify()
        record = verify_record(json.loads(self.receipt_json))
        if record["software_intake_sha256"] != self.software.sha256:
            raise RtlIntakeRefusal("operator schema software origin changed")
        expected = {pin.path for pin in self.software.source_pins if pin.role == "independent-example-graph"}
        if expected != {row["graph_path"] for row in record["members"]}:
            raise RtlIntakeRefusal("operator schema observations lost original independent graph membership")

    def record(self):
        self.verify()
        return json.loads(self.receipt_json)

    def effects(self, *, graph_path):
        self.verify()
        found = [
            row
            for row in json.loads(self.receipt_json)["members"]
            if row["graph_path"] == str(Path(graph_path).absolute())
        ]
        if len(found) != 1:
            raise RtlIntakeRefusal("operator schemas do not observe this exact selected original graph")
        row = found[0]
        tensor_arguments = (
            json.loads(Path(row["tensor_arguments"]["observation"]).read_bytes()) if "tensor_arguments" in row else None
        )
        return original_operator_effects(
            json.loads(Path(graph_path).read_bytes()),
            json.loads(Path(row["observation"]).read_bytes()),
            tensor_arguments=tensor_arguments,
        )


def issue_independent_operator_schema_intake(*, software, selection, forbidden_roots, output):
    """Run the fixed schema observer over every protected original example.

    No authored alias sets, expressions, op-name effects, shapes or factories
    are accepted. Native observation and clean tracked declarations have their
    own scope; effects not represented by schemas remain required UNKNOWN.
    """
    if type(software) is not IndependentSoftwareIntake or not isinstance(forbidden_roots, tuple) or not forbidden_roots:
        raise RtlIntakeRefusal("operator schema intake needs live minimal semantics and protected exclusions")
    software.verify()
    forbidden = tuple(_exclusion_prefix(path) for path in forbidden_roots)
    selection = _plain(selection)
    _outside(selection, forbidden)
    selected = _selection(selection.read_bytes())
    if selected["software_intake_sha256"] != software.sha256:
        raise RtlIntakeRefusal("operator schema selection belongs to different minimal semantics")
    source = selected["canonical_source"]
    checkout, declarations = _plain(source["checkout"], directory=True), _plain(source["path"])
    python = Path(selected["python"]).absolute()
    native_python = python.resolve(strict=True)
    observer = module_source_path("merlin.targetgen.torch_schema_observer")
    for path in (checkout, declarations, python, native_python, observer):
        _outside(path, forbidden)
    git = _tracked_source(checkout, declarations, source["commit"])
    pins = [
        _pin("protected-operator-schema-selection", selection, forbidden),
        _pin("canonical-public-operator-declarations", declarations, forbidden),
        _pin("native-framework-python", native_python, forbidden),
        *(
            _pin("operator-schema-reader", module_source_path(name), forbidden)
            for name in (
                "merlin.targetgen.torch_schema_observer",
                "merlin.targetgen.frontend_operator_effects",
                "merlin.targetgen.frontend_use_def",
                "merlin.targetgen.frontend_trace",
                "merlin_experiments.phase0.command_intake",
                "merlin_experiments.phase0.rtl_intake",
                "merlin.common.invocation_record",
                __name__,
            )
        ),
    ]
    config = python.parent.parent / "pyvenv.cfg"
    if config.is_file():
        pins.append(_pin("selected-python-venv", config, forbidden))
    destination = Path(output).absolute()
    if (
        destination.exists()
        or ".." in destination.parts
        or any(path.is_symlink() for path in (destination, *destination.parents))
    ):
        raise RtlIntakeRefusal("operator schema observations need a fresh ordinary output root")
    _outside(destination, forbidden)
    if any(Path(pin.path).is_relative_to(destination) for pin in (*pins, *software.source_pins)):
        raise RtlIntakeRefusal("operator schema output may not contain protected source inputs")
    destination.mkdir(parents=True, mode=0o700)
    getter = None
    if selected["schema"] == TENSOR_SELECTION_SCHEMA:
        from .tensor_argument_intake import prepare_getter

        getter = prepare_getter(
            python=python,
            compiler=selected["tensor_arguments"]["compiler"],
            checkout=checkout,
            commit=source["commit"],
            forbidden=forbidden,
            output=destination / "tensor-getter",
        )
        pins += [
            _pin("observed-tensor-argument-sdk-dependency", Path(path), forbidden)
            for path in getter["dependency_paths"]
        ]
        pins += [
            _pin("tensor-argument-reader", module_source_path(name), forbidden)
            for name in (
                "merlin.targetgen.torch_tensor_argument_observer",
                "merlin_experiments.phase0.tensor_argument_intake",
            )
        ]
    members = []
    for index, graph_pin in enumerate(pin for pin in software.source_pins if pin.role == "independent-example-graph"):
        trace = json.loads(Path(graph_pin.path).read_bytes())
        relation = original_use_def_semantics(trace)
        request = destination / ("request-" + str(index) + ".json")
        request.write_bytes(
            _json(
                {
                    "namespace": selected["namespace"],
                    "captured_schemas": trace["graphs"]["original"].get("operator_schemas", {}),
                    "operations": list(relation.operations),
                }
            )
        )
        observation = destination / ("observation-" + str(index) + ".json")
        completed = I.run(
            [str(python), "-I", str(observer), str(request), str(declarations)],
            directory=destination,
            stage="native_operator_schema_observation",
            inputs=(request, declarations, observer),
            dependencies=(native_python,),
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            timeout=60,
        )
        completed.check_returncode()
        observation.write_bytes(completed.stdout)
        observed = json.loads(completed.stdout)
        for name in ("torch_module", "native_schema_parser", "yaml_module"):
            pins.append(_pin("observed-framework-runtime", observed["runtime"][name], forbidden))
        tensor_member = None
        tensor_arguments = None
        if getter is not None:
            from .tensor_argument_intake import observe_arguments

            tensor_root = destination / ("tensor-arguments-" + str(index))
            tensor_root.mkdir()
            tensor_member = observe_arguments(
                trace=trace, schema_observation=observed, getter=getter, output=tensor_root
            )
            tensor_arguments = json.loads(Path(tensor_member["observation"]).read_bytes())
        effects = original_operator_effects(trace, observed, tensor_arguments=tensor_arguments)
        invocation = next(
            path
            for path in destination.glob("invocations/*/invocation.json")
            if json.loads(path.read_bytes())["argv"][-2] == str(request)
        )
        members.append(
            {
                "graph_path": graph_pin.path,
                "request": str(request),
                "observation": str(observation),
                "invocation": str(invocation),
                "public_semantics": effects.public_semantics(),
                "witnesses": effects.witnesses(),
                "unknowns": effects.unknowns(),
            }
        )
        if getter is not None:
            members[-1].update(tensor_arguments=tensor_member, tensor_bindings=effects.tensor_bindings())
        pins.append(graph_pin)
    pins += [
        _pin("operator-schema-native-evidence", path, forbidden) for path in destination.rglob("*") if path.is_file()
    ]
    pins = tuple({(pin.role, pin.path): pin for pin in pins}.values())
    record = {
        "schema": TENSOR_SCHEMA if getter is not None else SCHEMA,
        "software_intake_sha256": software.sha256,
        "selection_path": str(selection),
        "observer_path": str(observer),
        "tracked_source": git,
        "source_pins": [pin.record() for pin in pins],
        "members": members,
        "unknowns": list(_UNKNOWN),
    }
    if getter is not None:
        record["tensor_argument_getter"] = getter
    receipt = destination / "intake.json"
    receipt.write_bytes(_json(record))
    authority = IndependentOperatorSchemaIntake(
        software, (*pins, _pin("operator-schema-intake-receipt", receipt, forbidden)), receipt.read_bytes()
    )
    _ISSUED[authority] = authority._identity()
    authority.verify()
    return authority
