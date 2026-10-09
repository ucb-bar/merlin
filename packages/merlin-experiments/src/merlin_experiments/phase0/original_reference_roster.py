"""Private original typed source/reference roster with actual native comparison.

Every original call/cohort remains required. Live source owners, exact fixed
observer invocations and independent full comparisons establish only bounded
software observations, never whole numerical domains or phase/target authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path, schemas_dir
from merlin.common.strict_json import loads
from merlin.targetgen import original_operator_sources as S
from merlin.targetgen.frontend_original_call import call_contracts
from merlin.targetgen.original_operator_reference import OriginalReferenceBudget, prepare_original_reference
from merlin.targetgen.original_reference_values import TypedReferenceTensor as T

from . import component_execution_budget as E
from . import original_reference_plan as P
from . import original_schema_defaults as D
from .component_semantic_basis import ComponentSemanticBasis
from .operator_schema_intake import IndependentOperatorSchemaIntake, _selection
from .rtl_intake import RtlIntakePin

SCHEMA = "merlin.original_reference_roster.v1"
BATCH_SCHEMA = "merlin.original_reference_roster.v2"
POINTWISE_SCHEMA = "merlin.original_reference_roster.v3"
_ISSUED = weakref.WeakKeyDictionary()
_UNKNOWN = (
    "original_numerical_domain",
    "general_framework_equivalence",
    "original_operation_correspondence",
    "physical_effects",
    "target_support",
    "phase1_candidate_execution",
    "native_runtime_dependency_closure",
)
_SCOPE = "private bounded original source/reference observations; no numerical-domain or phase admission"
_READERS = (
    "merlin_experiments.phase0.original_reference_products",
    __name__,
    P.__name__,
    D.__name__,
    "merlin_experiments.phase0.original_reference_observer",
    "merlin_experiments.phase0.original_pointwise_reference_observer",
    "merlin.targetgen.original_operator_reference",
    "merlin.targetgen.original_pointwise_reference",
    "merlin.targetgen.original_pointwise_sources",
    "merlin.targetgen.original_reference_values",
    "merlin.targetgen.original_operator_sources",
    "merlin.targetgen.frontend_original_call",
    "merlin.targetgen.frontend_typed_add",
    "merlin.targetgen.torch_schema_defaults_observer",
    E.__name__,
    "merlin_experiments.phase0.original_call_sources",
    "merlin.common.quant_formats",
    "merlin.targetgen.software_spec",
    "merlin.common.schemas",
    "merlin.common.yaml",
    "merlin.common.strict_json",
    "merlin.common.jsonio",
)


def _plain(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file():
        raise ValueError("original references require ordinary exact selected files")
    return path


def _pin(path):
    path = _plain(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _sources(selection):
    if os.environ.get("MERLIN_QUANT_FORMATS") is not None:
        raise ValueError("original reference roster has no explicitly selected format overlay")
    readers = list(_READERS)
    if P.transport(loads(Path(selection).read_bytes())) == "batch.v1":
        readers += [
            "merlin_experiments.phase0.original_schema_batch",
            "merlin.targetgen.torch_schema_batch_observer",
            "merlin.targetgen.torch_schema_observer",
        ]
    return [module_source_path(name) for name in readers] + [
        Path(selection),
        schemas_dir() / "quant_formats.registry.yaml",
        schemas_dir() / "quant_format.schema.yaml",
    ]


def record_schema(selection):
    if selection["schema"] == P.POINTWISE_SCHEMA:
        return POINTWISE_SCHEMA
    return BATCH_SCHEMA if P.transport(selection) == "batch.v1" else SCHEMA


def _observer(selection):
    name = (
        "original_pointwise_reference_observer"
        if selection["schema"] == P.POINTWISE_SCHEMA
        else "original_reference_observer"
    )
    return module_source_path("merlin_experiments.phase0." + name)


def _paths(owner):
    return {
        name: owner / filename
        for name, filename in {
            "source": "loader.py",
            "metadata": "metadata.json",
            "inputs": "inputs.json",
            "reference": "reference.json",
            "actual": "actual.json",
            "comparison": "comparison.json",
        }.items()
    }


def _products(paths):
    return {key: _pin(path) for key, path in paths.items() if path.is_file()}


def _comparison(contract, inputs, path):
    from .original_reference_products import native_outputs

    return contract.compare(inputs, native_outputs(contract, path))


def _originals(intake, basis):
    if type(intake) is not IndependentOperatorSchemaIntake or type(basis) is not ComponentSemanticBasis:
        raise ValueError("original references require exact live schema intake and selected semantic basis")
    intake.verify()
    software = json.loads(intake.software.receipt_json)
    if software["semantic_basis_sha256"] != basis.source.sha256:
        raise ValueError("original reference basis differs from the live original software selection")
    pins = {pin.path: pin.sha256 for pin in intake.software.source_pins}
    if pins.get(basis.source.path) != basis.source.sha256:
        raise ValueError("original reference basis lacks its original protected source membership")
    path = _plain(basis.source.path)
    reloaded = ComponentSemanticBasis.load(path.read_bytes(), source=basis.source, parent=path.parent, routing={})
    if reloaded != basis:
        raise ValueError("original reference semantic basis changed its complete private membership")
    schema = intake.record()
    if {row["graph_path"] for row in schema["members"]} != {row.path for row in basis.graph_sources}:
        raise ValueError("original reference schema and basis original source rosters differ")
    return schema


def _drafts(defaults, *, schema, basis, selection):
    original_ids = [row["id"] for row in json.loads(basis.declaration_json)["members"]]
    if [row["graph_path"] for row in defaults] != [row.path for row in basis.graph_sources]:
        raise ValueError("original references lost exact ordered graph/default membership")
    calls, sources = [], {}
    for original_id, row in zip(original_ids, defaults, strict=True):
        trace, schemas, observed = D.verify_member(
            row, schema_record=schema, version=2, transport=P.transport(selection)
        )
        forms = [
            form
            for factory in (S.matmul_forms, S.original_add_forms, S.conv2d_forms)
            for form in factory(trace, schemas, observed)
        ]
        if selection["schema"] == P.POINTWISE_SCHEMA:
            from merlin.targetgen.original_pointwise_sources import pointwise_forms

            forms += pointwise_forms(trace, schemas, observed, version=2)
        indexed = {form["node"]: form for form in forms}
        for call in call_contracts(trace, schemas, observed):
            for cohort in P.COHORTS:
                for extent in selection["cohorts"][cohort]:
                    member = {
                        "original_member_id": original_id,
                        "graph_path": row["graph_path"],
                        "node": call["node"],
                        "target": call["target"],
                        "call": call,
                        "cohort": cohort,
                        "extent": extent,
                        "state": "unavailable",
                        "required_unknowns": list(_UNKNOWN),
                    }
                    try:
                        form = copy.deepcopy(indexed[call["node"]])
                        if form["status"] != "supported":
                            raise ValueError(form["reason"])
                        selected = P.selected_policy(selection, form)
                        form["source_numerical_semantics"] = selected.record()
                        factories = {
                            "aten.matmul.default": S.matmul_source,
                            "aten.add.Tensor": S.add_source,
                            "aten.conv2d.default": S.conv2d_source,
                        }
                        if selection["schema"] == P.POINTWISE_SCHEMA:
                            from merlin.targetgen.original_pointwise_reference import OPERATIONS
                            from merlin.targetgen.original_pointwise_sources import pointwise_source

                            factories.update(dict.fromkeys(OPERATIONS, pointwise_source))
                        factory = factories[form["target"]]
                        source = factory(
                            form, extent=extent, max_tensor_elements=selection["source_budget"]["max_tensor_elements"]
                        )
                        contract = prepare_original_reference(
                            form,
                            source,
                            extent=extent,
                            policy=selected,
                            budget=OriginalReferenceBudget(**selection["reference_budget"]),
                            output_byteorder=selection["byteorder"],
                        )
                        for input_slot in source.metadata()["inputs"]:
                            P.palette(selection, input_slot["dtype"])
                        cost = P.measure(contract, selection)
                        member.update(
                            form=form,
                            policy=selected.record(),
                            contract_sha256=contract.sha256,
                            cost=cost,
                            source_cost={
                                "tensor_elements": source.metadata()["tensor_elements"],
                                "scalar_products": source.metadata()["scalar_products"],
                                "source_bytes": len(source.loader.encode()) + len(source.metadata_json.encode()),
                            },
                        )
                        sources[len(calls)] = contract
                    except (KeyError, TypeError, ValueError) as error:
                        member["reason"] = str(error)
                    calls.append(member)
    totals = dict.fromkeys(E._METRICS, 0)
    source_totals = dict.fromkeys(("tensor_elements", "scalar_products", "source_bytes"), 0)
    budget = selection["source_budget"]
    for index, member in enumerate(calls):
        if index not in sources:
            continue
        limits = []
        if len(calls) > budget["max_sources"]:
            limits.append("complete source member count")
        limits += [
            key
            for key, value in member["source_cost"].items()
            if value > budget["max_" + key] or source_totals[key] + value > budget["max_total_" + key]
        ]
        limits += E._exceeded(selection["execution_budget"], member["cost"], totals)
        if limits:
            member["reason"] = "original reference preallocation budget exceeded: " + ", ".join(limits)
            sources.pop(index)
            continue
        for key in totals:
            totals[key] += member["cost"][key]
        for key in source_totals:
            source_totals[key] += member["source_cost"][key]
        member.update(state="planned", reason="complete source/native/reference budgets checked before allocation")
    return calls, sources, {"execution": totals, "source": source_totals}


def _tensor_record(tensor):
    return {
        "name": tensor.name,
        "dtype": tensor.dtype,
        "shape": list(tensor.shape),
        "byteorder": tensor.byteorder,
        "data_hex": tensor.data.hex(),
    }


def _tensors(rows):
    return tuple(
        T(row["name"], row["dtype"], tuple(row["shape"]), bytes.fromhex(row["data_hex"]), row["byteorder"])
        for row in rows
    )


def _stimulus(contract, selection):
    import math

    tensors = []
    for index, row in enumerate(contract.verify()["inputs"]):
        values = P.palette(selection, row["dtype"])
        tensors.append(
            T.from_values(
                row["name"],
                row["dtype"],
                row["shape"],
                [values[(j + index) % len(values)] for j in range(math.prod(row["shape"]))],
                byteorder=selection["byteorder"],
            )
        )
    return tuple(tensors)


def _write(path, value):
    path.write_bytes(_json(value) + b"\n")
    path.chmod(0o600)


def _native(invocation, *, argv, inputs):
    document = loads(_plain(invocation).read_bytes())
    if document.get("status") == "completed":
        document = I.require_environment(Path(invocation), environment=D.ENVIRONMENT)
    else:
        if (
            document.get("schema") != I.SCHEMA
            or document.get("kind") != "subprocess"
            or document.get("status") != "failed"
            or type(document.get("returncode")) is not int
            or document["returncode"] == 0
            or document.get("environment") != I.environment_identity(D.ENVIRONMENT)
            or any(
                document.get(k) is not True
                for k in ("inputs_unchanged", "dependencies_unchanged", "executable_unchanged")
            )
        ):
            raise ValueError("original reference native failure lacks a completed unchanged process")
        for pin in [
            document["executable"],
            document["stdout"],
            document["stderr"],
            *document["inputs"],
            *document["dependencies"],
            *document["outputs"],
        ]:
            if pin.get("sha256") is None and pin in document["outputs"] and not Path(pin["path"]).exists():
                continue
            if _pin(pin["path"]) != pin:
                raise ValueError("original reference native failure product/source changed")
    if (
        document["stage"] != "native_original_reference"
        or document["argv"] != argv
        or {pin["path"] for pin in document["inputs"]} != {str(Path(path).resolve()) for path in inputs}
    ):
        raise ValueError("original reference native invocation lost its fixed source/input/observer membership")
    return document


def _evaluate(member, contract, *, selection, python, owner):
    observer = _observer(selection)
    paths = _paths(owner)
    paths["source"].write_text(contract.source.loader)
    paths["source"].chmod(0o600)
    paths["metadata"].write_text(contract.source.metadata_json)
    paths["metadata"].chmod(0o600)
    inputs = _stimulus(contract, selection)
    _write(paths["inputs"], [_tensor_record(tensor) for tensor in inputs])
    try:
        reference = contract.evaluate(inputs)
    except ValueError as error:
        member.update(
            state="reference_unavailable",
            reason=str(error),
            products={key: _pin(path) for key, path in paths.items() if path.is_file()},
        )
        return
    _write(paths["reference"], [_tensor_record(tensor) for tensor in reference])
    argv = [
        python,
        "-I",
        str(observer),
        str(paths["source"]),
        str(paths["metadata"]),
        str(paths["inputs"]),
        str(paths["actual"]),
    ]
    result = I.run(
        argv,
        directory=owner,
        stage="native_original_reference",
        inputs=(observer, paths["source"], paths["metadata"], paths["inputs"]),
        outputs=(paths["actual"],),
        env=D.ENVIRONMENT,
        capture_output=True,
        timeout=60,
    )
    invocation = next((owner / "invocations").glob("*/invocation.json"))
    member["invocation"] = _pin(invocation)
    if result.returncode:
        member.update(state="native_failed", reason="fixed original native source process failed")
    else:
        try:
            comparison = _comparison(contract, inputs, paths["actual"])
            _write(paths["comparison"], comparison)
            member.update(
                state="reference_checked" if comparison["passed"] else "comparison_refuted",
                reason="complete bounded native/source-reference comparison; domain remains unqualified",
            )
        except (ValueError, KeyError, TypeError) as error:
            member.update(state="comparison_unavailable", reason=str(error))
    member["products"] = _products(paths)


@dataclass(frozen=True, eq=False)
class OriginalReferenceRoster:
    schema_intake: IndependentOperatorSchemaIntake
    basis: ComponentSemanticBasis
    selection: Path
    selection_sha256: str
    source_pins: tuple[RtlIntakePin, ...]
    receipt_json: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt_json).hexdigest()

    def verify(self):
        if _ISSUED.get(self) != (self.sha256, self.selection_sha256):
            raise ValueError("original reference roster requires its actual live source preparation")
        for pin in self.source_pins:
            pin.verify()
        if _pin(self.selection)["sha256"] != self.selection_sha256:
            raise ValueError("original reference protected selection changed")
        verify(
            self.record_without_verification(),
            schema_intake=self.schema_intake,
            basis=self.basis,
            selection=self.selection,
        )

    def record_without_verification(self):
        return loads(self.receipt_json)

    def record(self):
        self.verify()
        return self.record_without_verification()


def prepare(*, schema_intake, basis, selection, destination):
    schema = _originals(schema_intake, basis)
    selected = P.validate(loads(_plain(selection).read_bytes()))
    if (
        selected["operator_schema_intake_sha256"] != schema_intake.sha256
        or selected["semantic_basis_sha256"] != basis.source.sha256
    ):
        raise ValueError("original reference contract does not select these exact live original sources")
    pins = tuple(
        RtlIntakePin("private-original-reference-source", str(path), _pin(path)["sha256"])
        for path in _sources(selection)
    )
    destination = Path(destination).absolute()
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("original reference products require an ordinary new private destination")
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    defaults = D.observe_members(
        schema_record=schema,
        basis=basis,
        destination=destination / "defaults",
        version=2,
        transport=P.transport(selected),
    )
    rows, contracts, totals = _drafts(defaults, schema=schema, basis=basis, selection=selected)
    python = _selection(Path(schema["selection_path"]).read_bytes())["python"]
    for index, member in enumerate(rows):
        if index in contracts:
            owner = destination / str(index)
            owner.mkdir(mode=0o700)
            _evaluate(member, contracts[index], selection=selected, python=python, owner=owner)
        for pin in pins:
            pin.verify()
        schema_intake.verify()
    record = {
        "schema": record_schema(selected),
        "operator_schema_intake_sha256": schema_intake.sha256,
        "software_intake_sha256": schema_intake.software.sha256,
        "semantic_basis_sha256": basis.source.sha256,
        "selection": _pin(selection),
        "destination": str(destination),
        "defaults": defaults,
        "members": rows,
        "totals": totals,
        "source_pins": [dict(path=pin.path, sha256=pin.sha256) for pin in pins],
        "scope": _SCOPE,
    }
    _write(destination / "roster.json", record)
    result = OriginalReferenceRoster(
        schema_intake, basis, Path(selection).absolute(), _pin(selection)["sha256"], pins, _json(record)
    )
    _ISSUED[result] = (result.sha256, result.selection_sha256)
    result.verify()
    return result


def verify(record, *, schema_intake, basis, selection):
    from .original_reference_products import verify as replay

    return replay(record, schema_intake=schema_intake, basis=basis, selection=selection)
