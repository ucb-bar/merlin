"""Live original scalar source slots joined to registered upstream products.

This opt-in consumer retains the exact original guard/private membership. Body
construction checks grant no numerical/reference, effect, hardware or coverage
admission, and leave historical source and automatic policies unchanged.
"""

from __future__ import annotations

import hashlib
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.jsonio import canonical_json
from merlin.common.paths import module_source_path
from merlin.common.selected_pin_replay import replay_selected_pins
from merlin.common.strict_json import loads
from merlin.targetgen import original_scalar_binary_correspondence as B
from merlin.targetgen import original_scalar_binary_sources as S

from . import original_call_sources as C
from . import original_schema_defaults as D
from . import original_standard_ir_plan as P
from .component_semantic_basis import ComponentSemanticBasis
from .operator_schema_intake import IndependentOperatorSchemaIntake, _selection
from .original_reference_roster import _pin, _plain

SCHEMA = "merlin.original_scalar_conversion.v1"
INTEGER_SCHEMA = "merlin.original_scalar_conversion.v2"
METADATA_SCHEMA = "merlin.original_scalar_conversion.v3"
TRIANGULAR_SCHEMA = "merlin.original_scalar_conversion.v4"
SELECTION_SCHEMA = "merlin.original_scalar_conversion_selection.v1"
INTEGER_SELECTION_SCHEMA = "merlin.original_scalar_conversion_selection.v2"
METADATA_SELECTION_SCHEMA = "merlin.original_scalar_conversion_selection.v3"
TRIANGULAR_SELECTION_SCHEMA = "merlin.original_scalar_conversion_selection.v4"
_ISSUED = weakref.WeakKeyDictionary()
_LIMITS = {
    *B._LIMITS,
    "max_members",
    "max_total_tensor_elements",
    "max_total_source_bytes",
    "timeout_s",
}
_INTEGER_LIMITS = {*_LIMITS, "max_promotion_tensor_elements", "max_total_promotion_tensor_elements"}
_UNKNOWN = (
    "registered_torch_tensor_literal_source_correspondence",
    "original_numerical_policy_and_reference",
    "general_framework_and_compiled_equivalence",
    "original_reviewed_operation_owner",
    "source_effect_and_resource_mapping",
    "target_support_and_runtime_dependency_closure",
    "mandatory_coverage_and_phase_admission",
)
_READERS = (
    __name__,
    "merlin_experiments.phase0.original_scalar_conversion_observer",
    B.__name__,
    "merlin.targetgen.application_graph",
    "merlin.targetgen.access_observations",
    "merlin.common.mlir_query",
    "merlin.targetgen.contract.mlir_source_admission",
    "merlin.xdsl_dialects._common",
    "merlin.xdsl_dialects.fp8",
    "merlin.common.jsonio",
    "merlin.common.strict_json",
    "merlin.common.selected_pin_replay",
    I.__name__,
    P.__name__,
    "merlin_experiments.phase0.original_reference_roster",
)


def _digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def validate_selection(selected, *, source_record, schema_intake, basis):
    if (
        type(selected) is not dict
        or set(selected)
        != {
            "schema",
            "source_record_sha256",
            "operator_schema_intake_sha256",
            "semantic_basis_sha256",
            "capture_checkout",
            "capture_commit",
            "mlir_opt",
            "budget",
        }
        or selected["schema"]
        not in {SELECTION_SCHEMA, INTEGER_SELECTION_SCHEMA, METADATA_SELECTION_SCHEMA, TRIANGULAR_SELECTION_SCHEMA}
        or selected["source_record_sha256"] != _digest(source_record)
        or selected["operator_schema_intake_sha256"] != schema_intake.sha256
        or selected["semantic_basis_sha256"] != basis.source.sha256
        or type(selected["capture_commit"]) is not str
        or len(selected["capture_commit"]) != 40
        or any(c not in "0123456789abcdef" for c in selected["capture_commit"])
    ):
        raise ValueError("scalar conversion requires its explicit exact original-source and public compiler selection")
    if (
        source_record.get("schema")
        != (
            {
                SELECTION_SCHEMA: C.SCALAR_BINARY_SCHEMA,
                INTEGER_SELECTION_SCHEMA: C.INTEGER_SCALAR_SCHEMA,
                METADATA_SELECTION_SCHEMA: C.METADATA_SCHEMA,
                TRIANGULAR_SELECTION_SCHEMA: C.TRIANGULAR_SCHEMA,
            }[selected["schema"]]
        )
    ):
        raise ValueError("scalar conversion selection cannot widen a prior original-source vocabulary")
    limits = selected["budget"]
    if (
        type(limits) is not dict
        or set(limits)
        != (
            _INTEGER_LIMITS
            if selected["schema"] in {INTEGER_SELECTION_SCHEMA, METADATA_SELECTION_SCHEMA, TRIANGULAR_SELECTION_SCHEMA}
            else _LIMITS
        )
        or any(type(value) is not int or value < 1 for value in limits.values())
        or limits["timeout_s"] > 180
    ):
        raise ValueError("scalar conversion requires complete bounded logical/parser/native reservations")
    B.validate_limits({key: limits[key] for key in B._LIMITS})
    root = Path(selected["capture_checkout"])
    if not root.is_absolute() or not root.is_dir() or any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("scalar conversion requires an ordinary explicitly selected public compiler checkout")
    _plain(selected["mlir_opt"])
    return selected


def required_members(source_record, *, basis, budget):
    """Preserve every scalar call's original three source slots before expansion."""
    if source_record.get("schema") not in {
        C.SCALAR_BINARY_SCHEMA,
        C.INTEGER_SCALAR_SCHEMA,
        C.METADATA_SCHEMA,
        C.TRIANGULAR_SCHEMA,
    }:
        raise ValueError("scalar conversion requires the opt-in original scalar source vocabulary")
    version = 2 if source_record["schema"] in {C.INTEGER_SCALAR_SCHEMA, C.METADATA_SCHEMA, C.TRIANGULAR_SCHEMA} else 1
    if version == 2 and (
        type(budget) is not dict
        or set(budget) != _INTEGER_LIMITS
        or any(type(value) is not int or value < 1 for value in budget.values())
    ):
        raise ValueError("integer scalar conversion needs complete explicit promotion payload reservations")
    originals = loads(basis.declaration_json)["members"]
    if len(originals) != len(source_record["members"]):
        raise ValueError("scalar conversion lost the complete original graph roster")
    count = sum(
        member["target"] in S.TARGETS for graph in source_record["members"] for member in graph["source_members"]
    )
    if count > budget["max_members"]:
        raise ValueError("scalar conversion complete original membership exceeds its preallocation reservation")
    result, totals = [], {"tensor_elements": 0, "source_bytes": 0}
    if version == 2:
        totals["promotion_tensor_elements"] = 0
    for original, graph in zip(originals, source_record["members"], strict=True):
        forms = {
            form["node"]: form
            for form in graph["forms"]
            if form["form_schema"] == (S.INTEGER_FORM_SCHEMA if version == 2 else S.FORM_SCHEMA)
        }
        calls = [call for call in graph["calls"] if call["target"] in S.TARGETS]
        expected = [
            (call["node"], call["target"], cohort, extent)
            for call in calls
            for cohort, extent in C.required_source_cohorts()
        ]
        members = [member for member in graph["source_members"] if member["target"] in S.TARGETS]
        if canonical_json(
            [(row["node"], row["target"], row["cohort"], row["extent"]) for row in members]
        ) != canonical_json(expected):
            raise ValueError("scalar conversion cannot drop, reorder or substitute an original guard/private slot")
        for member in members:
            row = {
                "index": len(result),
                "original_member_id": original["id"],
                "graph_path": graph["graph_path"],
                **{key: member[key] for key in ("node", "target", "cohort", "extent")},
                "state": "unavailable",
            }
            source = None
            try:
                if member["status"] != "source_constructed":
                    raise ValueError(member["reason"])
                form = forms[member["node"]]
                source = S.scalar_binary_source(
                    form, extent=member["extent"], max_tensor_elements=budget["max_tensor_elements"]
                )
                B.coefficient_bits(form["parameters"]["other"], version=version)
                cost = {
                    "tensor_elements": source.metadata()["tensor_elements"],
                    "source_bytes": len(source.loader.encode()),
                }
                if version == 2:
                    # Input, both retained dispatch readouts and wrapped scalar
                    # are explicitly reserved before the native source executes.
                    count = source.metadata()["scalar_products"]
                    cost["promotion_tensor_elements"] = 3 * count + 1 if form.get("tensor_binding") else 0
                    if cost["promotion_tensor_elements"] > budget["max_promotion_tensor_elements"]:
                        raise ValueError(
                            "integer promotion complete logical readouts exceed their preallocation budget"
                        )
                if cost["source_bytes"] > budget["max_source_bytes"] or any(
                    totals[key] + cost[key] > budget["max_total_" + key] for key in totals
                ):
                    raise ValueError("scalar conversion exceeds its complete original construction reservation")
                if (
                    canonical_json(member["metadata"]) != canonical_json(source.metadata())
                    or member["source"]["sha256"] != hashlib.sha256(source.loader.encode()).hexdigest()
                    or _plain(member["source"]["path"]).read_bytes() != source.loader.encode()
                ):
                    raise ValueError(
                        "scalar conversion original source changed its exact freshly reconstructed factory"
                    )
                for key in totals:
                    totals[key] += cost[key]
                row.update(state="requested", form=form, source=member["source"], cost=cost)
            except (KeyError, TypeError, ValueError) as error:
                row.update(reason=str(error))
                source = None
            result.append((row, source))
    # Reserve every potential complete product before native conversion can
    # materialize any source, trace or registry frame. The observation frame
    # receives its own full reservation; actual logical counts stay separate.
    reservation = (3 * sum(source is not None for _, source in result) + 1) * budget["max_source_bytes"]
    if totals["source_bytes"] + reservation > budget["max_total_source_bytes"]:
        raise ValueError("scalar conversion complete product reservation exceeds its aggregate byte budget")
    totals["reserved_product_bytes"] = reservation
    return result, totals


def _replay_originals(schema_intake, basis, source_record, numerical_semantics):
    if type(schema_intake) is not IndependentOperatorSchemaIntake or type(basis) is not ComponentSemanticBasis:
        raise ValueError("scalar conversion needs its actual live original public schema and semantic basis")
    schema = schema_intake.record()
    for pin in basis.sources():
        if _pin(pin["path"])["sha256"] != pin["sha256"]:
            raise ValueError("scalar conversion original public basis bytes changed")
    C.verify(source_record, schema_record=schema, basis=basis, numerical_semantics=numerical_semantics)
    return schema


def _request(members, capture, budget, *, version=1, getter=None):
    request = {
        "schema": "merlin.original_scalar_conversion_request.v2"
        if version == 2
        else "merlin.original_scalar_conversion_request.v1",
        "capture_sources": capture,
        "max_product_bytes": budget["max_source_bytes"],
        "members": [
            {
                "index": row["index"],
                "target": row["target"],
                "source": row["source"]["path"],
                "source_sha256": row["source"]["sha256"],
                **(
                    {
                        "original_tensor_binding": row["form"].get("tensor_binding"),
                        "input": {
                            "kind": "tensor",
                            "dtype": "torch.float32",
                            "layout": "torch.strided",
                            "device": "cpu",
                            "shape": source.metadata()["inputs"][0]["shape"],
                        },
                        "promotion_tensor_elements": row["cost"]["promotion_tensor_elements"],
                    }
                    if version == 2
                    else {}
                ),
            }
            for row, source in members
            if source is not None
        ],
    }
    if version == 2:
        if getter is None:
            raise ValueError("integer scalar conversion needs the original live schema Tensor getter selection")
        request["tensor_argument_getter"] = _pin(getter["getter"])
        request["max_tensor_elements"] = budget["max_tensor_elements"]
        request["max_promotion_tensor_elements"] = budget["max_promotion_tensor_elements"]
        request["max_total_promotion_tensor_elements"] = budget["max_total_promotion_tensor_elements"]
    return request


def _products(destination, index):
    return {
        name: destination / str(index) / filename
        for name, filename in (
            ("source", "source.mlir"),
            ("trace", "trace.json"),
            ("registry", "registry.json"),
        )
    }


def _pins(selection, capture, *, version=1, getter=None, source_version=None):
    readers = [
        module_source_path(name)
        for name in (
            *_READERS,
            *C.reader_modules(source_version if source_version is not None else 7 if version == 2 else 6),
        )
    ]
    if version == 2:
        readers.append(module_source_path("merlin.targetgen.torch_tensor_argument_observer"))
        readers.extend(Path(getter[key]) for key in ("getter", "cpp", "sdk"))
    readers.extend(module_source_path("xdsl").parent.rglob("*.py"))
    return [_pin(path) for path in sorted(set(readers))] + [_pin(selection), *capture]


@dataclass(frozen=True, eq=False)
class OriginalScalarConversion:
    schema_intake: IndependentOperatorSchemaIntake
    basis: ComponentSemanticBasis
    source_json: bytes
    numerical_json: bytes
    selection: Path
    receipt_json: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt_json).hexdigest()

    def record(self):
        if _ISSUED.get(self) != self.sha256:
            raise ValueError("scalar conversion requires its actual live registered observation")
        record = loads(self.receipt_json)
        actual = _record(self, Path(record["destination"]))
        if canonical_json(record) != canonical_json(actual):
            raise ValueError("scalar conversion record differs from complete live original/source/product replay")
        return actual


def _record(owner, destination):
    source_record, numerical = loads(owner.source_json), loads(owner.numerical_json)
    schema = _replay_originals(owner.schema_intake, owner.basis, source_record, numerical)
    selected = validate_selection(
        loads(owner.selection.read_bytes()),
        source_record=source_record,
        schema_intake=owner.schema_intake,
        basis=owner.basis,
    )
    capture = P.capture_sources(selected)
    members, totals = required_members(source_record, basis=owner.basis, budget=selected["budget"])
    version = (
        2
        if selected["schema"] in {INTEGER_SELECTION_SCHEMA, METADATA_SELECTION_SCHEMA, TRIANGULAR_SELECTION_SCHEMA}
        else 1
    )
    getter = schema.get("tensor_argument_getter") if version == 2 else None
    request = _request(members, capture, selected["budget"], version=version, getter=getter)
    request_path, observation_path = destination / "request.json", destination / "products/observation.json"
    if canonical_json(loads(request_path.read_bytes())) != canonical_json(request):
        raise ValueError("scalar conversion actual request changed original complete membership/source/budgets")
    native = loads(observation_path.read_bytes())
    if (
        set(native) != {"schema", "rows", "dependencies"}
        or native["schema"]
        != (
            "merlin.native_original_scalar_conversion.v2"
            if version == 2
            else "merlin.native_original_scalar_conversion.v1"
        )
        or any(type(row.get("index")) is not int for row in native["rows"])
        or [row["index"] for row in native["rows"]] != [row["index"] for row in request["members"]]
    ):
        raise ValueError("scalar conversion actual native observation lost its requested original members")
    pins = _pins(
        owner.selection,
        capture,
        version=version,
        getter=getter,
        source_version={METADATA_SELECTION_SCHEMA: 8, TRIANGULAR_SELECTION_SCHEMA: 9}.get(selected["schema"]),
    )
    observer = module_source_path(_READERS[1])
    invocations = tuple((destination / "native/invocations").glob("*/invocation.json"))
    if len(invocations) != 1:
        raise ValueError("scalar conversion lost its unique actual ordinary conversion process")
    invocation = invocations[0]
    actual = I.require_environment(invocation, environment=D.ENVIRONMENT)
    wanted_inputs = {observer, request_path, *(Path(row["source"]) for row in request["members"])}
    if (
        actual["argv"]
        != [
            _selection(Path(schema["selection_path"]).read_bytes())["python"],
            "-I",
            "-B",
            str(observer),
            str(request_path),
            selected["capture_checkout"],
            str(destination / "products"),
        ]
        or actual["stage"] != "original_registered_scalar_conversion"
        or actual["cwd"] != str(destination)
        or {pin["path"] for pin in actual["inputs"]} != {str(path) for path in wanted_inputs}
        or {pin["path"] for pin in actual["dependencies"]} != {pin["path"] for pin in pins}
    ):
        raise ValueError("scalar conversion lost fixed actual execution/source/member/ENV joins")
    for dependency in native["dependencies"]:
        if _pin(dependency["path"])["sha256"] != dependency["sha256"]:
            raise ValueError("scalar conversion observed runtime bytes changed; closure authority remains unknown")
    observations = {row["index"]: row for row in native["rows"]}
    results, outputs = [], {str(observation_path)}
    with replay_selected_pins((Path(selected["mlir_opt"]),), max_pins=1) as replay:
        for row, source in members:
            result = dict(row)
            if source is not None:
                observed = observations[row["index"]]
                if observed["status"] == "unavailable" and set(observed) == {"index", "status", "reason"}:
                    result.update(state="unavailable", reason=observed["reason"])
                elif observed != {"index": row["index"], "status": "converted"}:
                    raise ValueError("scalar conversion changed its native original outcome")
                else:
                    products = _products(destination / "products", row["index"])
                    if any(
                        _plain(path).stat().st_size > selected["budget"]["max_source_bytes"]
                        for path in products.values()
                    ):
                        raise ValueError("scalar conversion product exceeds its declared source byte budget")
                    outputs.update(str(path) for path in products.values())
                    binding = B.verify(
                        row["form"],
                        source,
                        extent=row["extent"],
                        text=products["source"].read_text(),
                        trace=loads(products["trace"].read_bytes()),
                        registry=loads(products["registry"].read_bytes()),
                        source_inventory={pin["path"]: pin["sha256"] for pin in capture},
                        limits={key: selected["budget"][key] for key in B._LIMITS},
                    )
                    parser_path = destination / "parsers" / str(row["index"])
                    parser_invocations = tuple((parser_path / "invocations").glob("*/invocation.json"))
                    if len(parser_invocations) != 1:
                        raise ValueError("scalar conversion lacks its selected actual stock parser observation")
                    parser = I.require_environment(parser_invocations[0], environment=D.ENVIRONMENT, pin_replay=replay)
                    verified = parser_path / "verified.mlir"
                    if (
                        parser["argv"] != [selected["mlir_opt"], str(products["source"]), "-o", str(verified)]
                        or parser["stage"] != "original_registered_scalar_stock_parser"
                        or parser["cwd"] != str(destination)
                        or parser["inputs"] != [_pin(products["source"])]
                        or parser["outputs"] != [_pin(verified)]
                        or parser["dependencies"]
                    ):
                        raise ValueError("scalar conversion changed its exact stock parser/product observation")
                    result.update(
                        state="registered_source_checked",
                        correspondence=binding,
                        products={key: _pin(path) for key, path in products.items()},
                        parser_invocation=_pin(parser_invocations[0]),
                        verified=_pin(verified),
                    )
            results.append(result)
    if {pin["path"] for pin in actual["outputs"]} != outputs:
        raise ValueError("scalar conversion process omitted or substituted an actual complete product")
    return {
        "schema": TRIANGULAR_SCHEMA
        if selected["schema"] == TRIANGULAR_SELECTION_SCHEMA
        else METADATA_SCHEMA
        if selected["schema"] == METADATA_SELECTION_SCHEMA
        else INTEGER_SCHEMA
        if version == 2
        else SCHEMA,
        "destination": str(destination),
        "source_record_sha256": _digest(source_record),
        "operator_schema_intake_sha256": owner.schema_intake.sha256,
        "semantic_basis_sha256": owner.basis.source.sha256,
        "selection": _pin(owner.selection),
        "source_pins": pins,
        "request": _pin(request_path),
        "native_observation": _pin(observation_path),
        "invocation": _pin(invocation),
        "members": results,
        "totals": totals,
        "unavailable_requirements": list(_UNKNOWN),
        "scope": "complete original scalar source construction observations only; no numerical or mandatory admission",
    }


def prepare(*, schema_intake, basis, source_record, numerical_semantics, selection, destination):
    schema = _replay_originals(schema_intake, basis, source_record, numerical_semantics)
    selection = _plain(selection)
    selected = validate_selection(
        loads(selection.read_bytes()), source_record=source_record, schema_intake=schema_intake, basis=basis
    )
    capture = P.capture_sources(selected)
    members, _ = required_members(source_record, basis=basis, budget=selected["budget"])
    version = (
        2
        if selected["schema"] in {INTEGER_SELECTION_SCHEMA, METADATA_SELECTION_SCHEMA, TRIANGULAR_SELECTION_SCHEMA}
        else 1
    )
    getter = schema.get("tensor_argument_getter") if version == 2 else None
    request = _request(members, capture, selected["budget"], version=version, getter=getter)
    destination = Path(destination).absolute()
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("scalar conversion needs a fresh ordinary destination")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    request_path = destination / "request.json"
    request_path.write_bytes(canonical_json(request))
    products = destination / "products"
    products.mkdir(mode=0o700)
    observer = module_source_path(_READERS[1])
    paths = [products / "observation.json"]
    for member in request["members"]:
        paths.extend(_products(products, member["index"]).values())
    I.run(
        [
            _selection(Path(schema["selection_path"]).read_bytes())["python"],
            "-I",
            "-B",
            str(observer),
            str(request_path),
            selected["capture_checkout"],
            str(products),
        ],
        directory=destination / "native",
        stage="original_registered_scalar_conversion",
        cwd=destination,
        env=D.ENVIRONMENT,
        inputs=(observer, request_path, *(Path(row["source"]) for row in request["members"])),
        outputs=paths,
        dependencies=tuple(
            Path(pin["path"])
            for pin in _pins(
                selection,
                capture,
                version=version,
                getter=getter,
                source_version={METADATA_SELECTION_SCHEMA: 8, TRIANGULAR_SELECTION_SCHEMA: 9}.get(selected["schema"]),
            )
        ),
        capture_output=True,
        timeout=selected["budget"]["timeout_s"],
    ).check_returncode()
    native = loads((products / "observation.json").read_bytes())
    for row in native["rows"]:
        if row["status"] != "converted":
            continue
        owner = destination / "parsers" / str(row["index"])
        owner.mkdir(parents=True, mode=0o700)
        source = _products(products, row["index"])["source"]
        I.run(
            [selected["mlir_opt"], str(source), "-o", str(owner / "verified.mlir")],
            directory=owner,
            stage="original_registered_scalar_stock_parser",
            cwd=destination,
            env=D.ENVIRONMENT,
            inputs=(source,),
            outputs=(owner / "verified.mlir",),
            capture_output=True,
            timeout=selected["budget"]["timeout_s"],
        ).check_returncode()
    prototype = OriginalScalarConversion(
        schema_intake, basis, canonical_json(source_record), canonical_json(numerical_semantics), selection, b""
    )
    record = _record(prototype, destination)
    issued = OriginalScalarConversion(
        schema_intake, basis, prototype.source_json, prototype.numerical_json, selection, canonical_json(record)
    )
    _ISSUED[issued] = issued.sha256
    (destination / "receipt.json").write_bytes(issued.receipt_json)
    return issued
