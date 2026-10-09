"""Private exact original references joined to ordinary upstream standard IR.

The full original roster persists. A successful row checks source production,
ordered typed ABI and complete native reference values, not compiled semantics,
stress-domain coverage, hardware support or any Phase1 release authority.
"""

from __future__ import annotations

import hashlib
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.execution_deadline import ExecutionDeadline
from merlin.common.paths import module_source_path
from merlin.common.strict_json import loads

from . import original_reference_roster as R
from . import original_standard_ir_plan as P
from .operator_schema_intake import _selection
from .rtl_intake import RtlIntakePin

SCHEMA = "merlin.original_reference_standard_ir.v1"
POINTWISE_SCHEMA = "merlin.original_reference_standard_ir.v2"
_SCOPE = (
    "complete original source/reference/standard-IR ABI observations only; no compiler, runtime or release authority"
)
_UNKNOWN = (
    "upstream_compiled_semantics",
    "mandatory_numerical_stress_coverage",
    "source_effect_test_contract",
    "resource_axis_tail_mapping",
    "physical_effects",
    "target_support",
    "phase1_candidate_execution",
    "native_runtime_dependency_closure",
)
_READERS = (
    __name__,
    P.__name__,
    "merlin_experiments.phase0.original_standard_ir_products",
    "merlin_experiments.phase0.original_standard_ir_observer",
)
_ISSUED = weakref.WeakKeyDictionary()


def schemas(references):
    if references.record_without_verification()["schema"] == R.POINTWISE_SCHEMA:
        return POINTWISE_SCHEMA, "merlin.original_standard_ir_request.v2", "merlin.native_original_standard_ir.v2"
    return SCHEMA, "merlin.original_standard_ir_request.v1", "merlin.native_original_standard_ir.v1"


def _paths(owner):
    return {
        key: owner / name
        for key, name in (
            ("source", "source.mlir"),
            ("trace", "trace.json"),
            ("actual", "actual.json"),
            ("verified", "verified.mlir"),
        )
    }


def ordered_abi(source, metadata, budget):
    """Parse actual standard source and check every ordered argument/return.

    Dense storage is counted before splat expansion. This is a bounded parser
    observation, never a proof of output stores or compiled numerical effects.
    """
    import math

    from xdsl.dialects import affine, arith, linalg, scf, tensor
    from xdsl.dialects import math as math_dialect
    from xdsl.dialects.builtin import NoneAttr, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.parser import Parser

    from merlin.common.quant_formats import machine_bits
    from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source
    from merlin.targetgen.contract.tensor_types import match_tensor_spec
    from merlin.xdsl_dialects._common import make_context

    if R._plain(source).stat().st_size > budget["max_source_bytes"]:
        raise ValueError("standard IR exceeds its explicit source byte budget")
    text = Path(source).read_text()
    admit_mlir_source(
        text,
        max_source_bytes=budget["max_source_bytes"],
        max_nesting=budget["max_nesting"],
        max_integer_bits=budget["max_integer_bits"],
        allow_dense=True,
        allow_dense_resource=False,
    )

    class BoundedParser(Parser):
        elements = 0
        payload_bytes = 0

        def _parse_dense_literal_type(self):
            value_type = super()._parse_dense_literal_type()
            bits = machine_bits(str(value_type.get_element_type()))
            count = math.prod(value_type.get_shape())
            if bits is None or count < 0:
                raise ValueError("standard IR dense literal has unsupported storage")
            self.elements += count
            self.payload_bytes += count * ((bits + 7) // 8)
            if self.elements > budget["max_dense_elements"] or self.payload_bytes > budget["max_dense_payload_bytes"]:
                raise ValueError("standard IR dense storage exceeds its explicit preallocation budget")
            return value_type

    module = BoundedParser(
        make_context(affine.Affine, arith.Arith, linalg.Linalg, math_dialect.Math, scf.Scf, tensor.Tensor), text
    ).parse_module()
    module.verify()
    functions = tuple(module.body.block.ops)
    if len(functions) != 1 or type(functions[0]) is not FuncOp:
        raise ValueError("standard IR has no unique original function")
    entry = functions[0]
    if len(entry.body.blocks) != 1 or type(entry.body.block.last_op) is not ReturnOp:
        raise ValueError("standard IR lacks the supported explicit ordered return")
    inputs, outputs = tuple(entry.function_type.inputs), tuple(entry.function_type.outputs)
    returned = entry.body.block.last_op.arguments
    if (
        tuple(argument.type for argument in entry.body.block.args) != inputs
        or tuple(value.type for value in returned) != outputs
    ):
        raise ValueError("standard IR function body changes its complete original argument/return types")
    result = {"entry_symbol": entry.sym_name.data, "inputs": [], "outputs": []}
    for role, types in (("inputs", inputs), ("outputs", outputs)):
        originals = metadata[role]
        if len(types) != len(originals):
            raise ValueError("standard IR omits an original ordered input/return slot")
        for slot, value_type in zip(originals, types, strict=True):
            if not isinstance(value_type, TensorType) or type(value_type.encoding) is not NoneAttr:
                raise ValueError("standard IR original slot has unsupported storage/encoding")
            observed = {"shape": list(value_type.get_shape()), "dtype": str(value_type.get_element_type())}
            match_tensor_spec(slot, observed)
            result[role].append({"name": slot["name"], **observed})
    return result


def _pin_sources(selection, capture):
    modules = [
        *_READERS,
        "merlin.targetgen.frontend_trace",
        "merlin.common.jsonio",
        "merlin.targetgen.contract.mlir_source_admission",
        "merlin.targetgen.contract.tensor_types",
        "merlin.xdsl_dialects._common",
        "merlin.xdsl_dialects.fp8",
    ]
    paths = {module_source_path(name) for name in modules}
    paths.update(module_source_path("xdsl").parent.rglob("*.py"))
    selected = loads(Path(selection).read_bytes())
    return [R._pin(path) for path in sorted(paths)] + [R._pin(selection), R._pin(selected["mlir_opt"]), *capture]


def _native(references, selected, request, destination, source_pins, deadline):
    schema = references.schema_intake.record()
    python = _selection(Path(schema["selection_path"]).read_bytes())["python"]
    observer = module_source_path(_READERS[-1])
    reference_observer = R._observer(loads(references.selection.read_bytes()))
    argv = [
        python,
        "-I",
        "-B",
        str(observer),
        str(request),
        selected["capture_checkout"],
        str(reference_observer),
        str(destination),
    ]
    outputs = [destination / "observation.json"]
    for member in loads(request.read_bytes())["members"]:
        outputs.extend(path for key, path in _paths(destination / str(member["index"])).items() if key != "verified")
    dependencies = [Path(pin["path"]) for pin in source_pins] + [reference_observer]
    inputs = [observer, request, reference_observer]
    for member in loads(request.read_bytes())["members"]:
        inputs.extend(Path(member[key]) for key in ("source", "metadata", "inputs"))
    result = I.run(
        argv,
        directory=destination / "native",
        stage="original_source_upstream_standard_ir",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        dependencies=tuple(dependencies),
        env=R.D.ENVIRONMENT,
        capture_output=True,
        timeout=deadline.remaining(),
    )
    invocation = next((destination / "native/invocations").glob("*/invocation.json"))
    return R._pin(invocation), result.returncode


@dataclass(frozen=True, eq=False)
class OriginalReferenceStandardIr:
    references: R.OriginalReferenceRoster
    selection: Path
    source_pins: tuple[RtlIntakePin, ...]
    receipt_json: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt_json).hexdigest()

    def verify(self):
        if _ISSUED.get(self) != self.sha256:
            raise ValueError("standard IR requires its actual live original source preparation")
        verify(loads(self.receipt_json), references=self.references, selection=self.selection)

    def record(self):
        self.verify()
        return loads(self.receipt_json)


def prepare(*, references, selection, destination):
    if type(references) is not R.OriginalReferenceRoster:
        raise ValueError("standard IR requires the exact live original reference owner")
    record = P.required_members(references)
    selected = P.validate(loads(R._plain(selection).read_bytes()), references)
    capture = P.capture_sources(selected)
    source_pins = _pin_sources(selection, capture)
    request_members, decisions, totals = P.preflight(record, selected)
    deadline = ExecutionDeadline.start(selected["budget"]["timeout_s"])
    destination = Path(destination).absolute()
    if destination.exists() or any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("standard IR needs a fresh ordinary private destination")
    destination.mkdir(parents=True, mode=0o700)
    request = destination / "request.json"
    R._write(
        request,
        {
            "schema": schemas(references)[1],
            "capture_sources": capture,
            "members": request_members,
            "budget": selected["budget"],
        },
    )
    invocation, returncode = (
        _native(references, selected, request, destination, source_pins, deadline) if request_members else (None, None)
    )
    rows = []
    for original, decision in zip(record["members"], decisions, strict=True):
        row = {
            "original": {
                key: original[key] for key in ("original_member_id", "graph_path", "node", "target", "cohort", "extent")
            },
            "reference_member_sha256": hashlib.sha256(R._json(original)).hexdigest(),
            "decision": decision,
            "state": "unavailable",
            "required_unknowns": [*original["required_unknowns"], *_UNKNOWN],
        }
        if decision["state"] == "planned":
            row.update(
                _evaluate(
                    original,
                    selected,
                    destination / str(decision["index"]),
                    returncode,
                    references=references,
                    deadline=deadline,
                )
            )
        rows.append(row)
    document = {
        "schema": schemas(references)[0],
        "scope": _SCOPE,
        "reference_roster_sha256": references.sha256,
        "selection": R._pin(selection),
        "destination": str(destination),
        "source_pins": source_pins,
        "request": R._pin(request),
        "invocation": invocation,
        "totals": totals,
        "members": rows,
    }
    if invocation is not None:
        document["observation"] = (
            R._pin(destination / "observation.json") if (destination / "observation.json").is_file() else None
        )
    R._write(destination / "roster.json", document)
    for pin in source_pins:
        if R._pin(pin["path"]) != pin:
            raise ValueError("standard IR source selection changed during observation")
    references.verify()
    owner = OriginalReferenceStandardIr(
        references,
        Path(selection).absolute(),
        tuple(RtlIntakePin("private-original-standard-ir", pin["path"], pin["sha256"]) for pin in source_pins),
        R._json(document),
    )
    _ISSUED[owner] = owner.sha256
    owner.verify()
    return owner


def _evaluate(original, selected, owner, returncode, *, references, run_parse=True, deadline=None):
    from xdsl.utils.exceptions import ParseError, VerifyException

    if deadline is not None:
        deadline.remaining()
    paths = _paths(owner)
    products = {key: R._pin(path) for key, path in paths.items() if path.is_file()}
    if returncode != 0:
        return {
            "state": "native_failed",
            "reason": "ordinary upstream native conversion process failed",
            "products": products,
        }
    try:
        if not {"source", "trace", "actual"} <= set(products):
            raise ValueError("ordinary upstream conversion omitted complete products")
        budget = selected["budget"]
        if any(
            path.stat().st_size > budget["max_source_bytes"]
            for key, path in paths.items()
            if key != "verified" and path.is_file()
        ):
            raise ValueError("ordinary upstream conversion exceeded explicit product byte limits")
        metadata = loads(Path(original["products"]["metadata"]["path"]).read_bytes())
        abi = ordered_abi(paths["source"], metadata, budget)
        trace = loads(paths["trace"].read_bytes())
        from merlin.targetgen.frontend_trace import original_operation_semantics

        original_operation_semantics(trace)
        if (
            trace["mlir"]["sha256"] != products["source"]["sha256"]
            or trace["mlir"]["bytes"] != paths["source"].stat().st_size
        ):
            raise ValueError("upstream trace does not join the actual complete standard IR")
        contract = _contract(original, references)
        inputs = R._tensors(loads(Path(original["products"]["inputs"]["path"]).read_bytes()))
        comparison = R._comparison(contract, inputs, paths["actual"])
        if not comparison["passed"]:
            return {
                "state": "comparison_refuted",
                "reason": "complete upstream native values differ from independent reference",
                "comparison": comparison,
                "ordered_abi": abi,
                "products": products,
            }
        if run_parse:
            _stock_verify(paths, selected, owner, deadline)
        from .original_standard_ir_products import parse_invocation

        parse, parse_returncode = parse_invocation(owner, paths, selected)
        if paths["verified"].is_file():
            if paths["verified"].stat().st_size > budget["max_source_bytes"]:
                raise ValueError("stock parser output exceeds its explicit product byte budget")
            products["verified"] = R._pin(paths["verified"])
        if parse_returncode:
            return {
                "state": "unavailable",
                "reason": "actual stock parser refused original standard IR",
                "products": products,
                "parse_invocation": parse,
            }
        return {
            "state": "source_reference_ir_checked",
            "reason": "actual upstream source, complete ordered ABI and independent full native comparison",
            "comparison": comparison,
            "ordered_abi": abi,
            "products": products,
            "parse_invocation": parse,
        }
    except (ValueError, KeyError, TypeError, OSError, ParseError, VerifyException, RecursionError) as error:
        return {"state": "unavailable", "reason": str(error), "products": products}


def _contract(original, references):
    from merlin.targetgen.original_operator_reference import OriginalReferenceBudget, prepare_original_reference
    from merlin.targetgen.original_operator_sources import OriginalOperatorSource

    from . import original_reference_plan

    source = OriginalOperatorSource(
        Path(original["products"]["source"]["path"]).read_text(),
        Path(original["products"]["metadata"]["path"]).read_text(),
    )
    selected = original_reference_plan.validate(loads(references.selection.read_bytes()))
    return prepare_original_reference(
        original["form"],
        source,
        extent=original["extent"],
        policy=original_reference_plan.policy(
            original["policy"], pointwise=selected["schema"] == original_reference_plan.POINTWISE_SCHEMA
        ),
        budget=OriginalReferenceBudget(**selected["reference_budget"]),
        output_byteorder=selected["byteorder"],
    )


def _stock_verify(paths, selected, owner, deadline):
    I.run(
        [selected["mlir_opt"], str(paths["source"]), "--verify-each", "-o", str(paths["verified"])],
        directory=owner / "parse",
        stage="original_standard_ir_native_parse",
        inputs=(paths["source"],),
        outputs=(paths["verified"],),
        dependencies=(Path(selected["mlir_opt"]),),
        env=R.D.ENVIRONMENT,
        capture_output=True,
        timeout=deadline.remaining(),
    )
    if deadline is not None:
        deadline.remaining()


def verify(document, *, references, selection):
    from .original_standard_ir_products import verify as replay

    return replay(document, references=references, selection=selection)
