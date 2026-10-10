"""Fixed complete replay of private original-reference products and decisions."""

import hashlib
import math
import operator
from pathlib import Path

from merlin.common.jsonio import canonical_json
from merlin.common.strict_json import loads
from merlin.targetgen.original_reference_values import TypedReferenceTensor, format_record

from . import original_reference_plan as P
from .operator_schema_intake import _selection

SCHEMA = "merlin.original_reference_roster.v1"


def _pointwise_equal(expected, actual):
    """Compare finite JSON without merging bool/int/float or signed-zero fields."""
    return canonical_json(expected) == canonical_json(actual)


def _decode_outputs(rows):
    return tuple(
        TypedReferenceTensor(
            row["name"], row["dtype"], tuple(row["shape"]), bytes.fromhex(row["data_hex"]), row["byteorder"]
        )
        for row in rows
    )


def native_outputs(contract, path):
    """Check the complete native frame and exact payload lengths before decoding.

    The explicit reference source budget bounds JSON metadata; the explicit
    payload budget bounds its hexadecimal representation. This bounds transfer
    content, not parser heap or framework workspace.
    """
    expected = contract.verify()["outputs"]
    if path.stat().st_size > contract.budget.max_source_bytes + 2 * contract.budget.max_payload_bytes:
        raise ValueError("original native output exceeds explicit predecode transfer limits")
    observed = loads(path.read_bytes())
    transpose = contract.policy.operation == "aten.transpose.int"
    keys = {"schema", "outputs", "runtime"} | ({"alias_observation"} if transpose else set())
    if (
        not isinstance(observed, dict)
        or set(observed) != keys
        or observed["schema"]
        != ("merlin.original_reference_native_output.v2" if transpose else "merlin.original_reference_native_output.v1")
        or not isinstance(observed["runtime"], dict)
        or set(observed["runtime"]) != {"torch_version", "git_version"}
        or any(type(value) is not str for value in observed["runtime"].values())
        or not isinstance(observed["outputs"], list)
        or len(observed["outputs"]) != len(expected)
    ):
        raise ValueError("original native reference output has an unsupported complete schema")
    for row, slot in zip(observed["outputs"], expected, strict=True):
        size = math.prod(slot["shape"]) * (format_record(slot["dtype"])["element_bits"] // 8)
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "dtype", "shape", "byteorder", "data_hex"}
            or not isinstance(row["shape"], list)
            or any(type(extent) is not int or extent <= 0 for extent in row["shape"])
            or {key: row[key] for key in ("name", "dtype", "shape")}
            != {key: slot[key] for key in ("name", "dtype", "shape")}
            or row["byteorder"] != contract.output_byteorder
            or type(row["data_hex"]) is not str
            or len(row["data_hex"]) != 2 * size
            or any(character not in "0123456789abcdef" for character in row["data_hex"])
        ):
            raise ValueError("original native output lost exact complete storage before decoding")
    return _decode_outputs(observed["outputs"])


def finite_transpose_alias(contract, inputs, path):
    """Reopen a distinct finite storage observation; equality grants no alias fact."""
    metadata = contract.verify()
    from merlin.targetgen.original_operator_reference import _roster
    from merlin.targetgen.original_transpose_reference import OriginalTransposeReferencePolicy

    if type(contract.policy) is not OriginalTransposeReferencePolicy:
        raise ValueError("finite transpose alias observation needs its exact original reference contract")
    _roster(inputs, metadata["inputs"])
    # Bounds/schema/complete output storage are independently checked first.
    native_outputs(contract, path)
    observed = loads(path.read_bytes())["alias_observation"]
    shape = metadata["inputs"][0]["shape"]
    strides = [shape[1], 1]
    expected = {
        "schema": "merlin.finite_transpose_alias_observation.v1",
        "input_index": 0,
        "output_index": 0,
        "input_strides": strides,
        "output_strides": [strides[axis] for axis in metadata["permutation"]],
        "input_storage_offset": 0,
        "output_storage_offset": 0,
        "input_storage_bytes": len(inputs[0].data),
        "output_storage_bytes": len(inputs[0].data),
        "same_storage_base": True,
        "input_bytes_sha256_before": hashlib.sha256(inputs[0].data).hexdigest(),
        "input_bytes_sha256_after": hashlib.sha256(inputs[0].data).hexdigest(),
        "scope": "actual finite native storage/stride contact only; no effect-domain or physical authority",
    }
    if canonical_json(observed) != canonical_json(expected):
        raise ValueError("finite original transpose alias/storage/stride observation differs")
    return {"contract_sha256": contract.sha256, "source_schema_alias": metadata["schema_alias"], **observed}


def verify(record, *, schema_intake, basis, selection):
    from . import original_reference_roster as R

    schema = R._originals(schema_intake, basis)
    selected = P.validate(loads(R._plain(selection).read_bytes()))
    equal = _pointwise_equal if selected["schema"] in {P.POINTWISE_SCHEMA, P.TRANSPOSE_SCHEMA} else operator.eq
    if (
        set(record)
        != {
            "schema",
            "operator_schema_intake_sha256",
            "software_intake_sha256",
            "semantic_basis_sha256",
            "selection",
            "destination",
            "defaults",
            "members",
            "totals",
            "source_pins",
            "scope",
        }
        or record.get("schema") != R.record_schema(selected)
        or record["selection"] != R._pin(selection)
        or record["operator_schema_intake_sha256"] != schema_intake.sha256
        or selected["operator_schema_intake_sha256"] != schema_intake.sha256
        or selected["semantic_basis_sha256"] != basis.source.sha256
        or record["semantic_basis_sha256"] != basis.source.sha256
        or record["software_intake_sha256"] != schema_intake.software.sha256
        or record["scope"] != R._SCOPE
        or record["source_pins"] != [R._pin(path) for path in R._sources(selection)]
    ):
        raise ValueError("original reference roster changed its exact original selection/origin")
    destination = Path(record["destination"])
    if (
        not destination.is_absolute()
        or not destination.is_dir()
        or any(path.is_symlink() for path in (destination, *destination.parents))
    ):
        raise ValueError("original reference roster lost its ordinary private owner")
    R.D.verify_members(
        record["defaults"],
        schema_record=schema,
        basis=basis,
        destination=destination / "defaults",
        version=2,
        transport=P.transport(selected),
    )
    rows, contracts, totals = R._drafts(record["defaults"], schema=schema, basis=basis, selection=selected)
    if len(rows) != len(record["members"]) or not equal(totals, record["totals"]):
        raise ValueError("original references lost required original slots or complete preallocation decisions")
    python = _selection(Path(schema["selection_path"]).read_bytes())["python"]
    observer = R._observer(selected)
    for index, (expected, actual) in enumerate(zip(rows, record["members"], strict=True)):
        if index not in contracts:
            if not equal(actual, expected):
                raise ValueError("original unavailable reference obligation was changed or dropped")
            continue
        contract = contracts[index]
        owner = destination / str(index)
        paths = R._paths(owner)
        products = actual["products"]
        if products != R._products(paths) or not {"source", "metadata", "inputs"}.issubset(products):
            raise ValueError("original reference complete private product/owner roster changed")
        if (
            paths["source"].read_text() != contract.source.loader
            or paths["metadata"].read_text() != contract.source.metadata_json
        ):
            raise ValueError("original reference source differs from original exact typed factory")
        inputs = R._stimulus(contract, selected)
        if not equal(loads(paths["inputs"].read_bytes()), [R._tensor_record(tensor) for tensor in inputs]):
            raise ValueError("original reference changed complete original typed input stress")
        try:
            reference = contract.evaluate(inputs)
        except ValueError as error:
            if set(products) != {"source", "metadata", "inputs"}:
                raise ValueError("unavailable reference retained unexpected downstream products") from error
            expected.update(state="reference_unavailable", reason=str(error), products=products)
        else:
            if not equal(loads(paths["reference"].read_bytes()), [R._tensor_record(tensor) for tensor in reference]):
                raise ValueError("original reference answers differ from complete independent replay")
            argv = [
                python,
                "-I",
                str(observer),
                *(str(paths[key]) for key in ("source", "metadata", "inputs", "actual")),
            ]
            invocation = actual["invocation"]
            if invocation != R._pin(invocation["path"]) or Path(invocation["path"]).parent.parent.parent != owner:
                raise ValueError("original native reference invocation changed its exact private owner")
            native = R._native(
                invocation["path"], argv=argv, inputs=(observer, paths["source"], paths["metadata"], paths["inputs"])
            )
            if [pin["path"] for pin in native["outputs"]] != [str(paths["actual"])]:
                raise ValueError("original reference native process lost exact output-file membership")
            expected["invocation"] = invocation
            if native["returncode"]:
                expected.update(state="native_failed", reason="fixed original native source process failed")
            else:
                try:
                    comparison = R._comparison(contract, inputs, paths["actual"])
                except (ValueError, KeyError, TypeError) as error:
                    if "comparison" in products:
                        raise ValueError("unavailable comparison retained an unexpected verdict product") from error
                    expected.update(state="comparison_unavailable", reason=str(error))
                else:
                    if "comparison" not in products or not equal(loads(paths["comparison"].read_bytes()), comparison):
                        raise ValueError("original complete comparison differs from independent replay")
                    expected.update(
                        state="reference_checked" if comparison["passed"] else "comparison_refuted",
                        reason="complete bounded native/source-reference comparison; domain remains unqualified",
                    )
            expected["products"] = products
        if not equal(expected, actual):
            raise ValueError("original reference observation/status/required premises differ from actual replay")
    return record
