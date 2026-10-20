"""Finite exact byte permutation for original rank-two transpose sources.

No floating or integer arithmetic, tolerance, aliasing or tensor library is used
to compute answers. Original format, each element's bits and byte order remain
explicit. Finite value/reference observations do not establish physical effects.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from merlin.common.jsonio import canonical_json
from merlin.common.strict_json import loads

from .original_operator_sources import OriginalOperatorSource
from .original_reference_values import TypedReferenceTensor, format_record
from .original_transpose_sources import TARGET, transpose_source

POLICY_SCHEMA = "merlin.original_transpose_reference_policy.v1"


@dataclass(frozen=True)
class OriginalTransposeReferencePolicy:
    operation: str
    operand_dtypes: tuple[str, ...]
    readout_dtypes: tuple[str, ...]
    movement: str
    finite_only: bool
    comparison: str

    # Logical work accounting uses storage width; this is no accumulator or
    # arithmetic declaration. Existing numerical policies remain unchanged.
    @property
    def accumulator_dtype(self):
        return self.operand_dtypes[0]

    @property
    def arithmetic(self):
        return "storage_permutation"

    def verify(self):
        if (
            type(self.operation) is not str
            or self.operation != TARGET
            or type(self.operand_dtypes) is not tuple
            or len(self.operand_dtypes) != 1
            or type(self.readout_dtypes) is not tuple
            or self.readout_dtypes != self.operand_dtypes
            or type(self.operand_dtypes[0]) is not str
            or self.operand_dtypes[0] not in {"float32", "int8", "int16", "int32", "int64"}
            or self.movement != "rank_two_axis_swap"
            or self.finite_only is not True
            or self.comparison != "exact_element_storage_bits"
        ):
            raise ValueError("transpose reference needs explicit original storage and finite exact permutation policy")
        format_record(self.operand_dtypes[0])

    def record(self):
        self.verify()
        return {
            "schema": POLICY_SCHEMA,
            **vars(self),
            "operand_dtypes": list(self.operand_dtypes),
            "readout_dtypes": list(self.readout_dtypes),
        }


@dataclass(frozen=True)
class OriginalTransposeReferenceContract:
    form_json: bytes
    source: OriginalOperatorSource
    extent: int
    policy: OriginalTransposeReferencePolicy
    budget: object
    output_byteorder: str
    formats_json: bytes

    def verify(self):
        from .original_operator_reference import OriginalReferenceBudget

        if (
            type(self.policy) is not OriginalTransposeReferencePolicy
            or type(self.budget) is not OriginalReferenceBudget
        ):
            raise ValueError("transpose reference needs its exact explicit policy/budget types")
        self.policy.verify()
        self.budget.verify()
        if type(self.source) is not OriginalOperatorSource or self.output_byteorder not in {"little", "big"}:
            raise ValueError("transpose reference needs an exact source and explicit output byte order")
        if (
            type(self.form_json) is not bytes
            or len(self.form_json) > self.budget.max_source_bytes
            or len(self.source.loader.encode()) + len(self.source.metadata_json.encode()) > self.budget.max_source_bytes
        ):
            raise ValueError("transpose reference exceeds source budget before parsing")
        form = loads(self.form_json)
        if (
            form["target"] != self.policy.operation
            or form["operand_dtypes"] != list(self.policy.operand_dtypes)
            or form["result_dtypes"] != list(self.policy.readout_dtypes)
            or canonical_json(form["source_numerical_semantics"]) != canonical_json(self.policy.record())
        ):
            raise ValueError("transpose source and explicit original storage policy differ")
        expected = transpose_source(form, extent=self.extent, max_tensor_elements=self.budget.max_tensor_elements)
        if self.source != expected:
            raise ValueError("transpose reference source differs from its rederived original factory")
        metadata = expected.metadata()
        formats = canonical_json({dtype: format_record(dtype) for dtype in self.policy.operand_dtypes})
        if formats != self.formats_json:
            raise ValueError("transpose reference selected storage descriptors changed")
        if (
            metadata["logical_payload_bytes"] > self.budget.max_payload_bytes
            or math.prod(metadata["outputs"][0]["shape"]) > self.budget.max_arithmetic_steps
        ):
            raise ValueError("transpose reference exceeds explicit logical storage/work budget before allocation")
        return metadata

    @property
    def sha256(self):
        return hashlib.sha256(
            canonical_json(
                {
                    "form": hashlib.sha256(self.form_json).hexdigest(),
                    "loader": hashlib.sha256(self.source.loader.encode()).hexdigest(),
                    "metadata": hashlib.sha256(self.source.metadata_json.encode()).hexdigest(),
                    "policy": self.policy.record(),
                    "extent": self.extent,
                    "budget": vars(self.budget),
                    "formats": loads(self.formats_json),
                    "output_byteorder": self.output_byteorder,
                }
            )
        ).hexdigest()

    def evaluate(self, inputs):
        from .original_operator_reference import _roster

        metadata = self.verify()
        _roster(inputs, metadata["inputs"])
        original = inputs[0]
        original.values()  # Explicit finite domain; no decoded value computes an answer.
        size = format_record(original.dtype)["element_bits"] // 8
        columns = original.shape[1]
        permutation = metadata["permutation"]
        chunks = []
        for row in range(metadata["outputs"][0]["shape"][0]):
            for column in range(metadata["outputs"][0]["shape"][1]):
                offset = row * columns + column if permutation == [0, 1] else column * columns + row
                chunk = original.data[offset * size : (offset + 1) * size]
                chunks.append(chunk if original.byteorder == self.output_byteorder else chunk[::-1])
        output = metadata["outputs"][0]
        return (
            TypedReferenceTensor(
                output["name"], output["dtype"], tuple(output["shape"]), b"".join(chunks), self.output_byteorder
            ),
        )

    def compare(self, inputs, actual):
        from .original_operator_reference import _roster

        metadata = self.verify()
        _roster(actual, metadata["outputs"])
        expected = self.evaluate(inputs)
        size = format_record(self.policy.readout_dtypes[0])["element_bits"] // 8
        mismatches, checked = [], 0
        for wanted, observed in zip(expected, actual, strict=True):
            observed.values()
            for offset in range(0, len(wanted.data), size):
                want = wanted.data[offset : offset + size]
                found = observed.data[offset : offset + size]
                if observed.byteorder != wanted.byteorder:
                    found = found[::-1]
                checked += 1
                if want != found:
                    mismatches.append(
                        {
                            "slot": wanted.name,
                            "index": offset // size,
                            "expected_hex": want.hex(),
                            "actual_hex": found.hex(),
                        }
                    )
        return {
            "schema": "merlin.original_transpose_reference_comparison.v1",
            "contract_sha256": self.sha256,
            "input_sha256": [hashlib.sha256(row.data).hexdigest() for row in inputs],
            "reference_sha256": [hashlib.sha256(row.data).hexdigest() for row in expected],
            "actual_sha256": [hashlib.sha256(row.data).hexdigest() for row in actual],
            "checked_elements": checked,
            "output_roster": metadata["outputs"],
            "passed": not mismatches,
            "mismatches": mismatches,
            "scope": "finite original byte-permutation values only; alias/effect/target/phase authority unproved",
        }


def prepare_transpose_reference(form, source, *, extent, policy, budget, output_byteorder):
    if type(policy) is not OriginalTransposeReferencePolicy:
        raise ValueError("transpose reference requires its explicitly selected storage policy")
    policy.verify()
    result = OriginalTransposeReferenceContract(
        canonical_json(form),
        source,
        extent,
        policy,
        budget,
        output_byteorder,
        canonical_json({dtype: format_record(dtype) for dtype in policy.operand_dtypes}),
    )
    result.verify()
    return result
