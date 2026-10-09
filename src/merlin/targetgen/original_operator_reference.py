"""Opt-in bounded numerical checks for exact original typed operator sources.

This is an explicitly selected software reference contract, not equivalence to
arbitrary framework reduction algorithms, target support or phase admission.
Original source factories rederive all types, arguments and geometry. Every
input/output element is checked; no loader, compiler or backend supplies answers.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction

from merlin.common.strict_json import loads

from . import original_operator_sources as S
from .original_reference_values import TypedReferenceTensor, format_record, integer_project, round_f32

POLICY_SCHEMA = "merlin.original_operator_reference_policy.v1"
_OPERATIONS = {"aten.matmul.default", "aten.add.Tensor", "aten.conv2d.default"}


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class OriginalReferenceBudget:
    max_tensor_elements: int
    max_payload_bytes: int
    max_arithmetic_steps: int
    max_source_bytes: int

    def verify(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("original reference budgets must be explicit positive integers")


@dataclass(frozen=True)
class OriginalReferencePolicy:
    operation: str
    operand_dtypes: tuple[str, ...]
    readout_dtypes: tuple[str, ...]
    accumulator_dtype: str
    arithmetic: str
    product_rounding: str
    reduction_order: str
    reduction_cadence: str
    rounding: str
    finite_only: bool
    subnormal_operand_flush: bool
    bias_position: str
    atol: float
    rtol: float
    zero_sign: str

    def record(self) -> dict:
        self.verify()
        return {
            "schema": POLICY_SCHEMA,
            **vars(self),
            "operand_dtypes": list(self.operand_dtypes),
            "readout_dtypes": list(self.readout_dtypes),
        }

    def verify(self):
        if self.operation not in _OPERATIONS:
            raise ValueError("original reference has no implementation of this operator")
        if (
            type(self.operand_dtypes) is not tuple
            or not self.operand_dtypes
            or type(self.readout_dtypes) is not tuple
            or len(self.readout_dtypes) != 1
        ):
            raise ValueError("original reference requires ordered complete original dtype rosters")
        formats = [format_record(dtype) for dtype in (*self.operand_dtypes, *self.readout_dtypes)]
        accumulator = format_record(self.accumulator_dtype)
        if any(type(tol) is not float or not math.isfinite(tol) or tol < 0 for tol in (self.atol, self.rtol)):
            raise ValueError("original reference requires explicit finite nonnegative float tolerances")
        if self.finite_only is not True or self.subnormal_operand_flush is not False:
            raise ValueError("original reference implements only finite values without operand flushing")
        if self.zero_sign not in {"preserve", "ignore"} or self.bias_position != "after_reduction":
            raise ValueError("original reference requires an implemented zero-sign and bias order")
        order = {
            "aten.matmul.default": "contracting_axis_sequential",
            "aten.add.Tensor": "elementwise",
            "aten.conv2d.default": "input_channel_kernel_row_kernel_column",
        }[self.operation]
        if self.reduction_order != order or self.reduction_cadence != "per_step":
            raise ValueError("original reference implements only the explicitly selected sequential order")
        if self.arithmetic == "finite_f32":
            if (
                any(fmt["kind"] != "float_ieee" for fmt in formats)
                or accumulator["kind"] != "float_ieee"
                or self.rounding != "rne"
                or self.product_rounding != "accumulator_format"
            ):
                raise ValueError("finite f32 reference requires f32 storage/accumulator and per-product RNE")
        elif self.arithmetic in {"bounded_exact", "modular_wrap"}:
            if (
                any(fmt["kind"] != "int_affine" for fmt in formats)
                or accumulator["kind"] != "int_affine"
                or self.rounding != "exact_integer"
                or self.product_rounding != "accumulator_format"
                or self.atol != 0.0
                or self.rtol != 0.0
                or self.zero_sign != "ignore"
            ):
                raise ValueError("signed integer reference requires exact integer arithmetic and comparison")
        else:
            raise ValueError("original reference arithmetic is unimplemented")


def _source(form, extent, budget):
    factory = {"aten.matmul.default": S.matmul_source, "aten.conv2d.default": S.conv2d_source}.get(form["target"])
    if form["target"] == "aten.add.Tensor":
        factory = getattr(S, "add_source", None)
    from .original_pointwise_reference import OPERATIONS

    if form["target"] in OPERATIONS:
        from .original_pointwise_sources import pointwise_source

        factory = pointwise_source
    if factory is None:
        raise ValueError("original reference lacks an independently supported original source factory")
    return factory(form, extent=extent, max_tensor_elements=budget.max_tensor_elements)


def _costs(metadata):
    from .original_pointwise_reference import OPERATIONS, reference_steps

    inputs, outputs = metadata["inputs"], metadata["outputs"]
    rows = inputs + outputs
    elements = sum(math.prod(row["shape"]) for row in rows)
    payload = sum(math.prod(row["shape"]) * (format_record(row["dtype"])["element_bits"] // 8) for row in rows)
    count = sum(math.prod(row["shape"]) for row in outputs)
    if metadata["target"] == "aten.matmul.default":
        products = count * inputs[0]["shape"][1]
    elif metadata["target"] == "aten.conv2d.default":
        products = count * math.prod(inputs[1]["shape"][1:])
    elif metadata["target"] in OPERATIONS:
        products = 0
    else:
        products = count
    steps = (
        reference_steps(metadata)
        if metadata["target"] in OPERATIONS
        else 2 * products + (count if metadata["parameters"].get("bias") else 0)
    )
    if (
        metadata["tensor_elements"] != elements
        or metadata["logical_payload_bytes"] != payload
        or metadata["scalar_products"] != products
    ):
        raise ValueError("original source costs disagree with its actual typed geometry")
    return elements, payload, steps


@dataclass(frozen=True)
class OriginalReferenceContract:
    form_json: bytes
    source: S.OriginalOperatorSource
    extent: int
    policy: OriginalReferencePolicy
    budget: OriginalReferenceBudget
    output_byteorder: str
    formats_json: bytes

    def verify(self) -> dict:
        self.budget.verify()
        self.policy.verify()
        if type(self.source) is not S.OriginalOperatorSource:
            raise ValueError("original reference requires an exact original source declaration")
        if self.output_byteorder not in {"little", "big"}:
            raise ValueError("original reference requires an explicit output byte order")
        if (
            type(self.form_json) is not bytes
            or len(self.form_json) > self.budget.max_source_bytes
            or len(self.source.loader.encode()) + len(self.source.metadata_json.encode()) > self.budget.max_source_bytes
        ):
            raise ValueError("original reference exceeds its source-byte budget before parsing")
        form = loads(self.form_json)
        if (
            form.get("target") != self.policy.operation
            or form.get("operand_dtypes") != list(self.policy.operand_dtypes)
            or form.get("result_dtypes") != list(self.policy.readout_dtypes)
            or _json(form.get("source_numerical_semantics")) != _json(self.policy.record())
        ):
            raise ValueError("original source and exact per-operation numerical policy are incompatible")
        expected = _source(form, self.extent, self.budget)
        if type(self.source) is not S.OriginalOperatorSource or self.source != expected:
            raise ValueError("original reference source/form binding changed")
        metadata = expected.metadata()
        from .original_pointwise_reference import OriginalPointwiseReferencePolicy, validate_parameters

        if type(self.policy) is OriginalPointwiseReferencePolicy:
            validate_parameters(metadata["parameters"], self.policy)
        current_formats = _json(
            {
                dtype: format_record(dtype)
                for dtype in (*self.policy.operand_dtypes, *self.policy.readout_dtypes, self.policy.accumulator_dtype)
            }
        )
        if current_formats != self.formats_json:
            raise ValueError("original reference selected software datatype descriptor changed")
        elements, payload, steps = _costs(metadata)
        if (
            elements > self.budget.max_tensor_elements
            or payload > self.budget.max_payload_bytes
            or steps > self.budget.max_arithmetic_steps
        ):
            raise ValueError("original reference exceeds explicit evaluation budget before allocation")
        return metadata

    @property
    def sha256(self):
        return _sha(
            _json(
                {
                    "form": _sha(self.form_json),
                    "loader": _sha(self.source.loader.encode()),
                    "metadata": _sha(self.source.metadata_json.encode()),
                    "policy": self.policy.record(),
                    "extent": self.extent,
                    "budget": vars(self.budget),
                    "formats": loads(self.formats_json),
                    "output_byteorder": self.output_byteorder,
                }
            )
        )

    def _inputs(self, inputs):
        metadata = self.verify()
        _roster(inputs, metadata["inputs"])
        return metadata, [tensor.values() for tensor in inputs]

    def evaluate(self, inputs: tuple[TypedReferenceTensor, ...]) -> tuple[TypedReferenceTensor, ...]:
        metadata, values = self._inputs(inputs)
        return _evaluate(metadata, values, _arithmetic(self.policy), self.output_byteorder)

    def observe_stress(self, inputs: tuple[TypedReferenceTensor, ...]) -> dict:
        """Observe realized inputs/products/partial sums in the same reference.

        Counters retain actual rounding, cancellation and wrap events under the
        declared policy. They cannot establish an untested numerical domain,
        framework reduction order, candidate semantics or hardware effects.
        """
        metadata, values = self._inputs(inputs)
        arithmetic = _arithmetic(self.policy, stress=True)
        for row in values:
            for value in row:
                arithmetic.counts["positive_inputs"] += value > 0
                arithmetic.counts["negative_inputs"] += value < 0
                arithmetic.counts["zero_inputs"] += value == 0
        output = _evaluate(metadata, values, arithmetic, self.output_byteorder)
        return {
            "schema": "merlin.original_reference_stress.v2"
            if "pointwise_values" in arithmetic.counts
            else "merlin.original_reference_stress.v1",
            "contract_sha256": self.sha256,
            "input_sha256": [_sha(tensor.data) for tensor in inputs],
            "output_sha256": [_sha(tensor.data) for tensor in output],
            "counts": arithmetic.counts,
            "logical_input_shapes": [row["shape"] for row in metadata["inputs"]],
            "logical_output_shapes": [row["shape"] for row in metadata["outputs"]],
            "scope": (
                "realized bounded reference arithmetic only; no whole-domain, framework, target or phase authority"
            ),
        }

    def compare(self, inputs, actual) -> dict:
        """Recompute and compare every original output slot and element."""
        metadata = self.verify()
        _roster(actual, metadata["outputs"])
        reference = self.evaluate(inputs)
        mismatches, checked = [], 0
        for want, observed in zip(reference, actual, strict=True):
            for index, (a, b) in enumerate(zip(want.values(), observed.values(), strict=True)):
                checked += 1
                if self.policy.arithmetic == "finite_f32":
                    tolerance = Fraction(self.policy.atol) + Fraction(self.policy.rtol) * abs(Fraction(a))
                    matches = abs(Fraction(b) - Fraction(a)) <= tolerance
                    if self.policy.zero_sign == "preserve" and a == b == 0:
                        matches = matches and math.copysign(1, a) == math.copysign(1, b)
                else:
                    matches = a == b
                if not matches:
                    mismatches.append({"slot": want.name, "index": index, "expected": a, "actual": b})
        return {
            "schema": "merlin.original_operator_reference_comparison.v1",
            "contract_sha256": self.sha256,
            "input_sha256": [_sha(tensor.data) for tensor in inputs],
            "reference_sha256": [_sha(tensor.data) for tensor in reference],
            "actual_sha256": [_sha(tensor.data) for tensor in actual],
            "checked_elements": checked,
            "output_roster": metadata["outputs"],
            "passed": not mismatches,
            "mismatches": mismatches,
            "scope": "selected bounded software reference only; framework/target/phase admission unproved",
        }


def _roster(tensors, roster):
    if type(tensors) is not tuple or len(tensors) != len(roster):
        raise ValueError("original reference requires the complete ordered tensor roster")
    for tensor, row in zip(tensors, roster, strict=True):
        if type(tensor) is not TypedReferenceTensor:
            raise ValueError("original reference requires explicitly typed raw tensor storage")
        if (tensor.name, tensor.dtype, tensor.shape) != (row["name"], row["dtype"], tuple(row["shape"])):
            raise ValueError("original reference ordered name/type/shape roster changed")
        tensor.verify()


class _Arithmetic:
    def __init__(self, policy):
        self.policy = policy

    def project(self, value):
        if self.policy.arithmetic == "finite_f32":
            return round_f32(value)
        return integer_project(value, self.policy.accumulator_dtype, self.policy.arithmetic)

    def product(self, a, b):
        return self.project(a * b)

    def add(self, a, b):
        return self.project(a + b)

    def reduce(self, pairs):
        value = 0.0 if self.policy.arithmetic == "finite_f32" else 0
        for a, b in pairs:
            value = self.add(value, self.product(a, b))
        return value

    def readout(self, value, dtype):
        return (
            round_f32(value)
            if self.policy.arithmetic == "finite_f32"
            else integer_project(value, dtype, self.policy.arithmetic)
        )


class _StressArithmetic(_Arithmetic):
    def __init__(self, policy):
        super().__init__(policy)
        self.counts = dict.fromkeys(
            (
                "positive_inputs",
                "negative_inputs",
                "zero_inputs",
                "products",
                "additions",
                "rounded_products",
                "rounded_additions",
                "wrapped_products",
                "wrapped_additions",
                "wrapped_outputs",
                "cancellation_additions",
                "exact_zero_cancellations",
                "zero_outputs",
            ),
            0,
        )

    def _observe(self, kind, exact, observed):
        self.counts[kind] += 1
        if Fraction(observed) != exact:
            prefix = "rounded_" if self.policy.arithmetic == "finite_f32" else "wrapped_"
            self.counts[prefix + kind] += 1
        return observed

    def product(self, a, b):
        return self._observe("products", Fraction(a) * Fraction(b), super().product(a, b))

    def add(self, a, b):
        exact = Fraction(a) + Fraction(b)
        opposite = (a > 0 and b < 0) or (a < 0 and b > 0)
        self.counts["cancellation_additions"] += opposite
        self.counts["exact_zero_cancellations"] += opposite and exact == 0
        return self._observe("additions", exact, super().add(a, b))

    def readout(self, value, dtype):
        observed = super().readout(value, dtype)
        self.counts["wrapped_outputs"] += self.policy.arithmetic != "finite_f32" and observed != value
        self.counts["zero_outputs"] += observed == 0
        return observed


def _arithmetic(policy, *, stress=False):
    if type(policy) is OriginalReferencePolicy:
        return _StressArithmetic(policy) if stress else _Arithmetic(policy)
    from .original_pointwise_reference import pointwise_arithmetic

    return pointwise_arithmetic(policy, stress=stress)


def _evaluate(metadata, values, arithmetic, byteorder):
    from .original_pointwise_reference import OPERATIONS

    if metadata["target"] == "aten.matmul.default":
        m, k = metadata["inputs"][0]["shape"]
        n = metadata["inputs"][1]["shape"][1]
        result = [
            arithmetic.reduce((values[0][i * k + t], values[1][t * n + j]) for t in range(k))
            for i in range(m)
            for j in range(n)
        ]
    elif metadata["target"] == "aten.add.Tensor":
        result = [arithmetic.add(a, arithmetic.product(1, b)) for a, b in zip(*values, strict=True)]
    elif metadata["target"] in OPERATIONS:
        result = [arithmetic.pointwise(metadata["target"], value, metadata["parameters"]) for value in values[0]]
    else:
        result = _conv(metadata, values, arithmetic)
    output = metadata["outputs"][0]
    result = [arithmetic.readout(value, output["dtype"]) for value in result]
    return (
        TypedReferenceTensor.from_values(
            output["name"],
            output["dtype"],
            output["shape"],
            result,
            byteorder=byteorder,
        ),
    )


def _conv(metadata, values, arithmetic):
    batch, channels, height, width = metadata["inputs"][0]["shape"]
    out_channels, local_channels, kh, kw = metadata["inputs"][1]["shape"]
    _, _, oh, ow = metadata["outputs"][0]["shape"]
    parameters = metadata["parameters"]

    def pair(name):
        return parameters[name] * 2 if len(parameters[name]) == 1 else parameters[name]

    sh, sw = pair("stride")
    ph, pw = pair("padding")
    dh, dw = pair("dilation")
    group_channels = out_channels // parameters["groups"]
    result = []
    for b in range(batch):
        for c in range(out_channels):
            group = c // group_channels
            for i in range(oh):
                for j in range(ow):

                    def pairs():
                        for channel in range(local_channels):
                            for r in range(kh):
                                for t in range(kw):
                                    y, x = i * sh - ph + r * dh, j * sw - pw + t * dw
                                    a = (
                                        values[0][
                                            ((b * channels + group * local_channels + channel) * height + y) * width + x
                                        ]
                                        if 0 <= y < height and 0 <= x < width
                                        else 0
                                    )
                                    w = values[1][((c * local_channels + channel) * kh + r) * kw + t]
                                    yield a, w

                    value = arithmetic.reduce(pairs())
                    if parameters["bias"]:
                        value = arithmetic.add(value, values[2][c])
                    result.append(value)
    return result


def prepare_original_reference(form, source, *, extent, policy, budget, output_byteorder):
    """Bind one independently selected source/policy, never mint phase authority."""
    from .original_pointwise_reference import OriginalPointwiseReferencePolicy

    if (
        type(policy) not in {OriginalReferencePolicy, OriginalPointwiseReferencePolicy}
        or type(budget) is not OriginalReferenceBudget
    ):
        raise ValueError("original reference requires explicit typed policy and evaluation budget")
    policy.verify()
    budget.verify()
    form_json = _json(form)
    formats = _json(
        {
            dtype: format_record(dtype)
            for dtype in (*policy.operand_dtypes, *policy.readout_dtypes, policy.accumulator_dtype)
        }
    )
    contract = OriginalReferenceContract(form_json, source, extent, policy, budget, output_byteorder, formats)
    contract.verify()
    return contract
