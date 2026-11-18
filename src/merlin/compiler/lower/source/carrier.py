"""Reify closed scalar observation helpers from their current typed source proof.

The helpers clone the proved arithmetic, including original operation order,
precision, attributes and literals. No expression grammar, workload label or
retained compiled body selects the implementation. Coordinates, ownership,
numeric-mode guards and subsequent helper-call binding remain caller obligations.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import struct
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

from merlin.compiler.lower.source.interval import (
    ClosedScalarObserver,
    IntervalEffectContract,
    build_source_interval_table,
    find_closed_scalar_i8_observers,
    validate_closed_scalar_observer,
)
from merlin.compiler.lower.source.stage import ScalarLeafPatch, operation_paths, value_locator

# ---- source_observation_helpers ----------------------------------------------------------------


def reify_closed_scalar_observer_helpers(
    proof: ClosedScalarObserver,
    *,
    effects: IntervalEffectContract,
    expression_symbol: str,
    observer_symbol: str,
):
    """Return fresh ``f32→f32`` expression and ``f32→i8`` observation functions.

    The observer accepts the original second rounded finishing multiply's result;
    neither finishing multiply is moved, folded or reproduced here. The current
    source witness and all enclosing numeric conditions are checked before and
    after cloning. A stale proof, missing dependency or invalid ABI symbol refuses.
    The source operations and their use lists are unchanged.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp, StringAttr, f32, i8
    from xdsl.ir import Block, Region

    if not isinstance(proof, ClosedScalarObserver):
        raise ValueError("current typed ClosedScalarObserver proof required")
    if not isinstance(effects, IntervalEffectContract):
        raise ValueError("explicit IntervalEffectContract required")
    effects.validate()
    validate_closed_scalar_observer(proof)
    for symbol in (expression_symbol, observer_symbol):
        if not isinstance(symbol, str) or not symbol.isascii() or not symbol.isidentifier():
            raise ValueError("explicit nonempty ASCII helper ABI symbol required")
    if expression_symbol == observer_symbol:
        raise ValueError("distinct expression and observer ABI symbols required")

    def trace(value):
        result = {}
        operation = value.owner
        while hasattr(operation, "parent_op"):
            for key, attribute in operation.attributes.items():
                if key.startswith("prov."):
                    result.setdefault(key, attribute)
            operation = operation.parent_op()
            if operation is None:
                break
        # Preserve enclosing trace data alongside a derived transformation record.
        additions = {
            "prov.reification": StringAttr("closed_scalar_observer_helper"),
            "prov.reified_expression_sha256": StringAttr(proof.expression.canonical_sha256),
        }
        for key, value in additions.items():
            if key in result and result[key] != value:
                raise ValueError("source provenance conflicts with helper reification record")
            result[key] = value
        return result

    def function(symbol, source_input, source_output, operations):
        block = Block(arg_types=[f32])
        mapping = {source_input: block.args[0]}
        for operation in operations:
            if any(operand not in mapping for operand in operation.operands):
                raise ValueError("source helper dependency is not inside its proved scalar DAG")
            block.add_op(operation.clone(mapping))
        if source_output not in mapping:
            raise ValueError("proved scalar helper output is not materialized")
        returned = func.ReturnOp(mapping[source_output])
        block.add_op(returned)
        result = func.FuncOp(symbol, ([f32], [source_output.type]), Region(block))
        result.attributes.update(trace(source_output))
        return result

    expression = function(expression_symbol, proof.cut, proof.endpoint, proof.expression_operations)
    scaled = proof.observer_operations[1].results[0]
    observer = function(observer_symbol, scaled, proof.integer_result, proof.observer_operations[2:])
    if expression.function_type.outputs.data != (f32,) or observer.function_type.outputs.data != (i8,):
        raise ValueError("proved scalar helper ABI changed")
    module = ModuleOp([expression, observer])
    module.verify()
    validate_closed_scalar_observer(proof)
    return module


# ---- source_scalar_carrier_policy --------------------------------------------------------------
# Explicit numerical permission for a source-bound scalar representation.
#
# The budget concerns the binary32 scalar carrier before its unchanged finishing
# operations. It is not a bound on the final integer code or an entire model.
# Those outputs keep their independent original validation gate. No policy is
# selected by default, and permission does not establish profitability.


@dataclass(frozen=True)
class ScalarCarrierErrorBudget:
    atol: float
    rtol: float

    def validate(self):
        if any(type(value) is not float or not math.isfinite(value) or value < 0 for value in (self.atol, self.rtol)):
            raise ValueError("explicit finite nonnegative scalar carrier tolerances required")
        if self.atol == self.rtol == 0:
            raise ValueError("an approximate scalar carrier requires a nonzero explicit budget")


@dataclass(frozen=True)
class ApproximateScalarCarrierPolicy:
    budget: ScalarCarrierErrorBudget
    leading_bits: int
    proof_leading_bits: int
    max_table_bytes: int
    max_proof_table_bytes: int
    approximate_output_permitted: bool
    independent_original_output_validation_required: bool
    original_source_fallback_retained: bool
    target_incoming_rne_predicate_required: bool

    def validate(self):
        if type(self.budget) is not ScalarCarrierErrorBudget:
            raise ValueError("typed scalar carrier budget required")
        self.budget.validate()
        if type(self.leading_bits) is not int or type(self.proof_leading_bits) is not int:
            raise ValueError("explicit integer carrier and proof partition widths required")
        if not 9 <= self.leading_bits <= self.proof_leading_bits <= 23:
            raise ValueError("binary32 carrier/proof widths must satisfy 9 <= carrier <= proof <= 23")
        for amount, required in (
            (self.max_table_bytes, (1 << self.leading_bits) * 12),
            (self.max_proof_table_bytes, (1 << self.proof_leading_bits) * 8),
        ):
            if type(amount) is not int or amount < required:
                raise ValueError("scalar carrier exceeds its explicit immutable storage budget")
        if any(
            value is not True
            for value in (
                self.approximate_output_permitted,
                self.independent_original_output_validation_required,
                self.original_source_fallback_retained,
                self.target_incoming_rne_predicate_required,
            )
        ):
            raise ValueError("complete explicit scalar approximation permission required")

    @property
    def canonical_sha256(self):
        self.validate()
        data = (
            self.budget.atol,
            self.budget.rtol,
            self.leading_bits,
            self.proof_leading_bits,
            self.max_table_bytes,
            self.max_proof_table_bytes,
            self.approximate_output_permitted,
            self.independent_original_output_validation_required,
            self.original_source_fallback_retained,
            self.target_incoming_rne_predicate_required,
        )
        return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


# ---- source_scalar_carrier ---------------------------------------------------------------------
# Prove and reify an explicitly selected compact rounded scalar carrier.
#
# Coefficients are immutable caller choices, not permission inferred from samples.
# Every fine raw-binary32 interval must satisfy the declared carrier error budget.
# The runtime representation is a coarse three-coefficient table and two ordered
# binary32 FMAs. Invalid cells and an unadmitted incoming rounding mode evaluate
# the original current expression. Finishing multiplies and the complete original
# integer observer are cloned unchanged. The target supplies the incoming-RNE
# predicate; this module reads no ISA or runtime floating environment.


def _words_digest(words):
    return hashlib.sha256(json.dumps(words, separators=(",", ":")).encode()).hexdigest()


def _validate_words(words, count):
    if type(words) is not tuple or len(words) != count:
        raise ValueError("complete immutable coefficient word tuple required")
    if any(
        type(cell) is not tuple
        or len(cell) != 3
        or any(type(word) is not int or not 0 <= word < 1 << 32 for word in cell)
        for cell in words
    ):
        raise ValueError("each coefficient cell requires three binary32 words")


@dataclass(frozen=True)
class SourceScalarCarrier:
    expression_sha256: str
    interval_sha256: str
    policy: ApproximateScalarCarrierPolicy
    coefficient_words: tuple[tuple[int, int, int], ...] = field(repr=False)
    admitted_cells: int
    original_valid_cells: int
    _observers: tuple = field(repr=False)
    _effects: IntervalEffectContract = field(repr=False)
    _witness: tuple = field(repr=False)

    @property
    def canonical_sha256(self):
        validate_source_scalar_carrier(self)
        return hashlib.sha256(json.dumps(self._witness, separators=(",", ":")).encode()).hexdigest()


def prepare_source_scalar_carrier(
    observers, *, effects: IntervalEffectContract, policy: ApproximateScalarCarrierPolicy, coefficient_words
):
    """Derive complete current-source admission for a caller's coefficients.

    Proof and physical granularity are independent. Every subcell is enclosed
    under original rounded source arithmetic and the two rounded carrier FMAs.
    The conservative comparison uses the minimum source magnitude, including
    zero for a zero-crossing interval. A rejected coefficient cell remains on
    the original route; no observed input chooses an exception or admission.
    """
    import numpy as np

    if type(policy) is not ApproximateScalarCarrierPolicy or type(effects) is not IntervalEffectContract:
        raise ValueError("explicit scalar carrier policy and floating effects required")
    policy.validate()
    effects.validate()
    observers = tuple(observers)
    if not observers or any(type(proof) is not ClosedScalarObserver for proof in observers):
        raise ValueError("current typed closed scalar observers required")
    for proof in observers:
        validate_closed_scalar_observer(proof)
        if proof.expression != observers[0].expression:
            raise ValueError("all scalar carriers must share exact current source arithmetic")
    count = 1 << policy.leading_bits
    _validate_words(coefficient_words, count)
    table = build_source_interval_table(
        observers[0].expression,
        effects=effects,
        leading_bits=policy.proof_leading_bits,
        max_table_bytes=policy.max_proof_table_bytes,
    )
    parts = 1 << (policy.proof_leading_bits - policy.leading_bits)
    coefficients = np.asarray(coefficient_words, dtype=np.uint32).view(np.float32)
    intervals = np.frombuffer(table.data, dtype="<f4").reshape(count, parts, 2).astype(np.float64)
    c0, c1, c2 = [coefficients[:, lane].astype(np.float64)[:, None] for lane in range(3)]
    low, high = intervals[:, :, 0], intervals[:, :, 1]
    source_finite = np.isfinite(intervals).all(axis=2) & (low <= high)
    width = 1 << (32 - policy.proof_leading_bits)
    tlo = (np.arange(parts, dtype=np.float64) * width)[None, :]
    thi = tlo + float(width - 1)
    with np.errstate(all="ignore"):
        first, last = c2 * tlo + c1, c2 * thi + c1
        hlow = np.nextafter(np.nextafter(np.minimum(first, last), -np.inf).astype(np.float32), np.float32(-np.inf))
        hhigh = np.nextafter(np.nextafter(np.maximum(first, last), np.inf).astype(np.float32), np.float32(np.inf))
        products = [h.astype(np.float64) * t for h in (hlow, hhigh) for t in (tlo, thi)]
        qlow = np.nextafter(
            np.nextafter(np.minimum.reduce(products) + c0, -np.inf).astype(np.float32), np.float32(-np.inf)
        ).astype(np.float64)
        qhigh = np.nextafter(
            np.nextafter(np.maximum.reduce(products) + c0, np.inf).astype(np.float32), np.float32(np.inf)
        ).astype(np.float64)
        error = np.nextafter(np.maximum(np.abs(qhigh - low), np.abs(high - qlow)), np.inf)
        minimum = np.where((low <= 0) & (high >= 0), 0.0, np.minimum(np.abs(low), np.abs(high)))
        tolerance = np.nextafter(policy.budget.atol + policy.budget.rtol * minimum, -np.inf)
    each = (
        source_finite
        & np.isfinite(hlow)
        & np.isfinite(hhigh)
        & np.isfinite(qlow)
        & np.isfinite(qhigh)
        & (error <= tolerance)
    )
    admitted = np.isfinite(coefficients).all(axis=1) & each.all(axis=1)
    # A typed binary32 quiet NaN marks a refused cell. It is not an executable
    # instruction encoding, nor a claim about physical target byte order.
    words = tuple(cell if bool(allowed) else (0x7FC00000, 0, 0) for cell, allowed in zip(coefficient_words, admitted))
    admitted_count, valid_count = int(admitted.sum()), int(source_finite.all(axis=1).sum())
    witness = (
        observers[0].expression.canonical_sha256,
        table.sha256,
        policy.canonical_sha256,
        _words_digest(words),
        admitted_count,
        valid_count,
        tuple(vars(effects).items()),
    )
    result = SourceScalarCarrier(
        witness[0], witness[1], policy, words, admitted_count, valid_count, observers, effects, witness
    )
    validate_source_scalar_carrier(result)
    return result


def validate_source_scalar_carrier(carrier):
    """Refuse stale source/effects, edited coefficients or changed permission."""
    if type(carrier) is not SourceScalarCarrier:
        raise ValueError("typed current-source scalar carrier required")
    carrier.policy.validate()
    carrier._effects.validate()
    _validate_words(carrier.coefficient_words, 1 << carrier.policy.leading_bits)
    if not carrier._observers:
        raise ValueError("current closed scalar observer witness required")
    for proof in carrier._observers:
        validate_closed_scalar_observer(proof)
        if proof.expression.canonical_sha256 != carrier.expression_sha256:
            raise ValueError("scalar carrier source arithmetic changed")
    witness = (
        carrier.expression_sha256,
        carrier.interval_sha256,
        carrier.policy.canonical_sha256,
        _words_digest(carrier.coefficient_words),
        carrier.admitted_cells,
        carrier.original_valid_cells,
        tuple(vars(carrier._effects).items()),
    )
    if witness != carrier._witness:
        raise ValueError("scalar carrier source, budget, effects or immutable coefficient witness changed")


def _reify_source_scalar_carrier_helper_ops(
    carrier: SourceScalarCarrier,
    proof: ClosedScalarObserver,
    *,
    table_symbol: str,
    expression_symbol: str,
    observer_symbol: str,
    carrier_symbol: str,
):
    """Clone detached helper operations for one validated current source proof.

    Arguments are the original cut, original independent finishing operand and
    a target-supplied incoming-RNE admission predicate. The caller must supply a
    true predicate only for the admitted floating mode/effect environment. False
    always evaluates the original expression. The literal second finishing
    multiplier and complete observer are bound from this current source proof.
    No source operation is moved or rewritten; invocation-local binding,
    ownership, original output validation and complete cost remain obligations.
    """
    from xdsl.dialects import arith, func, math, memref, scf
    from xdsl.dialects.builtin import (
        IndexType,
        MemRefType,
        StringAttr,
        f32,
        i1,
        i8,
        i32,
    )
    from xdsl.ir import Block, Region

    validate_source_scalar_carrier(carrier)
    if type(proof) is not ClosedScalarObserver or not any(proof is p for p in carrier._observers):
        raise ValueError("helper requires one of the carrier's current source proofs")
    symbols = (table_symbol, expression_symbol, observer_symbol, carrier_symbol)
    if any(type(name) is not str or not name.isascii() or not name.isidentifier() for name in symbols):
        raise ValueError("explicit nonempty ASCII helper/storage ABI symbols required")
    if len(set(symbols)) != len(symbols):
        raise ValueError("distinct scalar carrier helper/storage symbols required")
    original = reify_closed_scalar_observer_helpers(
        proof, effects=carrier._effects, expression_symbol=expression_symbol, observer_symbol=observer_symbol
    )
    expression, observation = tuple(original.body.block.ops)
    expression.detach()
    observation.detach()
    count = 1 << carrier.policy.leading_bits
    shape = [count, 3]
    table_type = MemRefType(f32, shape)
    block = Block(arg_types=[f32, f32, i1])
    cut, up, incoming_rne = block.args
    raw = arith.BitcastOp(cut, i32)
    shift = arith.ConstantOp.from_int_and_width(32 - carrier.policy.leading_bits, i32)
    cell = arith.ShRUIOp(raw, shift)
    index = arith.IndexCastOp(cell, IndexType())
    table = memref.GetGlobalOp(table_symbol, table_type)
    lane0 = arith.ConstantOp.from_int_and_width(0, IndexType())
    c0 = memref.LoadOp.get(table, [index, lane0])
    finite = arith.CmpfOp(c0, c0, "ord")
    allowed = arith.AndIOp(incoming_rne, finite)
    block.add_ops([raw, shift, cell, index, table, lane0, c0, finite, allowed])
    selected = Block()
    lane1 = arith.ConstantOp.from_int_and_width(1, IndexType())
    lane2 = arith.ConstantOp.from_int_and_width(2, IndexType())
    c1 = memref.LoadOp.get(table, [index, lane1])
    c2 = memref.LoadOp.get(table, [index, lane2])
    mask = arith.ConstantOp.from_int_and_width((1 << (32 - carrier.policy.leading_bits)) - 1, i32)
    low = arith.AndIOp(raw, mask)
    t = arith.UIToFPOp(low, f32)
    first = math.FmaOp(t, c2, c1)
    second = math.FmaOp(first, t, c0)
    selected.add_ops([lane1, lane2, c1, c2, mask, low, t, first, second, scf.YieldOp(second)])
    refused = Block()
    fallback = func.CallOp(expression_symbol, [cut], [f32])
    refused.add_ops([fallback, scf.YieldOp(fallback)])
    choose = scf.IfOp(allowed, [f32], Region(selected), Region(refused))
    block.add_op(choose)
    product, scaled = proof.observer_operations[:2]
    factor = next(value for value in scaled.operands if value is not product.results[0])
    mapping = {proof.endpoint: choose.results[0], proof.up: up}
    block.add_op(factor.owner.clone(mapping))
    block.add_op(product.clone(mapping))
    block.add_op(scaled.clone(mapping))
    observe = func.CallOp(observer_symbol, [mapping[scaled.results[0]]], [i8])
    block.add_ops([observe, func.ReturnOp(observe)])
    function = func.FuncOp(carrier_symbol, ([f32, f32, i1], [i8]), Region(block))
    function.attributes.update(expression.attributes)
    record = {
        "prov.scalar_carrier_sha256": StringAttr(carrier.canonical_sha256),
        "prov.scalar_carrier_policy_sha256": StringAttr(carrier.policy.canonical_sha256),
    }
    for key, value in record.items():
        if key in function.attributes and function.attributes[key] != value:
            raise ValueError("source provenance conflicts with scalar carrier reification")
        function.attributes[key] = value
    validate_source_scalar_carrier(carrier)
    return expression, observation, function


def _reify_source_scalar_carrier_storage(carrier, table_symbol):
    from xdsl.dialects import memref
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, MemRefType, StringAttr, TensorType, UnitAttr, f32

    validate_source_scalar_carrier(carrier)
    shape = [1 << carrier.policy.leading_bits, 3]
    values = [struct.unpack(">f", struct.pack(">I", word))[0] for cell in carrier.coefficient_words for word in cell]
    return memref.GlobalOp.get(
        StringAttr(table_symbol),
        MemRefType(f32, shape),
        DenseIntOrFPElementsAttr.from_list(TensorType(f32, shape), values),
        constant=UnitAttr(),
    )


def reify_source_scalar_carrier_helpers(
    carrier: SourceScalarCarrier,
    proof: ClosedScalarObserver,
    *,
    table_symbol: str,
    expression_symbol: str,
    observer_symbol: str,
    carrier_symbol: str,
):
    """Return typed immutable storage and one fresh ``(f32,f32,i1)→i8`` helper.

    The ABI accepts original cut, original independent finishing operand and
    target-supplied incoming-RNE permission. False evaluates the original source.
    The current literal finishing factor and complete original observer remain.
    Use the family reifier for multiple helpers sharing one physical table.
    Ownership, invocation-local binding, original whole validation and complete
    cost remain separate caller obligations.
    """
    from xdsl.dialects.builtin import ModuleOp

    helpers = _reify_source_scalar_carrier_helper_ops(
        carrier,
        proof,
        table_symbol=table_symbol,
        expression_symbol=expression_symbol,
        observer_symbol=observer_symbol,
        carrier_symbol=carrier_symbol,
    )
    storage = _reify_source_scalar_carrier_storage(carrier, table_symbol)
    result = ModuleOp([storage, *helpers])
    result.verify()
    validate_source_scalar_carrier(carrier)
    return result


@dataclass(frozen=True)
class ScalarCarrierHelperBinding:
    """One selected current source proof and its explicit helper ABI names."""

    proof: ClosedScalarObserver
    expression_symbol: str
    observer_symbol: str
    carrier_symbol: str


def reify_source_scalar_carrier_helper_family(
    carrier: SourceScalarCarrier, bindings: tuple[ScalarCarrierHelperBinding, ...], *, table_symbol: str
):
    """Own one immutable coefficient table for all selected current observers.

    Every member is independently revalidated and retains its original literal
    finishing multiply and complete observer. Names describe the explicit ABI;
    they select neither numerical permission nor source arithmetic. All names
    occupy one checked namespace. Shared physical storage is constructed here,
    before upstream lowering; LLVM deduplication is never an admission premise.
    The caller still owns invocation-local binding, runtime RNE predicates,
    publication/alias contracts, aggregate runtime admission and complete costs.
    Unselected source observers remain on their original path.
    """
    from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr

    validate_source_scalar_carrier(carrier)
    if (
        type(bindings) is not tuple
        or not bindings
        or any(type(item) is not ScalarCarrierHelperBinding for item in bindings)
    ):
        raise ValueError("nonempty immutable typed scalar carrier bindings required")
    symbols = (
        table_symbol,
        *(name for item in bindings for name in (item.expression_symbol, item.observer_symbol, item.carrier_symbol)),
    )
    if any(type(name) is not str or not name.isascii() or not name.isidentifier() for name in symbols):
        raise ValueError("explicit nonempty ASCII family ABI symbols required")
    if len(set(symbols)) != len(symbols):
        raise ValueError("scalar carrier family storage/helper namespace conflicts")
    for item in bindings:
        if type(item.proof) is not ClosedScalarObserver or not any(item.proof is proof for proof in carrier._observers):
            raise ValueError("every family member requires a selected current source proof")
        validate_closed_scalar_observer(item.proof)
    storage, functions = _reify_source_scalar_carrier_storage(carrier, table_symbol), []
    for item in bindings:
        helpers = _reify_source_scalar_carrier_helper_ops(
            carrier,
            item.proof,
            table_symbol=table_symbol,
            expression_symbol=item.expression_symbol,
            observer_symbol=item.observer_symbol,
            carrier_symbol=item.carrier_symbol,
        )
        functions.extend(helpers)
    result = ModuleOp([storage, *functions])
    result.attributes.update(
        {
            "prov.scalar_carrier_sha256": StringAttr(carrier.canonical_sha256),
            "prov.scalar_carrier_storage": StringAttr(table_symbol),
            "prov.scalar_carrier_members": ArrayAttr([StringAttr(item.carrier_symbol) for item in bindings]),
        }
    )
    result.verify()
    validate_source_scalar_carrier(carrier)
    return result


# ---- source_scalar_carrier_binding -------------------------------------------------------------
# One-shot current-source scalar-carrier binding for ordinary model lowering.


def _identity(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        or value == "0" * 64
    ):
        raise ValueError("explicit implementation SHA256 identity required")


@dataclass(frozen=True)
class CompilerHostNumericAdmission:
    """Trusted caller's scoped host capability; core implements no native fenv.

    The factory must restore the complete incoming environment on every exit.
    verify_rne must inspect the actual current host environment, not return an
    unconditional promise. Its implementation is separately qualified/pinned.
    """

    context_factory: Callable
    verify_rne: Callable
    implementation_sha256: str

    def validate(self):
        _identity(self.implementation_sha256)
        if not callable(self.context_factory) or not callable(self.verify_rne):
            raise ValueError("scoped actual compiler-host RNE capability required")

    @contextmanager
    def enter(self):
        self.validate()
        with self.context_factory():
            if self.verify_rne() is not True:
                raise ValueError("actual compiler-host RNE admission refused")
            yield
            if self.verify_rne() is not True:
                raise ValueError("compiler-host RNE changed during source preparation")


@dataclass(frozen=True)
class IncomingRNECapability:
    """Provider-owned runtime ()→i1 predicate, read once per scalar point.

    The provider proves the actual linked implementation and ordinary ABI.
    This declaration grants no all-region or all-model FRM stability theorem.
    """

    symbol: str
    implementation_sha256: str
    reads_actual_incoming_rounding: bool = False
    preserves_rounding_mode: bool = False
    preserves_flags: bool = False
    nontrapping: bool = False
    no_memory_writes: bool = False
    placement: str = "per_point"

    def validate(self):
        _identity(self.implementation_sha256)
        if not isinstance(self.symbol, str) or not self.symbol.isascii() or not self.symbol.isidentifier():
            raise ValueError("explicit runtime predicate ABI symbol required")
        if self.placement != "per_point":
            raise ValueError("cross-FRM predicate hoisting has no complete region theorem")
        if any(
            value is not True
            for value in (
                self.reads_actual_incoming_rounding,
                self.preserves_rounding_mode,
                self.preserves_flags,
                self.nontrapping,
                self.no_memory_writes,
            )
        ):
            raise ValueError("actual incoming-RNE predicate and complete runtime effects required")


@dataclass(frozen=True)
class SourceScalarCarrierSelection:
    policy: ApproximateScalarCarrierPolicy
    effects: IntervalEffectContract
    coefficient_proposal: Callable
    incoming_rne: IncomingRNECapability
    compiler_host: CompilerHostNumericAdmission
    max_total_table_bytes: int
    namespace: str = "merlin_scalar_carrier"
    max_source_bytes: int = 16 * 1024 * 1024
    max_response_bytes: int = 16 * 1024 * 1024

    def validate(self):
        if type(self.policy) is not ApproximateScalarCarrierPolicy or type(self.effects) is not IntervalEffectContract:
            raise ValueError("distinct typed scalar policy and explicit source effects required")
        self.policy.validate()
        self.effects.validate()
        if (
            type(self.incoming_rne) is not IncomingRNECapability
            or type(self.compiler_host) is not CompilerHostNumericAdmission
        ):
            raise ValueError("typed runtime and separate compiler-host capabilities required")
        self.incoming_rne.validate()
        self.compiler_host.validate()
        if not callable(self.coefficient_proposal):
            raise ValueError("invocation-local current-expression coefficient proposal required")
        if not isinstance(self.namespace, str) or not self.namespace.isascii() or not self.namespace.isidentifier():
            raise ValueError("explicit private helper namespace required")
        if any(type(value) is not int or value <= 0 for value in (self.max_source_bytes, self.max_response_bytes)):
            raise ValueError("explicit positive private source/response quotas required")
        if (
            type(self.max_total_table_bytes) is not int
            or self.max_total_table_bytes < (1 << self.policy.leading_bits) * 12
        ):
            raise ValueError("explicit aggregate immutable table budget required")

    def callback(self, payload, directory):
        """Discover and prove current IR; no retained helper or call-site proof input."""
        from xdsl.dialects import func
        from xdsl.dialects.builtin import ModuleOp, StringAttr, f32, i1, i8
        from xdsl.ir import Block, Region

        from merlin.compiler.dialects.xdsl_context import text
        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

        self.validate()
        with self.compiler_host.enter():
            module = parse_mlir_text(payload.decode("utf-8"))
            proofs, refusals = find_closed_scalar_i8_observers(module, effects=self.effects)
            if not proofs:
                raise ValueError("current source has no admitted closed scalar observers")
            # Authenticate ALL source witnesses before the coefficient callback
            # and again before constructing any edit. Source IR is never mutated.
            for proof in proofs:
                validate_closed_scalar_observer(proof)
                # Whole current IR is verified in its native session before
                # insertion. xDSL's unrelated operation constraints can lag
                # native resource support; verify every selected container,
                # never re-create or relax an original external resource.
                container = proof.integer_result.owner.parent_op()
                while container is not None and container.name not in ("linalg.generic", "func.func"):
                    container = container.parent_op()
                if container is None:
                    raise ValueError("current closed scalar observer has no typed source container")
                container.verify()
            paths, blocks = operation_paths(module)
            groups = {}
            for proof in proofs:
                groups.setdefault(proof.expression.canonical_sha256, []).append(proof)
            if len(groups) * (1 << self.policy.leading_bits) * 12 > self.max_total_table_bytes:
                raise ValueError("current expression families exceed aggregate table budget")
            fragments, replacements, source_records, private_paths = [], [], [], set()
            predicate = func.FuncOp.external(self.incoming_rne.symbol, [], [i1])
            fragments.append(predicate)
            physical_table_bytes = 0
            for family_index, members in enumerate(groups.values()):
                members = tuple(members)
                proposal = self.coefficient_proposal(members[0].expression, self.effects, self.policy)
                for proof in proofs:
                    validate_closed_scalar_observer(proof)
                carrier = prepare_source_scalar_carrier(
                    members,
                    effects=self.effects,
                    policy=self.policy,
                    coefficient_words=proposal,
                )
                prefix = f"{self.namespace}_{family_index}"
                bindings = tuple(
                    ScalarCarrierHelperBinding(
                        proof,
                        f"{prefix}_expression_{index}",
                        f"{prefix}_observer_{index}",
                        f"{prefix}_carrier_{index}",
                    )
                    for index, proof in enumerate(members)
                )
                table_name = f"{prefix}_coefficients"
                family = reify_source_scalar_carrier_helper_family(carrier, bindings, table_symbol=table_name)
                for operation in tuple(family.body.block.ops):
                    operation.detach()
                    if isinstance(operation, func.FuncOp):
                        operation.properties["sym_visibility"] = StringAttr("private")
                    fragments.append(operation)
                physical_table_bytes += (1 << self.policy.leading_bits) * 12
                for index, binding in enumerate(bindings):
                    proof = binding.proof
                    block = Block(arg_types=[f32, f32])
                    predicate_call = func.CallOp(self.incoming_rne.symbol, [], [i1])
                    call = func.CallOp(binding.carrier_symbol, [*block.args, predicate_call.results[0]], [i8])
                    block.add_ops([predicate_call, call, func.ReturnOp(call.results[0])])
                    adapter_name = f"{prefix}_point_{index}"
                    adapter = func.FuncOp(adapter_name, ([f32, f32], [i8]), Region(block), visibility="private")
                    fragments.append(adapter)
                    anchor = proof.integer_result.uses
                    uses = tuple(anchor)
                    if (
                        len(uses) != 1
                        or uses[0].index != 0
                        or uses[0].operation.name not in ("linalg.yield", "func.return", "tensor.insert")
                    ):
                        raise ValueError("current integer observation escaped its original terminator")
                    terminator = uses[0].operation
                    private = set((*proof.expression_operations, *proof.observer_operations))
                    private = {operation for operation in private if operation.name != "arith.constant"}
                    if proof.cut.owner in private or proof.up.owner in private:
                        raise ValueError("original cut/up producer overlaps the replaced private scalar DAG")
                    order = [operation for operation in terminator.parent.ops if operation in private]
                    if len(order) != len(private):
                        raise ValueError("current scalar closure is not private to its original block")
                    remove = []
                    for operation in order:
                        path = paths[operation]
                        if path in private_paths:
                            raise ValueError("selected scalar closures overlap")
                        private_paths.add(path)
                        remove.append({"path": path, "name": operation.name})
                    replacements.append(
                        {
                            "anchor": paths[terminator],
                            "cut": value_locator(proof.cut, paths, blocks),
                            "up": value_locator(proof.up, paths, blocks),
                            "adapter": adapter_name,
                            "remove": remove,
                        }
                    )
                    source_records.append(
                        {
                            "expression_sha256": proof.expression.canonical_sha256,
                            "quant_factor_bits": proof.quant_factor_bits,
                            "anchor": paths[terminator],
                            "carrier": binding.carrier_symbol,
                            "diagnostic_paths_not_policy": True,
                        }
                    )
            for proof in proofs:
                validate_closed_scalar_observer(proof)
            helper_module = ModuleOp(fragments)
            helper_module.verify()
            fragment = text(helper_module, generic=True)
            # Native fragment namespace verification repeats this before mutation.
            existing = {op.sym_name.data for op in module.body.block.ops if hasattr(op, "sym_name")}
            fresh = [op.sym_name.data for op in helper_module.body.block.ops if hasattr(op, "sym_name")]
            if len(set(fresh)) != len(fresh) or existing.intersection(fresh):
                raise ValueError("current source helper/runtime namespace conflict")
            permission = {
                "source_effects": asdict(self.effects),
                "runtime_capability": asdict(self.incoming_rne),
                "compiler_host_capability_sha256": self.compiler_host.implementation_sha256,
                "policy_sha256": self.policy.canonical_sha256,
                "original_source_fallback": True,
                "private_publication": True,
            }
            digest = hashlib.sha256(payload).hexdigest()
            record = {
                "schema": "merlin.current_scalar_carrier_binding.v1",
                "status": "CURRENT_TYPED_EDITS_PREPARED",
                "source_sha256": digest,
                "fragment_sha256": hashlib.sha256(fragment.encode()).hexdigest(),
                "members": source_records,
                "refusals": [{"operation": op.name, "reason": reason} for op, reason in refusals],
                "family_count": len(groups),
                "physical_table_bytes": physical_table_bytes,
                "permissions": permission,
                "default_policy": False,
                "profitability": "UNKNOWN",
                "whole_output_validation": "PENDING_ORIGINAL_GATE_REQUIRED",
                "predicate_placement": "per_point_no_cross_FRM_hoist",
                "native_original_module_or_producer_reparse": False,
                "whole_current_module_verifier": "original native context before any mutation",
                "parent_verification": "every current closed selected container; original external resources untouched",
                "lifetime": "one-shot packet bound to exact current stage bytes",
            }
            Path(directory, "binding.json").write_text(json.dumps(record, indent=2) + "\n")
            return ScalarLeafPatch(digest, fragment, tuple(replacements), permission)


def host_scope(selection):
    if selection is None:
        return nullcontext()
    if type(selection) is not SourceScalarCarrierSelection:
        raise ValueError("typed invocation-local scalar-carrier selection required")
    selection.validate()
    return selection.compiler_host.enter()


def host_admitted(function):
    """Cover direct ordinary APIs before their first xDSL operation."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def invoke(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        if arguments.get("source_scalar_carrier") is not None:
            from merlin.compiler.lower.options.prov_cse import FEATURE as provenance_stripping_feature

            if provenance_stripping_feature in (arguments.get("features") or ()):
                raise ValueError("current typed scalar rewrite cannot follow provenance-stripping CSE")
        with host_scope(arguments.get("source_scalar_carrier")):
            return function(*args, **kwargs)

    return invoke
