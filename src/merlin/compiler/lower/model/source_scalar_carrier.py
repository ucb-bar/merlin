"""Prove and reify an explicitly selected compact rounded scalar carrier.

Coefficients are immutable caller choices, not permission inferred from samples.
Every fine raw-binary32 interval must satisfy the declared carrier error budget.
The runtime representation is a coarse three-coefficient table and two ordered
binary32 FMAs. Invalid cells and an unadmitted incoming rounding mode evaluate
the original current expression. Finishing multiplies and the complete original
integer observer are cloned unchanged. The target supplies the incoming-RNE
predicate; this module reads no ISA or runtime floating environment.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field

from merlin.compiler.lower.model.source_expression_interval import (
    ClosedScalarObserver,
    IntervalEffectContract,
    build_source_interval_table,
    validate_closed_scalar_observer,
)
from merlin.compiler.lower.model.source_observation_helpers import reify_closed_scalar_observer_helpers
from merlin.compiler.lower.model.source_scalar_carrier_policy import ApproximateScalarCarrierPolicy


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
