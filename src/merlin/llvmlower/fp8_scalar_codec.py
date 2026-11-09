"""Explicit scalar FP8 representation legalization with byte-preserving ABI.

This does not select a quantizer or an arithmetic backend. A caller supplies
qualified source/helper/compiler bindings. Only F32/E4M3FN scalar casts change;
tensor and memref element types, source arithmetic and public types stay intact.
The saturating codec is used for truncation only after an SSA clamp proof rules
out overflow. Unconstrained standard truncation is deliberately unsupported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import write_pretty_json

from .host_llvm_helpers import HelperIR, _headers, _module_context


@dataclass(frozen=True)
class BoundedE4M3FNCodec:
    """Caller-qualified C10-compatible helpers, not numerical permission.

    Helpers use unsigned 64-bit containers for the low eight storage bits. This
    avoids a narrow-integer extension ABI assumption; the existing helper linker
    still checks complete actual scalar signatures and module contexts. Source
    and compilation records are pinned inputs, not inferred Torch installations.
    RNE, nontrapping/unobserved exceptions and NaN-payload nonobservation must be
    explicit. Signed zero and NaN sign/class are retained by the source codec.
    """

    helper_ir: HelperIR
    source_bindings: tuple[HelperIR, ...]
    compilation_record: HelperIR
    encode_symbol: str
    decode_symbol: str
    rounding: str
    floating_effects: str
    nan_payload_unobserved: bool

    def validate(self):
        if (
            type(self.helper_ir) is not HelperIR
            or type(self.compilation_record) is not HelperIR
            or type(self.source_bindings) is not tuple
            or not self.source_bindings
            or any(type(binding) is not HelperIR for binding in self.source_bindings)
            or type(self.rounding) is not str
            or type(self.floating_effects) is not str
            or self.rounding != "rne"
            or self.floating_effects != "nontrapping_unobserved"
            or self.nan_payload_unobserved is not True
        ):
            raise ValueError("explicit codec source/compiler and RNE/effect bindings required")
        names = (self.encode_symbol, self.decode_symbol)
        if len(set(names)) != 2 or any(
            type(name) is not str
            or not name
            or not (name[0].isalpha() or name[0] == "_")
            or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in name)
            for name in names
        ):
            raise ValueError("distinct ordinary C codec symbols required")
        paths = [self.helper_ir.validate(), self.compilation_record.validate()]
        paths.extend(binding.validate() for binding in self.source_bindings)
        if len(set(paths)) != len(paths):
            raise ValueError("distinct immutable codec input bindings required")
        text = paths[0].read_text()
        _, headers, _ = _headers(text)
        signatures = {
            self.encode_symbol: (("i64", ()), (("float", ()),)),
            self.decode_symbol: (("float", ()), (("i64", ()),)),
        }
        for name, signature in signatures.items():
            actual = headers.get(name, {})
            if not actual.get("defined") or actual.get("signature") != signature:
                raise ValueError("codec helper signature/extension ABI is unsupported: " + name)
        context = _module_context(text)
        if set(context) != {"triple", "datalayout"}:
            raise ValueError("codec helper requires explicit target triple and data layout")
        return {
            "input_sha256": {str(path): sha256_file(path) for path in paths},
            "helper_context": context,
            "codec": "caller-qualified C10-compatible E4M3FN conversion",
            "encode_symbol": self.encode_symbol,
            "decode_symbol": self.decode_symbol,
            "rounding": self.rounding,
            "floating_effects": self.floating_effects,
            "nan_payload_unobserved": True,
            "overflow": "source SSA bound required; unconstrained truncf refused",
        }


def _none_fastmath(op):
    flags = op.properties.get("fastmath")
    return flags is None or not flags.data


def has_fp8_casts(module):
    """Structural selection only; unsupported FP8 cast forms still reach refusal."""
    from xdsl.dialects.arith import ExtFOp, TruncFOp
    from xdsl.utils.type import get_element_type_or_self

    from merlin.xdsl_dialects.fp8 import Float8E4M3FNType, Float8E5M2Type

    return any(
        isinstance(op, ExtFOp | TruncFOp)
        and any(
            isinstance(get_element_type_or_self(v.type), Float8E4M3FNType | Float8E5M2Type)
            for v in (*op.operands, *op.results)
        )
        for op in module.walk()
    )


def scoped_codec(policy, output: Path, *, prefix: str):
    """Alpha-rename exact helper definitions/globals for one compiled object.

    All token references to defined functions/globals change together. External
    declarations remain untouched; quoted strings and metadata are not symbol
    references. LLVM still verifies the complete cloned module in the existing
    link/inlining seam. Inline assembly and special LLVM globals refuse.
    """
    from .late_quant_rne import _identity

    if type(policy) is not BoundedE4M3FNCodec or not prefix.isascii() or not prefix.isidentifier():
        raise ValueError("explicit codec and ordinary object symbol prefix required")
    bindings = policy.validate()
    original = policy.helper_ir.validate().read_text()
    tokens, headers, _ = _headers(original)
    if any(token.text == "asm" for token in tokens):
        raise ValueError("codec helper symbol scope cannot prove inline assembly references")
    defined = {name for name, row in headers.items() if row["defined"]}
    defined.update(
        _identity(token.text)
        for i, token in enumerate(tokens[:-1])
        if token.text.startswith("@") and tokens[i + 1].text == "="
    )
    if any(name.startswith("llvm.") for name in defined):
        raise ValueError("special LLVM global definitions are unsupported in a scoped codec")
    mapping = {name: prefix + "_codec_" + str(index) for index, name in enumerate(sorted(defined))}
    if set(mapping.values()) & set(headers):
        raise ValueError("derived codec namespace conflicts with original helper symbols")
    changes = [
        (token.start, token.end, "@" + mapping[_identity(token.text)])
        for token in tokens
        if token.text.startswith("@") and _identity(token.text) in mapping
    ]
    selected = original
    for start, end, value in reversed(changes):
        selected = selected[:start] + value + selected[end:]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    helper = output / "helper.ll"
    helper.write_text(selected)
    result = replace(
        policy,
        helper_ir=HelperIR(helper, sha256_file(helper)),
        source_bindings=(*policy.source_bindings, policy.helper_ir),
        encode_symbol=mapping[policy.encode_symbol],
        decode_symbol=mapping[policy.decode_symbol],
    )
    result.validate()
    if policy.validate() != bindings:
        raise ValueError("codec input bindings changed during symbol scoping")
    write_pretty_json(
        output / "scope.json",
        {
            "schema": "merlin.fp8_codec_symbol_scope.v1",
            "source": bindings,
            "symbols": mapping,
            "selected_helper_sha256": sha256_file(helper),
        },
    )
    return result


def _constant(value):
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import Float32Type, FloatAttr

    owner = value.owner
    if isinstance(owner, ConstantOp):
        if set(owner.properties) != {"value"} or not _known_provenance(owner):
            return None
        attr = owner.value
        if isinstance(attr, FloatAttr) and isinstance(attr.type, Float32Type):
            result = attr.value.data
            if math.isfinite(result):
                return result
    return None


def _known_provenance(op):
    return _known_attributes(op.attributes)


def _known_attributes(attributes):
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr

    arrays = {"prov.source_node_ids", "prov.origin_node_ids"}
    strings = {
        "prov.region_id",
        "prov._pattern_hint",
        "prov.op",
        "prov.family",
        "prov.trace_role",
        "prov.aten",
        "prov.orig_dtype",
        "prov.module",
        "prov.fqn",
        "prov.dispatch_id",
        "prov.fp8_cast_domain",
        "prov.explicit_numeric_policy",
        "prov.scaled_mm_stage",
        "prov.scaled_mm_source_layouts",
    }
    for key, value in attributes.items():
        if key == "prov.fused_elementwise_sources":
            if (
                not isinstance(value, ArrayAttr)
                or len(value) != 2
                or any(not isinstance(item, DictionaryAttr) or not _known_attributes(item.data) for item in value)
            ):
                return False
            continue
        if key in arrays:
            if not isinstance(value, ArrayAttr) or any(not isinstance(item, StringAttr) for item in value):
                return False
        elif key not in strings or not isinstance(value, StringAttr):
            return False
    return True


def _complete_elementwise(op):
    """Prove every output element receives exactly one pure body evaluation."""
    from xdsl.dialects.builtin import AffineMapAttr, NoneAttr, TensorType
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.ir.affine import AffineMap
    from xdsl.traits import is_side_effect_free

    if (
        not isinstance(op, GenericOp)
        or len(op.inputs) != 1
        or len(op.outputs) != 1
        or len(op.results) != 1
        or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        or not _known_provenance(op)
        or len(op.body.blocks) != 1
        or any(
            not _known_provenance(child)
            or not (is_side_effect_free(child) or isinstance(child, YieldOp) and child is op.body.block.last_op)
            for child in op.body.block.ops
        )
    ):
        return False
    types = [value.type for value in (*op.inputs, *op.outputs, *op.results)]
    if any(not isinstance(t, TensorType) or not isinstance(t.encoding, NoneAttr) for t in types):
        return False
    shape = types[0].get_shape()
    if (
        any(type(d) is not int or d <= 0 for d in shape)
        or any(t.get_shape() != shape for t in types)
        or types[1] != types[2]
        or len(op.iterator_types) != len(shape)
        or any(it.data is not IteratorType.PARALLEL for it in op.iterator_types)
        or len(op.indexing_maps) != 2
        or any(not isinstance(m, AffineMapAttr) or m.data != AffineMap.identity(len(shape)) for m in op.indexing_maps)
    ):
        return False
    return True


def _ordered_clamp_body(op):
    from xdsl.dialects.arith import CmpfOp, SelectOp
    from xdsl.dialects.linalg.ops import YieldOp

    body = tuple(op.body.block.ops)
    if len(body) != 7 or not isinstance(body[-1], YieldOp) or len(body[-1].arguments) != 1:
        return False
    upper = body[-1].arguments[0].owner
    if not isinstance(upper, SelectOp):
        return False
    upper_cmp, lower = upper.cond.owner, upper.rhs.owner
    if not isinstance(upper_cmp, CmpfOp) or not isinstance(lower, SelectOp):
        return False
    lower_cmp = lower.cond.owner
    if not isinstance(lower_cmp, CmpfOp):
        return False
    lo, hi = _constant(lower.lhs), _constant(upper.lhs)
    if (
        lo is None
        or hi is None
        or lo > hi
        or lower_cmp.predicate.value.data != 4
        or upper_cmp.predicate.value.data != 2
        or not _none_fastmath(lower_cmp)
        or not _none_fastmath(upper_cmp)
        or tuple(lower_cmp.operands) != (lower.rhs, lower.lhs)
        or tuple(upper_cmp.operands) != (lower.result, upper.lhs)
        or set(body) != {lower.lhs.owner, lower_cmp, lower, upper.lhs.owner, upper_cmp, upper, body[-1]}
    ):
        return False
    return all(
        set(child.properties)
        <= (
            {"value"}
            if child.name == "arith.constant"
            else {"predicate", "fastmath"}
            if child.name == "arith.cmpf"
            else set()
        )
        for child in body
    )


def _domain(value, visited=frozenset()):
    """A finite ordered bound or NaN, proved from current scalar/tensor SSA.

    Ordered clamp selects preserve a possible NaN operand. The codec's NaN
    sign/class preservation and caller's payload/effect contract are separate
    from this range proof. No example values or provenance tags grant a bound.
    """
    from xdsl.dialects.arith import CmpfOp, SelectOp
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir import BlockArgument, OpResult

    if value in visited:
        return -math.inf, math.inf
    visited = visited | {value}
    constant = _constant(value)
    if constant is not None:
        return constant, constant
    if isinstance(value, BlockArgument):
        parent = value.owner.parent_op()
        if _complete_elementwise(parent) and value.index < len(parent.inputs):
            return _domain(parent.inputs[value.index], visited)
    if not isinstance(value, OpResult):
        return -math.inf, math.inf
    owner = value.owner
    if isinstance(owner, GenericOp):
        body = tuple(owner.body.block.ops)
        if _complete_elementwise(owner) and value is owner.results[0] and _ordered_clamp_body(owner):
            return _domain(body[-1].arguments[0], visited)
    if isinstance(owner, SelectOp):
        condition = owner.cond.owner
        if (
            isinstance(condition, CmpfOp)
            and _none_fastmath(condition)
            and not owner.properties
            and _known_provenance(owner)
            and _known_provenance(condition)
            and set(condition.properties) <= {"predicate", "fastmath"}
        ):
            source, constant_value = condition.operands
            limit = _constant(constant_value)
            if owner.lhs is constant_value and owner.rhs is source and limit is not None:
                low, high = _domain(source, visited)
                predicate = condition.predicate.value.data
                # xDSL/Arith's ordered predicates: olt=4, ogt=2.
                if predicate == 4:
                    return max(low, limit), max(high, limit)
                if predicate == 2:
                    return min(low, limit), min(high, limit)
    return -math.inf, math.inf


def legalize_file(source: Path, output: Path, policy: BoundedE4M3FNCodec | None):
    """Use as an explicit prepared-model transform; default None performs no I/O.

    Only the proved scalar casts become private pure helper calls and bitcasts.
    Callers link the exact helper IR using ``host_llvm_helpers.link_and_inline``
    and require both codec symbols inlined. This operation alone makes no claim
    that an entire graph, guards or device arithmetic has compiled or executed.
    """
    if policy is None:
        return source
    if type(policy) is not BoundedE4M3FNCodec:
        raise ValueError("an explicit BoundedE4M3FNCodec is required")
    bindings = policy.validate()
    from xdsl.dialects.arith import BitcastOp, ExtFOp, ExtUIOp, TruncFOp, TruncIOp
    from xdsl.dialects.builtin import Float32Type, i8, i64
    from xdsl.dialects.func import CallOp, FuncOp
    from xdsl.pattern_rewriter import Rewriter
    from xdsl.utils.type import get_element_type_or_self

    from merlin.common.proc import run_checked
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.deployable_ir import checked_generic_text
    from merlin.xdsl_dialects.fp8 import Float8E4M3FNType, Float8E5M2Type, register_fp8_bitcast_constraints

    from . import stock_arith_compat, toolchain
    from .compilation_recipe import CompilationRecipe

    register_fp8_bitcast_constraints()

    source, output = Path(source), Path(output)
    original = source.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    snapshot, normalized = output / "input.mlir", output / "normalized.mlir"
    snapshot.write_bytes(original)
    normalizer = toolchain.llvm_install() / "bin" / "mlir-opt"
    (output / "normalization").mkdir()
    recipe = CompilationRecipe(output / "normalization", producer=Path(__file__))
    compatibility_source = Path(stock_arith_compat.__file__)
    compatibility_sha256 = sha256_file(compatibility_source)
    recipe.bind_preparation("stock_integer_compatibility", compatibility_source)
    argv = [normalizer, snapshot, "--mlir-print-op-generic", "-o", normalized]
    recipe.run(argv, runner=lambda args: run_checked(args, timeout=180), inputs=[snapshot], output=normalized)
    recipe.completed_product(normalized, kind="mlir")
    if snapshot.read_bytes() != original or source.read_bytes() != original:
        raise ValueError("codec normalization modified its immutable source")
    # Custom-form xDSL parsing can discard unknown linalg attributes. Analyze
    # only the stock compiler's lossless generic form, never a lossy reprint.
    module = parse_mlir_text(normalized.read_text())
    stock_integer_compatibility = stock_arith_compat.unflagged_truncations(module)
    module.verify()
    casts, counts = [], {"decode": 0, "encode": 0}
    for op in module.walk():
        if not isinstance(op, ExtFOp | TruncFOp):
            continue
        incoming, result = op.operands[0].type, op.results[0].type
        involves_fp8 = any(
            isinstance(get_element_type_or_self(dtype), Float8E4M3FNType | Float8E5M2Type)
            for dtype in (incoming, result)
        )
        if not involves_fp8:
            continue
        parent = op.parent_op()
        while parent is not None:
            if parent.name == "linalg.generic" and (
                set(parent.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
                or not _known_provenance(parent)
            ):
                raise ValueError("FP8 cast is nested under an unknown generic contract")
            parent = parent.parent_op()
        decode = isinstance(op, ExtFOp) and isinstance(incoming, Float8E4M3FNType) and isinstance(result, Float32Type)
        encode = isinstance(op, TruncFOp) and isinstance(incoming, Float32Type) and isinstance(result, Float8E4M3FNType)
        if not (decode or encode) or op.properties or not _known_provenance(op):
            raise ValueError("unsupported FP8 scalar conversion, rounding or effects")
        if encode:
            low, high = _domain(op.operands[0])
            if not (-448.0 <= low <= high <= 448.0):
                raise ValueError("FP8 truncation lacks a current SSA bound within E4M3FN finite range")
        casts.append((op, decode))
    if not casts:
        raise ValueError("selected codec found no supported actual scalar FP8 casts")
    symbols = {op.sym_name.data for op in module.body.block.ops if isinstance(op, FuncOp)}
    if {policy.encode_symbol, policy.decode_symbol} & symbols:
        raise ValueError("codec symbols conflict with original module definitions/declarations")
    for op, decode in casts:
        if decode:
            bits = BitcastOp(op.operands[0], i8)
            wide = ExtUIOp(bits.result, i64)
            call = CallOp(policy.decode_symbol, [wide.result], [Float32Type()])
            replacement = [bits, wide, call]
            counts["decode"] += 1
        else:
            call = CallOp(policy.encode_symbol, [op.operands[0]], [i64])
            narrow = TruncIOp(call.results[0], i8)
            bits = BitcastOp(narrow.result, op.results[0].type)
            replacement = [call, narrow, bits]
            counts["encode"] += 1
        for new in replacement:
            new.attributes.update(op.attributes)
        Rewriter.replace_op(op, replacement)
    if counts["decode"]:
        module.body.block.add_op(FuncOp.external(policy.decode_symbol, [i64], [Float32Type()]))
    if counts["encode"]:
        module.body.block.add_op(FuncOp.external(policy.encode_symbol, [Float32Type()], [i64]))
    module.verify()
    text = checked_generic_text(module)
    if (
        source.read_bytes() != original
        or policy.validate() != bindings
        or sha256_file(compatibility_source) != compatibility_sha256
    ):
        raise ValueError("codec source or selected bindings changed during legalization")
    result = output / "model.mlir"
    result.write_text(text)
    write_pretty_json(
        output / "report.json",
        {
            "schema": "merlin.explicit_fp8_scalar_codec.v1",
            "source_sha256": sha256_file(source),
            "output_sha256": sha256_file(result),
            "bindings": bindings,
            "casts": counts,
            "stock_integer_compatibility": stock_integer_compatibility,
            "stock_integer_compatibility_input_sha256": {str(compatibility_source): compatibility_sha256},
            "normalizer": {
                "path": str(normalizer),
                "sha256": sha256_file(normalizer),
                "argv": [str(arg) for arg in argv],
                "normalized_sha256": sha256_file(normalized),
            },
            "public_storage_abi": "unchanged original one-byte E4M3FN types",
            "remaining_obligations": [
                "helper linkage/inlining",
                "RNE execution context",
                "actual full-output target gate",
            ],
            "arithmetic_backend_selected": False,
        },
    )
    return result
