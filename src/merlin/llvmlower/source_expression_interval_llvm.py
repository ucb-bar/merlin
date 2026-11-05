"""Bind an explicit typed closed-source table proof to actual scalar LLVM SSA.

The LLVM lexer/productions retain types, exact literal bits, operation order and
all live uses. Source labels never select a route. This optional bridge replaces
only an unobserved scalar endpoint; it moves no load, store or arithmetic. Table
data, callbacks, runtime FENV guard and final object/link closure are supplied by
the normal build caller. There is no target instruction implementation here.
"""

from __future__ import annotations

import hashlib
import math
import struct

from .late_quant_rne import (
    _functions,
    _identity,
    _Instruction,
    _instruction,
    _match,
    _match_intrinsic,
    _number,
    _statement_boundary,
    _tokens,
)
from .source_expression_interval import IntervalEffectContract, SourceIntervalTable, validate_closed_scalar_observer


def _extended_instruction(tokens, index):
    existing = _instruction(tokens, index)
    if existing is not None:
        return existing
    t = [token.text for token in tokens[index : index + 20]]
    if len(t) < 5 or not t[0].startswith("%") or t[1] != "=":
        return None
    opcode, callee, output = t[2], "", ""
    if opcode in ("fmul", "fadd", "fdiv", "shl") and len(t) >= 7 and t[5] == ",":
        dtype, args, count = t[3], (t[4], t[6]), 7
    elif opcode == "bitcast" and len(t) >= 7 and t[5] == "to":
        dtype, args, output, count = t[3], (t[4],), t[6], 7
    elif (
        opcode == "call"
        and len(t) >= 15
        and t[3] == "float"
        and t[4] == "@llvm.fma.f32"
        and t[5] == "("
        and t[6] == t[9] == t[12] == "float"
        and t[8] == t[11] == ","
        and t[14] == ")"
    ):
        dtype, callee, args, count = "float", "llvm.fma.f32", (t[7], t[10], t[13]), 15
    else:
        return None
    if not _statement_boundary(tokens, index + count):
        return None
    return _Instruction(
        t[0], opcode, dtype, args, tokens[index].start, tokens[index + count - 1].end, callee=callee, output_type=output
    )


def _literal(token, dtype):
    if dtype == "i32":
        value = int(token)
        if not -(2**31) <= value < 2**32:
            raise ValueError("i32 literal is out of range")
        return ("literal", "i32", value % 2**32)
    if dtype != "float":
        raise ValueError("unsupported expression literal type")
    if token.startswith("f0x") and len(token) == 11:
        word = int(token[3:], 16)
        value = struct.unpack("<f", struct.pack("<I", word))[0]
    else:
        value = _number(token)
        if value is None:
            raise ValueError("unsupported binary32 literal")
        word = struct.unpack("<I", struct.pack("<f", value))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite expression literal")
    return ("literal", "float", word)


_SOURCE_CODES = {
    "arith.negf": "fneg",
    "arith.maximumf": "llvm.maximum.f32",
    "arith.minimumf": "llvm.minimum.f32",
    "arith.mulf": "fmul",
    "arith.addf": "fadd",
    "arith.subf": "fsub",
    "math.fma": "llvm.fma.f32",
    "arith.divf": "fdiv",
    "arith.fptosi": "fptosi",
    "arith.addi": "add",
    "arith.shli": "shl",
    "arith.bitcast": "bitcast",
}


def _typed_tree(expression):
    values = [("input", "float")]
    for step in expression.steps:
        if step.opcode == "arith.constant":
            values.append(
                (
                    "literal",
                    "float" if step.dtype == "f32" else "i32",
                    step.literal if step.dtype == "f32" else step.literal % 2**32,
                )
            )
        else:
            values.append(
                (
                    _SOURCE_CODES[step.opcode],
                    "float" if step.dtype == "f32" else "i32",
                    tuple(values[i] for i in step.arguments),
                )
            )
    return values[-1]


def _compiled_tree(value, definitions, cut, *, dtype="float", visited=None):
    if value == cut:
        if dtype != "float":
            raise ValueError("source cut type changed")
        return ("input", "float"), set()
    if not value.startswith("%"):
        return _literal(value, dtype), set()
    visited = frozenset() if visited is None else visited
    if value in visited or value not in definitions:
        raise ValueError("unknown or cyclic source expression input")
    op = definitions[value]
    code = op.callee if op.opcode == "call" else op.opcode
    if code not in _SOURCE_CODES.values() or op.predicate:
        raise ValueError("unknown source expression operation")
    result_type = op.output_type or op.dtype
    if result_type != dtype:
        raise ValueError("source expression precision changed")
    arguments, nodes = [], {value}
    for operand in op.operands:
        tree, dependencies = _compiled_tree(operand, definitions, cut, dtype=op.dtype, visited=visited | {value})
        arguments.append(tree)
        nodes |= dependencies
    return (code, result_type, tuple(arguments)), nodes


def _rewrite_source_interval_lookup(
    source,
    *,
    proofs=(),
    table: SourceIntervalTable | None = None,
    lookup_symbol=None,
    effects: IntervalEffectContract | None = None,
    integer_result=False,
):
    """Explicit structural binding before any target RNE/ISA legalization.

    Source bounded-RNE arithmetic or its exact standard roundeven intrinsic
    must still be present: unknown intrinsics and inline instructions refuse. The build verifies the
    returned complete LLVM module and supplies the original source callbacks.
    Empty selection returns exactly the original bytes without parsing.
    """
    report = {
        "schema": "source_expression_interval_i8_llvm_binding_v1"
        if integer_result
        else "source_expression_interval_llvm_binding_v1",
        "routes": [],
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    if not proofs:
        return source, report
    if effects is None:
        raise ValueError("explicit numerical/effect policy required")
    effects.validate()
    if (
        table is None
        or not isinstance(lookup_symbol, str)
        or not lookup_symbol
        or lookup_symbol[0].isdigit()
        or not all(c.isascii() and (c.isalnum() or c == "_") for c in lookup_symbol)
    ):
        raise ValueError("explicit table and scalar callback symbol required")
    if (
        type(table.leading_bits) is not int
        or not 9 <= table.leading_bits <= 24
        or len(table.data) != (1 << table.leading_bits) * 8
    ):
        raise ValueError("invalid immutable table layout")
    admitted = {}
    for proof in proofs:
        validate_closed_scalar_observer(proof)
        if proof.expression.canonical_sha256 != table.expression_sha256:
            raise ValueError("physical table identity does not match source arithmetic")
        admitted.setdefault(proof.quant_factor_bits, []).append(_typed_tree(proof.expression))
    tokens = _tokens(source)
    if any(t.text == "strictfp" or t.text.startswith("@llvm.experimental.constrained.") for t in tokens):
        raise ValueError("strict or constrained compiled floating context")
    if any(t.text.startswith("@") and _identity(t.text) == lookup_symbol for t in tokens):
        raise ValueError("lookup symbol collides with an existing declaration or definition")
    edits = []
    for function_index, body in enumerate(_functions(tokens)):
        definitions, uses, instructions, locations = {}, {}, [], {}
        block = 0
        for index, token in enumerate(body):
            if index + 1 < len(body) and body[index + 1].text == ":":
                block += 1
            if token.text.startswith("%") and index + 1 < len(body) and body[index + 1].text == "=":
                locations[token.text] = (token.start, block)
                operation = _extended_instruction(body, index)
                if operation is not None:
                    definitions[operation.result] = operation
                    instructions.append(operation)
        call_positions = [index for index, token in enumerate(body) if token.text == "call"]
        recognized_calls = {operation.start for operation in instructions if operation.opcode == "call"}
        if any(index < 2 or body[index - 2].start not in recognized_calls for index in call_positions):
            # Unknown calls can change or observe rounding/flags even without
            # a data dependency on the endpoint. The typed source block is pure.
            continue
        if any(
            operation.opcode == "call"
            and operation.callee not in ("llvm.maximum.f32", "llvm.minimum.f32", "llvm.fma.f32", "llvm.roundeven.f32")
            for operation in instructions
        ):
            continue
        # Every SSA use is counted, including operands of unsupported memory or
        # call instructions. Unknown users cannot escape through a parser gap.
        owner = None
        for index, token in enumerate(body):
            if index + 1 < len(body) and (body[index + 1].text == ":" or body[index + 1].text == "="):
                owner = token.text if body[index + 1].text == "=" else None
                continue
            if token.text in ("store", "ret", "br", "switch", "invoke", "fence", "atomicrmw", "cmpxchg", "unreachable"):
                owner = None
            if token.text == "call" and (index == 0 or body[index - 1].text != "="):
                owner = None
            if token.text.startswith("%"):
                uses.setdefault(token.text, []).append(owner)
        legacy = {_identity(name): operation for name, operation in definitions.items()}
        for final in instructions:
            rne = _match(final, legacy) or _match_intrinsic(final, legacy)
            if rne is None or rne["integer_dtype"] != "i8" or rne["bounds"] != [-128, 127]:
                continue
            scaled = definitions.get(rne["raw_input"])
            if scaled is None or scaled.opcode != "fmul" or scaled.dtype != "float":
                continue
            literal_factors = [x for x in scaled.operands if not x.startswith("%")]
            if len(literal_factors) != 1:
                continue
            factor = literal_factors[0]
            factor_bits = _literal(factor, "float")[2]
            if factor_bits not in admitted:
                continue
            product_name = next(x for x in scaled.operands if x != factor)
            product = definitions.get(product_name)
            if product is None or product.opcode != "fmul" or product.dtype != "float":
                continue
            for endpoint_name, up in (product.operands, product.operands[::-1]):
                endpoint = definitions.get(endpoint_name)
                if endpoint is None or endpoint.opcode != "fmul" or endpoint.dtype != "float":
                    continue
                for cut in endpoint.operands:
                    try:
                        tree, expression_nodes = _compiled_tree(endpoint_name, definitions, cut)
                    except ValueError:
                        continue
                    if tree not in admitted[factor_bits]:
                        continue
                    if any(
                        value.startswith("%")
                        and value in locations
                        and (
                            locations[value][0] >= endpoint.start
                            or locations[value][1] != locations[endpoint.result][1]
                        )
                        for value in (cut, up)
                    ):
                        continue
                    if uses.get(endpoint_name) != [product_name] or uses.get(product_name) != [scaled.result]:
                        continue
                    # Recover the complete quant observer graph back to scaled,
                    # then close every internal expression/observer use. Its i8
                    # endpoint can retain multiple original integer consumers.
                    observer_nodes, pending = set(), [final.result]
                    while pending:
                        name = pending.pop()
                        if name == scaled.result or not name.startswith("%") or name in observer_nodes:
                            continue
                        operation = definitions.get(name)
                        if operation is None:
                            observer_nodes = set()
                            break
                        observer_nodes.add(name)
                        pending.extend(operation.operands)
                    if not observer_nodes:
                        continue
                    allowed = expression_nodes | observer_nodes | {product_name, scaled.result}
                    if any(
                        any(user not in allowed for user in uses.get(name, [])) for name in allowed - {final.result}
                    ):
                        continue
                    # Both routes retain complete source/consumer closure. The
                    # integer route publishes precisely that closed observation
                    # at its source location, eliminating only dead pure nodes
                    # under the explicit unobserved-effects contract.
                    replaced = final if integer_result else endpoint
                    result_type = "i8" if integer_result else "float"
                    edits.append(
                        (
                            replaced.start,
                            replaced.end,
                            f"{replaced.result} = call {result_type} @{lookup_symbol}(float {cut}, float {up}, float {factor})",
                        )
                    )
                    report["routes"].append(
                        {
                            "function_body_index": function_index,
                            "endpoint_source_span": [endpoint.start, endpoint.end],
                            "endpoint": endpoint.result,
                            "input": cut,
                            "up": up,
                            "quant_factor_bits": factor_bits,
                            "source_expression_sha256": table.expression_sha256,
                            "integer_observation": final.result,
                            "complete_scalar_uses_closed": True,
                            "source_observer_proof": rne,
                        }
                    )
                    if integer_result:
                        report["routes"][-1].update(
                            integer_source_span=[final.start, final.end],
                            result_type="i8",
                            floating_carrier_retained=False,
                            effect_contract="RNE/gradual/nontrapping/flags-and-signed-zero-unobserved",
                        )
                    break
                else:
                    continue
                break
    if len({start for start, _, _ in edits}) != len(edits):
        raise ValueError("ambiguous duplicate endpoint binding")
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    if edits:
        result_type = "i8" if integer_result else "float"
        source += f"\ndeclare {result_type} @{lookup_symbol}(float,float,float)\n"
    report.update(
        table_sha256=table.sha256,
        leading_bits=table.leading_bits,
        rewritten_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    return source, report


def rewrite_source_interval_lookup(source, *, proofs=(), table=None, lookup_symbol=None, effects=None):
    """Retain the default floating carrier route and its original byte output."""
    return _rewrite_source_interval_lookup(
        source, proofs=proofs, table=table, lookup_symbol=lookup_symbol, effects=effects
    )


def rewrite_source_interval_i8_lookup(source, *, proofs=(), table=None, lookup_symbol=None, effects=None):
    """Publish the existing closed i8 observation through an explicit callback.

    Complete source and compiled SSA-use closure must prove no floating escape.
    The callback retains original rounded finishing operations on refusal and
    returns the endpoint certificate's original saturated ties-even i8 on success.
    Unknown floating modes require the caller's retained original-source guard;
    this API supplies no target ISA, arithmetic relaxation or default policy.
    """
    return _rewrite_source_interval_lookup(
        source,
        proofs=proofs,
        table=table,
        lookup_symbol=lookup_symbol,
        effects=effects,
        integer_result=True,
    )
