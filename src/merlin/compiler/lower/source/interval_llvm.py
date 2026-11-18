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
from dataclasses import dataclass, field

from merlin.compiler.lower.quant.late_quant_rne import (
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
from merlin.compiler.lower.source.finite import (
    BroadcastScaledIntegerFiniteProof,
    emit_finite_scale_scan,
    prove_broadcast_scaled_integer_finite,
    validate_broadcast_scaled_integer_finite,
)
from merlin.compiler.lower.source.interval import (
    IntervalEffectContract,
    SourceIntervalTable,
    validate_closed_scalar_observer,
)

# ---- source_expression_interval_llvm -----------------------------------------------------------


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


# ---- scaled_integer_finite_llvm ----------------------------------------------------------------
# Explicit source-bound scale scans before repeated finite-input observers.
#
# The portable wrapper is admitted from exact source arithmetic and bounded LLVM
# scale loads. Immutable input storage and disjoint fresh output are mandatory
# caller ownership contracts. No target mode test, ISA or automatic policy lives
# here; the caller retains the original source helper outside stable RNE.


@dataclass(frozen=True)
class FiniteScaleScan:
    argument: int
    count: int
    proof: BroadcastScaledIntegerFiniteProof


@dataclass(frozen=True)
class FiniteScaleHelper:
    symbol: str
    arguments: tuple[str, ...]
    scans: tuple[FiniteScaleScan, ...]
    source_sha256: str
    body_sha256: str
    _source: str = field(repr=False)
    _body_source: str = field(repr=False)
    _routes: tuple = field(repr=False)
    _observers: tuple = field(repr=False)
    _effects: object = field(repr=False)


def _plain(words, opcode, dtype):
    if not words or words[0] != opcode:
        raise ValueError("unexpected compiled producer operation")
    tail = list(words[1:])
    while tail and tail[0] in ("nuw", "nsw", "inbounds"):
        tail.pop(0)
    if not tail or tail.pop(0) != dtype:
        raise ValueError("compiled producer precision/flags changed")
    return tail


class _Body:
    """Narrow ordinary SSA/branch grammar with exact loop induction bounds."""

    def __init__(self, source, body):
        self.defs, self.blocks, self.locations = {}, {}, {}
        current = None
        for line in source[body[0].start : body[-1].end].splitlines():
            tokens = [t.text for t in _tokens(line)]
            if not tokens:
                continue
            if len(tokens) == 2 and tokens[1] == ":":
                current = _identity(tokens[0])
                if current in self.blocks:
                    raise ValueError("duplicate compiled basic block")
                self.blocks[current] = []
                continue
            if current is None:
                raise ValueError("explicit compiled entry label required")
            self.blocks[current].append(tokens)
            if len(tokens) > 2 and tokens[0].startswith("%") and tokens[1] == "=":
                if tokens[0] in self.defs:
                    raise ValueError("duplicate compiled SSA definition")
                self.defs[tokens[0]], self.locations[tokens[0]] = tokens[2:], current
        self.succ, self.pred = {}, {name: set() for name in self.blocks}
        for name, rows in self.blocks.items():
            if not rows:
                raise ValueError("empty compiled block")
            end = rows[-1]
            if end[0] == "br":
                targets = [_identity(end[i + 1]) for i, word in enumerate(end[:-1]) if word == "label"]
                if len(targets) not in (1, 2):
                    raise ValueError("ordinary branch required")
            elif end[0] == "ret":
                targets = []
            else:
                raise ValueError("unsupported compiled control flow")
            self.succ[name] = targets
            for target in targets:
                if target not in self.pred:
                    raise ValueError("unknown compiled branch target")
                self.pred[target].add(name)
        entry = next(iter(self.blocks))
        reachable, todo = set(), [entry]
        while todo:
            node = todo.pop()
            if node not in reachable:
                reachable.add(node)
                todo.extend(self.succ[node])
        if reachable != set(self.blocks):
            raise ValueError("unreachable compiled blocks require separate proof")
        self.dom = {n: ({entry} if n == entry else set(reachable)) for n in reachable}
        changed = True
        while changed:
            changed = False
            for node in reachable - {entry}:
                new = {node} | set.intersection(*(self.dom[p] for p in self.pred[node]))
                if new != self.dom[node]:
                    self.dom[node], changed = new, True

    def pointer(self, name, dtype):
        op = self.defs.get(name)
        if op is None:
            return name, "0"
        tail = _plain(op, "getelementptr", dtype)
        if len(tail) != 6 or tail[:2] != [",", "ptr"] or tail[3:5] != [",", "i64"]:
            # Element type, one borrowed pointer and one integer index only.
            raise ValueError("one-dimensional scalar GEP required")
        return tail[2], tail[5]

    def load(self, name, dtype):
        op = self.defs.get(name, [])
        if len(op) < 5 or op[:4] != ["load", dtype, ",", "ptr"]:
            raise ValueError("ordinary typed scalar load required")
        if op[5:] and not (len(op) == 8 and op[5:7] == [",", "align"] and op[7].isdigit()):
            raise ValueError("unsupported scalar load properties")
        return self.pointer(op[4], dtype)

    def index_range(self, name, block, active=frozenset()):
        if not name.startswith("%"):
            value = int(name)
            return value, value
        if name in active or name not in self.defs:
            raise ValueError("cyclic or unknown scale index")
        op, active = self.defs[name], active | {name}
        if op[0] == "phi":
            if len(op) != 13 or op[:3] != ["phi", "i64", "["]:
                raise ValueError("two-input ordinary index phi required")
            # Parse bracket pairs using tokens, including their comma separator.
            incoming = []
            for i, token in enumerate(op):
                if token == "[":
                    if op[i + 2] != "," or op[i + 4] != "]":
                        raise ValueError("malformed induction phi")
                    incoming.append((op[i + 1], _identity(op[i + 3])))
            header = self.locations[name]
            if len(incoming) != 2 or {p for _, p in incoming} != self.pred[header]:
                raise ValueError("complete induction predecessors required")
            seed = [row for row in incoming if row[0] == "0"]
            back = [row for row in incoming if row[0] != "0"]
            if len(seed) != 1 or len(back) != 1:
                raise ValueError("zero-seeded increasing induction required")
            inc, predecessor = back[0]
            tail = _plain(self.defs.get(inc, []), "add", "i64")
            if len(tail) != 3 or tail[:2] != [name, ","] or not tail[2].isdigit():
                raise ValueError("literal positive induction step required")
            step = int(tail[2])
            if (
                step <= 0
                or self.locations.get(inc) != predecessor
                or self.succ[predecessor] != [header]
                or header not in self.dom[predecessor]
            ):
                raise ValueError("unproved induction backedge")
            branch = self.blocks[header][-1]
            if len(branch) != 9 or branch[:2] != ["br", "i1"]:
                raise ValueError("induction header bound branch required")
            comparison = self.defs.get(branch[2], [])
            if self.locations.get(branch[2]) != header:
                raise ValueError("induction comparison must belong to its header")
            if len(comparison) != 6 or comparison[:4] != ["icmp", "slt", "i64", name] or comparison[4] != ",":
                raise ValueError("signed positive induction bound required")
            bound = int(comparison[5])
            true = _identity(branch[5])
            if not 0 < bound < 2**63 - step or true not in self.dom[block]:
                raise ValueError("scale access not dominated by induction bound")
            return 0, ((bound - 1) // step) * step
        if op[0] not in ("add", "mul"):
            raise ValueError("unsupported scale index arithmetic")
        tail = _plain(op, op[0], "i64")
        if len(tail) != 3 or tail[1] != ",":
            raise ValueError("ordinary scalar integer index operation required")
        a, b = (self.index_range(v, block, active) for v in (tail[0], tail[2]))
        if op[0] == "add":
            result = a[0] + b[0], a[1] + b[1]
        else:
            products = [x * y for x in a for y in b]
            result = min(products), max(products)
        if not -(2**63) <= result[0] <= result[1] < 2**63:
            raise ValueError("scale index arithmetic may overflow")
        return result


def bind_finite_scale_helpers(
    source, *, routes, observers, effects, immutable_inputs=False, fresh_disjoint_output=False
):
    """Bind typed original casts/products to every physical scale read.

    Ownership facts are supplied by normal tensor bufferization, never inferred
    from pointer values. Unknown pointer escapes, calls, output aliasing or
    unbounded scale accesses refuse before any source rewrite.
    """
    if not routes:
        return ()
    effects.validate()
    if immutable_inputs is not True or fresh_disjoint_output is not True:
        raise ValueError("explicit immutable inputs/fresh disjoint output required")
    tokens, results = _tokens(source), []
    if any(
        t.text == "strictfp"
        or t.text.startswith("@llvm.experimental.constrained.")
        or t.text in ("fast", "nnan", "ninf", "reassoc", "contract", "nsz", "afn", "arcp")
        for t in tokens
    ):
        raise ValueError("strict/relaxed compiled floating context refuses finite preparation")
    bodies = _functions(tokens)
    grouped = {}
    for route in routes:
        grouped.setdefault(route["function_body_index"], []).append(route)
    for index, selected in grouped.items():
        body = bodies[index]
        start = max(t.start for t in tokens if t.text == "define" and t.start < body[0].start)
        header = [t.text for t in tokens if start <= t.start < body[0].start]
        names = [i for i, word in enumerate(header) if word.startswith("@")]
        if len(names) != 1 or header[names[0] - 1] != "void":
            raise ValueError("ordinary void helper ABI required")
        symbol = _identity(header[names[0]])
        a, b = header.index("("), header.index(")")
        args = tuple(header[a + 2 : b : 3])
        if header[a + 1 : b] != [item for j, arg in enumerate(args) for item in (([","] if j else []) + ["ptr", arg])]:
            raise ValueError("plain borrowed pointer helper ABI required")
        parsed = _Body(source, body)
        arithmetic = {}
        for i in range(len(body)):
            op = _extended_instruction(body, i)
            if op is not None:
                arithmetic[op.result] = op
        scans, integer_roots = {}, set()
        for route in selected:
            matches = [
                p
                for p in observers
                if p.quant_factor_bits == route["quant_factor_bits"]
                and p.expression.canonical_sha256 == route["source_expression_sha256"]
            ]
            if len(matches) != 1:
                raise ValueError("ambiguous typed observer preparation")
            observer = matches[0]
            validate_closed_scalar_observer(observer)
            for role, value in (("input", observer.cut), ("up", observer.up)):
                proof = prove_broadcast_scaled_integer_finite(value, effects=effects)
                if len(proof.scale_axes) != 1:
                    raise ValueError("one-dimensional broadcast scale preparation required")
                count = math.prod(proof.operation.inputs[proof.scale_operand].type.get_shape())
                if not 0 < count < 2**61:
                    raise ValueError("finite scale span exceeds ordinary signed pointer range")
                final = arithmetic.get(route[role])
                if final is None or final.opcode != "fmul" or final.dtype != "float":
                    raise ValueError("original compiled final scale multiply required")
                candidates = []
                for scale, current in (final.operands, final.operands[::-1]):
                    try:
                        root, offset = parsed.load(scale, "float")
                        words = []
                        while current in arithmetic and arithmetic[current].opcode == "fmul":
                            op = arithmetic[current]
                            if op.dtype != "float":
                                raise ValueError("original producer precision changed")
                            literals = [v for v in op.operands if not v.startswith("%")]
                            if len(literals) != 1:
                                raise ValueError("single literal producer multiply required")
                            words.append(_literal(literals[0], "float")[2])
                            current = next(v for v in op.operands if v != literals[0])
                        conversion = arithmetic[current]
                        if (
                            conversion.opcode != "sitofp"
                            or conversion.dtype != f"i{proof.domain.integer_bits}"
                            or conversion.output_type != "float"
                            or tuple(reversed(words)) != proof.domain.constant_words
                        ):
                            raise ValueError("compiled original conversion/constants changed")
                        integer_root, _ = parsed.load(conversion.operands[0], conversion.dtype)
                        lo, hi = parsed.index_range(offset, parsed.locations[scale])
                        if root not in args or integer_root not in args or not 0 <= lo <= hi < count:
                            raise ValueError("scale load not covered by immutable scan")
                        candidates.append((root, integer_root))
                    except (ValueError, KeyError, IndexError):
                        continue
                if len(candidates) != 1:
                    raise ValueError("compiled scaled integer producer not proved")
                root, integer_root = candidates[0]
                integer_roots.add(integer_root)
                scan = FiniteScaleScan(args.index(root), count, proof)
                if root in scans and (scans[root].count != count or scans[root].proof.domain != proof.domain):
                    raise ValueError("inconsistent source scale domain")
                scans[root] = scan
        readonly = set(scans) | integer_roots
        output = set(args) - readonly
        if len(output) != 1 or readonly & output:
            raise ValueError("one fresh output and complete input mapping required")
        for block, rows in parsed.blocks.items():
            for words in rows:
                op = words[2:] if len(words) > 2 and words[1] == "=" else words
                if op[0] in ("load", "store"):
                    ptrpos = op.index("ptr") + 1
                    root, _ = parsed.pointer(op[ptrpos], op[1])
                    if op[0] == "store" and root not in output or op[0] == "load" and root not in readonly:
                        raise ValueError("unproved input mutation or output initializer read")
                elif op[0] == "call":
                    callees = [w for w in op if w.startswith("@")]
                    if callees not in (["@llvm.maximum.f32"], ["@llvm.minimum.f32"], ["@llvm.fma.f32"]):
                        raise ValueError("unknown compiled effects in finite-input consumer")
                elif op[0] not in (
                    "getelementptr",
                    "phi",
                    "add",
                    "mul",
                    "icmp",
                    "br",
                    "ret",
                    "sitofp",
                    "fptosi",
                    "fmul",
                    "fdiv",
                    "fadd",
                    "fsub",
                    "fneg",
                    "shl",
                    "bitcast",
                    "fcmp",
                    "select",
                    "and",
                    "or",
                    "xor",
                    "trunc",
                ):
                    raise ValueError("unsupported compiled helper operation/effect")
                # Pointer values may occur only as borrowed GEP bases or actual
                # load/store addresses, never stored, returned or inspected.
                for arg in args:
                    if arg in op and not (op[0] == "getelementptr" and op[op.index("ptr") + 1] == arg):
                        raise ValueError("borrowed helper pointer escapes exact GEP use")
        stop = next(t.end for t in tokens if t.text == "}" and t.start > body[-1].end)
        body_source = source[start:stop]
        retained_routes = tuple({**row, "function_body_index": 0} for row in selected)
        results.append(
            FiniteScaleHelper(
                symbol,
                args,
                tuple(scans.values()),
                hashlib.sha256(source.encode()).hexdigest(),
                hashlib.sha256(source[body[0].start : body[-1].end].encode()).hexdigest(),
                source,
                body_source,
                retained_routes,
                tuple(observers),
                effects,
            )
        )
    return tuple(results)


def validate_finite_scale_helper(binding):
    if not isinstance(binding, FiniteScaleHelper):
        raise ValueError("typed compiled finite-scale binding required")
    for scan in binding.scans:
        validate_broadcast_scaled_integer_finite(scan.proof)
    if (
        hashlib.sha256(binding._source.encode()).hexdigest() != binding.source_sha256
        or binding._body_source not in binding._source
    ):
        raise ValueError("compiled source/parent context changed")
    current = bind_finite_scale_helpers(
        binding._body_source,
        routes=binding._routes,
        observers=binding._observers,
        effects=binding._effects,
        immutable_inputs=True,
        fresh_disjoint_output=True,
    )
    if len(current) != 1 or (
        current[0].symbol,
        current[0].arguments,
        current[0].body_sha256,
        [(s.argument, s.count, s.proof.domain) for s in current[0].scans],
    ) != (
        binding.symbol,
        binding.arguments,
        binding.body_sha256,
        [(s.argument, s.count, s.proof.domain) for s in binding.scans],
    ):
        raise ValueError("compiled finite-scale semantic fields changed")


def emit_finite_scale_helper(binding, *, wrapper_symbol, finite_symbol, fallback_symbol):
    """Portable guard preserving the unchanged helper on any scan refusal.

    All symbols are explicit ordinary C identifiers. Stable RNE/gradual mode is
    the external provider's obligation; this wrapper contains no FP operations.
    Readonly scales can alias each other, but must remain immutable and disjoint
    from fresh output throughout both scans and the synchronous helper call.
    """
    validate_finite_scale_helper(binding)
    for symbol in (wrapper_symbol, finite_symbol, fallback_symbol):
        if not isinstance(symbol, str) or not symbol.isascii() or not symbol.isidentifier():
            raise ValueError("explicit C helper identifier required")
    if len({wrapper_symbol, finite_symbol, fallback_symbol}) != 3:
        raise ValueError("helper symbol collision")
    declarations, tests = [], []
    for index, scan in enumerate(binding.scans):
        symbol = f"{wrapper_symbol}_scan_{index}"
        declarations.append(
            emit_finite_scale_scan(scan.proof.domain, symbol=symbol).replace(
                f"int {symbol}", f"static inline int {symbol}", 1
            )
        )
        tests.append(f"{symbol}((const float*)a{scan.argument},{scan.count},1)")
    args = ",".join(f"void*a{i}" for i in range(len(binding.arguments)))
    types = ",".join("void*" for _ in binding.arguments)
    values = ",".join(f"a{i}" for i in range(len(binding.arguments)))
    return (
        "\n".join(declarations)
        + f"\nextern void {finite_symbol}({types});\nextern void {fallback_symbol}({types});\nvoid {wrapper_symbol}({args}){{if({'&&'.join(tests)}){finite_symbol}({values});else {fallback_symbol}({values});}}\n"
    )
