"""Explicit source-bound scale scans before repeated finite-input observers.

The portable wrapper is admitted from exact source arithmetic and bounded LLVM
scale loads. Immutable input storage and disjoint fresh output are mandatory
caller ownership contracts. No target mode test, ISA or automatic policy lives
here; the caller retains the original source helper outside stable RNE.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from merlin.compiler.lower.model.late_quant_rne import _functions, _identity, _tokens
from merlin.compiler.lower.model.scaled_integer_finite import (
    BroadcastScaledIntegerFiniteProof,
    emit_finite_scale_scan,
    prove_broadcast_scaled_integer_finite,
    validate_broadcast_scaled_integer_finite,
)
from merlin.compiler.lower.model.source_expression_interval import validate_closed_scalar_observer
from merlin.compiler.lower.model.source_expression_interval_llvm import _extended_instruction, _literal


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
