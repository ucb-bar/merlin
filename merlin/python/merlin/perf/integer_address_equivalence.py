"""Narrow whole-IR identity check modulo unflagged integer affine identities.

This compares retained parsed modules; it neither imports compilers nor runs a
model. Floating operations, memory effects, calls, inline assembly, CFG edges,
and unknown operations retain their exact order/types/properties/operands.
Unsupported differences are UNKNOWN, never inferred numerical mismatches.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time

from xdsl.dialects.builtin import IntegerAttr, IntegerType


class Unsupported(ValueError):
    pass


@dataclass(frozen=True)
class Affine:
    width: int
    constant: int
    terms: tuple
    poison: frozenset

    def key(self):
        return ("integer", self.width, self.constant, self.terms, tuple(sorted(self.poison)))


def _width(value):
    return value.type.width.data if isinstance(value.type, IntegerType) else None


def _combine(a, b, coefficient=1):
    if not isinstance(a, Affine) or not isinstance(b, Affine) or a.width != b.width:
        raise Unsupported("integer operand widths are not identical")
    modulus = 1 << a.width
    terms = dict(a.terms)
    for atom, amount in b.terms:
        terms[atom] = (terms.get(atom, 0) + coefficient * amount) % modulus
    return Affine(a.width, (a.constant + coefficient * b.constant) % modulus,
        tuple(sorted((atom, amount) for atom, amount in terms.items() if amount)), a.poison | b.poison)


def _scale(value, amount, poison=frozenset()):
    modulus = 1 << value.width
    return Affine(value.width, value.constant * amount % modulus,
        tuple((atom, coefficient * amount % modulus) for atom, coefficient in value.terms
              if coefficient * amount % modulus), value.poison | poison)


def _attributes(op):
    return (tuple(sorted((key, str(value)) for key, value in op.properties.items())),
            tuple(sorted((key, str(value)) for key, value in op.attributes.items())))


def _normal_form(module, *, deadline, max_operations, defined_integer_domain=False, pruned=frozenset()):
    operations = list(module.walk())
    if len(operations) > max_operations:
        raise Unsupported("operation bound exceeded")
    operation_ids = {op: index for index, op in enumerate(operations)}
    blocks = [block for op in operations for region in op.regions for block in region.blocks]
    block_ids = {block: index for index, block in enumerate(blocks)}
    values = {}
    normalized_counts = {}
    retained_count = 0
    semantic_nodes = []

    def atom(value, identity):
        width = _width(value)
        if width is None:
            return ("ssa", identity, str(value.type))
        if not 0 < width <= 256:
            raise Unsupported("unsupported integer width")
        return Affine(width, 0, ((identity, 1),), frozenset([identity]))

    for block in blocks:
        for index, argument in enumerate(block.args):
            values[argument] = atom(argument, f"b{block_ids[block]}a{index}")

    def operand(value):
        if value not in values:
            raise Unsupported("unresolved or out-of-order SSA definition")
        return values[value]

    def safe_arithmetic(op):
        # These attributes are provenance tags only. Their omission is explicitly
        # scoped to deleted, pure integer operations, not retained effects.
        if set(op.attributes) - {"merlin.global_task"}:
            return False
        for key, value in op.properties.items():
            if key != "overflowFlags" or not isinstance(value, IntegerAttr) or value.value.data != 0:
                raise Unsupported(f"unsupported integer arithmetic property {key}")
        return True

    def normalize(op):
        if len(op.results) != 1 or _width(op.results[0]) is None:
            return None
        width = _width(op.results[0])
        if not 0 < width <= 256:
            raise Unsupported("unsupported integer width")
        if op.name == "llvm.mlir.constant":
            value = op.properties.get("value")
            if (isinstance(value, IntegerAttr) and set(op.properties) == {"value"}
                    and not (set(op.attributes) - {"merlin.global_task"}) and not op.operands):
                return Affine(width, value.value.data % (1 << width), (), frozenset())
            return None
        if op.name not in {"llvm.add", "llvm.sub", "llvm.mul", "llvm.udiv", "llvm.urem"}:
            return None
        if len(op.operands) != 2 or op.regions or op.successors or not safe_arithmetic(op):
            return None
        a, b = map(operand, op.operands)
        if not isinstance(a, Affine) or not isinstance(b, Affine) or a.width != width or b.width != width:
            raise Unsupported("integer arithmetic type mismatch")
        if op.name == "llvm.add":
            return _combine(a, b)
        if op.name == "llvm.sub":
            return _combine(a, b, -1)
        if op.name == "llvm.mul":
            if not a.terms:
                return _scale(b, a.constant, a.poison)
            if not b.terms:
                return _scale(a, b.constant, b.poison)
        if not b.terms and b.constant == 1 and not b.poison:
            if op.name == "llvm.udiv":
                return a
            if op.name == "llvm.urem":
                return Affine(width, 0, (), a.poison)
        if (not a.terms and not b.terms and b.constant
                and (defined_integer_domain or (not a.poison and not b.poison))):
            if op.name in {"llvm.udiv", "llvm.urem"}:
                value = a.constant // b.constant if op.name == "llvm.udiv" else a.constant % b.constant
                return Affine(width, value, (), frozenset())
        return None

    def key(value):
        if isinstance(value, Affine):
            if defined_integer_domain:
                return ("integer", value.width, value.constant, value.terms, ())
            return value.key()
        return value

    def references(value):
        if value[0] == "integer":
            return {name for name, _ in value[3]} | set(value[4])
        return {value[1]} if value[0] == "ssa" else set()

    def total_pure(op):
        if (op.name not in {"llvm.add", "llvm.sub", "llvm.mul", "llvm.udiv", "llvm.urem"}
                or len(op.results) != 1 or _width(op.results[0]) is None
                or op.regions or op.successors or len(op.operands) != 2 or not safe_arithmetic(op)):
            return False
        if op.name in {"llvm.udiv", "llvm.urem"}:
            divisor = operand(op.operands[1])
            return (isinstance(divisor, Affine) and not divisor.terms and divisor.constant != 0
                    and (defined_integer_domain or not divisor.poison))
        return True

    def operation(op, identity):
        nonlocal retained_count
        if time.monotonic() > deadline:
            raise Unsupported("analysis deadline exceeded")
        if op.name in {"llvm.mlir.undef", "llvm.mlir.poison"}:
            raise Unsupported("explicit undef/poison is outside the defined-integer domain")
        normalized = normalize(op)
        if normalized is not None:
            values[op.results[0]] = normalized
            normalized_counts[op.name] = normalized_counts.get(op.name, 0) + 1
            return None
        operands = tuple(key(operand(value)) for value in op.operands)
        if op in pruned:
            if not total_pure(op):
                raise Unsupported("pruned operation lacks integer totality proof")
            for index, result in enumerate(op.results):
                values[result] = atom(result, f"dead{operation_ids[op]}r{index}")
            return None
        for index, result in enumerate(op.results):
            values[result] = atom(result, identity + f"r{index}")
        semantic_nodes.append((op, tuple(identity + f"r{index}" for index in range(len(op.results))),
            set().union(*(references(value) for value in operands)), total_pure(op)))
        regions = []
        for region in op.regions:
            region_blocks = []
            for block in region.blocks:
                body = []
                for child in block.ops:
                    row = operation(child, f"b{block_ids[block]}o{len(body)}")
                    if row is not None:
                        body.append(row)
                region_blocks.append((block_ids[block], tuple(str(arg.type) for arg in block.args), body))
            regions.append(region_blocks)
        retained_count += 1
        return (op.name, _attributes(op), tuple(str(result.type) for result in op.results),
                operands, tuple(block_ids[block] for block in op.successors), regions)

    result = operation(module, "root")
    producers = {name: node for node in semantic_nodes for name in node[1]}
    live = set().union(*(node[2] for node in semantic_nodes if not node[3]))
    queue = list(live)
    while queue:
        node = producers.get(queue.pop())
        if node is not None and node[3]:
            extra = node[2] - live
            live.update(extra)
            queue.extend(extra)
    dead = {op for op, results, _, pure in semantic_nodes if pure and not (set(results) & live)}
    return result, {"input_operations": len(operations), "blocks": len(blocks),
                    "retained_operations": retained_count, "normalized_operations": normalized_counts,
                    "pruned_total_integer_operations": len(pruned)}, dead


def compare_integer_address_identity(before_module, after_module, *, timeout_s=30.0,
                                     max_operations=1000000, defined_integer_domain=False) -> dict:
    """Prove exact retained program structure modulo narrow integer identities.

    Callers bind these parsed modules to exact emitted bytes. The report's normal
    form digests are not substitutes for source/LLVM/compiler provenance pins.
    Poison dependence is retained through cancellation and multiplication by zero.
    """
    result = {"schema": "integer_address_identity_v1", "status": "UNKNOWN",
        "proof_scope": "ordered LLVM structure modulo unflagged affine bit-vector integer identities",
        "floating_arithmetic_normalized": False, "poison_dependencies_preserved": not defined_integer_domain,
        "normalized_operation_metadata_ignored": ["merlin.global_task"],
        "source_numerics_qualified": False, "timing_benefit": "UNMEASURED",
        "llvm_undef_semantics_qualified": False,
        "defined_integer_domain_explicitly_assumed": defined_integer_domain,
        "domain_assumptions_discharged": False,
        "dead_integer_pruning": "semantic liveness; unflagged total integer ops only; constant nonzero divisors",
        "required_domain": "integer SSA operands denote stable defined bit-vector values; not LLVM undef",
        "full_model_executed": False}
    started = time.monotonic()
    try:
        if (before_module is None or after_module is None or timeout_s <= 0 or max_operations <= 0
                or type(defined_integer_domain) is not bool):
            raise Unsupported("missing parsed module or invalid bound")
        forms = []
        for label, module in (("before", before_module), ("after", after_module)):
            options = dict(deadline=started + timeout_s, max_operations=max_operations,
                           defined_integer_domain=defined_integer_domain)
            form, counts, dead = _normal_form(module, **options)
            if dead:
                form, counts, _ = _normal_form(module, **options, pruned=dead)
            forms.append(form)
            result[label] = {**counts, "normal_form_sha256": hashlib.sha256(
                json.dumps(form, separators=(",", ":")).encode()).hexdigest()}
        if forms[0] == forms[1]:
            result["status"] = "conditional_integer_identity"
        else:
            result["reason"] = "ordered non-integer operations, CFG, or canonical operands differ"
            def difference(a, b, path=()):
                if type(a) is not type(b):
                    return path, repr(a)[:250], repr(b)[:250]
                if isinstance(a, (tuple, list)):
                    if len(a) != len(b):
                        return path + ("length",), len(a), len(b)
                    for at, (left, right) in enumerate(zip(a, b)):
                        if left != right:
                            return difference(left, right, path + (at,))
                return path, repr(a)[:250], repr(b)[:250]
            result["first_difference"] = difference(*forms)
    except (Unsupported, KeyError, TypeError, ValueError, RecursionError) as exc:
        result["reason"] = str(exc)
    result["elapsed_seconds"] = time.monotonic() - started
    return result
