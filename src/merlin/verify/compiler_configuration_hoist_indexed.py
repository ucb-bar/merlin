"""Current-only retained integer data admission for the indexed hoist model.

Full actual expression tokens and loop metadata are preserved. Configuration
field indices and values stay syntactic integer constants. Host interval and
congruence analysis only narrows admission; it is trusted Python analysis, not a
Lean-derived native/encoding theorem. Models still quantifies over EVERY index.
"""

from __future__ import annotations

import copy
import json
from math import gcd

from .configuration_hoist_bridge import (
    Unsupported,
    call_shape,
    canonical,
    constant,
    decode_effect,
    node,
    program,
    project_current,
    sequence,
)


def _expr(value, kind, fields):
    if (
        type(value) is not dict
        or set(value) != {"class", "fields"}
        or value["class"] != "merlin.sched.ir.expr." + kind
        or type(value["fields"]) is not dict
        or set(value["fields"]) != set(fields)
    ):
        raise Unsupported("unsupported exact indexed integer expression")
    return value["fields"]


def _literal_domain(value, domains, *, depth=0):
    """C index64 renders every literal, including both arms of a Select."""
    if depth > 64:
        raise Unsupported("indexed integer expression depth budget exceeded")
    kind = value.get("class", "") if type(value) is dict else ""
    literals, children = [], []
    if kind.endswith(".Const"):
        literals.append(constant(value))
    elif kind.endswith(".Var"):
        field = _expr(value, "Var", ("name",))
        if field["name"] not in domains:
            raise Unsupported("unbound actual indexed integer variable")
    elif kind.endswith(".Sum"):
        children = sequence(_expr(value, "Sum", ("terms",))["terms"])
        if len(children) > 4096:
            raise Unsupported("indexed integer expression width budget exceeded")
    elif kind.endswith(".Scaled"):
        field = _expr(value, "Scaled", ("expr", "factor"))
        literals.append(field["factor"])
        children.append(field["expr"])
    elif kind.endswith(".Select"):
        field = _expr(value, "Select", ("var", "equals", "then", "other"))
        if field["var"] not in domains:
            raise Unsupported("unbound actual indexed integer selector")
        literals.append(field["equals"])
        children.extend((field["then"], field["other"]))
    else:
        raise Unsupported("unsupported indexed integer expression form")
    if any(type(number) is not int or not -(1 << 63) <= number < (1 << 63) for number in literals):
        raise Unsupported("indexed integer literal exceeds signed index64 rendering domain")
    for child in children:
        _literal_domain(child, domains, depth=depth + 1)


def bounds(value, domains, *, depth=0):
    """Conservative min/max and congruence class, without dynamic expansion."""
    if depth == 0:
        _literal_domain(value, domains)
    if depth > 64:
        raise Unsupported("indexed integer expression depth budget exceeded")
    kind = value.get("class", "") if type(value) is dict else ""
    if kind.endswith(".Const"):
        number = constant(value)
        result = number, number, 0, number
    elif kind.endswith(".Var"):
        field = _expr(value, "Var", ("name",))
        if field["name"] not in domains:
            raise Unsupported("unbound actual indexed integer variable")
        low, high = domains[field["name"]]
        result = (low, high, 0, low) if low == high else (low, high, 1, 0)
    elif kind.endswith(".Sum"):
        field = _expr(value, "Sum", ("terms",))
        low = high = divisor = remainder = 0
        terms = sequence(field["terms"])
        if len(terms) > 4096:
            raise Unsupported("indexed integer expression width budget exceeded")
        for term in terms:
            a, b, d, r = bounds(term, domains, depth=depth + 1)
            low, high, divisor, remainder = low + a, high + b, gcd(divisor, d), remainder + r
            if not -(1 << 63) <= low <= high < (1 << 63):
                raise Unsupported("indexed integer sum can exceed signed index64")
        result = low, high, divisor, remainder
    elif kind.endswith(".Scaled"):
        field = _expr(value, "Scaled", ("expr", "factor"))
        if type(field["factor"]) is not int:
            raise Unsupported("exact integer factor required")
        low, high, divisor, remainder = bounds(field["expr"], domains, depth=depth + 1)
        scale = field["factor"]
        low, high = sorted((low * scale, high * scale))
        result = low, high, abs(scale) * divisor, scale * remainder
    elif kind.endswith(".Select"):
        field = _expr(value, "Select", ("var", "equals", "then", "other"))
        if field["var"] not in domains or type(field["equals"]) is not int:
            raise Unsupported("exact scoped integer selector required")
        low, high = domains[field["var"]]
        parts = []
        if low <= field["equals"] <= high:
            parts.append(
                bounds(field["then"], {**domains, field["var"]: (field["equals"], field["equals"])}, depth=depth + 1)
            )
        if low != high or low != field["equals"]:
            parts.append(bounds(field["other"], domains, depth=depth + 1))
        a, b, d, r = parts[0]
        for x, y, e, s in parts[1:]:
            a, b, d = min(a, x), max(b, y), gcd(gcd(d, e), r - s)
        result = a, b, d, r
    else:
        raise Unsupported("unsupported indexed integer expression form")
    if not -(1 << 63) <= result[0] <= result[1] < (1 << 63):
        raise Unsupported("indexed integer expression can exceed signed index64")
    return result


def _configuration_arguments(declaration):
    result = set()
    for kind in ("reads", "writes", "clobbers"):
        for item in declaration[kind]:
            for key in ("index", "value"):
                binding = item.get(key)
                if type(binding) is dict and set(binding) == {"argument"}:
                    result.add(binding["argument"])
    return result


def _substitute(call, isa, declarations, domains, records):
    fields, pairs = call_shape(call)
    # This profile is consumed with signed index64 C expressions. Check every
    # subtree before any constant/unknown-declaration fast path can return.
    for _, value in pairs:
        _literal_domain(value, domains)
    declaration = declarations.get(fields["instr"])
    if declaration is None:
        return call
    # Existing admission rejects these annotations; do not turn unsupported
    # completion semantics into a body frame through symbolic data analysis.
    if sequence(fields["sets"]) or sequence(fields["assumes"]):
        return call
    instruction = next((row for row in isa["instructions"] if row["name"] == fields["instr"]), None)
    if instruction is None or instruction["results"]:
        return call
    arguments = {row["name"]: row for row in (*instruction["operands"], *instruction["immediates"])}
    encoded = {row["argument"] for row in instruction["fields"] if row["argument"] is not None}
    if {name for name, _ in pairs} != encoded:
        raise Unsupported("actual indexed call must cover exact selected encoding arguments")
    sensitive = _configuration_arguments(declaration)
    replacements = {}
    for key, value in pairs:
        try:
            constant(value)
            continue
        except Unsupported:
            pass
        if key in sensitive:
            raise Unsupported("configuration field index/value must remain a syntactic checked constant")
        if declaration["pure_write"]:
            raise Unsupported("pure setter operands must remain entirely constant")
        if key not in arguments or arguments[key]["type"]["kind"] != "integer":
            raise Unsupported("indexed data requires an exact declared integer argument")
        low, high, divisor, remainder = bounds(value, domains)
        argument = arguments[key]
        alignment = argument["alignment"]
        for field in instruction["fields"]:
            if field["argument"] != key:
                continue
            width = min(field["width"], argument["type"]["bits"])
            signed = argument["type"]["signed"]
            lower = -(1 << (width - 1)) if signed else 0
            upper = (1 << (width - int(signed))) - 1
            if not lower <= low <= high <= upper or remainder % alignment or divisor % alignment:
                raise Unsupported("actual indexed data can exceed selected encoded range/alignment")
        # This representative is used ONLY to reuse the static field-declaration
        # resolver: by construction no field index/value reads this argument.
        # Original expression bytes are restored to every emitted atom below.
        representative = low
        replacements[key] = {"class": "merlin.sched.ir.expr.Const", "fields": {"value": representative}}
        records.append(
            {
                "instruction": fields["instr"],
                "argument": key,
                "expression": value,
                "range": [low, high],
                "alignment": alignment,
                "trusted_host_analysis": True,
            }
        )
    if not replacements:
        return call
    result = copy.deepcopy(call)
    for pair in result["fields"]["args"]["tuple"]:
        key, _ = sequence(pair)
        if key in replacements:
            pair["tuple"][1] = replacements[key]
    return result


def program_current(value, isa, effects, *, analyses, allow_multiple=False):
    if type(allow_multiple) is not bool:
        raise Unsupported("explicit multiple-loop parsing profile required")
    original_calls = []
    substituted = copy.deepcopy(value)
    kernel = node(substituted, "Kernel", ("name", "args", "body", "attrs"))
    for position, statement in enumerate(sequence(kernel["body"])):
        if statement.get("class") == "merlin.sched.ir.kernel.Call":
            original_calls.append(copy.deepcopy(statement))
            # Outer calls keep static effects but share this profile's
            # eager signed-index64 literal admission, including constants.
            for _, argument in call_shape(statement)[1]:
                _literal_domain(argument, {})
            decode_effect(statement, isa, effects)
            continue
        loop = node(statement, "Loop", ("var", "extent", "body", "iter_args", "yields", "results"))
        if type(loop["extent"]) is not int or not 0 < loop["extent"] <= 1_000_000:
            raise Unsupported("positive bounded indexed loop domain required")
        domains = {loop["var"]: (0, loop["extent"] - 1)}
        for index, call in enumerate(sequence(loop["body"])):
            original_calls.append(copy.deepcopy(call))
            loop["body"]["tuple"][index] = _substitute(call, isa, effects, domains, analyses)
    projected, descriptor = program(substituted, isa, effects, allow_multiple=allow_multiple)
    if len(projected["atoms"]) != len(original_calls):
        raise ValueError("actual indexed atom roster changed")
    for atom, call in zip(projected["atoms"], original_calls, strict=True):
        atom["token"] = canonical(call)
    return projected, descriptor


def project(source, final, *, isa, effects):
    analyses = []

    def actual_program(value, isa, effects):
        return program_current(value, isa, effects, analyses=analyses)

    projection = project_current(source, final, isa=isa, effects=effects, program_decoder=actual_program)
    projection.update(
        indexed_data_analysis=analyses,
        indexed_data_analysis_is_trusted_host=True,
        all_index_model_premise_retained=True,
        native_or_encoder_fidelity_proved=False,
    )
    return projection


def proposal_adapter(selected_contracts, qualification_identity):
    """Explicit untrusted IV-data proposal, independently checked by the gate.

    The original selected adapter remains unchanged. This helper permits only
    integer Expr data unrelated to every configuration field index/value; moved
    pure setters stay completely constant. A legal representative is used only
    to obtain a proposal from the original selected adapter. Actual full tokens,
    loop domains, expressions/ranges and effects are independently reprojected
    by canonical intake. Neither this helper nor its label grants approval.
    """
    from dataclasses import replace

    from merlin.compile.scheduling.configuration import ConfigurationAdapter
    from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
    from merlin.sched.ir import Call, Const
    from merlin.sched.ir.expr import Expr

    from .compiler_configuration_kernel import tree

    if type(selected_contracts) is not SelectedSchedulingContracts:
        raise ValueError("proposal requires original typed selected ISA/effect contracts")
    selected_contracts.revalidate()
    ordinary = selected_contracts.configuration_adapter(qualification_identity)
    isa = selected_contracts.isa.record()
    declarations = json.loads(selected_contracts.effects_json)
    instructions = {row["name"]: row for row in isa["instructions"]}

    def propose(call):
        selected_contracts.revalidate()
        if type(call) is not Call:
            return None
        declaration, instruction = declarations.get(call.instr), instructions.get(call.instr)
        if declaration is None or instruction is None:
            return ordinary.project(call)
        if instruction["results"]:
            return None
        sensitive = _configuration_arguments(declaration)
        arguments = {row["name"]: row for row in (*instruction["operands"], *instruction["immediates"])}
        rewritten = []
        for key, value in call.args:
            if type(value) is Const:
                if type(value.value) is not int:
                    return None
                rewritten.append((key, value))
                continue
            if (
                key in sensitive
                or declaration["pure_write"]
                or key not in arguments
                or arguments[key]["type"]["kind"] != "integer"
                or not isinstance(value, Expr)
            ):
                return None
            # All encoded signed/unsigned integer fields include aligned zero.
            # This value is a proposal-only placeholder, never a source token or
            # a range/field admission witness. The canonical worker checks the
            # untouched actual expression using its actual enclosing loop.
            if type(value).__module__ != "merlin.sched.ir.expr":
                return None
            tree(value)  # reject incomplete/foreign expression dataclasses
            rewritten.append((key, Const(0)))
        result = ordinary.project(replace(call, args=tuple(rewritten)))
        selected_contracts.revalidate()
        return result

    return ConfigurationAdapter(propose, selected_contracts.hardware_identity, qualification_identity)
