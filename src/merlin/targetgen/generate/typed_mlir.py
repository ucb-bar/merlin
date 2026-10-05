"""Checked ODS emission for reviewed, fully typed target dialect plans.

This is a deliberately small declarative vocabulary. Unknown types, effects,
attributes, and lowering requests fail before emitting a C++ package. Numerical
semantics and physical instruction encoding remain separate target obligations.
"""

from __future__ import annotations

import json
import copy
from typing import Any

from .target_repo import camel

_VALUE_TYPES = {
    "i1": "I1", "i8": "I8", "i16": "I16", "i32": "I32", "i64": "I64",
    "index": "Index", "f32": "F32", "f64": "F64", "ranked_tensor": "AnyRankedTensor",
}
_ATTR_TYPES = {"i32": "I32Attr", "i64": "I64Attr", "string": "StrAttr", "bool": "BoolAttr"}
_EFFECTS = {"read": "Read", "write": "Write", "allocate": "Allocate", "free": "Free"}
_CPP_KEYWORDS = set((
    "alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t char32_t "
    "class compl concept const consteval constexpr constinit const_cast continue co_await co_return co_yield "
    "decltype default delete do double dynamic_cast else enum explicit export extern false float for friend "
    "goto if inline int long mutable namespace new noexcept not not_eq nullptr operator or or_eq private "
    "protected public register reinterpret_cast requires return short signed sizeof static static_assert "
    "static_cast struct switch template this thread_local throw true try typedef typeid typename union "
    "unsigned using virtual void volatile wchar_t while xor xor_eq"
).split())


def _identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str) or not value or not value.isascii()
        or not (value[0].isalpha() or value[0] == "_")
        or any(not (char.isalnum() or char == "_") for char in value)
        or value in _CPP_KEYWORDS
    ):
        raise ValueError(f"{label} must be an ASCII identifier")
    return value


def _literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _cpp_integer(value: int) -> str:
    # Unary minus on the unsigned-magnitude INT64_MIN literal is not portable C++.
    return "(-9223372036854775807LL - 1)" if value == -(2**63) else str(value)


def _value_type(value: Any, dialect: str, cls: str, known_types: set[str]) -> str:
    if isinstance(value, str) and value in _VALUE_TYPES:
        return _VALUE_TYPES[value]
    prefix = f"!{dialect}."
    if isinstance(value, str) and value.startswith(prefix) and value[len(prefix):] in known_types:
        return f"{cls}_{camel(value[len(prefix):])}"
    raise ValueError(f"unsupported typed dialect value type {value!r}")


def _type_parameter_refs(
    signature: dict[str, Any], types: list[dict[str, Any]], dialect: str,
) -> dict[tuple[str, str], tuple[str, str]]:
    declared = {row["name"]: row for row in types}
    refs = {}
    for role in ("operands", "results"):
        for index, field in enumerate(signature[role]):
            source = field["source_type"]
            prefix = f"!{dialect}."
            if not source.startswith(prefix):
                continue
            name = source[len(prefix):]
            for param in declared[name]["parameters"]:
                container = f"getOperation()->get{'Operand' if role == 'operands' else 'Result'}({index})"
                expression = (
                    f"mlir::cast<{camel(name)}Type>({container}.getType())"
                    f".get{camel(param['name'])}()"
                )
                refs[(field["name"], param["name"])] = (
                    "integer" if param["kind"] == "unsigned" else "type", expression,
                )
    return refs


def _predicate_expression(
    node: Any, fields: dict[str, str], refs: dict[tuple[str, str], tuple[str, str]],
    *, depth: int = 0,
) -> tuple[str, str]:
    """Type-check and lower a bounded declarative attribute/type predicate to C++."""
    if depth > 32 or not isinstance(node, dict) or len(node) != 1:
        raise ValueError("predicate must be one bounded expression node")
    op, value = next(iter(node.items()))
    if op == "attr":
        if not isinstance(value, str) or value not in fields:
            raise ValueError("predicate references an undeclared attribute")
        kind = fields[value]
        attr = f'getOperation()->getAttrOfType<{"StringAttr" if kind == "string" else "BoolAttr" if kind == "bool" else "IntegerAttr"}>({_literal(value)})'
        return kind, f"{attr}.getValue()" if kind in {"string", "bool"} else f"{attr}.getInt()"
    if op == "type_param":
        if (
            not isinstance(value, dict) or set(value) != {"value", "name"}
            or not isinstance(value["value"], str) or not isinstance(value["name"], str)
        ):
            raise ValueError("predicate type_param requires value and name")
        key = (value["value"], value["name"])
        if key not in refs:
            raise ValueError("predicate references an undeclared value type parameter")
        return refs[key]
    if op == "integer":
        if type(value) is not int or not -(2**63) <= value < 2**63:
            raise ValueError("predicate integer is outside signed 64-bit range")
        return "integer", _cpp_integer(value)
    if op == "string":
        if not isinstance(value, str):
            raise ValueError("predicate string literal is malformed")
        return "string", _literal(value)
    if op == "boolean":
        if type(value) is not bool:
            raise ValueError("predicate Boolean literal is malformed")
        return "bool", str(value).lower()
    if op == "not":
        kind, expression = _predicate_expression(value, fields, refs, depth=depth + 1)
        if kind != "bool":
            raise ValueError("predicate not requires Boolean input")
        return "bool", f"(!({expression}))"
    if op in {"and", "or"}:
        if not isinstance(value, list) or len(value) < 2:
            raise ValueError(f"predicate {op} requires at least two arguments")
        parts = [_predicate_expression(item, fields, refs, depth=depth + 1) for item in value]
        if any(kind != "bool" for kind, _ in parts):
            raise ValueError(f"predicate {op} requires Boolean arguments")
        operator = " && " if op == "and" else " || "
        return "bool", "(" + operator.join(f"({expression})" for _, expression in parts) + ")"
    if op in {"implies", "eq", "lt", "le", "gt", "ge", "mod"}:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"predicate {op} requires exactly two arguments")
        left_kind, left = _predicate_expression(value[0], fields, refs, depth=depth + 1)
        right_kind, right = _predicate_expression(value[1], fields, refs, depth=depth + 1)
        if op == "implies":
            if left_kind != "bool" or right_kind != "bool":
                raise ValueError("predicate implication requires Boolean arguments")
            return "bool", f"((!({left})) || ({right}))"
        if op == "eq":
            if left_kind != right_kind and {left_kind, right_kind} != {"i32", "integer"} and {left_kind, right_kind} != {"i64", "integer"}:
                raise ValueError("predicate equality requires matching argument types")
            return "bool", f"(({left}) == ({right}))"
        if left_kind not in {"i32", "i64", "integer"} or right_kind not in {"i32", "i64", "integer"}:
            raise ValueError(f"predicate {op} requires integer arguments")
        if op == "mod" and (
            not isinstance(value[1], dict) or type(value[1].get("integer")) is not int
            or value[1]["integer"] <= 0
        ):
            raise ValueError("predicate modulo requires a positive literal divisor")
        operator = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "mod": "%"}[op]
        return ("integer" if op == "mod" else "bool"), f"(({left}) {operator} ({right}))"
    raise ValueError(f"unsupported predicate operator {op!r}")


def validate(plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize one MLIR-only plan; never infer Pure or a variadic signature."""
    if not isinstance(plan, dict):
        raise ValueError("typed dialect plan must be a mapping")
    dialect = _identifier(plan.get("dialect_name"), "dialect_name")
    target = _identifier(plan.get("target"), "target")
    cls = camel(target)
    if plan.get("lowering"):
        raise ValueError("typed dialect plans require checked lowering, not name-only renaming")
    types = plan.get("types")
    ops = plan.get("ops")
    if not isinstance(types, list) or not isinstance(ops, list) or not ops:
        raise ValueError("typed dialect plan requires types and nonempty ops lists")
    known_types = set()
    normalized_types = []
    for row in types:
        if not isinstance(row, dict) or set(row) - {"name", "summary", "parameters"}:
            raise ValueError("typed dialect type has unsupported fields")
        name = _identifier(row.get("name"), "type.name")
        if name in known_types:
            raise ValueError(f"duplicate typed dialect type {name}")
        known_types.add(name)
        params = row.get("parameters", [])
        if not isinstance(params, list):
            raise ValueError(f"{name}.parameters must be a list")
        normalized_params = []
        for param in params:
            if not isinstance(param, dict) or set(param) - {"name", "kind", "min", "max", "intervals", "choices", "unit"}:
                raise ValueError(f"{name}.parameters contains unsupported fields")
            pname = _identifier(param.get("name"), f"{name}.parameter.name")
            kind = param.get("kind")
            if kind not in {"unsigned", "type"}:
                raise ValueError(f"{name}.{pname} has unsupported parameter kind")
            if "unit" in param and (not isinstance(param["unit"], str) or not param["unit"].strip()):
                raise ValueError(f"{name}.{pname} has invalid physical unit")
            if kind == "type" and set(param) & {"min", "max", "intervals", "choices"}:
                raise ValueError(f"{name}.{pname} type parameter cannot have numeric bounds")
            if kind == "unsigned" and (
                type(param.get("min", 0)) is not int or type(param.get("max", 2**32 - 1)) is not int
                or param.get("min", 0) < 0 or param.get("max", 2**32 - 1) > 2**32 - 1
                or param.get("max", 2**32 - 1) < param.get("min", 0)
            ):
                raise ValueError(f"{name}.{pname} has invalid unsigned bounds")
            if "intervals" in param or "choices" in param:
                if kind != "unsigned" or set(param) & {"min", "max"} or (
                    "intervals" in param and "choices" in param
                ):
                    raise ValueError(f"{name}.{pname} has conflicting unsigned domains")
                if "intervals" in param:
                    intervals = param["intervals"]
                    if not isinstance(intervals, list) or not intervals:
                        raise ValueError(f"{name}.{pname} has invalid intervals")
                    previous_max = -1
                    for interval in intervals:
                        if not isinstance(interval, dict) or set(interval) != {"min", "max", "step"}:
                            raise ValueError(f"{name}.{pname} has invalid intervals")
                        low, high, step = (interval[key] for key in ("min", "max", "step"))
                        if (
                            any(type(item) is not int for item in (low, high, step))
                            or low < 0 or low <= previous_max or high < low or high > 2**32 - 1
                            or step < 1 or (high - low) % step
                        ):
                            raise ValueError(f"{name}.{pname} has invalid intervals")
                        previous_max = high
                else:
                    choices = param["choices"]
                    if (
                        not isinstance(choices, list) or not choices
                        or any(type(item) is not int or not 0 <= item <= 2**32 - 1 for item in choices)
                        or len(set(choices)) != len(choices)
                    ):
                        raise ValueError(f"{name}.{pname} has invalid choices")
            normalized_params.append({**param, "name": pname, "kind": kind})
        if len({p["name"] for p in normalized_params}) != len(normalized_params):
            raise ValueError(f"{name} has duplicate parameter names")
        normalized_types.append({"name": name, "summary": str(row.get("summary") or name), "parameters": normalized_params})
    normalized_ops = []
    op_names = set()
    for row in ops:
        if not isinstance(row, dict) or set(row) - {"name", "summary", "signature"}:
            raise ValueError("typed dialect operation has unsupported fields")
        name = _identifier(row.get("name"), "op.name")
        if name in op_names:
            raise ValueError(f"duplicate typed dialect operation {name}")
        op_names.add(name)
        signature = row.get("signature")
        if (
            not isinstance(signature, dict)
            or not {"operands", "results", "attributes", "effects"} <= set(signature)
            or set(signature) - {"operands", "results", "attributes", "effects", "predicates"}
        ):
            raise ValueError(f"{name}.signature requires operands, results, attributes, and effects")
        fields = []
        parsed = {}
        for role in ("operands", "results", "attributes"):
            values = signature[role]
            if not isinstance(values, list):
                raise ValueError(f"{name}.{role} must be a list")
            parsed[role] = []
            for field in values:
                if not isinstance(field, dict):
                    raise ValueError(f"{name}.{role} has a malformed field")
                allowed = {"name", "type"} if role != "attributes" else {
                    "name", "type", "min", "max", "intervals", "choices", "role", "unit",
                }
                if set(field) - allowed:
                    raise ValueError(f"{name}.{role} has unsupported field properties")
                fname = _identifier(field.get("name"), f"{name}.{role}.name")
                if fname in fields:
                    raise ValueError(f"{name} repeats argument/result/attribute name {fname}")
                fields.append(fname)
                ftype = field.get("type")
                if role == "attributes":
                    if ftype not in _ATTR_TYPES:
                        raise ValueError(f"{name}.{fname} has unsupported attribute type")
                    if field.get("role", "binding") not in {"mode", "binding", "policy"}:
                        raise ValueError(f"{name}.{fname} has unsupported attribute role")
                    if "unit" in field and (not isinstance(field["unit"], str) or not field["unit"].strip()):
                        raise ValueError(f"{name}.{fname} has invalid physical unit")
                    choices = field.get("choices")
                    if choices is not None and (
                        not isinstance(choices, list) or not choices
                        or len({json.dumps(choice, sort_keys=True) for choice in choices}) != len(choices)
                        or any(type(choice) is not (str if ftype == "string" else bool if ftype == "bool" else int) for choice in choices)
                    ):
                        raise ValueError(f"{name}.{fname} has invalid choices")
                    if choices is not None and set(field) & {"min", "max", "intervals"}:
                        raise ValueError(f"{name}.{fname} has conflicting finite domains")
                    if ("min" in field or "max" in field) and ftype not in {"i32", "i64"}:
                        raise ValueError(f"{name}.{fname} has noninteger bounds")
                    if "intervals" in field:
                        if ftype not in {"i32", "i64"} or set(field) & {"min", "max", "choices"}:
                            raise ValueError(f"{name}.{fname} has conflicting integer domains")
                        intervals = field["intervals"]
                        if not isinstance(intervals, list) or not intervals:
                            raise ValueError(f"{name}.{fname} has invalid intervals")
                        previous_max = -(2**63) - 1
                        width = 32 if ftype == "i32" else 64
                        for interval in intervals:
                            if not isinstance(interval, dict) or set(interval) != {"min", "max", "step"}:
                                raise ValueError(f"{name}.{fname} has invalid intervals")
                            low, high, step = (interval[key] for key in ("min", "max", "step"))
                            if (
                                any(type(item) is not int for item in (low, high, step))
                                or low < -(2 ** (width - 1)) or high >= 2 ** (width - 1)
                                or low <= previous_max or high < low or not 1 <= step <= 2**64 - 1
                                or (high - low) % step
                            ):
                                raise ValueError(f"{name}.{fname} has invalid intervals")
                            previous_max = high
                    if any(type(field[key]) is not int for key in ("min", "max") if key in field):
                        raise ValueError(f"{name}.{fname} has invalid bounds")
                    if "min" in field and "max" in field and field["min"] > field["max"]:
                        raise ValueError(f"{name}.{fname} min exceeds max")
                    if ftype in {"i32", "i64"}:
                        width = 32 if ftype == "i32" else 64
                        for key in ("min", "max"):
                            if key in field and not -(2 ** (width - 1)) <= field[key] < 2 ** (width - 1):
                                raise ValueError(f"{name}.{fname}.{key} exceeds {ftype} range")
                        if choices is not None and any(
                            not -(2 ** (width - 1)) <= choice < 2 ** (width - 1) for choice in choices
                        ):
                            raise ValueError(f"{name}.{fname}.choices exceed {ftype} range")
                    parsed[role].append({**field, "role": field.get("role", "binding")})
                else:
                    parsed[role].append({
                        "name": fname, "type": _value_type(ftype, dialect, cls, known_types),
                        "source_type": ftype,
                    })
        effects = signature["effects"]
        if (
            not isinstance(effects, list)
            or any(not isinstance(effect, str) or effect not in _EFFECTS for effect in effects)
            or len(set(effects)) != len(effects)
        ):
            raise ValueError(f"{name}.effects must be a distinct known effect list (empty means Pure)")
        parsed["effects"] = effects
        predicates = signature.get("predicates", [])
        if not isinstance(predicates, list) or len(predicates) > 32:
            raise ValueError(f"{name}.predicates must be a bounded list")
        predicate_ids = set()
        attr_types = {field["name"]: field["type"] for field in parsed["attributes"]}
        type_refs = _type_parameter_refs(parsed, normalized_types, dialect)
        for predicate in predicates:
            if not isinstance(predicate, dict) or set(predicate) != {"id", "expr"}:
                raise ValueError(f"{name}.predicate requires id and expr")
            identity = _identifier(predicate["id"], f"{name}.predicate.id")
            if identity in predicate_ids:
                raise ValueError(f"{name} has duplicate predicate id {identity}")
            predicate_ids.add(identity)
            kind, _ = _predicate_expression(predicate["expr"], attr_types, type_refs)
            if kind != "bool":
                raise ValueError(f"{name}.{identity} predicate must be Boolean")
        parsed["predicates"] = copy.deepcopy(predicates)
        normalized_ops.append({"name": name, "summary": str(row.get("summary") or name), "signature": parsed})
    return {"target": target, "dialect_name": dialect, "class": cls, "types": normalized_types, "ops": normalized_ops}


def types_td(spec: dict[str, Any]) -> str:
    cls = spec["class"]
    chunks = [f'include "{cls}Dialect.td"', ""]
    for row in spec["types"]:
        name = row["name"]
        params = row["parameters"]
        lines = [f'def {cls}_{camel(name)} : {cls}_Type<{_literal(camel(name))}, {_literal(name)}> {{',
                 f'  let summary = {_literal(row["summary"])};']
        if params:
            pieces = [f'"{"unsigned" if p["kind"] == "unsigned" else "::mlir::Type"}":${p["name"]}' for p in params]
            lines.append(f'  let parameters = (ins {", ".join(pieces)});')
            fmt = " `,` ".join(f'${p["name"]}' for p in params)
            lines.append(f'  let assemblyFormat = "`<` {fmt} `>`";')
            if any(p["kind"] == "unsigned" and set(p) & {"min", "max", "intervals", "choices"} for p in params):
                lines.append("  let genVerifyDecl = 1;")
        else:
            lines.append('  let assemblyFormat = "";')
        lines.append("}")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) + "\n"


def ops_td(spec: dict[str, Any]) -> str:
    cls = spec["class"]
    chunks = [f'include "{cls}Dialect.td"', f'include "{cls}Types.td"',
              'include "mlir/Interfaces/SideEffectInterfaces.td"', ""]
    for row in spec["ops"]:
        sig = row["signature"]
        traits = "[Pure]" if not sig["effects"] else "[DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]"
        operands = [f'{field["type"]}:${field["name"]}' for field in sig["operands"]]
        operands += [f'{_ATTR_TYPES[field["type"]]}:${field["name"]}' for field in sig["attributes"]]
        results = [f'{field["type"]}:${field["name"]}' for field in sig["results"]]
        verify = bool(sig["predicates"]) or any(
            any(key in field for key in ("min", "max", "intervals", "choices")) for field in sig["attributes"]
        )
        chunks.append("\n".join([
            f'def {cls}_{camel(row["name"])}Op : {cls}_Op<{_literal(row["name"])}, {traits}> {{',
            f'  let summary = {_literal(row["summary"])};',
            f'  let arguments = (ins {", ".join(operands)});',
            f'  let results = (outs {", ".join(results)});',
            *(["  let hasVerifier = 1;"] if verify else []),
            "}",
        ]))
    return "\n\n".join(chunks) + "\n"


def ops_cpp(spec: dict[str, Any], pkg: str) -> str:
    cls = spec["class"]
    dialect = spec["dialect_name"]
    chunks = [f'#include "{pkg}/Dialect/{cls}/IR/{cls}Dialect.h"',
              '#include "mlir/Interfaces/SideEffectInterfaces.h"',
              'using namespace mlir;', f'using namespace merlin::{dialect};', ""]
    for row in spec["ops"]:
        name = f'{camel(row["name"])}Op'
        sig = row["signature"]
        if sig["effects"]:
            body = "\n".join(f'  effects.emplace_back(MemoryEffects::{_EFFECTS[effect]}::get());' for effect in sig["effects"])
            chunks.append(
                f'void {name}::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {{\n'
                f'{body}\n}}'
            )
        checks = []
        for field in sig["attributes"]:
            attr = f'getOperation()->getAttrOfType<{"StringAttr" if field["type"] == "string" else "BoolAttr" if field["type"] == "bool" else "IntegerAttr"}>({_literal(field["name"])})'
            value = f'{attr}.getValue()' if field["type"] == "string" else f'{attr}.getValue()' if field["type"] == "bool" else f'{attr}.getInt()'
            if "min" in field:
                checks.append(f'  if ({value} < {_cpp_integer(field["min"])}) return emitOpError({_literal(field["name"] + " below minimum")});')
            if "max" in field:
                checks.append(f'  if ({value} > {_cpp_integer(field["max"])}) return emitOpError({_literal(field["name"] + " above maximum")});')
            if "choices" in field:
                terms = [
                    f'{value} == {_literal(choice) if isinstance(choice, str) else str(choice).lower() if isinstance(choice, bool) else _cpp_integer(choice)}'
                    for choice in field["choices"]
                ]
                checks.append(f'  if (!({" || ".join(terms)})) return emitOpError({_literal(field["name"] + " has invalid choice")});')
            if "intervals" in field:
                terms = [
                    f'({value} >= {_cpp_integer(interval["min"])} && '
                    f'{value} <= {_cpp_integer(interval["max"])} && '
                    f'((static_cast<unsigned long long>({value}) - '
                    f'static_cast<unsigned long long>({_cpp_integer(interval["min"])})) '
                    f'% {interval["step"]}ULL) == 0ULL)'
                    for interval in field["intervals"]
                ]
                checks.append(
                    f'  if (!({" || ".join(terms)})) return emitOpError('
                    f'{_literal(field["name"] + " outside declared domain")});'
                )
        fields = {field["name"]: field["type"] for field in sig["attributes"]}
        refs = _type_parameter_refs(sig, spec["types"], dialect)
        for predicate in sig["predicates"]:
            _, expression = _predicate_expression(predicate["expr"], fields, refs)
            checks.append(f'  if (!({expression})) return emitOpError({_literal("predicate " + predicate["id"] + " failed")});')
        if checks:
            chunks.append(f'LogicalResult {name}::verify() {{\n' + "\n".join(checks) + '\n  return success();\n}')
    return "\n\n".join(chunks) + "\n"


def type_verifiers(spec: dict[str, Any]) -> str:
    cls = spec["class"]
    chunks = []
    for row in spec["types"]:
        params = row["parameters"]
        if not any(p["kind"] == "unsigned" and set(p) & {"min", "max", "intervals", "choices"} for p in params):
            continue
        signature = ", ".join(f'{"unsigned" if p["kind"] == "unsigned" else "::mlir::Type"} {p["name"]}' for p in params)
        checks = []
        for param in params:
            if "min" in param:
                checks.append(f'  if ({param["name"]} < {param["min"]}) return emitError() << {_literal(param["name"] + " below minimum")};')
            if "max" in param:
                checks.append(f'  if ({param["name"]} > {param["max"]}) return emitError() << {_literal(param["name"] + " above maximum")};')
            if "intervals" in param:
                cases = [
                    f'({param["name"]} >= {interval["min"]}u && {param["name"]} <= {interval["max"]}u'
                    f' && (({param["name"]} - {interval["min"]}u) % {interval["step"]}u) == 0u)'
                    for interval in param["intervals"]
                ]
                checks.append(
                    f'  if (!( {" || ".join(cases)} )) return emitError() << '
                    f'{_literal(param["name"] + " outside declared domain")};'
                )
            if "choices" in param:
                cases = [f'{param["name"]} == {choice}u' for choice in param["choices"]]
                checks.append(
                    f'  if (!( {" || ".join(cases)} )) return emitError() << '
                    f'{_literal(param["name"] + " outside declared domain")};'
                )
        chunks.append(
            f'LogicalResult {camel(row["name"])}Type::verify(llvm::function_ref<InFlightDiagnostic()> emitError, {signature}) {{\n'
            + "\n".join(checks) + '\n  return success();\n}'
        )
    return "\n\n".join(chunks)
