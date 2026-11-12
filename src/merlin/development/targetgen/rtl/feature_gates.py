"""Exact structural FIRRTL boolean feature-gate parsing; missing stays unknown."""

from typing import Any


def _firrtl_bool_literal(expr: str) -> bool | None:
    """A FIRRTL one-bit UInt literal, parsed exactly (not substring/regex matched)."""
    expr = expr.strip()
    prefix = "UInt<1>("
    if not expr.startswith(prefix) or not expr.endswith(")"):
        return None
    token = expr[len(prefix) : -1].strip().lower()
    try:
        value = int(token[2:], 16) if token.startswith("0h") else int(token, 10)
    except ValueError:
        return None
    return bool(value) if value in (0, 1) else None


def _firrtl_call_args(expr: str, callee: str) -> list[str] | None:
    """Top-level operands of one FIRRTL primitive call, preserving nested expressions."""
    expr = expr.strip()
    head = f"{callee}("
    if not expr.startswith(head) or not expr.endswith(")"):
        return None
    body = expr[len(head) : -1]
    args: list[str] = []
    depth = 0
    start = 0
    for i, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
        elif char == "," and depth == 0:
            args.append(body[start:i].strip())
            start = i + 1
    if depth != 0:
        return None
    args.append(body[start:].strip())
    return args


def extract_boolean_feature_gate(fir_text: str, *, gate: dict[str, str]) -> dict[str, Any] | None:
    """Read a declared FIRRTL ``and`` build gate with one literal and one dynamic operand.

    Missing, duplicate or differently-shaped nodes are unknown, not inferred from adjacent ISA fields.
    """
    found: list[dict[str, Any]] = []
    prefix = f"node {gate['node']} = "
    module: str | None = None
    for line_no, raw in enumerate(fir_text.splitlines(), 1):
        code = raw.split("@[", 1)[0].strip()
        if code.startswith("module ") and " :" in code:
            module = code[len("module ") : code.index(" :")].strip()
            continue
        if module != gate["module"]:
            continue
        if not code.startswith(prefix):
            continue
        expr = code[len(prefix) :].strip()
        args = _firrtl_call_args(expr, "and")
        if args is None or len(args) != 2:
            continue
        literals = [(idx, _firrtl_bool_literal(arg)) for idx, arg in enumerate(args)]
        literals = [(idx, val) for idx, val in literals if isinstance(val, bool)]
        if len(literals) != 1:
            continue
        literal_index, value = literals[0]
        dynamic = args[1 - literal_index]
        # The other operand must be the target-declared dynamic enable, not merely a second literal.
        if gate["dynamic_operand_contains"].lower() not in dynamic.lower():
            continue
        found.append({"value": value, "line": line_no, "expression": expr})
    if len(found) != 1:
        return None
    return found[0]
