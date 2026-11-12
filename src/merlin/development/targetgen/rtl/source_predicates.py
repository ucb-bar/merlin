"""Read exact pure source predicates over a selected declared integer operand.

This reader recognizes only comparisons against source declaration symbols and
parenthesized conjunction/disjunction. It refuses other Scala expressions. The
result describes the selected source predicate over the declared finite domain;
it establishes neither elaboration correspondence nor hardware effects.
"""

from __future__ import annotations


class SourcePredicateError(ValueError):
    """Selected source predicate cannot be followed exactly."""


def _tokens(expression: str) -> list[str]:
    tokens, position = [], 0
    while position < len(expression):
        character = expression[position]
        if character.isspace():
            position += 1
            continue
        if character.isascii() and (character.isalpha() or character == "_"):
            end = position + 1
            while (
                end < len(expression)
                and expression[end].isascii()
                and (expression[end].isalnum() or expression[end] in "_.")
            ):
                end += 1
            tokens.append(expression[position:end])
            position = end
            continue
        operator = next(
            (
                op
                for op in ("===", "=/=", ">=", "<=", "&&", "||", ">", "<", "(", ")")
                if expression.startswith(op, position)
            ),
            None,
        )
        if operator is None:
            raise SourcePredicateError("source predicate contains an unsupported expression")
        tokens.append(operator)
        position += len(operator)
    return tokens


class _Parser:
    def __init__(self, tokens: list[str], operand: str, symbols: dict[str, int]):
        self.tokens, self.operand, self.symbols = tokens, operand, symbols
        self.position = 0
        self.references: set[str] = set()

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise SourcePredicateError("source predicate is incomplete")
        self.position += 1
        return token

    def parse_or(self):
        node = self.parse_and()
        while self.peek() == "||":
            self.take()
            node = ("or", node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_atom()
        while self.peek() == "&&":
            self.take()
            node = ("and", node, self.parse_atom())
        return node

    def parse_atom(self):
        if self.peek() == "(":
            self.take()
            node = self.parse_or()
            if self.take() != ")":
                raise SourcePredicateError("source predicate parentheses disagree")
            return node
        if self.take() != self.operand:
            raise SourcePredicateError("source predicate is not over the exact selected operand")
        comparator, symbol = self.take(), self.take()
        if comparator not in {"===", "=/=", ">=", "<=", ">", "<"} or symbol not in self.symbols:
            raise SourcePredicateError("source predicate needs an exact comparison to an admitted declaration symbol")
        self.references.add(symbol)
        return (comparator, self.symbols[symbol])


def _evaluate(node, value: int) -> bool:
    operation = node[0]
    if operation in {"and", "or"}:
        first, second = _evaluate(node[1], value), _evaluate(node[2], value)
        return first and second if operation == "and" else first or second
    selected = node[1]
    return {
        "===": value == selected,
        "=/=": value != selected,
        ">=": value >= selected,
        "<=": value <= selected,
        ">": value > selected,
        "<": value < selected,
    }[operation]


def _active_source(source: str) -> str:
    """Mask Scala comments and quoted text without moving original line spans.

    This lexical screen prevents text from impersonating a source declaration;
    it does not establish scope, elaboration or hardware correspondence.
    """
    masked, position = list(source), 0
    while position < len(source):
        start = position
        if source.startswith("//", start):
            end = source.find("\n", start)
            position = len(source) if end < 0 else end
        elif source.startswith("/*", start):
            depth, position = 1, start + 2
            while depth and position < len(source):
                if source.startswith("/*", position):
                    depth += 1
                    position += 2
                elif source.startswith("*/", position):
                    depth -= 1
                    position += 2
                else:
                    position += 1
            if depth:
                raise SourcePredicateError("source predicate has an unclosed block comment")
        elif source.startswith('"""', start):
            end = source.find('"""', start + 3)
            if end < 0:
                raise SourcePredicateError("source predicate has unclosed quoted text")
            position = end + 3
        elif source[start] == '"':
            position = start + 1
            while position < len(source) and source[position] != '"':
                if source[position] in "\r\n":
                    raise SourcePredicateError("source predicate has invalid quoted text")
                position += 2 if source[position] == "\\" else 1
            if position >= len(source):
                raise SourcePredicateError("source predicate has unclosed quoted text")
            position += 1
        elif source[start] == "'":
            end = start + 1
            if end >= len(source):
                raise SourcePredicateError("source predicate has unclosed character text")
            end += 2 if source[end] == "\\" else 1
            if end >= len(source) or source[end] != "'":
                raise SourcePredicateError("source predicate has unsupported quoted syntax")
            position = end + 1
        else:
            position += 1
            continue
        for index in range(start, position):
            if masked[index] not in "\r\n":
                masked[index] = " "
    return "".join(masked)


def _span(source: str, binding: str) -> tuple[str, int, int]:
    lines = source.splitlines()
    prefix = "val " + binding + " ="
    starts = [
        index for index, line in enumerate(_active_source(source).splitlines()) if line.strip().startswith(prefix)
    ]
    if len(starts) != 1:
        raise SourcePredicateError("selected predicate binding must occur exactly once")
    start = end = starts[0]
    expression = lines[start].strip()[len(prefix) :].strip()
    while expression.count("(") > expression.count(")") or expression.rstrip().endswith(("&&", "||")):
        end += 1
        if end >= len(lines):
            raise SourcePredicateError("selected multiline source predicate is incomplete")
        expression += "\n" + lines[end].strip()
    return expression, start + 1, end + 1


def source_predicate(source: str, *, binding: str, operand: str, declarations: dict[int, str]) -> dict:
    """Derive exact selected-source membership without role inference or constants."""
    if not binding or not operand or not declarations:
        raise SourcePredicateError("source predicate needs explicit binding, operand and admitted declarations")
    symbols = {name: value for value, name in declarations.items()}
    if len(symbols) != len(declarations) or any(type(value) is not int or value < 0 for value in declarations):
        raise SourcePredicateError("source predicate declaration domain is ambiguous or invalid")
    expression, start, end = _span(source, binding)
    parser = _Parser(_tokens(expression), operand, symbols)
    parsed = parser.parse_or()
    if parser.position != len(parser.tokens):
        raise SourcePredicateError("source predicate contains an unconsumed expression")
    return {
        "binding": binding,
        "operand": operand,
        "expression": expression,
        "source_lines": [start, end],
        "symbols": sorted(parser.references),
        "declared_domain": sorted(declarations),
        "selected_source_values": [value for value in sorted(declarations) if _evaluate(parsed, value)],
        "scope": "selected public source predicate over the admitted declaration domain",
    }
