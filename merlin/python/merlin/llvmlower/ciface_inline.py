"""Keep exceptionally wide MLIR C-interface boundaries out of target backends.

``llvm.emit_c_interface`` deliberately exposes one descriptor pointer per memref, but its
generated wrapper expands those pointers into the flattened implementation ABI.  Large models can
therefore contain a wrapper call with hundreds of scalar arguments.  Once the implementation gains
an external call, LLVM may stop inlining that boundary and some targets cannot legally materialize
the enormous outgoing frame (LLVM's RISC-V backend fails in register scavenging, for example).

This repair is ABI-driven, not model- or target-driven: mark only implementation functions that
have a matching ``_mlir_ciface_<name>`` wrapper and exceed the declared flattened-argument limit.
Inlining removes the artificial wrapper-to-implementation call while preserving both public
symbols and all runtime-visible argument semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from string import ascii_letters, digits


DEFAULT_MAX_FLATTENED_ARGUMENTS = 256

#: LLVM identifier characters, per the grammar this module accepts.  The leading character may be
#: neither a digit nor a dash, which ``_define_name`` enforces separately.
_NAME_CHARS = frozenset(ascii_letters + digits + "$._-")
_NAME_HEAD = frozenset(ascii_letters + "$._")
#: Word characters, so an attribute lookup matches whole words only.
_WORD_CHARS = frozenset(ascii_letters + digits + "_")


def _define_name(line: str) -> tuple[str, int] | None:
    """The symbol a ``define`` line declares, and the offset its name starts at.

    Parsed structurally rather than by pattern.  A too-narrow pattern silently drops a
    valid-but-differently-spelled definition, which produces a wrong ABI repair instead of a loud
    failure.  Quoted symbol names (``@"with space"``) are deliberately not accepted, matching this
    module's rule of never guessing at LLVM syntax.
    """
    if not line.startswith("define"):
        return None
    if not line[len("define"):][:1].isspace():
        return None
    at = line.find("@")
    while at != -1:
        start = end = at + 1
        while end < len(line) and line[end] in _NAME_CHARS:
            end += 1
        name = line[start:end]
        if name and name[0] in _NAME_HEAD and line[end:end + 1] == "(":
            return name, start
        at = line.find("@", at + 1)
    return None


def _ends_with_value_name(part: str) -> bool:
    """Whether an argument fragment ends in a ``%ssa`` value name.

    A parameter carrying a name is a value argument; a bare type is not.  Scanned back to the final
    ``%`` rather than matched, for the same reason as ``_define_name``.
    """
    text = part.rstrip()
    at = text.rfind("%")
    if at == -1 or at == len(text) - 1:
        return False
    return all(character in _NAME_CHARS for character in text[at + 1:])


def _has_attribute(attributes: str, wanted: str) -> bool:
    """Whether a whole-word attribute appears in a function-attribute fragment.

    Whole-word, so ``alwaysinlinehint`` does not count as ``alwaysinline``.
    """
    token = ""
    for character in attributes:
        if character in _WORD_CHARS:
            token += character
            continue
        if token == wanted:
            return True
        token = ""
    return token == wanted


@dataclass(frozen=True)
class InlinedCInterfaceBoundary:
    implementation: str
    wrapper: str
    flattened_arguments: int


def _signature_close(line: str, open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(line)):
        token = line[index]
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _value_argument_count(arguments: str) -> int:
    """Count top-level LLVM arguments without splitting aggregate/function-pointer types."""
    parts: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0, "<": 0}
    closes = {")": "(", "]": "[", "}": "{", ">": "<"}
    quoted = False
    escaped = False
    for index, token in enumerate(arguments):
        if quoted:
            if escaped:
                escaped = False
            elif token == "\\":
                escaped = True
            elif token == '"':
                quoted = False
            continue
        if token == '"':
            quoted = True
        elif token in depths:
            depths[token] += 1
        elif token in closes:
            depths[closes[token]] -= 1
        elif token == "," and not any(depths.values()):
            parts.append(arguments[start:index])
            start = index + 1
    if arguments.strip():
        parts.append(arguments[start:])
    return sum(_ends_with_value_name(part.strip()) for part in parts)


def inline_wide_ciface_implementations(
    llvm_ir: str, *, max_flattened_arguments: int = DEFAULT_MAX_FLATTENED_ARGUMENTS
) -> tuple[str, dict]:
    """Add ``alwaysinline`` to wide implementations that have an MLIR C wrapper.

    The LLVM translator currently prints each function signature on one line.  Refuse malformed or
    multiline definitions by leaving them untouched; this helper must never guess at LLVM syntax.
    The returned report makes the automatic ABI repair auditable in lowering receipts.
    """
    if max_flattened_arguments < 1:
        raise ValueError("max_flattened_arguments must be positive")
    lines = llvm_ir.splitlines(keepends=True)
    definitions: dict[str, tuple[int, int, int, int]] = {}
    for line_index, line in enumerate(lines):
        declared = _define_name(line)
        if declared is None:
            continue
        name, name_start = declared
        open_index = line.find("(", name_start + len(name))
        close_index = _signature_close(line, open_index)
        if close_index is None or "{" not in line[close_index + 1 :]:
            continue
        arguments = line[open_index + 1 : close_index]
        argument_count = _value_argument_count(arguments)
        definitions[name] = (line_index, open_index, close_index, argument_count)

    repairs: list[InlinedCInterfaceBoundary] = []
    for name, (line_index, _open, close, argument_count) in definitions.items():
        if name.startswith("_mlir_ciface_") or argument_count <= max_flattened_arguments:
            continue
        wrapper = f"_mlir_ciface_{name}"
        if wrapper not in definitions:
            continue
        line = lines[line_index]
        brace = line.find("{", close + 1)
        attributes = line[close + 1 : brace]
        if _has_attribute(attributes, "alwaysinline"):
            continue
        lines[line_index] = f"{line[:brace].rstrip()} alwaysinline {line[brace:]}"
        repairs.append(InlinedCInterfaceBoundary(name, wrapper, argument_count))

    return "".join(lines), {
        "schema": "wide_ciface_inline_v1",
        "max_flattened_arguments": max_flattened_arguments,
        "inlined": [asdict(repair) for repair in repairs],
        "count": len(repairs),
    }
