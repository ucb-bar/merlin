"""Explicit source analysis for independent FMA packets with finite constants.

This module owns typed SSA grouping and source witnesses. An explicitly supplied
provider owns instruction emission, resource legality and exact FMA semantics.
Default calls preserve source bytes. No policy or profitable width is inferred.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Callable

from merlin.compiler.lower.model.constant_float_clamp import _finite_f32, _has_environment_scope, _ssa_key
from merlin.compiler.lower.model.late_quant_rne import _functions, _identity, _statement_boundary, _tokens


@dataclass(frozen=True)
class SourceFma:
    result: str
    operands: tuple[str, str, str]
    start: int
    end: int


@dataclass(frozen=True)
class ConstantFmaPacket:
    calls: tuple[SourceFma, ...]
    mode: str
    constant_words: tuple[int, ...]
    source_function: str
    source_block: str
    source_sha256: str


def _call(tokens, index: int) -> SourceFma | None:
    values = [token.text for token in tokens[index : index + 15]]
    if len(values) != 15 or not values[0].startswith("%") or values[1:4] != ["=", "call", "float"]:
        return None
    if (
        not values[4].startswith("@")
        or _identity(values[4]) != "llvm.fma.f32"
        or values[5] != "("
        or values[6] != values[9]
        or values[6] != values[12]
        or values[6] != "float"
        or values[8] != ","
        or values[11] != ","
        or values[14] != ")"
        or not _statement_boundary(tokens, index + 15)
    ):
        return None
    return SourceFma(values[0], (values[7], values[10], values[13]), tokens[index].start, tokens[index + 14].end)


def _unsupported_environment(tokens) -> bool:
    if _has_environment_scope(tokens):
        return True
    unsafe = {"unsafe-fp-math", "no-nans-fp-math", "no-infs-fp-math", "no-signed-zeros-fp-math"}
    for index, token in enumerate(tokens):
        identity = _identity(token.text)
        words = "".join(c if c.isalnum() or c in "_." else " " for c in identity).split()
        if "frm" in words:
            return True
        if index + 2 < len(tokens) and tokens[index + 1].text == "=":
            value = _identity(tokens[index + 2].text)
            if identity in unsafe and value == "true":
                return True
            if identity.startswith("denormal-fp-math") and value != "ieee,ieee":
                return True
    return False


def analyze(
    text: str, *, width: int = 4, ordinary_nontrapping: bool = False, exception_flags_unobserved: bool = False
) -> tuple[ConstantFmaPacket, ...]:
    """Find consecutive independent, ordinary scalar F32 FMA calls.

    The normal compile seam must verify the complete source LLVM module. Dynamic
    operands must be defined earlier in this block or be explicit float formal
    parameters. No instruction, memory operation, label or unknown call is
    crossed. Unknown call attributes/metadata, strict/constrained or observed
    floating environments refuse. Constants use integer RNE decoding and must
    be finite and nonzero. These premises grant no target emission permission.
    """
    if width not in (2, 4):
        raise ValueError("supported explicit FMA packet widths are 2 and 4")
    if not ordinary_nontrapping or not exception_flags_unobserved:
        return ()
    tokens = _tokens(text)
    if _unsupported_environment(tokens):
        return ()
    digest = hashlib.sha256(text.encode()).hexdigest()
    packets = []
    for body in _functions(tokens):
        if not body:
            continue
        first = next(i for i, token in enumerate(tokens) if token.start == body[0].start)
        define = max(i for i, token in enumerate(tokens[:first]) if token.text == "define")
        header = tokens[define:first]
        symbol = next(i for i, token in enumerate(header) if token.text.startswith("@"))
        name = _identity(header[symbol].text)
        arguments = set()
        index, depth = symbol + 2, 1
        while index < len(header) and depth:
            depth += (header[index].text == "(") - (header[index].text == ")")
            if index + 1 < len(header) and header[index].text == "float" and header[index + 1].text.startswith("%"):
                arguments.add(_ssa_key(header[index + 1].text))
            index += 1
        defined, block, index = set(arguments), "entry", 0
        while index < len(body):
            if index + 1 < len(body) and body[index + 1].text == ":":
                block, defined = body[index].text, set(arguments)
                index += 2
                continue
            group = tuple(_call(body, index + 15 * lane) for lane in range(width))
            if all(group):
                results = {_ssa_key(call.result) for call in group}
                values = tuple(tuple(call.operands[axis] for call in group) for axis in range(3))
                variable = tuple(
                    all(
                        value.startswith("%") and _ssa_key(value) in defined and _ssa_key(value) not in results
                        for value in axis
                    )
                    for axis in values
                )
                constants = []
                for axis in values:
                    decoded = tuple(_finite_f32(value) for value in axis)
                    same = all(decoded) and len({item[0] for item in decoded}) == 1
                    constants.append(decoded[0][0] if same else None)
                mode, words = "", ()
                if constants[1] is not None and variable[0] and variable[2]:
                    mode, words = "constant_rhs", (constants[1],)
                elif constants[0] is not None and constants[2] is not None and variable[1]:
                    mode, words = "constant_lhs_addend", (constants[0], constants[2])
                elif constants[2] is not None and variable[0] and variable[1]:
                    mode, words = "constant_addend", (constants[2],)
                if mode:
                    packets.append(ConstantFmaPacket(group, mode, words, name, block, digest))
                    defined.update(results)
                    index += 15 * width
                    continue
            if index + 1 < len(body) and body[index].text.startswith("%") and body[index + 1].text == "=":
                defined.add(_ssa_key(body[index].text))
            index += 1
    return tuple(packets)


def validate_packet_source(text: str, packet: ConstantFmaPacket) -> None:
    """Bind every retained instruction and ancestor numeric attribute to source."""
    if hashlib.sha256(text.encode()).hexdigest() != packet.source_sha256 or packet not in analyze(
        text, width=len(packet.calls), ordinary_nontrapping=True, exception_flags_unobserved=True
    ):
        raise ValueError("FMA packet or complete source context changed after analysis")


def rewrite(
    text: str,
    *,
    emitter: Callable | None = None,
    width: int = 4,
    ordinary_nontrapping: bool = False,
    exception_flags_unobserved: bool = False,
    temporary_prefix: str = "constant.fma.packet",
) -> tuple[str, dict]:
    """Apply an explicit provider while retaining original result SSA identities.

    The provider must preserve every source FMA input position and single
    rounding, input/result live uses, and independently validate ISA/resource
    legality. This source utility adds no default routing or numeric permission.
    The normal build hook verifies selected LLVM before object identity.
    """
    report = dict(
        schema="explicit_constant_fma_packet_v1",
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        width=width,
        packets=[],
        default_unchanged=emitter is None,
    )
    if emitter is None:
        return text, report
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$"
    if not temporary_prefix or temporary_prefix[0].isdigit() or any(char not in allowed for char in temporary_prefix):
        raise ValueError("invalid generated SSA prefix")
    packets = analyze(
        text,
        width=width,
        ordinary_nontrapping=ordinary_nontrapping,
        exception_flags_unobserved=exception_flags_unobserved,
    )
    reserved = {_ssa_key(token.text) for token in _tokens(text) if token.text.startswith("%")}
    edits, counter = [], 0
    for packet in packets:
        while _ssa_key("%" + temporary_prefix + "." + str(counter)) in reserved:
            counter += 1
        temporary = "%" + temporary_prefix + "." + str(counter)
        reserved.add(_ssa_key(temporary))
        counter += 1
        replacement = emitter(packet, temporary)
        if not isinstance(replacement, str) or not replacement:
            raise ValueError("emitter must return LLVM for the exact original results")
        edits.append((packet.calls[0].start, packet.calls[-1].end, replacement))
        report["packets"].append(asdict(packet))
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    report["selected_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return text, report
