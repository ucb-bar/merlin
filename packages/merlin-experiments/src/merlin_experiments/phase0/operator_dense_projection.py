"""Bounded initializer intake for the unchanged typed exclusion projection.

This is trusted source intake, not executable IR or numerical equivalence.
Complete original bytes are hashed. Only checked function-body arith.constant
dense values become resource placeholders, which the existing exclusion
canonicalizer deliberately omits. Scalar-region constants can never be elided.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from math import prod

from xdsl.dialects import builtin

from merlin.common import mlir_query as Q

MARKER = "prov.merlin_operator_dense_projection_v1"
HANDLE = "merlin_operator_dense_projection_v1_"
_HEADER = re.compile(rb"(?P<head>[ \t]*%[A-Za-z0-9_.$-]+[ \t]*=[ \t]*arith\.constant)\b")
_HEX = re.compile(rb"[0-9a-fA-F]*")
_WIDTHS = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8}
_WIDTHS.update({prefix + str(bits): bits // 8 for prefix in ("i", "si", "ui") for bits in (8, 16, 32, 64)})


@dataclass(frozen=True)
class Projection:
    source_sha256: str
    source_bytes: int
    compact: bytes
    projected_constants: int
    hexadecimal_payload_bytes: int


def _header(pending):
    """A bounded lexical prefix; original attributes are subsequently typed/verified.

    Balance dictionaries and quoted strings rather than matching across literal
    contents. Unknown syntax is retained and remains subject to the parse cap.
    """
    match = _HEADER.match(pending)
    if match is None:
        return None
    at = match.end()
    while at < len(pending) and pending[at] in b" \t":
        at += 1
    attrs = None
    if at < len(pending) and pending[at] == ord("{"):
        start, depth, quoted, escaped = at, 0, False, False
        while at < min(len(pending), 4096):
            ch = pending[at]
            if ch in b"\r\n":
                return None
            if quoted:
                if escaped:
                    escaped = False
                elif ch == ord("\\"):
                    escaped = True
                elif ch == ord('"'):
                    quoted = False
            elif ch == ord('"'):
                quoted = True
            elif ch == ord("{"):
                depth += 1
            elif ch == ord("}"):
                depth -= 1
                if depth == 0:
                    at += 1
                    attrs = bytes(pending[start:at])
                    break
            at += 1
        if attrs is None:
            return None
        while at < len(pending) and pending[at] in b" \t":
            at += 1
    if at + 6 > 4096 or pending[at:at + 6] != b"dense<":
        return None
    return bytes(match["head"]), attrs, at + 6


def _literal_module(value, typ):
    return Q.parse(f"module {{ func.func @literal() -> {typ} {{\n"
                   f"%v = arith.constant dense<{value}> : {typ}\n"
                   f"return %v : {typ}\n}} }}")


def _validate(compact, constants, *, max_decoded_bytes):
    module = Q.parse(compact.decode("utf-8"))
    return validate_module(module, constants, max_decoded_bytes=max_decoded_bytes)


def validate_module(module, constants, *, max_decoded_bytes):
    """Recheck ALL projected initializer owners in the complete typed module."""
    module.verify()
    seen, decoded, hex_bytes = set(), 0, 0
    for op in Q.walk(module):
        marker = op.attributes.get(MARKER)
        if marker is None:
            continue
        if (Q.op_name(op) != "arith.constant" or not isinstance(marker, builtin.IntegerAttr)
                or op.parent_op() is None or Q.op_name(op.parent_op()) != "func.func"
                or len(op.results) != 1 or op.regions):
            raise ValueError("dense projection is not an immediate function-body initializer")
        if (any(not key.startswith("prov.") for key in op.attributes)
                or set(op.properties) != {"value"}):
            raise ValueError("dense projection has unsupported initializer attributes or properties")
        ordinal = marker.value.data
        if ordinal in seen or not 0 <= ordinal < len(constants):
            raise ValueError("dense projection initializer ownership is ambiguous")
        seen.add(ordinal)
        value, typ = op.properties.get("value"), op.results[0].type
        # The registered builtin resource interface may uniquify a previously
        # parsed handle. No original source may contain this reserved prefix;
        # ownership is the separately checked marker and ordinal, not a handle
        # spelling or a claim that resource values exist.
        handle = HANDLE + str(ordinal)
        if (not isinstance(value, builtin.DenseResourceAttr)
                or re.fullmatch(re.escape(handle) + r"(?:_[0-9]+)*", value.resource_handle.data) is None
                or not isinstance(typ, builtin.TensorType) or not isinstance(typ.encoding, builtin.NoneAttr)
                or value.type != typ or any(n < 0 for n in typ.get_shape())):
            raise ValueError("dense projection requires exact static unencoded tensor storage")
        element = str(typ.get_element_type())
        if element not in _WIDTHS:
            raise ValueError("dense projection has no checked byte-aligned element representation")
        count = prod(typ.get_shape())
        expected = count * _WIDTHS[element]
        record = constants[ordinal]
        if record["hex_chars"] is not None:
            if record["hex_chars"] != 2 * expected:
                raise ValueError("dense projection hexadecimal length differs from its complete typed storage")
            # Check the same registered dense parser at one element, without
            # materializing the original potentially multi-gigabyte attribute.
            literal = '"0x' + '00' * _WIDTHS[element] + '"'
            _literal_module(literal, f"tensor<1x{element}>").verify()
            hex_bytes += expected
        else:
            payload = record["literal"].decode("ascii").strip()
            if not payload:
                raise ValueError("dense projection initializer literal is empty")
            if payload.startswith("[") or count == 0:
                decoded += expected
                if decoded > max_decoded_bytes:
                    raise ValueError("dense projection literal decode exceeds its independent memory budget")
                _literal_module(payload, str(typ)).verify()
            else:
                # A scalar dense literal is a splat. Its value is checked by
                # the original registered parser at one element, independent
                # of the source tensor's size.
                _literal_module(payload, f"tensor<1x{element}>").verify()
    if seen != set(range(len(constants))):
        raise ValueError("dense projection did not recover every original initializer")
    return hex_bytes


def project(stream, *, max_input_bytes, max_compact_bytes, max_literal_bytes=65536,
            max_retained_literal_bytes=4 << 20, max_decoded_bytes=4 << 20, max_constants=8192,
            chunk_bytes=65536):
    """The original complete compact parse and initializer-ownership route."""
    metadata, compact, constants = scan(
        stream, max_input_bytes=max_input_bytes, max_compact_bytes=max_compact_bytes,
        max_literal_bytes=max_literal_bytes, max_retained_literal_bytes=max_retained_literal_bytes,
        max_decoded_bytes=max_decoded_bytes, max_constants=max_constants, chunk_bytes=chunk_bytes)
    hex_bytes = _validate(compact, constants, max_decoded_bytes=max_decoded_bytes)
    return Projection(metadata["source_sha256"], metadata["source_bytes"], compact,
                      len(constants), hex_bytes)


def scan(stream, *, max_input_bytes, max_compact_bytes, max_literal_bytes=65536,
         max_retained_literal_bytes=4 << 20, max_decoded_bytes=4 << 20, max_constants=8192,
         chunk_bytes=65536, sink=None):
    """Hash full binary input and bound every parser/retained-literal allocation.

    This low-level scan is not a checked Projection. Complete typed parsing and
    validate_module remain mandatory before its metadata can support intake.
    Pretty-printed custom-form constants beginning a physical line are the
    supported intake domain. Other forms remain unchanged and must fit the
    compact parse ceiling. Hex values require their full typed byte count;
    hexadecimal splat encodings are conservatively refused. Output is private
    transient IR, never compiler input, an answer file or a publication grant.
    """
    limits = (max_input_bytes, max_compact_bytes, max_literal_bytes, max_retained_literal_bytes,
              max_decoded_bytes, max_constants, chunk_bytes)
    if any(type(n) is not int or n < 1 for n in limits) or chunk_bytes > 1 << 20:
        raise ValueError("dense projection requires explicit finite source and allocation bounds")
    digest, size, emitted, compact, pending = hashlib.sha256(), 0, 0, bytearray(), bytearray()
    constants, retained, reserved_tail = [], 0, b""
    state, line_projected, current = "normal", False, None
    reserved = HANDLE.encode().removesuffix(b"_")

    def emit(data):
        nonlocal emitted
        if emitted + len(data) > max_compact_bytes:
            raise ValueError("dense projection exceeds the unchanged compact IR parse budget")
        emitted += len(data)
        if sink is None:
            compact.extend(data)
        elif sink(data) != len(data):
            raise ValueError("dense projection scratch sink did not retain complete syntax")

    while True:
        piece = stream.readline(chunk_bytes)
        if not isinstance(piece, bytes) or len(piece) > chunk_bytes:
            raise ValueError("dense projection requires a bounded binary source stream")
        if not piece:
            break
        size += len(piece)
        if size > max_input_bytes:
            raise ValueError("dense projection source exceeds its streaming byte budget")
        digest.update(piece)
        search = reserved_tail + piece
        if reserved in search:
            raise ValueError("dense projection reserved ownership marker occurs in original source")
        reserved_tail = search[-len(reserved):]
        pending.extend(piece)
        while pending:
            if state == "normal":
                match = None if line_projected else _header(pending)
                if match:
                    if len(constants) >= max_constants:
                        raise ValueError("dense projection initializer prefix or count exceeds its bound")
                    ordinal = len(constants)
                    current = {"hex_chars": None, "literal": bytearray()}
                    constants.append(current)
                    head, attrs, end = match
                    original = attrs[1:-1].strip() if attrs is not None else b""
                    emit(head + b" {" + original + (b", " if original else b"")
                         + MARKER.encode() + b" = " + str(ordinal).encode()
                         + b" : i64} dense_resource<" + HANDLE.encode() + str(ordinal).encode())
                    del pending[:end]
                    state, line_projected = "kind", True
                    continue
                if pending.endswith(b"\n"):
                    emit(pending)
                    pending.clear()
                    line_projected = False
                elif len(pending) > max_compact_bytes - emitted:
                    raise ValueError("dense projection exceeds the unchanged compact IR parse budget")
                break
            if state == "kind":
                if len(pending) < 3:
                    break
                if pending.startswith(b'"0x'):
                    current["hex_chars"] = 0
                    del pending[:3]
                    state = "hex"
                else:
                    state = "literal"
                continue
            if state == "hex":
                end = pending.find(b'"')
                data = pending if end < 0 else pending[:end]
                if _HEX.fullmatch(data) is None:
                    raise ValueError("dense projection hexadecimal initializer is malformed")
                current["hex_chars"] += len(data)
                del pending[:len(data)]
                if end < 0:
                    break
                del pending[:1]
                state = "close"
                continue
            if state == "literal":
                end = pending.find(b">")
                data = pending if end < 0 else pending[:end]
                if len(current["literal"]) + len(data) > max_literal_bytes:
                    raise ValueError("dense projection nonhex initializer exceeds its literal budget")
                current["literal"].extend(data)
                retained += len(data)
                if retained > max_retained_literal_bytes:
                    raise ValueError("dense projection retained literals exceed their allocation budget")
                del pending[:len(data)]
                if end < 0:
                    break
                state = "close"
                continue
            if state == "close":
                while pending and pending[0] in b" \t\r":
                    del pending[:1]
                if not pending:
                    break
                if pending[0] != ord(">"):
                    raise ValueError("dense projection initializer has no exact closing delimiter")
                del pending[:1]
                emit(b">")
                state = "normal"
                continue
    if state != "normal":
        raise ValueError("dense projection source ends inside an initializer")
    emit(pending)
    # This scan is NOT verified evidence. A consumer must reconstruct and
    # verify the complete module, then call validate_module for every marker.
    return dict(source_sha256=digest.hexdigest(), source_bytes=size, compact_bytes=emitted), bytes(compact), constants
