"""Live, bounded original dense storage identities for private source exclusions.

These identities never authorize execution. Hex storage is hashed while reading
the original source; textual values use the registered dense attribute reader.
The complete module and every projected owner must independently verify first.
"""

from __future__ import annotations

import hashlib
from math import prod

from xdsl.dialects import builtin

_TOKEN = object()


class _HexStorage:
    __slots__ = ("_digest", "_tail", "_bytes")

    def __init__(self, token):
        if token is not _TOKEN:
            raise ValueError("initializer identity needs its original streaming owner")
        self._digest, self._tail, self._bytes = hashlib.sha256(), b"", 0

    def feed(self, value):
        value = self._tail + value
        end = len(value) // 2 * 2
        storage = bytes.fromhex(value[:end].decode("ascii"))
        self._digest.update(storage)
        self._bytes += len(storage)
        self._tail = bytes(value[end:])

    def identity(self, expected):
        if self._tail or self._bytes != expected:
            raise ValueError("initializer identity has incomplete original typed storage")
        return self._digest.hexdigest()


def new_hex_storage():
    return _HexStorage(_TOKEN)


def identities(module, constants, *, max_decoded_bytes):
    """Bind original streamed payloads to ALL independently checked AST owners."""
    from . import operator_dense_projection as P

    P.validate_module(module, constants, max_decoded_bytes=max_decoded_bytes)
    result = {}
    for op in module.walk():
        marker = op.attributes.get(P.MARKER)
        if marker is None:
            continue
        typ = op.results[0].type
        record = constants[marker.value.data]
        count = prod(typ.get_shape())
        width = P._WIDTHS[str(typ.get_element_type())]
        if count * width > 6 << 30:
            raise ValueError("initializer identity exceeds its explicit streamed storage work bound")
        if record["hex_chars"] is not None:
            storage = record.get("storage")
            if type(storage) is not _HexStorage:
                raise ValueError("initializer identity lost its original streaming owner")
            digest = storage.identity(count * width)
        else:
            literal = record["literal"].decode("ascii").strip()
            if literal.startswith("[") or count == 0:
                if count * width > max_decoded_bytes:
                    raise ValueError("initializer identity exceeds its bounded literal allocation")
                parsed = P._literal_module(literal, str(typ))
                attr = next(v.value for v in parsed.walk() if v.name == "arith.constant")
                digest = hashlib.sha256(attr.data.data).hexdigest()
            else:
                parsed = P._literal_module(literal, f"tensor<1x{typ.get_element_type()}>")
                attr = next(v.value for v in parsed.walk() if v.name == "arith.constant")
                # No dense expansion. Hash complete repeated storage in bounded chunks.
                piece = attr.data.data
                if len(piece) != width:
                    raise ValueError("initializer identity has unsupported scalar storage")
                digestor = hashlib.sha256()
                chunk = piece * min(count, 65536 // width)
                remaining = count * width
                while remaining:
                    take = min(remaining, len(chunk))
                    digestor.update(chunk[:take])
                    remaining -= take
                digest = digestor.hexdigest()
        result[op] = dict(type=str(typ), storage_sha256=digest)
    return result


def direct_identity(op):
    """Canonical storage identity for a bounded original, unprojected constant."""
    value = op.properties.get("value")
    if not isinstance(value, builtin.DenseIntOrFPElementsAttr):
        return None
    if len(value.data.data) > 4 << 20:
        raise ValueError("captured initializer exceeds the bounded original literal reader")
    return dict(type=str(op.results[0].type), storage_sha256=hashlib.sha256(value.data.data).hexdigest())
