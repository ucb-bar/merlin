"""Read protected coordinator selections; declarations confer no authority."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from merlin.common.strict_json import loads

SCHEMA = "merlin.private_instruction_selection.v1"
MAX_DECLARATION_BYTES = 1024 * 1024


def _fail():
    raise ValueError("instruction coordinator requires a complete reviewed source declaration")


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        _fail()


def _text(value):
    if type(value) is not str or not value:
        _fail()
    return value


def _ordinary(value, *, directory=False, absent=False):
    path = Path(_text(value))
    if not path.is_absolute() or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        _fail()
    if not absent and not (path.is_dir() if directory else path.is_file()):
        _fail()
    if absent and path.exists() and not path.is_dir():
        _fail()
    return path


def _digest(value, length=64):
    if type(value) is not str or len(value) != length or any(c not in "0123456789abcdef" for c in value):
        _fail()
    return value


def _bytes(path):
    with path.open("rb") as stream:
        raw = stream.read(MAX_DECLARATION_BYTES + 1)
    if len(raw) > MAX_DECLARATION_BYTES:
        _fail()
    return raw


@dataclass(frozen=True)
class Declaration:
    path: Path
    raw: bytes
    pins: tuple[tuple[Path, str], ...]
    roots: tuple[Path, ...]

    def document(self):
        return loads(self.raw, max_bytes=MAX_DECLARATION_BYTES)

    def verify(self):
        if _bytes(_ordinary(str(self.path))) != self.raw:
            raise ValueError("instruction coordinator source declaration changed")
        document = self.document()
        if type(document) is not dict:
            _fail()
        closed = read(self.path, target=document.get("target"))
        if (closed.raw, closed.pins, closed.roots) != (self.raw, self.pins, self.roots):
            raise ValueError("instruction coordinator source declaration or membership changed")


def _reopen(declaration):
    if _bytes(_ordinary(str(declaration.path))) != declaration.raw:
        raise ValueError("instruction coordinator source declaration changed")
    for path, digest in declaration.pins:
        if hashlib.sha256(_ordinary(str(path)).read_bytes()).hexdigest() != digest:
            raise ValueError("instruction coordinator selected source or tool changed")
    for root in declaration.roots:
        _ordinary(str(root), directory=True)


def read(path, *, target):
    """Close a host-only v1 request before delegating any fresh native issuer.

    Reviewed status expresses protected caller selection, not source semantics,
    ISA policy completeness, imported runtime or hardware qualification.
    """
    try:
        selected = _ordinary(str(path))
        raw = _bytes(selected)
        doc = loads(raw, max_bytes=MAX_DECLARATION_BYTES)
        _keys(
            doc,
            (
                "schema",
                "status",
                "target",
                "hardware",
                "command",
                "predicates",
                "accessor",
                "policy",
                "decoder",
                "forbidden_roots",
            ),
        )
        if doc["schema"] != SCHEMA or doc["status"] != "reviewed" or _text(doc["target"]) != target:
            _fail()
        forbidden = doc["forbidden_roots"]
        if type(forbidden) is not list or not forbidden:
            _fail()
        exclusions = tuple(_ordinary(value, absent=True) for value in forbidden)
        pins, roots = [], []

        def pin(value):
            _keys(value, ("path", "sha256"))
            member, digest = _ordinary(value["path"]), _digest(value["sha256"])
            pins.append((member, digest))

        def root(value):
            roots.append(_ordinary(value, directory=True))

        hardware = doc["hardware"]
        _keys(hardware, ("descriptor", "source_bundle"))
        for value in hardware.values():
            pin(value)
        command = doc["command"]
        _keys(command, ("checkout", "commit", "isa_source", "function_span", "circt_opt"))
        root(command["checkout"])
        _digest(command["commit"], 40)
        pin(command["isa_source"])
        pin(command["circt_opt"])
        if not Path(command["isa_source"]["path"]).is_relative_to(Path(command["checkout"])):
            _fail()
        span = command["function_span"]
        if type(span) is not list or len(span) != 2:
            _fail()
        for value in span:
            _text(value)
        predicates = doc["predicates"]
        if type(predicates) is not list or not predicates:
            _fail()
        identities = set()
        for row in predicates:
            _keys(row, ("source", "binding", "operand"))
            pin(row["source"])
            if not Path(row["source"]["path"]).is_relative_to(Path(command["checkout"])):
                _fail()
            identity = (row["source"]["path"], _text(row["binding"]), _text(row["operand"]))
            if identity in identities:
                _fail()
            identities.add(identity)
        accessor = doc["accessor"]
        _keys(accessor, ("checkout", "commit", "include_root", "native_compiler", "reviewed_spec"))
        root(accessor["checkout"])
        root(accessor["include_root"])
        _digest(accessor["commit"], 40)
        pin(accessor["native_compiler"])
        pin(accessor["reviewed_spec"])
        pin(doc["policy"])
        decoder = doc["decoder"]
        _keys(decoder, ("match_constant", "mask_constant", "selector_member", "expected_elf_machine"))
        for name in ("match_constant", "mask_constant", "selector_member"):
            _text(decoder[name])
        if type(decoder["expected_elf_machine"]) is not int or decoder["expected_elf_machine"] < 0:
            _fail()
        for member in (selected, *(path for path, _ in pins), *roots):
            if any(member == denied or member.is_relative_to(denied) for denied in exclusions):
                _fail()
        if len(dict(pins)) != len(pins):
            # Shared byte-identical files can supply distinct input roles.
            if any(len({digest for path, digest in pins if path == member}) != 1 for member, _ in pins):
                _fail()
        declaration = Declaration(selected, raw, tuple(pins), tuple(roots))
        _reopen(declaration)
        return declaration
    except (OSError, TypeError, KeyError, ValueError) as error:
        raise ValueError("instruction coordinator requires unchanged reviewed public source selections") from error
