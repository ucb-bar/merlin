"""Observe explicitly selected plain source declarations as conditional bit relations.

Only directly allocated literal-width or explicitly selected fixed-width fields
and one exact reinterpret cast are supported. Packing rules are pinned selected
premises, not inferred Scala semantics. This grants no native ABI, source/binary,
elaboration, HW, instruction-length, ELF, policy, effect or runtime authority.
"""

from __future__ import annotations

import hashlib
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .source_predicates import SourcePredicateError, _active_source

SCHEMA = "merlin.conditional_plain_word_relation.v1"
_UNKNOWN = (
    "primitive_source_semantic_review",
    "complete_source_scope_and_language_semantics",
    "compiler_plugin_and_binary_correspondence",
    "source_elaboration_correspondence",
    "original_hw_word_and_field_occurrence_correspondence",
    "raw_storage_to_cast_value_correspondence",
    "instruction_length_and_complete_executable_walk",
    "elf_machine_and_storage_abi",
    "complete_prohibited_source_role_policy",
    "command_effects_acceptance_completion_payload_reuse",
    "observer_physical_resource_and_timing",
)
_PREMISE_ROLES = frozenset({"field_roster", "allocation_order", "field_order", "flatten", "cast", "slice"})


class WordRelationError(ValueError):
    """The selected bounded declaration/cast cannot be observed exactly."""


@dataclass(frozen=True)
class WordSourceSpan:
    path: Path
    sha256: str
    first_line: int
    last_line: int


@dataclass(frozen=True)
class WordPrimitive:
    constructor: str
    width_kind: str
    width_suffix: str | None
    fixed_width: int | None
    construction_source: WordSourceSpan
    width_source: WordSourceSpan


@dataclass(frozen=True)
class WordPackingPremise:
    role: str
    source: WordSourceSpan


@dataclass(frozen=True)
class WordSourceSemantics:
    bundle_base: str
    field_order: str
    slice_order: str
    cast_method: str
    primitives: tuple[WordPrimitive, ...]
    premises: tuple[WordPackingPremise, ...]


@dataclass(frozen=True)
class WordDeclarationSelection:
    source: WordSourceSpan
    class_name: str


@dataclass(frozen=True)
class WordCastSelection:
    source: WordSourceSpan
    receiver: str
    destination: str


@dataclass(frozen=True)
class WordRelationLimits:
    source_bytes: int
    aggregate_source_bytes: int
    tokens: int
    fields: int
    word_bits: int
    nesting: int

    def verify(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise WordRelationError("word relation budgets must be explicit positive integers")
        # Count/read representations are bounded independently of target word or
        # physical resource authority. The extra byte detects truncated reads.
        if any(value >= sys.maxsize for value in asdict(self).values()):
            raise WordRelationError("word relation budgets must be representable by the bounded reader")


def _identifier(value):
    return (
        type(value) is str
        and bool(value)
        and value.isascii()
        and (value[0].isalpha() or value[0] == "_")
        and all(character.isalnum() or character == "_" for character in value)
    )


def _path_tokens(value):
    if type(value) is not str or not all(_identifier(part) for part in value.split(".")):
        raise WordRelationError("cast selection requires exact plain member paths")
    result = []
    for part in value.split("."):
        if result:
            result.append(".")
        result.append(part)
    return result


class _Sources:
    def __init__(self, spans, limits):
        self.limits, self.sources, self.span_records = limits, {}, []
        declarations = {}
        for span in spans:
            if type(span) is not WordSourceSpan or not isinstance(span.path, Path):
                raise WordRelationError("source selection requires exact source spans")
            path = span.path.absolute()
            if ".." in path.parts or any(member.is_symlink() for member in (path, *path.parents)):
                raise WordRelationError("source selection requires canonical unlinked files")
            if (
                type(span.sha256) is not str
                or len(span.sha256) != 64
                or any(character not in "0123456789abcdef" for character in span.sha256)
                or type(span.first_line) is not int
                or type(span.last_line) is not int
                or not 1 <= span.first_line <= span.last_line
            ):
                raise WordRelationError("source hash or original line selection is malformed")
            if path in declarations and declarations[path] != span.sha256:
                raise WordRelationError("one source has conflicting exact byte selections")
            declarations[path] = span.sha256
        total = 0
        for path, digest in declarations.items():
            try:
                metadata = path.stat()
                if not stat.S_ISREG(metadata.st_mode):
                    raise WordRelationError("selected source must be a regular file")
                size = metadata.st_size
                total += size
                if size > limits.source_bytes or total > limits.aggregate_source_bytes:
                    raise WordRelationError("complete selected source byte budget exceeded")
                with path.open("rb") as stream:
                    raw = stream.read(limits.source_bytes + 1)
            except OSError as error:
                raise WordRelationError("selected source file is unavailable") from error
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
                raise WordRelationError("selected complete source bytes changed")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise WordRelationError("selected source is not UTF-8") from error
            self.sources[path] = (digest, text)
        for span in spans:
            self.fragment(span)

    def fragment(self, span):
        path = span.path.absolute()
        digest, text = self.sources[path]
        lines = text.splitlines(keepends=True)
        if span.last_line > len(lines):
            raise WordRelationError("original source span lies outside complete source")
        fragment = "".join(lines[span.first_line - 1 : span.last_line])
        record = {
            "path": str(path),
            "sha256": digest,
            "first_line": span.first_line,
            "last_line": span.last_line,
            "span_sha256": hashlib.sha256(fragment.encode()).hexdigest(),
        }
        if record not in self.span_records:
            self.span_records.append(record)
        return fragment

    def active_fragment(self, span):
        try:
            text = _active_source(self.sources[span.path.absolute()][1])
        except SourcePredicateError as error:
            raise WordRelationError("selected source has unavailable lexical membership") from error
        return "".join(text.splitlines(keepends=True)[span.first_line - 1 : span.last_line])

    def verify(self):
        for path, (digest, _) in self.sources.items():
            if any(member.is_symlink() for member in (path, *path.parents)):
                raise WordRelationError("selected source path changed during observation")
            try:
                if not stat.S_ISREG(path.stat().st_mode):
                    raise WordRelationError("selected source must remain a regular file")
                with path.open("rb") as stream:
                    raw = stream.read(self.limits.source_bytes + 1)
            except OSError as error:
                raise WordRelationError("selected source became unavailable") from error
            if hashlib.sha256(raw).hexdigest() != digest:
                raise WordRelationError("selected source bytes changed during observation")

    def record(self):
        return {
            "files": [{"path": str(path), "sha256": digest} for path, (digest, _) in self.sources.items()],
            "spans": self.span_records,
        }


def _tokens(text, limits):
    result, position, stack = [], 0, []
    closes = {")": "(", "}": "{", "]": "["}
    while position < len(text):
        character = text[position]
        if character.isspace():
            position += 1
            continue
        end = position + 1
        if character.isascii() and (character.isalpha() or character == "_"):
            while end < len(text) and text[end].isascii() and (text[end].isalnum() or text[end] == "_"):
                end += 1
        elif character in "0123456789":
            while end < len(text) and text[end] in "0123456789":
                end += 1
        elif text.startswith(":=", position):
            end += 1
        token = text[position:end]
        result.append(token)
        if len(result) > limits.tokens:
            raise WordRelationError("selected syntax token budget exceeded")
        if token in {"(", "{", "["}:
            stack.append(token)
            if len(stack) > limits.nesting:
                raise WordRelationError("selected syntax nesting budget exceeded")
        elif token in closes:
            if not stack or stack.pop() != closes[token]:
                raise WordRelationError("selected syntax has mismatched boundaries")
        position = end
    if stack:
        raise WordRelationError("selected syntax boundary is incomplete")
    return result


class _Parser:
    def __init__(self, tokens):
        self.tokens, self.position = tokens, 0

    def take(self, expected=None):
        if self.position == len(self.tokens):
            raise WordRelationError("selected declaration is incomplete")
        value = self.tokens[self.position]
        self.position += 1
        if expected is not None and value != expected:
            raise WordRelationError("selected declaration/cast has unsupported syntax or identity")
        return value

    def done(self):
        if self.position != len(self.tokens):
            raise WordRelationError("selected declaration/cast has unconsumed source")


def _selection(semantics, declaration, cast, limits):
    if (
        type(semantics) is not WordSourceSemantics
        or type(declaration) is not WordDeclarationSelection
        or type(cast) is not WordCastSelection
        or type(limits) is not WordRelationLimits
    ):
        raise WordRelationError("word relation requires exact explicit typed selections")
    limits.verify()
    if (
        not all(_identifier(value) for value in (semantics.bundle_base, semantics.cast_method, declaration.class_name))
        or type(semantics.field_order) is not str
        or semantics.field_order not in {"definition", "reverse_definition"}
        or type(semantics.slice_order) is not str
        or semantics.slice_order not in {"low_to_high", "high_to_low"}
        or type(semantics.primitives) is not tuple
        or not semantics.primitives
        or len(semantics.primitives) > limits.fields
        or type(semantics.premises) is not tuple
        or len(semantics.premises) != len(_PREMISE_ROLES)
        or any(type(premise) is not WordPackingPremise for premise in semantics.premises)
        or any(type(premise.role) is not str for premise in semantics.premises)
        or {premise.role for premise in semantics.premises} != _PREMISE_ROLES
    ):
        raise WordRelationError("word relation primitive/packing premises are incomplete")
    primitives = {}
    for primitive in semantics.primitives:
        if type(primitive) is not WordPrimitive or not _identifier(primitive.constructor):
            raise WordRelationError("primitive constructor selection is malformed")
        if primitive.constructor in primitives:
            raise WordRelationError("primitive constructor selection is duplicated")
        if primitive.width_kind == "literal":
            valid = _identifier(primitive.width_suffix) and primitive.fixed_width is None
        elif primitive.width_kind == "fixed":
            valid = (
                primitive.width_suffix is None
                and type(primitive.fixed_width) is int
                and 0 < primitive.fixed_width <= limits.word_bits
            )
        else:
            valid = False
        if not valid:
            raise WordRelationError("primitive width premise is unavailable")
        primitives[primitive.constructor] = primitive
    _path_tokens(cast.receiver)
    _path_tokens(cast.destination)
    return primitives


def _declaration(tokens, declaration, semantics, primitives, limits):
    parser = _Parser(tokens)
    for expected in ("class", declaration.class_name, "extends"):
        parser.take(expected)
    parser.take(semantics.bundle_base)
    parser.take("{")
    fields, total = [], 0
    while parser.tokens[parser.position : parser.position + 1] != ["}"]:
        parser.take("val")
        name = parser.take()
        if not _identifier(name) or any(row["name"] == name for row in fields):
            raise WordRelationError("original field identity is invalid or duplicated")
        parser.take("=")
        constructor = parser.take()
        if constructor not in primitives:
            raise WordRelationError("original field uses an unselected primitive")
        primitive = primitives[constructor]
        parser.take("(")
        if primitive.width_kind == "literal":
            digits = parser.take()
            if (
                not digits
                or any(character not in "0123456789" for character in digits)
                or len(digits) > len(str(limits.word_bits))
            ):
                raise WordRelationError("literal field width is unavailable or exceeds budget")
            width = int(digits)
            parser.take(".")
            parser.take(primitive.width_suffix)
        else:
            width = primitive.fixed_width
        parser.take(")")
        total += width
        if width <= 0 or total > limits.word_bits or len(fields) >= limits.fields:
            raise WordRelationError("complete original field/word budget exceeded")
        fields.append({"name": name, "ordinal": len(fields), "constructor": constructor, "width": width})
        if parser.tokens[parser.position : parser.position + 1] == [";"]:
            parser.take(";")
    parser.take("}")
    parser.done()
    if not fields:
        raise WordRelationError("plain word declaration has no original fields")
    packed = fields if semantics.field_order == "definition" else list(reversed(fields))
    cursor = 0 if semantics.slice_order == "low_to_high" else total
    for field in packed:
        if semantics.slice_order == "high_to_low":
            cursor -= field["width"]
        field["low_bit"] = cursor
        if semantics.slice_order == "low_to_high":
            cursor += field["width"]
    return total, fields


def _record(value):
    if isinstance(value, Path):
        return str(value.absolute())
    if isinstance(value, dict):
        return {key: _record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_record(item) for item in value]
    return value


def observe_plain_word_relation(*, declaration, cast, semantics, limits):
    """Retain complete field intervals conditional on explicit source packing premises.

    Source bytes and syntax are checked here. Selected primitive/packing premise
    interpretation, language scope, binary, HW and ELF correspondence stay unknown.
    """
    primitives = _selection(semantics, declaration, cast, limits)
    spans = (declaration.source, cast.source)
    spans += tuple(premise.source for premise in semantics.premises)
    spans += tuple(
        span for primitive in semantics.primitives for span in (primitive.construction_source, primitive.width_source)
    )
    sources = _Sources(spans, limits)
    try:
        complete = _tokens(_active_source(sources.sources[declaration.source.path.absolute()][1]), limits)
    except SourcePredicateError as error:
        raise WordRelationError("selected declaration has unavailable lexical membership") from error
    if sum(complete[index : index + 2] == ["class", declaration.class_name] for index in range(len(complete))) != 1:
        raise WordRelationError("selected original class identity is absent or duplicated")
    tokens = _tokens(sources.active_fragment(declaration.source), limits)
    word_bits, fields = _declaration(tokens, declaration, semantics, primitives, limits)
    expected = [
        *_path_tokens(cast.destination),
        ":=",
        *_path_tokens(cast.receiver),
        ".",
        semantics.cast_method,
        "(",
        "new",
        declaration.class_name,
        "(",
        ")",
        ")",
    ]
    if _tokens(sources.active_fragment(cast.source), limits) != expected:
        raise WordRelationError("selected cast does not bind the exact source and complete declared class")
    sources.verify()
    return {
        "schema": SCHEMA,
        "status": "conditional_source_relation",
        "scope": "bounded selected plain declaration and cast; explicit packing premises only",
        "class_name": declaration.class_name,
        "word_bits": word_bits,
        "fields": fields,
        "cast": {"receiver": cast.receiver, "destination": cast.destination, "method": semantics.cast_method},
        "selection": _record(
            {
                "declaration": asdict(declaration),
                "cast": asdict(cast),
                "semantics": asdict(semantics),
            }
        ),
        "source_membership": sources.record(),
        "limits": asdict(limits),
        "required_unknowns": list(_UNKNOWN),
        "capabilities_issued": 0,
    }
