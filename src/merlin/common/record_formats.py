"""Structured record readers: strict JSON, deterministic YAML, schema validation, lexical tokens and the
regex-call scan.

Each reader fails closed on input it cannot represent exactly; none of them interprets the records it
reads.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# ---- strict_json -------------------------------------------------------------------------------
# Bounded, unambiguous JSON parsing for records used as authority.

MAX_RECORD_BYTES = 128 * 1024 * 1024


def loads(raw: bytes | str, *, max_bytes: int = MAX_RECORD_BYTES) -> Any:
    """Reject duplicate keys and non-finite numbers before interpreting claims.

    The bound applies to encoded input bytes. This reader grants no provenance or
    admission; consumers still validate their independently selected identities.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("JSON reader limit must be a nonnegative integer")
    if not isinstance(raw, (bytes, str)):
        raise TypeError("JSON record must be bytes or text")
    if len(raw) > max_bytes or (isinstance(raw, str) and len(raw.encode("utf-8")) > max_bytes):
        raise ValueError("JSON record exceeds the bounded reader limit")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON value: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON value: {value}")
        return parsed

    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant, parse_float=finite_float)


# ---- yaml --------------------------------------------------------------------------------------
# Deterministic YAML load/dump wrappers.
#
# TargetGen emits YAML artifacts that must be byte-stable across runs (so diffs are
# meaningful and tests are reproducible). ``dump_yaml`` therefore sorts keys and disables
# PyYAML's line-wrapping and aliasing. Stdlib + PyYAML only.


def load_yaml(path: str | Path) -> Any:
    """Parse a YAML file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


#: Parsed documents by the sha256 of the bytes they were parsed from. A build reads the same few
#: contract and provider files thousands of times (measured: 1,800 parses, 37 of a 49 s whole-model
#: statement); the bytes are re-read and re-hashed every time, so an edited file is a new key and a
#: cached document is never served for bytes it was not parsed from.
_PARSED: dict[str, Any] = {}
_PARSED_LOCK = threading.Lock()
_PARSED_MAX = 256


#: libyaml's scanner and parser under PyYAML's own safe constructor: the same documents, built by the
#: same Python code, about six times faster (measured: a 22 MB golden in 2.6 s against 16.5 s).
_FAST_LOADER = getattr(yaml, "CSafeLoader", None)


def fast_safe_load(data: str | bytes) -> Any:
    """``yaml.safe_load(data)``'s value, scanned and parsed by libyaml when PyYAML was built with it.

    The constructor and implicit-tag resolver are PyYAML's own safe ones either way, so the same text
    builds the same document. A document libyaml refuses is parsed again by PyYAML, so a malformed one
    raises exactly the error (message and source labels) it always did; without the C extension this
    is plain ``yaml.safe_load``. Nothing is cached.
    """
    if _FAST_LOADER is not None:
        try:
            return yaml.load(data, Loader=_FAST_LOADER)  # noqa: S506 -- CSafeLoader is the safe loader
        except yaml.YAMLError:
            pass
    return yaml.safe_load(data)


def _parse(text: str) -> Any:
    """``yaml.safe_load(text)``'s value (see :func:`fast_safe_load`)."""
    return fast_safe_load(text)


def safe_load_text(text: str) -> Any:
    """``yaml.safe_load(text)``, parsed once per distinct text; every caller gets its own copy.

    Exactly the stdlib-PyYAML result (the same errors raise, and a failed parse is never cached); the
    copy means a caller that mutates what it got cannot change what the next caller reads.
    """
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    with _PARSED_LOCK:
        hit = _PARSED.get(key, _PARSED_LOCK)
    if hit is not _PARSED_LOCK:
        return copy.deepcopy(hit)
    parsed = _parse(text)
    with _PARSED_LOCK:
        if len(_PARSED) >= _PARSED_MAX:
            _PARSED.pop(next(iter(_PARSED)))
        _PARSED[key] = parsed
    return copy.deepcopy(parsed)


def dump_yaml(obj: Any) -> str:
    """Serialize ``obj`` to a deterministic YAML string.

    Keys are sorted, flow style is block, lines are not wrapped, and no anchors/aliases are
    emitted. The same object always yields the same bytes.
    """
    return yaml.safe_dump(
        obj,
        sort_keys=True,
        default_flow_style=False,
        width=10**9,
        allow_unicode=True,
    )


def write_yaml(path: str | Path, obj: Any, header: str | None = None) -> Path:
    """Write ``obj`` as deterministic YAML to ``path`` (creating parents).

    An optional ``header`` comment line is prepended (without the leading ``# ``; it is
    added here). Returns the written path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = dump_yaml(obj)
    if header:
        text = f"# {header}\n{text}"
    p.write_text(text, encoding="utf-8")
    return p


# ---- schemas -----------------------------------------------------------------------------------
# Load and validate artifacts against the YAML schemas in ``src/merlin/resources/schemas/``.
#
# The schemas are intentionally lightweight (see ``src/merlin/resources/schemas/AGENT.md``): each is a
# YAML doc with a ``title``, ``purpose``, a ``required_top_level_fields`` list, and an
# ``example`` block. They are *not* formal JSON Schema yet. Validation here therefore means:
# "the object is a mapping that contains every required top-level field." That is enough to
# keep cross-workstream artifacts honest without over-engineering a type system.
#
# This module is the single dependency every kernel-mining emitter relies on, so it stays
# dependency-light (stdlib + PyYAML) and side-effect free.


def schemas_dir() -> Path:
    """Return the directory holding ``*.schema.yaml`` files.

    Delegates to :func:`merlin.common.paths.schemas_dir` — honors ``MERLIN_SCHEMAS_DIR``, resolves
    the in-repo ``<repo>/src/merlin/resources/schemas`` when a checkout is present, and falls back to the copy
    bundled into an installed wheel (``merlin/_data/schemas``).
    """
    from merlin.common.paths import schemas_dir as _schemas_dir

    return _schemas_dir()


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict[str, Any]:
    """Load a schema by short name (e.g. ``"kernel_record"``).

    Accepts either the short name or a full ``<name>.schema.yaml`` filename.
    """
    stem = name[: -len(".schema.yaml")] if name.endswith(".schema.yaml") else name
    path = schemas_dir() / f"{stem}.schema.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"schema not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"schema {path} did not parse to a mapping")
    return data


def required_fields(name: str) -> list[str]:
    """Return the ``required_top_level_fields`` list for a schema."""
    return list(load_schema(name).get("required_top_level_fields", []))


def validate(obj: Any, schema_name: str) -> list[str]:
    """Validate ``obj`` against a schema. Return a list of problems (empty == valid).

    Checks that ``obj`` is a mapping and contains every required top-level field with a
    non-None value. Does not (yet) type-check nested fields.
    """
    problems: list[str] = []
    if not isinstance(obj, dict):
        return [f"expected a mapping for schema '{schema_name}', got {type(obj).__name__}"]
    for field in required_fields(schema_name):
        if field not in obj:
            problems.append(f"missing required field '{field}'")
        elif obj[field] is None:
            problems.append(f"required field '{field}' is null")
    return problems


def validate_or_raise(obj: Any, schema_name: str) -> None:
    """Validate and raise ``ValueError`` listing all problems if invalid."""
    problems = validate(obj, schema_name)
    if problems:
        joined = "; ".join(problems)
        raise ValueError(f"{schema_name} validation failed: {joined}")


# ---- text_tokens -------------------------------------------------------------------------------
# Small lexical primitives for explicitly bounded source-format readers.
#
# These tokens preserve quoted literals and discard C-style comments. Callers
# must validate the complete grammar they accept. Lexing supplies no semantics.


def is_identifier(value: object) -> bool:
    """Whether a value is a nonempty portable C/LLVM or shell identifier."""
    return isinstance(value, str) and value.isascii() and value.isidentifier()


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def source_tokens(text: str) -> list[Token]:
    """Lex words, quoted literals and individual punctuation, refusing open literals/comments."""
    tokens, position = [], 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        if text.startswith("//", position):
            end = text.find("\n", position + 2)
            position = len(text) if end < 0 else end + 1
            continue
        if text.startswith("/*", position):
            end = text.find("*/", position + 2)
            if end < 0:
                raise ValueError("unterminated source comment")
            position = end + 2
            continue
        start = position
        if text[position] in "\"'":
            quote = text[position]
            position += 1
            while position < len(text) and text[position] != quote:
                position += 2 if text[position] == "\\" else 1
            if position >= len(text):
                raise ValueError("unterminated source literal")
            position += 1
        elif text[position].isalnum() or text[position] in "_$":
            position += 1
            while position < len(text) and (text[position].isalnum() or text[position] in "_$"):
                position += 1
        else:
            position += 1
        tokens.append(Token(text[start:position], start, position))
    return tokens


# ---- regex_scan --------------------------------------------------------------------------------
# Structural Python regex-call findings, without repository or grading exemptions.
#
# This is the shared AST mechanism, not a sandbox or complete dynamic-import analysis.
# Callers own exemptions and failure policy. Malformed Python and unreadable files raise;
# absence of inspection must never be indistinguishable from an empty finding list.

REGEX_FUNCS = frozenset(
    {
        "compile",
        "match",
        "search",
        "fullmatch",
        "sub",
        "subn",
        "findall",
        "finditer",
        "split",
        "escape",
    }
)


class _RegexVisitor(ast.NodeVisitor):
    """Collect line numbers of ``re``-module call sites, following the file's import aliases."""

    def __init__(self) -> None:
        self.aliases: set[str] = set()  # module aliases bound to `re` (e.g. {"re", "_re"})
        self.from_funcs: dict[str, str] = {}  # local binding -> canonical function imported from re
        self.hits: list[tuple[int, str]] = []  # (lineno, what)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "re":
                self.aliases.add(alias.asname or "re")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "re":
            for alias in node.names:
                self.from_funcs[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        # <alias>.<func>(...)
        if (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Name)
            and fn.value.id in self.aliases
            and fn.attr in REGEX_FUNCS
        ):
            self.hits.append((fn.lineno, f"{fn.value.id}.{fn.attr}"))
        # bare <func>(...) from `from re import <func>`
        elif isinstance(fn, ast.Name) and self.from_funcs.get(fn.id) in REGEX_FUNCS:
            self.hits.append((fn.lineno, fn.id))
        self.generic_visit(node)


def scan_source(source: str, *, filename: str = "<unknown>") -> list[tuple[int, str]]:
    """Return ordered distinct ``(line, call)`` findings; do not honor inline markers."""
    tree = ast.parse(source, filename=filename)
    visitor = _RegexVisitor()
    visitor.visit(tree)
    return sorted(set(visitor.hits))


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Scan UTF-8 source using the repository's historical replacement decoding."""
    return scan_source(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
