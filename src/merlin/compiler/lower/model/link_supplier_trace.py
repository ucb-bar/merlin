"""Opt-in, linker-observed defining-symbol suppliers for explicit link inputs.

This recognizes one conservative GNU-compatible ``--trace-symbol`` diagnostic
grammar. An unsupported linker, missing trace, duplicate definition, indirect
library search, or changed input refuses. It proves neither that source calls
reach the symbol nor numerical equivalence of the selected implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_file

SCHEMA = "merlin.link_supplier_trace.v1"
SCOPE = "observed defining supplier in selected link; source-call and numerical proof excluded"
_MAX_TRACE_BYTES = 65536
_MAX_SYMBOLS = 64
_SYMBOL_HEAD = "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_SYMBOL_TAIL = _SYMBOL_HEAD + "0123456789"
_MEMBER_CHARS = _SYMBOL_TAIL + ".-+"


def _symbol(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or value[0] not in _SYMBOL_HEAD
        or any(character not in _SYMBOL_TAIL for character in value)
    ):
        raise ValueError("link supplier symbol is not a simple C symbol")
    return value


def trace_symbol_flags(symbols: Sequence[str]) -> tuple[str, ...]:
    """Flags for the *actual* link, never a post-link diagnostic rerun."""
    if isinstance(symbols, (str, bytes)) or not 1 <= len(symbols) <= _MAX_SYMBOLS:
        raise ValueError("link supplier proof needs a bounded nonempty symbol roster")
    selected = [_symbol(symbol) for symbol in symbols]
    if len(set(selected)) != len(selected):
        raise ValueError("link supplier proof repeats a symbol")
    return tuple(f"-Wl,--trace-symbol={symbol}" for symbol in selected)


def _definition(stderr: str, symbol: str) -> tuple[str, str]:
    suffix = f": definition of {symbol}"
    definitions = []
    for line in stderr.splitlines():
        if line.endswith(suffix):
            prefix = line[: -len(suffix)]
            # GNU ld and lld prefix diagnostics with their own executable path.
            # A path containing this delimiter is intentionally unsupported.
            if ": " not in prefix:
                raise ValueError("link supplier trace has no linker-owned origin")
            definitions.append(tuple(prefix.rsplit(": ", 1)))
    if len(definitions) != 1:
        raise ValueError(f"link supplier trace has no unique defining supplier for {symbol}")
    return definitions[0]


def _emitter_identity(name: str) -> dict[str, Any]:
    printed = Path(name)
    if not printed.is_absolute() or any(character in name for character in "\r\n\x00"):
        raise ValueError("link supplier trace has no absolute diagnostic emitter")
    resolved = printed.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("link supplier diagnostic emitter is not a file")
    return {
        "printed_path": name,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def observe_link_suppliers(
    *,
    argv: Sequence[str],
    input_identities: Sequence[dict[str, Any]],
    expected_suppliers: Mapping[str, Path],
    stderr: str,
) -> dict[str, Any]:
    """Parse the completed link's captured trace and bind exact selected files.

    The caller must independently select ``expected_suppliers``. Every expected
    path must be an explicit, already byte-pinned link input; ``-l`` search and
    merely present but unextracted archives never satisfy a requested symbol.
    """
    if not isinstance(expected_suppliers, Mapping):
        raise ValueError("link supplier proof needs selected suppliers")
    symbols = list(expected_suppliers)
    flags = trace_symbol_flags(symbols)
    if any(argv.count(flag) != 1 for flag in flags):
        raise ValueError("link supplier proof was not traced by the actual link")
    if not isinstance(stderr, str) or "\x00" in stderr or len(stderr.encode("utf-8")) > _MAX_TRACE_BYTES:
        raise ValueError("link supplier trace is absent or oversized")
    identities = {identity["path"]: identity for identity in input_identities}
    observed = {}
    emitter = None
    for symbol in symbols:
        selected = Path(expected_suppliers[symbol])
        if (
            not selected.is_absolute()
            or selected.is_symlink()
            or any(character in str(selected) for character in "\r\n\x00()")
            or ": " in str(selected)
        ):
            raise ValueError("link supplier has no direct selected input")
        selected_path = str(selected.resolve(strict=True))
        if selected_path not in identities:
            raise ValueError("link supplier has no exact selected link input")
        emitter_name, definition = _definition(stderr, symbol)
        selected_emitter = _emitter_identity(emitter_name)
        if emitter is not None and selected_emitter != emitter:
            raise ValueError("link supplier trace came from multiple diagnostic emitters")
        emitter = selected_emitter
        member = definition[len(selected_path) + 1 : -1]
        if definition != selected_path and not (
            definition.startswith(selected_path + "(")
            and definition.endswith(")")
            and member
            and all(character in _MEMBER_CHARS for character in member)
        ):
            raise ValueError(f"link supplier trace names another defining supplier for {symbol}")
        observed[symbol] = {"selected_input": identities[selected_path], "definition": definition}
    return {"schema": SCHEMA, "scope": SCOPE, "emitter": emitter, "symbols": observed, "stderr": stderr}


def verify_link_suppliers(
    observation: object,
    *,
    argv: Sequence[str],
    input_identities: Sequence[dict[str, Any]],
    expected_suppliers: Mapping[str, Path] | None,
) -> dict[str, Any]:
    """Reparse the recorded observation and check an independent caller selection."""
    if (
        not isinstance(observation, dict)
        or set(observation) != {"schema", "scope", "emitter", "symbols", "stderr"}
        or observation.get("schema") != SCHEMA
        or observation.get("scope") != SCOPE
        or not isinstance(observation.get("symbols"), dict)
    ):
        raise ValueError("link supplier proof has no complete observed record")
    recorded = observation["symbols"]
    if any(
        not isinstance(item, dict)
        or set(item) != {"selected_input", "definition"}
        or not isinstance(item["selected_input"], dict)
        or not isinstance(item["selected_input"].get("path"), str)
        for item in recorded.values()
    ):
        raise ValueError("link supplier proof has malformed selected inputs")
    selected = {symbol: Path(item["selected_input"]["path"]) for symbol, item in recorded.items()}
    reproduced = observe_link_suppliers(
        argv=argv,
        input_identities=input_identities,
        expected_suppliers=selected,
        stderr=observation["stderr"],
    )
    if reproduced != observation:
        raise ValueError("link supplier proof differs from recorded linker trace")
    if expected_suppliers is not None:
        if not isinstance(expected_suppliers, Mapping):
            raise ValueError("link supplier proof has no independently selected supplier roster")
        expected = {symbol: Path(path) for symbol, path in expected_suppliers.items()}
        if (
            set(expected) != set(selected)
            or any(
                not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path
                for path in expected.values()
            )
            or expected != selected
        ):
            raise ValueError("link supplier proof differs from independently selected supplier")
    return observation
