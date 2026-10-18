"""Deterministic YAML load/dump wrappers.

TargetGen emits YAML artifacts that must be byte-stable across runs (so diffs are
meaningful and tests are reproducible). ``dump_yaml`` therefore sorts keys and disables
PyYAML's line-wrapping and aliasing. Stdlib + PyYAML only.
"""

from __future__ import annotations

import copy
import hashlib
import threading
from pathlib import Path
from typing import Any

import yaml


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
