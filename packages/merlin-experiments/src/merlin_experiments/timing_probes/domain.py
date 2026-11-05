"""The Phase 2 operation domain as whole-call quantity ranges (MACs, bytes read, bytes written).

The domain is read from a corpus ledger of declared operations (``{"entries": [...]}``, the Phase 0
release's ``phase2-deferred.json`` form) or from an explicit ranges document. Volumes are the
declared operation's minimum work -- every input read once, every output written once -- computed
from framework-level operation semantics (matmul, 2-D convolution, elementwise binary add). An
entry whose operation or shape this module cannot quantify is counted and reported, never guessed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RANGES = "merlin.timing_probe_domain.v1"
QUANTITIES = ("macs", "read_bytes", "write_bytes")


def _bytes(dtype) -> int | None:
    text = str(dtype or "").lower()
    for prefix in ("int", "uint", "float", "i", "u", "f"):
        if text.startswith(prefix) and text[len(prefix) :].isdigit():
            bits = int(text[len(prefix) :])
            return bits // 8 if bits % 8 == 0 and bits > 0 else None
    return None


def _ints(entry: dict, *names: str) -> list[int] | None:
    values = [entry.get(name) for name in names]
    return values if all(type(v) is int and v > 0 for v in values) else None


def _pair(value, default: int) -> tuple[int, int]:
    if isinstance(value, list) and len(value) == 2 and all(type(v) is int for v in value):
        return value[0], value[1]
    return default, default


def operation_volume(entry: dict) -> dict[str, int] | None:
    """``{macs, read_bytes, write_bytes}`` of one declared operation, or None when it is not quantified."""
    width = _bytes(entry.get("operand_dtype"))
    if width is None:
        return None
    op = entry.get("op")
    if op == "matmul":
        dims = _ints(entry, "M", "K", "N")
        if dims is None:
            return None
        m, k, n = dims
        return {"macs": m * k * n, "read_bytes": (m * k + k * n) * width, "write_bytes": m * n * width}
    if op == "conv2d":
        dims = _ints(entry, "Himg", "Wimg", "ci", "kh", "kw", "N")
        if dims is None:
            return None
        h, w, ci, kh, kw, co = dims
        sh, sw = _pair(entry.get("stride"), 1)
        dh, dw = _pair(entry.get("dilation"), 1)
        padding = entry.get("padding") if isinstance(entry.get("padding"), list) else [0, 0, 0, 0]
        if len(padding) != 4 or not all(type(p) is int and p >= 0 for p in padding):
            return None
        ho = (h + padding[0] + padding[1] - dh * (kh - 1) - 1) // sh + 1
        wo = (w + padding[2] + padding[3] - dw * (kw - 1) - 1) // sw + 1
        if ho <= 0 or wo <= 0:
            return None
        return {
            "macs": ho * wo * co * ci * kh * kw,
            "read_bytes": (h * w * ci + kh * kw * ci * co) * width,
            "write_bytes": ho * wo * co * width,
        }
    if op == "residual_add":
        dims = _ints(entry, "M", "N")
        if dims is None:
            return None
        m, n = dims
        return {"macs": 0, "read_bytes": 2 * m * n * width, "write_bytes": m * n * width}
    return None


def from_document(document: dict, *, source: str, sha256: str) -> dict:
    """Quantity ranges from a corpus ledger or an explicit ``merlin.timing_probe_domain.v1`` document."""
    if document.get("schema") == RANGES:
        ranges = document.get("ranges") or {}
        for name, bounds in ranges.items():
            if name not in QUANTITIES or not (
                isinstance(bounds, list) and len(bounds) == 2 and 0 < bounds[0] <= bounds[1]
            ):
                raise ValueError(f"domain range {name!r} must be [min, max] with 0 < min <= max")
        return {
            "schema": RANGES,
            "source": source,
            "source_sha256": sha256,
            "ranges": ranges,
            "entries": document.get("entries_quantified"),
            "unquantified": document.get("unquantified", []),
        }
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{source}: expected a corpus ledger with entries or a {RANGES} document")
    values: dict[str, list[int]] = {q: [] for q in QUANTITIES}
    unquantified = []
    for entry in entries:
        volume = operation_volume(entry) if isinstance(entry, dict) else None
        if volume is None:
            unquantified.append(
                {
                    "name": entry.get("name") if isinstance(entry, dict) else None,
                    "op": entry.get("op") if isinstance(entry, dict) else None,
                }
            )
            continue
        for name, value in volume.items():
            if value > 0:
                values[name].append(value)
    ranges = {q: [min(v), max(v)] for q, v in values.items() if v}
    return {
        "schema": RANGES,
        "source": source,
        "source_sha256": sha256,
        "ranges": ranges,
        "entries_quantified": len(entries) - len(unquantified),
        "unquantified": unquantified,
    }


def load(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = path.read_bytes()
    return from_document(json.loads(raw), source=str(path), sha256=hashlib.sha256(raw).hexdigest())


def targets(domain: dict, budget: dict[str, int]) -> dict[str, dict]:
    """Per quantity: the range labels must span, clipped by an explicit simulation budget."""
    out = {}
    for name, (low, high) in (domain.get("ranges") or {}).items():
        cap = budget.get(name)
        clipped = high if cap is None else min(high, cap)
        out[name] = {"min": low, "max": max(low, clipped), "domain_max": high, "clipped": clipped < high}
    return out
