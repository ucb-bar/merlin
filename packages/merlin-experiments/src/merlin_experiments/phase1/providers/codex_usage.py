"""Convert reported provider usage while preserving unreported token accounting."""
from __future__ import annotations


def usage_to_claude_shape(usage: dict) -> tuple[dict, bool]:
    """Translate a Codex ``turn.completed`` usage payload to the Claude shape.

    Returns ``(usage_dict, reported)``. Codex's ``input_tokens`` is a TOTAL that
    already contains the cache reads and writes; the Claude shape's
    ``input_tokens`` is the uncached remainder. Subtracting is therefore the
    correct translation, clamped at zero so a provider inconsistency cannot
    produce a negative count. When nothing was reported, ``reported`` is False
    and the caller must omit usage rather than emit zeros.
    """

    def _int(key: str) -> int | None:
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(value)

    total_in = _int("input_tokens")
    cache_read = _int("cached_input_tokens")
    cache_write = _int("cache_write_input_tokens")
    out = _int("output_tokens")
    reasoning = _int("reasoning_output_tokens")
    if all(v is None for v in (total_in, cache_read, cache_write, out, reasoning)):
        return {}, False

    uncached = None if total_in is None else max(total_in - (cache_read or 0) - (cache_write or 0), 0)
    shaped = {
        "input_tokens": uncached or 0,
        "output_tokens": out or 0,
        "cache_read_input_tokens": cache_read or 0,
        "cache_creation_input_tokens": cache_write or 0,
    }
    # Kept alongside, not folded in: reasoning is a subset of output_tokens and
    # adding it again would double-count the most expensive bucket.
    if reasoning is not None:
        shaped["reasoning_output_tokens"] = reasoning
    if total_in is not None:
        shaped["codex_input_tokens_total"] = total_in
    return shaped, True
