"""Transient model-capacity refusals in a continuous codex session.

A refusal may resume the same session after a backoff, never switch its
model or start another grader, and retries stay within its wall budget.
"""

from __future__ import annotations

# ---- codex_capacity ----------------------------------------------------------------------------

MAX_CONSECUTIVE = 160

_CAPACITY_BACKOFF_S = 20

_CAPACITY_BACKOFF_MAX_S = 300

_CAPACITY_MESSAGE = "selected model is at capacity"

RESUME_NOTE = (
    "\n\nStatus: the previous turn was ended by a provider capacity error. Helper threads "
    "started in that turn were stopped with it and will not report back; their file edits "
    "remain on disk. Start again any helper work that is still needed.\n"
)


def refusal(errors) -> bool:
    """Whether every error of a failed turn is a transient model-capacity refusal.

    Capacity refusals also surface wrapped, e.g. from the remote compaction
    task, so the provider message is matched anywhere in the error text."""
    return bool(errors) and all(_CAPACITY_MESSAGE in str(error).lower() for error in errors)


_NOT_PROGRESS = frozenset({"reasoning", "error"})


def progress(item) -> bool:
    """Whether a completed turn item is served work (a command, tool call, file change, message)."""
    return isinstance(item, dict) and isinstance(item.get("type"), str) and item["type"] not in _NOT_PROGRESS


def backoff_s(attempt: int) -> int:
    """Exponential backoff for the ``attempt``-th consecutive refusal (1-based)."""
    return min(_CAPACITY_BACKOFF_S * 2 ** max(attempt - 1, 0), _CAPACITY_BACKOFF_MAX_S)
