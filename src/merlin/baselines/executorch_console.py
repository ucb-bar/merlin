"""Structural, tri-state parsing of ExecuTorch timing console output."""

from __future__ import annotations

from dataclasses import dataclass

_PARSED = "parsed"
_ABSENT = "absent"
_UNPARSEABLE = "unparseable"

_EXEC_MARKER = "Model executed successfully "  # then `.* in <ms> ms`
_EXEC_JOINER = " in "
_LOAD_MARKER = "Model loaded in "  # the number follows the marker directly


@dataclass
class _ConsoleReading:
    """One millisecond field read out of the executor_runner console."""

    state: str  # _PARSED / _ABSENT / _UNPARSEABLE
    ms: float | None = None
    detail: str = ""  # the offending line, when state is _UNPARSEABLE

    @property
    def ns(self) -> int | None:
        return None if self.ms is None else int(round(self.ms * 1e6))


def _leading_ms(text: str) -> float | None:
    """Read ``<number> ms`` off the FRONT of ``text``; ``None`` if it is not shaped like that.

    This is the old ``([\\d.]+) ms`` capture group, which sat immediately after a literal, so the
    number starts at character 0 here. The old character class was digits-and-dots only, so we
    refuse anything else rather than coerce it — a token that is not a plain decimal (``inf``,
    ``nan``, ``1.2.3``, a negative) yields UNKNOWN. (The old code would have crashed on ``1.2.3``:
    the pattern matched it and ``float()`` then raised.)
    """
    token, sep, _rest = text.partition(" ms")
    if not sep or not token:
        return None
    if any(c not in "0123456789." for c in token):
        return None
    try:
        return float(token)
    except ValueError:  # e.g. "." or "1.2.3" — matched the old class, is not a number
        return None


def _read_console_ms(console: str, marker: str, *, joiner: str = "") -> _ConsoleReading:
    """First line containing ``marker`` whose millisecond field parses, as a tri-state reading.

    ``joiner`` is the literal the old pattern required between the marker and the number (the
    ``.* in `` of the execute line); empty when the number follows the marker directly. The old
    ``.*`` was greedy, so on a line with several ``... in <n> ms`` the LAST one won — hence the
    right-to-left search for the joiner.
    """
    saw_marker = ""
    for line in console.splitlines():
        idx = line.find(marker)
        while idx >= 0:
            # `re.search` restarts at the next position when a match attempt fails, so a marker
            # that occurs several times on one line gets tried at EVERY occurrence, left to right.
            saw_marker = saw_marker or line
            tail = line[idx + len(marker) :]
            if not joiner:
                ms = _leading_ms(tail)
                if ms is not None:
                    return _ConsoleReading(_PARSED, ms)
            else:
                pos = tail.rfind(joiner)
                while pos >= 0:
                    ms = _leading_ms(tail[pos + len(joiner) :])
                    if ms is not None:
                        return _ConsoleReading(_PARSED, ms)
                    pos = tail.rfind(joiner, 0, pos)
            idx = line.find(marker, idx + 1)
    if saw_marker:
        return _ConsoleReading(_UNPARSEABLE, None, saw_marker.strip()[:200])
    return _ConsoleReading(_ABSENT)
