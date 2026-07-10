"""Parse our own C-driver / simulator stdout structurally (token scans) — never regex.

Our bare-metal / spike / K1 / cost-calibration drivers print fixed, whitespace-delimited line
markers: ``CYCLES <n>``, ``KIND <k> COUNT <c> CYCLES <n>``, ``REGION_CYCLES <n>``,
``MERLIN_E2E key=val ...``. These helpers read them by splitting on whitespace and looking up
labelled fields, replacing the ad-hoc ``re.search(r"CYCLES\\s+(\\d+)")`` scrapes that were spread
across the runtime / cost-model / benchmark drivers. No ``re`` here by design.
"""
from __future__ import annotations


def int_after(text: str, label: str) -> int | None:
    """First integer token immediately following a whitespace-delimited ``label`` token anywhere in
    ``text`` (e.g. ``int_after("... CYCLES 421 ...", "CYCLES") == 421``), else None."""
    toks = text.split()
    for i, tok in enumerate(toks):
        if tok == label and i + 1 < len(toks) and toks[i + 1].lstrip("-").isdigit():
            return int(toks[i + 1])
    return None


def _leading_int(s: str) -> int | None:
    i = 1 if s[:1] == "-" else 0
    j = i
    while j < len(s) and s[j].isdigit():
        j += 1
    return int(s[:j]) if j > i else None


def int_field(text: str, key: str) -> int | None:
    """Integer value of a labelled field in ``text``, accepting either ``KEY <int>`` or ``KEY=<int>``
    (leading digits of the value), e.g. ``MR=4`` or ``errors=7``. None if absent."""
    for tok in text.split():
        head, sep, val = tok.partition("=")
        if sep and head == key:
            got = _leading_int(val)
            if got is not None:
                return got
    return int_after(text, key)


def line_after_marker(text: str, marker: str) -> str | None:
    """Remainder of the first line whose first whitespace-token equals ``marker`` (e.g.
    ``MERLIN_E2E``); ``""`` if the marker line has no payload; None if no such line."""
    for line in text.splitlines():
        parts = line.split(None, 1)
        if parts and parts[0] == marker:
            return parts[1] if len(parts) > 1 else ""
    return None


def kv_pairs(text: str) -> dict[str, str]:
    """Whitespace-separated ``key=value`` tokens in ``text`` -> dict (later tokens win)."""
    out: dict[str, str] = {}
    for tok in text.split():
        key, sep, val = tok.partition("=")
        if sep and key:
            out[key] = val
    return out


def is_vector_mnemonic(mnem: str) -> bool:
    """True if ``mnem`` is an RVV vector mnemonic token — the structured equivalent of the pattern
    ``^v[a-z0-9]+(?:\\.[a-z0-9]+)*$`` (a ``v``-led head of >=2 lowercase/digit chars, then optional
    ``.``-separated lowercase/digit groups, e.g. ``vsetvli``, ``vfmacc.vv``, ``vle32.v``)."""
    parts = mnem.split(".")
    head = parts[0]
    if len(head) < 2 or head[0] != "v":
        return False
    return all(p and all(c.islower() or c.isdigit() for c in p) for p in parts)
