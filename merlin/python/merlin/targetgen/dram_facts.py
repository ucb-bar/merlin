"""DERIVE a self-hosted-ISA target's DRAM region (base, and its SIZE) from its declared hardware spec —
target-general.

An ``external_backend`` program oracle preloads/reads operands in the model's 0-based DRAM aperture, but
a correct kernel addresses DRAM at the target's real region base — the start of the DRAM region in the
target's ISA memory map (a card may map cacheable DRAM at a nonzero start). The functional oracle
relocates every DRAM index by this base so the 0-based aperture and the kernel's absolute addresses
agree. The base is DERIVED here from the target's shipped ISA reference (the green-card memory-map
table) — never a hardcoded per-target literal — and is 0 when the target ships no memory map (a 0-based
target, e.g. a 0-based RoCC target, is unaffected). Nothing here holds a target name or an address
literal.

The card row is ``| <label> | <start> ~ <end> |`` and the END ADDRESS IS PART OF THE FACT:
:func:`dram_window_for` returns ``(base, size_bytes | None, provenance_why)``, so a consumer that needs
the region's upper bound (the liveness screen's address-map check) gets it derived rather than assumed.
When the end is not in the card, the size is ``None`` — UNKNOWN, with ``provenance_why`` naming which
gap it is — never a plausible default, because a fabricated window would manufacture "unmapped address"
faults for exactly the programs the real window contains.
"""
from __future__ import annotations

import os
from pathlib import Path

from merlin.common.paths import merlin_dir, repo_root

_CACHE: dict[str, int] = {}
_WINDOW_CACHE: dict[str, tuple[int, int | None, str]] = {}


def _resolve(rel: str) -> Path:
    """Resolve a descriptor-relative path. The ``experiments/…`` bundle-convention paths are
    ``merlin/``-relative; a few refs are repo-root-relative — try ``merlin/`` first, then repo root."""
    for base in (merlin_dir(), repo_root()):
        p = base / rel
        if p.exists():
            return p
    return merlin_dir() / rel


def _descriptor_for(target: str) -> Path | None:
    """The target's ``target_experiment.yaml`` — honor ``MERLIN_TARGET_EXPERIMENT`` when it names THIS
    target, else the standard capsule-bench location. None when neither exists."""
    env = os.environ.get("MERLIN_TARGET_EXPERIMENT")
    if env:
        p = Path(env)
        try:
            if p.is_file():
                from .target_experiment import load_target_experiment
                if load_target_experiment(p).target == target:
                    return p
        except Exception:  # noqa: BLE001 — a malformed env pointer must not mask the standard location
            pass
    std = merlin_dir() / "experiments" / "capsule_bench" / "targets" / target / "target_experiment.yaml"
    return std if std.is_file() else None


def _hex_tokens(text: str) -> list[int]:
    """EVERY ``0x…`` hex token in ``text``, in order (backticks/underscores/whitespace tolerated).
    Structured tokenization — no regex. Empty list when no hex token is present.

    This is the shape the memory-map row actually has: ``| `DRAM` | `0x8000_0000` ~ `0x8_8000_0000` |``
    carries TWO addresses, and reading only the first one threw the region's upper bound away (which is
    why every DRAM window read as "size unknown" even for a target that ships the end address)."""
    out: list[int] = []
    for tok in text.replace("`", " ").replace("~", " ").split():
        cleaned = tok.strip().strip("`").replace("_", "")
        low = cleaned.lower()
        if low.startswith("0x") and len(low) > 2:
            try:
                out.append(int(cleaned, 16))
            except ValueError:
                continue
    return out


def _first_hex(text: str) -> int | None:
    """The first ``0x…`` hex token in ``text``, or None. Thin view over :func:`_hex_tokens`."""
    toks = _hex_tokens(text)
    return toks[0] if toks else None


def _dram_addrs_from_memory_map(md_text: str) -> list[int] | None:
    """Parse a markdown memory-map table for the DRAM region's ADDRESSES. A row is
    ``| <label> | <start> ~ <end> |``; the DRAM region is the row whose label carries the word ``DRAM``
    (so ``IMEM``/``VMEM``/``PERIPH`` and a ``DONT TELL PROF`` region do not match). Returns the row's
    hex tokens in order (``[start]`` or ``[start, end]``), or None if no DRAM row is found. No regex —
    split on table cells + whitespace tokens."""
    for raw in md_text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        label_words = cells[0].replace("`", " ").upper().split()
        if "DRAM" not in label_words:
            continue
        addrs = _hex_tokens(cells[1])
        if addrs:
            return addrs
    return None


def _dram_base_from_memory_map(md_text: str) -> int | None:
    """The DRAM region start from a markdown memory map, or None. Thin view over
    :func:`_dram_addrs_from_memory_map` (kept so the base-only reading stays byte-identical)."""
    addrs = _dram_addrs_from_memory_map(md_text)
    return addrs[0] if addrs else None


#: The three real shapes a target's DRAM window derivation can take. A *target-data* gap (the card
#: ships no upper bound, or the target ships no memory map at all) must read differently from a
#: *tooling* gap (the caller never asked) — otherwise "unknown" says nothing about who can fix it.
_WHY_NO_CARD = ("target ships no memory-map card (no .md ISA reference among its declared "
                "isa_headers) — DRAM window not derivable")
_WHY_NO_DRAM_ROW = ("the target's memory-map card declares no DRAM region row — DRAM window not "
                    "derivable")


def _why_start_only(start: int) -> str:
    return (f"memory-map green card DRAM row {start:#x} declares no upper bound (a single address "
            f"token) — DRAM window size not derivable")


def _why_window(start: int, end: int) -> str:
    return f"memory-map green card DRAM row {start:#x} ~ {end:#x}"


def _why_bad_span(start: int, end: int) -> str:
    return (f"memory-map green card DRAM row {start:#x} ~ {end:#x} spans no bytes (end <= start) — "
            f"unusable, DRAM window size not derivable")


def dram_window_for(target: str) -> tuple[int, int | None, str]:
    """The target's DRAM region as ``(base, size_bytes | None, provenance_why)``, DERIVED from the
    target's shipped memory map (the ``.md`` green card among its descriptor's ISA headers).

    The card's row is ``| <label> | <start> ~ <end> |`` and the END ADDRESS IS PART OF THE FACT: with
    both tokens present the size is ``end - start`` (the region's ``hi`` is an EXCLUSIVE bound
    everywhere it is consumed, so that subtraction is the consistent reading). With only a start token,
    or with no card at all, the size is **UNKNOWN** — ``None``, never a plausible default: a fabricated
    window would manufacture false "unmapped address" faults for every program the real window contains.

    ``base`` is 0 when nothing is derivable (a 0-based aperture — the model default), so a
    non-external-backend or 0-based target is byte-identically unaffected. Memoized; never raises."""
    hit = _WINDOW_CACHE.get(target)
    if hit is not None:
        return hit
    base, size, why = 0, None, _WHY_NO_CARD
    try:
        desc = _descriptor_for(target)
        if desc is not None:
            from .target_experiment import load_target_experiment
            te = load_target_experiment(desc)
            saw_card = False
            for h in te.isa_headers:
                if not str(h).endswith(".md"):
                    continue
                p = _resolve(str(h))
                if not p.is_file():
                    continue
                saw_card = True
                addrs = _dram_addrs_from_memory_map(p.read_text(encoding="utf-8", errors="replace"))
                if not addrs:
                    continue
                base = int(addrs[0])
                if len(addrs) >= 2:
                    end = int(addrs[1])
                    span = end - base
                    if span > 0:
                        size, why = span, _why_window(base, end)
                    else:
                        why = _why_bad_span(base, end)
                else:
                    why = _why_start_only(base)
                break
            else:
                if saw_card:
                    why = _WHY_NO_DRAM_ROW
    except Exception:  # noqa: BLE001 — an unresolvable/absent spec means 0-based (fail to the default)
        base, size, why = 0, None, _WHY_NO_CARD
    out = (base, size, why)
    _WINDOW_CACHE[target] = out
    return out


def dram_base_for(target: str) -> int:
    """The DRAM region base (byte address) a target's kernels address, DERIVED from the target's shipped
    memory map (the ``.md`` green card among its descriptor's ISA headers). 0 when the target ships no
    memory map / no DRAM row (a 0-based aperture — the model default), so a non-external-backend or
    0-based target is byte-identically unaffected. Memoized per target; never raises."""
    if target in _CACHE:
        return _CACHE[target]
    base = dram_window_for(target)[0]
    _CACHE[target] = base
    return base
