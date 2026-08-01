"""DERIVE a self-hosted-ISA target's DRAM region base from its declared hardware spec — target-general.

An ``external_backend`` program oracle preloads/reads operands in the model's 0-based DRAM aperture, but
a correct kernel addresses DRAM at the target's real region base — the start of the DRAM region in the
target's ISA memory map (a card may map cacheable DRAM at a nonzero start). The functional oracle
relocates every DRAM index by this base so the 0-based aperture and the kernel's absolute addresses
agree. The base is DERIVED here from the target's shipped ISA reference (the green-card memory-map
table) — never a hardcoded per-target literal — and is 0 when the target ships no memory map (a 0-based
target, e.g. gemmini, is unaffected). Nothing here holds a target name or an address literal.
"""
from __future__ import annotations

import os
from pathlib import Path

from merlin.common.paths import merlin_dir, repo_root

_CACHE: dict[str, int] = {}


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


def _first_hex(text: str) -> int | None:
    """The first ``0x…`` hex token in ``text`` (backticks/underscores/whitespace tolerated). Structured
    tokenization — no regex. Returns None when no hex token is present."""
    for tok in text.replace("`", " ").replace("~", " ").split():
        cleaned = tok.strip().strip("`").replace("_", "")
        low = cleaned.lower()
        if low.startswith("0x") and len(low) > 2:
            try:
                return int(cleaned, 16)
            except ValueError:
                continue
    return None


def _dram_base_from_memory_map(md_text: str) -> int | None:
    """Parse a markdown memory-map table for the DRAM region start. A row is
    ``| <label> | <start> ~ <end> |``; the DRAM region is the row whose label carries the word ``DRAM``
    (so ``IMEM``/``VMEM``/``PERIPH`` and a ``DONT TELL PROF`` region do not match). Returns the start
    address, or None if no such row is found. No regex — split on table cells + whitespace tokens."""
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
        start = _first_hex(cells[1])
        if start is not None:
            return start
    return None


def dram_base_for(target: str) -> int:
    """The DRAM region base (byte address) a target's kernels address, DERIVED from the target's shipped
    memory map (the ``.md`` green card among its descriptor's ISA headers). 0 when the target ships no
    memory map / no DRAM row (a 0-based aperture — the model default), so a non-external-backend or
    0-based target is byte-identically unaffected. Memoized per target; never raises."""
    if target in _CACHE:
        return _CACHE[target]
    base = 0
    try:
        desc = _descriptor_for(target)
        if desc is not None:
            from .target_experiment import load_target_experiment
            te = load_target_experiment(desc)
            for h in te.isa_headers:
                if str(h).endswith(".md"):
                    p = _resolve(str(h))
                    if p.is_file():
                        found = _dram_base_from_memory_map(p.read_text(encoding="utf-8", errors="replace"))
                        if found is not None:
                            base = int(found)
                            break
    except Exception:  # noqa: BLE001 — an unresolvable/absent spec means 0-based (fail to the default)
        base = 0
    _CACHE[target] = base
    return base
