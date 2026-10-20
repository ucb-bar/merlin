"""Rejoin disassembler alias spelling to exact selected linked functions.

The old text-only counter review stays unchanged. This separately versioned
extension admits only aliases with identical positive executable extents in the
actual ELF. It changes no code, selected symbol, counter boundary or simulator.
"""

from merlin.perf import compiled_static_features as SYMBOLS
from merlin.perf import debug_companion as ELF
from merlin_experiments.phase1 import annotated_spike_evidence as ORIGINAL

REVIEW_SCHEMA = "merlin.counter_window_review.v2"


def source_modules():
    """Original parser/definition owners for live queue and process pinning."""
    return (SYMBOLS, ELF, ORIGINAL)


def window(rows, *, elf_bytes, counter_asm, entry_symbol, completion_symbol):
    """Return original review unchanged, or an explicit source-bound alias review."""
    selection = dict(counter_asm=counter_asm, entry_symbol=entry_symbol, completion_symbol=completion_symbol)
    try:
        return ORIGINAL.counter_window(rows, **selection), ORIGINAL.REVIEW_SCHEMA
    except ORIGINAL.EvidenceError as original_error:
        error = original_error
    if type(elf_bytes) is not bytes or not 0 < len(elf_bytes) <= 64 << 20:
        raise ORIGINAL.EvidenceError("counter alias review requires exact bounded ELF bytes")
    names = {name for row in rows for name in row["symbols"]}
    if len(names) > 128:
        raise ORIGINAL.EvidenceError("counter alias review exceeds its bounded function reference roster")
    extents = {}
    for name in names | {entry_symbol, completion_symbol}:
        try:
            extents[name] = SYMBOLS.linked_entry_extent(elf_bytes, name)
        except ValueError:
            continue
    aliases = {}
    proof = []
    for selected in (entry_symbol, completion_symbol):
        extent = extents.get(selected)
        if not extent or extent["size_bytes"] <= 0:
            continue  # A zero-sized label cannot supply a complete function-alias proof.
        identity = tuple(extent[key] for key in ("address", "size_bytes", "section"))
        for name in sorted(names - {selected}):
            candidate = extents.get(name)
            if candidate and tuple(candidate[key] for key in ("address", "size_bytes", "section")) == identity:
                aliases.setdefault(name, set()).add(selected)
                proof.append(dict(alias=name, selected_symbol=selected,
                                  **{key: extent[key] for key in ("address", "size_bytes", "section")}))
    if not proof:
        raise error
    augmented = [{**row, "symbols": sorted(set(row["symbols"]) | set().union(
        *(aliases.get(name, set()) for name in row["symbols"]))) } for row in rows]
    result = ORIGINAL.counter_window(augmented, **selection)
    result["symbol_aliases"] = proof
    result["alias_authority"] = "exact linked executable extent only; no counter, completion or timing qualification"
    return result, REVIEW_SCHEMA
