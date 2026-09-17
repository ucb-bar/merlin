"""Deterministic marker table: the heart of cheap, reproducible motif extraction.

A *marker* is a regex whose presence in a kernel's text is evidence that an optimization
*decision* was made. Markers are keyed by ``(isa_family, motif)`` rather than by producer:
because every source ultimately becomes C (XNNPACK ships C; Autocomp emits C; Exo is
compiled to C), the intrinsics that appear depend on the *target ISA*, not on who wrote it.
This means Autocomp-Gemmini C and Exo-Gemmini C share one marker set, and XNNPACK-RVV and a
hypothetical Exo-RVV share another.

We deliberately match *decisions* (a packed-weight pointer advance, an accumulator address
flag, a vector-length-agnostic loop) and never *constants* (tile sizes, LMUL values).

The table is DATA. Each ISA family's patterns (``markers``) and the ``kernel.target`` spellings that
select it (``targets``) live in ``framework_contracts/feature_extraction/<family>.yaml``, beside the
family's other feature-extraction facts; this module only loads, merges and compiles them. Adding an ISA
family is a new data file, not an edit here.

``markers_for(target)`` resolves a kernel's ``target`` to its family and returns the compiled
regex table for that family, merged over the ``generic`` baseline.
"""

from __future__ import annotations

import re  # regex-ok: compiles the motif patterns each family declares as data (feature_extraction/)
from functools import lru_cache

from merlin.kernels.framework_contracts import feature_families, load_feature_contract

# Canonical motif vocabulary (also the keys produced by classify.py). A family file may declare markers
# for these motifs only.
MOTIFS = (
    "packed_rhs",
    "accumulator_lifetime",
    "epilogue_before_commit",
    "vector_length_polymorphic",
    "tiling_blocking",
    "double_buffering",
    "weight_stationary_dataflow",
    "intrinsic_lowering",
    # RVV intrinsic decisions (evidence snippets; the classified motifs of the same name in
    # classify.py drive promotion — these markers give audit the matched substrings).
    "lmul_grouping",
    "scalar_broadcast_fma",
    "int8_widening_mac",
    "vl_polymorphic_tail",
    "vector_reduction",
    "requant_narrowing",
)

#: The family an unmapped ``kernel.target`` resolves to. Its markers (``feature_extraction/generic.yaml``)
#: are the baseline merged under every family's own.
GENERIC_FAMILY = "generic"


@lru_cache(maxsize=None)
def _target_families() -> dict[str, str]:
    """``{kernel.target spelling: ISA family}``, built from every family file's ``targets`` list.

    A spelling claimed by two families raises: resolving it by file order would make a kernel's evidence
    depend on which file happened to sort first."""
    out: dict[str, str] = {}
    for family in feature_families():
        for spelling in load_feature_contract(family).get("targets") or []:
            key = str(spelling).lower()
            owner = out.setdefault(key, family)
            if owner != family:
                raise ValueError(f"kernel.target {key!r} is claimed by two ISA families: {owner!r} and {family!r}")
    return out


def target_family(target: str) -> str:
    return _target_families().get((target or "").lower(), GENERIC_FAMILY)


def _raw(family: str) -> dict[str, list[str]]:
    """The family's uncompiled ``{motif: [pattern, ...]}`` table; a motif fires if ANY pattern matches.

    A motif outside :data:`MOTIFS` raises rather than being skipped: the merge walks MOTIFS, so an unknown
    (e.g. misspelled) key would otherwise never fire and nothing would say so."""
    table = load_feature_contract(family).get("markers") or {}
    unknown = sorted(set(table) - set(MOTIFS))
    if unknown:
        raise ValueError(
            f"feature_extraction/{family}.yaml declares markers for unknown motif(s) "
            f"{unknown}; the vocabulary is markers.MOTIFS"
        )
    return {motif: [str(p) for p in (pats or [])] for motif, pats in table.items()}


@lru_cache(maxsize=None)
def _compiled_for_family(family: str) -> tuple[tuple[str, tuple[re.Pattern, ...]], ...]:
    base = _raw(GENERIC_FAMILY)
    fam = _raw(family)
    merged: dict[str, list[str]] = {}
    for motif in MOTIFS:
        pats = list(base.get(motif, [])) + list(fam.get(motif, []))
        if pats:
            merged[motif] = pats
    compiled = []
    for motif, pats in merged.items():
        compiled.append((motif, tuple(re.compile(p) for p in pats)))  # regex-ok: data-declared markers
    return tuple(compiled)


def markers_for(target: str) -> dict[str, tuple[re.Pattern, ...]]:
    """Return ``{motif: (compiled_regex, ...)}`` for the ISA family of ``target``."""
    return dict(_compiled_for_family(target_family(target)))


def fired_markers(text: str, target: str) -> dict[str, list[str]]:
    """Return, per motif, the list of matched marker substrings found in ``text``.

    The returned snippets become the ``evidence.code_markers`` of a kernel record. Only
    motifs that fired appear in the result.
    """
    out: dict[str, list[str]] = {}
    for motif, patterns in markers_for(target).items():
        hits: list[str] = []
        for pat in patterns:
            m = pat.search(text)
            if m:
                hits.append(m.group(0).strip())
        if hits:
            # de-dup while preserving order
            out[motif] = list(dict.fromkeys(hits))
    return out
