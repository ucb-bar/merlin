"""Section selection: turn a human "profile just this part" spec into the set of ``prov.region_id``s
the dispatch-DAG slicer (``xdsl_dialects.lowering.dispatch_program.slice_program``) keeps.

Compile the whole model, then profile a SECTION — an inner region, a layer, or several layers —
referenced the way a user thinks about the ORIGINAL model (its nn.Module FQNs, or GGUF layers whose
reconstructed HF FQNs share this key space). The selectable menu is exactly what the structural
recognizer already found (:func:`attribution.recognize_regions`); this module resolves a spec against
it. Deterministic, no model run, no regex (glob via ``fnmatch``; layer indices via structured parsing).

Spec forms (a single string or a list, mixed freely):
  * ``"whole"`` / ``"all"`` / ``"*"``        — every region.
  * an exact ``prov.region_id`` (e.g. ``"matmul_3"``).
  * ``"layers:3"`` / ``"layers:3-5"``        — the region(s) in layer 3 (or the inclusive range 3..5).
  * ``"fqn:<glob>"`` or a bare token         — fqn glob (``*self_attn``) or, without wildcards,
                                               an fqn substring (``self_attn``).
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from .attribution import recognize_regions

# nn.Module container tokens whose FOLLOWING integer is the layer index (structured, no regex).
_LAYER_TOKENS = ("layers", "blocks", "transformer_blocks", "block", "layer", "h")


@dataclass(frozen=True)
class Section:
    """One selectable section: what it is + the join/slice keys it owns."""
    label: str                     # region label (attention / linear / mlp / conv / norm / softmax)
    fqn: str | None                # the nn.Module path the section occupies
    role: str | None               # backbone_once / repeated_head / ... (role_from_fqn)
    region_ids: tuple[str, ...]    # prov.region_ids the slicer keeps for this section


def list_sections(capture_dir: str) -> tuple[Section, ...]:
    """The menu of selectable sections for a capture (compute regions only — those with region_ids)."""
    return tuple(
        Section(label=r.region_label, fqn=r.fqn_group, role=r.role, region_ids=r.region_ids)
        for r in recognize_regions(capture_dir) if r.region_ids)


def _layer_index(fqn: str | None) -> int | None:
    """The layer index a section lives at (the int following a container token like ``layers``/
    ``blocks``), or None. Handles the varying schemas (``blocks.N`` / ``layers.N`` / ...) structurally."""
    if not fqn:
        return None
    parts = fqn.split(".")
    for i, tok in enumerate(parts[:-1]):
        if tok.lower() in _LAYER_TOKENS and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _parse_layer_range(spec: str) -> tuple[int, int]:
    spec = spec.strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return int(lo), int(hi)
    v = int(spec)
    return v, v


def _resolve_token(token: str, sections: tuple[Section, ...], all_ids: set[str]) -> set[str]:
    t = token.strip()
    if t in all_ids:                                          # exact region_id
        return {t}
    if t.startswith("layers:"):
        lo, hi = _parse_layer_range(t[len("layers:"):])
        return {rid for s in sections if (li := _layer_index(s.fqn)) is not None and lo <= li <= hi
                for rid in s.region_ids}
    pattern = t[len("fqn:"):] if t.startswith("fqn:") else t
    wild = any(c in pattern for c in "*?[")
    out: set[str] = set()
    for s in sections:
        if s.fqn and (fnmatch.fnmatch(s.fqn, pattern) if wild else pattern in s.fqn):
            out.update(s.region_ids)
    return out


def resolve(capture_dir: str, spec) -> set[str]:
    """Resolve a selection spec (string or list of tokens) to the set of ``prov.region_id``s to keep.
    ``None`` / whole / all / ``*`` selects every region. Raises if the spec matches nothing (fail-closed
    — a typo'd section name must not silently profile the whole model)."""
    sections = list_sections(capture_dir)
    all_ids = {rid for s in sections for rid in s.region_ids}
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("whole", "all", "*")):
        return set(all_ids)
    tokens = [spec] if isinstance(spec, str) else list(spec)
    out: set[str] = set()
    for tok in tokens:
        out |= _resolve_token(str(tok), sections, all_ids)
    if not out:
        raise ValueError(f"section selection {spec!r} matched no region in {capture_dir}")
    return out
