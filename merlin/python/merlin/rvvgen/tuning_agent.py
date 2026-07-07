"""LLM-based ForkProposal proposer — the judgment alternative to the deterministic gap-router.

`propose_forks_llm` is signature-compatible with `merlin.kernels.rvv_knobs.propose_forks`, so it
drops straight into `run_beam(proposer=propose_forks_llm)`. Where the gap-router enumerates a fixed
motif->knob table, the tuning agent asks an LLM to read the S4 divergences (+ optional mined-policy
/ curated-fingerprint context) and PROPOSE schedule-knob changes to close the structural gap toward
the expert kernel.

Honest contract:
  * The LLM is the injectable ``llm_fn`` (default merlin.common.llm.complete — Anthropic when
    ANTHROPIC_API_KEY is set, else None so we degrade to []). The prompt is a versioned artifact
    (merlin/prompts/rvv_tuning_v{V}.md).
  * Every proposed override is VALIDATED/CLAMPED against the known knob vocabulary before it can be
    minted; an override key the generator (rvvgen.from_strategy.render_schedule) cannot render is
    DROPPED with a note rather than emitted (the beam never tries to render an unknown knob).
  * Graceful: llm_fn None / parse failure / empty list -> [] (the beam simply finds no forks and
    falls back on whatever else it has). Non-actionable suggestions are returned as
    forkable=False ForkProposals so they are recorded as work-items, never silently lost.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from ..kernels.rvv_knobs import ForkProposal

from merlin.common.paths import merlin_dir

_PROMPT_DIR = merlin_dir() / "prompts"

# The knob vocabulary the RVV generator (from_strategy.render_schedule) can actually render. Any
# override key outside this set is dropped with a note (we never emit a knob the generator can't
# turn into schedule.mlir). Keep in lock-step with from_strategy.render_schedule's consumed knobs.
_KNOWN_OVERRIDE_KEYS = {"op_match", "contraction_strategy", "lowering_patterns", "dtype_strategy"}

# Enumerated value vocabularies (used to clamp / reject scalar knobs).
_CONTRACTION_STRATEGIES = {"outerproduct", "dot", "matmulintrinsics", "parallelarith"}
_DTYPE_STRATEGIES = {"fp32", "int8_w8a8", "bf16_f32acc"}
# The vector-lowering patterns the schedule knows how to apply (transform.apply_patterns.vector.*).
_LOWERING_PATTERNS = {
    "lower_contraction", "lower_masked_transfers", "lower_transpose", "lower_shape_cast",
    "lower_outerproduct", "lower_broadcast", "lower_transfer", "lower_multi_reduction",
}


def prompt_path(version: int = 1) -> Path:
    return _PROMPT_DIR / f"rvv_tuning_v{version}.md"


def _default_llm(prompt: str) -> str | None:
    from ..common.llm import complete
    return complete(prompt, max_tokens=800)


def build_prompt(divergences: list[str], knobs: dict[str, Any], *, context: Any = None,
                 version: int = 1) -> str:
    """Render the versioned tuning prompt from the parent knobs + S4 divergences + optional
    mined-policy / curated-fingerprint context."""
    tmpl = prompt_path(version).read_text(encoding="utf-8")
    return tmpl.format(
        divergences=json.dumps(divergences, indent=2),
        knobs=json.dumps(knobs, indent=2),
        context=json.dumps(context, indent=2, default=str) if context else "(none provided)",
        contraction_strategies=", ".join(sorted(_CONTRACTION_STRATEGIES)),
        dtype_strategies=", ".join(sorted(_DTYPE_STRATEGIES)),
        lowering_patterns=", ".join(sorted(_LOWERING_PATTERNS)),
    )


def _parse_proposals(text: str | None) -> list[dict]:
    """Tolerant parse of an agent reply into a list of {overrides, rationale, targets} dicts.

    Accepts a bare JSON array, or an object with a "proposals"/"forks" key, possibly wrapped in
    code fences / prose. Returns [] on any failure (never raises)."""
    if not text:
        return []
    # Try a JSON array first, then an enclosing object.
    for pat in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pat, text, re.S)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(obj, list):
            return [p for p in obj if isinstance(p, dict)]
        if isinstance(obj, dict):
            for key in ("proposals", "forks", "items"):
                v = obj.get(key)
                if isinstance(v, list):
                    return [p for p in v if isinstance(p, dict)]
            # a single proposal object
            if "overrides" in obj:
                return [obj]
    return []


def _clamp_op_match(value: Any) -> tuple[list[dict] | None, list[str]]:
    """Validate an op_match override: a list of {op, tile, vector} with int tile/vector lists of
    equal length. Returns (clamped|None, notes)."""
    notes: list[str] = []
    if not isinstance(value, list) or not value:
        return None, ["op_match must be a non-empty list of {op, tile, vector}"]
    out = []
    for entry in value:
        if not isinstance(entry, dict) or "op" not in entry:
            notes.append("op_match entry missing 'op'")
            return None, notes
        tile, vec = entry.get("tile"), entry.get("vector")
        if not (isinstance(tile, list) and isinstance(vec, list) and len(tile) == len(vec)):
            notes.append(f"op_match[{entry.get('op')}] needs equal-length int tile/vector lists")
            return None, notes
        try:
            tile = [int(x) for x in tile]
            vec = [int(x) for x in vec]
        except (TypeError, ValueError):
            notes.append(f"op_match[{entry.get('op')}] tile/vector must be integers")
            return None, notes
        out.append({"op": str(entry["op"]), "tile": tile, "vector": vec})
    return out, notes


def _clamp_overrides(overrides: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate/clamp a proposed override dict against the known knob vocabulary. Returns
    (clean_overrides, drop_notes). Unknown keys and invalid values are dropped (with a note);
    never raises."""
    clean: dict[str, Any] = {}
    notes: list[str] = []
    if not isinstance(overrides, dict):
        return {}, ["overrides was not an object"]
    for key, value in overrides.items():
        if key not in _KNOWN_OVERRIDE_KEYS:
            notes.append(f"dropped unknown knob {key!r} (not in {sorted(_KNOWN_OVERRIDE_KEYS)})")
            continue
        if key == "op_match":
            clamped, om_notes = _clamp_op_match(value)
            notes.extend(om_notes)
            if clamped is not None:
                clean[key] = clamped
        elif key == "contraction_strategy":
            if value is None or value in _CONTRACTION_STRATEGIES:
                clean[key] = value
            else:
                notes.append(f"dropped contraction_strategy={value!r} (not in "
                             f"{sorted(_CONTRACTION_STRATEGIES)})")
        elif key == "dtype_strategy":
            if value in _DTYPE_STRATEGIES:
                clean[key] = value
            else:
                notes.append(f"dropped dtype_strategy={value!r} (not in {sorted(_DTYPE_STRATEGIES)})")
        elif key == "lowering_patterns":
            if isinstance(value, list) and all(isinstance(p, str) for p in value):
                kept = [p for p in value if p in _LOWERING_PATTERNS]
                dropped = [p for p in value if p not in _LOWERING_PATTERNS]
                if dropped:
                    notes.append(f"dropped unknown lowering_patterns {dropped}")
                if kept:
                    clean[key] = kept
            else:
                notes.append("lowering_patterns must be a list of pattern-name strings")
    return clean, notes


def propose_forks_llm(divergences: list[str], knobs: dict[str, Any], *, context: Any = None,
                      llm_fn: Callable[[str], "str | None"] | None = None,
                      version: int = 1) -> list[ForkProposal]:
    """LLM proposer, drop-in for `propose_forks(divergences, knobs)`.

    Builds the versioned tuning prompt from the parent ``knobs`` + S4 ``divergences`` (+ optional
    mined-policy/curated-fingerprint ``context``), calls ``llm_fn`` (default common.llm.complete),
    parses a JSON list of {overrides, rationale, targets}, validates/clamps each override to the
    known knob vocabulary, and returns ForkProposal objects:

      * a proposal with at least one renderable override -> forkable=True, lever='knob'
      * a proposal whose overrides are all dropped/empty -> forkable=False (recorded work-item),
        note carries the drop reasons + rationale

    Graceful: llm_fn None / parse failure -> [] (beam finds no forks that gen)."""
    llm_fn = llm_fn or _default_llm
    raw = llm_fn(build_prompt(divergences, knobs, context=context, version=version))
    proposals = _parse_proposals(raw)
    out: list[ForkProposal] = []
    for p in proposals:
        targets = str(p.get("targets") or p.get("target") or "llm_proposal")
        rationale = str(p.get("rationale") or p.get("why") or "")
        clean, notes = _clamp_overrides(p.get("overrides") or {})
        note = rationale
        if notes:
            note = (note + " | " if note else "") + "; ".join(notes)
        if clean:
            out.append(ForkProposal(overrides=clean, lever="knob", targets=targets,
                                    evidence=["llm_tuning_agent"], forkable=True, note=note))
        else:
            # nothing renderable survived clamping -> record as a non-actionable work-item.
            out.append(ForkProposal(overrides={}, lever="llm_suggestion", targets=targets,
                                    evidence=["llm_tuning_agent"], forkable=False,
                                    note=note or "no renderable knob override in proposal"))
    return out
