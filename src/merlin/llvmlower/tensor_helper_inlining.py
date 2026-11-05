"""Expose private tensor helper allocations to ordinary result bufferization.

The stock inliner preserves tensor semantics. Upstream alias analysis, ownership
and result conversion still own destination legality; this adds no pointer facts.
"""

from __future__ import annotations

FEATURE = "inline_tensor_helpers_before_bufferization"


def composition_features(features, *, has_writers):
    """Expose closed writer helpers by default; an explicit feature set wins."""
    return frozenset({FEATURE}) if features is None and has_writers else features


def edit_pipeline(passes):
    """Inline before bufferization so result forwarding can see helper storage."""
    result = list(passes)
    bufferize = [i for i, p in enumerate(result) if p.partition("{")[0] == "one-shot-bufferize"]
    outputs = [i for i, p in enumerate(result) if p.partition("{")[0] == "buffer-results-to-out-params"]
    if len(bufferize) != 1 or len(outputs) != 1 or bufferize[0] >= outputs[0]:
        raise ValueError("tensor helper inlining requires one ordered bufferization/result conversion pair")
    at = bufferize[0]
    if result[max(0, at - 3):at] != ["inline", "canonicalize", "cse"]:
        result[at:at] = ["inline", "canonicalize", "cse"]
    return result


def ensure_registered():
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(ImprFeature(
            name=FEATURE,
            action_class="PASS",
            description=(
                "Inline tensor helpers before upstream bufferization and output forwarding; explicit and default off."
            ),
            edit_pipeline=edit_pipeline,
        ))
    return FEATURE
