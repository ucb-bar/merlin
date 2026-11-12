"""Explicit upstream buffer identity exposure before public result conversion.

Bufferization can produce loops carrying an unchanged memref identity. Exposing
that identity before result conversion lets upstream forward a complete static
allocation to its caller-owned output. Upstream tensor alias and ownership
analysis remains authoritative; this feature adds no pointer or numeric facts.
"""

from __future__ import annotations

FEATURE = "canonicalize_bufferized_result_identity"


def edit_pipeline(passes: list[str]) -> list[str]:
    """Insert upstream canonicalization immediately before result conversion."""
    bufferize = [i for i, item in enumerate(passes) if item.partition("{")[0] == "one-shot-bufferize"]
    result_conversion = [i for i, item in enumerate(passes) if item.partition("{")[0] == "buffer-results-to-out-params"]
    if len(bufferize) != 1 or len(result_conversion) != 1 or bufferize[0] >= result_conversion[0]:
        raise ValueError("buffer identity requires one ordered bufferization/result conversion pair")
    i = result_conversion[0]
    out = list(passes)
    if out[i - 1] != "canonicalize":
        out.insert(i, "canonicalize")
    return out


def ensure_registered() -> str:
    from merlin.compiler.selection.shared import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description="Expose upstream bufferized loop/result identities before output conversion; retain source tensor alias and ownership semantics. Explicit and default off.",
                edit_pipeline=edit_pipeline,
            )
        )
    return FEATURE
