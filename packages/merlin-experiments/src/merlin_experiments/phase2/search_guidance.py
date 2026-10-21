"""Author guidance for scaling a search to its measured performance deficit.

This changes exploration advice, never the evidence used for ranking or stopping.
"""

ADAPTIVE_SCOPE = (
    "Scale the scope of changes to the gap from the configured target. For each valid measured "
    "attainment, compute target attainment / current attainment. Above 4x, prioritize replacing "
    "costly lowering strategies, restructuring whole loop nests, changing tiling, loop order, "
    "data reuse, operation grouping and coherent combinations of "
    "admitted passes; estimate the recoverable dominant cost before spending a measurement. "
    "Above 1.25x through 4x, explore pass combinations and broader parameter sweeps. At 1.25x "
    "or less, prioritize granular tuning and regression control. These are exploration heuristics, "
    "not predicted speedups or evidence of convergence. Unknown or partial bounds stay UNKNOWN; "
    "do not assume those members are close to the target. Continue high-impact improvements "
    "using admitted semantics while resolving their missing measurements. Preserve all numerical "
    "and source gates at every scale. "
    "Use available queue capacity for distinct frozen contenders instead of waiting after each "
    "submission. Give each contender a recorded parent and hypothesis, serialize shared source "
    "edits, and keep evidence attached to its immutable revision. Reconsider hypotheses as "
    "results arrive. Combine retained improvements and measure the combination; individual "
    "gains do not establish the combined gain. "
)
