# src/merlin/development/targetgen/semantic_search

This module owns bounded, deterministic selection of static tensor operations against
selected instruction semantics. It must compare exact typed semantics, never op family
names or target names. Unknown semantics, effects, memory facts, and solver timeouts
remain visible in the receipt. A selection is a compile plan, not execution evidence.
