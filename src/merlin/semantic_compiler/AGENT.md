# AGENT.md — src/merlin/semantic_compiler

This package owns target-independent typed semantic graphs, generated instruction
rules, native search, extraction, allocation, and independent checking. Keep
target-specific facts in explicitly selected out-of-tree support packages.
The production path must not import or invoke ACT. Reference adapters belong in
a separate module and may only enter through an explicit engine selection.

The Rust `egg_bridge` is a general-purpose e-graph dependency boundary. Merlin
owns all rule eligibility, buffer-aware extraction, constraint construction,
fallback and result checking. Unknown semantics or solver results are failures,
not evidence that a target operation is unsupported.
