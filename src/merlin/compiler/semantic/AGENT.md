# src/merlin/compiler/semantic

This package owns target-independent typed semantic graphs, generated instruction
rules, native search, extraction, allocation, and independent checking. Keep
target-specific facts in explicitly selected out-of-tree support packages.
The production path must not import or invoke ACT. Reference adapters belong in
a separate module and may only enter through an explicit engine selection.

The Rust `egg_bridge` is a general-purpose e-graph dependency boundary. Merlin
owns all rule eligibility, buffer-aware extraction, constraint construction,
fallback and result checking. Unknown semantics or solver results are failures,
not evidence that a target operation is unsupported.

Portable pointwise/reduction tile planning and byte transport now live in
`compiler/transforms/pointwise.py` and `reduction.py`. Historical modules here
alias those exact owners, including class/function globals and patches; record
pickle spellings remain stable. `_compiler_sources` in `snapshot.py` observes
both implementation files and the aliases for active artifacts. Earlier saved
source inventories remain historical inputs and are not rewritten by this move.

Ordered contraction tiling likewise lives in compiler/transforms:
`contraction_k.py`, `contraction_mn.py`, `contraction_mnk.py` and `panels.py`.
The old `ordered_k_tiling.py`, `ordered_mn_tiling.py`, `ordered_mnk_tiling.py`
and `streamed_mnk_tiling.py` are exact aliases. Actual shared C transport
consumes these owners; existing OOT callers
retain the same records, functions and module-global patch identities through
the aliases. The active snapshot observes each new implementation explicitly.
