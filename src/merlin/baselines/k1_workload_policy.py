"""K1 board feasibility for the optional cross-framework baseline study.

These classifications describe one measured board/corpus pairing. They are not
properties of capture bundles or a reusable compiler capability declaration.
"""

from __future__ import annotations

# The eight models whose int8 footprint fits the 3.8 GB board (~3.4 GB usable).
# From the full-fidelity recaptures; an unlisted model has no inferred fit.
K1_RUNNABLE: frozenset[str] = frozenset(
    {"tiny_llama", "smolvla", "bitvla", "groot_n1d7", "rdt", "rdt2", "xr0", "small_llama"}
)

# The three 7B-class VLAs are RAM-infeasible for whole-model on-board runs even
# at int8 (fp32 embeddings dominate). They are attempted and RAM-gapped, not
# silently omitted.
K1_RAM_INFEASIBLE: frozenset[str] = frozenset({"openvla", "molmoact", "pi05"})

# ResNet50 v1.5 is not in either K1 board cohort yet. Its earlier max-pool
# capture blocker is fixed: the capture now supplies a shape-only window
# operand, and its fp32 bundle lowers and matches on x86. The W8A8 path also
# has torchao affine decompositions and lifted FC weights. Those host checks
# do not establish that the model fits or runs on this board, so neither
# cohort should infer board feasibility from them.
