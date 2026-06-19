# Capture-shape sensitivity (P20-S3) — resolving two headline caveats with evidence

Variant recaptures (env-override only, no loader surgery; raw MLIR in `recaptures_decode/`, gitignored,
regenerable via `variant_capture.py`). Structural only; magnitudes are random-init.

## Caveat 1 — "GEMV-like is capture-M-induced" → RESOLVED (holds at true decode)
tiny_llama recaptured at **M=1 (true single-token decode, `M2M_SEQ=1`)** vs the committed **M=4 prefill**:

| variant | M | n linear ops | dominant geometry |
|---|---|---|---|
| prefill (committed) | 4 | 15 | gemv_like |
| decode (variant) | 1 | 15 | gemv_like |

Both are `gemv_like` (the projections are N≈2048 ≫ M for M∈{1,4}). **So the GEMV finding is genuine for the
decode regime — it is NOT a small-M capture artifact that vanishes at true decode; M=1 confirms it.** The
artifact concern would only apply at a *large prefill* M (hundreds), where these become skinny/square — that
is the regime to capture separately if a prefill-sized DSE point is wanted. Net: the caveat is refined, not
a defect — the captured M=4 was already in the decode regime for this workload.

## Caveat 2 — "rdt giant op doesn't generalize" → RESOLVED (depth-2 artifact, confirmed)
rdt recaptured at **depth=6 (`M2M_RDT_DEPTH=6`)** vs the committed **depth=2**:

| variant | depth | n linear ops | top-op MAC share | #cross_attn.kv ops | cross_attn.kv class share |
|---|---|---|---|---|---|
| committed | 2 | 20 | **0.871** (1 op) | 2 | 0.878 |
| variant | 6 | 48 | **0.292** | 6 | 0.883 |

The single `blocks.1.cross_attn.kv` op falls from **87.1% → 29.2%** of MACs as depth grows; the cross-attn-
to-image-context **op class** stays ~88% but is now spread across 6 ops. **So "one giant op dominates" is a
depth-2 capture artifact** — the honest finding is "cross-attention to the 4096-token image context is the
dominant compute *class* for RDT (~88%), distributed across the blocks at real depth," NOT a single hot op.

## Deferred
Full true-decode M=1 recapture of the other autoregressive workloads (small_llama, openvla, molmoact,
bitvla) needs per-loader edits (their loaders force `use_cache=False` / hardcode seq) — higher-risk,
lower marginal value since M=1 here already demonstrates the decode regime. Tracked as a follow-on.
