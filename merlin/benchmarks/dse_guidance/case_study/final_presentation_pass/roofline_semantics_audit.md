# Roofline semantics audit (`arithmetic_intensity_roofline`)

The roofline is conceptually strong but high-risk for overclaim, so its semantics are pinned here before
the plot is shown. Source: `arithmetic_intensity.csv` (computed by `arithmetic_intensity.py`).

### The six questions, answered
1. **Is AI computed from repeated-head WEIGHT bytes only?** — **Yes.** AI = MACs ÷ repeated-head *weight*
   bytes. That is why the axis is labeled **"weight-stream arithmetic intensity (MAC / repeated-head weight
   byte)"**, not generic "arithmetic intensity".
2. **Are activations / outputs / KV-cache / scales / partial sums / intermediates excluded?** — **Yes**, all
   excluded. This is a *weight-stream* quantity, explicitly **not** full-memory arithmetic intensity. (Stated
   in the subtitle caveat.)
3. **Why does reload-every-step collapse all workloads to the same AI?** — Because reloading the weight for
   every MAC gives AI = MACs / (MACs · dtype_bytes) = **1/dtype_bytes** — independent of the model. At bf16
   that is 0.5 MAC/byte for every workload. It is a *floor*, not a per-model number.
4. **Is residency gain exactly K?** — **No.** gain = **(prefix + repeated·K) / (prefix + repeated)**. It
   approaches K only as prefix → 0; for prefix-heavy workloads it is much less than K. The CSV stores the
   exact formula value.
5. **Captured-config or deployment-composition magnitudes?** — **Deployment-composition** (the params come
   from `real_config` composition feeding `arithmetic_intensity.csv`), so the AI values are deployment-scale.
6. **Are the machine-balance values illustrative or grounded?** — **Illustrative / parametric.** The ridge is
   drawn as a **band over a range of hypothetical machine balances** (and one illustrative B). No chip is
   assumed; the y-axis is a **normalized roofline bound**, never attainable throughput on real hardware.

### Wording (enforced by `check_final_plots.py`)
- Title: *"Residency shifts weight-stream arithmetic intensity across machine-balance regimes."*
- x: *weight-stream arithmetic intensity (MAC / repeated-head weight byte)* · y: *normalized roofline bound
  (hypothetical balance)*.
- Subtitle caveat: *"requirement / modeling view — not measured performance, not full-memory AI."*
- **Banned:** "hardware-independent roofline", "attainable throughput", "compute-bound on real hardware",
  "predicted performance".
