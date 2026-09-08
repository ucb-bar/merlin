# q534 whole-model headroom and fail-closed roofline

## Decision

Optimize the macro path next: fuse exact quantized epilogue/residual work across task boundaries,
narrow intermediate encodings, and remove spatial im2col/repeated movement. Do not lead with another
matmul tile search. q534 is only **3.479%** loop-matmul-active
and **3.505%** reservation-station-active over the full
measured interval, while it still retires 751,827,149 instructions and reports
1,196,945,312 Gemmini DMA-interface bytes.

## Exact q534 observation

| Quantity | Value |
| --- | ---: |
| FireSim/U250 warm-measured cycles | **1,537,416,019** |
| Retired instructions | 751,827,149 |
| Read DMA bytes | 1,142,808,576 |
| Write DMA bytes | 54,136,736 |
| Loop-matmul active cycles | 53,479,788 |
| Reservation-station active cycles | 53,892,695 |
| Exact logits | 1,000; bad=0 |

The content-addressed compiler analysis records 4,091,330,560 MACs
for the frozen workload. Combined with the hardware counters, this gives arithmetic diagnostics of
**2.661173 MAC/cycle** end to end and
**3.418143 MAC/byte** at the Gemmini DMA interface. During
`loop_matmul_active`, the diagnostic rate is 76.502371 MAC/cycle,
or 29.884% of the 16x16 geometry ceiling.
The 256 MAC/cycle value is geometry only, not a measured sustainable peak. DMA counters are not a
cache/DRAM-level traffic decomposition.

## Controlled q530 -> q534 response

The accelerator schedule and DMA are attested unchanged. Removing
229,709,973 retired instructions saved 166,648,204 target cycles:
**0.725472 cycles per removed instruction** in this
one interval. DMA deltas are 0 read bytes and
0 write bytes. The two-point zero-instruction intercept is
991,986,308 cycles. It is not a physical fixed term or
a general predictor. Its margin below the 1B goal is only
8,013,692 cycles, so scalar peepholes alone do not
provide a robust path through the remaining 537,416,019-cycle gap.

## Why there is no roofline curve yet

The repository's `merlin.perf.roofline.empirical_roofline` API was invoked on q534 and returned
**refused**. This refusal is retained in `receipt.json`. The stricter
`build_rtl_roofline.py` contract additionally requires all of the following:

1. complete convolution geometry from compiler command buffers (current exact MAC recovery is
   `UNKNOWN`);
2. physical byte readings with counter bindings tied to the exact RTL-facts digest;
3. at least four plan-matched samples per compute/movement calibration sweep;
4. at least four structurally empty runs under this exact measurement protocol; and
5. a complete joint-occupancy partition proving overlap eta and composition.

q534 has useful marginal activity counters, but no joint partition. Summing them or defaulting to
`max` would fabricate composition. Likewise, no sustainable bandwidth is inferred from one
workload point. The exact diagnostic intensity therefore must not be plotted against an invented
bandwidth roof.

## Evidence that unlocks the next decision

A candidate deserves cheap warm A/B execution when four-model static analysis shows fewer host
tasks/dynamic operations, fewer i32 device-host boundaries, and lower encoded/im2col/repeated-weight
bytes without legality, capacity, residency, hazard, ownership, or semantic regressions. A new
queued hardware point is justified only after that structural change is material. Formal remaining
headroom requires the five calibration/provenance items above.
