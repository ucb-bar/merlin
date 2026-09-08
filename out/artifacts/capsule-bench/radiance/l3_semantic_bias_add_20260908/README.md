# Radiance semantic-family bias-add L3 qualification (2026-09-08)

## Outcome

The first compiler-side family-selection bridge is live at the real Muon LLVM-MLIR emitter.  Merlin
parsed the public `RP16_bias_add_fp32_pt` linalg workload, lowered it to a command buffer, derived a
`bias_add(fp32, rows=16, cols=16)` request and Radiance hardware capabilities from contracts, selected
`kernels/bias_add`, and emitted its own scalar LLVM-dialect kernel.  No ModelBlaster/Radiance reference
source is copied, linked, called, or dispatched.

| Arm | Private oracle change | GSIM verdict | Elements | Active simulation |
|---|---|---:|---:|---:|
| positive | none | **PASS** | 256/256 | 146.120 s |
| negative control | first expected value +100 | **FAIL (expected)** | 256 | 149.254 s |

Both arms have byte-identical public command buffers, selection reports, emitted MLIR, and submitted
Muon ELF (`02d0707f624ff54a67f34f24bde26fa5b5c39c90d521332eeca42d2cea1f60a7`).  Only the trusted
Rocket-side carrier receives the private expected answer.  The pair therefore shows that the exact
selected compiler output computes the capsule correctly and that the L3 result-page witness can reject
it when the answer key is wrong.

The report retains decisions for all 23 PR1 families: one selected qualified family, 20 refused
qualified alternatives, and the two experimental flash-attention families disabled.  Selection is
not itself numeric qualification; this L3 pair qualifies only this generated bias-add implementation
at shape 16x16 fp32.  The remaining 20 qualified strategies still need command-buffer semantic
extractors, emitter registrations, and per-shape L3 qualification before whole-corpus use.

This is correctness evidence, not a performance measurement.  The 360,000-cycle bound deliberately
keeps the GSIM mailbox observation window open; `sim_active_s` is simulator wall time.

## Replay and verification

Replay either arm into fresh directories (maximum wall timeout is enforced at 300 seconds):

```bash
PYTHONPATH=merlin/python .venv/bin/python \
  out/artifacts/capsule-bench/radiance/l3_semantic_bias_add_20260908/run_qualification.py \
  --case positive \
  --emulator /scratch/agustin/tmp/gsim-radiance-l3-v6-20260907/emulator \
  --work /scratch/agustin/tmp/radiance-bias-positive-replay \
  --publish /scratch/agustin/tmp/radiance-bias-positive-public
```

Use `--case negative` for the fail-capable control.  The recorded emulator SHA-256 is
`458d11844538463e5f4ab1f2d6314934dc0d4fe71df07131eebc5aa29d2633be`; the receipts also pin the
selection and generated hardware-contract bytes used for the run.

Verify the sealed artifact without simulation:

```bash
PYTHONPATH=merlin/python .venv/bin/python \
  out/artifacts/capsule-bench/radiance/l3_semantic_bias_add_20260908/verify.py
```
