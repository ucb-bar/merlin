# Radiance L3 streaming-result evidence (v2, updated 2026-09-08)

## Outcome

The production GSIM adapter now streams numeric results through a runner-owned,
32-word mailbox. After copying a chunk and its count, the Muon harness executes
a release fence and publishes the single changing word `READY_BASE ^ sequence`.
Rocket acquires after observing that word, reads the mailbox, orders those reads
before publishing `ACK_BASE ^ sequence`, and Muon acquires that ACK before
reuse. A stale constant token cannot authorize a different transaction. Rocket
therefore never coherently reads a full 1 KiB, 256-element output page. Output
backing buffers remain private to the Muon-side harness and their addresses are
absent from the public manifest.

The carrier alone contains comparison intervals derived from the private,
post-submission golden. Expected values are absent from the submitted MLIR,
Muon harness/ELF, and Muon build-cache identity. The adapter accepts only an
exact final PC at `merlin_numeric_pass` or `merlin_numeric_fail` as the numeric
witness. A cycle-cap line is observation metadata, not a verdict.

This artifact establishes:

| Case | Shape | Expected use | GSIM numeric result | Bound | Active simulation |
|---|---:|---|---|---:|---:|
| `rp10_pass` | 32 f32 words | legacy v1 positive control | **PASS** | 120,000 cycles | 41.283 s |
| `rp10_negative_control` | 32 f32 words | legacy v1 perturbed-golden control | **FAIL (expected)** | 120,000 cycles | 41.190 s |
| `r4_rmsnorm_observed_fail` | 256 f32 words | historical initial mailbox, PR RMSNorm | **PASS** | 360,000 cycles | 122.951 s |
| `rp12_embed_scale` | 256 f32 words | changing-token mailbox, independent elementwise control | **PASS** | 360,000 cycles | 123.012 s |
| `rp12_negative_control` | 256 f32 words | same submitted RP12 ELF; first golden value perturbed by +100 | **FAIL (expected)** | 360,000 cycles | 122.898 s |

The RP12 positive and negative submitted Muon ELFs are byte-identical, SHA-256
`cfbc6cdcff0704e46c047eca85b9d46912d3c1cbafde7abaa455e1ccf5bdda64`.
Their trusted carriers and fused SoC ELFs differ. This proves that the streaming
protocol is fail-capable and that its verdict is neither hard-wired nor embedded
in the submitted kernel.

## Closed diagnosis

The old direct-read ABI passed 32 words but failed both independent 256-word
cases, pointing to long-page visibility rather than an RMSNorm arithmetic bug.
With the mailbox, RP12 and R4 each pass 256/256 through eight acknowledged
chunks. RP12 was rerun after changing-token publication and acquire/release
hardening; its negative arm reaches the exact fail PC with an unchanged Muon
ELF. The R4 receipt is retained as historical initial-mailbox evidence and was
not used to validate the ordering hardening. The RP12 pair closes the observed
1 KiB cross-master transport failure without weakening numeric grading.

This is correctness evidence only. `bounded_observation.performance_measurement`
is false; the cycle caps are not kernel performance measurements.

## Replay

The submitted MLIR, command buffers, capsule contracts, public inputs, numeric
receipts, and GSIM logs are frozen below `cases/`. Private `golden.yaml` files,
generated result carriers, and fused SoC ELFs are deliberately not published:
the carriers encode the answer intervals. The replay resolves the answer from
the local benchmark corpus after submission. Run one case into a fresh work
directory:

```bash
PYTHONPATH=merlin/python .venv/bin/python \
  out/artifacts/capsule-bench/radiance/l3_result_page_v2_20260908/run_representative.py \
  --case rp10_pass \
  --emulator /scratch/agustin/tmp/gsim-radiance-l3-v6-20260907/emulator \
  --work /scratch/agustin/tmp/radiance-rp10-replay
```

The script enforces a wall timeout no greater than 300 seconds. Select
`rp10_negative_control`, `r4_rmsnorm_observed_fail`, `rp12_embed_scale`, or
`rp12_negative_control` for the other frozen arms. It writes a complete
`adapter_result.json` into the requested work directory.

Verify the sealed evidence without simulation:

```bash
.venv/bin/python out/artifacts/capsule-bench/radiance/l3_result_page_v2_20260908/verify.py
```

The emulator used for the recorded runs is
`/scratch/agustin/tmp/gsim-radiance-l3-v6-20260907/emulator`, SHA-256
`458d11844538463e5f4ab1f2d6314934dc0d4fe71df07131eebc5aa29d2633be`.
The implementation extends the declared-result work in `a0222bff` and the
original evidence snapshot in `88ec5bb7`; all published files are sealed in
`SHA256SUMS`.
