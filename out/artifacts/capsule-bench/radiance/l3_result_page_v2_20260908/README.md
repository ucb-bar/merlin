# Radiance L3 declared result-page evidence (v2, 2026-09-08)

## Outcome

The production GSIM adapter now has an opt-in, declared result-memory ABI. The
Muon harness allocates linker-visible output buffers and a READY/ACK status
line; a runner-generated Rocket carrier reads those buffers and grades them
against the private post-submission golden. The adapter accepts only an exact
final PC at `merlin_numeric_pass` or `merlin_numeric_fail` as the numeric
witness. A cycle-cap line is observation metadata, not a verdict.

This artifact establishes:

| Case | Shape | Expected use | GSIM numeric result | Bound | Active simulation |
|---|---:|---|---|---:|---:|
| `rp10_pass` | 32 f32 words | PR batched GEMV | **PASS** | 120,000 cycles | 41.283 s |
| `rp10_negative_control` | 32 f32 words | same ELF; first golden value perturbed by +100 | **FAIL (expected)** | 120,000 cycles | 41.190 s |
| `r4_rmsnorm_observed_fail` | 256 f32 words | PR RMSNorm | **FAIL (real)** | 360,000 cycles | 121.974 s |
| `rp12_embed_scale` | 256 f32 words | independent elementwise control | **FAIL (real)** | 360,000 cycles | 123.319 s |

The RP10 kernel ELF is byte-identical between the positive and negative arms;
only the trusted carrier/SoC ELF differs. This proves that the new verdict is
not a hard-wired completion marker.

## Honest limitation and narrowed diagnosis

R4 is not relabelled as a pass. Its exact submitted LLVM MLIR and command
buffer produce 256 correct values in the independent Cyclotron run. Replaying
those values through the carrier's IEEE ordered-integer interval algorithm
accepts 256/256. Its argument order is also correct (`G`, `X`, `Y0`). Thus the
R4 GSIM failure is neither golden/dtype binding nor interval construction.

The distinct RP12 elementwise case was rerun with the exact canonical input
bytes and also fails at 256 words, while RP10 passes at 32 words. Both
256-word cases use the same 1 KiB result-page geometry and linked addresses.
Their GSIM logs report no writes on the instrumented Rocket DRAM port, and
Radiance's LSU contract documents relaxed coherence requiring explicit
flushes. The harness emits a RISC-V `fence` before READY, but this evidence now
points to incomplete long-page visibility/flush in this GSIM/SoC path rather
than an RMSNorm-only sqrt/div defect. The current one-bit final-PC ABI cannot
expose the first stale word, so it does not prove which cache level loses
visibility. Until that transport issue is repaired, L3 numeric grading is
demonstrated for 32 words but must fail closed for these 256-word cases. The
next implementation is a runner-owned 32-word streaming mailbox with a
READY(sequence,count)/ACK handshake.

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
PYTHONPATH=merlin/python python \
  out/artifacts/capsule-bench/radiance/l3_result_page_v2_20260908/run_representative.py \
  --case rp10_pass \
  --emulator /scratch/agustin/tmp/gsim-radiance-l3-v6-20260907/emulator \
  --work /scratch/agustin/tmp/radiance-rp10-replay
```

The script enforces a wall timeout no greater than 300 seconds. Select
`rp10_negative_control`, `r4_rmsnorm_observed_fail`, or `rp12_embed_scale` for
the other frozen arms. It writes a complete `adapter_result.json` into the
requested work directory.

Verify the sealed evidence without simulation:

```bash
python out/artifacts/capsule-bench/radiance/l3_result_page_v2_20260908/verify.py
```

The emulator used for the recorded runs is
`/scratch/agustin/tmp/gsim-radiance-l3-v6-20260907/emulator`, SHA-256
`458d11844538463e5f4ab1f2d6314934dc0d4fe71df07131eebc5aa29d2633be`.
The repository base was `3edd9334fd39ce97b5cba50d715be5b6d8c0d6c1`; the
uncommitted source hashes are sealed in `SHA256SUMS`.
