# Automatic exact Gemmini residual-add placement

This public, isolated compiler child adds automatic placement for the exact subset of residual
addition implemented by Gemmini's `LOOP_WS is_resadd` mode. It recognizes two row-major i8 tensors
whose source semantics are exactly

```
clamp_i8(add_i32(sign_extend_i8(a), sign_extend_i8(b)))
```

with an optional ReLU lower bound. It emits `K=0`, `is_resadd=1`, identity A/B/C scales, tiles to
the pinned `ACC_ROWS/2` partition, and records the choice in the command buffer. Any changed add,
clamp, layout, dtype, or scale/rounding order stays on the host with a reason. Exhaustive testing
covers all 65,536 i8 input pairs, with and without ReLU.

## Canonical PT2E ResNet-50 result

The automatic census finds all 16 residual adds but selects **0/16**. This is an exactness result,
not a missing pattern:

- The first 15 joins consume f32 values. Their convolution branches apply a scalar f32 scale,
  a per-channel f32 scale, and a per-channel f32 bias before the join. The source rounds only once,
  after the f32 residual addition. `is_resadd` instead consumes i8, rounds/clamps each scaled input
  before adding, has only one scalar scale per input, and has no bias input.
- Eleven of those 15 joins retain one live f32 residual branch rather than an exclusive i8 value.
  The four stage-transition joins have two i32-derived f32 branches, but still have the scale,
  bias, input-width, and rounding-order blockers.
- Source op 1204 is the sixteenth residual join and has no unique exact round-even/clamp i8 sink.

Therefore lowering any canonical site to `is_resadd` would change logits. The full generated target
remains byte-identical to the compute-only parent (`ac1fb50a...`), proving the refusal path adds no
runtime work. This compiler descends from the q545 compute-only candidate, which was exact but
5.11% slower than q535 on FireSim; this artifact is not a promotion and makes no performance claim.

The right follow-up is the explicit native-aligned per-tensor recapture: make the model contract
materialize i8 branch values and hardware-order scaling/rounding before the residual join, then
requalify all 1,000 logits. Silently approximating the canonical per-channel graph is not allowed.

## Evidence

Run the self-contained verifier:

```sh
artifact=/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-bench/gemmini/development_automatic_loop_ws_resadd_20260908
PYTHONDONTWRITEBYTECODE=1 python "$artifact/verify.py"
```

`validation/resnet50_resadd_census.json` is payload-free and binds the canonical prepared-source
hash, all 16 source operation indices, every refusal reason, the unchanged target hash, and the
scheduled instruction count. No model payload, logits, FireSim run, L3 run, queue action, or change
to the working 92/96 compiler is included.

The hardware contract was checked against Jack's Chipyard commit
`e27c6561c0066c1f60bf4eb4885a38391c850ac0`: `gemmini.h`'s `tiled_resadd`, generated float scale
types/round-near-even macros, and `LoopMatmul.scala`'s `is_resadd` load/store path.
