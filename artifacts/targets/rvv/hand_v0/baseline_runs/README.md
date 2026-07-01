# hand_v0 baseline measurements (the comparison anchor — do not lose)

These are the **durable baseline `results.yaml`** for the `hand_v0` RVV package on the kernel-sized
workloads. Every tuned fork's `certify_rvv(..., baseline_run_dir=...)` compares against the matching
directory here (`delta_vs_baseline.speedup` is credited only if the fork's correctness gate passes).
Without these the beam-search loses its reference point.

One subdir per workload key `<op>_<dtype>_<shape>`:
- `matmul_f32_64/` — fp32 GEMM 64×64×64. spike cycles ≈ 27.1M; gate fp32_cos 1.0. KNOWN GAPS:
  emits `vfmul.vv`+`vfadd.vv` (no fused `vfmacc`), `vsetivli` not `vsetvli`.
- `softmax_f32_64/` — fp32 softmax 64×64. gate cos 1.0 but `any_rvv=False` — NOT vectorized
  (schedule matches only matmul/batch_matmul; softmax generics fall to scalar loops).

The workload BUNDLES are not stored (regenerable, deterministic seed):
```
python -m merlin.rvvgen.workloads matmul_f32  -M 64 -N 64 -K 64 --out-root <root>
python -m merlin.rvvgen.workloads softmax_f32 -M 64 -N 64        --out-root <root>
```
Re-measure the baseline (only if the toolchain/target changes — cycles are target-specific):
```
python -m merlin.rvvgen.runner --package generated_targets/rvv/hand_v0 \
    --workload <root>/matmul_f32_64x64x64 --targets spike \
    --run-id hand_v0_matmul_f32_64 --runs-root <tmp>
# then copy <tmp>/hand_v0_matmul_f32_64/results.yaml here.
```
K1 (real-silicon cycles) baselines land here once the K1 runtime is wired (S2.4–S2.6).
