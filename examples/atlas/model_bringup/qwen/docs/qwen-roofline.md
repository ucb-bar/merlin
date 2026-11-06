# Qwen kernels through the NPU roofline model

Integration note (2026-11-05): This is historical evidence replayed from `5884524e3cf85027e08734526c1e3dbf0a093686`; its original measurements and unsupported-variant observations have not been rerun by this port. Current D128 attention and width-1536 RMSNorm support, explicit Qwen epsilon, local source selection, and validation status are described in the [Qwen integration README](../README.md).

Ran all five compiled Qwen kernels, each in unscheduled, boundary-DMA-only, and scheduled form, through `npu_speed_of_light.roofline.speed_of_light` from the selected npu_model checkout at `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`. The model checkout was read without modification. These are simulator results.

There are 30 measured runs: five kernels × three pipelines × two memory backends. Each run decodes the existing emitted words, requires exact instruction re-encoding, uses RTL numerical mode, and checks virtual-reference output bytes and the entire initialized DRAM region, including input and guard preservation. All 30 passed with zero runtime errors and no outstanding execution-unit work. The roofline tool's nine tests also passed when run from the model repository. An initial invocation from the Atlas directory failed two built-in-program tests because their assembly paths depend on the working directory; custom emitted programs used here do not depend on those paths.

## Default memory curve: all five are DRAM-bound

The default backend uses `ee290sim_vcs_probe.json`, measured on VCS EE290SimConfig with DRAMSim2 refresh disabled. The roofline tool uses the fastest curve peak across read/write mixes for its optimistic DRAM throughput lower bound.

| Scheduled kernel | Measured cycles | Resource lower bound | Bound efficiency | Remaining cycle reduction ceiling | Intensity, FLOP/B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention | 16,127 | 15,231 (DRAM) | 94.4% | 5.6% | 23.27 |
| SwiGLU | 7,691 | 6,912 (DRAM) | 89.9% | 10.1% | 19.20 |
| Projection K=128 | 7,717 | 6,912 (DRAM) | 89.6% | 10.4% | 25.60 |
| Projection K=256 | 13,727 | 12,458 (DRAM) | 90.8% | 9.2% | 28.44 |
| Projection K=512 | 25,745 | 23,551 (DRAM) | 91.5% | 8.5% | 30.12 |

**Bound efficiency = lower-bound cycles / measured cycles.** It describes throughput relative to the best bound for this exact work, not utilization of peak compute. For example, attention is 94.4% of its resource-bound throughput, while achieving only 32.51 MXU FLOP/cycle against the configured two-MXU peak of 4,096 FLOP/cycle. Those statements are consistent: this program moves too much data per FLOP to approach compute peak.

The ridge point is approximately 2,765–2,771 FLOP/B, while these kernels have only 19–30 FLOP/B. Their points lie on the memory side of the roofline. The corresponding optimistic DRAM rate is approximately 1.48 B/cycle. Small per-program ridge differences arise from finite-transfer interval counting.

With work and traffic unchanged, the optimistic remaining speedup is only 1.059× for attention and 1.093–1.116× for the other kernels. Dependencies, startup and transfer latency may prevent even those gains. The bound ignores those effects and is not a guaranteed reachable schedule.

## Scheduling moves the programs closer to the same bound

| Kernel | Unscheduled efficiency | Boundary-only efficiency | Scheduled efficiency |
| --- | ---: | ---: | ---: |
| Attention | 80.7% | 81.7% | 94.4% |
| SwiGLU | 78.3% | 79.7% | 89.9% |
| Projection K=128 | 83.0% | 84.9% | 89.6% |
| Projection K=256 | 82.7% | 86.5% | 90.8% |
| Projection K=512 | 82.5% | 87.4% | 91.5% |

Within each workload, the measured DRAM bytes and lower bound are identical across the three pipelines. Scheduling improves how well that work is overlapped; it does not reduce the byte traffic in these programs. The default-memory measurements exactly reproduce the earlier simulator cycle counts despite using the newer checkout's roofline implementation.

## Faster-memory sensitivity: other limits become visible

This scenario uses fixed 43-cycle response latency and one 32-byte beat per cycle. It is a hypothetical faster link, not the default hardware measurement.

| Scheduled kernel | Measured cycles | Largest resource charge | Bound efficiency |
| --- | ---: | --- | ---: |
| Attention | 2,513 | DRAM: 703 cycles | 28.0% |
| SwiGLU | 1,407 | VPU: 391 cycles | 27.8% |
| Projection K=128 | 987 | DRAM: 319 cycles | 32.3% |
| Projection K=256 | 1,651 | DRAM: 575 cycles | 34.8% |
| Projection K=512 | 2,979 | VLOAD path: 1,120 cycles | 37.6% |

The classical FLOP/DRAM-byte roofline still classifies all five as memory-side: their intensities are below its approximately 128 FLOP/B ridge. The fuller resource model is more informative: SwiGLU's largest charge becomes the VPU, and K=512's becomes the VLOAD path that moves VMEM data into matrix registers. A FLOP/DRAM-byte chart alone misses those limits.

The larger gap to the fast-memory bound is consistent with serial dependencies, startup/transfer latency and incomplete overlap; this run does not attribute that gap precisely. It is not evidence that a scheduler can automatically recover the full gap. Additional dependency-aware bounds and timelines are needed before making that claim.

## Compiler implications

1. **Reduce traffic and increase reuse first on the default memory system.**
   Attention already approaches the bound for its existing transfers. Keep
   tiles resident across windows/kernels, reuse K/V across grouped query heads,
   and amortize weights/activation loads across more useful work.
2. **Use faster-memory experiments to evaluate scheduling improvements.**
   The default memory link masks compute/dependency costs. On the faster link,
   test software pipelining, double buffering and overlap of loads with MXU/VPU
   work, while accounting for register pressure and real dependencies.
3. **Model the VPU and VMEM-to-register path explicitly.** Optimizing only MXU
   arithmetic cannot address the tightest fast-memory resource charge for
   SwiGLU or the longest compiled projection.
4. **Keep numerical qualification independent.** A good roofline position does
   not fix the previously observed signed SwiGLU underflow or projection error.
   These runs use the original positive input panel and diagnostic FP8/BF16
   policies. FLOP counts include MXU MACs only, at two FLOPs per MAC; nonlinear
   and elementwise VPU operations are covered by resource charges, not added to
   the plot's FLOP numerator. Compute peak includes both configured MXUs; the
   emitted programs use MXU0, and using MXU1 also requires precision qualification.

Full K=1536 projection, RMSNorm/QKV and full residual MLP have no emitted program from the earlier campaign, so the executable roofline API cannot run them. Their compilation blockers remain recorded in [the main report](qwen-results.md).

## Reproduction

The tables above summarize all five kernels. Detailed per-resource reports and provenance remain local in `out/qwen-roofline/roofline.txt` and `out/qwen-roofline/roofline.json`. Raw datasets and plot images are not tracked in the repository.

From the Merlin repository root, set `NPU_MODEL_ROOT` to the selected physical npu_model checkout and use the prepared model environment and emitted campaign artifacts:

```sh
PYTHONPATH="$NPU_MODEL_ROOT" out/model-venv/bin/python examples/atlas/model_bringup/qwen/roofline_workloads.py --campaign out/qwen-final -o out/qwen-roofline-new
```

The runner reads the selected model checkout through `PYTHONPATH`; unlike the original campaign's older pinned model, that checkout must include `npu_speed_of_light/roofline.py`. It records model revision, roofline/curve hashes, program hashes, and per-resource charges. Existing results are not overwritten.
