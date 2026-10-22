> Historical comparison notes. Begin with [README.md](README.md) for the current
> TensorIR/MetaSchedule and EL4 handoff. Results below retain their original scope.

# TVM–Gemmini comparison

The new [BYOC backend](byoc/README.md) uses TVM pattern partitioning, fused TensorIR and replayable schedules. Its [bounded validation](byoc/VALIDATION.md) passes matmul, BMM and convolution on Spike and Verilator; [operation coverage](byoc/SUPPORT.md) records native subsets and justified limits for all 51 requested names.

Apache TVM is a machine-learning compiler framework. It imports or constructs a model as an IRModule, transforms its graph and tensor programs, and generates executable code for a selected target. Relax represents graphs and stateful interfaces; TensorIR represents operator computations, loops and tensor intrinsics. The [Apache v0.19 architecture](https://github.com/apache/tvm/blob/v0.19.0/docs/arch/index.rst) explicitly supports manually constructed IR, custom transformations and target-specific lowering. Those extension mechanisms establish how this integration uses TVM; they do not prescribe a research benchmark's numerical tolerances, quantization, tuning budget or measurement boundary.

This directory owns model recipes, independent numerical references, input preparation and session verification. The independent pinned TVM fork owns ONNX import, Relax optimization, integer decomposition, primitive scheduling, bounded learned search and static export. Its [pinned integration guide](https://github.com/reed-nicolas/tvm/blob/6b376cc8895a83387ec38cc06ef8f7b09827b071/apps/gemmini/README.md) documents backend APIs, adapter builds and exported-graph execution. TVM is not a Merlin submodule; all commands below run from the Merlin comparison checkout.

## Current bounded compiler-development scope — 2026-10-19

The current task uses checkpoint-backed **cutouts**, with synthetic inputs and explicit FP32 conversion before W8A8. It does not require the historical source-exact environment or qualify full-model quality. The selected SmolVLA scope is **one original transformer block plus action/update**, not all 16 denoiser layers. The [block verification driver](verify_blocks.py), [workload recipes](block_workloads.py) and [SmolVLA assembly](smolvla_denoise_block.py) implement this smaller experiment. The [detailed report](../../../../out/artifacts/verification/gemmini/tvm/int8-blocks-20261019T215051Z/final-report.md) and [machine-readable evidence](../../../../out/artifacts/verification/gemmini/tvm/int8-blocks-20261019T215051Z/final-report.json) retain exact commands, scopes, source snapshots, dispatch, resources and failed attempts.

| Cutout | Gemmini learned projections per call | Lowest INT8-versus-FP32 output cosine over two fixtures | Functional Spike result |
| --- | --- | --- | --- |
| TinyLlama layer-zero query head zero, its shared KV head and O-column contribution, then full 2048→5632→2048 FFN; four tokens | 7: Q/K/V/O, gate/up/down | Attention head 0.874792; FFN 0.998351; final residual hidden 0.999999946 | Four calls pass; all seven exposed INT32 projections exact; floating outputs pass `rtol=1e-4, atol=1e-5`; wrong-oracle control reports 22. |
| SmolVLA action projection and Euler update, external expert hidden fixture; chunk50 | 1: action output | Velocity 0.999903; updated latent 0.999999686; action slice 0.999999548 | Four calls pass with exact exposed INT32 projection, floating gate and wrong-oracle control. |
| SmolVLA original expert layer zero, original final norm, four action/time projections and Euler update; chunk50, prefix4, hidden720, all15 query/5 KV heads, FFN2048 | 11: block Q/K/V/O and gate/up/down; action input/time-in/time-out/output | Velocity 0.999905; updated latent 0.999996139; action slice 0.999996509 | Four calls pass floating gate; maximum compiler velocity error `4.7684e-7`; wrong-oracle control reports 22. Native contractions have complete INT64 checks; device INT32 intermediates are not exposed by this cutout. |

Compiler agreement is a separate comparison against quantized PyTorch: these passing cutouts have floating-output cosine approximately one. The lower TinyLlama attention cosine is quantization error and remains visible even though its residual-hidden cosine is high. Neither metric establishes language or policy quality. Two synthetic fixtures and repeated calls establish only their declared functional coverage; they are not representative calibration or held-out evaluation.

Weights use signed INT8 symmetric per-output-channel maxabs scaling; activations use dynamic per-token power-of-two scales, nearest-ties-even rounding and `[-127,127]` codes. Accumulation is INT32, with FP32 dequantization/bias. This is an explicit project recipe before TVM import, not automatic TVM quantization or a fully integer model. QK/PV products, softmax, RMSNorm, SiLU, rotary/time embeddings, quantization/dequantization, cache assembly and Euler/slice work stay on the CPU. SmolVLA time embeddings retain their source FP64 operations.

The actual path is selected PyTorch modules → public Dynamo ONNX export with the existing `_int_mm` translation to standard `MatMulInteger` → normal Relax ONNX import → existing Gemmini integer lowering and TensorIntrin scheduling → LLVM RV64 objects → static C graph orchestration plus the exact-header Gemmini adapter and audited newlib math → one-hart Gemmini Spike. This pass adds **no TVM source patch**; the previously disclosed pinned-fork core changes below still apply. There is no new FireSim/RTL or performance qualification.

Three opt-in development passes use public TVM APIs: [static buffer specialization](static_buffer_abi.py) binds symbolic compact-buffer offsets to the standalone zero-offset ABI; [CPU row scheduling](cpu_row_schedule.py) uses legal `compute_at` placement and standard allocation compaction to remove large hidden scratch; the same helper serializes CPU parallel loops for the declared one-hart guest. For the SmolVLA block, private attention scratch shrinks 162000→216 bytes, softmax reductions 6000→8 bytes and one mean 200→4 bytes. Illegal row schedules retain the original function; incompatible buffer ABIs are refused. No exporter/runtime safety guard is removed. The guest explicitly supplies single-hart `__errno` storage and links unchanged upstream math archives; unsupported TVM workspace/thread-pool callbacks, unresolved symbols, heap/file dependencies and hardware-loop instructions are rejected.

A separate **16-layer denoiser diagnostic failed** the same floating gate after executing a complete inference: velocity cosine versus quantized PyTorch `0.9999828591`, maximum error `0.0512841`, relative L2 `0.00585515`. It is preserved as a compiler/numerical limitation, outside the selected one-block scope. No tolerance was relaxed. No root cause has been established; near-one cosine alone does not make that run pass.

Header-only resource preflight precedes selected tensor loading. Runs enforce one CPU, no swap, a 2–3-GiB cgroup memory cap, a 180-second whole-job deadline and 60-second subprocess limits, with one model job at a time. Recorded cgroup peaks include charged file cache, not just process RSS. Exported tensor totals exclude executable, verifier fixtures/guards and the reserved 16-KiB stack; final ELF load reservations are recorded separately. Full models, longer contexts, complete policy recurrence and measured tuning remain future work.

The broader recorded full-model state remains:

| Workload | Recorded development evidence | Remaining qualification |
| --- | --- | --- |
| ResNet50 v1.5 / IMAGENET1K_V2 | Fresh V6 baseline/optimized host checks pass all development20 and frozen final100 images: top-one 16/20, matching FP32, and 87/100 versus FP32 88/100. Both complete development20 streams pass matched Spike twice, with preservation, negative controls and linked-memory accounting. | Canonical ImageNet accuracy, actual FireSim deployment and timing; final100 has host evidence only. |
| TinyLlama-1.1B-Chat-v1.0 | Historical opt-in v3/source-rotary/source-SiLU checks pass complete 128+32 host sessions in both modes, two repeats/reset: logits, full KV caches and positions are raw-exact. Native recurrence and unchanged development language gates pass. A fresh complete WSL candidate run remains pending. | Candidate adoption, complete static packaging, functional device checks, deployed memory fit and paper performance. |
| SmolVLA / SmolVLM2-500M | Historical recorded TRAIN observation passes retained prefix, ten denoising/Euler updates and actions in both modes, two fresh VMs and two resets each. Flow states/timesteps/actions are native-exact; three BF16 key-cache words differ within the unchanged tolerance. Fresh local eager execution and bounded SDE compiler checks are separate evidence. | Combined public export composition, original-policy quality/units, device, deployed memory and timing. |

These results qualify the recorded host/functional workloads only. Keep TRAIN/calibration, development and held-out inputs separate, freeze recipes before final evaluation and preserve numerical gates. Workload order is ResNet50, TinyLlama, then SmolVLA. Final timing requires the selected FireSim deployment; host and functional Spike supply no timing labels. Store weights, captures, binaries and immutable receipts under the configured `out/` roots, never alongside these sources.

## Local Verilator block verification — 2026-10-19

[verify_blocks_rtl.py](verify_blocks_rtl.py) prepares the retained block operator objects for actual `GemminiRocketConfig` execution. Gemmini/CPU placement is decided during TVM compilation: the selected learned integer products become Gemmini TensorIntrin functions; surrounding supported work becomes RV64 CPU functions. The guest has a fixed call sequence. Runtime pointer/shape admission can reject a call; it does not dynamically repartition the graph or move it to another backend.

The [RTL report](../../../../out/artifacts/verification/gemmini/tvm/rtl-blocks-20261020T050040Z/final-report.md), [hardware binding](../../../../out/artifacts/verification/gemmini/tvm/rtl-blocks-20261020T050040Z/hardware-binding.json) and [dispatch audit](../../../../out/artifacts/verification/gemmini/tvm/rtl-blocks-20261020T050040Z/dispatch-audit.json) distinguish compile-time placement from operations actually reached before a limit. The complete graph schedules remain TinyLlama 7 Gemmini/26 CPU calls, isolated action 1/6, and SmolVLA block/action 11/58 per inference. These are graph-function calls, not ISA instruction counts. The two attention QK/PV products in TinyLlama and the SmolVLA block remain CPU FP32 work.

The existing simulator was built from the preferred FireSim lineage: Chipyard `009e85b0`, Gemmini `63f0b68a`, Rocket `885dd596`, with the exact generated parameter-header digest `3758ae96…`. The FireSim wrapper selects this same base `chipyard.GemminiRocketConfig`. This establishes the local selected generator/configuration binding; it does not make the Verilator harness identical to the registered FireSim image. Local Verilator uses 256-MiB magic memory; the FireSim wrapper adds bridges/design/clock changes and declares 16 GiB. Final image/driver/DRAM/completion binding still requires actual FireSim execution.

All three original-shape block guests were prepared and run under a 2-GiB/no-swap/one-CPU cap, no waves, 175-second wall deadline and 850000-cycle testbench limit. Each reaches the cycle limit before an inference completes. The action guest remains in the first CPU activation-range reduction; TinyLlama completes its first CPU call and reaches RMSNorm; the SmolVLA block completes two CPU calls and reaches activation encoding. None reaches its first Gemmini call within this budget. There is no complete original-shape numerical RTL verdict or RTL cosine. The same instrumented guests pass fresh Spike positive and wrong-oracle controls. Partial execution is not a numerical pass, and these local limits are not evidence of an unsupported operation.

The trusted RTL verifier uses one original saved fixture and one inference. It retains the exact TVM operator object, weights/constants, numerical tolerance and INT32 oracle; guards around outputs and workspace, input rejection, stack canary and primitive counters remain. It addresses preloaded input fixtures directly and requires the supported ELF loader to initialize all PT_LOAD BSS. Guest full-weight hashing, input copying/preservation scans, workspace and output poisoning and alternate-output retention/repeats are outside this bounded scope. Trace hooks surround each existing graph call; actual output bytes use length-delimited HTIF records. Instrumentation and HTIF contribute simulation overhead, so these wall times and partial cycle counts are not performance measurements.

An explicit supplementary isolated-action diagnostic admits `verify_blocks.py --workload smolvla-action --action-chunk-size 1`; the default remains 50 and other workloads refuse this override. It retains the original full 720→32 learned projection and quantize/dequantize/bias/Euler/slice operations, with one action position. Its separate receipts do not replace or qualify the requested chunk50 workload. The detailed RTL report records its final result and any numerical comparisons.

The latest trusted verifier batches each trace/count summary into one HTIF write. Its stack watermark skips intact aligned words, then locates the exact first changed byte; a compiled native C regression checks all pointer alignments and changed-byte positions across a range of lengths. These changes reduce diagnostic cost without changing the graph, operator object, primitive counters, output checks or stack acceptance. Earlier probe binaries/logs remain preserved. Interrupted output records retain completed earlier tensors and simulator-failure evidence; a truncated stream can never qualify a passing run.

To prepare or execute a retained passing block capture, supply explicit fresh artifact directories and the previously qualified hardware receipt. Both commands require an externally enforced memory/no-swap/one-CPU cgroup; `run` also requires `+loadmem`, which the driver supplies. Refer to the recorded session plan for exact bounded service commands.

```bash
python "$setup_dir/verify_blocks_rtl.py" prepare --input-dir "$passing_block_capture" \
  --output-dir "$fresh_rtl_bundle" --hardware-receipt "$qualified_hardware_receipt"
python "$setup_dir/verify_blocks_rtl.py" run --bundle "$fresh_rtl_bundle" \
  --name rtl-positive --seconds 175 --cycles 850000
```

The preparation stage checks final ISA/runtime closure, unresolved callbacks and prohibited hardware-loop instructions. The execution stage requires actual guest PASS, simulator `$finish`, zero host status and complete independently checked raw outputs for success; a service wrapper returning zero alone establishes none of those. A wrong-oracle control additionally requires guest22 in the simulator's decoded assertion and the expected numerical-failure marker.

## Bring-up milestones

| Milestone | Current evidence | Remaining work |
| --- | --- | --- |
| 1. Full-model PyTorch intake into supported TVM IR | Recorded weighted FP32 ResNet and original-precision SmolVLA stage imports pass. Fresh bounded parameterized imports additionally pass saved ResNet forward, full TinyLlama FP32 prefill and all three SmolVLA stages through the normal ONNX frontend, preserving nodes and parameter dtypes/shapes. | Full TinyLlama weighted prefill/decode intake remains incomplete; original BF16 cache consistency fails before export and decode remains unverified. The parameterized diagnostic exposes external floating tensors as additional inputs and does not qualify weight binding, compiled numerics or quality. Direct integer ResNet IR does not establish automatic PyTorch INT8 import. |
| 2. TVM to a functional device backend | Complete integer ResNet: 53 convolutions plus classifier, 20 images, two repeats, both graph modes, 80 successful Spike invocations. A small generated recurrent integer graph also passes actual CLI/export/Spike checks. | This milestone is complete for the demonstrated workloads. The pinned Gemmini Spike extension checks functional behavior, not the Rocket microarchitecture, DMA/cache timing or RTL equivalence. |
| 3. Bind the TVM backend to the selected Rocket/Gemmini hardware | Pinned Chipyard `009e85b0`, Gemmini `63f0b68a` and Rocket build successfully. Bounded Verilator runs pass CPU startup/clean exit, the guest-12 failure control, and ten representative primitive calls over five shapes with a guest-22 wrong-oracle control. The adapter was rebuilt against the exact pinned software/generated headers; linked ELFs were audited without hardware loops. Its complete 13-call/max-K=131071 primitive suite now also passes fresh functional Spike checks. | The original large-buffer/max-K stress suite remains unqualified on RTL after its rejection harness reached a cycle limit. The functional extension does not establish a matched RTL revision or timing; local magic-memory RTL qualification does not bind FireSim's image, driver or timing model. |
| 4. Run a small TVM model on that RTL configuration | TVM-generated INT8 matmul+bias+ReLU, M=17/N=19/K=33, passes all six calls on actual Rocket/Gemmini Verilator, including outputs, primitive counts, preservation, guards, reuse and rejection/recovery. Its wrong-oracle control reports guest 22 through TSI. | The trusted verifier precomputes the same exact integer reference dot products on the host; all six inputs/checks and the production graph/operators remain intact. This supplies small-model functional RTL evidence, not full-model RTL qualification or timing. |

Original-model accuracy/adoption and FireSim performance are later qualification gates. Missing canonical ImageNet data or the original native Flash libm must not be used to block an independent small-model RTL smoke test. The latter library still blocks reproduction of the selected source-exact SmolVLA Flash profile.

## Bounded local RTL follow-up — 2026-10-19

The [local execution report](../../../../out/artifacts/verification/gemmini/tvm/local-rtl-qualification-20261019T095639Z/local-execution-report.json) supersedes earlier statements that startup and tiny-model RTL execution are wholly unverified. Every simulation uses an aggregate 2-GiB cgroup cap, zero swap, one CPU, one simulator at a time, a wall deadline and a testbench cycle limit, without waveform tracing. CPU startup completes in 20.4 seconds, the ten-call primitive smoke in 41.2 seconds and the complete six-call tiny TVM test in 155.6 seconds. Their negative controls require exact decoded guest codes and application markers where present; a service wrapper's success only means its receipt was saved.

The [hardware/runtime qualification](../../../../out/artifacts/verification/gemmini/tvm/local-rtl-qualification-20261019T095639Z/runtime-qualification.json), [exact-header adapter build](../../../../out/artifacts/verification/gemmini/tvm/local-rtl-qualification-20261019T095639Z/adapter-binding.json), [guest bindings](../../../../out/artifacts/verification/gemmini/tvm/local-rtl-qualification-20261019T095639Z/guest-binding.json) and [throughput/resource assessment](../../../../out/artifacts/verification/gemmini/tvm/local-rtl-qualification-20261019T095639Z/resource-estimates-and-throughput.json) record the individual scopes and limits. No TVM core, adapter source, accelerator RTL or Scala configuration was modified in this follow-up. The primitive smoke is supplementary to the retained full stress harness. Moving the tiny verifier's reference calculations to a trusted host table does not change the TVM graph, generated operator object, model constants, six inputs or numerical gate; fresh Spike controls also check the revised verifier.

ResNet uses the complete existing optimized graph and learned weights with one saved image. A separate bounded preload probe avoids large guest-side weight hashing, workspace poisoning and input copying; its startup relies on the supported `+loadmem` ELF loader to zero PT_LOAD BSS. The full graph starts, then reaches its 250,001-cycle limit after 50.0 seconds at about 5,000 testbench cycles/second and 60.7 MiB peak cgroup memory, without OOM. Inference does not finish: there is no full ResNet numerical RTL verdict. Those verifier/runtime adaptations are disclosed in the execution report and do not establish the original full preservation/repeat stress scope. A completed one-image numerical RTL verdict is required before claiming full ResNet RTL correctness; starting the graph or reaching a cycle limit is insufficient. None of these host wall times is a paper performance result or a measured tuning label.

A [fresh software functional follow-up](../../../../out/artifacts/verification/gemmini/tvm/bound-software-spike-20261019T180545Z/software-functional-report.json) runs the exact pinned-header software through the existing provisional Gemmini Spike extension. The original 18-case rejection test, complete 13-call primitive suite including K=131071, and primitive wrong-oracle control pass. The same full one-image ResNet preload verifier used by the unfinished RTL probe passes both output tensors, guards and required primitive counts in 24.4 seconds; a fresh wrong-oracle build reports guest/Spike 22 with the expected numerical-failure marker in 24.8 seconds. Each run has a 512-MiB/no-swap/one-CPU cap and a 55-second deadline; peak batch memory is 263.0 MiB, with no OOM. The only negative-build change to compilation inputs is the verifier's BAD_ORACLE define. This adds functional evidence for the rebound software and adapted verifier, without rerunning RTL or expanding its declared stress/quality scope. The existing Spike plugin is not newly qualified against the pinned physical RTL revision and supplies no Rocket microarchitectural or timing verdict. The [follow-up ledger](../../../../out/artifacts/verification/gemmini/tvm/bound-software-spike-20261019T180545Z/change-ledger.json) records commands, identities and generated controls.

The [fresh three-model import audit](../../../../out/artifacts/verification/gemmini/tvm/local-import-followup-20261019T102023Z/imports-consolidated-current.json) adds actual bounded parameterized imports to the earlier weighted/source evidence. Five recorded stages pass: ResNet forward (122 nodes), TinyLlama FP32 prefill (1,424), SmolVLA prefix (2,217), denoiser (1,302) and action (one). External floating initializers become ordinary typed ONNX inputs and unbound Relax parameters. All original nodes, original inputs/outputs and opsets are preserved byte-for-byte; parameter dtypes/shapes, Relax well-formedness and JSON structural roundtrip pass. This changes the import signature and is a separate diagnostic, not a claim that original weighted intake completed. No floating-weight payload or PyTorch model is loaded; 63,925 bytes of nonfloating auxiliary data are loaded exactly. The validated batch takes 14.6 seconds and peaks at 628.5 MiB under a 1-GiB/no-swap/one-CPU/60-second cap. These runs add no TVM core patch or monkeypatch and perform no backend compilation; previously disclosed pinned-fork deviations remain. The [follow-up change ledger](../../../../out/artifacts/verification/gemmini/tvm/local-import-followup-20261019T102023Z/change-ledger.json) records the derived graphs, parameter bindings, IR and serialized modules.

The [earlier import reconciliation](../../../../out/artifacts/verification/gemmini/tvm/local-rtl-qualification-20261019T095639Z/imports-consolidated-current.json) remains preserved. TinyLlama's native BF16 numerical gate failure, original weighted FP32 intake and decode coverage remain separate from the new prefill diagnostic; no compiled numerical comparison follows from either report. The [current environment assessment](../../../../out/artifacts/verification/gemmini/tvm/local-rtl-qualification-20261019T095639Z/environment-closure-current.json) rechecks the actual libm hash: the original manifest/library closure remains unavailable. The [RTL change ledger](../../../../out/artifacts/verification/gemmini/tvm/local-rtl-qualification-20261019T095639Z/change-ledger.json) lists the earlier source edits, probes, rebuilt artifacts and corrected attempts; prior evidence is preserved.

## Earlier local closeout and deferred execution — 2026-10-19

The earlier [local final report](../../../../out/artifacts/verification/gemmini/tvm/remaining-local-20261019T071526Z/local-final-report.json) records the closeout before bounded RTL execution resumed. Work remains local; no remote server or SSH destination is required. The earlier [consolidated import report](../../../../out/artifacts/verification/gemmini/tvm/remaining-local-20261019T071526Z/imports-consolidated.json) summarizes its saved local runs without rerunning models. Older remote-deferral wording and unverified-runtime statements are historical; the bounded local execution report above records current qualification. The earlier [corrected execution handoff](../../../../out/artifacts/verification/gemmini/tvm/remaining-local-20261019T071526Z/remote-handoff-v2.json) retains command and portability records for future use, including the correction that guest failure codes are not simulator process return codes. Existing per-run reports remain unchanged, including unfinished reports whose last saved status was `running`.

The [local working-tree patch](../../../../out/artifacts/verification/gemmini/tvm/remaining-local-20261019T071526Z/local-working-tree.patch) and [source manifest](../../../../out/artifacts/verification/gemmini/tvm/remaining-local-20261019T071526Z/local-source-manifest.json) preserve the seven uncommitted comparison/documentation/test files, including the untracked import-only driver. Reconstruct Merlin at `a4745c66aaead245a58979f0d1d85cf244aee117`, apply that patch and verify the resulting file digests; cloning the commit alone omits those changes. Forward patch applicability was checked against the commit using an isolated temporary Git index, without staging in this checkout. The independent protocol checkout is clean at model2MLIR `7915e23475c6db446a3c404847b11e8bc72c8a27`; the TVM fork remains clean at its declared pin. This source snapshot does not contain checkpoints, external ONNX tensor payloads, Python installations or a complete toolchain/library closure. Earlier patch/manifest versions remain preserved as historical evidence.

| Full model / precision | Fresh structural evidence | Unsupported operations | Numerical evidence |
| --- | --- | --- | --- |
| ResNet50, original FP32 | One complete forward graph imports: 122 ONNX nodes, seven operator types, 53 convolutions and the classifier. Source parameters are preserved. | None encountered in the recorded graph on the pinned fork. Unmodified Apache TVM was not tested by this run. | No compiled-output comparison in this import-only run. Its seeded input is for structural coverage, not accuracy. |
| SmolVLA, original LeRobot precision, no W8A8 | Prefix (2,217 nodes), denoiser (1,302), action slice (one); 33 operator types across the three graphs. Native state supplies prefix, ten denoising steps and action decoding, 12 calls total. Source parameters are preserved. | None encountered in the recorded stages on the pinned fork. All nine named source-arithmetic policies are `native`; no fused BF16 export profile is selected. | No compiled-output comparison, repeats/reset qualification, physical-action-unit validation or policy-quality qualification follows from these imports. |
| TinyLlama, original BF16 | Factory's native full-sequence/cache consistency check fails before ONNX export. The intended workload remains the full 22 layers, 128 prefill tokens and 32 decode tokens. | Not evaluated: this is a native numerical failure before TVM intake. | At 130 tokens, seven of 32,000 logits fail the existing `rtol=atol=0.05` check; maximum absolute difference is 0.076171875. No tolerance was relaxed. |
| TinyLlama, explicitly converted FP32 diagnostic | Complete prefill ONNX is saved: 1,424 nodes, 19 operator types, 4.40-GB external tensor data. The original weighted import was interrupted. A fresh parameterized prefill import passes with 245 external floating tensors exposed as additional inputs; decode export/import is unverified. | None encountered in the new parameterized prefill import. This does not prove the original weighted import or unexecuted decode; earlier registry-name screening alone remains weaker evidence. | No compiled-output comparison. This diagnostic does not qualify the original BF16 recipe or candidate quality. |

These native-profile graphs still use the comparison's exporter helpers. SmolVLA translates nine `aten.scalar_tensor` sites into standard typed ONNX constants (five prefix, four denoiser); its `_int_mm` translation has zero sites. The saved graphs contain standard `ai.onnx` operators, not the named `merlin.source_preserving` operators. This is narrower evidence than stock-exporter or stock-Apache compatibility. A metadata audit of the saved SmolVLA inventory finds integer index types at all 40 Gather nodes and matching data/update types at all 110 ScatterND nodes, so it finds no need for the fork's invalid-input dtype coercions in these recorded graphs. Other frontend fixes remain in the selected fork.

The full TinyLlama FP32 import ran without an aggregate memory cap while RTL work was active. WSL was interrupted; the previous-boot journal contains a global OOM kill of a Python process with about 13 GiB resident anonymous memory. Its PID is not conclusively linked to the importer. Repeated journal entries replay the same kernel timestamp and do not establish repeated OOM events. The [incident record](../../../../out/artifacts/verification/gemmini/tvm/remaining-local-20261019T071526Z/wsl-incident.json) preserves these limits. The interrupted heavy jobs were not automatically restarted; subsequent bounded RTL and parameterized-import runs are documented above. Before the user deferred compute, a separate 64-MiB cgroup probe verified that an oversized allocation is killed inside the workload's enforced limit; this is host resource-control evidence only. The full floating-weight import remains deferred.

Current work stays local. Heavy imports, builds, compiled regressions and simulations are marked **unable to complete in this pass**, as the user requested, and are left for a later time. This does not establish that every small RTL test is inherently impossible in WSL. Before later heavy execution, use one workload at a time with an enforced aggregate memory/swap limit and a deadline covering its children; JVM `-Xmx` and build `-j` settings alone do not enforce that limit. Required peak memory is unmeasured, and the interrupted uncapped run does not establish a minimum. The commands below are reproduction recipes, not runs performed during this report-only closeout.

The [SmolVLA closure assessment](../../../../out/artifacts/verification/gemmini/tvm/remaining-local-20261019T071526Z/environment-closure-assessment.json) verifies all 26 selected authenticated artifacts. It cannot establish the complete historical installation: the original install manifest is unavailable, the originating OS/library package is unidentified, and the Flash profile requires libm SHA256 `f0734a09a0602ccde6b98ea7fbb3c9cfc62b3a4cf4a7b17ccd36d774ca22b8df` while local Ubuntu `libc6 2.39-0ubuntu8.9` supplies `fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0`. The latest bounded search inspected 27 additional official package payloads and found no exact match; this does not prove the artifact is unavailable everywhere. A hash alone does not identify its package or recover its contents. Current Transformers 4.57.1, NumPy 1.26.4, Hugging Face Hub 0.36.0 and packaging 25.0 overrides remain disclosed alongside dependency-metadata conflicts. Using a different library or disabling admission would change the selected strict-source contract, not recover it. Native structural imports and the independent RTL smoke test do not require that missing historical Flash profile.

The earlier seven deliverables fit the four milestones as follows. Original environment/source-profile recovery supports milestone 1. The TVM/LLVM build supports all four and already exists. Fresh compiled SmolVLA qualification follows structural milestone 1 and additionally needs real compiled state chaining, repeats and resets; native import success does not complete it. Packaging and memory accounting support milestones 2–4; full ResNet functional accounting exists, full TinyLlama/SmolVLA accounting does not. Complete matched Spike workloads extend milestone 2; full TinyLlama/SmolVLA device sessions remain pending. Model quality/adoption is a separate scientific gate. FireSim binding begins in milestone 3, while actual FPGA benchmarking and measured tuning extend beyond the small-model RTL demonstration in milestone 4. Those remaining compute tasks are deferred, not relabeled complete.

## TVM implementation and comparison scope

This is a TVM-based Gemmini backend, pinned to an Apache TVM v0.19.0 fork with 61 additional commits, not an unmodified Apache release. The fork changes 31 files, including the ONNX frontend, Relax backend passes, TensorIR scheduling, a static exporter, shared arithmetic/dtype lowering and tests. Model recipes and numerical acceptance remain explicit comparison choices rather than Apache TVM requirements.

The stateful verifier exports PyTorch stage signatures through ONNX into Relax. The integer ResNet path instead constructs semantic Relax directly from its TRAIN-calibrated recipe; its separate FP32 path uses ONNX. Host verification uses LLVM and the Relax VM. Device preparation decomposes supported integer convolutions/linear operations into matmul plus CPU work, substitutes genuine TensorIntrins and compiles primitives through LLVM. The guest uses custom static C orchestration and the pinned Gemmini C adapter, rather than a guest Relax VM or the stock BYOC registration path.

The current schedule family has nine I/J macro-tile choices and fixed reduction chunks of at most 16. Accelerator calls remain opaque to fusion; surrounding CPU work can be fused, while packing, requantization and state handling remain part of the workload. Search uses TVM's PerStoreFeature and XGBModel in a custom finite-budget controller. Public stateful export accepts explicit per-shape choices and records actual prepared schedules, but no qualified FireSim timing labels exist. Selection alone does not establish a measured-autotuned deployment.

Named source-arithmetic profiles bind selected shapes, library artifacts and CPU dispatch; they are bounded custom extensions, not general replacements for standard ONNX operators. Fresh native WSL attempts at the complete selected SmolVLA profile set refuse this host's AVX2 dispatch, as intended. Authorized Intel SDE 10.13.1 actual AVX512 instruction emulation admits eight of nine source contracts and passes the existing language-scaling, rotary-table and FP32-SiLU public export/compiler numerical tests without skips. Flash still refuses the missing original host libm; these bounded checks do not qualify the complete combined-profile model or target timing. Eager native-profile diagnostics and historical host passes do not establish fresh combined-profile compilation, original-model quality, full-model device execution or deployed memory fit. Compare the same accepted numerical recipe and full execution boundary across systems, and report each of these qualifications separately.

### Extension mechanisms and deviations relevant to the paper

The framework supports custom passes and tensorization; the [v0.19 TensorIntrin API](https://github.com/apache/tvm/blob/v0.19.0/python/tvm/tir/function.py) represents a mathematical description and its implementation. This backend registers those objects and tensorizes semantic integer reductions into Gemmini C-ABI calls. CPU loops, packing, activation and requantization still compile through TVM/LLVM. This is a substantive TVM backend, although its integration and deployment are project-specific. It does not implement a native Gemmini TargetKind, the stock BYOC partition/codegen path, or a guest Relax VM/microTVM runtime.

| Area | Actual implementation / deviation | Consequence for a fair comparison |
| --- | --- | --- |
| Compiler identity | Apache v0.19.0 `c4dc0c29…` plus 61 fork commits, pinned at `6b376cc8…`; 31 changed files, 10,718 added and 153 removed lines in the retained source audit. | Name the fork, full source pin and actual built libraries. Results cannot be attributed to an unmodified Apache release. |
| Model intake | Stateful stages and fresh FP32 ResNet use PyTorch → ONNX → Relax. The calibrated integer ResNet path constructs semantic Relax directly from the explicit recipe. | Direct IR construction is supported by TVM, but it does not demonstrate automatic PyTorch quantization/import. Report model preparation as part of the methodology. |
| Shared ONNX frontend | `python/tvm/relax/frontend/onnx/onnx_frontend.py` contains generic fixes and bounded source-arithmetic extensions, beyond accelerator dispatch. Inherited Gather float-index casting and ScatterND update-type casting accept inputs outside the corresponding standard type contracts. | Separate standard operator support, invalid-input permissiveness and named custom domains. Do not infer ONNX conformance from successful import. The fresh SmolVLA typed inventory finds no dependency on those two coercions. |
| Numerical source profiles | Nine opt-in families cover mean, softmax, LayerNorm, convolution, SGEMM, scalar scaling, rotary tables, Flash and SiLU, with source/library/dispatch/shape guards. Export also has an explicit fused BF16 candidate. | These profiles and FP32/W8A8 conversions are declared arithmetic choices. Require the same accepted recipe in each paper arm and retain original-model quality checks separately. Strict-source guards are not evidence of missing generic IR operators. |
| Shared arithmetic core | `src/arith/rewrite_simplify.cc` removes the unconditional float-constant division-to-reciprocal-multiplication rewrite. | This is a core compiler change affecting rounding, not Gemmini-only glue. Preserve its source diff and numerical regressions; disclose it even if introduced to improve source fidelity. |
| Shared dtype legalization | `src/tir/transforms/unsupported_dtype_legalize.cc` preserves explicit BF16/FP8 narrowing and logical storage rounding while internally promoting storage/computation. | This changes shared low-level numerical behavior. The patch and its BF16/FP8 tests belong in the compiler identity and reproduction record. |
| Restricted host build | `CMakeLists.txt` enables the optional `apps/gemmini/auto_copy_guard.cc` replacement for the unavailable automatic-copy implementation. Unannotated IR is preserved; automatic-copy markers are rejected. | It is a restricted build subset, not a replacement implementation of the full optimization. Record the flag; never silently accept those annotations. |
| Scheduling / verification | Nine I/J macro-tile choices, reduction chunks ≤16, synchronous load/compute/store C calls. `tir.Schedule(..., debug_mask=1)` retains sref verification and tensorization matching while disabling the optional cached region-cover verification. | Report the bounded search space and reduced optional checking. No asynchronous DMA overlap or unrestricted schedule-search claim is supported. |
| Runtime / packaging | Custom static C graph/session orchestration calls TVM-generated RV64 primitives and the pinned Gemmini C adapter. Buffers, constants, state routes, startup and linker placement are explicit. | End-to-end comparisons must include required host work and memory. Import-only or per-operator success does not qualify the custom orchestrator, complete workload or platform fit. |
| Tuning | A custom finite controller uses TVM PerStoreFeature and XGBModel; qualified FireSim timing labels are absent. | Describe a bounded tuner implementation. Do not call a selected tile choice a measured-autotuned winner or use Spike wall time as accelerator performance. |
| Instruction policy | TVM forbids hardware-loop/FSM instructions and selects OS dataflow. Physical default Gemmini supports both dataflows. The generic Merlin experiment exempts its Phase 2 vendor-reference arm from one loop restriction. | Record the actual policy and linked instructions in every arm before claiming equal restrictions; an exemption does not prove instructions were used. |
| Hardware / measurement | Matching default array configuration and parameter header do not bind the entire SoC, FireSim wrapper, DRAM, driver or image. The fresh Verilator build is a 256-MiB board, while the preferred FireSim DTS declares 16 GiB. | Bind actual run artifacts to the same selected image, runtime and measurement boundary. Functional Spike, Verilator execution and FPGA timing are separate qualifications. |

The retained [file audit](../../../../out/artifacts/verification/gemmini/tvm/four-step-audit-20261019T053243Z/file-audit.json) and [Apache-to-fork patch](../../../../out/artifacts/verification/gemmini/tvm/four-step-audit-20261019T053243Z/tvm-apache-v0.19.patch) enumerate the complete 31-file fork delta. No TVM source was changed during the latest local import/RTL bring-up. Extension through supported APIs is a framework fit; preservation of model semantics, disclosed compiler patches, matched tuning budgets and matched measured hardware are independent evidence needed for the paper. This audit is not an Apache certification of the backend or benchmark fairness.

### Software and hardware meet at the guest workload

The implemented software path is pinned model/preprocessing → PyTorch stage export → ONNX (standard or explicitly selected custom profiles) → Relax → graph and integer convolution/linear decomposition → semantic TensorIR → Gemmini TensorIntrin scheduling → LLVM RV64 objects → static C graph/session glue, Gemmini adapter, weights/constants, startup and linker → guest ELF and provenance/fixture package. Integer ResNet enters through the separately declared direct-Relax recipe. The fresh import-only checks stop at Relax; they perform neither this later compilation nor hardware execution.

The hardware path is separately pinned Chipyard/Gemmini Scala → Chisel elaboration → FIRRTL → CIRCT/firtool → RTL. Verilator compiles that RTL into the local simulator executable. A FireSim build applies its own simulation transformations and platform build flow, then uses Vivado to produce an FPGA image and matching driver/platform artifacts. TVM generates the guest software; it does not generate or modify the FPGA bitstream. A final deployment joins the guest workload with the independently selected hardware image, memory map, driver and completion interface.

The current source build restores Chipyard `009e85b05ca817a15fcd6c3fe4eedd86cc2a3617`, Gemmini `63f0b68a68f185a016a2993f2fb32e1661ea9e6c`, Rocket `885dd5966a49ce19512e3cc3275515926e729ba2` and CIRCT firtool 1.75.0. The actual `simulator-chipyard.harness-GemminiRocketConfig` build exits zero. Generated header SHA256 is `3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924`; generated DTS SHA256 is `a45460168248bd08e47a237f78eb1256d589b5410c046f0031e648207a83dfc3`. The configuration entry is `generators/gemmini/chipyard/GemminiConfigs.scala`; array defaults are in `generators/gemmini/src/main/scala/gemmini/Configs.scala` and `GemminiConfigs.scala` within that isolated Chipyard checkout.

There are two build adaptations to disclose: SBT resolves its legacy plugin through the official Ivy download endpoint, and its source list excludes `HeteroConfigs.scala` and `MerlinSaturnConfigs.scala` because the pinned optional Saturn repository is unavailable. The selected Gemmini configuration does not reference those classes. No Scala/RTL source file was edited. The [exclusion receipt](../../../../out/artifacts/verification/gemmini/tvm/remaining-local-20261019T071526Z/rtl-build-exclusions.json) records the exact setting, excluded file digests and reference search. FireSim sources were restored for MIDAS annotations; this did not build an FPGA image.

The deferred execution order remains CPU startup and its deliberate bad-model control, Gemmini primitive correctness/rejection/recovery controls, then the TVM tiny graph and wrong-oracle control. These runs are unable to complete in this pass. Use `+permissive +max-cycles=200000000 +permissive-off` around simulator arguments; the first local attempt omitted the permissive wrapper and FESVR treated the plusarg as an ELF path. That infrastructure failure is not a numerical negative-control pass.

The [static runtime binding](../../../../out/artifacts/verification/gemmini/tvm/remaining-local-20261019T071526Z/runtime-static-binding.json) rechecks all seven small guest ELF hashes, executable entry points, load segments, BSS/stack bounds, HTIF symbols, lack of unresolved symbols and absence of a dynamic interpreter. Every image fits the actual generated enabled DRAM range `[0x80000000, 0x90000000)`. CPU startup reserves 19,088 bytes, the largest primitive negative-control image 293,376 bytes, and the tiny TVM graph 32,528 bytes, each including its 16-KiB guest stack. These are static image reservations, not measured RTL stack use or full-model memory qualification. The generated machine has one Rocket, 32-KiB instruction/data caches and a 512-KiB inclusive L2.

Completion has three distinct statuses: guest return code, direct simulator process termination, and an outer service/scheduler/SSH wrapper result. The actual build uses `verilator --main` with `TestDriver.v`, not the older Rocket `emulator.cc` main. TSI encodes a completed guest as `(code << 1) | 1`. Guest zero drives the harness success signal and `$finish`; require the application PASS marker, TestDriver finish and host success. A nonzero guest triggers the SimTSI assertion, which prints the decoded guest code before generated C++ invokes `VL_STOP_MT`. Actual bounded local controls observe direct host status **255** for guest codes `12` and `22`. This supersedes the earlier SIGABRT/status-134 prediction: Verilator calls `abort`, but FESVR installs a SIGABRT handler that calls `exit(-1)`. The [runtime qualification](../../../../out/artifacts/verification/gemmini/tvm/local-rtl-qualification-20261019T095639Z/runtime-qualification.json) binds the observed control logs and the supporting local FESVR source; it does not authenticate the entire source-to-installed-library build closure. For negative controls require decoded guest `12` or `22` in that specific TSI assertion and the application failure marker where present, plus attributable simulator assertion termination. Record outer wrapper results separately. An arbitrary host failure, trap, missing-file error or timeout is insufficient. The CPU bad-model guest has no dedicated failure marker, so its exact TSI guest code must follow a successful positive startup control with the same runtime.

The x86-64 simulator is tied to the local toolchain: its embedded RPATH points into the Chipyard Conda environment. It needs `libriscv.so` and host libraries with symbol versions including `GLIBC_2.34` and `GLIBCXX_3.4.32`. If the executable is moved later, its toolchain/library closure must be rebound or rebuilt and its new identity recorded. The complete resolved transitive library contents are not yet authenticated. This portability observation is not a remote-server prerequisite for the current local work, and loader compatibility is separate from the exact historical SmolVLA source-profile libm hash.

The generated SimDRAM instance declares 256 MiB and a 500-MHz clock parameter. Its C++ implementation selects `mm_magic_t` by default; the current smoke commands omit `+dramsim`, so linked DRAMSim2 support is not evidence that its timing model is enabled. Enabling `+dramsim` requires an explicit INI directory and identities for the selected memory/system configuration, followed by fresh qualification. Neither the clock parameter nor magic-memory operation establishes FireSim DRAM timing or measured target frequency. Only after actual execution can RTL startup, completion, primitives and tiny-model correctness be qualified; FireSim still needs its separate image/driver/DRAM/measurement binding.

## Target and numerical contract

[target.yaml](target.yaml) is the source/arithmetic/runtime declaration. Its provisional integer sources are Gemmini `6ad65b90`, gemmini-rocc-tests `7c540b3a` and libgemmini `ea8f7ed7`; full commits/digests live in YAML. The parameter header SHA256 is `3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924`: 16×16 geometry, 256-KiB scratchpad and 64-KiB accumulator. Adapter verification stages exact Git blobs and checks resolved includes; the live floating-point header is incompatible. TVM's binding to a deployed hardware revision/generated configuration, usable guest DRAM and completion/coherence remains unconfirmed.

The shared [hardware pin registry](../../../../merlin/contract/hardware_pins.yaml) records a preferred full `FireSimGemminiRocketConfig` U250 design: queue configuration `alveo_u250_firesim_gemmini_rocket_30mhz`, a 30-MHz build, bitstream archive SHA256 `0af6f26e1cbad1c03f9e5873217505a5100bf2d0d068205cfa486522b24fc603` (`firesim.tar.gz`). That build label alone does not establish the simulated target frequency or DRAM timing model. The [board catalog](../../target/board-catalog.yaml) records one non-vector Rocket hart and a 16-GiB DTS memory node; the [Phase 2 machine registry](../../phase2/whole-model-machines.yaml) selects this image and the matching parameter-header digest. These are recorded platform inputs, not a TVM execution receipt. The generator sources live in the selected external Chipyard/Gemmini trees, not in this recipe directory.

Keep the three source selections distinct: Merlin's Verilator facts use Gemmini `8c3f9923` plus a declared LoadController assertion edit; the preferred FireSim registry records `63f0b68a` in Chipyard `009e85b0`; TVM development uses `6ad65b90`. A fresh public-source comparison verifies that their primary `Configs.scala`, `GemminiConfigs.scala` and `chipyard/GemminiConfigs.scala` files are byte-identical. The first two revisions differ only in README content; the FireSim revision adds optional DSE configurations, changes software submodule pins and disables the stale load assertion. Its LoadController bytes match Merlin's declared edit. This narrows the source compatibility question without proving the whole SoC or an executed image identical. The registry flags incomplete build-time tree capture and an untracked FireSim build recipe, so the recorded source lineage is weaker evidence than its elaborated-design and bitstream digests. It also retains another full image (`a9a190b9…`) and a Lean design without full-width accumulator readout. Matching configuration names or the parameter header does not make these interchangeable. As the [Phase 0 walkthrough](../../phase0/README.md) requires, compare hardware facts across the selected elaborations before a shared-hardware paper claim, then bind both implementations' actual runs to the same image, driver, header, memory map and measurement boundary. TVM's hardware revision/configuration fields remain unset until that binding is established.

The device boundary is dense row-major signed INT8 operands and INT32 results, positive static shapes and K≤131071. Generic linear lowering flattens left leading dimensions against shared rank-two weights; convolution uses explicit CPU packing/restoration. Bias, scales, activation, attention and state remain where each recipe declares them. TVM schedules C-library primitives in output-stationary mode; WS/FSM/hardware-loop paths are forbidden, including final linked ELFs. The separate whole-matmul reference adapter uses one-array reduction tiles to protect provisional 20-bit PE output and INT32 outer accumulation. The library CPU branch is not a full-accumulator oracle; independent bounded int64 references check integer arithmetic.

Instruction restrictions are comparison policy, distinct from physical hardware support. The development default Scala configuration supports both dataflows. Merlin's generic [experiment definition](../../experiment.yaml) prohibits the `loop_descriptor` role for candidates but explicitly exempts its Phase 2 vendor-reference arm; TVM's comparison declaration forbids hardware loops. Record the policy actually applied to every paper arm and inspect its linked ELF before claiming equal restrictions; the generic exemption alone does not prove a reference used hardware loops.

The selected development ABI is baremetal RV64GC/lp64d. YAML paths are relative to Merlin; environment references are declarations, not automatic expansions. The 8-GiB map is hypothetical, with no confirmed capacity. Functional export defaults to an explicitly requested 256-MiB map. Tensor admission includes aligned constants, serial scratch/workspace and simultaneously live routed state/outputs; executable, stack, runtime/platform reservations and retained fixtures need separate budgets. Shared-stage export deduplicates constants and refuses hidden runtime allocators. Hashes bind selected inputs/sources/artifacts and loaded libraries, not complete runtime source closure or target fit.

## Pinned host environment

Use absolute `TVM_ROOT` and `TVM_BUILD` paths. The TVM source pin in `target.yaml` is published in the fork; check out that exact commit rather than the moving branch head. Tested tools are Python 3.10.14, GCC 11.5, CMake 3.26.5 and Ninja 1.13.2. Do not source Chipyard's environment for this build. Override tools with absolute `TVM_HOST_PYTHON`, `TVM_HOST_CMAKE`, `TVM_HOST_CC`, `TVM_HOST_CXX`; `TVM_BUILD_JOBS` defaults to 4.

For a new checkout, apply mandatory case-insensitive name exclusions before materializing files; preserve an existing matching checkout. Reject any supplied path containing either prohibited name before invoking tools. Optional dependencies must not be initialized recursively.

```bash
merlin_root="$PWD"
setup_dir="$merlin_root/examples/gemmini/comparisons/tvm"
export TVM_ROOT=/absolute/path/to/tvm-gemmini TVM_BUILD=/absolute/path/to/tvm-host-build
tvm_commit="$(awk '$1=="source:" {selected=1; next} selected && $1=="commit:" {print $2; exit}' "$setup_dir/target.yaml")"
git clone --filter=blob:none --no-checkout https://github.com/reed-nicolas/tvm.git "$TVM_ROOT"
git -C "$TVM_ROOT" sparse-checkout set --no-cone --stdin <<'PATTERNS'
/*
!*[hH][aA][mM][mM][eE][rR]*
!*[vV][lL][sS][iI]*
PATTERNS
git -C "$TVM_ROOT" checkout --detach "$tvm_commit"
test "$(git -C "$TVM_ROOT" rev-parse HEAD)" = "$tvm_commit" || { echo 'TVM source pin mismatch' >&2; exit 1; }
```

Initialize only `dmlc-core`, `dlpack` and `rang` at their recorded TVM gitlinks. For fresh dependencies, clone without checkout and use the same exclusions:

```bash
git -C "$TVM_ROOT" submodule init -- 3rdparty/dmlc-core 3rdparty/dlpack 3rdparty/rang
while read -r dependency repository; do
  revision="$(git -C "$TVM_ROOT" rev-parse "HEAD:3rdparty/$dependency")"
  git clone --no-checkout "$repository" "$TVM_ROOT/3rdparty/$dependency"
  git -C "$TVM_ROOT/3rdparty/$dependency" sparse-checkout set --no-cone --stdin <<'PATTERNS'
/*
!*[hH][aA][mM][mM][eE][rR]*
!*[vV][lL][sS][iI]*
PATTERNS
  git -C "$TVM_ROOT/3rdparty/$dependency" checkout --detach "$revision"
done <<'DEPENDENCIES'
dmlc-core https://github.com/dmlc/dmlc-core.git
dlpack https://github.com/dmlc/dlpack.git
rang https://github.com/agauniyal/rang.git
DEPENDENCIES
git -C "$TVM_ROOT" submodule status -- 3rdparty/dmlc-core 3rdparty/dlpack 3rdparty/rang
```

Each status line must start with a space; `-`, `+` or `U` requires repair. Prepare an isolated environment and helper-derived output root:

```bash
host_python="${TVM_HOST_PYTHON:-$(command -v python3.10)}"
host_cmake="${TVM_HOST_CMAKE:-$(command -v cmake)}"
host_cc="${TVM_HOST_CC:-$(command -v gcc)}"
host_cxx="${TVM_HOST_CXX:-$(command -v g++)}"
: "${host_python:?Set TVM_HOST_PYTHON}" "${host_cmake:?Set TVM_HOST_CMAKE}"
: "${host_cc:?Set TVM_HOST_CC}" "${host_cxx:?Set TVM_HOST_CXX}"
build_jobs="${TVM_BUILD_JOBS:-4}"
host_tools_path="$(dirname "$host_cmake"):$(dirname "$host_cc"):$(dirname "$host_cxx"):/usr/bin:/bin"
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
unset CMAKE_PREFIX_PATH PYTHONPATH PYTHONHOME CC CXX CFLAGS CXXFLAGS LDFLAGS
export PYTHONDONTWRITEBYTECODE=1
build_root="$(PYTHONPATH="$merlin_root/src" "$host_python" -c 'from merlin.common.paths import build_dir; print(build_dir() / "baselines" / "tvm-gemmini")')"
mkdir -p "$build_root"/{cache/pip,tmp,logs}
export PIP_CACHE_DIR="$build_root/cache/pip" TMPDIR="$build_root/tmp"
"$host_python" -m venv "$build_root/venv"
source "$build_root/venv/bin/activate"
python -m pip install -r "$setup_dir/host-requirements.txt"
host_ninja="$build_root/venv/bin/ninja"
export PATH="$build_root/venv/bin:$host_tools_path"
```

Prefer an existing complete LLVM 18 SDK with X86/RISCV, matching headers and shared libraries; set `llvm_config` and `llvm_lib_dir` to it. If unavailable, this is the tested LLVM 18.1.8 source-build fallback (commit `3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff`):

```bash
llvm_source="$build_root/llvm-project"
llvm_build="$build_root/llvm-build"
git clone --filter=blob:none --no-checkout --depth 1 --branch llvmorg-18.1.8 https://github.com/llvm/llvm-project.git "$llvm_source"
git -C "$llvm_source" sparse-checkout set --no-cone --stdin <<'PATTERNS'
/*
!/*/
/llvm/
/cmake/
/third-party/
!/llvm/test/
!/llvm/unittests/
!*[hH][aA][mM][mM][eE][rR]*
!*[vV][lL][sS][iI]*
PATTERNS
git -C "$llvm_source" checkout --detach llvmorg-18.1.8
"$host_cmake" -S "$llvm_source/llvm" -B "$llvm_build" -G Ninja -DCMAKE_MAKE_PROGRAM="$host_ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$host_cc" -DCMAKE_CXX_COMPILER="$host_cxx" \
  '-DLLVM_TARGETS_TO_BUILD=X86;RISCV' -DLLVM_ENABLE_PROJECTS= -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DLLVM_DYLIB_COMPONENTS=all -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_INCLUDE_TOOLS=ON -DLLVM_BUILD_TOOLS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_ZLIB=OFF
"$host_cmake" --build "$llvm_build" --target LLVM llvm-config --parallel "$build_jobs"
"$host_cmake" -E create_symlink libLLVM.so.18.1 "$llvm_build/lib/libLLVM-18.so"
llvm_config="$llvm_build/bin/llvm-config"
llvm_lib_dir="$llvm_build/lib"
```

Keep source/generated include directories when consuming this build tree. Configure TVM using the checked-in host cache and LLVM RPATH:

```bash
"$llvm_config" --version --targets-built --shared-mode --link-shared
"$host_cmake" -S "$TVM_ROOT" -B "$TVM_BUILD" -G Ninja -C "$setup_dir/host-config.cmake" -DCMAKE_MAKE_PROGRAM="$host_ninja" -DCMAKE_C_COMPILER="$host_cc" -DCMAKE_CXX_COMPILER="$host_cxx" -DUSE_LLVM="$llvm_config --link-shared" -DCMAKE_BUILD_RPATH="$llvm_lib_dir"
"$host_cmake" --build "$TVM_BUILD" --parallel "$build_jobs"
export PYTHONPATH="$TVM_ROOT/python" TVM_LIBRARY_PATH="$TVM_BUILD" TVM_FFI=ctypes
export TEST_DATA_ROOT_PATH="$build_root/cache/tvm-test-data"
python "$TVM_ROOT/apps/gemmini/verify_host.py" --output "$build_root/host-smoke.json"
python -m pip install -r "$setup_dir/frontend-requirements.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TVM_NUM_THREADS=1
```

`host-config.cmake` opts into `USE_HOST_ONLY_AUTO_COPY_GUARD`, OFF upstream by default. This local guard preserves unannotated IR and rejects automatic-copy markers because the restricted checkout lacks `LowerAutoCopy`; it implements no optimization and must not coexist with the full implementation. Record the enabled patch and actual C++ build identity; later Python source changes do not relabel an older library. In a new shell, restore the paths, activate the same venv and reapply these exports. Every numerical run requires a fresh output directory.

## Frontend, graph and search checks

Frontend pins are Torch 2.10.0 (tested `+cu128`, executing on CPU), ONNX 1.17.0, protobuf 5.29.5 and NumPy 1.26.4. Legacy export is explicit `dynamo=False`; ONNX 1.17 retains the importer API. Synthetic opset17/18 fixtures compare Torch, ONNX ReferenceEvaluator and Relax LLVM at `rtol=1e-4, atol=1e-5`; importer integer dtype/shape/values are exact. Neither suite qualifies models/devices.

```bash
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --output-dir "$build_root/frontend/new-run"
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --suite importer --output-dir "$build_root/frontend/importer-new-run"
python -m pip install 'xgboost==1.7.6' 'pytest==8.3.5'
```

Focused learned-search regressions live in TVM's `tests/python/relax/test_backend_contrib_gemmini_tuning.py`; use a test launcher that excludes workspace ancestor imports and unrelated discovery. TVM's `prepare_gemmini_graph` folds constants/fuses CPU work while preserving the device boundary. `BoundedGemminiSearch`/`GemminiModelSearch` use `PerStoreFeature`/`XGBModel`, finite reproducible budgets and candidate-bound repeated measurement requests. Invalid measurements preserve search state; synthetic labels check machinery only. Real timing requires one compiler/adapter/platform/protocol domain and independent semantic/device gates. The [graph simulator recipe](https://github.com/reed-nicolas/tvm/blob/6b376cc8895a83387ec38cc06ef8f7b09827b071/apps/gemmini/README.md#complete-exported-graphs-in-gemmini-spike) supports baseline/optimized modes; its generated graph checks also admit `--search-seed 7` and `--workload conv-residual --conv-kernel 3` (kernels 1/7 are checked). These are bounded fixtures, not performance winners.

## ResNet50 artifacts and frozen evaluation

[verify_resnet.py](verify_resnet.py) defaults to a complete random-weight torchvision ResNet50 v1.5 structural diagnostic: 53 convolutions, one classifier, `[1,3,224,224]`→`[1,1000]`, opset17 and unchanged FP32 `rtol=atol=1e-4`. Supply model2MLIR loader revision `7915e23475c6db446a3c404847b11e8bc72c8a27`; frontend requirements include torchvision 0.25.0/Pillow 12.1.0. `--graph-mode both` checks baseline and explicit `zero`/`default_build` optimization. `--batchnorm-folding` checks a separate evaluation-only folded copy before export.

Supplied mode requires `--checkpoint`, `--inputs`, `--input-source` and `--preprocessing` together. Checkpoints are plain finite CPU state dicts loaded with `weights_only=True`; exact keys/dtypes/shapes are required. NPZ `images` must be finite float32 `[N,3,224,224]` or `[N,1,3,224,224]`, N>0; no conversion/preprocessing occurs. Optional labels JSON has exactly `input_sha256`, 1000 unique ordered `class_ids`, and one `{id, label}` sample per image (integer 0–999, no Boolean). Reports preserve every occurrence and rank ties by class index. Labels/dataset/training truth remain declarations.

The v6 integer recipe in [resnet_quantized.py](resnet_quantized.py) uses signed symmetric INT8, power-of-two per-channel weights/per-tensor activations, checked INT32 contractions/bias, ties-away requantization, accumulator residuals and integer pooling. Scales coarsen for overflow instead of saturating bias. TRAIN-only bounded MSE calibration records ranges/bounds, propagates sole-consumer ReLU ranges and rounds scaled spatial sums once. All learned convolutions/classifier use Gemmini; packing, requantization, residuals, pooling and logit dequantization are CPU work. Evaluation never selects coefficients.

Set `model2mlir_source` to the selected local checkout. Obtain the local checkpoint/archive and prepare disjoint TRAIN80/development20/final100 streams:

```bash
model_cache="$(PYTHONPATH="$merlin_root/src:$PYTHONPATH" python -c 'from merlin.common.paths import artifacts_dir; print(artifacts_dir() / "cache")')" || exit 1
test -n "$model_cache" || exit 1
mkdir -p "$model_cache/resnet50"
curl --fail --location --retry 3 --continue-at - --output "$model_cache/resnet50/resnet50-11ad3fa6.pth" https://download.pytorch.org/models/resnet50-11ad3fa6.pth
curl --fail --location --retry 3 --continue-at - --output "$model_cache/resnet50/imagenette2-320.tgz" https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
input_run="$model_cache/resnet50/train80-final100-new"
python "$setup_dir/prepare_resnet_inputs.py" --archive "$model_cache/resnet50/imagenette2-320.tgz" \
  --calibration-per-class 8 --evaluation-per-class 2 --final-per-class 10 --selection-seed resnet50-v2-train-final-v1 --output-dir "$input_run"
export RESNET_CHECKPOINT="$model_cache/resnet50/resnet50-11ad3fa6.pth" RESNET_TRAIN_INPUTS="$input_run/calibration.npz"
export FROZEN_QUANTIZATION="$input_run/frozen-quantization.json"
PYTHONPATH="$setup_dir:$merlin_root/src:$PYTHONPATH" python - <<'PYTHON'
import json, os
from pathlib import Path
import numpy as np
import torch, torchvision
import verify_resnet, resnet_quantized

torch.set_num_threads(1)
model = torchvision.models.resnet50(weights=None).cpu().eval()
verify_resnet.load_checkpoint(model, os.environ["RESNET_CHECKPOINT"], torch)
calibration, identity = verify_resnet.load_images(os.environ["RESNET_TRAIN_INPUTS"], np)
plan = resnet_quantized.calibrate_resnet50(model, calibration,
    calibration_ids=[f"{identity['sha256']}:{i}" for i in range(len(calibration))],
    checkpoint_declaration="local supplied state_dict", activation_estimator="mse", residual_policy="accumulator")
with Path(os.environ["FROZEN_QUANTIZATION"]).open("x") as stream:
    json.dump(plan.manifest(), stream, indent=2, sort_keys=True)
PYTHON
python "$setup_dir/verify_resnet.py" --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" \
  --checkpoint "$RESNET_CHECKPOINT" --inputs "$input_run/evaluation.npz" --labels "$input_run/labels.json" \
  --input-source 'Imagenette320 val; ordered samples in preparation.json' --preprocessing 'torchvision V2 RGB/resize232/crop224/normalization' \
  --quantized --calibration-inputs "$RESNET_TRAIN_INPUTS" --calibration-source 'Imagenette320 train; ordered samples in preparation.json' \
  --activation-estimator mse --residual-policy accumulator --frozen-quantization "$FROZEN_QUANTIZATION" \
  --graph-mode both --export-baremetal --max-top1-drop 0.02 --memory-limit-bytes 268435456 --output-dir "$build_root/resnet-v6-development-new"
```

The [preparer](prepare_resnet_inputs.py) reads without archive extraction, applies installed V2 transforms and refuses unsafe entries/duplicate/overlapping content. It binds archive/member/transform bytes without authenticating the official archive or training provenance. Freeze calibration with one Torch thread, matching the verifier: the manifest compares observed statistics exactly, including statistics whose small rounding differences leave selected coefficients unchanged. For final evaluation, replace inputs/labels with `final.npz`/`final_labels.json`, add `--evaluation-role held-out` and use a fresh output directory; recomputed TRAIN coefficients must exactly match the frozen manifest. The unchanged 0.02 top-one rate-loss gate compares the original FP32 model and is a development proposal, not paper approval. All images require exact raw INT32 and dequantized FP32 agreement. `--device-sample-indices 0,18` (development) or predeclared `0,90` (final) selects fixtures after full host gates; the device stream remains a subset. Exported `reference.npz`/`reference.json` bind independent integer results; execute `baremetal_baseline`/`baremetal_optimized` with the TVM graph-simulator guide. Export alone executes no device workload.

## Stateful verifier

[verify_session.py](verify_session.py) consumes `get_model_and_inputs` plus model2MLIR's `ExternalRuntimeSession` protocol (optional `get_session_spec` for a single program). Static signatures track changing state shapes and are reused across repeats. Torch/TVM state is independently routed; every output and state update and full reset must pass. Factory arguments are a JSON keyword object; repeat `--artifact` for additional local files. Factory loading disables Python network/HuggingFace downloads, without providing a process sandbox or implicit weight conversion.

```bash
python "$setup_dir/verify_session.py" --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" --tvm-build "$TVM_BUILD" \
  --factory /absolute/path/to/loader.py --factory-arguments /absolute/path/to/arguments.json \
  --artifact /absolute/path/to/checkpoint --output-dir "$build_root/session-new-run"
```

Legacy export defaults to opset17; `--onnx-exporter dynamo` defaults to opset18, with explicit `--opset` (BF16 convolution requires 22). Optional modern dependencies are `ml_dtypes==0.5.4`, `onnxscript==0.5.6`, `onnx_ir==0.1.12`. Export copies stages to protect native cache/reset state, preserves BF16 raw uint16 payloads and typed FP64 scalars, validates external-weight paths/ranges and rechecks selected files after execution. `get_verification_artifacts` binds factory helpers/checkpoint/tokenizer/input sources; `get_onnx_export_options` may supply reviewed translations without overriding verification controls. Failed comparisons retain metrics and bounded 64-MiB raw snapshots; input-state bytes are diagnostic accounting, not peak memory.

Opt-in `--bf16-fused-export` retains public linear/attention/single-output LayerNorm/SiLU boundaries, BF16 interfaces and FP32 internal math while disabling ONNX optimization. It checks masks/causal scale/static normalized-affine shapes and rejects dropout, grouped-query expansion and incompatible dtypes. It avoids an extra BF16 SiLU sigmoid store; it does not guarantee all native CPU reduction/rounding behavior or relax source gates.

Only after host gates pass, `--export-baremetal --memory-limit-bytes BYTES` derives shared static stages and `baremetal/session/{route.json,fixture.npz,provenance.json}` from original Torch outputs with exact reset bytes. TVM's device verifier consumes `--session-route`, `--fixture`, `--fixture-provenance`; host allclose does not imply exact-byte device acceptance. Nonfinite tensors/implicit conversions/hidden allocators are refused. Bind CPU math closure explicitly with repeated `--cpu-runtime-library /absolute/path/to/library.a`; no library is selected by default. Bounded INT8/math/BF16/FP64 proofs do not qualify full models; a nontrivial sinf fixture retains a one-ULP device failure. ELF/stack/platform and retained output and fixture budgets remain separate.

Optional `--tile-choices /absolute/path/to/selection.json` requires baremetal export and forwards per-shape deployment choices to TVM's backend. The JSON format is `{"version":1,"choices":[{"shape":[17,19,33],"tiles":[4,2]}]}`, with shapes ordered M/N/K and tiles I/J. Duplicate or absent workload shapes are refused, and the backend checks schedule legality. The report records selected tiles, actual prepared contractions and the selection file identity; session provenance binds that file. Unspecified shapes retain backend defaults. This is a deployment input, not authenticated tuning evidence: retain the independent correctness, platform, finite-budget and repeated-measurement receipts for any claimed measured winner.

## TinyLlama host recipe

[tinyllama_session.py](tinyllama_session.py) requires the full [checkpoint/tokenizer](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/fe8a4ea1ffedaf415f4da2f062534de366a451e6), all 22 layers and local UTF-8 corpus. Explicit FP32 conversion of official BF16 weights is a compiler/development diagnostic, not original-precision quality qualification. Use Transformers 5.4.0's cache API/modern exporter in a path isolated from SmolVLA:

```bash
python -m pip install --target "$build_root/tinyllama-python-v5" 'transformers==5.4.0' 'huggingface_hub==1.5.0' \
  'tokenizers==0.22.1' 'safetensors==0.6.2' 'numpy==1.26.4' 'onnx==1.17.0' 'ml_dtypes==0.5.4' 'onnxscript==0.5.6' 'onnx_ir==0.1.12'
export PYTHONPATH="$build_root/tinyllama-python-v5:$merlin_root/src:$TVM_ROOT/python"
```

The default factory is an FP32 8+3 frontend diagnostic, `quantized: false`, with native arithmetic/rotary/SiLU. Explicit W8A8 (`quantized: true`) covers all 155 learned projections using per-output-channel INT8 weights, dynamic token INT8 activations, exact INT32 contractions and FP32 dequantization; other math/state is CPU FP32. Activation scale defaults to `power_of_two`; `maxabs` retains the older diagnostic. Complete-K integer references use actual operands.

Development language requires 128 prefill+32 teacher-forced decode, two repeats/reset, eight distinct attributed 128-token articles, NLL increase≤0.20 nats and top-one agreement≥0.80 versus the unchanged explicitly FP32-converted checkpoint. Prepare the [pinned WikiText validation parquet](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/validation-00000-of-00001.parquet) locally (`pyarrow` required), setting the five environment variables below to local snapshots/new outputs:

```bash
PYTHONPATH="$setup_dir:$merlin_root/src:$PYTHONPATH" python - <<'PYTHON'
import json, os
from pathlib import Path
from tinyllama_integer import prepare_wikitext_corpus
arguments = prepare_wikitext_corpus(os.environ["WIKITEXT_PARQUET"], os.environ["TINYLLAMA_CHECKPOINT"], os.environ["CORPUS_OUTPUT"])
arguments.update(checkpoint_dir=os.environ["TINYLLAMA_CHECKPOINT"], model2mlir_root=os.environ["MODEL2MLIR_ROOT"], precision="float32", cpu_threads=2,
    arithmetic_policy="fixed_tree_fp32_v3_softmax_reduce", rotary_policy="source_table_v1", silu_export_policy="torch_avx512_sleef_v1")
Path(os.environ["FACTORY_ARGUMENTS"]).write_text(json.dumps(arguments, indent=2) + "\n")
PYTHON
for graph_mode in baseline optimized; do
  python "$setup_dir/verify_session.py" --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" --tvm-build "$TVM_BUILD" \
    --factory "$setup_dir/tinyllama_session.py" --factory-arguments "$FACTORY_ARGUMENTS" --graph-mode "$graph_mode" \
    --onnx-exporter dynamo --repeats 2 --max-signatures 2 --output-dir "$build_root/tinyllama-v3-$graph_mode-new-run" || exit 1
done
```

Preparation verifies pinned parquet bytes and selects the first eight eligible articles without model feedback. `workload: synthetic_feasibility` separately permits 8+3/128+32 without a quality corpus. Native defaults and failed historical receipts remain intact; the candidate is unadopted. Recorded v3 quality is NLL increase 0.01698/top-one agreement 90.059% across 1,016 predictions.

| Explicit option | Arithmetic/scope limit |
| --- | --- |
| `arithmetic_policy: fixed_tree_fp32_v1` | Separately rounded FP32 products/adjacent balanced sums for QK and RMSNorm; no initial zero, FMA/reassociation/hidden FP64. Version-1 `merlin.fixed_order` helpers require positive static FP32/power-of-two K≤4096, finite nonoverflowing operands and visible `4*K` scratch. |
| `fixed_tree_fp32_v2_pv` / `fixed_tree_fp32_v3_softmax_reduce` | Add ordered PV, then active-mask max/subtract/backend-exp/ordered-sum/division softmax. Common 160-token bound pads positive zero to 256 leaves (may change signed zero); reject incompatible masks/cache before mutation. Backend exponentials remain distinct. |
| `rotary_policy: source_table_v1` | Configuration-derived native words for positions 0–159, checked full/cached equality; 80 KiB constants. Export requires admitted source masks/positions. |
| `silu_export_policy: torch_avx512_sleef_v1` | Binds pinned Torch/library/headers, AVX512 dispatch and two threads; proved 5,632-column 128-token/one-token calls. Source exp/eight FMAs/scaling/direct division, no Torch/SLEEF runtime/scratch. Dimension guards do not prove arbitrary thread-partition equivalence; LLVM only, C unsupported, target execution pending. |

All candidate CPU work, padding, scratch/reshape/output allocations count toward eventual memory/timing. Native/quality references and `rtol=atol=1e-4` gates remain unchanged.

## SmolVLA host recipe and source profiles

[smolvla_session.py](smolvla_session.py) preserves original LeRobot precision, including BF16 prefix/cache, for the full [policy](https://huggingface.co/lerobot/smolvla_base/tree/c83c3163b8ca9b7e67c509fffd9121e66cb96205) and supplied [backbone](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/dc831d5df58eb65ae2fc25b50b7f5a896edec069). The neural boundary is preprocessed observations→ten denoising steps→`[1,50,6]` actions, excluding robot execution/postprocessing. Training-time backbone/camera identity is unverified. Checkpoint namespaced statistics do not map to plain action/state keys; observed transforms are identity, with no silent namespace selection or physical-unit qualification.

Use a separate Python 3.12 environment with original LeRobot/Transformers versions:

Fresh installation of the historical version selection below is currently blocked by package metadata: `lerobot[smolvla]==0.5.1` requires Transformers 5.3.0, NumPy >=2.0 and Hugging Face Hub >=1.0, which conflict with this recorded recipe. Recover and authenticate the original environment/install manifest before complete model reproduction; do not silently upgrade the reference stack or treat a dependency override as qualification. The install command below records the unresolved combination.

```bash
python3.12 -m venv "$build_root/smolvla-venv"
"$build_root/smolvla-venv/bin/python" -m pip install 'lerobot[smolvla]==0.5.1' 'torch==2.10.0' 'transformers==4.57.1' \
  'onnx==1.19.1' 'onnxscript==0.7.2' 'numpy==1.26.4' 'ml_dtypes==0.5.4' 'pyarrow==21.0.0' 'pandas==2.3.3' 'av==15.1.0'
PYTHONPATH="$setup_dir:$merlin_root/src:$TVM_ROOT/python" "$build_root/smolvla-venv/bin/python" "$setup_dir/prepare_smolvla_observation.py" \
  --dataset-root "$SO100_DATASET" --checkpoint-dir "$SMOLVLA_CHECKPOINT" --backbone-dir "$SMOLVLM_BACKBONE" \
  --frame-index 0 --split train --noise-seed 0 --video-backend pyav --output-dir "$OBSERVATION_OUTPUT"
```

The [preparer](prepare_smolvla_observation.py) uses the [pinned SO100 dataset](https://huggingface.co/datasets/lerobot/svla_so100_pickplace/tree/3d6d687a25cdf1565cdf24550814f72d999a861d), saved processor and explicit top→camera1/wrist→camera2 mapping. It binds schema/parquet/video/decoder identities, never invents a camera/refits normalization and separates splits by whole episodes. Recorded replay supplies no closed-loop quality result. Create a factory arguments JSON with local paths:

```json
{
  "checkpoint_dir": "/absolute/path/to/smolvla-policy",
  "backbone_dir": "/absolute/path/to/smolvlm-backbone",
  "model2mlir_root": "/absolute/path/to/model2MLIR",
  "fixture_kind": "dataset", "input_npz": "/absolute/path/to/observation.npz",
  "input_manifest": "/absolute/path/to/observation.json",
  "input_source": "pinned revision/episode/frame and preprocessing",
  "selected_cameras": ["observation.images.camera1", "observation.images.camera2"],
  "seed": 0, "cpu_threads": 1, "precision": "original"
}
```

```bash
for graph_mode in baseline optimized; do
  PYTHONPATH="$merlin_root/src:$TVM_ROOT/python" TVM_FFI=ctypes "$build_root/smolvla-venv/bin/python" "$setup_dir/verify_session.py" \
    --model2mlir-root "$model2mlir_source" --tvm-source "$TVM_ROOT" --tvm-build "$TVM_BUILD" \
    --factory "$setup_dir/smolvla_session.py" --factory-arguments /absolute/path/to/smolvla-arguments.json \
    --graph-mode "$graph_mode" --onnx-exporter dynamo --opset 22 --repeats 2 --max-signatures 3 --output-dir "$build_root/smolvla-$graph_mode-new-run" || exit 1
done
```

Each graph mode requires a fresh process/output directory; an integer run also needs a distinct `integer_diagnostic_dir` per invocation. Defaults remain original/native; synthetic staging uses `fixture_kind: synthetic`. Explicit W8A8 requires `quantized: true` and a fresh `integer_diagnostic_dir` beneath Merlin artifacts. A target copy replaces learned linears with INT8/INT32, FP32 dequantization/bias and one original BF16/FP32 interface cast; other work keeps original precision. Eager/reset, sampled complete-K checks and same-state velocity/independent trajectory checks do not establish original-policy quality.

The following opt-in factory options require that integer recipe, original precision and one native thread. Each binds pinned native source/library/headers/dispatch, checks the admitted original graph/layout and preserves native references/gates; unmatched geometry retains ordinary export or is refused. Fused profiles require explicit Dynamo/`--bf16-fused-export`. These are bounded CPU source-route proofs, not general native/RISC-V equivalence. Finite admitted intermediates, round-to-nearest and gradual underflow are proof assumptions, not arbitrary runtime-value scans.

| Factory key and non-native value | Admitted source scope and visible storage |
| --- | --- |
| `source_mean_policy`: `torch_cpu_mean720_v1`, `torch_cpu_mean720_960_v1` | FP32 keepdim `[1,50,720]`, plus `[1,177,960]`; pinned AVX2 cascade/direct division. 260-byte reusable scratch plus outputs; square/epsilon/rsqrt/affine/casts unchanged. |
| `source_softmax_policy`: `torch_cpu_softmax177_v1`, `torch_cpu_softmax177_227_v1`, `torch_cpu_softmax177_227_prefix_v1` | FP32 `[1,15,50,177/227]`, plus `[1,15,177,177]`; AVX512 lane/exp/reciprocal order, 836-byte scratch. |
| `source_layernorm_policy`: `torch_cpu_layernorm768_v1` | BF16 `[1,1024,768]`, original BF16 affine/epsilon `1e-6`; AVX2 Welford/cascade/FMA, 320-byte scratch. |
| `source_conv_policy`: `torch_cpu_conv3x16_v1` | BF16 `[1,3,512,512]`, `[768,3,16,16]` weights/bias, stride16/padding0/dilation1/groups1; oneDNN software-BF16 order, 128-byte scratch. Reject unproved BF16/AMX dispatch and ISA/fpmath overrides. |
| `source_sgemm_policy`: `torch_cpu_sgemm177_v1`, `torch_cpu_attention177_227_v1`, `torch_cpu_attention177_227_prefix_v1` | Manual QK/PV177, add QK/PV227, then prefix PV177. FP32 increasing-K or software-BF16 paired FMA/odd-tail order; 4-byte scratch, bound physical layouts/highest precision. Alpha1/beta0 is inferred for ordinary MatMul; original internal variant/buffers were not observed. |
| `source_scalar_policy`: `torch_cpu_language_scale960_v1` | One BF16 `[1,48,960]` embedding site/original INT64 tokens: native FP32 sqrt960 scalar/opmath and one BF16 cast on a copied ExportedProgram. |
| `source_rotary_policy`: `torch_cpu_denoise_rotary64_table_v1` | Boolean prefix177/suffix50 position DAG, D64/wavelength10000, all reachable positions; source-derived FP32 sine/cosine tables occupy 58,368 bytes. Prefix rotary/FP64 time trigonometry unchanged; not portable MKL. |
| `source_flash_policy`: `torch_cpu_flash_bf16_vision1024_v1` | Static strided vision Q/K/V/positive-zero mask DAG; query256/key512 tiling, BF16 exp stores/FP32 online normalization, 985,472-byte serial scratch and pinned host `expf`. No Torch/MKL runtime; newlib/device unqualified. |
| `source_silu_policy`: `torch_cpu_silu_fp32_v1` | Contiguous FP32 complete 32-element vector groups, source exp/direct division; refuse offsets/strides/symbolic shapes/scalar tails. LLVM `fmaf`/`floorf` bind libm bytes; recorded process addresses unavailable. |

The factory forwards `source_layernorm_policy` as a named verifier option. With explicit Dynamo and `--bf16-fused-export`, StageCompiler selects the existing source-bound LayerNorm translation in its own slot; arbitrary factory translation overlaps remain rejected. The original shape/epsilon/affine checks and ordinary-shape fallback are unchanged. Receipts identify the selected LayerNorm profile separately from Flash. Public constructor/export regressions exercise this composition, not complete-policy arithmetic.

For a fresh qualification of the recorded integer/source-profile recipe, retain the observation/checkpoint fields above, use a distinct `integer_diagnostic_dir` beneath configured artifacts for each invocation, and add these factory arguments:

```json
{
  "quantized": true,
  "integer_diagnostic_dir": "/absolute/path/beneath/configured/artifacts/new-smolvla-integer-run",
  "source_mean_policy": "torch_cpu_mean720_960_v1",
  "source_softmax_policy": "torch_cpu_softmax177_227_prefix_v1",
  "source_layernorm_policy": "torch_cpu_layernorm768_v1",
  "source_conv_policy": "torch_cpu_conv3x16_v1",
  "source_sgemm_policy": "torch_cpu_attention177_227_prefix_v1",
  "source_scalar_policy": "torch_cpu_language_scale960_v1",
  "source_rotary_policy": "torch_cpu_denoise_rotary64_table_v1",
  "source_flash_policy": "torch_cpu_flash_bf16_vision1024_v1",
  "source_silu_policy": "torch_cpu_silu_fp32_v1"
}
```

Add `--bf16-fused-export` to the verifier command. The original precision and one-thread settings above remain required. A different PyTorch binary or CPU dispatch is refused by the source bindings; installing the same package version alone does not qualify that environment.

Recorded TRAIN qualification used authenticated retained ExportedPrograms/compiled prefix artifacts and freshly compiled flow/action stages (Comparisons `fddec47f`, TVM `6b376cc889`). Actual compiled outputs feed subsequent stages; references are comparison-only. Prefix masks/all value-cache words are native-exact; only three key-cache words differ within `rtol=atol=1e-4` (maximum `3.8147e-6`). Byte identity across modes/VMs/resets and source/input/artifact/IEEE checks pass. The generic command above uses native defaults; the explicit source-profile recipe requires fresh complete qualification and does not reproduce retained artifacts automatically. Full public packaging, original-policy quality, device/memory fit and timing remain pending. CPU casts, helpers, tables and all live storage count in eventual budgets.

[diagnose_smolvla_backends.py](diagnose_smolvla_backends.py) records original native flash/MATH trajectories, processor/statistics inventory and review-only numerical proposals without changing the active `1e-4` gate. Use `--help` for dataset/manifest attribution and `--velocity-replay`; output must be fresh beneath configured artifacts. Proposal inputs are TRAIN/development only, report `NO_ACCEPTANCE` and do not authenticate declared fixture identity or select action/robot limits. Ordered-arithmetic explorations remain unadopted local proofs; complete-policy native equivalence is not established by them. Preserve immutable failed receipts under `out/`.

## Target, simulator and verification entry points

For bounded compiler development, `verify_blocks.py` accepts explicit checkpoint
prefixes: `--action-hidden-size` (default 720), `--tiny-hidden-size` (default 2048),
and `--tiny-intermediate-size` (default 5632). Action positions remain explicitly
1 or 50; TinyLlama accepts `--tokens 1` through 16. Overrides apply only to the
matching isolated workload and cannot exceed the original tensor dimensions.
References, quantization scales and compiler output are regenerated for the
selected prefixes; reports retain original dimensions and exact selected weight
slice identities. These are reduced-dimension compiler diagnostics, with no
claim of original-model numerical equivalence or policy quality. A one-token
attention case also has a trivial singleton softmax and zero-position RoPE, so
it does not replace a multi-token attention test.

The original geometry remains the default. For example, a smaller TinyLlama
capture uses `--workload tinyllama-head-ffn --tokens 1 --tiny-hidden-size 1024
--tiny-intermediate-size 2816` alongside the required explicit tool/checkpoint
arguments. A smaller 50-position action projection uses `--workload
smolvla-action --action-chunk-size 50 --action-hidden-size 360`. Use fresh
artifact directories, enforced cgroup memory/no-swap/CPU limits and finite
deadlines. Feed a passing capture into `verify_blocks_rtl.py prepare`, then
`run`; RTL placement follows the resulting compiled manifest.

`verify_blocks.py` now defaults to `--macro-tiles largest`, independently of
the surrounding graph `--mode`. This explicitly selects the maximum-area
grouping from the pinned backend's existing 1/2/4 tiles per axis, subject to
authenticated parameter-header scratchpad/accumulator capacities and at least
90% occupied micro-tile slots across the macro grid. `--macro-fill-percent`
makes that threshold explicit; `--macro-tiles one` reproduces the earlier 1×1
grouping. `--gemmini-params` can bind the actual generated header, whose digest
must match the adapter receipt; otherwise its authenticated header is resolved.

A 1×1 grouping means one 16×16 hardware tile, not one processing element.
Larger groups reuse A across column tiles and B across row tiles. The backend
emits exact tail blocks, so macro grouping does not add tile-compute calls.
One-token cases still have a single useful output row; their intrinsic row
underutilization is separate from the macro-grid fill threshold. Typical
selections are 1×4 for one-token projections, 4×2 for a 50×32 action output,
and 4×4 for wider 50-row projections. This is a declared size heuristic, not
measured autotuning or a proof of maximum performance. Reports preserve the
candidates, hardware header, selected groups, utilization, resource use and
verified compiled attributes. No TVM-core, schedule implementation or RTL patch
is required. Earlier block runs used the default 1×1 grouping even when their
surrounding graph mode was named `optimized`.

```bash
python "$setup_dir/check_target.py" --config "$setup_dir/target.yaml"
python "$TVM_ROOT/apps/gemmini/verify_matmul.py" --gemmini-repo "$GEMMINI_ROOT" --build-root "$build_root" \
  --cc "$RISCV_BIN/riscv64-unknown-elf-gcc" --objdump "$RISCV_BIN/riscv64-unknown-elf-objdump"
python "$setup_dir/build_simulator.py" --libgemmini-repo "$LIBGEMMINI_REPO" --spike-prefix "$SPIKE_PREFIX" \
  --cxx "$HOST_CXX" --runtime-lib-dir "$SPIKE_SUPPORT/lib" --runtime-bin-dir "$SPIKE_SUPPORT/bin" \
  --smoke-elf "$CPU_PROOF/model.elf" --build-root "$build_root" --output-dir "$build_root/simulator-new"
python "$TVM_ROOT/apps/gemmini/verify_matmul_runtime.py" --adapter-receipt "$ADAPTER_BUILD/receipt.json" --simulator-receipt "$build_root/simulator-new/receipt.json" \
  --riscv-gcc "$RISCV_BIN/riscv64-unknown-elf-gcc" --spike "$SPIKE_PREFIX/bin/spike" --dtc "$SPIKE_SUPPORT/bin/dtc" \
  --spike-library-dir "$SPIKE_SUPPORT/lib" --plugin "$build_root/simulator-new/libgemmini.so" --output-dir "$build_root/matmul-runtime-new"
```

Supply existing matching reference repositories/tool installations and a passing CPU ELF. Adapter/simulator stages extract exact integer Git blobs without modifying references, check headers/tools/dependencies and bind immutable identities. Plugin loading alone executes no Gemmini arithmetic. Runtime fixtures check extremes/tails/strides, guards, repeated-call recovery and intentional failures; audit the final ELF for no FSM. Link audited `matmul.o`, never diagnostic `matmul.raw.o`. Final deployment/coherence/capacity and timing remain separate.

`check_target.py --memory-plan /path/to/reservations.json` accepts disjoint coarse `regions` with `name`, integer `address`, integer `size_bytes`; account for code/static data, weights, workspace, stack and runtime without double-counting arena children. Gaps are not measured free memory; omitted allocations stay unknown. The checker rejects incompatible arithmetic/WS/loop policy/ranges/confirmed-capacity requests, but does not verify hardware execution.

Use each CLI's `--help` and the TVM integration guide for focused cases. Comparison regression entry points are `merlin/tests/gemmini/test_tvm_{target,resnet_artifacts,resnet_quantized,session}.py`, `test_tinyllama_session.py` and `test_smolvla_session.py`; invoke explicit files with the selected model environment and `PYTHONPATH="$merlin_root/src:$PYTHONPATH"`. Optional real integration fixtures use `TVM_SESSION_MODERN_ONNX=1 TVM_SESSION_MODEL2MLIR="$model2mlir_source"` or `TVM_TINYLLAMA_INTEGER_ONNX=1`; Smol staging uses `MODEL2MLIR_ROOT`. Run the source/library-bound TVM regressions appropriate to a backend change. Static inspection does not substitute for numerical execution, device qualification or timing.
