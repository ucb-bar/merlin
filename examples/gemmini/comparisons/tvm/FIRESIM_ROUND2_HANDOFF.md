# FireSim runner handoff: ResNet50 and TinyLlama

This is the current execution plan. **SmolVLA is paused by the user: do not
prepare, compile, tune or run it in the next FireSim campaign.** Its existing
captures and interrupted reference attempt are preserved for later work.

Start ready ResNet work while TinyLlama numerical preparation runs on the
remote CPU host. Do not wait for TinyLlama to finish before restoring ResNet
or qualifying the installed launcher. No remote jobs have been launched here.

## Deliverables and readiness

| Deliverable | Location | Status |
| --- | --- | --- |
| ResNet baseline ELF | `out/artifacts/handoff/gemmini-firesim-dse-publication/v1/latest/ELFS/resnet-baseline.elf` | Ready for the existing ELF launcher; positive/wrong-oracle Spike controls passed, all 1,000 logits checked |
| ResNet parent export, 18 complete chunks, fixtures and study | Same publication, `restore.py` plus five checksum-verified archive parts | Ready to restore; 18 host comparisons have zero difference; 21 distinct contraction workloads assigned to 10 owners |
| TinyLlama calibration/input/range seed | `out/artifacts/handoff/gemmini-tinyllama-remote-seed/v1/latest/` | Committed small portable inputs and source snapshot; **not completed quantized exports or golden outputs** |
| TinyLlama preparation code | `prepare_remote_tinyllama.py`, `selected_tinyllama{,_functional}_loader.py`, `calculate_selected_references.py` in this directory | Resume on remote CPU; focused cache/repeated-export controls passed |
| TVM compiler | `https://github.com/grammy117/tvm`, branch `gemmini/firesim-chunks`, commit `3423bd6818ab752ddd9cdb5d5773743e48c65f1d` | Existing published backend; no TVM source change in this preparation |
| TinyLlama 16/1 chunks and 32/32 session | Produced by the jobs below | **Pending**; legacy/synthetic ELFs do not fill this gap |

Baseline ELF SHA256:
`4ec5e14fa82724d6aff3526a6beeb273dfa18637d4e316b386f5cbe487d44acf`.
ResNet's new identity moves all 54 biases into INT32, with bounds/equivalence
proofs and unchanged weights/scales. Hardware checks and fixed-set accuracy
remain separate. The old publication README and archive describe their frozen
preparation; this file takes precedence for current scope and job ordering.

## Work in parallel

| Job | Work | Dependency / resource |
| --- | --- | --- |
| R0 | Restore ResNet, relocate fixture proofs, verify baseline ELF and qualify installed runner/cycle observer | Start immediately; host setup plus assigned physical FireSim slots |
| R1 | ResNet baseline E2E and initial contraction DSE on stem / representative full bottlenecks | R0; independent owner campaigns across actual free slots |
| R2 | Compile/replay winners; confirm connected chunks, then baseline versus tuned full ResNet with stage profiling | R1; use the same image/input/numerical source and hardware |
| T0 | Restore TinyLlama seed, fetch exact checkpoint, prepare quantized prefill/decode exports and independent outputs | Start on remote CPU alongside R0/R1; no FPGA reservation needed |
| T1 | Derive 16/1 iteration fixtures and meaningful complete MLP/attention chunks; extract/deduplicate tasks and check correctness | T0; CPU/build workers |
| T2 | TVM DSE on those TinyLlama chunks, then connected confirmation | T1 plus qualified runner; share the physical pool and global budget with ResNet |
| T3 | Compile baseline and tuned complete TinyLlama **32-input/32-output-token** session, check all 99 arrays, run E2E comparison | T2, with actual matching full-model traces; keep baseline as well as tuned result |

Keep independent CPU preparation/build jobs and FPGA slots busy concurrently.
Use the actual assigned slot count; never attach two drivers to one physical
slot. All campaigns share slot locks, one proposal ledger and one study
supervisor. Add ready TinyLlama rows for the next scheduling wave using the
same ledger/deadline; do not launch a second supervisor against that ledger.
Bound CPU build/preparation concurrency from observed RSS and available RAM,
retain memory headroom, and increase only when the measured resource use fits.
There is no assumed F2 slot count, clock, memory capacity or job duration.

## Restore ResNet now

From the checked-out `comparisons/tvm-workload-tuning` branch:

```bash
export MERLIN="$PWD"
export DELIVERY="$MERLIN/out/artifacts/handoff/gemmini-firesim-dse-publication/v1/latest"
(cd "$DELIVERY" && sha256sum -c SHA256SUMS)
python3 "$DELIVERY/restore.py" --output-dir "$DELIVERY/unpacked"
export INPUTS="$DELIVERY/unpacked"
```

Read the restored `study.json`, `task-owners.json`, `resnet-checks.json` and
`relocate_proofs.py`. Use committed source directly; **skip the archive's old
source-patch/overlay application block**. Its frozen README retains the concrete
pool/toolchain templates, compilation/session commands and required receipts.
Replace template paths with actual installed tools and image identities.
Existing baseline execution needs only the ELF launcher, not a TVM rebuild.

`run_firesim_study.py --study ... --pool ... --toolchain ... --output-dir ...
--target-clock-hz ...` first emits a plan. For execution add `--execute`, the
same `--deadline-epoch`, `--ledger` and available `--jobs`. Supply the documented
**target cycle-counter frequency**, never guessed FPGA/emulation throughput.
The initial archive's TinyLlama rows are intentionally pending. Mark them ready
only after replacing their paths with the actual checked new-release chunks.

## TinyLlama authority files: read these, do not infer major choices

Paths in this table are repository-relative unless prefixed by `$SEED` or
`$TINY_RELEASE`. The first three are the files the user identified earlier.
They are source definitions; they do not themselves contain golden tensors.

| Information | Exact file / section |
| --- | --- |
| Model/checkpoint, full dimensions, inputs/outputs, calibration, host math, recurrence | [`examples/gemmini/protocol/experiment.yaml`](../../protocol/experiment.yaml), `models.tiny_llama` (line 103) |
| Meaning of protocol, prepared-IR oracle and all-output checks | [`examples/gemmini/protocol/README.md`](../../protocol/README.md), “Full networks and exact interfaces” and “Two profiles and three comparisons” |
| Argument order/carry, 32/32 chain, numerical reporting and functional tolerance | [`examples/gemmini/numerics/models.yaml`](../../numerics/models.yaml), `numerics` and `models.tiny_llama` |
| Latest user-directed arithmetic and quality override | [`examples/gemmini/numerics/CURRENT_RUN_POLICY.md`](../../numerics/CURRENT_RUN_POLICY.md) and `.json`; these supersede older quality descriptions |
| Function-preserving splitting algebra (threshold itself is not defined there) | [`examples/gemmini/numerics/QUANTIZATION.md`](../../numerics/QUANTIZATION.md), section 8 |
| Exact quantization recipe/planes/observers | [`examples/gemmini/protocol/recipes/tiny_llama.json`](../../protocol/recipes/tiny_llama.json) |
| Selected input token IDs | [`examples/gemmini/numerics/inputs/tiny_llama/visible.json`](../../numerics/inputs/tiny_llama/visible.json), `v00`; derived bytes `$SEED/prepared-inputs/tiny-v00.npz` |
| Exact training calibration roster | [`examples/gemmini/numerics/calibration/tiny_llama.json`](../../numerics/calibration/tiny_llama.json); realized tokens `$SEED/prepared-inputs/tiny-calibration.npz` |
| Checkpoint config/tokenizer and download identity | `$SEED/source/tiny/model/{config.json,generation_config.json,tokenizer*}`, `$SEED/seed.json` |
| Frozen range statistics / concrete transform realization | `$SEED/tiny-range-statistics.npz`, `tiny-range-identity-v2.json`, `tiny-range-realization-v2.json`; implementation [`selected_tinyllama_loader.py`](selected_tinyllama_loader.py) |
| Canonical cache/session implementation | `$MERLIN_M2M_DIR/m2m/capture/causal_session.py` in the supplied pinned source; public FX boundary in [`selected_tinyllama_functional_loader.py`](selected_tinyllama_functional_loader.py) |
| Quantized programs/scales/weights — **pending production** | `$TINY_RELEASE/tiny-stages/stages/{prefill,decode}/program.pt2` and corresponding `model.mlir`, weights and capture receipts |
| Independent expected outputs — **pending production** | `$TINY_RELEASE/references/all-outputs.npz`, `reference.json`, `prepared/` and `arguments/`, generated by [`calculate_selected_references.py`](calculate_selected_references.py) |
| Chunk boundaries and expected live outputs — **pending extraction/checks** | Explicit region manifest plus each chunk's `chunk.json`, `fixture.npz`, `fixture-provenance.json` and parent hashes, produced with [`prepare_exported_chunks.py`](prepare_exported_chunks.py) |

Concrete full-profile dimensions: batch 1; 22 layers; hidden 2048; original MLP
5632, transformed buckets 5728/6096; 32 query heads / 4 KV heads / head dimension
64; vocabulary 32000. Prefill tokens are I64 `[1,32]`; decode token I64 `[1,1]`.
Every call returns logits F32 `[1,32000]`, cache F32 `[2,22,1,4,64,64]` and
positions I64 `[22]`. Carry actual cache/positions between calls; reset each
complete input/repetition. The 16/1 iteration profile changes sequence length
and call count only; freeze its explicit derived profile and check all six
output arrays. The 32/32 full profile checks all 99 arrays.

The pending full-reference NPZ keys are `prefill_00_output_0` through
`prefill_00_output_2`, and `decode_00_output_0` through
`decode_31_output_2`; outputs 0/1/2 are logits/cache/positions. The independent
reference must use actual carried quantized-program state, not cached FP32
calibration state. Chunk fixtures must include expected values for **every
live output**, not just the GEMM or final logits.

**Do not guess major choices:** checkpoint/revision, token IDs, calibration
roster/counts, cache layout/capacity, transformed widths/algorithm, attention
planes, arithmetic/host math, output selection and tolerances are fixed by
these bindings. Do not replace them with defaults, a smaller model, synthetic
calibration, free generation or a legacy export. The local outlier-threshold
formula is explicitly recorded below; historical equivalence remains unproved.
If any required major decision is genuinely unspecified or conflicts across
these files, report the exact conflict and continue independent ResNet work;
do not silently choose a new numerical experiment or relax a check.

The remaining TinyLlama work is **finish quantized exports → calculate
independent full outputs → derive/check 16/1 chunks → DSE and connected/E2E
confirmation**. It is not yet only splitting an already completed quantized
model and calculating chunk outputs.

## Resume TinyLlama preparation on the remote

The seed contains the exact WikiText training token roster, exact selected v00
token input, frozen 32-segment range statistics, realized widths, model/tokenizer
configuration, checksums, and a standalone model2MLIR Git bundle plus the reviewed patch.
The large checkpoint is fetched once from its pinned public revision, or reused
with `TINYLLAMA_CHECKPOINT_FILE` after verifying its SHA256.

Use an isolated Python 3.12 environment. Install the seed's pinned requirements;
CPU Torch 2.10.0 is sufficient. Do not install SmolVLA/LeRobot/Diffusers for this
run or reproduce the old combined Python overlays. Keep the RISC-V/TVM build
environment separately pinned as in the restored toolchain instructions.

```bash
export SEED_DELIVERY="$MERLIN/out/artifacts/handoff/gemmini-tinyllama-remote-seed/v1/latest"
(cd "$SEED_DELIVERY" && sha256sum -c SHA256SUMS)
mkdir "$SEED_DELIVERY/unpacked"
tar -xzf "$SEED_DELIVERY/tinyllama-seed.tar.gz" -C "$SEED_DELIVERY/unpacked"
export SEED="$SEED_DELIVERY/unpacked"
(cd "$SEED" && sha256sum -c SHA256SUMS)
python3 -m venv "$MERLIN/out/build/tiny-preparation-venv"
export PY="$MERLIN/out/build/tiny-preparation-venv/bin/python"
"$PY" -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install -r "$SEED/requirements.txt"
export MERLIN_REPO_ROOT="$MERLIN"
export MERLIN_M2M_DIR="$MERLIN/out/build/upstreams/model2mlir-tiny-preparation"
git clone "$SEED/source/model2mlir.bundle" "$MERLIN_M2M_DIR"
git -C "$MERLIN_M2M_DIR" checkout --detach 5fec4a2dc106668e4e74605eeaa54729df36e1b4
# The reviewed patch is already in this commit; do not apply it twice.
export PYTHONPATH="$MERLIN_M2M_DIR:$MERLIN/examples/gemmini/comparisons/tvm:$MERLIN/src"
"$PY" "$MERLIN/examples/gemmini/comparisons/tvm/prepare_remote_tinyllama.py"   --phase bootstrap --seed-dir "$SEED"
# The command prints the fresh product directory; assign that path below.
export TINY_RELEASE="<printed-directory>"
for PHASE in prefill decode references; do
  "$PY" "$MERLIN/examples/gemmini/comparisons/tvm/prepare_remote_tinyllama.py"     --phase "$PHASE" --release "$TINY_RELEASE" || exit 1
done
```

Do not execute this block over an existing output tree. Each guarded phase keeps
its log/status; preserve failed attempts and use fresh product/attempt paths for
retries. The loop is sequential for constrained RAM; it runs alongside ResNet's
remote work. A larger host may run independent prefill/decode CPU jobs together
only after checking memory use. References require both completed stages.

The model2MLIR patch releases dead integer-reference intermediates while retaining
extra quantized-operand lifetimes needed for contraction diagnostics. Sixteen
focused tests and old/new linear/conv/batched controls pass with identical outputs
and partial-sum evidence. This is a normal source patch, **not a TVM patch or a
runtime monkeypatch**. The functional cache loader uses public Torch export/FX
APIs and restores verified Linear leaves for the existing recipe inventory.
Its repeated-export control passes; the full-model capture still needs to pass.

Preparation must retain the selected canonical FP32 cache `[2,22,1,4,64,64]`,
I64 positions `[22]`, FP32 logits `[1,32000]`, actual calibration, attention
planes and `portable_exp_trig` host math. The local range realization explicitly
uses split threshold `8 * median(positive observed channel absmax)`, producing
padded widths 5728/6096. This formula is recorded, not proof of byte-identical
historical preparation; freeze the fresh identity and report that distinction.
Do not silently change it, calibration, cache ABI or numerical tolerances to
make conversion pass. Record and report any remaining full-model export failure.

`references/all-outputs.npz` and `reference.json` must contain **99** checked
arrays from the independent ungrouped prepared-IR evaluator, including carried
cache and positions. Torch fake-quant outputs or the older synthetic release
are not substitutes. No independent 99-array reference has yet been completed
for this fresh selected-profile preparation.

## TinyLlama chunks and record mapping

Iteration/tuning is **16 prefill tokens + 1 decode token**. Final baseline/tuned
E2E is **32 prefill + 32 teacher-forced decode calls**, using the same frozen
weights/scales and physical cache capacity 64. Keep the supplied 160-token v00
input; use its first 17 or 64 IDs respectively. Do not recalibrate for 16/1.
The preparation command above first freezes the full 32/32 numerical source;
**deriving/checking the 16/1 programs and fixtures is still a remote job**.

Extract complete connected regions from that frozen source: one full MLP
(norm, gate/up, SiLU/product, down and residual) for each actual width bucket;
one full attention region (Q/K/V/O, RoPE, cache/mask, QK, softmax, PV and every
selected operand plane) in prefill and decode. Inventory embedding/output-head
and other missing contractions as well. Keep full reduction/output dimensions,
all heads, real boundary activations/cache and all live outputs. One layer and
one decode call make a small meaningful chunk; an isolated cropped output panel
is not the connected confirmation. `prepare_exported_chunks.py` preserves exact
explicit FX regions; its source-FX control does not replace prepared-IR oracle
admission or the selected portable host math.

For each chunk retain parent export hash, node interval, fixtures, scale hashes
and source call mapping. Match schedule records to E2E by **structural TIR
equality** plus target/capability/measurement compatibility. M16 prefill records
are not automatically M32 records; even M1 decode records need an actual match.
Inventory full-parent coverage, replay matching traces, retain default schedules
on misses, and confirm whole chunks and measured E2E. Never sum operator minima
and call that a model speedup.

## Fixed scope, cost and correctness rules

- **Do not implement the proposed optimization knobs.** Existing TVM/Gemmini
  contraction scheduling only; graph fusion fixed at 2. No new CPU loop,
  producer-placement, panel-packing pass or graph-fusion sweep.
- TVM EvolutionarySearch/XGB selects schedules through the existing APIs. No
  handwritten winning parameter tuple or static sweep replacing that search.
- Initial ResNet: 48 proposals; optional continuation: 20; TinyLlama: 24 once
  ready. Preserve the existing shared cap of 96 and **one absolute four-hour
  study deadline**, including paid baselines, controls, confirmations and E2E.
  Do not reset the deadline for later waves or spend the confirmation reserve
  on extra tuning. Report unfinished work if preparation uses up the window.
- Abort a DSE candidate at **active measured cycles >= current best for the
  same workload**, including memory stalls/final completion, excluding startup,
  loading, initialization and warmup. Tighten an incumbent in flight.
- Qualify that actual observer/cancellation path using [ADAPTER_CONTRACT.md](ADAPTER_CONTRACT.md).
  The previous adapter's total-cycle cap and completion-only filtering did not
  enforce this. Standalone baselines may use the existing launcher while this
  is checked; paid DSE requires demonstrated enforcement, not boolean labels.
- The supplied bitstream already sets memory; **DDR3 is expected**. Deliver
  ELFs to the installed launcher; do not rebuild/configure the memory model or
  invent hardware capabilities. Keep eligible contractions on Gemmini.
- Preserve positive/wrong-oracle controls and every output and state check. Apply
  [CURRENT_RUN_POLICY.md](../../numerics/CURRENT_RUN_POLICY.md): fixed W8A8,
  group16 weights, static per-tensor activations, ties-even/saturation and INT32
  bias. Record actual extra attention-plane work and all allowed FP32 host work.
- Quality remains separate: ResNet <=1 percentage point top-1 loss; TinyLlama
  WikiText-2 test PPL <=1.10x matched FP32. Publish evaluation settings/baselines
  or report NOT_RUN; do not claim quality from one input or from compiler checks.

Return the study ledger/status, source/ELF/hardware hashes, numerical verdicts,
coverage/misses, winning TVM databases/traces, per-stage and E2E active cycles,
wall/cost records, cancelled candidates and remaining blockers. No new speedup
or full-model numerical/quality pass is claimed by this handoff.
