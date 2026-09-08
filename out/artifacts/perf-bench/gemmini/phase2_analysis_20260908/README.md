# Performance campaign analysis — gemmini, 2026-09-08

The functional lane's report asks *how many capsules pass over time*. A performance campaign has no
such curve: its output is a compiler change and a measurement, and a round that authors nothing still
costs money. So this analysis answers three different questions — **where the budget went**, **what
was attempted and on what evidence**, and **what none of it can decide** — and it is generated, not
assembled: `gen_phase2_analysis.py --target gemmini` rewrites `campaign_analysis.json` beside this
file.

Every figure below is paired with a status in that JSON (`measured` / `derived` /
`unavailable(reason)`), the same discipline the functional lane uses, because **a zero plots as a
finding and a refusal plots as a gap**, and confusing the two makes a stalled campaign look like a
converged one. This run's availability score is **1.000** — every field reported here was measured,
none derived, none refused.

| | |
|---|---|
| stage/campaign directories with telemetry | **69** |
| tool spans | **1,535** (+324 point events excluded) |
| brokered actions | **1,296 calls, 11.25 h** |
| of which refused | **154 calls (11.9%), 1.97 h (17.5% of wall)** |
| optimization attempts | **56** — 35 helped, 13 refuted, 7 blocked, 1 unmeasured |
| distinct instruments credited | **56** |
| scopes reached | **7 of 7** |
| ledger integrity problems | **0** |

---

# PART I — Where the budget actually went, and why

## The three cost tiers differ by two orders of magnitude

| tier | calls | wall | share of wall | refused | what it is |
|---|---:|---:|---:|---:|---|
| `free` | 120 | 0.00 h | 0.0% | 0 | answered from artifacts already on disk |
| `compile` | 799 | 0.72 h | 6.4% | 58 | a lowering or emission step, ~3 s |
| `measure` | 377 | **10.53 h** | **93.6%** | **96** | a simulation or profile, ~100 s |

**29% of the calls consumed 93.6% of the wall time.** That single ratio decides how a campaign should
be shaped: compiles are effectively free and can be run speculatively, while every measurement has to
be earned. It is also why the cheap-instrument stack built this round matters — `offload`,
`lane_cost`, `work_volume` and `compose_estimate.band` all answer in milliseconds from the emitted
buffer, and each one that eliminates a candidate saves a ~100 s measurement.

## A quarter of the expensive tier refused

| action | tier | calls | wall | mean | refused | rate |
|---|---|---:|---:|---:|---:|---:|
| `tuning-gsim-feedback` | measure | 197 | **5.35 h** | 97.8 s | 61 | **31.0%** |
| `analyze-whole-model` | measure | 142 | 4.11 h | 104.3 s | 24 | 16.9% |
| `qualify-changed-region` | measure | 38 | 1.06 h | 100.7 s | 11 | 28.9% |
| `candidate-emit-command-buffer` | compile | 216 | 0.19 h | 3.2 s | 0 | 0.0% |
| `candidate-lower-target-to-llvm` | compile | 176 | 0.15 h | 3.1 s | 0 | 0.0% |
| `candidate-lower-interface-to-target` | compile | 137 | 0.12 h | 3.1 s | 1 | 0.7% |
| `inspect-optimization-surfaces` | compile | 140 | 0.09 h | 2.4 s | 28 | 20.0% |
| `candidate-parse` | compile | 94 | 0.08 h | 3.2 s | 1 | 1.1% |
| `profile-reduced-global-witness` | compile | 17 | 0.06 h | 11.7 s | 17 | **100.0%** |
| `prepare-source-contraction` | compile | 4 | 0.02 h | 18.9 s | 3 | 75.0% |
| `prepare-source-convolution` | compile | 7 | 0.00 h | 1.9 s | 7 | **100.0%** |
| `compare-controlled-context` | compile | 1 | 0.00 h | 8.0 s | 1 | **100.0%** |
| `analyze-command-buffers` | free | 120 | 0.00 h | 0.0 s | 0 | 0.0% |

A refusal is not a failure: the action declined to run, so it produced no evidence and cost the same
wall time. Reporting the two separately is the point — 11 calls actually *failed* (rc 1) against 154
that refused.

**Three actions refused every single time they were called.** `profile-reduced-global-witness`
(17/17) is the probe path: `--probe-interface` was never passed, so the provider stayed `None` and
every call refused before doing anything. That is why `probe_receipts` is `[]` in every binding
receipt, and it is a launcher argument rather than a code inversion — which is the whole reason the
plan's original "flip `full_model_simulation_allowed`" step was wrong.

**The single largest line item is an action that cannot succeed in this mode.**
`tuning-gsim-feedback` is 197 calls and 5.35 h — 47.6% of all brokered wall time — and it is
*hard-refused in global mode* by the driver. Its 61 refusals alone are **1.66 h**. The campaign spent
its largest single block of time asking for feedback the mode forbids.

## The lane is essentially serial, and that is a shape not a defect

1,535 tool spans across 23 stages, with 324 `file_change` rows excluded because they are
instantaneous events (99% land under 10 ms) and would deflate any occupancy figure. Peak concurrency
in this lane is low by construction: a candidate must be emitted before it can be lowered, lowered
before it can be measured. The parallelism story belongs to the functional lane; here the lever is
**making each step cheaper or unnecessary**, not overlapping them.

---

# PART II — What was attempted: why, where, how

56 attempts, each carrying the instrument that surfaced it, the hypothesis it tested, the scope it
acted at, and the measured delta that decided it. `decomposition.json` holds every row in full;
`optimization_ledger.txt` is the flat table.

## Where the work landed, and the two scopes that could not be reached

| scope | helped | refuted | blocked | unmeasured | total | what it means |
|---|---:|---:|---:|---:|---:|---|
| `global` | 17 | 4 | 1 | 0 | 22 | whole-graph: placement, offload, what reaches the unit at all |
| `build` | 10 | 4 | 0 | 1 | 15 | toolchain and pipeline order, no compiler change |
| `encoding` | 4 | 2 | 1 | 0 | 7 | instruction selection and packing |
| `host_lane` | 2 | 2 | 0 | 0 | 4 | code quality on the scalar lane |
| `local` | 2 | 1 | 0 | 0 | 3 | inside one region: loop shape, tiling |
| **`frontend`** | **0** | 0 | **3** | 0 | 3 | **what the compiler can ingest at all** |
| **`inter_layer`** | **0** | 0 | **2** | 0 | 2 | **epilogue fusion, residency across a block** |

**The only two scopes with zero `helped` rows are precisely the two that depend on another lane.**
`frontend` is the model export (attention quantization); `inter_layer` is the epilogue narrowing,
which is another session's active lane. Every scope this campaign could reach on its own produced
measured improvements. That is the strongest available evidence that the campaign was
instrument-limited and dependency-limited rather than idea-limited — and it is why the remaining work
is coordination, not search.

## The measured deltas span the fleet, not one model

12 deltas on `smolvla`, 11 on `resnet50`, 6 on `tiny_llama`, 1 on `lstmnetvit`, 10 on the `fleet` as
a whole, plus 12 on the instrument `corpus` and 8 on the `campaign` itself. The compiler changes were
required to help every workload, and the fleet rows are where that is asserted.

## The four findings that changed what the campaign believed

### 1. The entry stack frame was a fleet-wide defect, not a fixed historical one

*Instrument:* the target's own kernel stack-frame preflight, run over every emission in the cache.

The plan recorded this as already fixed, on the strength of a receipt reporting a **816-byte**
ResNet-50 frame. Compiling every emission measures all four models over the 65,536-byte budget:

| model | allocas bound | before | after | over budget by |
|---|---:|---:|---:|---:|
| resnet50 | 66 | 23,406,400 B | **784 B** | 357× |
| lstmnetvit | 449 | 25,956,928 B | **384 B** | 396× |
| tiny_llama | 1,636 | 48,976,448 B | **9,152 B** | 747× |
| smolvla | 1,534 | 99,897,984 B | **496 B** | 1,524× |

Zero allocations refused on any of them. The two ResNet-50 numbers are not a regression — they
describe *different emissions of one model*, and a receipt is evidence about the bytes it was
measured on. **How** it is fixed matters as much as that it is: the repair runs only after the emitted
frame has been *measured* over budget and the rebuilt object is *re-measured*, so an already-fitting
build takes the identical path and is byte-identical. That is what keeps the one bundle known to have
run correctly on hardware a valid acceptance test.

**Why not byte reuse**, which is the obvious fix: on every model the largest *single* allocation
already exceeds the whole budget (489,000 · 1,024,000 · 3,179,520 · 3,326,976 B), so no colouring
makes any frame legal. And an `alloca` has no `free` — 0 `llvm.lifetime`, 0 `llvm.stacksave` in the
emitted module — so no two live ranges are provably disjoint. Moving the storage is sound with no
liveness analysis at all; sharing it would be a guess.

### 2. Every whole-model readout is full-width, so the entire host lane is stranded

*Instrument:* a command-buffer census plus the RoCC model's own source.

All four models emit `output_dtype: i32` on every readout, and **zero epilogue stages are attached on
any of them**. On the full-width path the accelerator writes the raw accumulator, discarding the
scale and activation it computed. So requant, ReLU and pooling cannot offload separately — 100 of
ResNet-50's 119 host regions are behind that one property. Proven on hardware, not inferred: a
capsule declaring `epilogue=['relu']` with an i32 readout returned 126 of 256 outputs negative,
`min = -85`, while its numeric floor and trace both passed.

This is the campaign's headline explanation for **≥93% of the measured window being off the
accelerator**, and it reconciles the two hardware numbers: ResNet-50 is *provably compute-bound in
intensity* (23.01 > 15.882 MACs/byte) yet runs at **1.15% of arithmetic peak**, because the binding
constraint is on neither roofline axis.

### 3. A tolerance declared before the run can still be arbitrary

*Instrument:* the capture's own pair of references, measured after a gate failed.

`atol=rtol=1e-4` was declared up front for SmolVLA, which is the right discipline. Comparing the
capture's **own** two references to each other, **15,740 of 16,000 elements (98.4%) fail that
tolerance** and the worst disagreement is 5.36e-2. The device run duly reported `bad=1600/1600` on
every step — a verdict that says nothing about the device, because a gate that tight fails for any
conforming datapath. This is the repo's cos-0.484 incident in subtler form: there the wrong
*reference* was chosen; here the right reference with an impossible threshold. The floor is now
measured from the capture and a tighter tolerance is refused with the number to use.

### 4. A baremetal console has no float conversions

*Instrument:* the emitted ELF's own output.

The target's `vprintfmt` implements `c s d u x l`. A `%.9g` there prints the specifier **literally**
and then mis-consumes the varargs — 16,000 value lines reading `MERLIN_OUT 0 %.9g` and a header
claiming `elements=-350469331`. The verdict was still sound (the comparison is C arithmetic; the
counters use integer conversions), but no magnitude could be recovered and the run had to be
repeated. Values now travel as IEEE bit patterns, and a render-time scanner refuses any float
conversion — scanning structurally, because `"%.9g"` does not contain `"%g"` and a substring check
passes exactly the specifier that shipped.

---

# PART III — Where NOT to optimize

13 rows carry a `refuted` verdict. They are kept so a later campaign does not re-walk them, and each
names the measurement that closed it. The five that matter most:

- **Byte reuse cannot fix the entry frame** — on all four models, not just the largest. See above.
- **GSIM calibration cannot narrow the cost band.** Width is `peak / slowest_rate`, a per-class
  constant, and `rates_for` keeps the *minimum*, so any new measurement can only widen it. The
  declared compute-class split did the work instead: median band width **2816× → 211.5×**, with
  `n_decided` 81 → 96 and `below_floor` still 0.
- **`full_model_simulation_allowed` is not a toggle.** A literal `False` at 8 write sites and an
  invariant asserted at ~10 read sites that refuse when it is *not* `False`. Flipping it would make
  the work-order extractors refuse the run. The intended path is the probe provider — which the
  telemetry above shows refused 17/17 times because it was never installed.
- **The machine balance cannot be fitted from the corpus.** The same declared byte volume measured
  cycle counts disagreeing by **13.3×**, because they are different emitted programs at one volume.
  A controlled transfer-size series on GSIM measured it instead: **16.000 B/busy-cycle**, 4-point
  plateau, giving a two-sided ridge at 16.000 MACs/byte.
- **An image's medany span cannot be read off the symbol table.** TLS symbols carry offsets *within*
  the thread block, so `buf.2` at `0x40` made a 0.689 GiB image look like 2.689 GiB and produced a
  FAULT that belonged to the measurement. The span is the `PT_LOAD` virtual footprint.

**And the architectural refutation that reframed the whole campaign.** Counting flag polarity across
the phase-2 driver: `global_cost_validated` (9 write sites), `global_speedup_proven` (22),
`full_model_simulation_allowed` (10), `changed_context_validated` (1) and
`timing_calibration_admissible` (7) are **never set `True` anywhere**, and three are actively
asserted `False` at 16 sites combined. `probe_relevance.classify_probe_relevance` sets
`global_cost_validated: False` at *construction*, and its docstring states the intent: *"this
function never promotes a local sample into a global performance verdict."*

So the global driver is an **authoring and structural-validation loop**. "It proved a global speedup"
is not a state that code has. `promotion_blockers` has no clearing logic because nothing could clear
it. The consequence is not a bug to fix but a design decision to respect: **the measurement loop
belongs outside that driver**, which is affordable precisely because the instruments are now real.

---

# PART IV — What the campaign produced

All three fleet binaries link. Each frame pair is `clang -fstack-usage` before and after the static
arena; each blob was verified byte-for-byte against its capture rather than merely sized.

| model | ELF | args | frame | const blob | layout | gate |
|---|---:|---:|---:|---:|---|---|
| resnet50 | shipped bundle (ran on FireSim) | 393 | 23,406,400 → **784** | 26,563,328 B, byte-identical to that bundle | 0.25 GiB near | all-1000-logit exact |
| **smolvla flow_denoise** | **505,790,424 B** | 1,163 | 99,897,984 → **496** | 505,299,072 B, 809/809 unpadded byte-exact | 0.689 GiB near | 10-step trajectory, contract-derived, digest-verified, tolerance derived |
| **tiny_llama** | **1,300,782,520 B** | 825 | 48,976,448 → **9,152** | 1,298,638,208 B, 358/358 unpadded byte-exact | 1.028 GiB near **+ 1.209 GiB blob at 0x200000000** | per-token top-1 over 8 rows, execution-scoped |

**How the third one was made possible**, since it had been recorded as blocked on compile wall-clock
and on its golden: the wall-clock cause was the 48.9 MB frame the preflight rejected, and the golden
had an *execution*-reference generator needing no model repo. Its image spans 7.209 GiB, so the blob
cannot be linked beside the code at all — it sits at a fixed absolute address reached by a
compile-time literal, with **zero relocations against its symbol**, and the linker's placement
matches that literal exactly. Two further limits surfaced only at this size: `crt.S` reaches `_init`
with a ±1 MiB JAL and this kernel's `.text` is 1,061,906 bytes on its own, so link order decides
whether the image links; and the near region is 1.028 GiB, a **WARN** at over half the window.

## The correctness verdicts, stated as they stand

SmolVLA's first graded run failed on a tolerance that was demonstrably impossible (Part II, finding
3) and is being re-run against the derived one. tiny_llama's graded run **fails**: `bad=256000`, and
all 8 token rows report the same argmax while the reference has 8 distinct ones. Its harness was
ruled out first — all 825 pointers match the ABI order exactly, no mutable tensors overlap, and two
independent references agree on the row structure — so the divergence is real and localized to the
emitted program. A diagnostic dump run is decoding the values now. **The per-token ranking is what
made this visible**; a single global argmax over 256,000 logits agrees whenever the largest logit
anywhere lands in the same place.

---

# PART V — What this campaign cannot decide

Five live blockers, each outside this lane, with what blocks it:

1. **Attention quantization.** All 66 off-unit contractions are f32 attention `batch_matmul`, and
   the captures carry no attention quantization parameters at all — 0 `quantize_per_tensor`, 0
   `dequantize_per_tensor`, 0 `choose_qparams`. Quantizing them in the compiler would mean inventing
   scales. **Fix the export.** This is the single largest offload gap for two of the three models.
2. **`ATTENTION_PV` pricing.** Zero programs on this tree emit the opcode — the compiler lowers
   attention PV to a plain `MATMUL`. No amount of simulation prices it; only a compiler change does.
3. **Epilogue attach and readout narrowing.** Another session's active lane, six commits in flight.
   Their own refusal ledger names the blocker in its own words:
   `accumulator_scale_is_not_proven_identity` — the unit's `acc_scale` is one f32 per command while
   the graph's scale is a per-channel affine, so native formation only fires when there is
   effectively no scaling to do. The enabling change is per-output-channel mvout.
4. **tiny_llama's independent golden.** `make_w8a8_independent_golden.py` refuses because that
   loader is session-shaped (prefill+decode) while the capture on disk is a single forward, so the
   *arithmetic* cannot be decided from this tree. The execution reference was producible and is what
   the gate uses, with that scope declared in the gate itself.
5. **A FireSim run with the full 8-counter bracket.** Everything downstream is built and tested;
   `attribution` refuses the existing 3-of-7 reading by design, because the four missing `MAIN_*`
   counters are exactly the overlap terms. The FPGA is in use by another person.

## The instrument stack, and what each one is licensed to conclude

| instrument | cost | decides | does NOT decide |
|---|---|---|---|
| `work_volume` + `offload` | ms | routed MACs, contractions on/off unit, exactly | anything about time |
| `lane_cost` | ms | bytes by role and dtype | whether they are moved twice |
| `compose_estimate.band` | ms | a 211.5×-wide *elimination* band | never certifies a candidate |
| `movement_balance` | done | ridge at 16.000 MACs/byte, two-sided | a specific program's bandwidth |
| `attribution` + `envelope` | needs 1 FireSim bracket | per-bucket gap and its family, incl. `NONE` | refuses the 3-of-7 reading it has |
| GSIM per block | 13 s + cycles/11,000 | real cycles, no FPGA | whole-model totals |
| Spike whole model | ~2.5 min | correctness gate, relative host metric | **never** cycles as performance |

The last row is load-bearing: Spike prices every RoCC command at one cycle, so it systematically
flatters offloading. Measured: a retarget nearly doubled mesh instructions (14,864 → 28,499) and
moved the Spike metric by 0.2%. **The proxy degrades exactly as the optimization succeeds**, which is
why every ledger row records which instrument decided it.

---

# Reproducing this

```
gen_phase2_analysis.py --target gemmini \
  --ledger .../phase2_instrument_ledger_20260908/optimization_ledger.json \
  --out .../phase2_decomposition_20260908
```

- `campaign_analysis.json` (beside this file) — the machine-readable analysis, with the
  availability ledger every figure above is paired with.
- `../phase2_decomposition_20260908/decomposition.json` — all 56 attempts in full, plus the
  by-instrument index.
- `../phase2_decomposition_20260908/optimization_ledger.txt` — the flat table.
- `../phase2_decomposition_20260908/README.md` — the change-level decomposition this document
  summarizes: each compiler change, its measurement, and the mutation that proves its test.
