# Larger-model compiler gate demonstrations — 2026-10-29

All fourteen compiler phases passed on the recorded v3 source cut: **34 accepted changing
boundary checks**, same-options ordinary/gated LLVM equality, and zero unverified
observed boundaries. Twelve complete compiler-to-native cases pass; the two stored
TinyLlama native cases hit the demo wrapper's 300-second object-build deadline.
All owned processes were reaped. A finite-budget correction and two-case retry
are being prepared; the failed attempts retain zero native qualification credit. Every case selects all ten checks and must establish an actual
changing Q/DQ or MACC boundary. Absent and unchanged transformations earn no proof
credit.

V3 started from `fbe68e88846f109c792f7e2eb0aa43be99d858a1`, which matched the
remote at the pre-run fetch.
That source includes the colleague's complete-result guards, typed source-refusal
classification and generic-query diagnostic fixes. The fresh run additionally
uses the reviewed finite tensor intake policy described below. Exact working
bytes, helpers, tools and inputs are pinned in the retained source cut; these
records do not pretend the dirty run was a clean checkout of its base commit.
Lean relations, `SPEC_LOCK`, producer algorithms and approval flags are unchanged.

## Complete model inputs and actual gates

Each row uses both xDSL and textual routes, with two deterministic native input
scenarios per route. [INPUTS.json](INPUTS.json) pins source/provenance and complete
interface totals; [RESULTS.csv](RESULTS.csv) records terminal per-route results.
The [summary](summary.json) keeps exact retained artifact references and coverage.

| Model input | Original IR operations | Additional changing gate required | Compiler / native status |
| --- | ---: | --- | --- |
| Consistent small-Llama recapture | 1,175 | Q/DQ, including per-channel DQ with i32 zero points | Both routes pass / both routes pass |
| Existing M0 small-Llama capsule | 1,280 | Q/DQ | Both routes pass / both routes pass |
| Existing M2 MicroViT capsule | 818 | Q/DQ | Both routes pass / both routes pass |
| Existing M4 VLA-denoise capsule | 1,688 | Q/DQ | Both routes pass / both routes pass |
| Constructed twelve-layer f16 MLP | 123 | Draft MACC at all twelve live contractions | Both routes pass / both routes pass |
| Stored ResNet50 forward IR | 2,574 | Q/DQ at its actual boundary | Both routes pass / both routes pass |
| Stored TinyLlama forward slice | 10,134 | Q/DQ at its actual boundary | Both routes pass / native build timeout |

The first four sources and both larger stored sources are unchanged. M0/M2/M4
are scaled model capsules; M4 is not the original large SmolVLA model. Stored
ResNet50's metadata contains a checkpoint declaration **and** random/synthetic
capture settings (`M2M_RESNET_RANDOM=1`); it supplies no pretrained-accuracy claim.
Stored TinyLlama's input provenance is undeclared. Its complete interface has
203 arguments, **1,296,933,056 logical input bytes** and an `1x8x32000xf32`
output. ResNet50 has 109 arguments, 26,215,264 logical input bytes and 1,000
output values. Those sizes include every runtime weight/scalar argument; they
are not a recovered checkpoint or a peak-memory estimate.

The MLP is explicitly constructed, rather than recaptured or pretrained. It has
batch 8, width 256, twelve connected runtime-weight layers and 31 final logits.
Each +0-initialized half-format matrix product feeds a separate f16 bias add and
ReLU, except the final layer's logits have no ReLU. All contractions contribute
to the output. The independent reference performs every reduction in order with
f32 multiply/add, truncates the contraction once to f16, then applies the separate
half-precision operations. It retains complete outputs and verifies distinct
batch rows at every layer. The ordinary and checked compilers both explicitly
select the draft accumulation policy; comparison does not claim original half
accumulation equals that policy.

## What is checked end to end

The portable runner starts from each complete original MLIR interface, compiles
ordinary and all-selected versions with the same transformation options, and
audits actual requests, raw before/after endpoints, typed source-support results,
full-function composition and accepted receipts. A required pass must actually
change the graph and receive complete evidence. A saved status cannot substitute
for those joins. Recorded v3 counts are DCE **6**, Q/DQ **12**, CSE **14** and MACC **2**
whole-boundary jobs; each MACC job additionally covers twelve live sites.

Only after the compiler/receipt audit passes does the runner link actual LLVM
products through Merlin's existing `build_host_shared`, unchanged `mlir_runtime.c`
and `-lm`. It binds selected tools, objects, runtime sources and link commands.
It compares every public output byte for seeds `20261029` and `20261030`, checks
all original inputs unchanged, checks buffer guards and borrowed descriptors,
and requires finite floating outputs. The MLP additionally matches the complete
independent policy reference. Captured graphs use recorded generated inputs:
positive finite float parameters/scales, small signed i8 data, zero integer zero
points/gather indices and true booleans. They are not recovered trained weights
or calibration data.

These are full forward observations, rather than isolated local rewrites. They
remain finite native regression evidence. Case walltime includes input binding,
compilation, proofs, audits, native builds, comparisons and cleanup; it is not
kernel latency or a speedup measurement. Source-framework capture qualification
is a separate optional interface and is not inferred from this run.

## Why other gates do not fire on these models

All ten checks are integrated and selected. Validation selection does not enable
transformations. The original seven sources have **no source SCF loops or private
memrefs**. No artificial probes are added to manufacture coverage.

| Family absent here | Actual reason and next integration route |
| --- | --- |
| Unrolling and tiling | No factor/tile is selected and no source SCF loop exists. The current hook runs before tensor lowering creates backend loops. A real post-bufferization/SCF boundary needs its own source admission for generated bodies; floating reductions, carries and multiple events exceed the narrow generic singleton model. |
| Private buffer reuse | No allocator model is selected and no original private storage exists. A later bufferization boundary must retain real lifetimes, initialization and allocator premises. Current proofs cover bounded static rank-one/two integer storage, rather than arbitrary floating model buffers. |
| Configuration elision/hoisting | This host tensor route does not enter the selected ISA/Kernel scheduling consumers. A target model demo must use the real mapping owner and original ISA/effect contracts. Configuration hoisting is not general floating LICM. |
| Host graph fusion | No fusion contract/groups are selected. The current profile accepts integer pointwise graphs, while these entries contain floating operations and reductions. A broader contextual/typed bridge is needed for those real stages. |

Backend-generated loops and allocations do not automatically revisit the early
source hook. The existing per-pass positive/negative qualifications remain in
[the test inventory](TESTS.md); this campaign does not relabel them as large-model
coverage. Unchanged boundaries also receive zero positive proof credit.

## Bounded claims and limits

- Q/DQ remains refinement under the **proposed source model**. MACC remains
  conformance to the selected **draft f32 policy**. Neither gains numerical
  approval, general PyTorch/IEEE equivalence or unconditional quantizer conversion
  success from this campaign.
- The required tensor claims are typed refinement under the declared primitive
  interpretation and index64. Optional source-definedness and unrestricted
  strengthenings keep their actual individual verdicts; an abstracted strengthening
  is not established. Typed chain joins do not prove intervening ordinary passes.
- Ordinary LLVM lowering, native runtime/allocator behavior and hardware execution
  stay outside the Lean pass boundary. Full output comparisons do not constitute
  a whole-model or whole-pipeline theorem.
- The tensor/QDQ/MACC host intake permits 4 MiB raw source, 32,768 walked
  operations, 64 MiB aggregate decoded dense storage and 8,388,608 dense elements.
  Dense limits are charged before expansion. These are finite host policies,
  not new theorem premises. Legacy loop/memory budgets stay unchanged. Parser
  resource failures still block; no admitted failed target becomes unverified.
- Native generation/copies use a 2 GiB logical-input and 64 MiB logical-output
  guard. It bounds logical arrays, not peak RSS or compiler workspaces. The runner
  uses at most four nice-19, single-thread CPU lanes, one Lean worker per case,
  finite owned timeouts and confirmed cleanup. The corrected native observer
  allows 1,200 seconds for the whole shared build, leaving normal codegen's
  900-second per-command limit intact; the case maximum is 5,400 seconds.

The distinct external `tiny_llama_int8_consistent` recapture has an original
non-dominating scale operand; it is not the stored TinyLlama source above. The
original half-format large SmolVLA source exceeds the 4 MiB intake budget and
contains broader tensor-carrying conditional loops. Neither excluded source was
repaired, downcast or reduced to force proof credit.

## Retained iterations and tests

The earlier `dbab423b0` ten-case compiler phase passed 25 checks, but eight captured
native cases failed because its isolated test linker omitted `rsqrtf` or
`memrefCopy`. The separate native-only v2 repeat passed all ten cases using the
normal runtime and re-audited those prior proof inputs; it ran no fresh Merlin
pass proofs. Those failures, helper snapshots and source cuts remain immutable.
V3 repeated actual compiler checks and native observations
rather than importing their earlier success. The earlier ResNet
resource refusal is likewise retained; the finite intake correction is tested
against the same unchanged source.

[TESTS.md](TESTS.md) gives suite locations, exact retained cohorts and reproduction.
Current source/controller host controls pass 217 tests. The latest canonical
harness cohort passes 58 tests after the deadline correction; the earlier 53-test
cohort and 18-test native-cap refresh retain their own cuts and overlap. Layout/access checks pass
111 and skip two unavailable release/security checks. These counts overlap and
are not summed. Ruff and documentation checks pass. The structure check retains
16 unrelated inherited findings; no full-repository green result is claimed.

## Reproduction

Configure the compiler Python, Lean SDK and host tools as in the
[gate guide](../../../../docs/guides/lean_pass_gate.md). Use fresh output directories
and available CPU IDs:

```sh
python merlin/tests/ir/larger_model_prepare.py --out out/build/model-demo-inputs \
  --include-large-captures --recaptures /path/to/original/recaptures
nice -n 19 taskset -c 28-31 python merlin/tests/ir/larger_model_campaign.py \
  --manifest-list out/build/model-demo-inputs/CAMPAIGN-MANIFEST.json \
  --out out/build/model-demo-run --jobs 4 --cpus 28..31 --case-timeout 5400
```

Preparation defaults to eight routes: three tracked capsules plus MLP, each on
both routes. `--include-large-captures` adds stored ResNet50/TinyLlama, giving
twelve. The optional original small-Llama recapture adds two more, giving fourteen.
Unset `MERLIN_RECAPTURES` and omit `--recaptures` when unavailable. A missing
explicitly selected recapture fails rather than silently skipping.

- [Portable model preparation](../../../../merlin/tests/ir/larger_model_prepare.py)
- [Parallel compiler/native campaign](../../../../merlin/tests/ir/larger_model_campaign.py)
- [Boundary and receipt audit](../../../../merlin/tests/ir/larger_model_gate_demo.py)
- [Connected MLP and independent reference](../../../../merlin/tests/ir/larger_model_mlp.py)
- [Complete native observation](../../../../merlin/tests/ir/larger_model_native_demo.py)

Raw model, LLVM, proof, process and output-array records remain local under
`out/build/larger-model-gate-demo-20261029/`. The final source cut and independent
review must match before terminal credit is published. Saved reports provide
test evidence and reproducibility information, never live delivery authority.
