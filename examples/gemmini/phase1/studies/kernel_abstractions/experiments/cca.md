# Experiment 1: learning a projection-family rule from expert code

**Question:** Does actual lifting and CCA comparison help an agent reach a useful
optimization sooner, with performance that transfers across shapes?

Both authors evolve the full graded Fresh6 backend and identical current
construction tools. Control gets raw expert kernels and trusted measurements;
treatment additionally gets captured lifting, comparison and resulting
experiment feedback. Neither gets the oracle implementation or winning settings.

## Seed, public development and frozen sweep

Keep complete Fresh6 tree
`8dd194ae90a076b9990341cec4cd4a71f713a9654bee529257f668503f310ea9`,
including working passes, library, notes and rejected attempts. New conversations
start a fresh clock. Historical evidence is not a new certificate; do not rerun
the established Fresh6 suite.

Use `inputs.py --profile projection-family-v2`. Public cases, in **M × K × N**:
16×2048×2048, 8×2048×2048 and 16×1024×2048. Inputs are signed INT8; output is
complete raw INT32, without bias, scaling, rounding or activation. Tell both
agents to improve this family through source/resource predicates and safe
fallbacks, rather than selecting schedules by benchmark name or exact triples.

After freezing the compiler and rules, evaluate 14 one-axis changes from the
16×2048×2048 anchor:

| Changed axis | Frozen values | Fixed axes |
|---|---|---|
| M | 4, 24, 32, 48, 64 | K=N=2048 |
| K | 512, 1536, 3072, 4096 | M16,N2048 |
| N | 512, 1024, 1536, 3072, 4096 | M16,K2048 |

Together these are 17 cases, including the three public ones. Keep M32 hidden.
They span smaller, intermediate and larger sizes, including partial M tiles.
M1/M17/reduction tails are later stress tests. Label seed-known cases as transfer
checks rather than claiming wholly unseen training history.

## What is lifted

The C source is the expert TinyLlama `mm_o` function extracted unchanged from
the supplied archive. It is one representation of the expert kernel; parsing
its syntax alone is insufficient. Both arms receive the same approximately 217 KB
bank: that function, necessary headers/ISA protocol and trusted metric card.
There are no complete archives or exact hidden-shape expert kernels.

The existing bounded capture recorded the original function's 37,029 accelerator
issue/fence commands. The ordinary lift endpoint now replays that stream through
the selected observer and ISA protocol and recovers a typed region/dataflow:

```text
A: i8[16,2048]       Y: i32[16,2048], no epilogue
packed W[n//64,k,n%64] = logical W[k,n]
transfers: A32,768 bytes; W4,194,304 bytes
reuse: full A serves all output-column panels
coverage:16,384 compute tiles,128 output tiles, complete ordered K
```

The read-only `cca-evidence/` selection binds source/function-body, raw stream,
facts, contract and trusted observer hashes. The same lift request on the raw
reference automatically selects this view. Changed sources, wrong contexts or
incomplete coverage are refused. It grants no arbitrary C execution or
author-selected observer. Report capture preparation and replay costs separately
and in total authoring cost.

Issue/dataflow lifting does not establish accelerator arithmetic, hardware
completion, overlap, generalized loop proofs or elapsed cycles. Supplied expert
physical measurements retain their original binary/ROI/cache/sample authority.

## The automatic loop we need to establish

1. Top down, bind math, layouts, numerical behavior and every live output.
2. Lift the reference stream and compare supported axes with actual emitted
   candidate views. Keep incompatible contracts and absent witnesses explicit.
3. Feed differences to the EXISTING experiment planner. Generate mechanism
   hypotheses and library-contract routes, or a construction task when a needed
   pass/adapter or goal-bound alternative is absent.
4. The agent implements a general alternative and submits ordinary A/B checks.
   Reopen the emitted result to establish whether the requested mechanism happened.
5. Keep or reject using exact outputs and same-domain complete-call cost. Freeze
   the chosen rule before the 14 tests; there is no hidden-shape retuning.

Observed function-body counts are loads 4,128 versus candidate 18,432,
configurations 4 versus 4, and fences 1 versus 2,083. The candidate timed wrapper has
one additional fence; retain the distinct scope. Equal configuration work must
not generate an optimization task merely because its fraction of all commands
changed. Candidate operand traffic and residency remain UNKNOWN until separately
witnessed; excess load commands do not establish their cause.

The tool must turn those facts into useful experiments, not merely print a
report. Missing explicit mechanism-to-knob bindings remain construction gaps;
we must not fabricate runnable alternatives or claim declarations are consumed.
Both authors may add new passes and scheduling consumers. The treatment supplies
structured evidence and routes, not a completed optimization.

Ordinary linked baseline/variant trials also compare the evaluator's emitted
program identity. Changed settings with identical ELF and counts, under matching
capsule/model/domain/scope, produce a requested-but-no-emitted-change construction
gap. This is advisory negative evidence. A different ELF still leaves the claimed
mechanism unknown; absent or incompatible witnesses remain unknown. The offline
Fresh6 retention trial motivates this check: its setting was read, but an existing
construction guard returned the incumbent and left emitted work unchanged.

## Fair execution and evidence

Reuse `prepare_runs.py --study cca --seed-continuation PATH`, the existing
sandbox/brokers/evaluator and `--wall-s 1800 --replicas 1`. Keep equal 200% quotas in four shared cores, an optional fifth
for preparation, and one evaluator per author. Match model/effort. Both arms
have identical current library bytes; a common migration is not agent credit.
The base-config `cca_lifting: {path,sha256}` selects the captured view only in
treatment; optional `cca_guidance` adds its human-readable summary.

Screen on the anchor, then check promising rules on the other public cases.
All three have correctness/performance feedback. A selected case may pass while
the full suite is incomplete; inspect case rows. Do not repeat the old full suite.
Freeze one final compiler, dependencies, rules, settings and timestamp. Measure
all 17 supported cases on FireSim under matched preparation/cache/completion
conditions, retaining ordered samples, failures and regressions. Expert/oracle/
SW/FSM comparisons require exact supported contracts; otherwise leave a reasoned
blank. Provisional model estimates shortlist and never substitute for timing.

Measure time to a correct improvement, failed/invalid attempts, tokens/CPU/time,
consumed CCA tasks, actual emitted mechanisms and per-shape cycles. Retain
intermediate candidate evidence: equal final performance can still expose an
effort difference. Report M/K/N curves separately. One pair is diagnostic;
replicate promising effects. The experiment must allow CCA to fail its hypothesis.
Use the same external evidence rule for both authors: a CCA task ID alone is not
a useful hypothesis or successful attempt. Count proposed, requested, emitted,
refused, unchanged and unknown attempts separately, using existing request records.

## Qualification and current limits

The current-tool anchor checks all 32,768 outputs and reproduces the original ELF
and 1,380,088 provisional cycles/model/domain/scope. New public M8 checks all 16,384
outputs on pristine Fresh6; no timing was requested. The new lift/compare replays
the actual capture and retains unknown candidate mechanisms. A fresh exact
family/lifting launch configuration and smoke test supersede the earlier
one-anchor setup. No author or FPGA study has launched. Useful autonomous
experiment selection, generalized optimizations and physical wins remain study
outcomes to demonstrate, rather than established benefits of this tooling.
