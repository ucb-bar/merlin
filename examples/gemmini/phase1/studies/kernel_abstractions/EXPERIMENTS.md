# Small experiments evolving Fresh6

We test whether CCA, reusable transformations and checked scheduling help an
agent improve an established compiler. Separate two questions: whether a
mechanism produces faster code, and whether its tooling helps an agent build it.
Every agent experiment measures correctness, effort, performance and frozen-shape
transfer. A failed experiment remains a result.

Detailed component designs: [CCA](experiments/cca.md),
[library](experiments/library.md), [scheduling](experiments/scheduling.md).
Actual immutable configurations and results take precedence over older proposals.

## Starting point and controls

Start new conversations from the complete graded Fresh6 compiler, its custom
backend, working passes, reports, previous attempts and handoff. The seed is
`$TASK/.tmp/fresh8/seed/fresh6-frozen`, tree SHA256
`8dd194ae90a076b9990341cec4cd4a71f713a9654bee529257f668503f310ea9`.
Do not ask agents to recreate that work. Record the effective initial source after
its explicit dependency migration as well as the historical seed identity.

Fresh6 supplies 1,026 compiler members. Controls retain that inherited machinery,
including scheduling IR. Treatments receive an explicit reviewed addition, with
its dependencies and API documentation. They do not receive unrelated implicit
SDK upgrades. Both agents can write general passes and edit their compiler.
These are additions-since-Fresh6 comparisons, not a compiler with no abstractions
versus a compiler with abstractions.

Match references, hardware facts, ordering laws, numerical contracts, feedback,
author model/effort, CPU quota and time within each comparison. Isolate workspaces,
conversations, private feedback and other arms' outputs. Keep operator oracle
implementations and winning settings outside author grants. A neutral API sketch
is treatment assistance; an oracle adapter or solved schedule is not supplied.

Reuse prior numerical results only through existing owners that prove matching
code, inputs and judges. The old certificate store is absent: no fabricated cache
hits. Known seed checks can be deliberately skipped under the user's instruction;
show those cells as not newly measured. First correct changed code is distinct
from an already-correct seed. Unsupported baselines remain blank with a reason.

## Agent comparisons

| Study | Control | Treatment | Question |
|---|---|---|---|
| CCA | Raw expert references and common current tools | Same plus lifting/comparison | Does structured reference evidence help find a useful general rule sooner? |
| Convolution library | Inherited tools; own composition | Reviewed convolution construction APIs | Does reusable construction reduce errors/effort and handle capacity changes? |
| M-panel scheduling | Inherited construction and scheduling IR | Current checked M-panel toolkit | Does added checked construction help improve the family? |
| Completion scheduling | Inherited tools; own implementation of the same rule | Exactly one checked completion consumer | Does the helper reduce effort to construct a legal completion change? |
| Combined dense | Four arms: CCA off/on × construction bundle off/on | Same within-factor inputs | Do CCA and the transformation/scheduling bundle help each other? |

The first pilots used 30-minute authors. Current replications use 60 minutes,
200% CPU quota per author, one evaluation worker and no author helpers. Keep
pilot and replication results separate. Each pair has a four-CPU pool; three
pools use at most 12 reserved CPUs. The combined study has four authors and an
eight-CPU pool. The existing reservation owner and dispatcher manage resources.

The M-panel study changes a dependency closure, not just dialect syntax: it adds
12 and updates 85 compiler members. The focused completion study instead adds
only `merlin/compile/scheduling/dense_panel_epochs.py` (1,026 versus 1,027 members).
Both focused-study starting compilers emit identical interface, command buffer
and LLVM. CCA is off in the library and both scheduling comparisons.

## Training and frozen shapes

Dense dimensions are **M × K × N**. The standard 17-case projection panel has
three public cases: 16×2048×2048, 8×2048×2048 and 16×1024×2048. Freeze before
revealing the remaining 14. Change one axis at a time, keeping the others at the
16×2048×2048 anchor:

- M = 4, 8, 16, 24, 32, 48, 64.
- K = 512, 1024, 1536, 2048, 3072, 4096.
- N = 512, 1024, 1536, 2048, 3072, 4096.

Deduplicate the anchor. M32 is frozen. The M-panel and combined studies make M24
public instead of M8; their own pinned rosters define this swap. Compare effects
within a study rather than pooling different curricula. These are one-axis
sweeps, not a Cartesian product. Unchanged Fresh6 fallbacks are explicit.

Convolution has 18 cases. Its public shapes are SAME 3×3 B4/H28/C128→128,
B2/H28/C128→128 and B4/H21/C128→128. With other axes at the anchor, sweep
B={1,2,4,6,8}, H={14,21,28,32,35,42}, Cin={64,96,128,192,256} and
Cout={64,96,128,192,256}. Use NHWC, stride/dilation 1, padding 1, original INT32
bias, identity scale and INT8 saturation without activation. Input and output
spatial sizes match. Preserve the original halo, wrapper and quantization.
Time the full batch and disclose one B4 compiler call versus four expert B1 calls.

Attention Q×K trains on 32 query heads, four KV heads, 64 channels per head,
16 query positions and 16 context positions. Freeze, then vary query count to
8/24/32, context to 8/24/32, query boundary 17, context tail 19 and decode query
count 1: ten cases total. Keep exact strided inputs and raw INT32 outputs.
P×V is a separate zero-shot transfer panel with its original strided values,
scale 1/128, nearest-even rounding and saturating INT8 output. It is not the same
numerical contract as Q×K. Generalization can fail without a numerical failure:
an unchanged fallback is not a learned optimization.

M1 dense decode, reduction/channel tails, crossed-axis shapes and another actual
hardware configuration belong to separately frozen follow-ups. Do not imply
these current panels establish those capabilities or that every shape was absent
from the seed's earlier graded history.

## Two focused scheduling questions

**1. Does checked completion enable a useful optimization?** No new authors.
Freeze the qualified Fresh6 algorithm, tiling, segmentation, storage, addresses,
configuration and data-command issue order. Compare conservative completion with
a checked K-stream rule. Keep readout, output-block and final completions.
Unknown ordering premises retain the original boundary. Prove the emitted change
by comparing actual LLVM after removing only recognized fence statements; retain
unchanged and confounded cases separately. Check every output and measure both
variants under the same physical protocol.

The prior isolated anchor has 2,084→36 dynamic fences and modeled cycles
1,380,088→1,156,159. Its controlled physical follow-up is now complete:

- All 34 programs pass; 103 timed calls and 34 separately checked warm-ups.
- Anchor: 1,544,232→1,332,662 cycles, **13.7% fewer**.
- All 14 held-out shapes: T/C geometric mean **0.88741**, including four unchanged
  fallbacks. Thirteen emitted pairs differ only in completion statements.
- M24/M32/M48/M64 have fully identical LLVM: no optimization-transfer claim.
- Order drift is **0.1874%**, within the declared 1% bound.

This demonstrates the frozen optimization's efficacy. The epoch change is a
checked kernel transformation following consumed scheduling IR, not a new dialect
operation. Selected ordering laws are conditional target premises; numerical
checks and physical measurements remain separate from an RTL certification.

**2. Does the helper help an agent construct that optimization?** Two new
one-hour authors ran 15:50:49–16:50:49 UTC on CPUs 52–55. Same seed, laws,
fixed algorithm/data order, objective and public tests. Only treatment gets the
checked consumer and a neutral API sketch. Neither receives oracle code, winning
settings, counts or a solved law instance. Measure time/tokens to a correct
emitted change, attempts, repairs, legal applicability and frozen performance.
A final speed tie can still accompany a construction-effort benefit. Audit route
adherence after freeze; do not repair or exclude a violating author.

The treatment first produced the same correct 36-fence anchor pattern at 4.26
minutes versus 11.10 for control. Completed-response fresh input plus output
was 104,496 versus 161,984 tokens at that milestone; these are recorded lower
bounds. Whole-hour treatment usage was instead 2.5% higher. Both final compilers
pass all 17 numerical cases, but control changes completion on all 17 while
treatment changes 13 and leaves M24/M32/M48/M64 unchanged. Its adapter accepts
one pipeline result type after an inherited transformation can return another.
This is a source-supported composition gap; exact per-case runtime refusal
records are unavailable. Keep construction effort, applicability and physical
performance as separate outcomes.

The final physical panel passes all 68 programs, 205 timed calls and 68 separate
checked warm-ups. Order drift is 0.1123%, within 1%. Treatment/control cycles are
1.03775 across 14 hidden shapes: treatment is 3.8% slower overall, with 10–15%
regressions on the four larger-M fallbacks. The other 13 final LLVM and object
pairs are identical; their small timing differences are not optimization gains.
Passing the first/last order check does not eliminate all within-image variation.

## CCA and references

Supply only references relevant to the bounded task, identically within a pair.
Supplied expert metrics are trusted targets with their original source, shape,
quantizer, timing and preparation scope. Preserve them rather than rebinding a
nearby score to another contract. Source-based dense studies use the expert
`mm_o` function and required headers/ISA information. Complementary-reference
studies also supply `mm_down`. Machine-code-only studies supply identical complete
disassembly and mathematical/layout contracts without expert C source.

Top down, bind the operation, layouts, fusion, quantization and every live output.
Bottom up, decode ISA operands and recover movement, reuse, configuration and
producer order. Compare only supported common axes with the candidate's actual
emitted code. Choose or create a general transformation, lower it through the
existing library/scheduler, then verify that the intended mechanism occurred.
Counts and source intent are diagnostics, not proof of residency, overlap or time.

Complete function extents matter: the original QK listing was truncated at a local
label (100/438 bytes). Corrected extraction covers all 145 instructions; its new
replication remains separate. Both original attention arms already had complete
expert C. Complete ISA alone does not solve candidate semantic correspondence.
Current QK comparison still lacks checked multi-product/subview binding; old CCA
comparisons exposed only 1/59 axes. Preserve UNKNOWN instead of inventing a lift.

Common protected linked-code readback and artifact retention were selected
equally in the completion pair. Real calls exposed a copied-versus-original ELF
path join defect despite retained valid products. Prospective fix `598c4c6c9`
reopens four exact retained case chains in an independent read-only replay. It is
not injected into running authors or older CCA runs. Preserve original refusals
and prove the source→issued request→native code chain before comparison.

## Other queued studies

- **Frozen convolution mechanisms:** disable legal mechanisms in the existing
  frozen compiler and verify what changed. Array reuse is separable; single-row
  grouping changes several mechanisms. A refusal is retained, not a legal toggle.
- **Storage budgets:** freeze policies, reduce admitted scratchpad/accumulator
  windows on the same RTL, then check retiling, phases, correctness and fallback.
  This tests resource adaptation, not execution on another hardware design.
- **QK→PV transfer:** run frozen QK-authored rules on exact PV contracts without
  retuning. Identical final code establishes no learned transfer.
- **Combined system:** the four-arm factorial identifies CCA's interaction with
  a transformation/scheduling bundle. It does not identify separate library-by-
  scheduling effects or justify combining unrelated convolution and dense scores.

## Lean execution and evidence

Reuse `prepare_runs.py`, `run.py`, `launch.py`, the existing sandbox, brokered
self-check, evaluators, metrics observer and guarded FPGA wrapper. No new author
runner or whole-model cold preparation. Before clocks start, validate pins,
explicit dependency selections, initial source emission, common public inputs,
no-author sandbox preflights and cross-arm/private/process access denials.

Freeze one compiler, dependencies, rules and settings per arm before hidden
evaluation. Compile without retuning; check every output. Use the model to
shortlist, then compare final variants physically under matched contracts,
preparation, cache policy and timing boundaries. Preserve ordered samples,
failures, unknown timings and regressions. Hardware repetitions and shapes are
not independent author replications.

Historical physical images clone each slot's kernel at a different text address;
several fail the 1% first/last order control. Do not discard inconvenient samples
or claim small advantages. The two new scheduling studies explicitly reuse each
program entry and record one checked warm-up separately before three timed rounds.
All resets and order controls remain. The first mechanism panel passes; this is
not a causal proof that warm-up or entry reuse cures every source of variation.

FSM, restored software WS and expert rows require qualified exact contracts and
fixtures. A verified output-view bridge can support a separately labeled semantic
comparison; do not silently fill a strict byte-identical fixture column. Keep
cross-invocation and same-image comparisons separate. Runtime consumed-ELF and
deployed-domain custody remains UNKNOWN where records are unavailable, even when
source-bound raw observations and all numerical checks are complete.

| Area | Evidence per arm |
|---|---|
| Reproducibility | Exact source/dependency/tool/reference/test/model/settings/CPU/budget identities. |
| Progress | Actual initial/30/60-minute snapshots; 120/240 not applicable to one-hour runs. |
| Correctness | Every case, final coverage, source-coherent coverage AUC and unknown observation intervals. |
| Convergence | First correct changed executable and subsequent correct performance progress. |
| Tokens | Native lead/helper input, cache, output and reasoning where reported; no double counting cache. |
| Rework | Observed compiler, legality, numerical and ordering failures; repairs and their durations. |
| Abstraction use | Consumed schedule/emission witnesses and exact executed fallback identity. |
| Bottlenecks | Calls, queue latency, stage timings and oracle waits without summing nested durations. |
| Interruptions | Capacity failures, retries, timeouts, continuations and lost exposure. |
| Performance | Per-case model estimates and separately qualified ordered physical measurements. |
| Generalization | Frozen axis sweeps, failures, regressions and unchanged routes. |

Unavailable counters stay UNKNOWN. Milestone tokens are completed-response lower
bounds, not complete billing. Do not infer correctness between observations or
count source edits as failed attempts without evidence. Operator preparation,
frozen mechanism tests and author effort are separate costs.

## Current interpretation and records

The strongest demonstrated agent result is convolution capacity transfer: both
one-hour authors pass 18/18, and the library arm is 1.33× faster across 15 hidden
shapes, driven mainly by two capacity cases. It wins four of the 15. Its 8.80×
improvement over its own Fresh6 start is a different baseline. Earlier 30-minute
controls had genuine selected-target failures; preserve those historical results.

CCA has not demonstrated a consistent performance or effort benefit. Updated
dense, machine-only and complementary-reference pairs are near ties with order
warnings. Corrected QK control passes 10/10 while treatment passes 6/10, retaining
four authored query-size compile crashes. PV transfer leaves identical programs.
The original M-panel scheduling pair also has near-tied final performance; its
control already contains scheduling IR. The new completion-only physical result
is positive; the separate construction pair has an earlier treatment anchor
milestone but broader final control applicability and 3.8% slower hidden physical
performance for treatment.
The combined
four-arm study passes all 17 shapes in all arms, with all 136 physical cells
correct. Its 1.069% order drift exceeds the declared bound; retain its raw
contrasts without claiming small physical advantages.

For the paper figures, omit CCA comparisons at the user's request. Show a main
library/scheduling outcome matrix covering correctness and failures, author
effort, and frozen performance. Supporting plots should expose the full M/K/N,
batch/spatial/channel and query/context sweeps, with public cases and unchanged
fallbacks identified. Keep GEMM, convolution, QK and PV separate; do not pool
their speedups. Link mechanism ablations to actual emitted changes. Instruction,
traffic and model activity counts explain behavior but are not an additive
decomposition of measured physical cycles. Existing attention measurements
characterize transfer gaps; they are not a clean library-versus-scheduling agent
ablation. Keep the older pilot, one-hour authors and frozen mechanism cohorts
separate in every caption and source table.

These are single author runs per condition. They support bounded mechanism and
case-study claims, not universal agent gains, autonomous invention or broad
hardware portability. Repeat well-defined agent pairs independently before
making a robust population-level claim. Never tune frozen authors to improve a
paper narrative.

Authoritative operator root: `$TASK/.tmp/cca-abstractions-next-20261028/`.
Use `queue/queue.json` and `queue/dispatch-state.json` for actual execution;
`paper-current/STATUS.md`, `metrics-final/`, `metrics-postprofile-v1/`, immutable
source snapshots and original profile arrays contain the evidence. The focused
mechanism report is `paper-current/scheduling-completion-mechanism-final-v1/`.
The complete 14-study tables are `paper-current/all-studies-v3/`: 934 original
cells, with 58 explicit recovery/alias records and original failures retained.
All stages in that completed cohort have ended; historical failed attempts have
separate recovery or documented terminal outcomes. The dispatcher exited with
no occupied study CPUs before the new follow-up below. Plot preparation and
evidence review do not alter authors.
Old pilot artifacts remain under `.tmp/cca-abstractions-study-20261028/`.

## New partial-feedback CCA pilots

The user requested another GEMM/convolution CCA pilot using the latest field-branch
work. It uses frozen framework `80e65d7a6392`, identical current libraries and
the same migrated Fresh6 source `f57f7d1d01ef…` in both arms. Only CCA feedback
differs. GEMM has three canonical comparison axes and separately scoped role
counts; convolution has source/ABI and static ISA diagnostics but incomplete
mechanism matching. This tests partial assistance, not the complete lifting loop.

Prospective `prepared-v4` corrects empty common GEMM objective metadata without
declaring seed knobs applicable or supplying a solved transformation. Four native
sandbox and four private/cross-arm checks pass. Existing selected public numerical
anchors retain their exact source/tool scope; no baseline sweep is repeated.
Two 30-minute authors per operator share CPUs 52–55 at equal 200% quotas, one
operator pair at a time, through existing reservation, author and metrics owners.
GEMM started 2026-10-28 17:46:35 UTC and freezes at 18:16:35; convolution follows.
Freeze before the fixed 17-GEMM and 18-convolution panels. Attention stays separate.

Configuration, source and launch authority:
`$TASK/.tmp/cca-gemm-conv-pilot-20261028/README.md` and `prepared-v4/protocol.json`.
These are new cohorts with a common library migration; do not attribute migration
gains to CCA or replace historical negative results. CCA remains omitted from the
paper figures unless the user changes that preference.
