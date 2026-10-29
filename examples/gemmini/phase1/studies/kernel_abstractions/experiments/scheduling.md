# Experiment 3: learning a new completion rule

**Question.** Does a checked scheduling consumer help an agent improve the
projection family sooner, with a safe completion rule that transfers across
smaller, intermediate and larger shapes?

Both arms evolve the complete graded Fresh6 compiler. Keep its existing passes,
scheduling dialect, working optimizations, prior evidence and editing freedom.
This measures assistance for a new mechanism, not rebuilding the compiler or
removing its existing scheduling dialect.

## Task and actual difference from Fresh6

Fresh6 already consumes `schedule.plan` and has ready-readout support. Changing
an existing issue-order flag would be too small an experiment. Instead, remove
unnecessary full drains between K steps of its checked two-window dense stream.
For each input shape, keep its original blocking, operand windows, data commands
and issue order unchanged; keep output stores in their separate completion epoch.

The new `dense_panel_epochs.py` is absent from the original 1,026-file Fresh6
library. Its consumer proves that each operand load was consumed before reuse,
requires selected RAW/WAR and ordered-accumulator laws, and bounds the merged
issued work. Output-block boundaries and final completion remain. UNKNOWN
premises keep the corresponding drains. This supplies a reusable rule, not a
schedule selected by benchmark name or one fixed shape.

Derive issued-work bounds from the selected block and full K extent before
constructing the checked stream. Hardware pending-queue bounds stay unchanged;
a larger compiler bound does not grant extra queue capacity or an ordering law.

## Matched arms

- **Inherited tools:** the agent implements this completion rule using Fresh6's
  existing primitives and its own optimization/checking code.
- **New checked consumer:** the agent additionally receives
  `dense_panel_epochs.{DensePanelEpochLaw,schedule_dense_panel_epochs}` and its
  scoped API instructions. The remaining inherited library stays unchanged.

Both get the same mechanism objective, raw references, numerical/model feedback,
public ordering record and hardware facts. CCA is off in both. Grant only the new
capability closure; do not refresh the whole SDK or silently add unrelated passes.
The control may implement an equivalent rule. Record effort, failures and actual
route adherence rather than treating a build as proof of the mechanism.

Give both authors this family task, rather than an anchor-specific optimization:

> Improve projection GEMM across the public training shapes. Derive a reusable
> completion rule from operand lifetimes, selected target ordering and resource
> bounds. Preserve existing fast routes and fall back when premises are unknown
> or the new route is inapplicable. Do not dispatch on case names or a list of
> measured M/K/N triples.

The common hint identifies redundant K-step completion and the relevant
RAW/WAR/accumulator dependencies. Neither arm gets winning epoch counts,
thresholds, an oracle implementation or hidden shapes. Only the treatment gets
the new checked module and scoped instructions for calling it. Both may write
new general code; an API refusal is an applicability result, not permission to
erase a necessary completion or discard a shape.

The existing scheduling plan remains common to both arms. The new execution law
is supplied beside that plan, so this is a checked scheduling assistance study,
not an isolated test of dialect syntax.

## Shapes, budget and evaluation

Dimensions below are **M × K × N**, signed INT8 inputs and complete raw INT32
output, with no epilogue:

Use **three public training cases**: 16×2048×2048, 8×2048×2048 and
16×1024×2048. The smaller M checks that the rule is not tied to one live tile;
the smaller K exposes a different reduction extent. Keep 32×2048×2048 hidden
as the requested aligned generalization check.

The operator predeclares this **17-case panel: three training and 14 frozen
tests**. Starting from 16×2048×2048, change one dimension at a time:

| Sweep | Public values | Frozen values | Other dimensions |
|---|---|---|---|
| M | 8, 16 | 4, 24, 32, 48, 64 | K=N=2048 |
| K | 1024, 2048 | 512, 1536, 3072, 4096 | M=16, N=2048 |
| N | 2048 | 512, 1024, 1536, 3072, 4096 | M=16, K=2048 |

The repeated anchor is one case. Authors know they are optimizing the family,
but receive only the three public sources and their feedback. This operator
table and exact frozen source files remain outside their sandbox.

Preserve existing retained-LHS and other routes when the new two-window rule
does not apply. Report that fallback explicitly. M1 decode, M17 and reduction
tails are later stress tests. M4/M8/M24 already exercise partial M tiles;
the rest of this panel provides ordinary aligned scaling. Distinguish shapes
withheld from this round from those absent from Fresh6's earlier history.

Reuse bounded `prepare_runs.py` / `run.py`: 1,800 seconds per author, one evaluator
request per arm, equal 200% quotas in its selected four-CPU pool. An optional fifth core serves
common preparation. No full-model cold preparation or old-suite correctness sweep.
Both retain the same seed, evidence, dependencies and author settings. Supply
`COMMAND_ORDERING_SELECTION.json` through the existing public staging owner;
the base facts alone do not contain that ordering record.

Keep the same 30-minute clock despite the three public cases. Do not time every
candidate on all three: check the main anchor first, then validate promising
family rules on the other public cases. Reuse exact unchanged emissions where
the evaluator permits. Summarize all three public results, not only the fastest
anchor. Preparation must qualify the new M8 route/numerics or report its refusal;
a neutral tiny fixture is not a substitute.

Check changed emissions with the existing evaluator. Freeze one compiler, rule
and settings per arm before the 14-case evaluation, without retuning. Report every
failure, fallback, UNKNOWN and regression. Measure final variants on the same
FireSim panel under matched preparation, timing and cache conditions; retain
ordered samples. Model estimates shortlist alternatives and are not physical
timing. An exact supported oracle, expert or SW/FSM comparator may be included;
unsupported or mismatched contracts remain blank with a reason.

Report per-shape results and separate M/K/N sweep summaries, with the number of
cases compiled and correct. A retained safe fallback counts as such; it is not
a newly improved schedule. Larger cases that exhaust a model deadline remain
UNKNOWN until physical measurement. Do not silently drop them from averages.

## Offline qualification

The isolated operator-only rule preserves the original kernel's data projection
and block `(1,1,8)`. Its checked completion count falls **2,080 → 32**; real target
fences fall **2,084 → 36**. Stores remain separate and ten operand-consumption
proofs pass. The ordinary target L2 Spike evaluation checks all **32,768 outputs
exactly**. Same-domain provisional cycles fall **1,380,088 → 1,156,159**, or **16.23%**.

The operator recipe is outside author inputs. It adds only the new epoch module
to pristine Fresh6, derives its laws from the existing selected ordering owner,
and uses the original target emitter. Actual ordering/physical certification is
separate; this is not a FireSim result or autonomous discovery result. The older
M16 oracle combines other transformations and does not isolate this pass.

Receipts: task `attention-shared-epochs-20261027`,
`.tmp/panel-epochs-qualification-v1/`. No author or FPGA campaign was launched
for this qualification. Measure time/tokens/CPU, attempts, rejected unsafe
changes, time to the first correct changed kernel, frozen shape performance,
and the emitted completion mechanism.

The new public M8 case also passes ordinary target L2: **16,384 INT32 outputs
exact**, zero mismatches, with the original 1,026-file Fresh6 library unchanged.
This was a numerical-only check of one public case; its aggregate suite result
is incomplete because the other two cases were not requested. No timing was
selected. Receipt: `.tmp/projection-family-m8-qualification-v1/receipt.json` in
the same task.

The wider family panel is a new input/configuration revision. These two
qualifications do not validate all 17 cases or replace final matched-config
checks. No author or FPGA campaign is launched by this plan.
