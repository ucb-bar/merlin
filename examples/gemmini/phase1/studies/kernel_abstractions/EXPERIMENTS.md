# Small experiments evolving Fresh6

We test whether CCA, reusable transformations and checked scheduling help an
agent improve an established compiler. Run three separate matched pairs first;
leave the combined system until their controls work. Detailed plans:
[CCA](experiments/cca.md), [library](experiments/library.md),
[scheduling](experiments/scheduling.md).

## Common starting point

Use the complete graded Fresh6 seed, including its working library:
`$TASK/.tmp/fresh8/seed/fresh6-frozen`, SHA256
`8dd194ae90a076b9990341cec4cd4a71f713a9654bee529257f668503f310ea9`.
Its original full-compiler selection contains 1,026 files. Preserve its custom
backend, passes, reports, rejected attempts and previous-session handoff. Start
new conversations and clocks; do not ask agents to reconstruct earlier work.

Use explicit reviewed dependency selections. CCA gets the same current tools in
both arms. Library and scheduling controls keep inherited tools; their treatments
add only the specified capability and necessary reviewed dependencies. No implicit
SDK fill or unrelated upgrade. Record each arm's initial effective source and
migration. Automatic dependency gains are not agent gains; compare final results
with each arm's own starting compiler as well as the original seed.

Historical grades remain evidence. The retained Fresh6 certificate store is
absent, so no invented cache hits or blanket passing claims. Reuse numerical
or timing results only where the existing owner proves matching executed code,
inputs, judge and measurement domain. Check new anchors and changed emissions;
do not run the old full suite.

## The three comparisons

| Experiment | Control | Treatment | Bounded task |
|---|---|---|---|
| CCA | Current tools and raw reference | Same plus lifting/comparison and CCA report | Learn a general projection optimization from expert code |
| Transformation library | Inherited tools and own implementation | Same plus new convolution composition APIs | Compose rolling input reuse, output groups, weight reuse and wider stores |
| Scheduling | Inherited tools and own implementation | Same plus the new checked epoch consumer | Remove redundant K-step completion while keeping data issue order fixed |

These test additions since Fresh6. Existing library code and scheduling IR stay
available in every arm. Both agents may add general passes and edit their compiler;
controls are not restricted to frozen backend code. Give both arms the same task,
reference inputs, numerical contract and feedback. CCA is off in the library and
scheduling pairs. Record route adherence and any crossover.

For scheduling, Fresh6 already uses `schedule.plan` and ready readout. The new
capability is `dense_panel_epochs.py`, absent from its original library. It uses
explicit hardware ordering laws to prove merged K-stream epochs. Keep stores
in their separate epoch. The selected execution law is supplied beside the
existing plan: this measures checked scheduling assistance, not dialect syntax.

## Small training and shape panels

Dense dimensions are **M × K × N**. Reveal one training source to each author;
keep the three test sources private until the compiler and rules are frozen.

| Experiment | Training | Frozen shape changes, one at a time |
|---|---|---|
| CCA | Projection 16×2048×2048, signed-i8 inputs/raw-i32 output | M32; K1024; N1024 |
| Scheduling | Same projection | Same three aligned changes |
| Library | B4, valid 3×3, input30×30, output28×28, C128→128 | B8; input34/output32; Cout64 |

Convolution uses NHWC, stride/dilation 1, no padding, original INT32 bias and
INT8 saturation with identity scale, without activation. The retained expert
comparison is conditional on its zero-halo/batch correspondence; it is not
an exact arbitrary-input or whole-model performance baseline.

Start with aligned changes, including the requested M16→M32 test. M1 decode,
M17, reduction/channel tails and capacity boundaries are later stress tests.
Preserve safe existing routes where a new rule does not apply. Label a previously
graded shape as transfer/regression evidence rather than wholly unseen data.
After this pilot, extend M first, then K/N and convolution axes separately.

## References and oracle

The first CCA pair gets the same curated approximately 217 KB bank in both arms:
expert `mm_o` source, required headers/ISA information and trusted metric card.
Do not mount complete archives or unrelated hidden-shape references. Only CCA
gets the pinned derived report and enabled analysis endpoints. The other studies
receive the same selected references in their own matched pair.

Top down, bind the mathematical operation, layouts, fusion and every live output.
Bottom up, lift operand addresses, transfers, reuse and producer order from expert
code. Compare supported common axes with the candidate's emitted code, choose or
create a general transformation, and verify that the claimed mechanism happened.
Issue/dataflow evidence does not establish hardware completion or cycles.

Keep operator oracle implementations and winning settings outside author inputs.
Use frozen oracle rules as evaluation ceilings only where source, quantization
and measurement contracts match. Supplied expert measurements are trusted targets;
preserve their exact binary, ROI and ordered samples rather than rebinding scores.

## Lean execution and fair accounting

Reuse `prepare_runs.py --study cca|library|scheduling`, `run.py` and `launch.py`.
Use `--seed-continuation`, `--wall-s 1800 --replicas 1`, the existing sandbox,
brokered self-check and one evaluator per arm. A named study creates its matched
control/treatment pair. No new runner or whole-model cold preparation.

Reserve four author cores with equal 200% CPU quotas, plus an optional fifth for
common preparation. Run pairs sequentially in this pool rather than oversubscribing
six authors. Match author model/effort and budgets. Record preparation separately;
allow a separate compact final-check/evaluation budget after the 30-minute clock.
The offline runner has no formal final drain.

Stage public hardware ordering and source-bound workload-role metadata identically.
Keep discovery metadata consistent with each actual grant. Before authoring, check
pins, migrations, training visibility, exact baseline context and both no-author
sandbox preflights. Stop or revise an unsupported or already-solved task.

Freeze one compiler, dependencies, rules and settings per arm before revealing tests.
Compile without retuning and check every output. Report failures, UNKNOWNs,
fallbacks and regressions. Measure effort, tokens/CPU/time, attempts, refused unsafe
changes, time to a correct improvement and final performance on every shape.
Use the common model for shortlisting; do not substitute it for physical timing.
Measure frozen variants together on FireSim under matched preparation/cache/ROI
conditions, keeping ordered samples. Expert/oracle/SW/FSM rows require exact
supported contracts; otherwise show a blank and a reason.

## Evidence and remaining preparation

The isolated scheduling rule passed all 32,768 outputs. Actual fences fell
2,084→36; same-domain provisional cycles fell 1,380,088→1,156,159 (16.23%).
Only the new epoch module was added over original Fresh6; stores and all data
commands remained unchanged. The convolution composition checked 401,408 outputs
exactly on its selected retained fixture. Neither establishes autonomous learning,
hidden-shape speedups or a new physical performance result.

All three named pairs now pass exact input/tool checks and all six no-author
sandbox preflights. The current-tool projection baseline also reproduces the
original ELF and modeled cycles; automatic migration adds no anchor speedup.
The collector accepts verified issued numerics behind compact public results;
71 integrated focused checks pass. Exact plans and receipts are in the operator
handoff under `.tmp/cca-abstractions-study-20261028/`.

Older V3 preparation added seven files through implicit SDK fill: matched for
its historical CCA comparison, but not a clean no-new-scheduling control.
Prepared authors are unlaunched and remain Claude-owned. Frozen-author evaluation
and matched FireSim measurements follow authoring; they are not completed results.

- **CCA**
  - Baseline: raw-reference agent with identical current tools and budget.
  - Test/hypothesis: add lifting/comparison; identify useful mechanisms sooner and improve frozen-shape performance or agent cost.
- **Transformation library**
  - Baseline: agent builds the same convolution composition using inherited tools.
  - Test/hypothesis: new reusable APIs reduce implementation effort/errors while preserving gains across batch, spatial and channel changes.
- **Scheduling**
  - Baseline: agent implements the same new completion rule using inherited tools.
  - Test/hypothesis: checked epoch construction reaches a safe general rule sooner and improves or preserves frozen-shape performance.
- **Combined system — later**
  - Baseline: competitive raw-reference agent evolving the same seed.
  - Test/hypothesis: CCA identifies mechanisms, transformations implement them and scheduling checks execution; compare total performance and agent cost under matched budgets.
