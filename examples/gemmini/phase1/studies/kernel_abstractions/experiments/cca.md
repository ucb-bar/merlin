# Experiment 1: learning from one expert projection kernel

**Question.** Does CCA help an agent improve an established compiler sooner,
with a rule that retains its benefit when shapes change?

Both agents start from complete graded Fresh6 and the same explicitly migrated
current tools. Keep working passes, library, scheduling IR and editing freedom.
The control receives raw expert code; the treatment additionally gets qualified
lifting/comparison, its pinned derived report and CCA feedback. Neither receives
an oracle implementation or winning pass sequence.

## Inputs and shape panel

Use the full Fresh6 seed with tree hash
`8dd194ae90a076b9990341cec4cd4a71f713a9654bee529257f668503f310ea9`.
Carry the same actual notes, prior successes/rejections and session handoff.
Historical evidence is not a new certificate; no retained Fresh6 certificate
store was found. Do not repeat a baseline sweep of established cases.

Train only on `X_mm_o_m16k2048n2048`: signed-i8 inputs, M16×K2048×N2048,
complete raw-i32 output, no bias, scaling or activation. Freeze before evaluating
M32×K2048×N2048, M16×K1024×N2048 and M16×K2048×N1024. These are aligned,
one-axis shape changes. Decode M1, M17 and K tails belong in a subsequent stress
panel. Prior related seed cases are transfer evidence, not wholly unseen data.

Both arms receive the same curated approximately 217 KB reference bank: expert
`mm_o` only, its required headers/ISA context and trusted measurement card.
Do not expose complete archives or exact hidden-shape reference kernels. This
keeps the author focused and prevents raw-bank generalization leakage. Record
bank contents/hashes, current tool selections and any common dependency migration.
Qualify each starting compiler; automatic migration gains are not agent gains.

## What the lifting supplies

The existing bounded native capture observes unchanged expert C control at issue
sinks. The scoped observer recovers this selected contract and dataflow:

```text
A: i8[16,2048]       Y: i32[16,2048], no epilogue
packed W[n//64,k,n%64] = logical W[k,n]
transfers: A 32,768 bytes; W 4,194,304 bytes
reuse: the full A serves all output-column panels
```

Tile replay checks complete ordered-K coverage and every output tile. This is
issue/dataflow evidence, not execution of accelerator arithmetic, hardware
completion, overlap or elapsed cycles. The ordinary C syntax endpoint alone
cannot establish dynamic traffic. Supplied expert measurements are trusted
performance targets with their original binary/ROI/sample contract.

Only the CCA arm gets `CCA_REFERENCE.md` and enabled analysis feedback. This
pilot tests preprocessing plus analysis assistance; report preprocessing wall/CPU
cost separately and include it in total cost. Agent-requested native capture is
a later broker extension, not an extra harness or an already-shipped endpoint.
The original raw kernel and metric card remain identical in both arms.

## Small automatic loop

Give both agents the same task: improve this projection family using a general
shape/resource rule, preserve exact semantics and existing rules, and retain a
safe fallback. They may compose existing passes or implement missing generic
passes/scheduling consumers. Do not supply the operator's completed optimization.

CCA top-down analysis binds mathematics, layouts, fusion and all live outputs.
Bottom-up lifting recovers operands, traffic, reuse and producer order. Compare
those supported axes with actual candidate emission, identify a mechanism gap,
consult transformation contracts, and test a legal implementation. Reanalyze
emitted effects; a changed knob is not proof that the mechanism happened.
Use ordinary self-check for changed candidates and the shared model to shortlist.
Unknown or unsupported mechanisms remain explicit rather than guessed laws.

## Existing harness and evaluation

Use the shared bounded preparer with `--study cca --seed-continuation PATH`,
`--wall-s 1800 --replicas 1`. It creates the matched control/treatment pair with
the same current tool selection. Keep one evaluator per arm, equal 200% quotas
in the four-core author pool, and optionally a fifth core for common preparation.
Match author model/effort. Preparation and compact final evaluation have separate
recorded budgets; no full-model cold preparation or old-suite grading sweep.

Before launch, reopen bank, source/tool, evidence and model/ROI pins; verify the
common migration, source-bound training role and both no-author sandboxes. The
pinned base-config `cca_guidance: {path,sha256}` stages the report only in treatment.
Do not inherit old maps/tasks or fill missing SDK files implicitly.

Freeze one final compiler, rules, settings and dependency identity per arm before
opening the three test cases. No retuning or selection on test feedback. Check
every output and report every failure, fallback and regression. Compare final
variants and starting compilers on a matched four-shape FireSim panel, retaining
ordered samples and first/warm-call distinctions. Exact supported expert/oracle/
SW/FSM comparisons are useful; unsupported contracts remain blank with a reason.
Static counts and provisional modeled cycles are not physical comparisons.

Measure time/tokens/CPU, attempts, time to a correct changed kernel, mechanism
adherence, final cycles and frozen-shape regressions. CCA may improve effort
without changing final performance. One short pair is diagnostic; repeat before
claiming a reliable agent advantage. No author or FPGA campaign is launched here.

## Preparation status

The curated bank, aligned source panel and exact named pair are prepared. Both
sandbox preflights pass. The migrated current-tool anchor passes all 32,768
outputs and produces the identical original ELF, 1,380,088 provisional cycles
and model/domain/scope. This migration is not an agent performance improvement.
The actual lift endpoint succeeds only in treatment; source-only semantics stay
UNKNOWN while the separately pinned native-capture report supplies richer evidence.

Earlier V3 workspaces added seven SDK files after full-source staging; their
matched historical CCA preparation is not the clean selective-grant control.
Use the final operator handoff and exact named-study pins. Authors remain
unlaunched; hidden-shape and physical study results are not yet available.
