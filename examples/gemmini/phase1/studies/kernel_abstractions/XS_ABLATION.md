# Four-hour xs tooling and abstraction study

Material passport: experiment-agent implementation, 2026-10-28. Results are
unmeasured until source-bound numerical and physical observations are collected.

This is a new versioned study. Historical EL definitions and previous bounded
studies remain unchanged. The independent experimental unit is an author run;
workloads and ordered hardware calls are not independent author replicas.

## Six environments

The selected first wave is **EL1, EL2 and EL3, all from scratch**, per the
2026-10-28 operator decision. EL4 is handled by a separate session. The Fresh6
pair below is an unused optional mechanism study and is not in this wave.

| Arm | Origin | Supplied author environment |
|---|---|---|
| EL1 | Empty target backend | Ordinary C++/MLIR/LLVM tools |
| EL2 | Empty target backend | EL1 plus C++ compiler scaffolding/integration |
| EL3 | Empty target backend | EL2 plus lightweight Python/xDSL authoring plumbing |
| EL4 | Empty target backend | Full current Merlin, scheduling, passes, analyses and live CCA |
| Fresh6 control | Original frozen Fresh6 | Inherited implementations and common objective |
| Fresh6 treatment | Identical Fresh6 | Additionally the checked dense K-completion consumer |

The EL ladder is a broad environment comparison. The Fresh6 pair estimates the
incremental assistance of the new consumer, not the effect of removing Fresh6's
existing scheduling dialect. Both seeded arms have CCA off. EL4 has live CCA
with prior generated task cards off. All arms receive the same reviewed raw
reference bank; hardware timing targets and operator recipes stay private.

EL3 excludes modern scheduling/pass libraries, algorithmic tilers and completed
target lowering. Profiles enforce source inventories, broker capabilities,
guidance and actual author/helper/candidate package visibility. The trusted
judge retains the same latest runtime. Independently authored equivalent
implementations are permitted.

EL3 is a lightweight Python/xDSL and ABI support bundle, not a parser-only
switch: its exact dependency closure also includes generic logical-ABI,
command-buffer and tensor/numerical support. These contain no supplied
Gemmini lowering, optimized target kernel, scheduling library or CCA.
Report the complete module inventory with results. The EL1/EL2/EL3 comparison
measures these environment bundles; the later nested tooling study holds its
EL3 baseline fixed to isolate added scheduling, transformation and CCA support.

Fresh6 treatment must add only `dense_panel_epochs.py` to its original 1,026
library members. Any replaced dependency changes the experiment and requires
new common-baseline qualification. Preserve data commands, issue order,
blocking, operand windows, separate output stores and legal fallback. Unknown
ordering premises retain drains.

## Shared workloads and frozen transfer

Every author receives the complete xs suite: 109 core, 40 scalability and three
connected programs. Fourteen protected historical cases retain their ordinary
evaluation policy. Additional private whole-model campaigns are outside this
study; public model capsules and the three selected programs are retained.

The seeded optimization anchor is raw-i32 GEMM M×K×N = 16×2048×2048. After all
authors freeze, evaluate independent one-axis sweeps, keeping other dimensions
fixed:

| Family | Axis | Values |
|---|---|---|
| GEMM | M | 8, 16, 32, 48, 64 |
| GEMM | K | 1024, 1536, 2048, 2560, 3072, 4096 |
| GEMM | N | 1024, 1536, 2048, 2560, 3072, 4096 |
| QK | Queries | 8, 16, 32, 48, 64 |
| QK | Keys | 8, 16, 32, 48, 64 |

QK fixes 32 query heads, four shared key heads and D64 at its 16-query/16-key
anchor. Keep all 32 outputs and exactly four shared key inputs. It is QK, not
full attention. Boundary probes change only GEMM M to 15/17, K or N to
2047/2049; or QK query/key length to 15/17. This is 34 unique cases: 21 GEMMs
and 13 QKs, including both anchors. There is no Cartesian product.

Label sub-tile, aligned and boundary behavior separately. Retain the PI's
32×2048×2048 point and label its exact historical Fresh6 exposure. Likewise,
16×2048×2560 has public geometry exposure through a different i8-output contract.
Availability in a roster is distinct from demonstrated author access.

`transfer_panel.py` pins canonical sources, preserves their complete contracts,
rebuilds shapes/numerical bounds and writes independent deterministic fixtures.
Pass `--evidence-root` for fresh ordinary source-placement screening. Source
admission alone is not executable qualification. Keep the generator, manifest,
fixtures and results outside every author/helper mount.

## Preparation and launch

Use the existing task and branch. All temporary material belongs in the task's
`.tmp`. `study.py specs` takes a base xs campaign spec, the panel manifest,
original seed tree/hash, explicit inherited/scheduling source selections and the
original legacy SDK selection. It produces six ordinary campaign specs and
`study.json`. Pass `--arms el1 el2 el3` for the selected scratch wave; seed
arguments are unnecessary for a wave without seeded arms. Its three CPU pools
are 38–47 plus 56–60, 61–75, and 76–90. CPUs 4–37 and 48–55 are reserved
by the operator for other experiments and agents; never allocate them. Verify live
ownership before use. The selected source revision and export ID are frozen in
the study manifest: pass that `--export-id` to every native preparation command.

Run native `fresh_campaign.py export`, `inputs`, `seal`, `ordering`, `plan` and
`preflight --execute` for each selected spec. These actions never launch an
experiment author. Prepare the selected wave in order: EL1, EL2, EL3.
Reviewed corpus products use the framework export's native `out/artifacts`
directory so public reference grants remain repository contained. Operator
logs, study manifests and temporary scratch stay in the task's `.tmp`.

`study.py prepare --study STUDY --plans PLAN... --review REVIEW` accepts the
first three or all six plans. Its pinned `merlin.study.launch_review.v1` receipt
requires true checks for common xs sources/raw references, reference-content
review, actual author/helper/candidate sandbox canaries, initial backend
qualification and transfer-contract qualification, plus the panel SHA. Seeded
waves additionally require time-zero seed qualification. Attach
the underlying evidence; source preparation and unit tests alone do not justify
claiming real numerical or sandbox readiness.

Only `study.py run --study STUDY --confirm` dispatches authors. It enforces one
task coordinator, at most three of our experiments and five machine-wide,
reserving two seats for the Claude coordinator's 12-hour runs even between
their launches. Pending startup claims count, CPU pools cannot overlap, and
launch intent is persisted before spawning. Interrupted launches are recorded
without automatic retry. Only prepared arms dispatch.

Each service is pinned to fifteen CPUs and at most eight author helpers. Its
peer-safe reservation does not change or restore existing host/peer CPU masks;
ordinary host interference remains an observed resource, not an exclusivity
claim. Recheck the live service inventory before every launch.

The native policy gives 14,400 seconds of authoring and 900 seconds of process
drain, rather than subtracting drain from a four-hour total. Snapshots target
0/30/60/120/240 minutes and record actual capture offsets, exact source digests
and unavailable captures. Final coverage is explicitly deferred and cannot
produce an inherited certificate or a false numerical pass. The next author
does not wait for FireSim profiling.

## Evaluation and metrics

Use one frozen compiler, rules and settings per arm. Evaluate the same public,
protected and private-transfer contracts without retuning or per-case winner
mosaics. The full final physical panel is 34 shapes. Intermediate physical
panels are fixed in advance: both anchors; GEMM M8/M64/K4096/N4096; QK
query64/key64. Reduce intermediate profiling before dropping final cells.

`physical_panel.py` prepares/builds local batch ELFs and imports actual wrapper
custody; it never dispatches hardware. Use only the installed `fpga-run-elf`
wrapper, one job at a time, after hardware correspondence, primitive-command,
ABI, stack and memory checks. Pin deployment/runtime/cache/ROI identity. Close
each complete-call timing window before comparing every output byte. Retain
three ordered samples and the order-control repetition. A primitive memory
sweep is not proof of a cold cache. Missing or wrong outputs remain failures or
missing cells, with no zero cycles and no historical-domain ratios.

`study_metrics.py` reuses native provider and timing telemetry. Its operator
context supplies the public roster, author window and source/ELF/model
baselines. It reports lead/helper actual tokens, time/tokens to matched
milestones, static/numeric/ordering failures, repair episodes, repeated
failures, coherent-snapshot coverage AUC, explicit emitted reuse/fallback
evidence and queue/stage distributions. Missing joins remain null. In-flight
token reconstruction is a labelled lower bound. Final quality and
tokens-to-matched-quality are separate outcomes.

Report all 34 points, exposure strata, rule applicability, fallbacks, worst
slowdowns and missing results. Better performance/faster convergence/fewer
tokens are hypotheses. The initial unreplicated six-arm study is descriptive;
repeat the strongest contrast before claiming robust stochastic effects.
