# CPU-host compiler experiment

This experiment isolates the value of generated compiler infrastructure and, in the fourth arm, agentic
pass construction.  It does not use the paper networks for development.  The generic corpus definition is
frozen first, candidates are accepted on deterministic train/validation partitions, and the compiler is
frozen before any paper model is compiled.

`experiment.yaml` is intentionally `draft`. The frozen generic corpus, Chia/Codex runner, sequential
launcher, and executable L0--L3 grader are implemented. The grader has been exercised through L3 on the K1:
scalar and RVV use one pinned hart; RVV multicore uses the requested hart set; all modes verify `vlenb=32`,
numeric results, fallback status, task/CPU use, wall time, `rdtime`, and peak RSS. A paid campaign remains
`NO_GO` until the input/tooling protocol digest is reviewed and frozen and every preflight blocker is
cleared. Selected-policy/compiler/runtime hashes are post-campaign seals, not launch prerequisites. The
registered schedule uses four Williams blocks (four cells per arm). Neither script performs paid work
without an explicit `--live`.

The current draft uses `development_corpus_v2.yaml`. Version 2 adds only generic VLEN-derived movement
tail probes and preflight asks the production grader to prove semantic-family, tail, and genuine-multicore
coverage in train, validation, and heldout before a protocol can freeze. Protocols v1--v3 and campaign
`20260831T074022Z_launcher_seed001_b3aafbc` are retained as protocol-development evidence but are not
experimental arm outcomes: v3 stopped after its first agent because the exact v1 corpus had no
`movement_layout` tail case in any split. Its AET token/time records remain attributable to harness
development; its compiler package is ineligible for selection or reuse.

Protocol v4 subsequently launched one valid Arm 1 attempt,
`20260831T082839Z_launcher_seed001_b8213a5__arm1_raw_cpp__r00__seed001`. It is a genuine negative
compiler result: L0 passed 143/143 records, L1 passed 64/71, L2 passed 5/6, and L3 passed 10/13. The
remaining failures include a concatenate memory-safety error and producer/consumer/runtime-parallel
semantic failures. Agent completion, input isolation, AET reconciliation, the compiler seal, and grader
evidence all verified. The v4 launcher then stopped because failure continuation was an unfrozen CLI
choice, exposing a lifecycle defect after the outcome had already been observed. That result remains
immutable and reportable as protocol-v4 evidence, but it is never mixed into or retried within the v5
confirmatory campaign.

The next frozen campaign, `20260831T213208Z_launcher_seed20260831_b66ab07`, is retained only as a
protocol-design pilot. Arm 1 and Arm 2 independently built sealed packages, but both failed all 143 L0
compiler invocations because the public workspace did not disclose the exact descriptor enum-field ABI.
The same audit found that the task recommended a persistent worker team while L3 could authenticate only
workers created inside the selected call. Those common protocol defects were discovered after treatment
started, so the campaign was stopped and excluded in full rather than interpreted as negative arm
outcomes. Arm 1/2 terminal bytes remain untouched; the partial Arm 4 raw stream has a separate controller
cancellation, not a fabricated treatment outcome. Its one-shot claim now has three consumed and thirteen
cancelled receipts plus content-addressed design-audit, campaign-exclusion, and claim-revocation records.
The successor publishes the exact stable descriptor renderer and six family fixtures, and explicitly
places persistent pools in the separate continuous-inference session runtime.

The current draft binds failure handling, `analysis_plan_v1.yaml`, and launch order into the protocol
digest. Its 16 exact cells use a predeclared balanced 4x4 Williams order. Every arm occurs once in every
position, and all 12 directed within-block first-order carryover pairs occur exactly once. Provider
sampling is unseeded: the legacy `launch_seed` field is campaign metadata only, while each row's `seed`
is the AET/run-ID representation of its paired block identifier and is never sent to Codex. Every cell
launches once even after a
terminal grader failure; a terminal outcome is never retried. Live launch atomically consumes the frozen
protocol once, and each exact arm/repeat cell has its own authorization that is atomically consumed only
after machinery preflight succeeds and immediately before the paid run. Before the first cell in each
block, the launcher enforces one real `board_environment.settle_interval_seconds` washout and retries the
frozen K1 environment gate up to `board_environment.settle_attempts`; its exact board-state evidence is
retained under `contracts/block_boundaries/`. This makes the three cross-block transitions explicit
requalification boundaries rather than unbalanced carryover observations. The finalizer requires all 16
consumption receipts and all four qualifying boundary receipts. It accepts a negative row only
when agent completion, input audit, AET, search/compiler seals, summary, and grader evidence consistently
establish `graded_fail`. Campaign completion is separate from compiler promotion: only the predeclared
Arm 4/repeat 0 may be promoted, and a failure there leaves the completed campaign explicitly unpromoted
even if later Arm 4 repeats pass. Holdout capture remains blocked in that state.

Preflight also re-expands the corpus definition byte-for-byte and pre-materializes each arm's complete
agent-visible input tree. Those four tree identities, the sandbox boundary, staging/sealing code, K1
measurement adapter, schema/path authorities, and AET/Chia source identities all enter the composite
protocol digest. A live arm rechecks its staged tree against the same preflight identity before dispatching
Codex.

Preview the exact schedule and live-authorization blockers without spending tokens:

```sh
build/chia-venv/bin/python merlin/experiments/cpu_host_compiler_v0/launch.py
```

To check only whether the grader, AET/Chia/Codex integration, and board are operational while the protocol
is still a draft:

```sh
PYTHONPATH=merlin/python .venv/bin/python \
  merlin/experiments/cpu_host_compiler_v0/preflight.py --probe-board --machinery-only
```

After reviewing the draft, calibration records, and reported protocol digest, atomically create a new
frozen spec (the draft is never edited and a failed check leaves no output):

```sh
cpu_host_stamp=$(date -u +%Y%m%dT%H%M%SZ)
cpu_host_frozen="out/runs/k1_cpu/cpu-host-compiler/protocols/${cpu_host_stamp}/experiment.protocol_frozen.yaml"
PYTHONPATH=merlin/python .venv/bin/python \
  merlin/experiments/cpu_host_compiler_v0/preflight.py --probe-board \
  --freeze-protocol "${cpu_host_frozen}"
```

The freeze controller requires that live board probe and publishes the frozen spec together with sibling
`experiment.protocol_frozen.environment.json` and
`experiment.protocol_frozen.environment.sources.tar` artifacts. Every paid cell then exact-rechecks the
frozen local tool/source identity and K1 device/kernel/OS/ISA identity before it starts.

Then preview that exact frozen schedule without paid work:

```sh
build/chia-venv/bin/python merlin/experiments/cpu_host_compiler_v0/launch.py \
  --spec "${cpu_host_frozen}"
```

After the draft is frozen, the same launcher runs the 16 frozen arm/block attempts sequentially under
Chia, with a frozen 32,000-second active-wall limit per attempt, one logical Codex slot, and one logical K1 slot. Each
attempt gets a separate deny-by-default
workspace and Codex home. Raw JSONL, arrival-time sidecars, retry attempts, token subsets, tool calls, AET
trajectory/reconciliation, agent time, grader time, and total wall time land under
`out/runs/k1_cpu/cpu-host-compiler/`.

After all 16 terminal observations exist, seal their exact launch record and promote only the predeclared
primary when eligible:

```sh
PYTHONPATH=merlin/python .venv/bin/python \
  merlin/experiments/cpu_host_compiler_v0/complete_campaign.py \
  --spec "${cpu_host_frozen}" \
  --launch-record out/runs/k1_cpu/cpu-host-compiler/<campaign>/contracts/launch.json \
  --output "${cpu_host_frozen%.protocol_frozen.yaml}.campaign_complete.yaml"
```

The grader authorities are deliberately layered:

- L0: isolated build, manifest/metadata schema, MLIR verification, and emitted-code change.
- L1: three post-codegen random seeds, trusted scalar goldens, input immutability, guard zones, ASan/UBSan.
- L2: sealed tail cases on Spike at VLEN=256 plus substantive RVV load/store/compute evidence.
- L3: SHA-verified K1 execution with exact mode, CSR VLEN, affinity, active harts/tasks, time, RSS,
  independently replayed shard coverage, and an untimed worker-suppression counterfactual.

Arms 3 and 4 receive the frozen `optimization_space_v1.yaml` and executable `beam_search.py`. The search
is non-agentic and staged. Every legal one-action extension is screened on twelve content-stable train
capsules--two from each of all six generic families--using trusted Spike correctness, emitted-code digests,
and cycle counts. The deterministic width-one top survivor is then confirmed on one controller-private
post-freeze shape from each of all six generic families, first on train and then on validation, with
exactly six K1 measurement pairs in a frozen balanced parent/child order per capsule. Validation is used
only for promotion, heldout remains sealed, and one complete deterministic empty sweep establishes
convergence without counting cached observations twice.

The frozen width-one bound is 20 confirmation requests, 40 policy-package builds, 280 compiler
invocations, 240 Spike confirmation checks, 1,440 expected K1 program invocations, and 1,920 K1 program invocations as the
planning upper bound that includes invalid-pair replacements. The 2026-08-31 public-only calibration
measured 5.23 seconds per K1 program on average (5.39 seconds maximum), so the calibrated budget uses
5.8 seconds expected and retains 7.5 seconds planning-upper per K1 program. Two complete twelve-capsule
Spike screens took 66.45 and 68.94 seconds, projecting 6,910.41 to 7,169.26 seconds for the derived maximum
104 screens; the calibrated allowances are 8,500 seconds expected and 10,000 seconds planning-upper.
Confirmation package and compiler costs remained inside their existing limits, while two Spike
confirmation panels measured 2.39 to 2.95 seconds mean and 9.01 to 11.97 seconds maximum. The final
per-check envelope is 4.0 seconds expected and 20.0 seconds planning-upper, yielding 1,031 seconds expected
and 4,942 seconds planning-upper for confirmation. The resulting complete search budget is 17,883 seconds
expected and 29,342 seconds planning-upper. This fits the 30,200-second search window after reserving
1,800 seconds of each 32,000-second arm for pre-search agent work, leaving 858 seconds of planning-upper
search headroom. Preflight recomputes all stage counts and costs
from the frozen action groups and fails closed if they do not fit; the broker enforces each count and a hard
monotonic wall deadline.

The failed 2026-08-31 cost runs are retained diagnostic calibration inputs: their toolchain and
correctness stages passed, but stale declared budgets failed. They cannot serve as authorities. The final
reissued A/A, Spike-screen, confirmation-overhead, and K1-program artifacts are content-bound in
`experiment.yaml`; preflight independently replayed all four semantic gates and returned GO. The A/A
authority measured six families by six balanced pairs and bound a 0.123 symmetric margin into
`optimization_space_v1.yaml`.

A separate public-only equivalent-semantic-work diagnostic then exercised the same 1,024-output
`runtime_parallel/static_partition` capsule in one-hart and eight-hart modes. All six balanced pairs were
correct and directionally consistent, but the current ephemeral-pthread reference was about 2,360 times
slower in eight-hart mode: roughly 1.1 microseconds per one-hart call versus 2.6 milliseconds per
eight-hart call. Seven worker threads were created and joined per invocation. This first artifact is
retained as engineering evidence, not a paper speedup authority; an independent audit requires stronger
on-disk receipt replay and physical timed-worker attribution before it can be cited. The result is exposed
to every experimental arm as the same pre-campaign public diagnostic. The capsule experiment requires a
generic granularity remedy and exact per-call worker creation because its L3 authority independently
replays those callbacks. Persistent workers are optimized and measured later in the separate continuous-
inference session runtime; they are not representable by this intentionally narrow capsule ABI.

Replacement calibration has one enforced, non-circular order. First run
`calibrate_search_noise.py` on the retained raw compiler package. Its artifact binds only the
pre-result A/A protocol projection and records `noise_margin` as its output. Put that exact derived
margin in `optimization_space_v1.yaml`; do not edit the A/A artifact. Then run each cost producer with
`--noise-authority <AET-run>/metrics/k1_aa_noise_calibration.json`. The cost producer reads and hashes
that predecessor, requires the final space to contain its exact margin, and requires its retained raw
compiler tree to be byte-identical to the A/A input tree. Preflight independently reopens the configured
A/A authority and rejects reordered, re-encoded, cross-package, or stale-space cost artifacts. The
predecessor content hash, rather than a filename timestamp, is the causal ordering edge.

A staged shim forwards only that fixed protocol to a driver-side broker, which validates the exact public
sample and frozen action space and keeps a private observation ledger. After Codex exits the driver replays
the frozen state machine and requires the submitted policy to byte-match its selected policy. Beam work
stays outside the compiler package; `submission/search` is empty during evaluation and finally contains
only the replay-verified record and selected policy, both covered by the compiler seal. The sealed
split is never opened by search. The driver then creates a deterministic compiler-package seal; the grader
verifies that seal before opening heldout. Arm 4's only additional authority is to propose a generally legal
pass, which enters the same screen and promotion gates as every knob.

The lifecycle separates a `protocol_frozen` input/tooling contract from `campaign_complete` or
`campaign_complete_unpromoted`. A live arm requires the frozen composite protocol digest, and the
finalizer additionally verifies that each run's own preflight and workspace tree match it. Selected
policy/compiler/runtime hashes are required only for a promoted completed campaign.
