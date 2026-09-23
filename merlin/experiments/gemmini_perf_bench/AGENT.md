# AGENT.md — merlin/experiments/gemmini_perf_bench

Status: active

Cross-approach **Gemmini perf benchmark**: runs the kernel corpus through backends (golden /
generated / IREE-dialect) and reports cycles/GFLOPs. Consumes merlin (`scripts/run_perf_bench.py`
imports `merlin.targetgen`).

- **Tracked source**: `scripts/` (harness + reporting), `kernels/` (capsule test corpus:
  `capsule.yaml` + `capsule.interface.mlir` per kernel), tracked `reports/`.
- **Generated output**: runs → `out/runs/<target>/perf-bench/`, products → `out/artifacts/perf-bench/<target>/`
  (constants `RUNS`/`REPORTS` in `scripts/_pbcommon.py`).
- Reproduce: `python scripts/run_perf_bench.py --help`.
- The three corpus-authoring scripts (`gen_category_kernels.py`, `gen_golden_kernels.py`,
  `harvest_model_kernels.py`) require the selected OOT Gemmini support package's
  `gemmini_conformance.model_slices`. The shared distribution no longer owns the
  target-specific instruction requirements or fixed model-slice recipes. Set the
  host-only `PYTHONPATH` to that checkout's `merlin-support/` and select the same
  directory with `MERLIN_TARGET_PATH`; never grant this private authoring package
  to compiler candidates. The pure MLIR emitter remains in core
  `merlin.targetgen.contract.matmul_interface` and requires an explicit target.
- `conv_downsample_ab.py` is a target-specific study, not a generic compiler interface.
  Its selected support backend must export `conv_downsample_flag` and own the
  `gemmini_sched` sibling module. Missing capabilities refuse the experiment;
  the driver never imports a native helper as a substitute for selected support.
- **External-kernel bench**: `scripts/external_kernel_table.py` (+ `external_kernel_bundle.py`) measures a
  third-party kernel bundle beside this repo's routes on one design. Every row is run under BOTH
  protocols, because a kernel with one-time host setup looks fast measured twice and slow called once;
  the rendered program `_Static_assert`s the kernel's own fit macro so a design it does not fit fails
  the build instead of silently measuring the stock library; and each arm declares what is inside its
  window, since the external kernels pad on the host and the library pads in hardware.
  A row measured on ANOTHER engine (`cross_engine_rows.py`) may join only when a stock-library control
  measured on both sides agrees within a declared tolerance; the licence is computed from the foreign
  console log, not accepted, and a row that fails any of its five checks is refused with the reason.
- **Functional input gate**: the campaign consumes exactly one frozen functional submission
  (`merlin_experiments.phase2.campaign.inspect_functional_run`). A run that predates the immutable
  bundle-input snapshot v2 schema is re-verified — snapshot re-materialized, public + hidden
  grades re-run at L3 against the same submission bytes — by
  `scripts/refreeze_functional_run.py --source-run-id <run> --new-run-id <run>_refreeze_<date>`.
  It carries the original run's authoring provenance and never back-fills evidence.
  V3 adds host-only frozen inputs. Phase 2 verifies the original archived bundle, exact snapshot
  marker and full payload against the host-owned functional record; both candidate planes receive
  only a public manifest projection and mask private inputs at every translated mount. The original
  full snapshot and bundle hashes remain in host-owned receipts. A re-freeze archives its new grading
  bundle separately, preserving the original authoring manifest. Legacy v2 behavior is unchanged.

- **Source ownership**: `merlin_experiments.source_snapshot` freezes explicitly selected sources;
  `scripts/source_snapshot_layout.py` supplies the retained native layout.
  `merlin_experiments.phase2.campaign` and `.prompt` own the reusable admission/fork/
  completion and deterministic task contracts; native consumers import those owners.
  The portfolio experiment and authoring owners are installed package modules; native
  launchers still select compatibility inputs and optional completion-contract adapters.
  Its optional `contract_root` is a single resource selection for analysis, host
  policy, cache identity, sandbox schema mounts, prompt vocabulary and current
  checkpoint consumption. Pass the same selection to current consumers; sequence
  continuation/recovery do so automatically. V3 source policy uses installed logical
  module identities plus complete Phase 2/performance namespace membership. It permits
  explicit external live resources. Static reuse still
  requires all policy inputs inside its explicitly named, verified source snapshot.
  This is not alternate-root historical replay qualification; archived native dispatch
  is unchanged, with V2/V3 checking their explicit controller role. Historical policy
  versions remain inspectable; new verified execution requires a fresh current freeze.
  Its source-snapshot schema is separate from the functional-input schemas above.
  `merlin_experiments.frozen_python` is trusted launch instrumentation over those seals, not a second
  grader or process supervisor. Record native commands separately from bootstrap transport.
  Authoritative Python children must explicitly inherit the guarded launch; candidate
  commands retain their existing answer-masked bwrap policy. Do not claim arbitrary
  descendants or external tool/dependency bytes are frozen merely because a parent is.
  Performance source snapshots V4 record explicit imports and the selected support owner:
  external provider code/resources are copied into the existing source seal, while
  native selection retains its reference role. Guarded launches replace ambient
  provider, target-directory, contract and schema overrides with sealed ownership.
  Provider directories are containment-only import roots, never general sys.path
  additions. V1/V2/V3 archives remain inspectable; current verified launch requires
  a newly frozen V4 run. This source schema is unrelated
  to functional bundle-input snapshot V4 ownership.

- **Current paired reports**: `scripts/gen_perf_report.py --experiment-manifest <path>
  --manifest-sha256 <sha256> --candidate-record <trial>=<record> ... --output <report.md>`
  validates each candidate with the installed `phase2.candidate_verification` owner, binds it to the supplied parent,
  rechecks parent-pinned paired child/result bytes, and recomputes the authoritative
  all-trial GSIM statistics. Candidate records are explicit: no latest-run discovery.
  The old `--run-id` mode remains a distinct historical Verilator-schema reader and
  refuses current paired campaigns; it must not invent historical fields or relabel
  GSIM as Verilator. A measurement/report fixture with prequalified candidate inputs
  does not qualify full candidate authoring, paid agents, hardware, or a whole campaign.
