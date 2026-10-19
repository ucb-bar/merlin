# Capsule authoring and external model validation

`phase1_el4_config.yaml` explicitly selects `validation_scope: capsules`.
Its 10,800-second capsule clock starts when the prepared native controller is
started: 7,200 seconds of authoring and the remaining hour for strict capsule
grading, freeze and publication. The supervisor records preparation wall time
separately. It retains the protected private-spec and instruction-selection
inputs needed by the public model capsules, without running whole-model source
preflight, checking every private tensor payload, or starting model workers.

Prepare a fresh launch with the ordinary checked-in entry point:

```sh
"$PYTHON" examples/gemmini/phase1/launch/launch_phase1.py \
  --config examples/gemmini/phase1/launch/phase1_el4_config.yaml \
  --task-root "$TASK_ROOT" --prep "$PREP" --run-id "$FRESH_RUN_ID"
```

The operator first binds the source revision, runtime hashes, sealed corpus,
validated input bundle, instruction selection and exclusions into a new config
and preparation directory. Add `--run` only for the authorized actual launch,
inside the CPU-reservation owner. No seed or resume is accepted. The selected
18 CPUs and runtime facts remain the same; preparation has no paid author.

After native exit zero, the capsule supervisor calls
`capsule_certification.verify_phase2_handoff` to reopen strict capsule grades,
resource checks, frozen compiler/OOT custody, author workflow and durable
telemetry. Successful publication sets `capsule_complete: true`. Its status
continues to show `external_full_models_complete: false` and the composite
`phase1_complete: false`; these do not erase completed capsule work.

An independent operator starts external validation only after this capsule
owner has exited and that frozen handoff has been independently reopened.
`external_validation` selects a separate 7,200-second budget and the exact
verified frozen compiler. The external owner selects full validation and
rebinds its own source, run ID, budget and receipt paths; it launches no author.
It retains complete source/request/routing, independently selected instruction
policy, actual supplier links, exact kernel checks and all declared full-chain
outputs. Capsule publication alone does not satisfy those obligations.

Configs that omit `validation_scope` retain legacy full supervision and its
readiness prerequisites. Capsule mode cannot be used as an external
qualification shortcut.

The formal engine writes `capsule_certification.json` only after the complete
selected public and hidden rosters pass their existing tier, numeric, structural,
integrity and resource policies against one unchanged compiler. The record pins
the actual grading invocations, source/contract/facts inputs, freeze, immutable
OOT commit and read-only `capsule_handoff/compiler` export. Its early technical
certificate does not assert author convergence or Phase 2 publication readiness.

After the author summary and telemetry are durable, `capsule_publication.json`
binds those exact bytes and a fresh workflow check on the frozen compiler.
`verify_phase2_handoff(run_dir, descriptor=...)` independently reopens this final
publication, including its technical certificate and workflow transcript. A
terminal operator checkpoint cannot become author convergence through this API.
The author cost record retains its existing authoring-only wall time; the
technical certificate separately records actual author-start to capsule-grade
completion wall time. A historical full run cannot resume under capsule scope.
