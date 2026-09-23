# Installed measured-claims coordinator

The catalog route in the [Phase 2 overview](README.md) now selects the installed
managed envelope and checkpoint coordinator, not the global model-portfolio
workflow. Keep the
[shared measured-claims template](../../../experiments/definitions/measured-claims-template.yaml)
as the canonical definition example. This guide explains its resource selections
and the lower-level coordinator diagnostics. There is no YAML import or automatic
conversion into the Bash arrays below.

## Select three groups of inputs

Use absolute paths and keep authored declarations outside generated run output.
The following Bash arrays separate deployment selection, retained scientific
evidence, and the budget. Set every referenced variable to a real reviewed input;
`${NAME:?}` refuses an unset or empty selection. These snippets do not provision
tools, generate certificates or approve spending.

Deployment roots select where resources are owned and where new results go.
`CORE_PACKAGE_ROOT` is the installed core `merlin` source directory;
`EXPERIMENTS_PACKAGE_ROOT` is `merlin_experiments`;
`EXPERIMENTS_NAMESPACE_ROOT` is the experiments-owned `merlin` namespace directory.
They are source inventories, not arbitrary parent directories. `SOURCE_ROOT`
resolves the declared target resources. The holdout catalog and Chia wrapper must
be the actual reviewed files for this deployment. For the installed envelope,
`CHIA_WRAPPER` is the installed `merlin_experiments.phase2.chia_envelope` source
file, not the CLI module or a copied script; the catalog resolves it automatically.
Output roots belong beneath
the configured output tree; do not place private holdouts or run payloads here
under `examples/`.

To inspect the three package roots in the same Python environment that runs Merlin:

```sh
python -c 'from merlin.common.paths import module_source_path as p; print("core_package_root:", p("merlin").parent); print("experiments_package_root:", p("merlin_experiments").parent); print("experiments_namespace_root:", p("merlin.targetgen.capsule_runner").parent.parent)'
```

A wheel installation may share one `merlin` directory for core and extension
sources; an editable installation may have separate roots. Copy the actual paths,
not the placeholder directories from the template. This inspection launches no job.

```bash
deployment=(
  --source-root "${SOURCE_ROOT:?}" --contract-root "${CONTRACT_ROOT:?}"
  --functional-runs-root "${FUNCTIONAL_RUNS_ROOT:?}"
  --stage-root "${STAGE_ROOT:?}" --measurement-root "${MEASUREMENT_ROOT:?}"
  --holdout-catalog "${HOLDOUT_CATALOG:?}"
  --core-package-root "${CORE_PACKAGE_ROOT:?}"
  --experiments-package-root "${EXPERIMENTS_PACKAGE_ROOT:?}"
  --experiments-namespace-root "${EXPERIMENTS_NAMESPACE_ROOT:?}"
  --chia-wrapper "${CHIA_WRAPPER:?}" --suite "${SUITE:?}"
)
```

Evidence selects an already qualified functional submission and its exact
descriptor, facts, performance profile and two cross-engine certificates. Copy
hashes from the original evidence; do not hash edited files to relabel them as
qualified. The tuning certificate and functional certificate serve different
purposes and are not interchangeable. No waiver is included in this example.

```bash
evidence=(
  --functional-run-id "${FUNCTIONAL_RUN_ID:?}"
  --functional-submission-sha256 "${FUNCTIONAL_SUBMISSION_SHA256:?}"
  --descriptor "${FROZEN_DESCRIPTOR:?}" --rtl-facts "${RTL_FACTS:?}"
  --perf-profile "${PERF_PROFILE:?}"
  --gsim-certificate "${TUNING_CERTIFICATE:?}"
  --gsim-certificate-sha256 "${TUNING_CERTIFICATE_SHA256:?}"
  --functional-gsim-certificate "${FUNCTIONAL_CERTIFICATE:?}"
  --functional-gsim-certificate-sha256 "${FUNCTIONAL_CERTIFICATE_SHA256:?}"
)
```

Choose a new experiment ID/root and an approved model. The small explicit budget
below is illustrative, not a prediction of sufficient time or monetary cost.
The campaign has multiple independent trials and later qualification/measurement
work; the authoring limit is not an all-inclusive campaign cost cap.

```bash
campaign=(
  --experiment-id "${EXPERIMENT_ID:?}" --root "${EXPERIMENT_ROOT:?}"
  --model "${MODEL:?}" --effort high
  --wall-budget-seconds 600 --rounds 1 --round-timeout-seconds 600
  --max-tool-calls 40 --tool-timeout-seconds 60
)
python -m merlin_experiments.phase2.checkpoint_cli \
  "${deployment[@]}" "${evidence[@]}" "${campaign[@]}" --dry-run
```

Read the returned declaration and blockers. Dry-run does not execute the campaign
and does not establish sandbox, simulator, Chia deployment or hardware readiness.
Use `python -m merlin_experiments.phase2.checkpoint_cli --help` to inspect all flags.

## Launch and resume only through admitted managed execution

Prefer `merlin experiment run` with the completed measured-claims definition.
It passes the explicit supervisor endpoint to the managed envelope, which obtains
the assigned-resource receipt before invoking the coordinator. Use
`merlin experiment resume RUN` to retry an interrupted frozen plan with its same
coordinator root. It does not resolve a newer definition or add a resume flag.

After approving resources, costs and evidence, the command without `--dry-run` is
the coordinator payload for the deployment's managed Chia launch. A direct shell
launch without its verified launch receipt is expected to refuse; do not fabricate
the receipt or bypass that check. Preserve the selected interpreter, wrapper,
source inventories and invocation in the managed launch declaration.

The coordinator commits hidden holdouts before authoring, seals independent
candidates, checks the common tuning evidence, regrades functionally, then reveals
and qualifies holdouts before the predeclared paired measurements and statistics.
It re-verifies saved evidence when invoked again with the same root and inputs;
there is no separate `--resume` flag. Changed evidence is a refusal, not permission
to overwrite history. A model-portfolio checkpoint cannot be adopted here.

Installed synthetic tests cover portions of this lifecycle, not a complete
Gemmini deployment. Keep original/private artifacts at their declared locations;
this guide neither relocates them nor renews old certification. See the
[architecture and qualification boundaries](../../../docs/reference/repo_structure.md)
and [shared execution guide](../../../experiments/README.md#definitions-and-execution).
