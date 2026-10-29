# Catalog-only allocation experiment

`gemmini-pass-catalog` selects **EL4-P**: EL4's tools and RTL feedback plus the
static [allocation catalog](../../../merlin/contract/pass_catalog.yaml).
The control is `gemmini-functional` (**EL4**). Use the same reviewed corpus,
hardware/semantics, provider, model, effort and budget for both. The new definition
contains Phase 1 only; it reuses Phase 0 output and adds no generated test cases.

The agent is asked to implement an inspectable resident-storage allocation stage
in its generated OOT compiler and retain before/after IR, assignments and the
production pipeline invocation. For Gemmini this means scratchpad/accumulator
placement under selected geometry and completion semantics. The agent chooses
the algorithm. Unknown facts remain unknown. Existing execution grading applies;
the catalog's independent-allocation and Lean checks are explicitly deferred.

## Prepare the inputs without launching an agent

Use this branch's Python sources and an environment with Merlin's `targetgen`
extra and `merlin-experiments` installed. In a worktree using another checkout's
venv, explicitly select this tree rather than importing the other editable install:

```sh
export PYTHONPATH="$PWD/src:$PWD/packages/merlin-experiments/src:$PWD/packages/merlin-dse/src"
```

Use an existing, completed and admission-ready Phase 0 run. Preparation can
validate and copy it without running a new derivation, agent or simulator. Both
arms must be requested explicitly; the default ladder remains unchanged.
`PHASE0_RUN` is an operator-selected run directory, not the diagnostic example.
Select the existing LLVM installation with `MERLIN_CLANG` as described in the
[Phase 1 guide](README.md) before preparation.

```sh
python -m merlin_experiments.cli corpus prepare "${PHASE0_RUN:?}" \
  --output "$PWD/out/artifacts/protocols/gemmini/pass-catalog-v1" \
  --bundle-arm merlin_rtlchecks --bundle-arm merlin_pass_catalog
```

Preparation creates an unsealed release. Follow the ordinary
[release review procedure](../../../experiments/README.md#reviewed-phase-0-handoff)
and seal its exact reviewed digest. Do not modify a previously sealed release to
add this variant. The new catalog grants will be frozen with the bundle at launch;
changes to catalog bytes require a new run, just like changes to other inputs.

Preparation produces these bundles under `payload/experiment/input_bundles/`:

- `merlin_assisted_rtlchecks_public_v0`: control, explicitly denies the catalog.
- `merlin_assisted_pass_catalog_public_v0`: treatment, grants the catalog and schema.

It also produces the corresponding realistic and hwbringup variants. Check
`tools.txt`, `allowed_files.txt`, `denied_files.txt` and the rendered prompt for
the selected public bundle. The catalog bundle must contain `pass_catalog`, retain
EL4's tools and deny the verification seam. The catalog is static; it starts no
new broker. Actual admission still requires complete hardware evidence, numerical
semantics, toolchains, independent target support, private gates and measured
oracle timing. This setup does not supply or approve those operator inputs.

## Inspect, then launch when authorized

After release review, inspect the resolved command without authoring or grading:

```sh
export PASS_CATALOG_RELEASE="$PWD/out/artifacts/protocols/gemmini/pass-catalog-v1"
python -m merlin_experiments.cli inspect gemmini-pass-catalog --phase 1 \
  --corpus-seal "$PASS_CATALOG_RELEASE/private/seal.json" \
  --bundle-manifest "$PASS_CATALOG_RELEASE/payload/experiment/input_bundles/merlin_assisted_pass_catalog_public_v0/input_bundle_manifest.yaml"
```

The resolved selection must be `level: EL4-P`, runner arm `merlin_assisted`,
treatment `rtlchecks`, and bundle arm `merlin_pass_catalog`. The example retains
the control's 12-hour budget and model; use `--phase1-driver`, `--phase1-provider`,
`--phase1-model` and `--phase1-effort` to select another matched pair deliberately.
Read-only inspection does not prove runtime readiness. Use the normal preflight
after provisioning the selected inputs; missing inputs must refuse.

**Later, only when ready to start authoring and grading**, replace `inspect` with
`run` in that command. For the control use `gemmini-functional --phase 1` with the
same seal and the `merlin_assisted_rtlchecks_public_v0` manifest. Use separate fresh
run directories; do not resume an EL4 run as EL4-P.

Compare build/capsule results and inspect whether the real compiler invokes the
allocation stage, correctly reuses dead storage and preserves retained values.
Record explicit refusals and missing declarations. Successful capsule execution
does not establish independent allocation verification or universal correctness.
