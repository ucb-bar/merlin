# CI workflows

| Workflow | Trigger | What it establishes |
| --- | --- | --- |
| [`ci.yml`](ci.yml) (`pr-fast`) | Push, pull request, merge queue | Source and documentation gates, changed-file formatting, five executed tutorials, the complete test collection budget, and the compact behavioral suite. Its aggregate also requires integration. |
| [`integration.yml`](integration.yml) | Called by CI, weekly, manual | Builds the wheel and source archive, checks distribution contents, installs the exact wheel outside the checkout, and checks six pinned OOT provider boundaries plus their source wiring. |
| [`release.yml`](release.yml) | Manual | Runs the same integration qualification and retains its distributions and results. Publication and hardware deployment are not configured. |

`ci.yml` resolves one real Git comparison revision for formatting and ledger
checks. Its `fast` aggregate requires every selected job to succeed; the retained
`docs-freshness` check reports the shared source/documentation job's result.
Source gates run once. Current-document freshness and shrinking debt ledgers
remain enforced. Advisory steps report their own limitations and do not establish
missing execution evidence.

The permanent Merlin suite is capped at **150 collected cases**, including
parameterization. Collection errors, collection-time skips, deselection and an
empty collection fail the budget check. Execution skips remain visible in JUnit;
they do not establish that the skipped obligation passed. CI retains collection
identities, test outcomes and built distributions under each run's artifacts.

Integration selects exact companion revisions from
[`backend-boundaries.json`](../../build_tools/upstreams/backend-boundaries.json).
Their selected installed cases must run without skips against the same main
wheel, using the declared xDSL environments. Source-consumer pins additionally
select read-only source files for the wiring graph; those files are not imported
or offered to compiler candidates. These checks establish packaging and the
selected provider contracts, not accelerator execution, accuracy or performance.

Pinned companion commits must be available remotely for hosted CI. Local review
can use `check_backend_boundaries.py --source ID=PATH` with a clean checkout at the
selected commit. Local results do not establish that hosted CI ran. Actions use
full commit pins, read-only repository permissions and checkout without persisted
credentials.

## Prepared corpus qualification

Generated corpora and conformance requirements belong in ignored `out/` storage.
They are not tracked fixtures or inputs supplied by an ordinary source checkout.
Run coverage checking after generating, reviewing and selecting a corpus release:

```bash
# Set these to the reviewed release, not a source-pool example descriptor.
: "${TARGET:?set the descriptor's declared target}"
: "${DESCRIPTOR:?set the prepared descriptor containing the selected grading roots}"
: "${CONFORMANCE_SPEC:?set the generated and reviewed requirement YAML}"
test -f "$DESCRIPTOR"
test -f "$CONFORMANCE_SPEC"

MERLIN_TARGET_EXPERIMENT="$DESCRIPTOR" \
uv run --no-sync python build_tools/scripts/check_conformance_coverage.py \
  --target "$TARGET" --spec "$CONFORMANCE_SPEC" \
  --fail-on-uncovered --fail-on-unverifiable --json
```

The descriptor selects the corpus roots; `--spec` selects the requirement and any
hash-bound application sidecar it declares. Use the release's reviewed provider,
contracts, facts and source environment. Missing inputs or unmeasured required
axes prevent a complete coverage result. This command omits `--allow-unresolved`
and does not exempt gaps through a repository-wide historical ratchet. It audits
the supplied declarations; actual numerical, runtime and hardware qualification
remain separate obligations.

Check derivation/claim separation against the selected capture artifacts as a
second explicit audit. `QUALIFICATION_OUT` is the chosen output root containing
`artifacts/recaptures/<bundle>/model.mlir`; the gate partitions those bundles using
the reviewed claim-model policy:

```bash
: "${QUALIFICATION_OUT:?set the selected capture artifact root}"
test -d "$QUALIFICATION_OUT/artifacts/recaptures"

MERLIN_OUT_ROOT="$QUALIFICATION_OUT" \
uv run --no-sync python build_tools/scripts/check_claim_set_disjointness.py \
  --target "$TARGET" --fail-on-circular --fail-on-unverifiable --json
```

An absent capture set, missing claim models, unresolved contract or empty derived
requirement cannot pass this strict invocation. Neither artifact audit runs as an
implicit source-checkout qualification in ordinary CI.

Native tools, independent references, generated inputs, device execution and model
quality require prepared qualification environments. The independent verification
implementation lives under `src/merlin/verification/` and evaluation owners.
FireSim provisioning and infrastructure launch remain external.
