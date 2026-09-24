# Gemmini artifact map

This folder is a navigation guide, **not an output destination**. Keep the
authored inputs in [`../`](../README.md) and generated files under the configured
`out` root. No capsule, captured model, golden, run, compiler payload, or private
review record is committed here. A generated artifact can always be rebuilt
from its selected inputs; its frozen receipt identifies which build was used.

| Inspect | Generated location (relative to configured `out`) | Source / next selection |
| --- | --- | --- |
| Model capture | `artifacts/recaptures/<capture-version>/<application>/model.mlir` plus `capture_receipt.json` and tensor sidecars | Exact descriptor-declared derivation roster; select each versioned bundle by label, never a held-out claim model |
| Hardware facts | `artifacts/cache/rtl_introspect/gemmini/facts.json` | Selected elaboration and [target contract](../target/README.md); extraction is evidence, not the software spec |
| Complete application demand | `artifacts/verification/gemmini/<version>.application-demands.json` | Generated beside the conformance requirement; exact operation inventory, including refusals |
| Conformance requirement | `artifacts/verification/gemmini/<version>.yaml` | Capability, software spec, captured application demand, and boundary derivation |
| Proposed synthesis | `artifacts/verification/gemmini/v1/<product>/synth.yaml` | Automatically versioned product from explicit conformance, recipe and workload-spec identities; review before selecting |
| Phase 0 capsules | `runs/gemmini/phase0/<run>/phase0/capsules/` | Frozen Phase 0 run: `MANIFEST.yaml`, per-member `README.md`, `capsule.yaml`, MLIR and owner-side golden |
| Functional corpus release | `artifacts/protocols/<release>/` | Human-reviewed Phase 0 run selected explicitly by Phase 1; private seal and goldens stay owner-side |
| Phase 1 functional compiler | `runs/gemmini/phase1/<run>/` | Frozen corpus selection, compiler submission, qualification and separate certification records |
| Phase 2 optimization | `runs/gemmini/phase2/<run>/` | Exact qualified Phase 1 compiler plus separately selected performance workload/evidence |

The `<version>`, `<run>` and `<release>` names above are illustrative, not hardcoded
lookup rules. Read the selected paths and hashes from the experiment definition,
run plan, `MANIFEST.yaml`, release inspection and status receipts. Fresh Phase 0
selection uses `--phase0-conformance-spec` and `--phase0-synth-profile` together,
plus `--phase0-hidden-profile` when an operator-owned private corpus is required;
the reviewed paths are frozen without hand-editing a definition under `out`. A Phase 1
functional corpus and a Phase 2 performance workload are different inputs; a
target name does not identify either one. Phase 2 must also pin the Phase 1
compiler it optimizes.

Start with these read-only navigation commands:

```sh
merlin experiment inspect gemmini-functional --phase 0
merlin experiment runs --target gemmini
merlin experiment status /absolute/path/to/a/run
merlin experiment corpus inspect /absolute/path/to/a/prepared-release
```

For the derivation itself, follow [Phase 0 step by step](../phase0/README.md).
The historical checked-in conformance/synthesis references are diagnostic: their
family/dtype cells do **not** establish complete application-op coverage. Their
original captures had unresolved external calls; fresh versioned captures
remove those calls and add materialization receipts, but an operation inventory
still does not prove compiler execution. Do not label a run using the historical
references verified. For model IR and optional
non-bloated weight sidecars, see [whole-model inspection](../whole-model/README.md).

Inspect one selected versioned capture without claiming full roster coverage
(the command exits 2 if operations remain unresolved):

```sh
python build_tools/scripts/check_conformance_coverage.py --target gemmini \
  --application-capture lstmnetvit_int8_w8a8_consistent=/absolute/capture/model.mlir \
  --inventory-out out/artifacts/verification/gemmini/diagnostic.application-demands.json
jq -r '.applications | to_entries[] | "\(.key): \(.value.status), \(.value.counts.unclassified // 0) unresolved"' \
  out/artifacts/verification/gemmini/diagnostic.application-demands.json
```

The diagnostic is not a selectable conformance requirement. With a complete
receipt-verified roster, generate a new conformance requirement and adjacent
inventory, review them, then synthesize and freeze a new Phase 0 run. The
`phase_corpora` entries in its `MANIFEST.yaml` identify the
disjoint generated public functional, performance and diagnostic member sets;
private holdouts remain owner-side and are bound by the release separately.
The manifest's count-only `claim_model_evaluation` record is an obligation for
after Phase 1 freezes, not a public `SY_model_*` capsule or Phase 1 input.
