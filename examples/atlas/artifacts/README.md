# Atlas artifact map

This folder is navigation, not an output destination. Start with the
[Phase 0 walkthrough](../phase0/README.md) and the
[shared specification guide](../../../docs/guides/10-compiler-generation.md).
Generated files are never committed here.

Each fresh experiment run places its derivation bundle beneath `<run>/phase0/`:

| Path | Meaning and next consumer |
| --- | --- |
| `hardware/circt/facts.json` | Exact selected extraction bytes; preserve their raw identity |
| `hardware/effective-views/loaded-facts.json` | Resolved facts used to derive corpus bindings |
| `hardware/effective-views/refreshed-facts.json` | Snapshot-resolved register layouts and memory/readout inputs |
| `hardware/effective-views/performance-facts.json` | Derived profile and execution capabilities used by performance-family gates |
| `hardware/effective-views/readout-inputs.json`, `quantization.json` | Actual readout/format selections, not blanket operation support |
| `hardware/effective-views/toolchain-inputs.json` | Recorded producer identity versus current tool observations |
| `software/software-spec.json`, `contract.json`, `datapath.json` | Selected declaration and normalized consumer inputs |
| `evidence-manifest.json` | Source/derived hashes, consumer map and qualification blockers |
| `software/framework/pytorch-opset.json` | Selected framework operator catalog; unavailable is not an empty registry |
| `coverage/application-inventory.json`, `operation-accounting.json` | Exact selected inventory and per-application/combined operation partitions |
| `coverage/README.md` | Automatically generated readable summary of the same partitions and format decisions |
| `software/frontend/index.json` | Original/quantized/prepared PyTorch traces, typed MLIR graphs and per-capture operator catalogs |
| `software/host-capabilities.json` | Explicit host operation/precision declarations bound to the selected compiler identity |
| `software/quantization-contract.json` | Format-specific recipes, parameter unknowns and operation-scoped decisions |
| `coverage/generation.json` | Selected synthesis identity, omissions, failures and generation status |
| `capsules/MANIFEST.yaml` | Actual members and separate `phase1`, `phase2`, `diagnostic` selections |

Use the manifest's paths rather than guessing a latest run. The same SW and hardware
selection appear in the evidence bundle and capsule provenance. Raw extraction and
effective views deliberately have separate hashes.

Frontend call counts and lowered-operation counts are different denominators. Inspect
the source lineage and typed value edges rather than counting repeated provenance tags.
Host-required work and accelerator candidates remain obligations until the selected
compiler and execution receipts qualify them. Missing precision or transfer coverage
blocks verified whole-workload admission; it remains visible in diagnostic output.

Inside a member, inspect the generated README, `capsule.yaml`, interface MLIR and
expected instruction coverage together. Independent goldens, private diagnostics
and source snapshots are owner-side evidence, not compiler-candidate grants.
For model captures, retain input/argument receipts and weights beside the captured
MLIR; see [whole-model inspection](../whole-model/README.md) for lowering stages.

Phase 1 uses the functional selection and records compiler-byte identity, numerical
results and admitted dispatch. Phase 2 selects different performance workloads and
retains that exact frozen compiler lineage. Certificates/publication records stay
alongside immutable compiler payloads. Current diagnostic bundles establish none of
those verdicts merely by existing.
