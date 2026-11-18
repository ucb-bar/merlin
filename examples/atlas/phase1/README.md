# Phase 1: functional compiler

Select the reviewed Phase 0 manifest's `phase_corpora.phase1` functional members;
retain its evidence-manifest identity and selected software/hardware declarations.
The example's current diagnostic derivation is not that admission release.
Resolve required semantics and source consistency, regenerate a fresh run, and
complete independent coverage/numerical review before verified execution.
The Phase 2 selection is distinct and must not replace the functional population.

The [catalog definition](../experiment.yaml) selects **EL4 (RTL-informed Merlin)**
with an explicit bundle identity, manifest and timing path. These
retained inputs must be regenerated and reviewed for the selected corpus before
verified execution. See [execution prerequisites](../../../experiments/README.md#definitions-and-execution)
for missing artifacts, tool provisioning and historical-resume limitations.

[The target descriptor](../target/descriptor.yaml) selects the public inputs in
`contracts/hwbringup_atlas_v0/`:

- Architecture overview and ISA green card.
- ISA definition requiring the external `npu_model` dependency.
- Curated RTL source evidence, with a note identifying the full external tree.
- Worked assembly examples.
- Preflight assembly fixtures and their target-owned assembler/runner adapter.

These are supplied experiment inputs, not newly generated compilers or certification
results. Their bytes are preserved from the former experiment directory. The curated
RTL subset is not a replacement for the complete external hardware repository.
Merlin's shared preflight protocol consumes the declared adapter; it contains no Atlas
instruction encoding. Path validation does not imply that a hardware probe passed.

Start from [the experiment definition](../experiment.yaml) and
[local tooling setup](../target/README.md). The full-mode task prompt is generated
by the shared renderer. Prepare and review a fresh release through the
[Phase 0 handoff](../../../experiments/README.md#reviewed-phase-0-handoff) before
verified execution. Generate new bundles for these paths; historical bundles and
receipts retain their original bytes. Other harness resources still use the
descriptor's retained `resources_root`.

## Shared passes in the EL4 authoring bundle

Atlas uses the same `capsule_bench` driver as Gemmini. The Atlas descriptor
selects the compiler sources in [shared-library.json](../shared-library.json).
To inspect exactly what the author will see, run this from the Merlin checkout
in its Python environment. The command builds an input bundle without invoking
a compiler author.

```sh
python -m merlin.targetgen.generate_bundles \
  --descriptor examples/atlas/target/descriptor.yaml --arms merlin_rtlchecks \
  --variants public_v0 --with-tool shared_passes --python-source-root "$PWD/src" \
  --dest out/build/atlas-authoring/input_bundles
```

The selected bundle is `merlin_assisted_rtlchecks_public_v0+shared_passes`.
Its manifest grants one read-only file, `shared-passes.zip`, in addition to
the ordinary EL4 inputs. Its tool inventory describes how to use it. A fresh
Python process can put the ZIP first on `sys.path` and import the public
Linalg graph, dense planning and scheduling APIs. The archive also contains
the explicit module roster and source hashes in `compiler-library.json`.
The existing pre-launch tool check imports its public APIs in a fresh
interpreter inside the selected sandbox.
The ordinary submitted backend retains the Gemmini self-contained package
contract. The standalone dense planner and block scheduler can be vendored
unchanged. The separately reviewed compiler-library execution route supports
direct public imports when its host selects that contract.

For a real run, normal corpus preparation builds this additional bundle
inside the release from the selected Python sources. Use the **release's**
bundle and seal together. The eight floating component cases are useful
source-level regressions. Their independent component-experiment admission
is a separate workflow.

```sh
merlin experiment preflight atlas-functional --phase 1 \
  --corpus-seal "$RELEASE/private/seal.json" \
  --bundle-manifest "$RELEASE/payload/experiment/input_bundles/merlin_assisted_rtlchecks_public_v0+shared_passes/input_bundle_manifest.yaml"
```

This checks the selected corpus, tools and runtime configuration. The
operator-owned timing record and RTL-informed provider still come from the
selected hardware environment.
