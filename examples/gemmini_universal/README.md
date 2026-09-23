# Gemmini Universal: separate compiler-generation example

Start with [the experiment definition](experiment.yaml), registered as
`gemmini-universal-functional` in the [catalog](../../experiments/catalog.yaml).
This is the Alveo U250 / FireSim Universal configuration, **not** the Gemmini
configuration in `examples/gemmini/`: its accumulator readout and supported
programs differ. Never reuse a Gemmini compiler certificate or hardware verdict.

| Step | Authored input | Generated or selected output |
| --- | --- | --- |
| [Target](target/README.md) | Descriptor, contract and attested ABI header | Extracted RTL facts and external tool qualification |
| [Phase 0](phase0/README.md) | Existing cohort policy; new derivation recipe still required | Run-owned capsules and reviewed corpus release |
| [Phase 1](phase1/README.md) | Functional experiment and retained public source bundle | Frozen compiler and separately attributed grade |
| [Phase 2](phase2/README.md) | Exact functional handoff; shared templates | New optimization run, only after qualification |
| [Whole model](whole-model/README.md) | Capture and lowering inspection | Audited IR; not a Universal deployment verdict |

Inspect the declared Phase 1 route and discover stored runs without executing an
agent or simulator:

```sh
merlin experiment inspect gemmini-universal-functional --phase 1
merlin experiment runs --target gemmini_universal
merlin experiment lineage /absolute/path/to/orchestration-run
```

`lineage` reads the frozen definition, selected input digests, handoffs and native
engine output locations. It does not inspect private capsule names or revalidate
live source bytes. `status` reports attempts; neither command certifies a result.
Historical run directories remain unchanged and may be inspected by explicit path.

This catalog entry is an inspectable authored input, **not execution-ready**:
the retained bundle, an operator timing record, a fresh reviewed corpus release,
portable ordinary input trees, toolchain and managed hardware worker have not
all been supplied. Phase 0 and Phase 2 are not silently inherited from Gemmini.
No generated artifact, grading answer, or third-party checkout was moved here.
