# Radiance target inputs

`descriptor.yaml` declares target policy, workload requirements and external inputs.
`tooling.env.example` lists the machine-specific tool locations used by this example.
The template is documentation: Merlin never loads it automatically.

For the descriptor's current `resources_root`, copy it from the repository root:

```sh
cp -n examples/radiance/target/tooling.env.example merlin/experiments/capsule_bench/targets/radiance/experiment.env
```

Edit the destination's placeholder paths before running. Existing process environment
variables take precedence. The destination is ignored by Git; never commit local paths
or credentials. If you change `resources_root`, put `experiment.env` in that selected
directory instead. A missing selected file does not fall back to another target's file.

The live environment file and retained contract resources have not moved. Frozen runs
retain their recorded inputs; changing tooling requires a newly frozen run for verified
execution. Start from [the experiment definition](../experiment.yaml) and the
[Phase 0 guide](../phase0/README.md).

## Host-owned Muon support

The backend and Cyclotron oracle now live in a separate local `muon-support`
repository, not inside Merlin. Select its root explicitly:

```sh
export MERLIN_TARGET_PATH=/absolute/path/to/muon-support
```

See [the migration record](../../../build_tools/upstreams/target_support.json)
for the exact local revision. No publication remote is configured for this
companion yet. Provider identity `muon` and experiment identity `radiance`
remain distinct; do not rename one to make a lookup succeed.

The descriptor's `backend_package_dir` still denotes retained metadata and
historical pin locations. The relocation does not supply missing RTL facts or
IRDL pins, and verified bundles must be regenerated with explicitly provisioned,
reviewed inputs. Selected support stays host-private, never candidate payload.
Pure import tests are not toolchain or simulator qualification.
