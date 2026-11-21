# Radiance target inputs

`descriptor.yaml` declares target policy, workload requirements and external inputs.
`tooling.env.example` lists the machine-specific tool locations used by this example.
The template is documentation: Merlin never loads it automatically.

For the descriptor's current `resources_root`, copy it from the repository root:

```sh
cp -n examples/compiler-generation/radiance/target/tooling.env.example examples/compiler-generation/radiance/backend-generation/experiment.env
```

Edit the destination's placeholder paths before running. Existing process environment
variables take precedence. The destination is ignored by Git; never commit local paths
or credentials. If you change `resources_root`, put `experiment.env` in that selected
directory instead. A missing selected file does not fall back to another target's file.

Frozen runs retain their recorded inputs. Changing the selected tooling or
resource root requires a newly frozen run for verified
execution. Start from [the experiment definition](../experiment.yaml) and the
[test generation guide](../test-generation/README.md).

## Host-owned Muon support

The backend and Cyclotron oracle live in the `muon` support provider vendored at
[`../../muon/support`](../../../targets/muon/support), which is selected when
`MERLIN_TARGET_PATH` is unset. Its source revision is recorded in
[`../../muon/SOURCE.yaml`](../../../targets/muon/SOURCE.yaml) and
[the migration record](../../../../build_tools/dependencies/target_support.json). Provider identity `muon` and experiment identity `radiance`
remain distinct; do not rename one to make a lookup succeed.

The descriptor's `backend_package_dir` still denotes retained metadata and
historical pin locations. The relocation does not supply missing RTL facts or
IRDL pins, and verified bundles must be regenerated with explicitly provisioned,
reviewed inputs. Selected support stays host-private, never candidate payload.
Pure import tests are not toolchain or simulator qualification.
