# Atlas target inputs

`descriptor.yaml` declares target policy, workload requirements and external inputs.
`tooling.env.example` lists the machine-specific tool locations used by this example.
The template is documentation: Merlin never loads it automatically.

Named-program oracle checks and explicit input-tensor layout require the Atlas OOT
support package selected through `MERLIN_TARGET_PATH=/path/to/atlas-mlir/merlin-support`.
Its contract declares `runner.program_emitter`, including the model-specific encoding
option. Merlin runs that provider-owned helper in the configured model environment;
there is no bundled assembler patch or same-name fallback. The support package is
host-private evidence, never a compiler-candidate input. The former Python keyword
`fix_itype_rd` is removed; encoding policy belongs in the support declaration.
Local migration/schema tests do not certify the upstream model or its ISA patch.

Check machine-specific prerequisites from the repository root with:

```sh
.venv/bin/python examples/atlas/target/setup.py
```

The default only reports availability. Use `--write-env` to append absent local
checkout settings, `--sync-npu-model` to synchronize that checkout's environment,
or `--materialize-target-package` to derive a target definition. Supplying
`--target-package-dir PATH` also explicitly requests derivation. These actions are
opt-in; deriving a contract does not install the OOT oracle support above or certify
the target. The former `build_tools/scripts/setup_atlas.py` entrypoint is retired.

For the descriptor's current `resources_root`, copy it from the repository root:

```sh
cp -n examples/atlas/target/tooling.env.example merlin/experiments/capsule_bench/targets/atlas/experiment.env
```

Edit the destination's placeholder paths before running. Existing process environment
variables take precedence. The destination is ignored by Git; never commit local paths
or credentials. If you change `resources_root`, put `experiment.env` in that selected
directory instead. A missing selected file does not fall back to another target's file.

The live environment file and retained contract resources have not moved. Frozen runs
retain their recorded inputs; changing tooling requires a newly frozen run for verified
execution. Start from [the experiment definition](../experiment.yaml) and the
[Phase 0 guide](../phase0/README.md).
