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

## Kernel library

`kernel_library/` is the Atlas counterpart of the Radiance
[kernel library](../../radiance/phase1/README.md): a pinned public selection of
atlas-kernels families and a script that prepares it as a generated artifact.

- `kernel_library/selection.yaml`: the nine atlas-kernels families mapped to
  semantic op, dtype, and shape rules, in the schema and field names of the Radiance
  `selection.yaml`. It is generated; do not edit it by hand.
- `kernel_library/generate_selection.py`: regenerates `selection.yaml` from the
  `kernels/*/kernel.yaml` manifests and the `coverage/` placement data at the pinned
  commit. The per-family rules are written in the generator because the manifests
  state shapes in prose; the generator refuses a rule that disagrees with a manifest,
  a recorded projection variant, or a model op that the coverage data places on that
  family.
- `kernel_library/materialize.py`: the Radiance script with Atlas names. It does not
  turn these examples into a link-time dependency of submitted compilers.

The pin is atlas-kernels `9f63c53` on `feat/kernel-library`, which is not yet on
GitHub. After pushing, update `SOURCE_COMMIT` and `SOURCE_PR` in the generator,
rerun it, and update the selection digest in
`merlin/tests/infra/test_atlas_kernel_library_example.py`.

To materialize the pinned library, from the repository root:

```sh
.venv/bin/python examples/atlas/phase1/kernel_library/materialize.py --checkout /path/to/atlas-kernels
```

`MERLIN_ATLAS_KERNELS` may name the checkout instead of `--checkout`. The default is
`out/artifacts/targets/atlas/kernel_library_v1`, with the output root honoring
`MERLIN_OUT_ROOT`; `--output` selects another destination. The export holds
`Makefile`, `atlas_kernels_path.py`, `lib/`, and the nine `kernels/<family>/`
directories, plus `selection.yaml`, `manifest.yaml`, and `README.md`. It omits
`coverage/`, which holds op inventories of the held-out models. Reuse, refusal, and
symlink rules are those of the Radiance script. This export alone is not
certification.

To regenerate or check the selection against a local checkout:

```sh
.venv/bin/python examples/atlas/phase1/kernel_library/generate_selection.py --checkout /path/to/atlas-kernels --check
```

Four families are stages fused into a `projection_gemm` launch rather than launches
of their own: `epilogue`, `rne_add`, `rope`, and `resnet_patch_gather`. The schema
has no field for that, so they are selected through compound semantic ops
(`matmul_silu`, `matmul_add`, `matmul_rope`, `conv2d_im2col_matmul`, and so on), as
the Radiance `rope_qkv_fused` family is selected by `matmul_rope`, and each rule's
comment repeats the manifest's stage. The schema also cannot express set or
divisibility constraints (RoPE and RMSNorm rows of 32 or 128, GQA head ratios,
power-of-two channels); the rule comments name each case. Within
`projection_gemm`, the library itself picks a traffic-reuse recipe for an exact
MxKxN, so the selection names the family, not the recipe.

The library is not enabled by default. The [target descriptor](../target/descriptor.yaml)
declares no information treatments, and the experiment holds TinyLlama, SmolVLA, and
ResNet50 out as validation models, while these kernels were written for exactly
those models. Enabling it is the experiment owner's decision. By analogy with the
Radiance descriptor, it would take a `hardware_pins` entry named `atlas_kernels`
in `merlin/contract/hardware_pins.yaml` (commit, `root_env: MERLIN_ATLAS_KERNELS`)
and a third information treatment in the descriptor:

```yaml
information_sets:
  hwbringup_kernellibrary_v0:
    condition: kernel-library
    allowed:
      - path: out/artifacts/targets/atlas/kernel_library_v1/
        as: atlas-kernels
        mode: ro
        note: exact pinned atlas-kernels snapshot for schedule and compiler-rule mining
    source_pins: [atlas_kernels]
```

Whether that treatment can coexist with the held-out roster, or must wait for
models outside it, is an open question.
