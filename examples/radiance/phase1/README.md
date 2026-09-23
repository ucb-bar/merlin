# Phase 1: functional compiler

The [catalog definition](../experiment.yaml) uses the installed RTLchecks
treatment with an explicit bundle identity, manifest and timing path. These
retained inputs must be regenerated and reviewed for the selected corpus before
verified execution. See [execution prerequisites](../../../experiments/README.md#definitions-and-execution)
for missing artifacts, tool provisioning and historical-resume limitations.

Start from [the experiment definition](../experiment.yaml) and
[target descriptor](../target/descriptor.yaml). Public inputs are grouped here:

- `contracts/hwbringup_radiance_v0/isa_include/`: the source-attributed ISA taxonomy
  and its sibling encoding patterns. Read their scope notes: this taxonomy is not
  the complete compute-instruction encoding.
- `contracts/hwbringup_radiance_v0/example_kernel/gemm_tile.S`: a retained worked
  assembly reference, supplied as an experiment input. Its original derivation notes
  and bytes are preserved; it is not a result from a new compiler experiment.
- `kernel_library/selection.yaml`: the pinned public kernel-library selection.
- `kernel_library/materialize.py`: prepares that selection as a generated artifact.
  It does not turn these examples into a
  link-time dependency of submitted compilers.

The descriptor distinguishes three information treatments: the ordinary worked
example, no kernels, and the pinned kernel library. The no-kernel treatment denies
the example-kernel subtree even though the enclosing contract is readable. Preserve
this distinction; more readable files would change the experiment.

The shared renderer generates the full-mode task prompt. Prepare and review a fresh
release through the [Phase 0 handoff](../../../experiments/README.md#reviewed-phase-0-handoff)
before verified execution. Generate new bundles for the example-owned paths; old
manifests and historical receipts are not rewritten. Generated capsules, compiler
payloads, library exports and certification records remain artifacts. The descriptor's
`resources_root` still selects retained harness resources. For local tool locations,
see [target setup](../target/README.md).

To materialize the pinned library, from the repository root:

```sh
.venv/bin/python examples/radiance/phase1/kernel_library/materialize.py --checkout /path/to/radiance-kernels
```

The default is `out/artifacts/targets/radiance/kernel_library_pr1_v1`, matching
the descriptor's library grant, with the output root honoring `MERLIN_OUT_ROOT`.
`--output` selects another destination; update the grant for a new experiment when
using either override. Reuse checks every exported file against a fresh export of
the pinned revision, including the manifest and README. Changed, missing or extra
files are refused, not overwritten; symlinked exports are rejected. This export
alone is not certification.
