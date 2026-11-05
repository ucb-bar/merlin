# Merlin paper artifact

This package preserves the reported compiler versions, inputs, original measurement records, experiment configurations and analysis scripts. The result registry uses stable study names rather than manuscript figure numbers. Supplied IREE, TVM, Voyager and iterative-agent baseline implementations are outside this artifact's reproduction scope. Local Gemmini software-library controls are included.

This source checkout includes restoration tools and manifests, but not the measurement or dependency archives. No public archive download is configured here. Obtain the named files separately before running restoration. Toolchain binaries retain their own redistribution terms and are not part of the Merlin source release.

## Restore and check the saved results

Use Python 3.12 or newer, `zstd`, and approximately 8 GB of free disk space for the main extracted collection. The accompanying `archives/` directory contains the files named in `archives.json`. Archive verification requires no model account or accelerator. From this source checkout, set `ARTIFACT_ARCHIVES` to that directory and `ARTIFACT_DATA` to a new directory for restored inputs:

```sh
python3 build_tools/scripts/restore_paper_artifact.py \
  --manifest experiments/reference-data/paper-artifact/archives.json \
  --archive-dir "$ARTIFACT_ARCHIVES" \
  --destination "$ARTIFACT_DATA"
python3 build_tools/scripts/recompute_paper_results.py \
  --registry experiments/reference-data/paper-studies/registry.json \
  --artifact-root "$ARTIFACT_DATA" \
  --output "$MERLIN_OUT_ROOT/artifacts/paper-recomputation/all-results.json"
```

The restoration command verifies complete archive checksums, membership, expanded sizes and safe paths. It restores ordinary files and archive-local hardlinks into a new directory. It executes no archived program. An existing destination is refused. The analysis command verifies each consumed input before recomputing these seven result groups:

| Recipe ID | Quantities recomputed |
| --- | --- |
| `frozen-performance` | Frozen-compiler utilization and speed across the sixteen development cases and seventy-three additional shapes |
| `single-source-cca` | Family geometric means for the retained CCA continuations and their earlier comparisons |
| `scaffolding` | E1–E3 pass counts, uncached tokens, failed grade results and self-check latency |
| `checked-scheduling` | Time to the first ten-percent modeled improvement in each of five studies |
| `convolution-library` | Per-shape and aggregate hardware comparisons from the convolution transformation study |
| `controlled-cca` | Per-boot sample medians and controlled-pair contrasts |
| `fsm-controls` | Current automatic-library and adapted-library controls, kept separate from the earlier common reference |

Select one group with `--recipe ID`. The output receipt records script, registry and input hashes. The [calculation guide](../../../docs/guides/reproducing_paper_results.md) defines the numerical rules, timing boundaries and comparison cohorts. Recomputing saved observations does not execute the compiler or remeasure hardware.

## Inspect the frozen compilers

The included Git bundle preserves all thirty-eight compiler snapshots. The [compiler index](https://github.com/ucb-bar/gemmini-mlir/tree/paper/index) maps original run IDs to source commits, immutable tags, setup records and metrics. The eleven principal experiment branches are convenient entry points. Use the recorded full commit or tag when reproducing a result.

To inspect the bundle without a network connection, initialize a new repository and fetch its preserved refs:

```sh
git init compiler-snapshots
git -C compiler-snapshots fetch "$ARTIFACT_ARCHIVES/gemmini-mlir-paper-compilers-20261105.bundle" \
  'refs/heads/paper/*:refs/heads/paper/*' \
  'refs/tags/mlsys-paper-v1/*:refs/tags/mlsys-paper-v1/*'
git -C compiler-snapshots switch --detach mlsys-paper-v1/refinement/exo-kernels-4plus4h
python3 compiler-snapshots/experiment/verify_snapshot.py
```

Historical compiler bytes are separate from improvements to the framework in this release. Replacing an archived compiler or library with the current checkout changes the experiment. Historical source configuration records are restored under `historical-source-records/`, retaining their original relative paths. They contain exact delivered prompts and configurations. Their old absolute paths are provenance, not portable launch settings.

The separate `authoring-records/` collection retains delivered prompts, configurations, input inventories and self-check requests and replies. Its usage export is explicitly derived from the native event streams. Raw provider transcripts remain in private preservation archives and are not part of the distributable source tree.

## Rebuild and repeat execution

Rebuilding requires the original selected compiler, shared library, runtime, input interfaces, LLVM and RISC-V toolchains. Physical execution additionally requires the recorded FireSim design and an operator-controlled `fpga-run-elf` deployment. Check all dependencies before launching. A changed toolchain, hardware image, numerical policy, timed interval or cache policy defines a different measurement.

The accompanying dependency archives preserve the selected LLVM installation, compiler Python environment, RISC-V environment, interpreter installation and runtime headers. Their inventories record every file, directory, symlink, mode and digest. They require approximately 15 GB of additional disk space. `dependencies.json` pins the capture manifest, inventories and archives. Set `ARTIFACT_PACKAGE` to the outer package directory and `ARTIFACT_RUNTIME` to a new restoration directory:

```sh
python3 build_tools/scripts/restore_paper_dependencies.py \
  --capture "$ARTIFACT_PACKAGE/dependency-capture.json" \
  --capture-sha256 e91a7716c8800406e0682325f847fdcea1c7549563ddc07bd4784b373aa57bff \
  --archive-dir "$ARTIFACT_PACKAGE/dependency-archives" \
  --metadata-dir "$ARTIFACT_PACKAGE/dependency-metadata" \
  --destination "$ARTIFACT_RUNTIME"
```

The restored environments retain their original symlinks. Use them through the namespace adapter rather than executing a virtual-environment launcher directly from its new host path. The [kernel replay guide](../../../docs/guides/paper_kernel_replay.md) provides input validation, source-rebuild planning, interpreter probing, rebuilding and explicit hardware replay commands. Its configuration pins twenty-nine original image selections, including Exo refinement and the updated 89-case automatic-library comparison. A wrapper's successful exit does not itself establish a new numerical or timing result.

Agent development is another reproduction level. Its prompt, exposed files, starting compiler, model and reasoning effort, tool selection, CPU allocation and time budget must match the selected run configuration. A fresh stochastic agent campaign is not expected to generate the same source bytes. Preserve its new compiler and measurements under a new identity. New authoring must keep protected references and expected tensors outside the author sandbox, including temporary evaluator directories. Historical sandbox claims require a separate audit of the actual mounts and file-access records. Saved numerical correctness does not establish historical reference isolation.

The artifact records distinguish saved-data validation, frozen executable replay, source reconstruction and fresh authoring. Passing one level does not imply the others were run. The source release has been tested locally through restoration and saved-statistics recomputation. Independent clean-machine rebuilds and fresh hardware execution must be recorded separately.

Historical source profiles must use their captured shared libraries. Re-exporting V6, V8 or V22 from the current source would require later helper modules and change the files disclosed to the agent. The preserved libraries avoid that substitution. A new campaign using the current framework needs an explicitly versioned library selection.
