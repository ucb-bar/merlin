# Historical study records

[`manifest.json`](manifest.json) identifies the exact configuration records,
delivered prompts, input inventories, receipts and analysis reports for the
recorded compiler studies. The accompanying artifact package must contain
`historical-study-records-20261028.tar.gz`. Its size, SHA-256, individual file
hashes, executable modes and original source-relative paths are in the manifest.
The archive was prepared locally. It has not yet been assigned a public download
URL. A release without this archive does not provide these historical records.

These files are immutable experimental evidence. They retain the original
machine paths and historical outcomes. They are separated from the active source
tree so they cannot be mistaken for portable launch presets. Their hashes do not
establish that every referenced dependency, run log or binary is in this archive.
Full execution needs the selected compiler, inputs, runtime and toolchain from
the result's artifact package.

From the Merlin source checkout, verify the archive without restoring files:

```sh
python3 build_tools/scripts/restore_reference_archive.py \
  --manifest experiments/reference-data/historical-studies/manifest.json \
  --archive "$ARTIFACT_ROOT/historical-study-records-20261028.tar.gz"
```

To inspect the original files, restore into a separate evidence directory:

```sh
python3 build_tools/scripts/restore_reference_archive.py \
  --manifest experiments/reference-data/historical-studies/manifest.json \
  --archive "$ARTIFACT_ROOT/historical-study-records-20261028.tar.gz" \
  --destination "$EVIDENCE_ROOT"
```

Set `ARTIFACT_ROOT` to the extracted artifact package and `EVIDENCE_ROOT` to a
new directory. Restoration rejects non-file members, path traversal, symlink
destinations and overwriting different content. It reuses only files whose bytes
and permissions already match. It never executes archive members.

For the configuration registry, inspect
`$EVIDENCE_ROOT/examples/gemmini/phase1/studies/kernel_abstractions/configurations/20261028/registry.json`.
For the delivered tasks, consult the `served-TASK.md`, prompt-receipt and receives
records selected by that registry. Original run IDs remain unchanged. The
compiler repository's `paper/index` branch supplies the run-to-compiler mapping:
<https://github.com/ucb-bar/gemmini-mlir/tree/paper/index>.

**Restoration is not relocation.** Do not run historical plans unchanged on a new
machine or substitute new paths into an original record. Prepare a new explicit
configuration using the source launchers, record its selected dependency hashes,
and retain it beside the original. See the
[artifact evaluation guide](../../../docs/guides/artifact_evaluation.md) for the
input and measurement requirements.
