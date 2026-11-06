# Package TorchAO source captures for Kevin

`pack_source_capture.py` writes a deterministic gzip/tar archive containing the
original `capture_receipt.json` and exactly the files in its `artifacts` map.
It requires the published SHA256 of the original receipt, verifies all source
artifacts before writing, and reads back every archive member afterward. It
rejects extra files, duplicates, symbolic links, an existing output, and low
scratch space. The default free-space floor is **190 GiB plus the uncompressed
capture size**. Never lower the floor while the compact VME Vivado build is
active. `--check-only` verifies the source without creating an archive.

On this shared host, set `SOURCE_CAPTURE_DIR` to the saved SpectFormer capture
directory. From the Merlin checkout, check it with:

```sh
PYTHONPATH=src python examples/kodiak/pack_source_capture.py \
  "$SOURCE_CAPTURE_DIR" \
  "$TMPDIR/spectformer-torchao-source.tar.gz" \
  --expected-receipt-sha256 bd554a8cba6caf1b9c9c133073b7a839d3e596b70b2304d99c66aa03cd31038d \
  --check-only
```

Omit `--check-only` when the resource guard permits packaging. Save the JSON
report beside the archive and publish its `archive_sha256` with the transfer
location. Kevin should verify that SHA256 **before extraction**, then pass the
published source receipt hash to `relocate_capture.py`. The archive contains
the original machine's absolute `prov.weights_file`; relocation emits a
derived receipt and retargeted model while preserving the original bytes.

The corresponding source receipt SHA256 values for bounded TinyLlama L2 s8
and full 22-layer TinyLlama s8 are, respectively,
`75634410bbfda3a89b7280586cd0cc49614acfd2ad5c17b9ff1582e2773683a3`
and `f8e4d1652d606492956504551cc4b10c848a051ac59f662b457ec731a50c92ee`.
Their saved source directories and all artifact hashes are listed in
`TORCHAO_CAPTURE_DELIVERY.md`. The full capture is about 2.4 GB uncompressed;
the physical Kodiak 512 MiB memory limit remains separate from simulator
delivery.

This archive proves materialized input bytes only. It does not establish
checkpoint/loader source closure, TorchAO framework parity, OPU/VME target
arithmetic, or FireSim execution. As of 2026-11-06, no source-authoritative
archive SHA or transfer location has been published.
