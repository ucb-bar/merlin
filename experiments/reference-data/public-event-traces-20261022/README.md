# Public diagnostic event traces

`event-traces.tar.gz` contains the original event streams from
`public-matmul-event-trace-20261022` and
`public-scratchpad-row-arrivals-20261022/matmul-trace`. The manifest records the
archive SHA-256 and each original member's path, size, permissions and SHA-256.
The original study indexes and manifests retain their original paths and hashes.
No event, diagnostic result or measurement has been regenerated or revised.

Verify the archive from the Merlin checkout:

```sh
python3 build_tools/scripts/restore_reference_archive.py \
  --manifest experiments/reference-data/public-event-traces-20261022/manifest.json \
  --archive experiments/reference-data/public-event-traces-20261022/event-traces.tar.gz
```

For a complete analysis copy, first copy the retained reference tree into a
separate artifact directory. Then restore the original streams beneath the same
root. For example, set `EVIDENCE_ROOT` to an absolute directory beneath your
configured artifact output root and run:

```sh
mkdir -p "$EVIDENCE_ROOT/experiments"
cp -R experiments/reference-data "$EVIDENCE_ROOT/experiments/"
python3 build_tools/scripts/restore_reference_archive.py \
  --manifest experiments/reference-data/public-event-traces-20261022/manifest.json \
  --archive experiments/reference-data/public-event-traces-20261022/event-traces.tar.gz \
  --destination "$EVIDENCE_ROOT"
```

The restored study directories have the original relative layout, including
cross-study references. Restore only into the analysis copy. The tool rejects
changed archive bytes, unsafe members, symlink destinations and overwriting
different files. It does not execute any archived content or rerun an experiment.
