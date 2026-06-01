# merlin-kernel-index — kernel indexer

Thin CLI entrypoint. **Not implemented yet** — this directory documents intent only.

## What it will do

Scan an external kernel repo and emit normalized kernel_record artifacts.

## Backing module

`merlin.python.merlin.kernels.ingest`

## Intended usage

```bash
merlin-kernel-index --source xnnpack --repo $MERLIN_XNNPACK_REPO --target rvv --out output/kernels/xnnpack_rvv_index.json
```

## Notes

CLI logic is deliberately absent at this scaffold stage. When implemented, this entrypoint
should stay thin and delegate to the backing Python module. Artifacts are written under
`output/` (gitignored).
