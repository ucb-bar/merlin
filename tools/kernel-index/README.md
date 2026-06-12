# kernel-index — kernel indexer

Thin CLI entrypoint. Scans an external kernel repo and emits a normalized kernel-record index.

## What it does

Streams kernels from one source, runs the deterministic feature/motif pipeline
(`merlin.kernels.{ingest,features,emit}`), and writes a JSON index
(`{source, target, count, diagnostics, records}`) under `output/` (gitignored).

## Backing module

`merlin.kernels.cli_index:main` (installed as the `kernel-index` console script).

## Usage

```bash
kernel-index --source xnnpack  --repo $MERLIN_XNNPACK_REPO  --target rvv      --out output/kernels/xnnpack_rvv_index.json
kernel-index --source autocomp --repo $MERLIN_AUTOCOMP_REPO --target gemmini  --out output/kernels/autocomp_gemmini_index.json
kernel-index --source exo      --repo $MERLIN_EXO_REPO                        --out output/kernels/exo_index.json   # needs .[kernels-exo]
```

`--repo` may be omitted if `MERLIN_<SOURCE>_REPO` is set. `--limit N` caps kernels for dev runs.

## Notes

Source-specific parsing only; no kernel is executed except Exo specs, which are compiled to C
(failures are skipped and logged). Per-kernel work is regex/filename/signature based — zero
LLM calls — so it scales to the full corpus.
