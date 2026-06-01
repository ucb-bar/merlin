# merlin-opt — merlin optimizer driver

Thin CLI entrypoint. **Not implemented yet** — this directory documents intent only.

## What it will do

Run merlin dialect passes over IR. Prototype plane is xDSL (`merlin.python.merlin.xdsl_dialects`); the stable plane is the C++ `merlin-opt` under `merlin/compiler/tools/merlin-opt`.

## Backing module

`merlin.python.merlin (xdsl driver) / merlin/compiler/tools/merlin-opt`

## Intended usage

```bash
merlin-opt input.mlir --pass-pipeline=...  ->  stdout
```

## Notes

CLI logic is deliberately absent at this scaffold stage. When implemented, this entrypoint
should stay thin and delegate to the backing Python module. Artifacts are written under
`output/` (gitignored).
