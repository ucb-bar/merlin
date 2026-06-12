# run — workload runner

Thin CLI entrypoint. **Not implemented yet** — this directory documents intent only.

## What it will do

Run a workload through the simulator / a runner backend.

## Backing module

`merlin.python.merlin.runtime / merlin/runtime/simulator`

## Intended usage

```bash
run --workload ... --backend simulator
```

## Notes

CLI logic is deliberately absent at this scaffold stage. When implemented, this entrypoint
should stay thin and delegate to the backing Python module. Artifacts are written under
`output/` (gitignored).
