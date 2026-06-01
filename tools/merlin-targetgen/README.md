# merlin-targetgen — target dialect generator

Thin CLI entrypoint. **Not implemented yet** — this directory documents intent only.

## What it will do

ISA/docs/RTL/examples -> target_contract.yaml -> dialect_plan.yaml -> generated dialect scaffold + tests.

## Backing module

`merlin.python.merlin.targetgen`

## Intended usage

```bash
merlin-targetgen --isa ... --arch ... --examples ... --out output/targetgen/<target>
```

## Notes

CLI logic is deliberately absent at this scaffold stage. When implemented, this entrypoint
should stay thin and delegate to the backing Python module. Artifacts are written under
`output/` (gitignored).
