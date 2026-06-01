# merlin-dse — design-space exploration

Thin CLI entrypoint. **Not implemented yet** — this directory documents intent only.

## What it will do

Compare baseline/software_visible/hardware_managed/oracle variants for a candidate feature using a measurable cost model.

## Backing module

`merlin.python.merlin.dse`

## Intended usage

```bash
merlin-dse --design-pressure output/dse/repeated_rhs/design_pressure.json --feature resident_packed_tensor --variants baseline,software_visible,hardware_managed,oracle --out output/dse/repeated_rhs/
```

## Notes

CLI logic is deliberately absent at this scaffold stage. When implemented, this entrypoint
should stay thin and delegate to the backing Python module. Artifacts are written under
`output/` (gitignored).
