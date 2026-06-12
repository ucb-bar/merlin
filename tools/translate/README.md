# translate — IR translation driver

Thin CLI entrypoint. **Not implemented yet** — this directory documents intent only.

## What it will do

Translate between merlin IR and external formats.

## Backing module

`merlin/compiler/tools/translate`

## Intended usage

```bash
translate --emit=... input  ->  output
```

## Notes

CLI logic is deliberately absent at this scaffold stage. When implemented, this entrypoint
should stay thin and delegate to the backing Python module. Artifacts are written under
`output/` (gitignored).
