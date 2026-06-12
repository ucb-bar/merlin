# merlin-kernel-extract — kernel feature/policy extractor

Thin CLI entrypoint. **Not implemented yet** — this directory documents intent only.

## What it will do

Turn kernel indexes into abstraction_candidates and policy_rules.

## Backing module

`merlin.python.merlin.kernels.features / .emit`

## Intended usage

```bash
merlin-kernel-extract --inputs output/kernels/*.json --out output/kernels/abstraction_candidates.yaml --policies output/kernels/policy_rules.yaml
```

## Notes

CLI logic is deliberately absent at this scaffold stage. When implemented, this entrypoint
should stay thin and delegate to the backing Python module. Artifacts are written under
`output/` (gitignored).
