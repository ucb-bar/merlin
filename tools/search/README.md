# search — candidate search (grid / evolutionary / MAP-Elites)

Thin CLI entrypoint. **Not implemented yet.**

## What it will do

Run a search over candidate compiler artifacts described by a `search_space.yaml`. Method is one
of `grid | evolutionary | map_elites`. Emits regime maps / Pareto frontiers / a MAP-Elites archive
and a decision report.

## Backing module

`merlin.search` (delegates scoring to `merlin.dse.harness`)

## Intended usage

```bash
search \
  --space output/dse/spaces/resident_regime.yaml \
  --method grid \
  --out output/dse/interface_dse/
```
