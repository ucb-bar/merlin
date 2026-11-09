# The compiler change, as a reviewable patch

This is a historical experiment. The compiler package is not shipped in this repository;
reconstruction requires a separately acquired package at the **exact** pinned digest. An unrelated
version of the published OOT backend is not a substitute. `recipe_surface.patch` is the entire
difference between that certified champion and this experiment's fork: two files, one
modified (`lowering/isa.py`) and one new (`lowering/recipe.py`).

**Why it lives here rather than only in the fork.** Codegen packages are tool-generated and
`.gitignore`d by repo convention, and forks are ignored too. That is right for a package and wrong
for a *result*: the compiler change IS the deliverable of this experiment, so it has to be reviewable
in a diff and reconstructible from the champion by anyone, not only present as bytes in an untracked
directory on one machine.

**Reconstructing the fork, only after obtaining the exact external package:** Set
`MERLIN_RECIPE_FROZEN_PACKAGE_ROOT` to its absolute root. Run
`PYTHONPATH=packages/merlin-experiments/src:src python
experiments/studies/agent_recipe_select_v0/scripts/_track.py` from the repository root before copying.
It checks the pinned `SHA256SUMS` digest and every listed file. If the matching package is
unavailable, replay is blocked; do not infer equivalence from a similarly named branch.

```
cp -r "$MERLIN_RECIPE_FROZEN_PACKAGE_ROOT" \
      out/artifacts/targets/gemmini/gemmini_xdsl_recipe_v0
cd    out/artifacts/targets/gemmini/gemmini_xdsl_recipe_v0/mlir_oot
patch -p1 < <repo>/experiments/studies/agent_recipe_select_v0/compiler/recipe_surface.patch
```

Then `_track.mint_fork()` freezes it into a content-addressed, 0444 package that a run may consume.
The patch was verified by reconstruction: applied to a fresh copy of the champion it reproduces the
working fork byte for byte.

**What the patch contains.**

* `lowering/recipe.py` (new) — the exposed surface and the compiler's own statement of what is legal
  for a shape: `Recipe`, `CHOICES`, `fit` (does the whole shape fit both stores at once),
  `derive_blocks` / `blocks` (the capacity-fit cut), `catalog` (the agent-facing view).
* `lowering/isa.py` — `_matmul_trace` takes a recipe, and its K→N→M nest is emitted per block.

**The two properties that make it citable**, both gated in
`tests/gemmini/test_recipe_surface.py` and re-measured before any run:

1. **With the default recipe it is the same compiler.** Byte-identical emission on 222 capsules; the
   11 that differ are exactly the shapes past the single-block bound, i.e. the ones the champion could
   not legitimately emit. Zero capsules are newly refused.
2. **It makes real model layers expressible.** 0 of 26 distinct ResNet-50 + TinyLlama contraction
   shapes satisfied both capacity bounds; all 26 do after blocking.
