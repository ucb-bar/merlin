# The compiler change, as a reviewable patch

`recipe_surface.patch` is the entire difference between the certified champion
(`out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0`) and this experiment's fork: two files, one
modified (`lowering/isa.py`) and one new (`lowering/recipe.py`).

**Why it lives here rather than only in the fork.** Codegen packages are tool-generated and
`.gitignore`d by repo convention, and forks are ignored too. That is right for a package and wrong
for a *result*: the compiler change IS the deliverable of this experiment, so it has to be reviewable
in a diff and reconstructible from the champion by anyone, not only present as bytes in an untracked
directory on one machine.

**Reconstructing the fork:**

```
cp -r out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0 \
      out/artifacts/targets/gemmini/gemmini_xdsl_recipe_v0
cd    out/artifacts/targets/gemmini/gemmini_xdsl_recipe_v0/mlir_oot
patch -p1 < <repo>/merlin/experiments/agent_recipe_select_v0/compiler/recipe_surface.patch
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
`merlin/tests/gemmini/test_recipe_surface.py` and re-measured before any run:

1. **With the default recipe it is the same compiler.** Byte-identical emission on 222 capsules; the
   11 that differ are exactly the shapes past the single-block bound, i.e. the ones the champion could
   not legitimately emit. Zero capsules are newly refused.
2. **It makes real model layers expressible.** 0 of 26 distinct ResNet-50 + TinyLlama contraction
   shapes satisfied both capacity bounds; all 26 do after blocking.
