# Actual MACC IR → checked terms → claim

Scope: current zero-init same-format bf16/f16 policy. The mapping is a trusted parser/exporter bridge with checked raw/export/round-trip bindings; source inspection is not a proof of that bridge.

| Actual IR field | Representation/check | What it establishes or leaves open |
|---|---|---|
| Whole before/after text | Receipt snapshots, source byte length/hash, authoritative MLIR export and round trip | Exact submitted boundary; fresh replay required, not a preselected fragment |
| Function symbol/visibility/signature and module metadata | `Module`/`Op`; `ParametricModule.skeleton`, `functionShapes`, `checkFunctions` | Complete outer structure and every defined function compared; declarations/call meaning share explicit context |
| Every op name, result type, attrs/props/successors | `Op`; `Alpha.opHeadEq` | No ignored arithmetic flag/map/unknown surrounding attribute; semantics-free policy annotations only through explicit filter |
| Scalar/tensor format, dimensions, tensor encoding | `Ty` and `halfTensor?`, `siteE`, local typing | Static same half format; unsupported encodings/shapes refuse rather than infer initialization |
| Ordered operands/results and region block arguments | Numbered SSA values, `alphaOp`, fresh-ID checks, region correspondence | Source operand binding, scopes, captures, exact operation order; source arg swaps are a policy choice rather than float algebra |
| Generic operand segments | `operandSegmentSizes=[2,1]`; source `splitInsOuts` and target canonical attrs | Exactly two tensor inputs and one destination |
| Named matmul maps or generic maps | `AffineMap`; `siteE`, `contractionMaps`, `structuredRanges?` | Own source maps; dimensions/symbol count, output parallel map, derived bounds; catches base pass's default-map substitution |
| Reduction iterator order | `IterKind` list and `allPoints` | Frozen lexicographic per-cell order; native lowering fidelity is ledger A5 |
| Literal +0.0 | Exact format bits in `.float 0` and source producer identity | Negative zero differs (six pure-helper controls); no arbitrary constant folding or nonzero producer admission |
| Splat/fill producer | `zeroDestination`; `Sites.initialization?` copies actual producers | Source literal/splat, or literal/empty/fill, is in each local obligation, not assumed by name |
| Target six operations | `Reference.emit`: empty-f32, zero-f32, fill, accumulate-generic, empty-half, truncate-generic | Whole-output match prevents missing/extra/reordered/unrelated operations |
| Accumulate body | `accRegion`: extf, extf, mulf, addf, yield | Exact operand types/SSA order, accumulator as left addend, output bound to produced sum |
| Final body | `truncRegion`: truncf, yield; identity maps | Exactly one scalar truncation per output cell, old empty element unused |
| Fastmath/rounding | `License` and attribute-free conversion `FloatOpKey`s | Exactly none/contract independently; no silent reassociation/extra rounding attr; concrete backend rounding/fusion separate |
| Local candidate position | `Sites.Pair.function/index/source/target`, `collectOps_coverage` | All recognized direct function-body candidates in order, no omitted/duplicated source op |
| Complete acceptance | `Sites.check` = exact output and all `pairCheck` | Local proof cannot hide unrelated full-module change; Q must use this report section |
| Typed initialized argument domain | `HasTys ib` and `ResultsDefined`; `operand_domain_nonempty` | Every local argument tensor in the stated domain, inhabited by zeros; no whole-source-success hypothesis |
| Calls/context | `ParametricAlpha.Laws` plus complete module fuel induction | Equal outcomes in the modeled semantics under laws; no unrestricted opaque effects, native recursive divergence or ABI theorem |

## Concrete retained example

`ACTUAL_SITE_0.json` contains the exact decoded source and target operations, initializer and data bindings. `forward` source operation7624 has source result17210:

```text
left  : tensor<15x113x113xbf16>
right : tensor<15x113x64xbf16>
out   : tensor<15x113x64xbf16>
maps  : (b,m,n,k) -> (b,m,k), (b,k,n), (b,m,n)
iters : parallel, parallel, parallel, reduction
init  : tensor.splat of literal +0.0 bf16
target: f32 empty/zero/fill;
        generic with same maps and ext/ext/mul[contract]/add[contract]/yield;
        half empty; identity-map generic truncf/yield
```

`RETAINED_IR_INVENTORY.json` enumerates24 source positions and checks their historical report roster independently. It also binds all raw/source/target/report data bytes. It does not execute the Lean checker. The old segment has the same source and output data and cannot increase the count; TinyLlama is identity-only.

## Theorem-to-formula boundary

The actual admission roots establish equality/refinement to `Reference.expand`, not a single exported theorem whose conclusion is directly `Bf16F32AccContract.result`. Separate scalar, tensor and op-dispatch bridge theorems explain that reference. This is sufficient to state the checked **reference-program** claim; do not cite `Sites.check_sound` alone as if its type directly contains the closed numerical result formula. `MACC_PROOF_NOTES.md` gives an optional additive composition target for a future more convenient numerical theorem, without adding assumptions or changing the current claim.
