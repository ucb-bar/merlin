# Actual position bounds and composed frontend execution

This additive, inactive candidate closes the position-embedding gap recorded in
`../index-bounds/`. It preserves that earlier packet as historical evidence. Compiler passes,
W8A8 files, frozen semantics, the approved A2 specification, the production lakefile and
SPEC_LOCK are unchanged. Work stays on `pv/frontend` in its separate checkout.

## What is proved

`FrontendPositionVerification.check_sound` accepts actual exported before/after programs.
It connects successful execution of the actual repaired initial segment to execution of the
original initial segment under A2's source-only f32 repair policy. The proof composes actual
intermediate SSA environments; it does not replace omitted producers with assumed cut inputs.

The authoritative SmolVLA boundary accepts a **239-operation initial segment**, zero-based
operations 0 through 238, including every nested operation (430 operations in total). The added observation returns the
selected positions, selection count and final segment result. This proves initialization of
those observations, not of the full `forward` return. The important subgraph ends at operation
156 and has its own accepted 157-operation test.

1. The actual 31-term reductions at operations 74 and 78 select integer 0 or 1 and accumulate
   counts in `[0,32)`. Every possible comparison result is covered. No finite-input, ordering
   or monotonicity premise is needed, including for NaNs and infinities.
2. Actual integer multiplication, addition and views derive positions in `[0,1024)`. Selection
   preserves this range, including its zero seed and empty selections.
3. The embedding at operation 117 uses that selected row bound and a checked column extent
   of 768. The attention gather at operation 156 derives row/column bounds from its actual
   producers. No embedding or surrounding operation is skipped in the composed execution.
4. A proved SSA frame preserves additional captured tables, images and masks across selection.
   The continuation checks all remaining operations through 238, including their structured
   bodies, typed results and initialization.

Inputs have the original function's declared types and initialized accessible elements, with
index64 and runtime loop budget at least 1024. **No numeric input interval is supplied.**
Inaccessible tensor storage is unconstrained. Dynamic allocations initially contain poison;
initialization is proved from actual writes. Poison consumed as a size or SCF condition is
refused as stuck in this candidate, pending runtime review. Bounded loops prove finite
iteration and safe final signed advancement, including zero iterations.

The bounds checker follows actual static reduction coordinates and accumulator updates with
a proved finite abstract execution. It admits at most 2048 iteration points per such range
trace; larger unsupported range traces refuse. Unknown tensor indices, unsupported flags,
unsafe signed arithmetic ranges and uninitialized reads refuse. Empty outputs/reductions have
explicit tests. Requested output intervals are goals checked against inferred producer facts.

## Policy and floating-point scope

The complete-program repair theorem remains
`FrontendRuntimeVerification.full_policy`. Acceptance establishes equality to A2's repair
interpretation of the **original** module, including poison, UB, stuck outcomes, calls and
surrounding regions. It does not assert equivalence to the raw, potentially poison-producing
numeric fptosi program.

The only numerical repair is signed/unsigned **f32 to i1**. Both signed zeros are false; every
other initialized f32 encoding, including every NaN encoding, is true; poison remains poison.
`UneCorrect` states that the no-fastmath f32 UNE comparison with positive zero implements this
bit-level truth rule. `FrontendF32Comparison.model_correct` proves it using Lean's existing IEEE
unpack/comparison model for all 2^32 encodings. No axiom or frozen semantics edit is introduced.
Poison propagation and tensor lifting are proved separately in the execution connection.
The user-approved A2/direct-instance decision is retained. No f16/bf16/f64 repair coverage is
claimed; their production dispatch inventory remains in `../format-inventory.json`.

`Differential.lean` now has an opt-in candidate execution mode. Its host-float bridge is a
**test oracle**, not a theorem that host floating operations implement the IEEE model. Default
frozen-interpreter differential requests retain their prior behavior.

## Same-interpretation pipeline suffixes

`FrontendRuntimeChain.compose` now accepts complete alpha correspondence or adjacent identical,
closed, pure, region-free **modelled arith.constant** sharing. The latter checks actual operation fields,
result types, freshness, replacement correspondence and every subsequent use; it descends into
nested regions and preserves internal calls and symbol lookup. Checker fuel is 262144; fuel
exhaustion refuses. Other expressions and unmodelled attributes refuse. This is a narrow family,
not a general CSE or downstream-pipeline proof.

`FrontendRuntimeCse.site_sound` proves **exact** outcomes: both programs execute the first closed
expression, so UB/stuck outcomes are retained as well as successful values. The chain remains
equal to A2's source policy under the very same runtime interpretation. Frozen-model suffix
receipts do not qualify. The non-alpha receipt test uses an actual 157-operation checkpoint
with two added synthetic constants, explicitly distinguished from the unmodified production
repair boundary. Other suffix rewrites remain unsupported and must refuse.

## Qualification

Reproduce using the installed tools/environment recorded in the preceding packets:

```sh
cd merlin/lean
lake build MerlinVerify merlin-lean merlin-lean-audit
lake -f FrontendRuntime.lakefile.toml build merlin-lean-frontend-runtime
cd ../..
python -m pytest merlin/tests/ir/test_lean_frontend*.py \
  merlin/tests/ir/test_lean_spec_frontend_torch.py merlin/tests/ir/test_bool_cast_poison.py \
  merlin/tests/ir/test_lean_chain.py merlin/tests/ir/test_lean_guard.py
python out/artifacts/verification/merlin-passes/frontend/position-bounds/qualify.py
```

Every mutation counted as MLIR-valid first passes the authoritative compiler verifier and then
runs the actual Lean checker. Independent count mutations cover wrong seeds/increments,
negative values, overflow flags and fast-math flags. Actual-program mutations cover changed
counter producers, comparison predicates and bypassed selection operands. Sharing mutations
also cover unknown constant attributes and unsupported nonconstant expressions. Safe-but-wrong
predicate/use mutations can pass bounds checking and are rejected by the complete policy
checker. This distinction is intentional. Existing bool, SSA, allocation, branch and selection
faults remain in the regression run.

Native tests compare actual count cuts and independent empty/nonempty reductions against the
candidate evaluator, and check both branch outcomes and calls for closed-expression sharing.
The actual embedding generic and nested body are cloned intact, with small synthetic bounded
position/table producers and scalar observations. This avoids a huge native ABI: Lean proves
initialized execution symbolically, while native MLIR and pinned PyTorch compare 36 observed
float encodings over three masks. No literal Lean execution of the large table is claimed.
The count samples include NaNs, infinities, signed zeros, fractions and ordinary values.
The pinned oracle is **PyTorch 2.10.0+cpu**. Newer-version historical results are supplemental.
Tests qualify the proof engine's interpretation of upstream operations, without proving LLVM's
implementations. Exact measured counts, versions, skips and bindings follow in the JSON packet.

## Remaining obligations

- **Technical numeric-model/domain obligation:** operation 240 contains the first non-boolean
  f32-to-i8 conversion. The 241-operation continuation refuses. Initialized float input does
  not constrain `FloatModel.toInt`; the approved UNE instance does not change it. The existing
  `FrontendRuntimeObligations.cast8_*` witnesses make this obstruction explicit. The intervening
  operation 239 is an allocation containing poison, so observing that allocation alone also
  correctly fails initialization; that is not a separate unsupported-operation diagnosis.
- **Technical input-domain obligation:** the later token embedding at operation 7118 reads
  `tensor<49280x960xbf16>` using unconstrained initialized token words. An independently justified
  token range contract and its checked propagation are still needed. This is indexing a stored
  bf16 table, not an extension of the f32 bool numerical specification.
- **Remaining technical coverage:** successful full-model execution/initialized returns and
  the remaining operation and suffix families are not proved. The 302 float-to-i8 sites and
  token-domain contract are known obligations, not evidence that all others have been found.
- **Review/admission:** the additive runtime interpretation, statement/claim admission and
  existing A2/schema lock differences await review. This work neither edits their locks nor
  self-approves them. The f32 policy/direct IEEE instance is already approved.

The f32 repair proof and requested dynamic selection/position/attention checkpoint are closed
under the explicitly named candidate interpretation. The complete model/pipeline is not fully
proved. Further numeric or domain contracts cannot be smuggled in as initialization facts.
