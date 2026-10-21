# Full-model execution: remaining obligations after the runtime checkpoint

This is the current technical inventory, **not a certificate that every other obligation has
been found**. It concerns the proof engine's interpretation of upstream operations. It does
not ask for a proof of LLVM's implementations.

## Established checkpoint

The candidate accepts the authoritative full SmolVLA before/after modules. The actual repaired
`forward` contains 30,502 top-level operations and 816 arguments. Its first 116 operations,
including their nested regions (212 actual operations in total), execute successfully under
`FrontendMaskRuntime.sem 1024`, index64 and the approved direct f32 comparison instance, from
any inputs satisfying the block's declared types and element initialization. The observation
return is added after that actual initial segment. Input tensors need not be canonical.

The proof checks all 91 prefix operations and composes their **actual intermediate SSA state**
with the 25-operation selection. It establishes the cut input shapes and initialization,
initialized sizes/conditions, bounds within the selection, loop termination without signed
counter overflow, and typed initialized selected results. No Python boolean establishes these
facts. `FrontendRuntimeCheckpoint.Scope.body_actual` identifies the actual source slice.

`FrontendRuntimeVerification.full_policy` separately proves complete-module equality to A2's
source-only f32 repair policy, including poison, UB and stuck outcomes. This is not raw-source
numeric-cast equivalence and does not claim successful full-model execution.

## Concrete technical obligations

| Obligation | Exact current evidence | What is still needed |
|---|---|---|
| Three tensor reads outside the selection | Reads in top-level generic operations 117, 156 and 7118; all eight reads are enumerated in `inventory.json` | Check and prove bounds from actual producer values; initialization alone is insufficient |
| Position embedding indices | Operation 117 reads argument 2, `tensor<1024x768xf32>`, using selected integer positions and loop column | A producer range proof that every position is in `[0,1024)`, plus the checked column bound |
| Attention-mask indices | Operation 156 reads value 1079 using index tensors 1119 and 1129 | Actual producer range proofs for both coordinates |
| Token embedding indices | Operation 7118 reads argument 271, `tensor<49280x960xbf16>`, indexed by argument 812's i64 token words and loop column | An explicit input-domain contract and its checked propagation; see proposal below |
| Non-boolean float conversions | All 302 remaining casts are `arith.fptosi` to i8; the Lean inventory checks all against `cast8` | A justified interpretation/domain contract for defined conversion at every site; the approved UNE instance does not constrain `FloatModel.toInt` |
| Remaining whole-model support, typing and definedness | Census contains 51,781 operations and four dynamic-result operations; the new initialization book checks the actual prefix | A complete checker/proof for the remaining program after the checkpoint, including all operations' safety and poison obligations |
| Transformed pipeline suffixes | `FrontendRuntimeChain.compose` discharges its suffix premise through an actual checker under the same interpretation; current checker requires complete alpha correspondence | Transport/reprove each non-alpha transformation's rule under the candidate, retaining its original policy claim; ordinary frozen-model receipts cannot be reused |

The numeric task remains bounded to f32. The BF16 token table is only a stored element type in
an indexing obstruction; no BF16 numerical theorem or frontend repair coverage is introduced.
The production bool pass's f16/bf16/f64 cases remain explicitly deferred.

### Why universal initialized full-model execution cannot follow from the current domain

These are **proved local obstructions**, not native samples and not whole-model counterexamples:

- `FrontendRuntimeObligations.token_upper_ub` and `token_negative_ub` prove that the actual
  matching `tensor.extract` interpretation returns UB for initialized indices 49280 and -1
  into a `[49280,960]` table. The current capsule declares the i64 token tensor's shape/type,
  without the required token-value bound. No claim is made that an entire model execution
  necessarily reaches that operation for these values.
- `FrontendRuntimeObligations.cast8_poison` proves that every matching actual i8 cast produces
  poison from an initialized f32 operand in a permitted `FloatModel` whose `toInt` returns
  `none`. Overriding f32 UNE preserves that field. Thus UNE correctness alone cannot prove
  defined i8 conversion, even for initialized inputs. This is not a counterexample to the
  bool-repair theorem; it exhibits a distinct missing numeric contract.

Even a future concrete IEEE float-to-integer instance will require addressing NaN and range
failures. Finite/range conditions must be proved from actual operations or explicitly specified
and reviewed; native samples cannot discharge these universal obligations. No new axiom,
conversion law, input restriction, or frozen-file change was introduced here.

## Small explicit domain/interpretation proposals for review

These proposals are **not active assumptions** and have not been self-approved.

1. **Token input domain:** for every in-bounds element of argument 812, `tensor<1x48xi64>`,
   require its signed i64 interpretation `t` to satisfy `0 ≤ t ∧ t < 49280`. Keep the existing
   shape/type/initialization requirements. This rejects negative/out-of-vocabulary tokens;
   it does not clamp or wrap them, alter table elements, or silently restrict other inputs.
   The embedding's column bound remains a separate proof obligation.
2. **Float-to-i8 contract:** first select and review an exact interpretation for the existing
   `FloatModel.toInt` keys used by these 302 sites. Preserve its poison result on invalid
   conversion. Then prove every actual operand meets its definedness conditions, or explicitly
   report that full-model return initialization is unavailable for the declared input domain.
   A blanket promise that all conversions succeed would be unjustified. This proposal supplies
   no new law and does not change A2's approved f32 bool policy.
3. **Runtime admission:** review the existing additive runtime interpretation and candidate
   integration patch. Its poison sizes/SCF conditions refuse with `stuck`; it does not introduce
   poison-to-UB semantics. The production dispatcher, lakefile, locks and compiler remain unchanged.

## Scope of the suffix result

The candidate chain checks every supplied stage from its actual exported AST. Qualification
supplies an alpha-identical stage and also tests refusal of a changed comparison in a suffix.
It does not claim that Merlin's entire production pipeline is alpha correspondence, that
W8A8 is ordinary equality, or that all transformed suffixes are already verified under this
candidate. Source-policy matching is preserved as the chain's conclusion.
