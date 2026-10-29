# Separate transcendental candidate — unqualified mathematical work

CURRENT — complete source softmax segment execution (new mathematical profile)

Strict builds v81-v105 now prove reviewed parent-annotation erasure, complete sixteen-op
SSA correspondence, exact last-axis max/sum tensor execution, row-major reshape/broadcast
reads, all five numerical source stages, and initialized execution of the entire segment.
PolicyW8A8AttentionExecution additionally recomputes static geometry and proves actual
straight-line execution plus the output binding. Finite cap128/nonnegative output still
requires an actual finite witness for EACH row, cap<=252 and width<=2^22; this is not an
extra assumed capture-domain condition. Captured diagonal-mask admission is STILL OPEN.
An all-masked one-by-one COMPLETE program is proved to return an initialized NaN word.
Eight new execution fixture roots pass (v2); failed v1 retained, including its mistaken
expectation that static geometry rejects a zero-width empty output. Geometry accepts it;
finite-row bounds correctly still cannot obtain a finite member of an empty row.

Actual earlier template probe v3 accepts22retained segments plus SSA renaming and rejects
three compiler-verified dataflow/seed mutations. The stronger execution/geometry probe
attention-execution-development-v1 also PASSES all22segments, rename and3refusals;
MLIR verifies5fragments and pinned source is unchanged. Full strict audit v4 PASSES
487modules/11985theorems with zero violations, including the execution modules.
No captured-mask/1533/native/review/receipt/full-pass flag has changed. Older capsules and
unexplained HOST failures remain untouched and provisional. The sealed mathematical model
source snapshot for the frontend remains byte-identical and independent of attention work.

Historical milestones below are preserved as history; worker-status TOP is authoritative.

LATEST: UnitRounding(v37), UnitAdd(v38), row denominator_unit(v39), new
DivisionBounds(v40), and strengthened row probabilities/weighted trace(v41/v42)
compile with strict options. Finite denominator≥1, finite/NN quotient cap128 and
eight-term/Vcap138 output cap146 are now proved in the mathematical candidate.
The expanded row fixture v4 passes13printed roots, standard3; all-masked NaN and
tiny nonzero divisor overflow remain. Actual source mask/cross-SSA binding is
open, so1533 still refuses. Full new audit strict-audit-development-v2 is tracked
at worker-status TOP; old10972audit does not cover these latest additions.
All older capsules and qualification failures stay preserved/unqualified as recorded.

All prior ordered/bound/joint profiles are sealed. The new HOST capsule remains provisional
after two preserved integrity-read failures; its bounded diagnostic is terminal and did not
reproduce them. The planner explicitly directs continued independent formal work here.
Scratch emits no MerlinVerify objects and is excluded from all prior delivery/qualification.
The separately named rounded-real candidate is authorized because no independently selected
transcendental interpretation exists yet. This does not claim an executable/native primitive.

The original ModelContext permits arbitrary fallback sin/cos/exp. The preserved kernel
countermodel proves finite0 can yield NaN under that context. No finite-angle rule can be
added universally to that old family. Source cos1180 andsin1193 have conservative SSA paths
to quantizer1533. Do not remove that refusal or call these dependencies an execution proof.

## Current scratch result

NEW `scratch-v2/RoundedReal.lean` compiles in `rounded-real-v10/` with only the standard three
axioms at all printed roots. RUN.json retains exact source/helper/argv/env/compiler/logs;
v1-v8 failed/successful development attempts remain intact. Explicit set interfaces avoid
unfolding the enormous finite-word universe; no recursion/fuel increase was used.

The new prototype proves all-nearest selection, preference for even significands, unique
raw-word tie resolution, zero/one encoding uniqueness, signed rounding-to-zero, exact0/1,
positive/negative half-least-subnormal ties, and overflow at2^128-2^103. It declares mathematical
rounded-real sin/cos/exp with explicit NaN/infinity/signed-zero behavior. Proved results include
finite-input sin/cos cap127, nonpositive-finite-input exp cap127/nonnegative (underflow to+0
allowed), exp(-Inf)=+0,exp(+Inf)=+Inf,cos(±0)=1,exp(±0)=1,sin(-0)=-0. Exact default unaryf32
operation keys are bound through WByte(WIEEE(transcendentalFallback base)); all7source+
3dynamic+9Agreement laws and the original ContextOk follow for that same construction.

This remains NONCOMPUTABLE mathematical semantics, unqualified and unreviewed. No native
equivalence or1533 acceptance is claimed. The six initial source modules were promoted into
NEW profile `out/build/lean-w8a8-rounded-real/repo`, then actual-key scalar, structured-region,
entry-prefix and diagnostic-driver integration compiled (module-build-v1/v3). The first actual
retained probe derives cap127 for cos1180/sin1193 on BOTH source/reference, with all prior
166/163demands still passing. Its prefixes1060/1087 and next refusal1533 remain unchanged.

The source trace exposed unconstrained arith.negf1360/1386 followed by ordinary add1372/1398.
New W8A8UnarySign defines explicit sign-bit flipping and proves unchanged magnitude/exponent;
it is a separate exact-key fallback layer inside ONLY the new unqualified model. The trig
fallback itself still overrides only exact default unaryf32 sin/cos/exp. The promoted model
is WByte(WIEEE(transcendentalFallback(W8A8UnarySign.model base))); all old7+3+9 and ContextOk
are reproved for this same constructor, and exact arith.negf binding is proved. No shared,
frozen or other worker's numerical model changed. This extra primitive remains part of the
independent numerical-review/native-fidelity obligations.

W8A8PairBounds proves a conservative actual pair-add cap=max(left,right)+2 for input caps≤252,
using the existing signed/aligned/rounded integer-unit addition theorem. No reassociation or
assumed rounding-error law. PolicyW8A8AttentionScalar checks negation/add source keys, types,
initialization and actual facts. New structured integration is in module-build-v8. The second
actual retained probe derives SOURCE neg1360cap142,neg1386cap143,rotary1372cap148/1398cap149,
logits1437cap180 and scaled1448cap179. Reference QKV facts remain absent. All prior166/163
demands pass and1533 still refuses. `retained-progress-v2.json` binds exact report/source and
25selected facts. These direct probes generate NO receipts and establish no release claim.

Raw audit-development-v1 covers447new-profile modules/10813theorems with zero violations.
Actual helper/env/argv/compiler/auditor/source/log identities are retained. Kernel fixture
v10 now passes all14printed roots standard3, including fastmath/roundingmode mutation,
arity/type/SSA/initialization controls and mathematical endpoints. Failed elaborations and
resource attempts remain, never promoted. v3 retained demands add13SOURCE/2REFERENCE bounds
(179/165total), with identical input caps/domain; actual retained-trig-development-v3 checks
all of them. Prefixes1060/1087,next1533 remain unchanged; no receipt/qualification claimed.

Three later softmax modules are additive and compile under explicit autoImplicit=false and
relaxedAutoImplicit=false; they postdate the10813audit and are NOT covered by that audit:
W8A8MaskedMaximum proves finite ordered max and attainment with at least one actual finite
member, preserving -Inf masks and the all-masked refusal. W8A8AttentionDifference proves
actual f32 self-subtraction=+0,(-Inf)-finite=-Inf, exp(selfdifference)=1 and maskedexp=0;
all-masked rows yield canonical NaN in this mathematical candidate. W8A8RealOrder proves
integer-unit magnitude monotonicity, soundness of actual byte ordering and finite maximum
domination. The new masked fixture passes7printed standard3roots. No actual masked-max/
sub/exp/sum/div checker integration or native claim follows from these components.

CURRENT next: ordered masked-row domination and subtraction nonpositivity/range, actual
diagonal mask/unmasked witness, and a proof linking the max/sub/exp/sum/div SSA values. The
existing per-value cap book alone does not carry this relation between separate tensors.
Keep that obligation explicit rather than asserting positive denominators from caps alone.
Reference QKV and final V bounds remain separate. Host qualification remains provisional.

### Later continuation: positive ordered denominator, strict full candidate

The numerical softmax components have progressed beyond the paragraph above. See
`../softmax/README.md` and worker-status TOP for the precise current state. Actual masked
row domination and signed subtraction order/range are proved. W8A8PositiveRounding and
W8A8PositiveAdd prove nonzero preservation from actual rounding/packing/addition; the new
W8A8SoftmaxRow.denominator_admissible combines finite and positive/nonzero summation for
count≤2^22. It does not yet prove a useful lower denominator/quotient cap or actual source
SSA/mask binding. The row fixture passes all6printed roots; the row closure audit3737/0
is retained separately and its theorem count is not additive with earlier audits.

All22new library/driver modules compiled with explicit strict options in module-build-v36.
The full post-strict raw audit10972/0 covers457modules. A strict --run retained probe again
checks179SOURCE/165REFERENCE bounds and preserves1060/1087,next1533. These are development
observations, not a freeze, independent source rebuild, native fidelity or HOST release.
The new standalone entry and exact interpretation remain visible in the report.

CURRENT lower-power scratch (`scratch-lower-v1/Lower.lean`) emits no objects. Its first
normal-mantissa lower bound compiles in lower-fixture-development-v1, standard3axioms.
Next derive a1threshold preserved by NN addition and a usable denominator/quotient bound.
Do not assert that positivity alone establishes finite division or a bound on its result.

The older component scratch below is historical and is preserved unchanged.

`scratch-v1/Nearest.lean` compiles in `nearest-build-v5.log`. Its printed roots have only
propext/Classical.choice/Quot.sound. Earlier failed elaborations remain separately retained.
The file defines a binary32 bit-to-real value using the exact integer-unit magnitude in
2^-149 units and the sign bit. It proves the representations of±1, finite nearest-candidate
existence over all finite words, and a range theorem: if the exact real input is in[-1,1],
every selected nearest candidate stays in[-1,1]. The integer-unit bridge gives finite raw
exponent≤127. Applying mathematical Real.sin/cos gives the corresponding candidate bounds.

This is **noncomputable mathematical selection** over the finite binary32 encoding space.
It is not an executable primitive, an implemented RNE routine, actual MLIR arithmetic or
native fidelity. Tie-even selection, signed-zero behavior, overflow, NaN/Inf rules, source
keys and FloatModel binding are NOT implemented yet. These component theorems do not change
any checker acceptance, receipt, source/reference coverage or broad verification flag.

## Dependency isolation and how to elaborate

Pinned mathlib's trig objects were already present, but eight required Batteries objects
were missing. Shared package caches remain read-only. Only missing pinned Batteries source
modules were compiled into `out/build/lean-w8a8-trig-dependencies/lib/lean`;819existing objects
are read-only symlinks to the pinned cache. Syntax was compiled first separately, then these
eight modules were compiled by `prepare_overlay.py` (some expected dependency failures are
retained): Nat.Basic,Nat.Lemmas,Tactic.Congr,List.Pairwise,Fin.Basic,List.Perm,Fin.Lemmas,RatCast.
`dependency-overlay-v1.json` records exact compiler/commands/source/output/log references,
the scratch-only search path and unchanged1452frozen records. The initial Syntax compile is
recorded in batteries-syntax-v1.log; it predates the eight-module manifest. This dependency
overlay is not a qualified new checker or a modification of any old runtime search path.

Use the pinned executable/sysroot/search_path from `../freeze-v1.json`, with only the scratch
LEAN_PATH prefixed by the overlay above, and run the standalone Nearest.lean **without -o**.
`prepare_overlay.py` is a one-shot retained attempt recorder; do not overwrite its old logs
by rerunning it. Future scratch elaborations should use new versioned logs and no library
build. No background scratch command is currently running.

## Continue next

1. Refine nearest candidates to a separately defined nearest/ties-even choice: collect all
   equal-distance nearest encodings, prefer even low significand bits when available, then
   choose a deterministic raw-word tie break. Preserve sign on rounded zero according to
   the exact real sign; handle sin(-0) explicitly because real numbers erase signed zero.
   Keep the original candidate theorem for all minimizers rather than weakening it.
2. Prove binary32 endpoint uniqueness from integer magnitudes: value1 has exponent127,
   fraction0 and positive sign, hence bits0x3f800000. Value0 has only±0 encodings. This gives
   exact exp(±0)=1 and cos(±0)=1 once the rounded-real recipe is actually defined. Retain
   subnormal/tie/underflow cases. Exp(-Inf)=+0, exp(+Inf)=+Inf; sin/cos nonfinite inputs give
   canonical NaN under the declared candidate, with no native equivalence inferred.
3. Declare finite overflow at the RNE midpoint2^128-2^103 instead of saturating all reals
   to the largest finite number. Sin/cos and exp of nonpositive arguments stay in unit
   intervals, so prove those bounds from their mathematical functions and selection theorem.
   A nearest finite candidate alone is not a full IEEE rounding definition at overflow.
4. Add a new fallback overriding only exact default unaryf32 sin/cos/exp keys, then wrap it
   with the unchanged W8A8IEEEPrimitives.model and W8A8ByteModel.extend. Existing universal
   seven-plus-three laws and nine Agreement fields then specialize to this new fallback;
   prove the new context implies the old context. No new axiom or assumed desired trig law.
5. Promote into a NEW isolated profile only after the current ordered release qualifies.
   New model/noncomputability/native/review scope stays explicit in roots/driver/receipts.
   Bind actual sin/cos source operations, finite operand facts and numerical bounds; then
   handle masking with real negative infinity, attained unmasked maxima, exp(-Inf)=0,
   exp(0)=1, positive softmax denominators and the final V contraction. Reference1533 and
   missing reconstructed reference QKV caps remain separate obligations until discharged.

Planner constraints: a rounded-real model existence/range theorem cannot be relabeled an
executable implementation or primitive fidelity. Independent semantic review and actual
capture/domain admission remain pending; all publication/shared/frozen holds persist.
