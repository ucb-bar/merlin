# Actual exact-integer→CSE/DCE→LICM→unroll→segment example

Producer4499, i32 data, signed64 index profile,12 original iterations, unroll
factor3 and segment tile2. This is an ACTUAL changed production chain. Complete
machine-checked generic IR buffers and digests are in EVIDENCE.json; excerpts
below use custom notation/alpha-renaming for readability and omit only context
that remains in the full checked buffers. No performance claim is made.

## Original and exact-integer result (stage0→stage1)

The actual original function takes(%x:i32,%bias:i32), declares0/12/1 index
controls and i32 constants0,1,-3,3. Its meaningful original edits are:

```mlir
%foldedzero = arith.addi %minus_three, %three : i32
%later = arith.addi %x, %foldedzero : i32
%out = scf.for %iv = %lb to %ub step %st iter_args(%acc = %later) -> i32 {
  %invariant = arith.addi %bias, %bias : i32
  %duplicate = arith.addi %bias, %bias : i32
  %neutral = arith.muli %acc, %one : i32
  %sum = arith.addi %neutral, %duplicate : i32
  %sum2 = arith.addi %sum, %invariant : i32
  func.call @watch(%sum2, %iv) : (i32,index) -> ()
  %v = func.call @provider(%sum2, %iv) : (i32,index) -> i32
  %next = arith.subi %v, %zero : i32
  scf.yield %next : i32
}
func.call @final(%out) : (i32) -> ()
func.return %out : i32
```

Every original word add/sub/mul above has actual overflowFlags=none. Stage1
replaces foldedzero with a constant0, uses%x as the initial carry, uses%acc
directly in%sum, and yields%v directly. Other operations and their order remain.
The reused literal%foldedzero may remain dead at this stage; the later DCE owns
its removal. The complete original also contains a dead subtraction, explicit
provider/watch/final declarations and an untouched identity function.

The first fold is a BitVec constant identity modulo32; the next use observes the
updated SSA owner and eliminates+0. Multiplication by1 and subtraction by0 require
canonical live%acc and%v. Their representation is derived for every reached
iteration from typed initial inputs and the LOCAL successful provider-result
contract, including captures/carries. Merely seeing an i32 slot would be
insufficient. On i8 input256, removing+0 is refuted by the preserved countermodel.

## Remaining actual changed stages

Stage1→stage2 removes the dead arithmetic/constants and CSEs the duplicate
bias+bias. Stage2→stage3 hoists the single total pure invariant to immediately
before the loop. Its body still computes acc+invariant, then adds invariant,
then executes watch followed by provider, and yields the provider result.
The final observer and untouched function remain explicitly bound.

Stage3→stage4 changes index step1 to3 and clones the entire scalar/effect body
three times. The lanes use iv,iv+1,iv+2, with the ordered provider carry flowing
into the next lane. Stage4→stage5 nests those grouped executions under an outer
step6 and inner step3 loop; the inner upper bound is minui(outer+6,12). Watch/
provider order, lane offsets, every carry/yield, the final observer and returned
value remain in the checked complete program. W-derived signed/control proofs
justify the grouping and minimum. The actual unabridged IR and kernel proof
are authoritative; this explanatory account is not a producer witness.

## Proof path and output

Both actual programs of the arithmetic prefix are independently mapped into
Scalar.Program and kernel-normalized into Code. A checked local rewrite sequence
relates every live expression under canonical word typing. Unused total pure
expressions are handled by the old normalize_sound theorem; unknown partial
operations are retained, never dropped as dead. The relation's operational
soundness preserves the complete returned/error/world outcome through nested
loops and calls.

The exactMiddle theorem identifies the whole prefix-result entry roster with
the suffix's actual original entries in the SAME registry/type table. It then
composes the sealed four-pass theorem instantiated on the SAME word/index/
external interpretation. There is no equation between independently numbered
cells or an assumed whole-final equivalence.

EVIDENCE.json records kernel returncode0, the exact root statements, actual
transitive std3 audit and the original/fresh-replay receipts once qualification
completes. For a separately named concrete provider model on the actual
factor2/tile3 chain, canonical inputs[3,5],world7 return[17],world32 at both ends;
an optional first-observer fault gives error17/world8. This demonstrates actual
inhabited success/failure in the disclosed model, not physical runtime delivery.

## Negative binding

The retained compiler-valid exposed-site mutation changes a later folded constant
consumed by a watch call. The complete graph check refuses the changed ordered
observation. A separate ordinary Lean refusal rejects the false bit certificate
word8(0)=word8(1). Exact source/mutated/checker/log paths are in EVIDENCE.json.
No timeout, signal or setup error is counted as a semantic refusal.

Qualification and independent review are distinct. The v1 measurement is held
because its host process classifier was incomplete; only corrected-source v2
can supply final engineering qualification. Native/poison/primitive/model/
broker/full-graph and gSIM pipeline-figure claims remain outside this example.
