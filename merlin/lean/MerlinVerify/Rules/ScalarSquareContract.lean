/- Provisional numerical contract. This is a hypothesis about a chosen FloatModel, not an
axiom, not a reviewed Spec entry, and not a theorem about libm or IEEE pow implementations. -/
import MerlinVerify.Sem.Ops
import MerlinVerify.Check.RulesArith

namespace MerlinVerify.Rules.ScalarSquareContract

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def admitted : FKind → Bool
  | .f32 | .f64 => true
  | _ => false

/-- Exact IEEE encodings of positive two; other formats are never admitted by the checker. -/
def two (k : FKind) : FBits k :=
  BitVec.ofNat k.width (if k = .f64 then 4611686018427387904 else 1073741824)

def primitive (name : String) (attrs : List (String × Attr)) (k : FKind) : Op :=
  .mk name [⟨0, .float k⟩] [1, 2] attrs [] [] []

/-- Absent/none or contract only. No inference of reassociation, approximate functions,
finite-only operands/results or insignificant signed zeros is made. -/
def flags (attrs : List (String × Attr)) : Bool :=
  let semantic := attrs.filter (fun entry => !Spec.Annotations.semanticsFree entry.1)
  decide (semantic = [] ∨ semantic = [("fastmath", .enum "arith" "fastmath" "none")] ∨
    semantic = [("fastmath", .enum "arith" "fastmath" "contract")])

/-- Exact returned-bit equality, including NaN payload/sign, signed zeros, infinities,
subnormals, underflow and overflow. Flags are part of each key and must be preserved exactly.
This deliberately strong law is provisional; real libm counterexamples must not be erased.
The model has no errno, floating exception flags/traps, or changing rounding environment. -/
def Law (M : FloatModel) (k : FKind) (attrs : List (String × Attr)) : Prop :=
  ∀ x : FBits k,
    M.op (Arith.floatKey (primitive "math.powf" attrs k)) k [x, two k] =
    M.op (Arith.floatKey (primitive "arith.mulf" attrs k)) k [x, x]

def multiply (source : Op) (base : Nat) : Op :=
  .mk "arith.mulf" source.results [base, base] source.attrs source.props [] []

/-- The actual scalar operations agree under the numerical law, for every bit pattern.
The claim is local and typed; no successful execution of a surrounding capture is assumed. -/
theorem scalar_sound (ctx : Ctx) (call : CallSem) (k : FKind)
    (r : Value) (base exponent : Nat) (attrs : List (String × Attr))
    (typed : r.ty = .float k)
    (powModelled : unmodelledAttr? (primitive "math.powf" attrs k).blind = none)
    (mulModelled : unmodelledAttr? (primitive "arith.mulf" attrs k).blind = none)
    (numerical : Law ctx.M k attrs) (x : FBits k) :
    opSem ctx call (.mk "math.powf" [r] [base, exponent] attrs [] [] [])
      [.scalar (.some (.flt k x)), .scalar (.some (.flt k (two k)))] [] =
    opSem ctx call (.mk "arith.mulf" [r] [base, base] attrs [] [] [])
      [.scalar (.some (.flt k x)), .scalar (.some (.flt k x))] [] := by
  cases r with
  | mk id ty =>
    cases typed
    have hp : unmodelledAttr? (Op.mk "math.powf" [⟨id, .float k⟩] [base, exponent]
      attrs [] [] []).blind = none := powModelled
    have hm : unmodelledAttr? (Op.mk "arith.mulf" [⟨id, .float k⟩] [base, base]
      attrs [] [] []).blind = none := mulModelled
    rw [opSem_modelled hp, opSem_modelled hm]
    have hdPow : dialectOf "math.powf" = "math" := by cbv
    have hdMul : dialectOf "arith.mulf" = "arith" := by cbv
    simp only [dispatch, Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
      hdPow, hdMul]
    simpa [Arith.sem, Arith.pureN, Arith.binOp, Arith.resultElt,
      Op.name, Op.results, Op.attrs, Op.props, Op.regions, Op.successors,
      liftEager, Arith.fltN, Arith.fltBin, Outcome.bind, primitive,
      Arith.floatKey, Arith.fastmathKey, Arith.attrKey, Op.attr?]
      using congrArg (fun bits => Outcome.ok [RVal.scalar (.some (.flt k bits))]) (numerical x)


/-- Poison is separate from NaN. Both operations propagate a poison base in the shared
semantics, without requiring the numerical law. -/
theorem poison_sound (ctx : Ctx) (call : CallSem) (k : FKind)
    (r : Value) (base exponent : Nat) (attrs : List (String × Attr))
    (typed : r.ty = .float k)
    (powModelled : unmodelledAttr? (primitive "math.powf" attrs k).blind = none)
    (mulModelled : unmodelledAttr? (primitive "arith.mulf" attrs k).blind = none) :
    opSem ctx call (.mk "math.powf" [r] [base, exponent] attrs [] [] [])
      [.scalar .poison, .scalar (.some (.flt k (two k)))] [] =
    opSem ctx call (.mk "arith.mulf" [r] [base, base] attrs [] [] [])
      [.scalar .poison, .scalar .poison] [] := by
  cases r with
  | mk id ty =>
    cases typed
    have hp : unmodelledAttr? (Op.mk "math.powf" [⟨id, .float k⟩] [base, exponent]
      attrs [] [] []).blind = none := powModelled
    have hm : unmodelledAttr? (Op.mk "arith.mulf" [⟨id, .float k⟩] [base, base]
      attrs [] [] []).blind = none := mulModelled
    rw [opSem_modelled hp, opSem_modelled hm]
    have hdPow : dialectOf "math.powf" = "math" := by cbv
    have hdMul : dialectOf "arith.mulf" = "arith" := by cbv
    simp only [dispatch, Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
      hdPow, hdMul]
    simp [Arith.sem, Arith.pureN, Arith.binOp, Arith.resultElt, Op.name, Op.results,
      liftEager, Arith.fltN, Arith.fltBin, Outcome.bind]

/-- Numerical-law assumptions are satisfiable; the theorem does not quantify an impossible
law over every FloatModel. This witness says nothing about a native implementation. -/
def zeroModel : FloatModel where
  op := fun _ _ _ => 0
  cmp := fun _ _ _ _ => false
  convert := fun _ _ _ _ => 0
  toInt := fun _ _ _ _ => none
  ofInt := fun _ _ _ _ => 0

theorem law_nonempty (k : FKind) (attrs : List (String × Attr)) :
    ∃ M, Law M k attrs := ⟨zeroModel, fun _ => rfl⟩

def counterModel : FloatModel :=
  { zeroModel with op := fun key _ _ => if key.op = "math.powf" then 0 else 1 }

/-- No universal pow-square identity follows from the existing parametric float semantics. -/
theorem law_not_universal : ¬ (∀ M, Law M .f32 []) := by
  intro claimed
  have witness := claimed counterModel 0
  cbv at witness
  contradiction

#print axioms law_not_universal

#print axioms scalar_sound
#print axioms law_nonempty
#print axioms poison_sound

end MerlinVerify.Rules.ScalarSquareContract
