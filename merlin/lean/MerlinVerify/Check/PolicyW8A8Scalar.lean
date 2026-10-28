/- Accepted scalar regions execute the ordered W8A8 components. Whole tensor sites are separate. -/
import MerlinVerify.Check.Alpha
import MerlinVerify.Check.AlphaFast
import MerlinVerify.Check.PolicyW8A8Captures
import MerlinVerify.Rules.W8A8Scalar

namespace MerlinVerify.Check.PolicyW8A8Scalar

open MerlinVerify Rules.W8A8Scalar

def absMaxCorrect (actual : Region) : Prop :=
  ∀ ctx call env x acc, evalRegion Sem.Ops.opSem ctx call env actual
    [.scalar (floatValue x), .scalar (floatValue acc)] = .ok [.scalar (absMax ctx.M x acc)]

def reciprocalCorrect (actual : Region) : Prop :=
  ∀ ctx call env s init, evalRegion Sem.Ops.opSem ctx call env actual
    [.scalar (floatValue s), .scalar init] = .ok [.scalar (reciprocal ctx.M s)]

def requantizeCorrect (count : Nat) (actual : Region) : Prop :=
  ∀ ctx call env acc left right init, evalRegion Sem.Ops.opSem ctx call env actual
    ([.scalar (intValue acc)] ++
      ([.scalar (floatValue left), .scalar (floatValue right)].take count) ++ [.scalar init]) =
    .ok [.scalar (requantize ctx.M acc ([left, right].take count))]

def checkAbsMax (actual : Region) : Bool := alphaRegion [] absMaxRegion actual
def checkReciprocal (actual : Region) : Bool := alphaRegion [] reciprocalRegion actual
def checkRequantize (count : Nat) (actual : Region) : Bool :=
  decide (count ≤ 2) && alphaRegion [] (requantizeRegion count) actual

private theorem closed {reference actual : Region} (h : alphaRegion [] reference actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) :
    evalRegion Sem.Ops.opSem ctx call env reference = evalRegion Sem.Ops.opSem ctx call env actual :=
  alphaRegion_sound call call [] reference actual ctx env env h
    (by intro x y hxy; simp [Rho.find, List.lookup] at hxy) (fun _ _ => rfl)

theorem checkAbsMax_sound {actual : Region} (h : checkAbsMax actual = true) : absMaxCorrect actual := by
  intro ctx call env x acc
  rw [← closed h ctx call env]
  exact absMax_sound ctx call env x acc

theorem checkReciprocal_sound {actual : Region} (h : checkReciprocal actual = true) :
    reciprocalCorrect actual := by
  intro ctx call env s init
  rw [← closed h ctx call env]
  exact reciprocal_sound ctx call env s init

theorem checkRequantize_sound {count : Nat} {actual : Region}
    (h : checkRequantize count actual = true) : requantizeCorrect count actual := by
  have hh := h
  simp only [checkRequantize, Bool.and_eq_true] at hh
  obtain ⟨hc, hb⟩ := hh
  intro ctx call env acc left right init
  rw [← closed hb ctx call env]
  exact requantize_sound count (of_decide_eq_true hc) ctx call env acc left right init

def upper : FBits .f32 := 0x42fe0000#32
def lower : FBits .f32 := 0xc2fe0000#32
def captureMap (hi lo : Nat) : Rho := [(100, hi), (101, lo)]

def checkScaleKnown (known : List ConstantFacts.Fact) (denom : Nat) (actual : Region) : Bool :=
  PolicyW8A8Captures.hasKnownFloat known denom Rules.W8A8Scale.denominator &&
  alphaRegion [(100, denom)] scaleRegion actual

/-- This proves the implemented guarded formula; agreement with the unguarded formula is separate. -/
theorem checkScaleKnown_sound {known : List ConstantFacts.Fact} {denom : Nat} {actual : Region}
    (h : checkScaleKnown known denom actual = true) {env : Env}
    (valid : BindingFacts.Valid ConstantFacts.model known env) (ctx : Ctx) (call : CallSem)
    (amax : Option (FBits .f32)) (init : Val) :
    evalRegion Sem.Ops.opSem ctx call env actual
      [.scalar (floatValue amax), .scalar init] = .ok [.scalar (scale ctx.M amax)] := by
  have hh := h
  simp only [checkScaleKnown, Bool.and_eq_true] at hh
  obtain ⟨hd, hb⟩ := hh
  have hdenom := PolicyW8A8Captures.hasKnownFloat_sound hd valid
  have hc : Corr [(100, denom)] (scaleCaptures []) env := by
    intro a b hab
    by_cases ha : a = 100
    · subst a
      have : b = denom := by simpa [Rho.find, List.lookup] using hab.symm
      subst b
      rw [hdenom]
      rfl
    · have hna : (a == 100) = false := by simpa using ha
      simp [Rho.find, List.lookup, hna] at hab
  have he := alphaRegion_sound call call [(100, denom)] scaleRegion actual ctx
    (scaleCaptures []) env hb hc (fun _ _ => rfl)
  rw [← he]
  exact scale_sound ctx call [] amax init

/-- Check literal producers in the real prefix, not only the body's references to them. -/
def checkQuantizeKnown (known : List ConstantFacts.Fact) (hi lo : Nat) (actual : Region) : Bool :=
  PolicyW8A8Captures.hasKnownFloat known hi upper &&
  PolicyW8A8Captures.hasKnownFloat known lo lower &&
  alphaRegion (captureMap hi lo) quantizeRegion actual

/-- Prefix execution supplies all capture facts. Callers cannot certify arbitrary clamp bounds. -/
theorem checkQuantizeKnown_sound {known : List ConstantFacts.Fact} {hi lo : Nat} {actual : Region}
    (h : checkQuantizeKnown known hi lo actual = true) {env : Env}
    (valid : BindingFacts.Valid ConstantFacts.model known env) (ctx : Ctx) (call : CallSem)
    (x inv : Option (FBits .f32)) (init : Val) :
    evalRegion Sem.Ops.opSem ctx call env actual
      [.scalar (floatValue x), .scalar (floatValue inv), .scalar init] =
      .ok [.scalar (quantize ctx.M x inv upper lower)] := by
  have hh := h
  simp only [checkQuantizeKnown, Bool.and_eq_true] at hh
  obtain ⟨⟨hhi, hlo⟩, hb⟩ := hh
  have hupper := PolicyW8A8Captures.hasKnownFloat_sound hhi valid
  have hlower := PolicyW8A8Captures.hasKnownFloat_sound hlo valid
  have hc : Corr (captureMap hi lo) (quantizeCaptures upper lower []) env := by
    intro a b hab
    by_cases ha : a = 100
    · subst a
      have : b = hi := by simpa [captureMap, Rho.find, List.lookup] using hab.symm
      subst b
      rw [hupper]
      rfl
    · by_cases ha' : a = 101
      · subst a
        have : b = lo := by simpa [captureMap, Rho.find, List.lookup] using hab.symm
        subst b
        rw [hlower]
        rfl
      · have hna : (a == 100) = false := by simpa using ha
        have hnb : (a == 101) = false := by simpa using ha'
        simp [captureMap, Rho.find, List.lookup, hna, hnb] at hab
  have he := alphaRegion_sound call call (captureMap hi lo) quantizeRegion actual ctx
    (quantizeCaptures upper lower []) env hb hc (fun _ _ => rfl)
  rw [← he]
  exact quantize_sound ctx call [] x inv init upper lower

/-- Prefix adapters preserve the existing API; streaming clients can reuse the known-fact checks. -/
def checkScale (ib : Nat) (pre : List Op) (denom : Nat) (actual : Region) : Bool :=
  checkScaleKnown (PolicyW8A8Captures.facts ib pre) denom actual

def checkQuantize (ib : Nat) (pre : List Op) (hi lo : Nat) (actual : Region) : Bool :=
  checkQuantizeKnown (PolicyW8A8Captures.facts ib pre) hi lo actual

theorem checkScale_sound {ib : Nat} {pre : List Op} {denom : Nat} {actual : Region}
    (h : checkScale ib pre denom actual = true) (ctx : Ctx) (call : CallSem)
    (hib : ctx.indexBits = ib) (env env' : Env)
    (hs : execOps Sem.Ops.opSem ctx call env pre = .ok env')
    (amax : Option (FBits .f32)) (init : Val) :
    evalRegion Sem.Ops.opSem ctx call env' actual
      [.scalar (floatValue amax), .scalar init] = .ok [.scalar (scale ctx.M amax)] :=
  checkScaleKnown_sound h (PolicyW8A8Captures.facts_sound ib ctx call hib pre env env' hs)
    ctx call amax init

theorem checkQuantize_sound {ib : Nat} {pre : List Op} {hi lo : Nat} {actual : Region}
    (h : checkQuantize ib pre hi lo actual = true) (ctx : Ctx) (call : CallSem)
    (hib : ctx.indexBits = ib) (env env' : Env)
    (hs : execOps Sem.Ops.opSem ctx call env pre = .ok env')
    (x inv : Option (FBits .f32)) (init : Val) :
    evalRegion Sem.Ops.opSem ctx call env' actual
      [.scalar (floatValue x), .scalar (floatValue inv), .scalar init] =
      .ok [.scalar (quantize ctx.M x inv upper lower)] :=
  checkQuantizeKnown_sound h (PolicyW8A8Captures.facts_sound ib ctx call hib pre env env' hs)
    ctx call x inv init

#print axioms checkAbsMax_sound
#print axioms checkReciprocal_sound
#print axioms checkRequantize_sound
#print axioms checkQuantize_sound
#print axioms checkScale_sound
#print axioms checkScaleKnown_sound
#print axioms checkQuantizeKnown_sound

end MerlinVerify.Check.PolicyW8A8Scalar
