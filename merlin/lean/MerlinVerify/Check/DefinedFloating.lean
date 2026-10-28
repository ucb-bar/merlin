/- Source totality for scalar floating operations. Every result remains a bit pattern under
every FloatModel; no numerical identity, reassociation or constant-folding law is assumed. -/
import MerlinVerify.Check.DefinedControl

namespace MerlinVerify.Check.Definedness

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

/-- A finite source vocabulary, independent of the model's generic math dispatcher. -/
def mathUnaryNames : List String :=
  ["math.absf", "math.sqrt", "math.rsqrt", "math.exp", "math.exp2", "math.expm1",
   "math.log", "math.log2", "math.log10", "math.log1p", "math.tanh", "math.erf",
   "math.sin", "math.cos", "math.ceil", "math.floor", "math.round", "math.roundeven", "math.trunc"]

def floatingUnaryRule : Rule := fun op fs =>
  match fs, op.results with
  | [⟨.scalar (.float k), true⟩], [r] =>
    if ((dialectOf op.name = "arith" ∧ op.name = "arith.negf") ∨
        (dialectOf op.name = "math" ∧ op.name ∈ mathUnaryNames ∧ op.name.startsWith "math." = true)) ∧
        r.ty = .float k ∧ Arith.fastmathKey op = "0" then
      some [⟨.scalar (.float k), true⟩]
    else none
  | _, _ => none

theorem floatingUnaryRule_sound (ib : Nat) : RuleSound ib floatingUnaryRule := by
  intro op fs rs hm h ctx call args _ ha
  unfold floatingUnaryRule at h
  split at h
  · next k r hr =>
    split at h
    · next hc =>
      obtain ⟨hname, hty, _⟩ := hc
      cases h
      obtain _ | ⟨hx, hnil⟩ := ha
      cases hnil
      obtain ⟨x, rfl⟩ := initialized_float hx
      have hresult : Arith.resultElt op.blind = some (.float k) := by
        simp [Arith.resultElt, Op.blind_results, hr, hty]
      rw [opSem_modelled hm]
      rcases hname with ⟨hd, hn⟩ | ⟨hd, _, hp⟩
      · simp only [dispatch, Op.blind_name, hd]
        simp only [Arith.sem, Op.blind_name, hn, Arith.unOp, Arith.pureN, hresult]
        rw [show [.scalar (.some (.flt k x))] =
          ([.some (.flt k x)] : List Val).map RVal.scalar from rfl, liftEager_scalars]
        simp only [Arith.fltN]
        exact ⟨[.scalar (.some (.flt k (ctx.M.op (Arith.floatKey op.blind) k [x])))],
          by simp, .cons ⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms],
          by simp [RVal.Defined]⟩ .nil⟩
      · simp only [dispatch, Op.blind_name, hd]
        rw [sem_math (by simpa only [Op.blind_name] using hp)]
        simp only [hresult]
        simp only [Arith.pureN, hresult]
        rw [show [.scalar (.some (.flt k x))] =
          ([.some (.flt k x)] : List Val).map RVal.scalar from rfl, liftEager_scalars]
        simp only [Arith.fltN]
        exact ⟨[.scalar (.some (.flt k (ctx.M.op (Arith.floatKey op.blind) k [x])))],
          by simp, .cons ⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms],
          by simp [RVal.Defined]⟩ .nil⟩
    · cases h
  · cases h

def floatingFmaRule : Rule := fun op fs =>
  match fs, op.results with
  | [⟨.scalar (.float k), true⟩, ⟨.scalar (.float k'), true⟩, ⟨.scalar (.float k''), true⟩], [r] =>
    if dialectOf op.name = "math" ∧ op.name = "math.fma" ∧ k' = k ∧ k'' = k ∧
        r.ty = .float k ∧ Arith.fastmathKey op = "0" then
      some [⟨.scalar (.float k), true⟩]
    else none
  | _, _ => none

theorem floatingFmaRule_sound (ib : Nat) : RuleSound ib floatingFmaRule := by
  intro op fs rs hm h ctx call args _ ha
  unfold floatingFmaRule at h
  split at h
  · next k k' k'' r hr =>
    split at h
    · next hc =>
      obtain ⟨hd, hn, hk', hk'', hty, _⟩ := hc
      subst k'
      subst k''
      cases h
      obtain _ | ⟨hx, hyz⟩ := ha
      obtain _ | ⟨hy, hz⟩ := hyz
      obtain _ | ⟨hz, hnil⟩ := hz
      cases hnil
      obtain ⟨x, rfl⟩ := initialized_float hx
      obtain ⟨y, rfl⟩ := initialized_float hy
      obtain ⟨z, rfl⟩ := initialized_float hz
      have hresult : Arith.resultElt op.blind = some (.float k) := by
        simp [Arith.resultElt, Op.blind_results, hr, hty]
      rw [opSem_modelled hm]
      simp only [dispatch, Op.blind_name, hd]
      rw [sem_math (by simp [Op.blind_name, hn])]
      simp only [hresult]
      simp only [Arith.pureN, hresult]
      rw [show [.scalar (.some (.flt k x)), .scalar (.some (.flt k y)), .scalar (.some (.flt k z))] =
        ([.some (.flt k x), .some (.flt k y), .some (.flt k z)] : List Val).map RVal.scalar from rfl,
        liftEager_scalars]
      simp only [Arith.fltN]
      exact ⟨[.scalar (.some (.flt k (ctx.M.op (Arith.floatKey op.blind) k [x, y, z])))],
        by simp, .cons ⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms],
        by simp [RVal.Defined]⟩ .nil⟩
    · cases h
  · cases h

def floatingBook (ib : Nat) : Book opSem ib where
  rule _ op fs :=
    if unmodelledAttr? op.blind = none ∧ op.regions = [] then
      (floatingUnaryRule op fs).orElse fun _ => floatingFmaRule op fs
    else none
  sound Γ op fs rs h ctx call env args hi _ ha := by
    split at h
    · next hc =>
      obtain ⟨hm, hregs⟩ := hc
      cases hu : floatingUnaryRule op fs with
      | some ys =>
        simp only [hu, Option.orElse_some, Option.some.injEq] at h
        subst rs
        simpa only [hregs, evalRegions] using floatingUnaryRule_sound ib op fs ys hm hu ctx call args hi ha
      | none =>
        simp only [hu, Option.orElse_none] at h
        simpa only [hregs, evalRegions] using floatingFmaRule_sound ib op fs rs hm h ctx call args hi ha
    · cases h

end MerlinVerify.Check.Definedness
