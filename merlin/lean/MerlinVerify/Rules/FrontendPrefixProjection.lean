/- Prefix initialization is checked directly on the authoritative full module.
No Python slice, assumed Candidate flag, or full-model successful execution is used. -/
import MerlinVerify.Rules.FrontendPrefixDefined
import MerlinVerify.Rules.FrontendMaskProjection
import MerlinVerify.Check.TypedModule

namespace MerlinVerify.Rules.FrontendPrefixProjection
open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness

/-- The prefix and its output roots come from the same actual block as the checked cut. -/
structure Prefix (m : Module) (name : String) (stop : Nat) where
  cut : FrontendMaskProjection.Cut m name stop

def Prefix.body {m : Module} {name : String} {stop : Nat} (p : Prefix m name stop) : Region :=
  .mk [.mk p.cut.block.args ((p.cut.block.ops.take stop) ++
    [.mk "func.return" [] (p.cut.args.map (·.id)) [] [] [] []])]

def Prefix.program {m : Module} {name : String} {stop : Nat} (p : Prefix m name stop) : Module :=
  {ops := [.mk "func.func" [] []
    [("function_type", .type (.function (p.cut.block.args.map (·.ty)) (p.cut.args.map (·.ty)))),
     ("sym_name", .str "subject")] [] [p.body] []]}

def project? (m : Module) (name : String) (stop : Nat) : Option (Prefix m name stop) :=
  (FrontendMaskProjection.project? m name stop).map fun cut => ⟨cut⟩

def check (m : Module) (name : String) (stop limit ib : Nat) : Bool :=
  match project? m name stop with
  | none => false
  | some p =>
    match TypedModule.signature? ib p.program "subject" with
    | none => false
    | some (a,r) => Definedness.check (FrontendPrefixDefined.candidateBook limit ib) p.program "subject" a r

/-- Keep the result type evidence as well as initialization when lifting the existing checker. -/
theorem checked_typed {sem : OpSem} {ib : Nat} (book : Definedness.Book sem ib)
    {m : Module} {name : String} {a r : List VTy}
    (h : Definedness.check book m name a r = true) (ctx : Ctx) (hib : ctx.indexBits = ib)
    (args : List RVal) (ha : HasTys ib args a) (hd : ResultsDefined args) :
    ∃ values, denote sem ctx m name args = .ok values ∧ HasTys ib values r ∧ ResultsDefined values := by
  unfold Definedness.check at h
  split at h
  · next f hfound =>
    simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at h
    obtain ⟨hdecl, hbody⟩ := h
    obtain ⟨values, hv, ht, hd⟩ := Definedness.checkRegion_sound book hbody hib
      (by intro _ _ h; cases h) (inputFacts ha hd)
    refine ⟨values, ?_, ht, hd⟩
    simpa only [denote, evalFunc, hfound, hdecl, Bool.false_eq_true, ↓reduceIte] using hv
  · cases h

/-- The actual prefix computes typed, initialized inputs for the selection cut on every declared
initialized input. This does not assert that the remaining model executes or returns clean data. -/
theorem check_sound {m : Module} {name : String} {stop limit ib : Nat}
    (h : check m name stop limit ib = true) :
    ∃ p : Prefix m name stop, project? m name stop = some p ∧
      ∃ a r, TypedModule.signature? ib p.program "subject" = some (a,r) ∧
        ∀ ctx, ctx.indexBits = ib → ∀ args, HasTys ib args a → ResultsDefined args →
          ∃ values, denote (FrontendMaskRuntime.sem limit) ctx p.program "subject" args = .ok values ∧
            HasTys ib values r ∧ ResultsDefined values := by
  unfold check at h
  split at h
  · cases h
  · next p hp =>
    split at h
    · cases h
    · next a r hsig =>
      exact ⟨p, hp, a, r, hsig, fun ctx hib args ha hd =>
        checked_typed (FrontendPrefixDefined.candidateBook limit ib) h ctx hib args ha hd⟩

#print axioms check_sound
end MerlinVerify.Rules.FrontendPrefixProjection
