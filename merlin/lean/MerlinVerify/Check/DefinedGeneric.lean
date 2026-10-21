/- Structured initialized execution with explicit operation semantics and refinement facts.
The supplied interpretation must prove agreement on the generic operation head; body meanings
are checked under that interpretation, including captures, bounds and initialization. -/
import MerlinVerify.Check.DefinedBounds

namespace MerlinVerify.Check.Definedness.Structured

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

/-- Reusing the frozen generic executor requires a proved semantic law, including arbitrary
body closures. A certificate or a successful runtime sample cannot supply this law. -/
def GenericAgreement (sem : OpSem) : Prop :=
  ∀ ctx call op args regions, op.name = "linalg.generic" →
    sem ctx call op args regions = Sem.Ops.opSem ctx call op args regions

variable {Info : Type} (M : Refined.Model Info) {sem : OpSem}

def initializedRule (ib : Nat) (bodyBook : Refined.Book M sem ib)
    (Γ : Refined.FEnv Info) (op : Op) (fs : List (Refined.Fact Info)) : Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts (α := Definedness.Fact) op (fs.map (·.base)), attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, outs), some maps, some iters, [body] =>
      if (∀ f ∈ outs, f.initialized = true) ∧ (∀ f ∈ outs, f.ty.isTensor = true) ∧
          op.results.length = outs.length ∧
          (structuredRanges? maps iters.length ((ins ++ outs).map (·.ty))).isSome ∧
          Refined.checkRegion bodyBook Γ body (((ins ++ outs).map elementFact).map (Refined.emptyFact M))
            (outs.map fun f => f.ty.eltScalar) then
        some ((clean (outs.map (·.ty))).map (Refined.emptyFact M))
      else none
    | _, _, _, _ => none
  else none

theorem initializedRule_sound (agreement : GenericAgreement sem) (ib : Nat) (bodyBook : Refined.Book M sem ib) : ∀ Γ op fs rs,
    initializedRule M ib bodyBook Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds M ib Γ env → Refined.FactsHold M ib args fs →
    ∃ vs, sem ctx call op args (evalRegions sem call env op.regions) = .ok vs ∧
      Refined.FactsHold M ib vs rs := by
  intro Γ op fs rs h ctx call env args hib hΓ ha
  unfold initializedRule at h
  split at h
  · next hc =>
    obtain ⟨hm, hd, hn⟩ := hc
    split at h
    · next ins outs maps iters body hsplit hmaps hiters hregs =>
      split at h
      · next hc =>
        obtain ⟨hi, ht, hres, hranges, hbody⟩ := hc
        obtain ⟨ranges, hranges⟩ := Option.isSome_iff_exists.1 hranges
        cases h
        obtain ⟨xs, ys, hio, hxs, hys⟩ := splitInsOuts_facts hsplit ha.base
        have hclean := facts_eq_clean outs hi
        have hys' : FactsHold ib ys (clean (outs.map (·.ty))) := hclean ▸ hys
        have hb : BodyTotal ib (fun ctx => evalRegion sem ctx call env body)
            ((ins ++ clean (outs.map (·.ty))).map elementFact)
            ((outs.map (·.ty)).map VTy.eltScalar) := by
          intro ctx hib args ha
          rw [← hclean] at ha
          simpa only [List.map_map, Function.comp_def] using
            Refined.checkRegion_sound bodyBook hbody hib hΓ (Refined.FactsHold.empty ha M)
        have hs : StructuredOk maps iters.length (ins.map (·.ty) ++ outs.map (·.ty)) ranges := by
          simpa only [List.map_append] using structuredRanges?_spec hranges
        have hout : ∀ t ∈ outs.map (·.ty), t.isTensor = true := by
          intro t ht'
          obtain ⟨f, hf, rfl⟩ := List.mem_map.1 ht'
          exact ht f hf
        obtain ⟨rs, hsem, hrf⟩ := genericSem_total hib hs hout hxs hys' hb
        refine ⟨rs, ?_, Refined.FactsHold.empty hrf M⟩
        have hlen : rs.length = op.blind.results.length := by
          rw [HasTys.length (clean_typed hrf), List.length_map, Op.blind_results,
            List.length_map, hres]
        rw [agreement ctx call op args (evalRegions sem call env op.regions) hn,
          linalg_sem_of hm hd (.inl hn)]
        simp only [hregs, evalRegions, Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hio,
          attrAffineMaps?_blind, hmaps, iterKinds?_blind, hiters, Outcome.bindOp_eq, hsem,
          Outcome.bind_ok, hlen, ↓reduceIte, Outcome.pure_eq]
      · cases h
    · cases h
  · cases h

def overwriteRule (ib : Nat) (bodyBook : Refined.Book M sem ib)
    (Γ : Refined.FEnv Info) (op : Op) (fs : List (Refined.Fact Info)) : Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts (α := Definedness.Fact) op (fs.map (·.base)), attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, [⟨.tensor sh e, _⟩]), some maps, some iters, [body] =>
      if op.results.length = 1 ∧ maps.drop ins.length = [identityMap sh.length] ∧
          structuredRanges? maps iters.length (ins.map (·.ty) ++ [.tensor sh e]) = some sh ∧
          Refined.checkRegion bodyBook Γ body
            (((ins ++ [(⟨.tensor sh e, false⟩ : Definedness.Fact)]).map elementFact).map (Refined.emptyFact M)) [.scalar e] then
        some ((clean [.tensor sh e]).map (Refined.emptyFact M))
      else none
    | _, _, _, _ => none
  else none

theorem overwriteRule_sound (agreement : GenericAgreement sem) (ib : Nat) (bodyBook : Refined.Book M sem ib) : ∀ Γ op fs rs,
    overwriteRule M ib bodyBook Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds M ib Γ env → Refined.FactsHold M ib args fs →
    ∃ vs, sem ctx call op args (evalRegions sem call env op.regions) = .ok vs ∧
      Refined.FactsHold M ib vs rs := by
  intro Γ op fs rs h ctx call env args hib hΓ ha
  unfold overwriteRule at h
  split at h
  · next hc =>
    obtain ⟨hm, hd, hn⟩ := hc
    split at h
    · next ins sh e init maps iters body hsplit hmaps hiters hregs =>
      split at h
      · next hc =>
        obtain ⟨hres, hout, hranges, hbody⟩ := hc
        cases h
        obtain ⟨xs, ys, hio, hxs, hys⟩ := splitInsOuts_facts hsplit ha.base
        obtain _ | ⟨ht, hnil⟩ := hys
        cases hnil
        rename_i t
        cases t with
        | scalar _ => cases ht.1
        | tensor t =>
          have hb : BodyTotal ib (fun ctx => evalRegion sem ctx call env body)
              ((ins ++ [(⟨.tensor sh e, false⟩ : Fact)]).map elementFact) [.scalar e] := by
            intro ctx hib args ha
            exact Refined.checkRegion_sound bodyBook hbody hib hΓ (Refined.FactsHold.empty ha M)
          obtain ⟨rs, hsem, hrf⟩ := overwriteSem_total hib (structuredRanges?_spec hranges) hout hxs ht.1 hb
          refine ⟨rs, ?_, Refined.FactsHold.empty hrf M⟩
          have hlen : rs.length = op.blind.results.length := by
            rw [HasTys.length (clean_typed hrf), Op.blind_results, List.length_map, hres]
            rfl
          rw [agreement ctx call op args (evalRegions sem call env op.regions) hn,
          linalg_sem_of hm hd (.inl hn)]
          simp only [hregs, evalRegions, Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hio,
            attrAffineMaps?_blind, hmaps, iterKinds?_blind, hiters, Outcome.bindOp_eq, hsem,
            Outcome.bind_ok, hlen, ↓reduceIte, Outcome.pure_eq]
      · cases h
    · cases h
  · cases h


/-- Select the proof for the actual destination facts. An uninitialized destination requires
complete overwrite, including the checked output mapping and every yielded element. -/
def genericBook (agreement : GenericAgreement sem) (ib : Nat)
    (bodyBook : Refined.Book M sem ib) : Refined.Book M sem ib where
  rule Γ op fs := match splitInsOuts (α := Definedness.Fact) op (fs.map (·.base)) with
    | some (_, outs) => if outs.all (·.initialized) then initializedRule M ib bodyBook Γ op fs
        else overwriteRule M ib bodyBook Γ op fs
    | none => none
  sound Γ op fs rs h ctx call env args hib hΓ ha := by
    split at h
    · split at h
      · exact initializedRule_sound M agreement ib bodyBook Γ op fs rs h ctx call env args hib hΓ ha
      · exact overwriteRule_sound M agreement ib bodyBook Γ op fs rs h ctx call env args hib hΓ ha
    · cases h

#print axioms initializedRule_sound
#print axioms overwriteRule_sound
#print axioms genericBook

end MerlinVerify.Check.Definedness.Structured
