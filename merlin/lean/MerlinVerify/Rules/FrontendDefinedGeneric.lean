/- Context-aware generic initialization, using the existing frozen loop evaluator.
The depth adaptation is a proof device; at every actual iteration it is the identity. -/
import MerlinVerify.Rules.FrontendDefinedIndex

namespace MerlinVerify.Rules.FrontendDefinedGeneric
open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness MerlinVerify.Sem.Ops
open Sem.Ops.Linalg FrontendDefinedFrame

/-- Supply an arbitrary frame only outside the contexts actually used by the generic.
No execution claim substitutes that outside-frame behavior for the original program. -/
def framed (depth : Nat) (body : Ctx → RegionSem) : Ctx → RegionSem := fun ctx =>
  body {ctx with loopIndex := if depth ≤ ctx.loopIndex.length then ctx.loopIndex else List.replicate depth 0}

theorem framed_eq (depth : Nat) (body : Ctx → RegionSem) (ctx : Ctx)
    (h : depth ≤ ctx.loopIndex.length) : framed depth body ctx = body ctx := by
  simp only [framed, h, ↓reduceIte]

theorem fold_congr {α β : Type} (f g : β → α → Outcome β) (xs : List α)
    (h : ∀ x ∈ xs, ∀ state, f state x = g state x) (initial : β) :
    xs.foldlM f initial = xs.foldlM g initial := by
  induction xs generalizing initial with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldlM_cons]
    rw [h x (List.mem_cons_self ..) initial]
    apply congrArg (Outcome.bind (g initial x))
    funext state
    exact ih (fun y hy => h y (List.mem_cons_of_mem _ hy)) state

/-- The loop evaluator supplies exactly one coordinate per iterator. Hence frame adaptation
cannot change any body call that the actual checked generic makes. -/
theorem generic_framed {ib : Nat} {maps : List AffineMap} {iters : List IterKind}
    {inTys outTys : List VTy} {ranges : List Nat} {ins : List RVal} {outs : List Tensor}
    (hs : StructuredOk maps iters.length (inTys ++ outTys) ranges)
    (hi : HasTys ib ins inTys) (ho : HasTys ib (outs.map RVal.tensor) outTys)
    (ctx : Ctx) (body : Ctx → RegionSem) :
    genericSem ctx maps iters ins (outs.map RVal.tensor) (framed iters.length body) =
      genericSem ctx maps iters ins (outs.map RVal.tensor) body := by
  rw [genericSem_execute hs hi ho, genericSem_execute hs hi ho]
  congr 1
  apply fold_congr
  intro pt hp cur
  have hb := (mem_allPoints ranges pt).mp hp
  have hlen : iters.length ≤ pt.length := by
    have he : ranges.length = iters.length := by simpa only [List.length_range] using optionMapM_length _ _ _ hs.loops
    have hp' : pt.length = ranges.length := hb.1
    omega
  simp only [loopStep, bodyAt]
  rw [framed_eq iters.length body {ctx with loopIndex := pt} hlen]

/-- A frame-checked body can use linalg.index, while ordinary leaves keep their old proofs. -/
theorem framed_total {ib depth : Nat} (book : BookAt opSem ib depth) {Γ : FEnv} {body : Region}
    {fs : List Fact} {ys : List VTy} (h : checkRegion book Γ body fs ys = true)
    (call : CallSem) (env : Env) (hΓ : EnvHolds ib Γ env) :
    BodyTotal ib (framed depth (fun ctx => evalRegion opSem ctx call env body)) fs ys := by
  intro ctx hib args ha
  unfold framed
  apply checkRegion_sound book h (ctx := {ctx with loopIndex :=
    if depth ≤ ctx.loopIndex.length then ctx.loopIndex else List.replicate depth 0}) hib ?_ hΓ ha
  split <;> simp_all

def overwriteRule (ib : Nat) (bodyBooks : (depth : Nat) → BookAt opSem ib depth)
    (Γ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts op fs, attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, [⟨.tensor sh e, _⟩]), some maps, some iters, [body] =>
      if op.results.length = 1 ∧ maps.drop ins.length = [identityMap sh.length] ∧
          structuredRanges? maps iters.length (ins.map (·.ty) ++ [.tensor sh e]) = some sh ∧
          FrontendDefinedFrame.checkRegion (bodyBooks iters.length) Γ body ((ins ++ [(⟨.tensor sh e, false⟩ : Fact)]).map elementFact) [.scalar e] then
        some (clean [.tensor sh e])
      else none
    | _, _, _, _ => none
  else none

theorem overwriteRule_sound (ib : Nat) (bodyBooks : (depth : Nat) → BookAt opSem ib depth) : ∀ Γ op fs rs,
    overwriteRule ib bodyBooks Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds ib Γ env → FactsHold ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      FactsHold ib vs rs := by
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
        obtain ⟨xs, ys, hio, hxs, hys⟩ := splitInsOuts_facts hsplit ha
        obtain _ | ⟨ht, hnil⟩ := hys
        cases hnil
        rename_i t
        cases t with
        | scalar _ => cases ht.1
        | tensor t =>
          have hb := framed_total (bodyBooks iters.length) hbody call env hΓ
          obtain ⟨rs, hsem, hrf⟩ := overwriteSem_total hib (structuredRanges?_spec hranges) hout hxs ht.1 hb
          have hf := generic_framed (outs := [t]) (structuredRanges?_spec hranges) hxs.typed
            (List.Forall₂.cons ht.1 .nil) ctx (fun inner => evalRegion opSem inner call env body)
          simp only [List.map_cons, List.map_nil] at hf
          rw [hf] at hsem
          refine ⟨rs, ?_, hrf⟩
          have hlen : rs.length = op.blind.results.length := by
            rw [HasTys.length (clean_typed hrf), Op.blind_results, List.length_map, hres]
            rfl
          rw [linalg_sem_of hm hd (.inl hn)]
          simp only [hregs, evalRegions, Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hio,
            attrAffineMaps?_blind, hmaps, iterKinds?_blind, hiters, Outcome.bindOp_eq, hsem,
            Outcome.bind_ok, hlen, ↓reduceIte, Outcome.pure_eq]
      · cases h
    · cases h
  · cases h

def initializedRule (ib : Nat) (bodyBooks : (depth : Nat) → BookAt opSem ib depth)
    (Γ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts op fs, attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, outs), some maps, some iters, [body] =>
      if (∀ f ∈ outs, f.initialized = true) ∧ (∀ f ∈ outs, f.ty.isTensor = true) ∧
          op.results.length = outs.length ∧
          (structuredRanges? maps iters.length ((ins ++ outs).map (·.ty))).isSome ∧
          FrontendDefinedFrame.checkRegion (bodyBooks iters.length) Γ body ((ins ++ outs).map elementFact)
            (outs.map fun f => f.ty.eltScalar) then
        some (clean (outs.map (·.ty)))
      else none
    | _, _, _, _ => none
  else none

theorem initializedRule_sound (ib : Nat) (bodyBooks : (depth : Nat) → BookAt opSem ib depth) : ∀ Γ op fs rs,
    initializedRule ib bodyBooks Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds ib Γ env → FactsHold ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      FactsHold ib vs rs := by
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
        obtain ⟨xs, ys, hio, hxs, hys⟩ := splitInsOuts_facts hsplit ha
        have hclean := facts_eq_clean outs hi
        have hys' : FactsHold ib ys (clean (outs.map (·.ty))) := hclean ▸ hys
        have hb : BodyTotal ib (framed iters.length (fun inner => evalRegion opSem inner call env body))
            ((ins ++ clean (outs.map (·.ty))).map elementFact)
            ((outs.map (·.ty)).map VTy.eltScalar) := by
          have hb := framed_total (bodyBooks iters.length) hbody call env hΓ
          intro inner hwidth vals hfacts
          rw [← hclean] at hfacts
          simpa only [List.map_map, Function.comp_def] using hb inner hwidth vals hfacts
        have hs : StructuredOk maps iters.length (ins.map (·.ty) ++ outs.map (·.ty)) ranges := by
          simpa only [List.map_append] using structuredRanges?_spec hranges
        have hout : ∀ t ∈ outs.map (·.ty), t.isTensor = true := by
          intro t ht'
          obtain ⟨f, hf, rfl⟩ := List.mem_map.1 ht'
          exact ht f hf
        obtain ⟨tensors, htensors⟩ := tensors_of_typed (clean_typed hys') hout
        have hf := generic_framed hs hxs.typed (htensors ▸ clean_typed hys') ctx
          (fun inner => evalRegion opSem inner call env body)
        rw [← htensors] at hf
        obtain ⟨rs, hsem, hrf⟩ := genericSem_total hib hs hout hxs hys' hb
        rw [hf] at hsem
        refine ⟨rs, ?_, hrf⟩
        have hlen : rs.length = op.blind.results.length := by
          rw [HasTys.length (clean_typed hrf), List.length_map, Op.blind_results,
            List.length_map, hres]
        rw [linalg_sem_of hm hd (.inl hn)]
        simp only [hregs, evalRegions, Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hio,
          attrAffineMaps?_blind, hmaps, iterKinds?_blind, hiters, Outcome.bindOp_eq, hsem,
          Outcome.bind_ok, hlen, ↓reduceIte, Outcome.pure_eq]
      · cases h
    · cases h
  · cases h


def book (ib : Nat) (bodyBooks : (depth : Nat) → BookAt opSem ib depth) : Book opSem ib where
  rule Γ op fs := (overwriteRule ib bodyBooks Γ op fs).orElse fun _ => initializedRule ib bodyBooks Γ op fs
  sound Γ op fs rs h ctx call env args hib hΓ ha := by
    cases ho : overwriteRule ib bodyBooks Γ op fs with
    | some ys =>
      simp only [ho, Option.orElse_some, Option.some.injEq] at h
      subst h
      exact overwriteRule_sound ib bodyBooks Γ op fs ys ho ctx call env args hib hΓ ha
    | none =>
      simp only [ho, Option.orElse_none] at h
      exact initializedRule_sound ib bodyBooks Γ op fs rs h ctx call env args hib hΓ ha

#print axioms book

end MerlinVerify.Rules.FrontendDefinedGeneric
