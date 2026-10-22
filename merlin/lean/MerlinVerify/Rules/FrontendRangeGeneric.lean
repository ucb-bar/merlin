/- Tensor-element intervals through identity-output structured overwrites. The
body is checked at actual loop coordinates and receives bounds at actual reads. -/
import MerlinVerify.Rules.FrontendRangeScalars

namespace MerlinVerify.Rules.FrontendRangeGeneric
open MerlinVerify MerlinVerify.Check Sem.Ops Sem.Ops.Linalg
open FrontendTensorRanges FrontendRangeFrame

def elementFact (f : Fact) : Fact := ⟨Definedness.elementFact f.base, f.info⟩

def emptyOutput (sh : List Nat) (e : Ty) : Fact := ⟨⟨.tensor sh e, false⟩, none⟩

theorem readAt_total {ib : Nat} {ranges pt : List Nat} (hp : Idx.InBounds ranges pt)
    {v : RVal} {f : Fact} {m : AffineMap} (hv : Definedness.Refined.Fact.Holds model ib v f)
    (hm : mapInBounds ranges m f.base.ty.shape = true) :
    ∃ x, readAt v m pt = .ok x ∧ Definedness.Refined.Fact.Holds model ib (.scalar x) (elementFact f) := by
  obtain ⟨x, hx, hbase⟩ := Definedness.readAt_total hp hv.1 hm
  refine ⟨x, hx, hbase, ?_⟩
  cases hi : f.info with
  | none => simp [model, elementFact, hi]
  | some r =>
    cases v with
    | scalar v =>
      have hxv : x = v := by
        simp only [readAt] at hx
        split at hx
        · cases hx; rfl
        · cases hx
      subst x
      simpa [model, Holds, elementFact, hi] using hv.2
    | tensor t =>
      cases ht : f.base.ty with
      | scalar _ => have h := hv.1.1; rw [ht] at h; cases h
      | tensor sh e =>
        have hs : t.shape = sh := by have h := hv.1.1; rw [ht] at h; exact h.1
        have hm' : mapInBounds ranges m sh = true := by simpa [ht, VTy.shape] using hm
        obtain ⟨i, he, hin⟩ := mapInBounds_sound hm' hp
        change evalMap m pt >>= toIdx sh = some i at he
        have heq : x = t.get i := by simpa only [readAt, hs, he, Outcome.ok.injEq] using hx.symm
        have hr : Holds r (.tensor t) := by simpa [model, hi] using hv.2
        simpa only [elementFact, hi, model, Holds, heq] using hr i (hs.symm ▸ hin)

theorem mapM_read_total {ib : Nat} {ranges pt : List Nat} (hp : Idx.InBounds ranges pt) :
    ∀ (vs : List RVal) (ms : List AffineMap) (fs : List Fact),
      Definedness.Refined.FactsHold model ib vs fs → MapsInBounds ranges ms (fs.map (·.base.ty)) →
      ∃ xs, (vs.zip ms).mapM (fun (v, m) => readAt v m pt) = .ok xs ∧
        Definedness.Refined.FactsHold model ib (xs.map RVal.scalar) (fs.map elementFact)
  | [], [], [], _, _ => ⟨[], rfl, .nil⟩
  | v :: vs, m :: ms, f :: fs, .cons hv hvs, .cons hm hms => by
    obtain ⟨x, hx, hxf⟩ := readAt_total hp hv hm
    obtain ⟨xs, hxs, hxsf⟩ := mapM_read_total hp vs ms fs hvs hms
    exact ⟨x :: xs, by simp only [List.zip_cons_cons, List.mapM_cons, Outcome.bindOp_eq,
      hx, hxs, Outcome.bind_ok, Outcome.pure_eq], .cons hxf hxsf⟩
  | [], _ :: _, _, h, hm => by cases h; cases hm
  | _ :: _, [], fs, h, hm => by cases fs; cases h; cases hm

theorem split_facts {ib : Nat} {op : Op} {args : List RVal} {fs ins outs : List Fact}
    (h : splitInsOuts op fs = some (ins, outs)) (ha : Definedness.Refined.FactsHold model ib args fs) :
    ∃ xs ys, splitInsOuts op args = some (xs, ys) ∧
      Definedness.Refined.FactsHold model ib xs ins ∧ Definedness.Refined.FactsHold model ib ys outs := by
  have hlen := List.Forall₂.length_eq ha
  unfold splitInsOuts at h ⊢
  split at h
  · next ni no _ =>
    split at h
    · next hc =>
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      refine ⟨args.take ni.toNat, args.drop ni.toNat, ?_, List.forall₂_take _ ha, List.forall₂_drop _ ha⟩
      rw [ite_eq_left ⟨hc.1, hc.2.1, by rw [hlen]; exact hc.2.2⟩]
    · cases h
  · cases h

/-- The semantic obligation includes successful execution and the inferred output interval.
Only actual in-bounds body calls are quantified, including zero-extent iteration spaces. -/
theorem overwrite_total {ib : Nat} {ctx : Ctx} {ins : List RVal} {fs : List Fact}
    {maps : List AffineMap} {iters : List IterKind} {t : Tensor} {sh : List Nat} {e : Ty}
    {body : Ctx → RegionSem} {info : Option Definedness.IndexRanges.Interval}
    (hs : StructuredOk maps iters.length (fs.map (·.base.ty) ++ [.tensor sh e]) sh)
    (hout : maps.drop fs.length = [identityMap sh.length])
    (hins : Definedness.Refined.FactsHold model ib ins fs)
    (ht : RVal.HasTy ib (.tensor t) (.tensor sh e))
    (hbody : ∀ pt, Idx.InBounds sh pt → ∀ args,
      Definedness.Refined.FactsHold model ib args
        ((fs ++ [emptyOutput sh e]).map elementFact) →
      ∃ ys, body {ctx with loopIndex := pt} args = .ok ys ∧
        Definedness.Refined.FactsHold model ib ys [⟨⟨.scalar e, true⟩, info⟩]) :
    ∃ rs, genericSem ctx maps iters ins [.tensor t] body = .ok rs ∧
      Definedness.Refined.FactsHold model ib rs [⟨⟨.tensor sh e, true⟩, info⟩] := by
  classical
  have hall : HasTys ib (ins ++ [.tensor t]) (fs.map (·.base.ty) ++ [.tensor sh e]) :=
    List.rel_append hins.typed (.cons ht .nil)
  obtain ⟨hL, hD, hR, hLR⟩ := structured_runtime hs hall
  have hk : ins.length = fs.length := List.Forall₂.length_eq hins
  have hinM : MapsInBounds sh (maps.take ins.length) (fs.map (·.base.ty)) := by
    have h := List.forall₂_take (fs.map (·.base.ty)).length hs.inBounds
    rw [List.take_left] at h
    simpa only [List.length_map, ← hk] using h
  have hpoint : ∀ p, Idx.InBounds sh p → ∃ y,
      bodyAt ctx ins (maps.take ins.length) [identityMap sh.length] [t] body p = .ok [y] ∧
      Definedness.Refined.Fact.Holds model ib (.scalar y) ⟨⟨.scalar e, true⟩, info⟩ := by
    intro p hp
    obtain ⟨xs, hx, hxf⟩ := mapM_read_total hp ins (maps.take ins.length) fs hins hinM
    have hp' : Idx.InBounds t.shape p := ht.1.symm ▸ hp
    have houtRead := readAt_identity hp'
    rw [ht.1] at houtRead
    have hfacts : Definedness.Refined.FactsHold model ib ((xs ++ [t.get p]).map RVal.scalar)
        ((fs ++ [emptyOutput sh e]).map elementFact) := by
      rw [List.map_append, List.map_append]
      exact List.rel_append hxf (.cons ⟨⟨ht.2.2 p hp, by simp [elementFact, emptyOutput, Definedness.elementFact]⟩, trivial⟩ .nil)
    obtain ⟨ys, hy, hyf⟩ := hbody p hp _ hfacts
    obtain _ | ⟨hyf, hnil⟩ := hyf
    cases hnil
    rename_i y
    cases y with
    | tensor _ => cases hyf.1.1
    | scalar y =>
      refine ⟨y, ?_, hyf⟩
      simp only [bodyAt, List.zip_cons_cons, List.zip_nil_right, List.mapM_cons, List.mapM_nil,
        Outcome.bindOp_eq, hx, houtRead, Outcome.bind_ok, Outcome.pure_eq, hy,
        List.length_cons, List.length_nil, ne_eq, not_true_eq_false, ↓reduceIte]
  let F : Idx → Val := fun p => if hp : Idx.InBounds sh p then Classical.choose (hpoint p hp) else .poison
  have hF : ∀ p, Idx.InBounds sh p →
      bodyAt ctx ins (maps.take ins.length) [identityMap sh.length] [t] body p = .ok [F p] ∧
      Definedness.Refined.Fact.Holds model ib (.scalar (F p)) ⟨⟨.scalar e, true⟩, info⟩ := by
    intro p hp
    simpa only [F, dite_eq_left hp] using Classical.choose_spec (hpoint p hp)
  refine ⟨[.tensor (Tensor.canon sh t.elt F)], ?_, .cons ⟨⟨⟨rfl, ht.2.1, ?_⟩, ?_⟩, ?_⟩ .nil⟩
  · exact genericSem_parallel1 ctx maps iters ins t body sh F
      (by simpa only [List.length_append, List.length_cons, List.length_nil] using hL)
      hD hR hLR ht.1 (hk ▸ hout) (fun p hp => (hF p hp).1)
  · intro p hp
    simpa only [Tensor.canon, hp, ↓reduceIte, RVal.HasTy] using (hF p hp).2.1.1
  · intro _ p hp
    change Idx.InBounds sh p at hp
    simpa only [Tensor.canon, hp, ↓reduceIte, RVal.Defined] using (hF p hp).2.1.2 rfl
  · cases info with
    | none => trivial
    | some r =>
      intro p hp
      change Idx.InBounds sh p at hp
      simpa only [Tensor.canon, hp, ↓reduceIte, model, Holds] using (hF p hp).2.2

def rule (ib : Nat) (bodyBooks : (ranges : List Nat) → BookAt ib ranges)
    (Γ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts op fs, attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, [⟨⟨.tensor sh e, _⟩, _⟩]), some maps, some iters, [body] =>
      if op.results.length = 1 ∧ maps.drop ins.length = [identityMap sh.length] ∧
          structuredRanges? maps iters.length (ins.map (·.base.ty) ++ [.tensor sh e]) = some sh ∧
          op.results.mapM (fun v => vtyOf ib v.ty) = some [.tensor sh e] then
        match inferRegion (bodyBooks sh) Γ body ((ins ++ [emptyOutput sh e]).map elementFact) with
        | some [⟨⟨.scalar ty, true⟩, info⟩] => if ty = e then some [⟨⟨.tensor sh e, true⟩, info⟩] else none
        | _ => none
      else none
    | _, _, _, _ => none
  else none

def book (ib : Nat) (bodyBooks : (ranges : List Nat) → BookAt ib ranges) : Definedness.Refined.Book model opSem ib where
  rule := rule ib bodyBooks
  sound Γ op fs rs h ctx call env args hib hΓ ha := by
    unfold rule at h
    split at h
    · next hc =>
      obtain ⟨hm, hd, hn⟩ := hc
      split at h
      · next ins sh e init oldinfo maps iters body hsplit hmaps hiters hregs =>
        split at h
        · next hc =>
          obtain ⟨hres, hout, hranges, _⟩ := hc
          split at h
          · next ty info hbody =>
            split at h
            · next hty =>
              subst ty
              cases h
              obtain ⟨xs, ys, hio, hxs, hys⟩ := split_facts hsplit ha
              obtain _ | ⟨ht, hnil⟩ := hys
              cases hnil
              rename_i t
              cases t with
              | scalar _ => cases ht.1.1
              | tensor t =>
                obtain ⟨rs, hsem, hrf⟩ := overwrite_total (ctx := ctx)
                  (body := fun inner => evalRegion opSem inner call env body) (info := info)
                  (structuredRanges?_spec hranges) hout hxs ht.1.1
                  (fun p hp vals hfacts => inferRegion_sound (ctx := {ctx with loopIndex := p})
                    (bodyBooks sh) hbody hib (.inr hp) hΓ hfacts)
                refine ⟨rs, ?_, hrf⟩
                have hlen : rs.length = op.blind.results.length := by
                  rw [List.Forall₂.length_eq hrf, Op.blind_results, List.length_map, hres]; rfl
                rw [linalg_sem_of hm hd (.inl hn)]
                simp only [hregs, evalRegions, Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hio,
                  attrAffineMaps?_blind, hmaps, iterKinds?_blind, hiters, Outcome.bindOp_eq, hsem,
                  Outcome.bind_ok, hlen, ↓reduceIte, Outcome.pure_eq]
            · cases h
          · cases h
        · cases h
      · cases h
    · cases h

/-- No body call is possible in a zero-extent iteration space. The checked
structured geometry proves this case, so it needs no invented element interval. -/
def emptyRule (ib : Nat) (_ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts op fs, attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, [⟨⟨.tensor sh e, _⟩, _⟩]), some maps, some iters, [_body] =>
      if op.results.length = 1 ∧ maps.drop ins.length = [identityMap sh.length] ∧
          structuredRanges? maps iters.length (ins.map (·.base.ty) ++ [.tensor sh e]) = some sh ∧
          shapeSize sh = 0 ∧ op.results.mapM (fun v => vtyOf ib v.ty) = some [.tensor sh e] then
        some [⟨⟨.tensor sh e, true⟩, none⟩]
      else none
    | _, _, _, _ => none
  else none

def emptyBook (ib : Nat) : Definedness.Refined.Book model opSem ib where
  rule := emptyRule ib
  sound Γ op fs rs h ctx call env args hib hΓ ha := by
    unfold emptyRule at h
    split at h
    · next hc =>
      obtain ⟨hm, hd, hn⟩ := hc
      split at h
      · next ins sh e init oldinfo maps iters body hsplit hmaps hiters hregs =>
        split at h
        · next hc =>
          obtain ⟨hres, hout, hranges, hempty, _⟩ := hc
          cases h
          obtain ⟨xs, ys, hio, hxs, hys⟩ := split_facts hsplit ha
          obtain _ | ⟨ht, hnil⟩ := hys
          cases hnil
          rename_i t
          cases t with
          | scalar _ => cases ht.1.1
          | tensor t =>
            obtain ⟨rs, hsem, hrf⟩ := overwrite_total (ctx := ctx)
              (body := fun inner => evalRegion opSem inner call env body) (info := none)
              (structuredRanges?_spec hranges) hout hxs ht.1.1
              (fun p hp _ _ => False.elim (by have hi := linearIndex_lt sh p hp; omega))
            refine ⟨rs, ?_, hrf⟩
            have hlen : rs.length = op.blind.results.length := by
              rw [List.Forall₂.length_eq hrf, Op.blind_results, List.length_map, hres]; rfl
            rw [linalg_sem_of hm hd (.inl hn)]
            simp only [hregs, evalRegions, Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hio,
              attrAffineMaps?_blind, hmaps, iterKinds?_blind, hiters, Outcome.bindOp_eq, hsem,
              Outcome.bind_ok, hlen, ↓reduceIte, Outcome.pure_eq]
        · cases h
      · cases h
    · cases h

#print axioms book
end MerlinVerify.Rules.FrontendRangeGeneric
