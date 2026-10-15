/- Totality and initialization proofs for structured tensor programs. -/
import MerlinVerify.Check.DefinedRules

namespace MerlinVerify.Check.Definedness

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

/-- Total region execution, including initialization of every yielded scalar or tensor. -/
def BodyTotal (ib : Nat) (body : Ctx → RegionSem) (args : List Fact) (yields : List VTy) : Prop :=
  ∀ ctx, ctx.indexBits = ib → ∀ vs, FactsHold ib vs args →
    ∃ ys, body ctx vs = .ok ys ∧ HasTys ib ys yields ∧ ResultsDefined ys

def clean (tys : List VTy) : List Fact := tys.map fun t => ⟨t, true⟩

def elementFact (f : Fact) : Fact := ⟨f.ty.eltScalar, f.initialized⟩

theorem clean_typed {ib : Nat} {vs : List RVal} {tys : List VTy}
    (h : FactsHold ib vs (clean tys)) : HasTys ib vs tys := by
  simpa only [clean, List.map_map, Function.comp_def, List.map_id_fun', id_eq] using h.typed

theorem clean_defined {ib : Nat} {vs : List RVal} {tys : List VTy}
    (h : FactsHold ib vs (clean tys)) : ResultsDefined vs :=
  h.defined (by simp [clean])

theorem readAt_total {ib : Nat} {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt) {v : RVal}
    {f : Fact} {m : AffineMap} (hv : Fact.Holds ib v f)
    (hm : mapInBounds ranges m f.ty.shape = true) :
    ∃ x, readAt v m pt = .ok x ∧ Fact.Holds ib (.scalar x) (elementFact f) := by
  rcases f with ⟨ty, init⟩
  cases v with
  | scalar x =>
    cases ty with
    | tensor _ _ => cases hv.1
    | scalar t =>
      simp only [mapInBounds, VTy.shape, List.length_nil, beq_iff_eq, List.zip_nil_right,
        List.all_nil, Bool.and_true, List.length_eq_zero_iff] at hm
      exact ⟨x, by simp [readAt, hm], hv⟩
  | tensor x =>
    cases ty with
    | scalar _ => cases hv.1
    | tensor sh e =>
      obtain ⟨⟨hs, he, hg⟩, hd⟩ := hv
      obtain ⟨i, hi, hin⟩ := mapInBounds_sound hm hpt
      change evalMap m pt >>= toIdx sh = some i at hi
      refine ⟨x.get i, ?_, ⟨hg i hin, fun hinit => hd hinit i (hs ▸ hin)⟩⟩
      simp only [readAt, hs, hi]

theorem mapM_read_total {ib : Nat} {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt) {α : Type}
    (toR : α → RVal) (g : α × AffineMap → Outcome Val) (hg : ∀ a m, g (a, m) = readAt (toR a) m pt) :
    ∀ (as : List α) (ms : List AffineMap) (fs : List Fact), FactsHold ib (as.map toR) fs →
      MapsInBounds ranges ms (fs.map (·.ty)) →
      ∃ xs, (as.zip ms).mapM g = .ok xs ∧ FactsHold ib (xs.map RVal.scalar) (fs.map elementFact)
  | [], [], [], _, _ => ⟨[], rfl, .nil⟩
  | a :: as, m :: ms, f :: fs, .cons ha hrest, .cons hm hms => by
    obtain ⟨x, hx, hxf⟩ := readAt_total hpt ha hm
    obtain ⟨xs, hxs, hxsf⟩ := mapM_read_total hpt toR g hg as ms fs hrest hms
    exact ⟨x :: xs, by simp only [List.zip_cons_cons, List.mapM_cons, Outcome.bindOp_eq,
      hg, hx, hxs, Outcome.bind_ok, Outcome.pure_eq], .cons hxf hxsf⟩
  | [], _ :: _, _, h, hm => by cases h; cases hm
  | _ :: _, [], fs, h, hm => by cases fs; cases h; cases hm

theorem writeAt_total {ib : Nat} {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt) {x : Tensor}
    {t : VTy} {m : AffineMap} {v : Val} (hx : Fact.Holds ib (.tensor x) ⟨t, true⟩)
    (hm : mapInBounds ranges m t.shape = true) (hv : Val.HasTy ib (eltOfVTy t) v)
    (hd : v ≠ .poison) :
    ∃ y, writeAt x m pt v = .ok y ∧ Fact.Holds ib (.tensor y) ⟨t, true⟩ := by
  cases t with
  | scalar _ => cases hx.1
  | tensor sh e =>
    obtain ⟨⟨hs, he, hg⟩, hxd⟩ := hx
    obtain ⟨i, hi, _⟩ := mapInBounds_sound hm hpt
    change evalMap m pt >>= toIdx sh = some i at hi
    refine ⟨x.set i v, by simp only [writeAt, hs, hi], ⟨⟨hs, he, ?_⟩, ?_⟩⟩
    · intro j hj
      simp only [Tensor.set]
      split
      · exact hv
      · exact hg j hj
    · intro _ j hj
      simp only [Tensor.set]
      split
      · exact hd
      · exact hxd rfl j hj

theorem mapM_write_total {ib : Nat} {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt)
    (g : Val × Tensor × AffineMap → Outcome Tensor) (hg : ∀ v x m, g (v, x, m) = writeAt x m pt v) :
    ∀ (vs : List Val) (cur : List Tensor) (ms : List AffineMap) (tys : List VTy),
      FactsHold ib (vs.map RVal.scalar) (clean (tys.map VTy.eltScalar)) →
      FactsHold ib (cur.map RVal.tensor) (clean tys) → MapsInBounds ranges ms tys →
      ∃ ys, (vs.zip (cur.zip ms)).mapM g = .ok ys ∧
        FactsHold ib (ys.map RVal.tensor) (clean tys)
  | [], [], [], [], _, _, _ => ⟨[], rfl, .nil⟩
  | v :: vs, x :: xs, m :: ms, t :: ts, .cons hv hvs, .cons hx hxs, .cons hm hms => by
    obtain ⟨y, hy, hyf⟩ := writeAt_total hpt hx hm hv.1 (hv.2 rfl)
    obtain ⟨ys, hys, hysf⟩ := mapM_write_total hpt g hg vs xs ms ts hvs hxs hms
    exact ⟨y :: ys, by simp only [List.zip_cons_cons, List.mapM_cons, Outcome.bindOp_eq,
      hg, hy, hys, Outcome.bind_ok, Outcome.pure_eq], .cons hyf hysf⟩
  | [], _, _, _ :: _, hvs, _, _ => by cases hvs
  | _ :: _, _, _, [], hvs, _, _ => by cases hvs
  | _, [], _, _ :: _, _, hxs, _ => by cases hxs
  | _, _ :: _, _, [], _, hxs, _ => by cases hxs
  | _, _, [], _ :: _, _, _, hms => by cases hms
  | _, _, _ :: _, [], _, _, hms => by cases hms

theorem mapM_scalars_total (g : RVal → Outcome Val) (h : ∀ v, g (.scalar v) = .ok v) :
    ∀ vs : List Val, (vs.map RVal.scalar).mapM g = .ok vs
  | [] => rfl
  | v :: vs => by
    simp only [List.map_cons, List.mapM_cons, Outcome.bindOp_eq, h, mapM_scalars_total g h vs,
      Outcome.bind_ok, Outcome.pure_eq]

theorem bodyAt_total {ib : Nat} {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt) {ctx : Ctx}
    (hib : ctx.indexBits = ib) {ins : List RVal} {inMaps outMaps : List AffineMap} {cur : List Tensor}
    {body : Ctx → RegionSem} {inFacts : List Fact} {outTys : List VTy}
    (hins : FactsHold ib ins inFacts) (hcur : FactsHold ib (cur.map RVal.tensor) (clean outTys))
    (hinM : MapsInBounds ranges inMaps (inFacts.map (·.ty)))
    (houtM : MapsInBounds ranges outMaps outTys)
    (hbody : BodyTotal ib body ((inFacts ++ clean outTys).map elementFact)
      (outTys.map VTy.eltScalar)) :
    ∃ vs, bodyAt ctx ins inMaps outMaps cur body pt = .ok vs ∧
      FactsHold ib (vs.map RVal.scalar) (clean (outTys.map VTy.eltScalar)) := by
  obtain ⟨xs, hxs, hxf⟩ := mapM_read_total hpt id (fun (v, m) => readAt v m pt) (by intro _ _; rfl) ins inMaps inFacts
    (by simpa using hins) hinM
  obtain ⟨ys, hys, hyf⟩ := mapM_read_total hpt RVal.tensor (fun (v, m) => readAt (.tensor v) m pt) (by intro _ _; rfl) cur outMaps
    (clean outTys) hcur (by simpa only [clean, List.map_map, Function.comp_def, List.map_id_fun', id_eq] using houtM)
  have hargs : FactsHold ib ((xs ++ ys).map RVal.scalar) ((inFacts ++ clean outTys).map elementFact) := by
    rw [List.map_append, List.map_append]
    exact List.rel_append hxf hyf
  obtain ⟨zs, hzs, hzt, hzd⟩ := hbody {ctx with loopIndex := pt} hib _ hargs
  have hlen : zs.length = cur.length := by
    rw [HasTys.length hzt, List.length_map, ← HasTys.length (clean_typed hcur), List.length_map]
  obtain ⟨vs, rfl, _⟩ := scalars_of_typed hzt
  refine ⟨vs, ?_, inputFacts hzt hzd⟩
  simp only [bodyAt, Outcome.bindOp_eq, hxs, hys, hzs, Outcome.bind_ok,
    hlen, ne_eq, not_true_eq_false, ↓reduceIte]
  exact mapM_scalars_total _ (by intro _; rfl) vs

theorem foldlM_total {α β : Type} (f : β → α → Outcome β) (I : β → Prop) (P : α → Prop)
    (hf : ∀ b a, I b → P a → ∃ b', f b a = .ok b' ∧ I b') :
    ∀ (l : List α) (b : β), (∀ a ∈ l, P a) → I b → ∃ b', l.foldlM f b = .ok b' ∧ I b'
  | [], b, _, hb => ⟨b, rfl, hb⟩
  | a :: l, b, hl, hb => by
    obtain ⟨b', hb', hi⟩ := hf b a hb (hl a (List.mem_cons_self ..))
    obtain ⟨c, hc, hci⟩ := foldlM_total f I P hf l b'
      (fun x hx => hl x (List.mem_cons_of_mem _ hx)) hi
    exact ⟨c, by rw [List.foldlM_cons, hb', Outcome.bindOp_eq, Outcome.bind_ok, hc], hci⟩

theorem mapM_tensors_total (g : RVal → Outcome Tensor) (h : ∀ t, g (.tensor t) = .ok t) :
    ∀ ts : List Tensor, (ts.map RVal.tensor).mapM g = .ok ts
  | [] => rfl
  | t :: ts => by
    simp only [List.map_cons, List.mapM_cons, Outcome.bindOp_eq, h, mapM_tensors_total g h ts,
      Outcome.bind_ok, Outcome.pure_eq]

theorem clean_canonize {ib : Nat} : ∀ {ts : List Tensor} {tys : List VTy},
    FactsHold ib (ts.map RVal.tensor) (clean tys) →
      FactsHold ib (ts.map fun t => RVal.tensor t.canonize) (clean tys)
  | [], [], _ => .nil
  | t :: ts, _ :: _, .cons ht hrest => by
    refine .cons ⟨hasTy_canonize ht.1, ?_⟩ (clean_canonize hrest)
    intro _ i hi
    change Idx.InBounds t.shape i at hi
    simpa only [Tensor.canonize, Tensor.canon, hi, ↓reduceIte] using ht.2 rfl i hi

/-- Static bounds checks imply that the semantic evaluator accepts the runtime shapes. -/
theorem structured_runtime {ib : Nat} {maps : List AffineMap} {n : Nat} {tys : List VTy}
    {ranges : List Nat} {args : List RVal} (hs : StructuredOk maps n tys ranges)
    (ht : HasTys ib args tys) :
    maps.length = args.length ∧
    maps.any (fun m => m.numDims ≠ n || m.numSyms ≠ 0) = false ∧
    (maps.zip (args.map shapeOf)).any (fun p => p.1.results.length ≠ p.2.length) = false ∧
    loopRanges maps (args.map shapeOf) n = .ok ranges := by
  have hshapes := shapes_typed ht
  refine ⟨hs.len.trans (HasTys.length ht).symm, ?_, ?_, ?_⟩
  · rw [List.any_eq_false]
    intro m hm
    simp [(hs.dims m hm).1, (hs.dims m hm).2]
  · rw [hshapes, List.zip_map_right, List.any_map, List.any_eq_false]
    intro p hp
    have hin := List.forall₂_iff_zip.1 hs.inBounds |>.2 hp
    simp only [mapInBounds, Bool.and_eq_true, beq_iff_eq] at hin
    simp [Function.comp, hin.1]
  · rw [hshapes]
    exact mapM_ofOption _ _ _ _ hs.loops

/-- Initialized destinations stay initialized across every reduction step, including empty loops.
Inputs may be uninitialized only when the checked body can still prove clean outputs. -/
theorem genericSem_total {ib : Nat} {ctx : Ctx} (hib : ctx.indexBits = ib) {maps : List AffineMap}
    {iters : List IterKind} {ins outs : List RVal} {body : Ctx → RegionSem} {inFacts : List Fact}
    {outTys : List VTy} {ranges : List Nat}
    (hs : StructuredOk maps iters.length (inFacts.map (·.ty) ++ outTys) ranges)
    (hout : ∀ t ∈ outTys, t.isTensor = true) (hins : FactsHold ib ins inFacts)
    (houts : FactsHold ib outs (clean outTys))
    (hbody : BodyTotal ib body ((inFacts ++ clean outTys).map elementFact)
      (outTys.map VTy.eltScalar)) :
    ∃ rs, genericSem ctx maps iters ins outs body = .ok rs ∧ FactsHold ib rs (clean outTys) := by
  have hint := hins.typed
  have houtt := clean_typed houts
  obtain ⟨ts, rfl⟩ := tensors_of_typed houtt hout
  have hall := List.rel_append hint houtt
  obtain ⟨hL, hD, hR, hLR⟩ := structured_runtime hs hall
  have hk : ins.length = (inFacts.map (·.ty)).length := HasTys.length hint
  have hinM : MapsInBounds ranges (maps.take ins.length) (inFacts.map (·.ty)) := by
    have := List.forall₂_take (inFacts.map (·.ty)).length hs.inBounds
    rwa [List.take_left, ← hk] at this
  have houtM : MapsInBounds ranges (maps.drop ins.length) outTys := by
    have := List.forall₂_drop (inFacts.map (·.ty)).length hs.inBounds
    rwa [List.drop_left, ← hk] at this
  have hstep : ∀ cur pt, FactsHold ib (cur.map RVal.tensor) (clean outTys) →
      Idx.InBounds ranges pt → ∃ ys,
      loopStep ctx ins (maps.take ins.length) (maps.drop ins.length) body cur pt = .ok ys ∧
      FactsHold ib (ys.map RVal.tensor) (clean outTys) := by
    intro cur pt hcur hpt
    obtain ⟨vs, hv, hvf⟩ := bodyAt_total hpt hib hins hcur hinM houtM hbody
    obtain ⟨ys, hy, hyf⟩ := mapM_write_total hpt (fun (v, x, m) => writeAt x m pt v)
      (by intro _ _ _; rfl) vs cur (maps.drop ins.length) outTys hvf hcur houtM
    exact ⟨ys, by simp only [loopStep, Outcome.bindOp_eq, hv, Outcome.bind_ok, hy], hyf⟩
  obtain ⟨final, hf, hfinal⟩ := foldlM_total _ _ (Idx.InBounds ranges) hstep
    (allPoints ranges) ts (fun pt hp => (mem_allPoints _ _).1 hp) houts
  refine ⟨final.map fun t => RVal.tensor t.canonize, ?_, clean_canonize hfinal⟩
  simp only [genericSem, hL, ne_eq, not_true_eq_false, ↓reduceIte, hD, Bool.false_eq_true,
    hR, Outcome.bindOp_eq]
  rw [mapM_tensors_total _ (by intro _; rfl) ts, Outcome.bind_ok, hLR, Outcome.bind_ok,
    hf, Outcome.bind_ok, Outcome.pure_eq]

/-- A full identity-mapped overwrite initializes an empty destination. The body is checked with
an explicitly uninitialized output argument, so reading that argument cannot certify the result. -/
theorem fillSem_total {ib : Nat} {ctx : Ctx} (hib : ctx.indexBits = ib) {v : RVal} {vf : Fact}
    {t : Tensor} {sh : List Nat} {e : Ty} {body : Ctx → RegionSem}
    (hs : StructuredOk (fillMaps sh.length) sh.length [vf.ty, .tensor sh e] sh)
    (hv : Fact.Holds ib v vf) (ht : RVal.HasTy ib (.tensor t) (.tensor sh e))
    (hbody : BodyTotal ib body [elementFact vf, ⟨.scalar e, false⟩] [.scalar e]) :
    ∃ rs, genericSem ctx (fillMaps sh.length) (List.replicate sh.length .parallel)
      [v] [.tensor t] body = .ok rs ∧ FactsHold ib rs (clean [.tensor sh e]) := by
  classical
  have hshape := ht.1
  have hall : HasTys ib [v, .tensor t] [vf.ty, .tensor sh e] := .cons hv.1 (.cons ht .nil)
  obtain ⟨hL, hD, hR, hLR⟩ := structured_runtime hs hall
  have hinM := hs.inBounds
  cases hinM with
  | cons hinM hrest =>
    have hpoint : ∀ p, Idx.InBounds sh p → ∃ y,
        bodyAt ctx [v] ((fillMaps sh.length).take 1) [identityMap sh.length] [t] body p = .ok [y] ∧
        Val.HasTy ib e y ∧ y ≠ .poison := by
      intro p hp
      obtain ⟨x, hx, hxf⟩ := readAt_total hp hv hinM
      have hp' : Idx.InBounds t.shape p := hshape ▸ hp
      have hout := readAt_identity hp'
      rw [hshape] at hout
      have hfacts : FactsHold ib [.scalar x, .scalar (t.get p)]
          [elementFact vf, ⟨.scalar e, false⟩] :=
        .cons hxf (.cons ⟨ht.2.2 p hp, by simp⟩ .nil)
      obtain ⟨ys, hy, hyt, hyd⟩ := hbody {ctx with loopIndex := p} hib _ hfacts
      obtain _ | ⟨hyf, hnil⟩ := hyt
      cases hnil
      rename_i y
      cases y with
      | tensor _ => cases hyf
      | scalar y =>
        refine ⟨y, ?_, hyf, hyd _ (List.mem_cons_self ..)⟩
        simp only [bodyAt, fillMaps, List.take_succ_cons, List.take_zero, List.zip_cons_cons,
          List.zip_nil_left, List.mapM_cons, List.mapM_nil, Outcome.bindOp_eq, hx, hout,
          Outcome.bind_ok, Outcome.pure_eq, List.singleton_append, List.map_cons, List.map_nil,
          hy, List.length_cons, List.length_nil, ne_eq, not_true_eq_false, ↓reduceIte]
    let F : Idx → Val := fun p => if hp : Idx.InBounds sh p then Classical.choose (hpoint p hp) else .poison
    have hF : ∀ p, Idx.InBounds sh p →
        bodyAt ctx [v] ((fillMaps sh.length).take 1) [identityMap sh.length] [t] body p = .ok [F p] ∧
        Val.HasTy ib e (F p) ∧ F p ≠ .poison := by
      intro p hp
      simpa only [F, dite_eq_left hp] using Classical.choose_spec (hpoint p hp)
    refine ⟨[.tensor (Tensor.canon sh t.elt F)], ?_, .cons ⟨⟨rfl, ht.2.1, ?_⟩, ?_⟩ .nil⟩
    · apply genericSem_parallel1 ctx (fillMaps sh.length) (List.replicate sh.length .parallel)
        [v] t body sh F hL
      · simpa only [List.length_replicate] using hD
      · exact hR
      · simpa only [List.length_replicate, List.singleton_append] using hLR
      · exact hshape
      · rfl
      · exact fun p hp => (hF p hp).1
    · intro p hp
      simpa only [Tensor.canon, hp, ↓reduceIte] using (hF p hp).2.1
    · intro _ p hp
      change Idx.InBounds sh p at hp
      simpa only [Tensor.canon, hp, ↓reduceIte] using (hF p hp).2.2

theorem splitInsOuts_facts {ib : Nat} {op : Op} {args : List RVal}
    {fs ins outs : List Fact} (h : splitInsOuts op fs = some (ins, outs))
    (ha : FactsHold ib args fs) : ∃ xs ys, splitInsOuts op args = some (xs, ys) ∧
      FactsHold ib xs ins ∧ FactsHold ib ys outs := by
  have hlen := List.Forall₂.length_eq ha
  unfold splitInsOuts at h ⊢
  split at h
  · next ni no _ =>
    split at h
    · next hc =>
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      refine ⟨args.take ni.toNat, args.drop ni.toNat, ?_, List.forall₂_take _ ha,
        List.forall₂_drop _ ha⟩
      rw [ite_eq_left ⟨hc.1, hc.2.1, by rw [hlen]; exact hc.2.2⟩]
    · cases h
  · cases h

def fillDefinedRule (ib : Nat) (Γ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.fill" then
    match splitInsOuts op fs, op.regions with
    | some ([vf], [⟨.tensor sh e, _⟩]), [body] =>
      if op.results.length = 1 ∧
          structuredRanges? (fillMaps sh.length) sh.length [vf.ty, .tensor sh e] = some sh ∧
          checkRegion (leafBook ib) Γ body [elementFact vf, ⟨.scalar e, false⟩] [.scalar e] then
        some (clean [.tensor sh e])
      else none
    | _, _ => none
  else none

theorem fillDefinedRule_sound (ib : Nat) : ∀ Γ op fs rs,
    fillDefinedRule ib Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds ib Γ env → FactsHold ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      FactsHold ib vs rs := by
  intro Γ op fs rs h ctx call env args hib hΓ ha
  unfold fillDefinedRule at h
  split at h
  · next hc =>
    obtain ⟨hm, hd, hn⟩ := hc
    split at h
    · next vf sh e init body hsplit hregs =>
      split at h
      · next hc =>
        obtain ⟨hres, hranges, hbody⟩ := hc
        cases h
        obtain ⟨xs, ys, hio, hxs, hys⟩ := splitInsOuts_facts hsplit ha
        obtain _ | ⟨hv, hnil⟩ := hxs
        cases hnil
        obtain _ | ⟨ht, hnil⟩ := hys
        cases hnil
        rename_i v t
        cases t with
        | scalar _ => cases ht.1
        | tensor t =>
          have hb : BodyTotal ib (fun ctx => evalRegion opSem ctx call env body)
              [elementFact vf, ⟨.scalar e, false⟩] [.scalar e] := by
            intro ctx hib args ha
            exact checkRegion_sound (leafBook ib) hbody hib hΓ ha
          obtain ⟨rs, hsem, hrf⟩ := fillSem_total hib (structuredRanges?_spec hranges) hv ht.1 hb
          dsimp only [fillMaps] at hsem
          refine ⟨rs, ?_, hrf⟩
          have hlen : rs.length = op.blind.results.length := by
            rw [HasTys.length (clean_typed hrf), Op.blind_results, List.length_map, hres]
            rfl
          rw [linalg_sem_of hm hd (.inr (.inl hn))]
          simp only [hregs, evalRegions, Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hio,
            ht.1.1, Outcome.bindOp_eq, hsem, Outcome.bind_ok, hlen, ↓reduceIte, Outcome.pure_eq]
      · cases h
    · cases h
  · cases h

theorem facts_eq_clean (fs : List Fact) (h : ∀ f ∈ fs, f.initialized = true) :
    fs = clean (fs.map (·.ty)) := by
  induction fs with
  | nil => rfl
  | cons f fs ih =>
    have hf := h f (List.mem_cons_self ..)
    have hfs := ih (fun f hf => h f (List.mem_cons_of_mem _ hf))
    rcases f with ⟨ty, init⟩
    change init = true at hf
    subst init
    change _ = ⟨ty, true⟩ :: clean (fs.map (·.ty))
    rw [← hfs]

def genericDefinedRule (ib : Nat) (Γ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts op fs, attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, outs), some maps, some iters, [body] =>
      if (∀ f ∈ outs, f.initialized = true) ∧ (∀ f ∈ outs, f.ty.isTensor = true) ∧
          op.results.length = outs.length ∧
          (structuredRanges? maps iters.length ((ins ++ outs).map (·.ty))).isSome ∧
          checkRegion (leafBook ib) Γ body ((ins ++ outs).map elementFact)
            (outs.map fun f => f.ty.eltScalar) then
        some (clean (outs.map (·.ty)))
      else none
    | _, _, _, _ => none
  else none

theorem genericDefinedRule_sound (ib : Nat) : ∀ Γ op fs rs,
    genericDefinedRule ib Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds ib Γ env → FactsHold ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      FactsHold ib vs rs := by
  intro Γ op fs rs h ctx call env args hib hΓ ha
  unfold genericDefinedRule at h
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
        have hb : BodyTotal ib (fun ctx => evalRegion opSem ctx call env body)
            ((ins ++ clean (outs.map (·.ty))).map elementFact)
            ((outs.map (·.ty)).map VTy.eltScalar) := by
          intro ctx hib args ha
          rw [← hclean] at ha
          simpa only [List.map_map, Function.comp_def] using
            checkRegion_sound (leafBook ib) hbody hib hΓ ha
        have hs : StructuredOk maps iters.length (ins.map (·.ty) ++ outs.map (·.ty)) ranges := by
          simpa only [List.map_append] using structuredRanges?_spec hranges
        have hout : ∀ t ∈ outs.map (·.ty), t.isTensor = true := by
          intro t ht'
          obtain ⟨f, hf, rfl⟩ := List.mem_map.1 ht'
          exact ht f hf
        obtain ⟨rs, hsem, hrf⟩ := genericSem_total hib hs hout hxs hys' hb
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

def book (ib : Nat) : Book opSem ib where
  rule Γ op fs := match (leafBook ib).rule Γ op fs with
    | some rs => some rs
    | none => (fillDefinedRule ib Γ op fs).orElse (fun _ => genericDefinedRule ib Γ op fs)
  sound Γ op fs rs h ctx call env args hib hΓ ha := by
    split at h
    · next _ hl =>
      cases h
      exact (leafBook ib).sound Γ op fs _ hl ctx call env args hib hΓ ha
    · cases hf : fillDefinedRule ib Γ op fs with
      | some fs' =>
        simp only [hf, Option.orElse_some, Option.some.injEq] at h
        subst h
        exact fillDefinedRule_sound ib Γ op fs fs' hf ctx call env args hib hΓ ha
      | none =>
        simp only [hf, Option.orElse_none] at h
        exact genericDefinedRule_sound ib Γ op fs rs h ctx call env args hib hΓ ha

/-- Acceptance proves successful, nonpoison results for every typed, nonpoison input. -/
theorem definedCheck_sound {ib : Nat} {m : Module} {f : String} {argTys resTys : List VTy}
    (h : check (book ib) m f argTys resTys = true) :
    FuncDefined opSem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys ∧ ResultsDefined args) m f := check_sound (book ib) h

end MerlinVerify.Check.Definedness
