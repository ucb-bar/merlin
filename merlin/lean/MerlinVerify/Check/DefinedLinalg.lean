/- Proved definedness rules for structured tensor programs and the assembled checker. -/
import MerlinVerify.Check.DefinedLoops
import MerlinVerify.Check.DefinedComposition
import MerlinVerify.Check.DefinedDequant
import MerlinVerify.Rules.Generalize

namespace MerlinVerify.Check.Definedness

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

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

/-- Single-output generics may start from an uninitialized destination when their output map is
identity and the iteration box is exactly the output shape. The body must be total even on a dirty
output element. This is a coverage proof, not an assumption that parallel iterators imply coverage. -/
def overwriteDefinedRule (ib : Nat) (Γ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts op fs, attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, [⟨.tensor sh e, _⟩]), some maps, some iters, [body] =>
      if op.results.length = 1 ∧ maps.drop ins.length = [identityMap sh.length] ∧
          structuredRanges? maps iters.length (ins.map (·.ty) ++ [.tensor sh e]) = some sh ∧
          checkRegion (leafBook ib) Γ body ((ins ++ [(⟨.tensor sh e, false⟩ : Fact)]).map elementFact) [.scalar e] then
        some (clean [.tensor sh e])
      else none
    | _, _, _, _ => none
  else none

theorem overwriteDefinedRule_sound (ib : Nat) : ∀ Γ op fs rs,
    overwriteDefinedRule ib Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds ib Γ env → FactsHold ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      FactsHold ib vs rs := by
  intro Γ op fs rs h ctx call env args hib hΓ ha
  unfold overwriteDefinedRule at h
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
          have hb : BodyTotal ib (fun ctx => evalRegion opSem ctx call env body)
              ((ins ++ [(⟨.tensor sh e, false⟩ : Fact)]).map elementFact) [.scalar e] := by
            intro ctx hib args ha
            exact checkRegion_sound (leafBook ib) hbody hib hΓ ha
          obtain ⟨rs, hsem, hrf⟩ := overwriteSem_total hib (structuredRanges?_spec hranges) hout hxs ht.1 hb
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

/-- Check a named contraction through its proved generic expansion, then transfer successful
execution back to the actual source. The input arity and actual body closure are checked as well. -/
def contractionDefinedRule (ib : Nat) (Γ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) := do
  let v ← Rules.Generalize.expand op
  match op.regions with
  | [_] => if fs.length = 3 then genericDefinedRule ib Γ v fs else none
  | _ => none

theorem contractionDefinedRule_sound (ib : Nat) : ∀ Γ op fs rs,
    contractionDefinedRule ib Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds ib Γ env → FactsHold ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      FactsHold ib vs rs := by
  intro Γ op fs rs h ctx call env args hib hΓ ha
  simp only [contractionDefinedRule] at h
  cases he : Rules.Generalize.expand op with
  | none => simp [he] at h
  | some v =>
    simp only [he] at h
    split at h
    · next body hregs =>
      split at h
      · next hlen =>
        obtain ⟨vs, hv, hvf⟩ := genericDefinedRule_sound ib Γ v fs rs h ctx call env args hib hΓ ha
        have hfields := Rules.Generalize.expand_sound op v he
        rw [hfields.2.2.2.2.1, hregs] at hv
        have hlegal : Rules.Generalize.legal op v := by
          dsimp only [Rules.Generalize.expand] at he
          split at he
          · next hc => cases he; exact hc.1
          · cases he
        refine ⟨vs, ?_, hvf⟩
        rw [hregs]
        exact Rules.Generalize.meaning_success hlegal ctx call
          ((List.Forall₂.length_eq ha).trans hlen) _ hv
      · cases h
    · cases h

/-- Map visits the entire destination with identity maps. Keep the destination's initialization
fact when checking the actual body: an initialized destination may be read, a dirty one may only
be overwritten. Captures are checked in the original environment. -/
def mapDefinedRule (ib : Nat) (Γ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.map" then
    match fs.reverse, op.regions with
    | (out@⟨.tensor sh e, _⟩) :: insRev, [body] =>
      let maps := List.replicate fs.length (identityMap sh.length)
      if op.results.length = 1 ∧
          structuredRanges? maps sh.length (insRev.reverse.map (·.ty) ++ [.tensor sh e]) = some sh ∧
          checkRegion (leafBook ib) Γ body ((insRev.reverse ++ [out]).map elementFact) [.scalar e] then
        some (clean [.tensor sh e])
      else none
    | _, _ => none
  else none

theorem mapDefinedRule_sound (ib : Nat) : ∀ Γ op fs rs,
    mapDefinedRule ib Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds ib Γ env → FactsHold ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      FactsHold ib vs rs := by
  intro Γ op fs rs h ctx call env args hib hΓ ha
  unfold mapDefinedRule at h
  split at h
  · next hc =>
    obtain ⟨hm, hd, hn⟩ := hc
    split at h
    · next sh e init insRev body hrev hregs =>
      dsimp only at h
      split at h
      · next hc =>
        obtain ⟨hres, hranges, hbody⟩ := hc
        cases h
        have hlen : fs.length = insRev.length + 1 := by
          simpa only [List.length_reverse, List.length_cons] using congrArg List.length hrev
        have ha' := forall₂_reverse ha
        rw [hrev] at ha'
        cases hv : args.reverse with
        | nil => rw [hv] at ha'; cases ha'
        | cons dst xs =>
          rw [hv] at ha'
          obtain _ | ⟨ht, hxs⟩ := ha'
          cases dst with
          | scalar _ => cases ht.1
          | tensor t =>
            have hins := forall₂_reverse hxs
            have hb : BodyTotal ib (fun ctx => evalRegion opSem ctx call env body)
                ((insRev.reverse ++ [(⟨.tensor sh e, init⟩ : Fact)]).map elementFact) [.scalar e] := by
              intro ctx hib args ha
              exact checkRegion_sound (leafBook ib) hbody hib hΓ ha
            have hs : StructuredOk (List.replicate fs.length (identityMap sh.length))
                (LinalgNamed.parallel sh.length).length
                (insRev.reverse.map (·.ty) ++ [.tensor sh e]) sh := by
              simpa only [LinalgNamed.parallel, List.length_replicate] using structuredRanges?_spec hranges
            have htotal : ∃ vs,
                genericSem ctx (List.replicate fs.length (identityMap sh.length))
                  (LinalgNamed.parallel sh.length) xs.reverse [.tensor t]
                  (fun ctx => evalRegion opSem ctx call env body) = .ok vs ∧
                FactsHold ib vs (clean [.tensor sh e]) := by
              cases init
              · exact overwriteSem_total hib hs (by simp [hlen]) hins ht.1 hb
              · exact genericSem_total hib hs (by simp [VTy.isTensor]) hins (.cons ht .nil) hb
            obtain ⟨vs, hsem, hrf⟩ := htotal
            refine ⟨vs, ?_, hrf⟩
            have hcount : vs.length = op.blind.results.length := by
              rw [HasTys.length (clean_typed hrf), Op.blind_results, List.length_map, hres]
              rfl
            rw [named_sem_of hm hd (by simp [hn])]
            simp only [hregs, evalRegions, LinalgNamed.sem, Op.blind_name, hn, hv,
              shapeOf_typed ht.1, VTy.shape, List.map_const', List.length_append,
              List.length_reverse, List.length_singleton, List.Forall₂.length_eq hxs,
              ← hlen, LinalgNamed.run, hsem, Outcome.bind_ok, hcount, ↓reduceIte]
      · cases h
    · cases h
  · cases h

def structuredBook (ib : Nat) : Book opSem ib where
  rule Γ op fs := match (leafBook ib).rule Γ op fs with
    | some rs => some rs
    | none => [fillDefinedRule ib, genericDefinedRule ib, overwriteDefinedRule ib,
        contractionDefinedRule ib, mapDefinedRule ib].findSome? fun r => r Γ op fs
  sound Γ op fs rs h ctx call env args hib hΓ ha := by
    split at h
    · next _ hl =>
      cases h
      exact (leafBook ib).sound Γ op fs _ hl ctx call env args hib hΓ ha
    · obtain ⟨r, hmem, hr⟩ := List.exists_of_findSome?_eq_some h
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem
      rcases hmem with rfl | rfl | rfl | rfl | rfl
      · exact fillDefinedRule_sound ib Γ op fs rs hr ctx call env args hib hΓ ha
      · exact genericDefinedRule_sound ib Γ op fs rs hr ctx call env args hib hΓ ha
      · exact overwriteDefinedRule_sound ib Γ op fs rs hr ctx call env args hib hΓ ha
      · exact contractionDefinedRule_sound ib Γ op fs rs hr ctx call env args hib hΓ ha
      · exact mapDefinedRule_sound ib Γ op fs rs hr ctx call env args hib hΓ ha

/-- Independent families compose through their totality proofs. The structured family retains
its existing behavior; dequantization adds source-definedness on its explicitly checked domain. -/
def book (ib : Nat) : Book opSem ib := (structuredBook ib).orElse (dequantBook ib)

/-- Acceptance proves successful, nonpoison results for every typed, nonpoison input. -/
theorem definedCheck_sound {ib : Nat} {m : Module} {f : String} {argTys resTys : List VTy}
    (h : check (book ib) m f argTys resTys = true) :
    FuncDefined opSem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys ∧ ResultsDefined args) m f := check_sound (book ib) h

end MerlinVerify.Check.Definedness
