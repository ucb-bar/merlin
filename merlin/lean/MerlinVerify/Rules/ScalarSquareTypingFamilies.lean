/- Relational, successful-result versions of the shared structured-operation rules.
The executable rules are reused unchanged; the proof accepts partial (possibly stuck) bodies. -/
import MerlinVerify.Rules.ScalarSquareTypingLinalg

namespace MerlinVerify.Rules.ScalarSquareTyping
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

abbrev BodyPair := (Ctx → RegionSem) × (Ctx → RegionSem)
def RelatedRuleSound (ib : Nat) (rule : OpRule) : Prop :=
  ∀ d op argTys resTys sigs, unmodelledAttr? op.blind = none →
    rule d op argTys resTys = some sigs →
    ∀ ctx callS callT args (bodies : List BodyPair), ctx.indexBits = ib →
      HasTys ib args argTys →
      List.Forall₂ (RegionRelated ctx.M ib) bodies sigs →
      Related (HasTys ib · resTys)
        (opSem ctx callS op args (bodies.map Prod.fst))
        (opSem ctx callT op args (bodies.map Prod.snd))

theorem run_related {M : FloatModel} {ib : Nat} {ctx : Ctx} (hm : ctx.M = M) (hib : ctx.indexBits = ib) {op : Op} {maps : List AffineMap}
    {iters : List IterKind} {ins outs : List RVal} {bodyS bodyT : Ctx → RegionSem} {inTys outTys : List VTy}
    {ranges : List Nat} (hs : StructuredOk maps iters.length (inTys ++ outTys) ranges)
    (hout : ∀ t ∈ outTys, t.isTensor = true) (hins : HasTys ib ins inTys)
    (houts : HasTys ib outs outTys)
    (hbody : RegionRelated M ib (bodyS, bodyT)
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, iters.length⟩)
    (hres : op.results.length = outTys.length) :
    Related (HasTys ib · outTys) (LinalgNamed.run ctx op maps iters ins outs bodyS)
      (LinalgNamed.run ctx op maps iters ins outs bodyT) := by
  unfold LinalgNamed.run
  refine Related.bind (genericSem_related hm hib hs hout hins houts hbody) fun rs hrs => ?_
  have hlen : rs.length = op.results.length := by rw [HasTys.length hrs, hres]
  simp only [hlen, ↓reduceIte]
  exact Related.ok hrs


theorem namedOk_related {M : FloatModel} {ib : Nat} {ctx : Ctx} (hm : ctx.M = M) (hib : ctx.indexBits = ib) {maps : List AffineMap}
    {iters : List IterKind} {inTys outTys : List VTy} {op : Op} {resTys : List VTy}
    {sigs : List RegionSig} (h : namedOk maps iters inTys outTys op resTys = some sigs)
    {ins outs : List RVal} {regions : List ((Ctx → RegionSem) × (Ctx → RegionSem))} (hins : HasTys ib ins inTys)
    (houts : HasTys ib outs outTys) (hregs : List.Forall₂ (RegionRelated M ib) regions sigs) :
    ∃ body, regions = [body] ∧
      Related (HasTys ib · resTys) (LinalgNamed.run ctx op.blind maps iters ins outs body.1)
        (LinalgNamed.run ctx op.blind maps iters ins outs body.2) := by
  unfold namedOk at h
  split at h
  · next hc =>
    obtain ⟨hout, rfl, hres, hsr⟩ := hc
    obtain ⟨ranges, hranges⟩ := Option.isSome_iff_exists.1 hsr
    cases h
    obtain _ | ⟨hbody, hnil⟩ := hregs
    cases hnil
    exact ⟨_, rfl, run_related hm hib (structuredRanges?_spec hranges) hout hins houts hbody
      (by rw [Op.blind_results, List.length_map, hres])⟩
  · cases h


theorem genericRule_related (ib : Nat) : RelatedRuleSound ib genericRule := by
  intro d op argTys resTys sigs hm h
  simp only [genericRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next inTys outTys maps iters hsplit hmaps hiters =>
      split at h
      · next hc2 =>
        obtain ⟨hout, rfl, hres, hsr⟩ := hc2
        obtain ⟨ranges, hranges⟩ := Option.isSome_iff_exists.1 hsr
        cases h
        intro ctx callS callT args regions hib hargs hregs
        obtain _ | ⟨hbody, hnil⟩ := hregs
        cases hnil
        obtain ⟨ins, outs, hio, hins, houts⟩ := splitInsOuts_typed hsplit hargs
        simp only [List.map_cons, List.map_nil, linalg_sem_of hm hd (.inl hn)]
        simp only [Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, attrAffineMaps?_blind,
          iterKinds?_blind, hio, hmaps, hiters, Outcome.bindOp_eq]
        refine Related.bind (genericSem_related rfl hib (structuredRanges?_spec hranges) hout hins houts hbody)
          fun rs hrs => ?_
        have hlen : rs.length = op.blind.results.length := by
          rw [HasTys.length hrs, Op.blind_results, List.length_map, hres]
        simp only [hlen, ↓reduceIte, Outcome.pure_eq]
        exact Related.ok hrs
      · cases h
    · cases h
  · cases h


theorem fillRule_related (ib : Nat) : RelatedRuleSound ib fillRule := by
  intro d op argTys resTys sigs hm h
  simp only [fillRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next vt sh e hsplit =>
      split at h
      · next hc2 =>
        obtain ⟨rfl, hres, hsr⟩ := hc2
        obtain ⟨ranges, hranges⟩ := Option.isSome_iff_exists.1 hsr
        cases h
        intro ctx callS callT args regions hib hargs hregs
        obtain _ | ⟨hbody, hnil⟩ := hregs
        cases hnil
        rename_i body
        obtain ⟨ins, outs, hio, hins, houts⟩ := splitInsOuts_typed hsplit hargs
        obtain _ | ⟨hv, _ | _⟩ := hins
        obtain _ | ⟨ht, _ | _⟩ := houts
        rename_i v x
        cases x with
        | scalar _ => exact (ht : False).elim
        | tensor t =>
        have hts : t.shape = sh := ht.1
        simp only [List.map_cons, List.map_nil, linalg_sem_of hm hd (.inr (.inl hn))]
        simp only [Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hio, hts, Outcome.bindOp_eq]
        have hs : StructuredOk (fillMaps sh.length) (List.replicate sh.length IterKind.parallel).length
            ([vt] ++ [VTy.tensor sh e]) ranges := by
          rw [List.length_replicate]; exact structuredRanges?_spec hranges
        have hbody' : RegionRelated ctx.M ib body
            ⟨([vt] ++ [VTy.tensor sh e]).map VTy.eltScalar, [VTy.tensor sh e].map VTy.eltScalar,
              (List.replicate sh.length IterKind.parallel).length⟩ := by
          rw [List.length_replicate]; exact hbody
        refine Related.bind (genericSem_related (ins := [v]) (outs := [.tensor t]) rfl hib hs
          (by simp [VTy.isTensor]) (List.Forall₂.cons hv .nil) (List.Forall₂.cons ht .nil) hbody')
          fun rs hrs => ?_
        have hlen : rs.length = op.blind.results.length := by
          rw [HasTys.length hrs, Op.blind_results, List.length_map, hres]; rfl
        simp only [hlen, ↓reduceIte, Outcome.pure_eq]
        exact Related.ok hrs
      · cases h
    · cases h
  · cases h


theorem matmulRule_related (ib : Nat) : RelatedRuleSound ib matmulRule := by
  intro d op argTys resTys sigs hm h
  simp only [matmulRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next ta tb tc hsplit =>
      intro ctx callS callT args regions hib hargs hregs
      obtain ⟨ins, outs, hio, hins, houts⟩ := splitInsOuts_typed hsplit hargs
      obtain _ | ⟨ha, _ | ⟨hb, _ | _⟩⟩ := hins
      obtain _ | ⟨hc', _ | _⟩ := houts
      obtain ⟨body, rfl, hsafe⟩ := namedOk_related rfl hib h (.cons ha (.cons hb .nil)) (.cons hc' .nil) hregs
      simp only [List.map_cons, List.map_nil, named_sem_of hm hd (by simp [hn])]
      simp only [LinalgNamed.sem, Op.blind_name, hn, splitInsOuts_blind, hio, mapsOr_blind]
      exact hsafe
    · cases h
  · cases h


theorem transposeRule_related (ib : Nat) : RelatedRuleSound ib transposeRule := by
  intro d op argTys resTys sigs hm h
  simp only [transposeRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next ti tout perm hperm =>
      split at h
      · next hcond =>
        intro ctx callS callT args regions hib hargs hregs
        obtain _ | ⟨hi, _ | ⟨ho, _ | _⟩⟩ := hargs
        obtain ⟨body, rfl, hsafe⟩ := namedOk_related rfl hib h (.cons hi .nil) (.cons ho .nil) hregs
        simp only [List.map_cons, List.map_nil, named_sem_of hm hd (by simp [hn])]
        simp only [LinalgNamed.sem, Op.blind_name, hn, natArray?_blind, hperm, shapeOf_typed ho]
        simp only [hcond, ↓reduceIte]
        exact hsafe
      · cases h
    · cases h
  · cases h


theorem reduceRule_related (ib : Nat) : RelatedRuleSound ib reduceRule := by
  intro d op argTys resTys sigs hm h
  simp only [reduceRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next ti ins outs dims hsplit hdims =>
      intro ctx callS callT args regions hib hargs hregs
      obtain ⟨xs, ys, hio, hxs, hys⟩ := reductionSplit_typed hsplit hargs
      obtain _ | ⟨hi, hrest⟩ := hxs
      obtain ⟨body, rfl, hsafe⟩ := namedOk_related rfl hib h (.cons hi hrest) hys hregs
      simp only [List.map_cons, List.map_nil, named_sem_of hm hd (by simp [hn])]
      dsimp only [reductionSplit] at hio
      simp only [LinalgNamed.sem, Op.blind_name, hn, natArray?_blind, hdims,
        splitInsOuts_blind, hio, shapeOf_typed hi,
        List.map_const', List.length_cons, HasTys.length hrest, HasTys.length hys]
      exact hsafe
    · cases h
  · cases h


end MerlinVerify.Rules.ScalarSquareTyping
