/- Bounds for actual linalg.index coordinates, integer constants and equal-width casts. -/
import MerlinVerify.Rules.FrontendRangeArithmetic

namespace MerlinVerify.Rules.FrontendRangeScalars
open MerlinVerify MerlinVerify.Check Sem.Ops
open FrontendTensorRanges FrontendRangeFrame

/-- Coordinates use the checked runtime loop extent, not an annotation or a depth alone. -/
def indexRule (ib : Nat) (ranges : List Nat) (_ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.index" ∧
      fs = [] ∧ op.results.map (·.ty) = [.index] ∧ op.regions = [] ∧ 0 < ib then
    match attrInt? op "dim" with
    | some d => if 0 ≤ d ∧ d.toNat < ranges.length ∧ ranges[d.toNat]! ≤ 2 ^ (ib - 1) then
        some [⟨⟨.scalar .index, true⟩, some ⟨0, ranges[d.toNat]!⟩⟩]
      else none
    | none => none
  else none

def indexBook (ib : Nat) (ranges : List Nat) : BookAt ib ranges where
  rule := indexRule ib ranges
  sound Γ op fs rs h ctx call env args hib hf _ ha := by
    unfold indexRule at h
    split at h
    · next hc =>
      obtain ⟨hm, hd, hn, rfl, _, _, hwidth⟩ := hc
      split at h
      · next d hdim =>
        split at h
        · next hc =>
          obtain ⟨hdpos, hdlt, hlimit⟩ := hc
          cases h
          cases ha
          have hframe : Idx.InBounds ranges ctx.loopIndex := by
            rcases hf with he | hf
            · subst ranges; simp at hdlt
            · exact hf
          have hlt : d.toNat < ctx.loopIndex.length := by rw [hframe.1]; exact hdlt
          have hi := List.getElem?_eq_getElem hlt
          have hcoord := hframe.2 d.toNat hlt hdlt
          have hdim' : attrInt? op.blind "dim" = some d := by simpa [attrInt?] using hdim
          have hget : ranges[d.toNat]! = ranges[d.toNat] := by simp [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hdlt]
          rw [hget] at hlimit
          have hencoded := FrontendMaskBounds.indexNat_encode hwidth (show ctx.loopIndex[d.toNat] < 2 ^ (ib - 1) by omega)
          obtain ⟨_, bits, he, hbits⟩ := FrontendMaskBounds.indexNat_spec hencoded
          have hbits' : (BitVec.ofNat ib ctx.loopIndex[d.toNat]).toInt = (ctx.loopIndex[d.toNat] : Int) := by
            cases he; exact hbits
          rw [opSem_modelled hm]
          have hd' : dialectOf "linalg.index" = "linalg" := by cbv
          simp only [dispatch, Op.blind_name, hn, hd', beq_self_eq_true, Bool.or_true, ↓reduceIte]
          simp only [Linalg.sem, Op.blind_name, hn, hdim', hi, hdpos, ↓reduceIte]
          refine ⟨[.scalar (.some (.bits ib (BitVec.ofNat ib ctx.loopIndex[d.toNat])))], ?_, .cons ⟨⟨?_, ?_⟩, ?_⟩ .nil⟩
          · cases ctx with
            | mk floatModel ext width loopIndex =>
              dsimp at hib ⊢
              subst width
              rfl
          · simp [RVal.HasTy, Val.HasTy, Scalar.conforms]
          · simp [RVal.Defined]
          · change ∃ z, Definedness.IndexRanges.value? (.scalar (.some (.bits ib _))) = some z ∧ _ ∧ _
            refine ⟨ctx.loopIndex[d.toNat], ?_, ?_, ?_⟩
            · simpa [Definedness.IndexRanges.value?, Tensor.indexVal?] using congrArg some hbits'
            · simp
            · dsimp; rw [hget]; exact_mod_cast hcoord
        · cases h
      · cases h
    · cases h

def constantRule (ib : Nat) (_ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "arith" ∧ op.name = "arith.constant" ∧
      op.regions = [] ∧ fs = [] then
    match op.results, op.attr? "value" with
    | [r], some (.int n (.int w)) =>
      let actual := (BitVec.ofInt ib n).toInt
      if w = ib ∧ r.ty = .int ib ∧ 0 ≤ actual then
        some [⟨⟨.scalar (.int ib), true⟩, some ⟨actual.toNat, actual.toNat + 1⟩⟩]
      else none
    | _, _ => none
  else none

def constantBook (ib : Nat) : Definedness.Refined.Book model opSem ib where
  rule := constantRule ib
  sound Γ op fs rs h ctx call env args hib _ ha := by
    unfold constantRule at h
    split at h
    · next hc =>
      obtain ⟨hm, hd, hn, hregs, rfl⟩ := hc
      cases ha
      split at h
      · next r n w hr hv =>
        dsimp only at h
        split at h
        · next hc =>
          obtain ⟨hw, _, hnonneg⟩ := hc
          subst w
          cases h
          refine ⟨[.scalar (.some (.bits ib (BitVec.ofInt ib n)))], ?_, .cons ⟨?_, ?_⟩ .nil⟩
          · rw [hregs, evalRegions, opSem_modelled hm]
            simp only [dispatch, Op.blind_name, hd]
            simp [Arith.sem, hn, hr, hv, scalarOfAttr]
          · exact ⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩
          · change ∃ z, Definedness.IndexRanges.value? (.scalar (.some (.bits ib (BitVec.ofInt ib n)))) = some z ∧ _ ∧ _
            refine ⟨(BitVec.ofInt ib n).toInt, rfl, ?_, ?_⟩ <;> dsimp <;> omega
        · cases h
      · cases h
    · cases h

def castRule (ib : Nat) (_ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  match fs, op.results with
  | [f], [r] =>
    if unmodelledAttr? op.blind = none ∧ op.regions = [] ∧ dialectOf op.name = "arith" ∧
        op.name = "arith.index_cast" ∧ hasUnit op "nonNeg" = false ∧ f.base.initialized = true ∧
        ((f.base.ty = .scalar .index ∧ r.ty = .int ib) ∨ (f.base.ty = .scalar (.int ib) ∧ r.ty = .index)) then
      some [⟨⟨.scalar r.ty, true⟩, f.info⟩]
    else none
  | _, _ => none

def castBook (ib : Nat) : Definedness.Refined.Book model opSem ib where
  rule := castRule ib
  sound Γ op fs rs h ctx call env args hib _ ha := by
    unfold castRule at h
    split at h
    · next f r hres =>
      split at h
      · next hc =>
        obtain ⟨hm, hregs, hd, hn, hnneg, hinit, hdir⟩ := hc
        cases h
        obtain _ | ⟨hv, hnil⟩ := ha
        cases hnil
        rcases hdir with ⟨hty, hr⟩ | ⟨hty, hr⟩
        all_goals
          have hb : f.base = ⟨f.base.ty, true⟩ := by rw [← hinit]
          rw [hty] at hb
          have hbase := hv.1
          rw [hb] at hbase
          first | have hv' := (FrontendDefinedIndex.index_int_fact ib _).mp hbase | have hv' := hbase
          obtain ⟨x, rfl⟩ := Definedness.initialized_int hv'
          have hresult : Arith.resultElt op.blind = some r.ty := by
            simp [Arith.resultElt, Op.blind_results, hres, hr]
          rw [hregs, evalRegions, opSem_modelled hm]
          simp only [dispatch, Op.blind_name, hd]
          simp only [Arith.sem, Op.blind_name, hn, Arith.widthOf, hresult, hr, Ty.bitWidth?, hib,
            Outcome.ofOption, Outcome.bindOp_eq, Outcome.bind_ok, Arith.unOp, Arith.pureN]
          rw [show [.scalar (.some (.bits ib x))] =
            ([.some (.bits ib x)] : List Val).map RVal.scalar from rfl, liftEager_scalars]
          simp only [Arith.intUn, Nat.le_refl, ↓reduceIte, Outcome.bind_ok, Arith.bv]
          exact ⟨_, rfl, .cons ⟨⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms],
            by simp [RVal.Defined]⟩, by simpa using hv.2⟩ .nil⟩
      · cases h
    · cases h

#print axioms indexBook
#print axioms castBook
end MerlinVerify.Rules.FrontendRangeScalars
