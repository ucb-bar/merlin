/- Totality and initialization for integer comparisons and selection. These
rules establish defined source executions; they assume no algebraic laws for floating values. -/
import MerlinVerify.Check.DefinedRules
import MerlinVerify.Check.DefinedComposition

namespace MerlinVerify.Check.Definedness

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

theorem initialized_integer {ib w : Nat} {ty : Ty} {v : RVal}
    (hw : intWidth? ib ty = some w) (h : Fact.Holds ib v ⟨.scalar ty, true⟩) :
    ∃ x : BitVec w, v = .scalar (.some (.bits w x)) := by
  cases v with
  | tensor t => cases h.1
  | scalar x =>
    rcases hasTy_intWidth hw h.1 with hp | ⟨a, ha⟩
    · exact (h.2 rfl hp).elim
    · exact ⟨a, congrArg RVal.scalar ha⟩

/-- The predicate is decoded by the actual semantics. Integer and index comparisons are total
for initialized operands of the same checked type. -/
def compareRule (ib : Nat) : Rule := fun op fs =>
  match fs, op.results with
  | [⟨.scalar t, true⟩, ⟨.scalar t', true⟩], [r] =>
    if dialectOf op.name = "arith" ∧ op.name = "arith.cmpi" ∧ t' = t ∧ r.ty = .int 1 then
      match intWidth? ib t, attrInt? op "predicate" >>= Arith.cmpiPred with
      | some _, some _ => some [⟨.scalar (.int 1), true⟩]
      | _, _ => none
    else none
  | _, _ => none

theorem compareRule_sound (ib : Nat) : RuleSound ib (compareRule ib) := by
  intro op fs rs hm h ctx call args _ ha
  unfold compareRule at h
  split at h
  · split at h
    · next hc =>
      obtain ⟨hd, hn, hty, hrty⟩ := hc
      subst hty
      split at h
      · next w p hw hp =>
        cases h
        obtain _ | ⟨hx, hy⟩ := ha
        obtain _ | ⟨hy, hnil⟩ := hy
        cases hnil
        obtain ⟨x, rfl⟩ := initialized_integer hw hx
        obtain ⟨y, rfl⟩ := initialized_integer hw hy
        have hresult : Arith.resultElt op.blind = some (.int 1) := by
          simp_all [Arith.resultElt, Op.blind_results]
        rw [opSem_modelled hm]
        simp only [dispatch, Op.blind_name, hd]
        simp only [Arith.sem, Op.blind_name, hn, attrInt?_blind, hp,
          Arith.binOp, Arith.pureN, hresult]
        rw [show [.scalar (.some (.bits w x)), .scalar (.some (.bits w y))] =
          ([.some (.bits w x), .some (.bits w y)] : List Val).map RVal.scalar from rfl,
          liftEager_scalars]
        simp only [Arith.intBin, ↓reduceDIte, Outcome.bind_ok]
        exact ⟨_, rfl, List.Forall₂.cons ⟨by simp [Arith.boolVal, RVal.HasTy,
          Val.HasTy, Scalar.conforms], by simp [Arith.boolVal, RVal.Defined]⟩ .nil⟩
      · cases h
    · cases h
  · cases h

theorem defined_canonize {v : RVal} (h : v.Defined) : v.canonize.Defined := by
  cases v with
  | scalar x => exact h
  | tensor t =>
    intro i hi
    change Idx.InBounds t.shape i at hi
    simpa only [RVal.canonize, Tensor.canonize, Tensor.canon, hi, ↓reduceIte] using h i hi

/-- Tensor facts justify both the type and initialization of an actual in-bounds element. -/
theorem Fact.element {ib : Nat} {t : Tensor} {sh : List Nat} {e : Ty} {initialized : Bool}
    (h : Fact.Holds ib (.tensor t) ⟨.tensor sh e, initialized⟩)
    {i : List Nat} (hi : Idx.InBounds t.shape i) :
    Fact.Holds ib (.scalar (t.get i)) ⟨.scalar e, initialized⟩ :=
  ⟨h.1.2.2 i (h.1.1 ▸ hi), fun hinit => h.2 hinit i hi⟩

/-- A scalar initialized i1 condition selects one of two initialized values of the same type.
The values may be scalars or static tensors, including floating-point tensors. -/
def scalarSelectRule (ib : Nat) : Rule := fun op fs =>
  match fs, op.results with
  | [⟨.scalar (.int 1), true⟩, ⟨t, true⟩, ⟨t', true⟩], [r] =>
    if dialectOf op.name = "arith" ∧ op.name = "arith.select" ∧ t' = t ∧
        vtyOf ib r.ty = some t then some [⟨t, true⟩]
    else none
  | _, _ => none

theorem scalarSelectRule_sound (ib : Nat) : RuleSound ib (scalarSelectRule ib) := by
  intro op fs rs hm h ctx call args _ ha
  unfold scalarSelectRule at h
  split at h
  · split at h
    · next hc =>
      obtain ⟨hd, hn, hty, _⟩ := hc
      subst hty
      cases h
      obtain _ | ⟨hc, hrest⟩ := ha
      obtain _ | ⟨ht, hrest⟩ := hrest
      obtain _ | ⟨hf, hnil⟩ := hrest
      cases hnil
      obtain ⟨c, rfl⟩ := initialized_int hc
      rw [opSem_modelled hm]
      simp only [dispatch, Op.blind_name, hd]
      simp only [Arith.sem, Op.blind_name, hn]
      split
      · exact ⟨_, rfl, .cons ⟨hasTy_canonize ht.1, fun _ => defined_canonize (ht.2 rfl)⟩ .nil⟩
      · exact ⟨_, rfl, .cons ⟨hasTy_canonize hf.1, fun _ => defined_canonize (hf.2 rfl)⟩ .nil⟩
    · cases h
  · cases h

/-- Elementwise selection requires initialized same-shape tensors, with an i1 condition. -/
def tensorSelectRule (ib : Nat) : Rule := fun op fs =>
  match fs, op.results with
  | [⟨.tensor sh (.int 1), true⟩, ⟨.tensor sh' e, true⟩,
      ⟨.tensor sh'' e', true⟩], [r] =>
    if dialectOf op.name = "arith" ∧ op.name = "arith.select" ∧ sh' = sh ∧ sh'' = sh ∧
        e' = e ∧ vtyOf ib r.ty = some (.tensor sh e) then
      some [⟨.tensor sh e, true⟩]
    else none
  | _, _ => none

theorem tensorSelectRule_sound (ib : Nat) : RuleSound ib (tensorSelectRule ib) := by
  intro op fs rs hm h ctx call args _ ha
  unfold tensorSelectRule at h
  split at h
  · split at h
    · next hc =>
      obtain ⟨hd, hn, hst, hsf, he, _⟩ := hc
      subst hst hsf he
      cases h
      obtain _ | ⟨hc, hrest⟩ := ha
      obtain _ | ⟨ht, hrest⟩ := hrest
      obtain _ | ⟨hf, hnil⟩ := hrest
      cases hnil
      rename_i c t f
      cases c with
      | scalar _ => cases hc.1
      | tensor c =>
        cases t with
        | scalar _ => cases ht.1
        | tensor t =>
          cases f with
          | scalar _ => cases hf.1
          | tensor f =>
            have hct : c.shape = t.shape := hc.1.1.trans ht.1.1.symm
            have htf : t.shape = f.shape := ht.1.1.trans hf.1.1.symm
            have point : ∀ i, Idx.InBounds t.shape i →
                Val.HasTy ib t.elt (match c.get i with
                  | .some (.bits 1 b) => if b = 1 then t.get i else f.get i
                  | _ => .poison) ∧
                (match c.get i with
                  | .some (.bits 1 b) => if b = 1 then t.get i else f.get i
                  | _ => .poison) ≠ .poison := by
              intro i hi
              have hic : Idx.InBounds c.shape i := hct ▸ hi
              have hif : Idx.InBounds f.shape i := htf ▸ hi
              obtain ⟨b, hb⟩ := initialized_int (Fact.element hc hic)
              have hb' : c.get i = .some (.bits 1 b) := RVal.scalar.inj hb
              rw [hb']
              simp only
              split
              · exact ⟨by simpa only [ht.1.2.1] using ht.1.2.2 i (ht.1.1 ▸ hi), ht.2 rfl i hi⟩
              · exact ⟨by simpa only [ht.1.2.1, hf.1.2.1] using hf.1.2.2 i (hf.1.1 ▸ hif), hf.2 rfl i hif⟩
            have hshapes : (c.shape == t.shape && t.shape == f.shape) = true := by
              simp [hct, htf]
            rw [opSem_modelled hm]
            simp only [dispatch, Op.blind_name, hd]
            simp only [Arith.sem, Op.blind_name, hn, hshapes, ↓reduceIte]
            refine ⟨_, rfl, .cons ⟨⟨ht.1.1, ht.1.2.1, ?_⟩, ?_⟩ .nil⟩
            · intro i hi
              have hit : Idx.InBounds t.shape i := ht.1.1.symm ▸ hi
              simpa only [Tensor.canon, hit, ↓reduceIte] using
                ht.1.2.1.symm ▸ (point i hit).1
            · intro _ i hi
              change Idx.InBounds t.shape i at hi
              have hic : Idx.InBounds c.shape i := hct ▸ hi
              obtain ⟨b, hb⟩ := initialized_int (Fact.element hc hic)
              have hb' : c.get i = .some (.bits 1 b) := RVal.scalar.inj hb
              simp only [Tensor.canon, hi, ↓reduceIte, hb']
              split
              · exact ht.2 rfl i hi
              · exact hf.2 rfl i (htf ▸ hi)
    · cases h
  · cases h

def controlBook (ib : Nat) : Book opSem ib where
  rule _ op fs :=
    if unmodelledAttr? op.blind = none ∧ op.regions = [] then
      [compareRule ib, scalarSelectRule ib, tensorSelectRule ib].findSome? fun r => r op fs
    else none
  sound Γ op fs rs h ctx call env args hib _ ha := by
    split at h
    · next hc =>
      obtain ⟨hm, hregs⟩ := hc
      obtain ⟨r, hmem, hr⟩ := List.exists_of_findSome?_eq_some h
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem
      rcases hmem with rfl | rfl | rfl
      · simpa only [hregs, evalRegions] using compareRule_sound ib op fs rs hm hr ctx call args hib ha
      · simpa only [hregs, evalRegions] using scalarSelectRule_sound ib op fs rs hm hr ctx call args hib ha
      · simpa only [hregs, evalRegions] using tensorSelectRule_sound ib op fs rs hm hr ctx call args hib ha
    · cases h

/-- Preserve all existing leaf cases and add independently proved control operations. -/
def controlLeafBook (ib : Nat) : Book opSem ib := (leafBook ib).orElse (controlBook ib)

end MerlinVerify.Check.Definedness
