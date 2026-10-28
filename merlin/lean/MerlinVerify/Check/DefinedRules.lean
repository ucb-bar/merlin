/- Proved transfer rules for the straight-line definedness checker. -/
import MerlinVerify.Check.Defined
import MerlinVerify.Check.Rulebook

namespace MerlinVerify.Check.Definedness

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

abbrev Rule := Op → List Fact → Option (List Fact)

def RuleSound (ib : Nat) (r : Rule) : Prop :=
  ∀ op fs rs, unmodelledAttr? op.blind = none → r op fs = some rs → ∀ ctx call args,
    ctx.indexBits = ib → FactsHold ib args fs →
    ∃ vs, opSem ctx call op args [] = .ok vs ∧ FactsHold ib vs rs

def constantRule (ib : Nat) : Rule := fun op fs =>
  if dialectOf op.name = "arith" ∧ op.name = "arith.constant" ∧ fs = [] then
    match op.results, op.attr? "value" with
    | [r], some a =>
      match scalarOfAttr ib a with
      | some s => if s.conforms ib r.ty then some [⟨.scalar r.ty, true⟩] else none
      | none => none
    | _, _ => none
  else none

theorem constantRule_sound (ib : Nat) : RuleSound ib (constantRule ib) := by
  intro op fs rs hm h ctx call args hib hargs
  unfold constantRule at h
  split at h
  · next hc =>
    obtain ⟨hd, hn, rfl⟩ := hc
    cases hargs
    split at h
    · next r a hr ha =>
      split at h
      · next s hs =>
        split at h
        · next hty =>
          cases h
          refine ⟨[.scalar (.some s)], ?_, List.Forall₂.cons ⟨hty, by simp [RVal.Defined]⟩ .nil⟩
          rw [opSem_modelled hm]
          simp only [dispatch, Op.blind_name, hd]
          simp only [Arith.sem, Op.blind_name, hn, Op.blind_results, hr,
            List.map_cons, List.map_nil, Op.blind_attr?, ha, hib]
          cases a
          all_goals first
            | (simp [scalarOfAttr] at hs; done)
            | simp only [hs]
        · cases h
      · cases h
    · cases h
  · cases h

/-- Uninitialized storage is allowed as an intermediate, but its fact never certifies an output. -/
def emptyRule : Rule := fun op fs =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.empty" ∧ fs = [] then
    (Sem.Ops.Tensor.resultTensorTy op).map fun (sh, e) => [⟨.tensor sh e, false⟩]
  else none

theorem emptyRule_sound (ib : Nat) : RuleSound ib emptyRule := by
  intro op fs rs hm h ctx call args _ hargs
  unfold emptyRule at h
  split at h
  · next hc =>
    obtain ⟨hd, hn, rfl⟩ := hc
    cases hargs
    cases hr : Sem.Ops.Tensor.resultTensorTy op with
    | none => simp [hr] at h
    | some p =>
      obtain ⟨sh, e⟩ := p
      simp only [hr, Option.map_some, Option.some.injEq] at h
      subst h
      refine ⟨[.tensor (MerlinVerify.Tensor.full sh e .poison)], ?_,
        List.Forall₂.cons ⟨hasTy_full (hasTy_poison _ _), by simp⟩ .nil⟩
      rw [tensor_sem_of hm hd]
      simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, hr]
  · cases h

/-- `tensor.splat` fully initializes a static tensor from one initialized scalar. -/
def splatRule : Rule := fun op fs =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.splat" then
    match fs, Sem.Ops.Tensor.resultTensorTy op with
    | [⟨.scalar t, true⟩], some (sh, e) =>
      if t = e then some [⟨.tensor sh e, true⟩] else none
    | _, _ => none
  else none

theorem splatRule_sound (ib : Nat) : RuleSound ib splatRule := by
  intro op fs rs hm h ctx call args _ hargs
  unfold splatRule at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next _ _ _ scalarTy shape _ hr =>
      split at h
      · next hte =>
        subst hte
        cases h
        obtain _ | ⟨hv, hnil⟩ := hargs
        cases hnil
        rename_i v
        cases v with
        | tensor t => cases hv.1
        | scalar x =>
          refine ⟨[.tensor (MerlinVerify.Tensor.full shape scalarTy x)], ?_, List.Forall₂.cons
            ⟨hasTy_full hv.1, ?_⟩ .nil⟩
          · rw [tensor_sem_of hm hd]
            simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, hr]
          · intro _ i hi
            change Idx.InBounds shape i at hi
            simpa only [RVal.Defined, MerlinVerify.Tensor.full, MerlinVerify.Tensor.canon, hi,
              ↓reduceIte] using hv.2 rfl
      · cases h
    · cases h
  · cases h

/-- Integer operations that are total on initialized operands when overflow flags are absent.
Division, shifts, casts, and flagged arithmetic require separate obligations and are refused. -/
def intBinaryNames : List String :=
  ["arith.addi", "arith.subi", "arith.muli", "arith.andi", "arith.ori", "arith.xori",
   "arith.maxsi", "arith.minsi", "arith.maxui", "arith.minui"]

def intBinaryRule : Rule := fun op fs =>
  match fs, op.results with
  | [⟨.scalar (.int w), true⟩, ⟨.scalar (.int w'), true⟩], [r] =>
    if dialectOf op.name = "arith" ∧ op.name ∈ intBinaryNames ∧ w' = w ∧
        r.ty = .int w ∧ overflowFlags op = [] then some [⟨.scalar (.int w), true⟩]
    else none
  | _, _ => none

theorem initialized_int {ib w : Nat} {v : RVal}
    (h : Fact.Holds ib v ⟨.scalar (.int w), true⟩) :
    ∃ x : BitVec w, v = .scalar (.some (.bits w x)) := by
  cases v with
  | tensor t => cases h.1
  | scalar x =>
    rcases hasTy_intWidth (t := .int w) rfl h.1 with hp | ⟨a, ha⟩
    · exact (h.2 rfl hp).elim
    · exact ⟨a, congrArg RVal.scalar ha⟩

theorem intBinaryRule_sound (ib : Nat) : RuleSound ib intBinaryRule := by
  intro op fs rs hm h ctx call args _ hargs
  unfold intBinaryRule at h
  split at h
  · split at h
    · next hc =>
      obtain ⟨hd, hn, hw, hty, hflags⟩ := hc
      subst hw
      cases h
      obtain _ | ⟨ha, hb⟩ := hargs
      obtain _ | ⟨hb, hnil⟩ := hb
      cases hnil
      obtain ⟨x, rfl⟩ := initialized_int ha
      obtain ⟨y, rfl⟩ := initialized_int hb
      rename_i w r hr
      have hresult : Arith.resultElt op.blind = some (.int w) := by
        simp only [Arith.resultElt, Op.blind_results, hr, List.map_cons, List.map_nil, hty]
      rw [opSem_modelled hm]
      simp only [dispatch, Op.blind_name, hd]
      simp only [intBinaryNames, List.mem_cons, List.not_mem_nil, or_false] at hn
      rcases hn with hn | hn | hn | hn | hn | hn | hn | hn | hn | hn
      all_goals
        simp only [Arith.sem, Op.blind_name, hn, Arith.binOp, Arith.pureN, hresult,
          Outcome.bindOp_eq,
          show overflowFlags op.blind = [] from by simpa [overflowFlags, attrEnum?] using hflags]
        rw [show [.scalar (.some (.bits w x)), .scalar (.some (.bits w y))] =
          ([.some (.bits w x), .some (.bits w y)] : List Val).map RVal.scalar from rfl,
          liftEager_scalars]
        simp only [Arith.intBin, ↓reduceDIte, Arith.arithWithFlags, List.contains_nil,
          Bool.false_and, Bool.false_eq_true, ↓reduceIte, Outcome.bind_ok, Arith.bv]
        exact ⟨_, rfl, List.Forall₂.cons ⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms],
          by simp [RVal.Defined]⟩ .nil⟩
    · cases h
  · cases h

/-- An initialized scalar float has an actual bit pattern, in every float interpretation. -/
theorem initialized_float {ib : Nat} {k : FKind} {v : RVal}
    (h : Fact.Holds ib v ⟨.scalar (.float k), true⟩) :
    ∃ x : BitVec k.width, v = .scalar (.some (.flt k x)) := by
  cases v with
  | tensor t => cases h.1
  | scalar x =>
    rcases hasTy_float h.1 with hp | ⟨a, ha⟩
    · exact (h.2 rfl hp).elim
    · exact ⟨a, congrArg RVal.scalar ha⟩

/-- Default same-format scalar arithmetic returns initialized bits under every float model.
This proves totality only: it does not assume any arithmetic identities or IEEE accuracy.
Fast-math promises are deliberately outside this definedness rule. -/
def floatBinaryRule : Rule := fun op fs =>
  match fs, op.results with
  | [⟨.scalar (.float k), true⟩, ⟨.scalar (.float k'), true⟩], [r] =>
    if dialectOf op.name = "arith" ∧ op.name ∈ fltBinNames ∧ k' = k ∧
        r.ty = .float k ∧ Arith.fastmathKey op = "0" then
      some [⟨.scalar (.float k), true⟩]
    else none
  | _, _ => none

theorem floatBinaryRule_sound (ib : Nat) : RuleSound ib floatBinaryRule := by
  intro op fs rs hm h ctx call args _ hargs
  unfold floatBinaryRule at h
  split at h
  · split at h
    · next hc =>
      obtain ⟨hd, hn, hk, hty, hfast⟩ := hc
      subst hk
      cases h
      obtain _ | ⟨ha, hb⟩ := hargs
      obtain _ | ⟨hb, hnil⟩ := hb
      cases hnil
      obtain ⟨x, rfl⟩ := initialized_float ha
      obtain ⟨y, rfl⟩ := initialized_float hb
      rename_i k r hr
      have hresult : Arith.resultElt op.blind = some (.float k) := by
        simp only [Arith.resultElt, Op.blind_results, hr, List.map_cons, List.map_nil, hty]
      rw [opSem_modelled hm]
      simp only [dispatch, Op.blind_name, hd]
      simp only [fltBinNames, List.mem_cons, List.not_mem_nil, or_false] at hn
      rcases hn with hn | hn | hn | hn | hn | hn | hn | hn | hn
      all_goals
        simp only [Arith.sem, Op.blind_name, hn, Arith.binOp, Arith.pureN, hresult,
          Outcome.bindOp_eq]
        rw [show [.scalar (.some (.flt k x)), .scalar (.some (.flt k y))] =
          ([.some (.flt k x), .some (.flt k y)] : List Val).map RVal.scalar from rfl,
          liftEager_scalars]
        simp only [Arith.fltBin, ↓reduceDIte, Outcome.bind_ok]
        exact ⟨_, rfl, List.Forall₂.cons ⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms],
          by simp [RVal.Defined]⟩ .nil⟩
    · cases h
  · cases h

/-- Static reassociation preserves initialization because every result element reads an in-bounds
source element. Reuse the existing reassociation validator and size/bounds proofs. -/
def reshapeRule : Rule := fun op fs =>
  if dialectOf op.name = "tensor" ∧
      (op.name = "tensor.collapse_shape" ∨ op.name = "tensor.expand_shape") then
    match fs, Sem.Ops.Tensor.resultTensorTy op, Sem.Ops.Tensor.reassociation? op with
    | [⟨.tensor sh e, true⟩], some (shape, elt), some groups =>
      if e = elt ∧ reshapeValid op groups sh shape = true then some [⟨.tensor shape elt, true⟩]
      else none
    | _, _, _ => none
  else none

theorem reshapeRule_sound (ib : Nat) : RuleSound ib reshapeRule := by
  intro op fs rs hm h ctx call args _ hargs
  unfold reshapeRule at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · split at h
      · next hc =>
        obtain ⟨he, hvalid⟩ := hc
        subst he
        cases h
        obtain _ | ⟨hv, hnil⟩ := hargs
        cases hnil
        rename_i a
        cases a with
        | scalar _ => cases hv.1
        | tensor t =>
          obtain ⟨hts, hte, htg⟩ := hv.1
          have hdefined := hv.2 rfl
          subst hts hte
          rename_i shape groups hr hg
          have hsize := reshapeValid_size hvalid
          rw [tensor_sem_of hm hd]
          have hsem : Sem.Ops.Tensor.sem ctx (restrictCall (calleeOf op.blind) call) op.blind
              [.tensor t] [] =
              .ok [.tensor (MerlinVerify.Tensor.canon shape t.elt fun i =>
                t.get (Sem.Ops.Tensor.delinearize t.shape (linearIndex shape i)))] := by
            unfold reshapeValid at hvalid
            rcases hn with hn | hn <;>
              simp_all [Sem.Ops.Tensor.sem, Op.blind_name, resultTensorTy_blind, reassociation?_blind]
          refine ⟨_, hsem, List.Forall₂.cons ⟨⟨rfl, rfl, ?_⟩, ?_⟩ .nil⟩
          · intro i hi
            simp only [MerlinVerify.Tensor.canon, hi, ↓reduceIte]
            exact htg _ (delinearize_inBounds _ _ (hsize ▸ linearIndex_lt _ _ hi))
          · intro _ i hi
            change Idx.InBounds shape i at hi
            simp only [MerlinVerify.Tensor.canon, hi, ↓reduceIte]
            exact hdefined _ (delinearize_inBounds _ _ (hsize ▸ linearIndex_lt _ _ hi))
      · cases h
    · cases h
  · cases h

theorem listMapM_total {α β : Type} (f : α → Outcome β) :
    ∀ (l : List α), (∀ x ∈ l, ∃ y, f x = .ok y) → ∃ ys : List β, l.mapM f = .ok ys
  | [], _ => ⟨[], rfl⟩
  | x :: xs, h => by
    obtain ⟨y, hy⟩ := h x (List.mem_cons_self ..)
    obtain ⟨ys, hys⟩ := listMapM_total f xs (fun x hx => h x (List.mem_cons_of_mem _ hx))
    exact ⟨y :: ys, by simp only [List.mapM_cons, Outcome.bindOp_eq, hy, hys,
      Outcome.bind_ok, Outcome.pure_eq]⟩

/-- Exact byte counts and a supported element decoder imply successful, fully initialized dense
constants. The array lookup fallback cannot occur at an in-bounds tensor index. -/
theorem denseTensor_defined {ib : Nat} {ty : Ty} {splat : Bool} {bytes : List UInt8}
    {sh : List Nat} {elt : Ty} (h : denseOk ty splat bytes sh elt = true) :
    ∃ t, denseTensor ib ty splat bytes = .ok t ∧
      Fact.Holds ib (.tensor t) ⟨.tensor sh elt, true⟩ := by
  have hsafety := denseTensor_safe (ib := ib) h
  have htotal : ∃ t, denseTensor ib ty splat bytes = .ok t ∧ (RVal.tensor t).Defined := by
    unfold denseOk at h
    split at h
    · next sh' elt' eb hst heb =>
      simp only [decide_eq_true_eq] at h
      obtain ⟨rfl, rfl, heb0, hlen⟩ := h
      unfold denseTensor
      simp only [hst, heb, heb0, ↓reduceIte, Outcome.bindOp_eq, Outcome.pure_eq]
      cases splat with
      | true =>
        obtain ⟨s, hs, _⟩ := scalarOfBits_conforms (ib := ib) heb (leNat bytes)
        simp only [↓reduceIte] at hlen ⊢
        simp only [hlen, ne_eq, not_true_eq_false, ↓reduceIte, hs]
        refine ⟨_, rfl, fun i hi => ?_⟩
        change Idx.InBounds sh i at hi
        simp [MerlinVerify.Tensor.full, MerlinVerify.Tensor.canon, hi]
      | false =>
        simp only [Bool.false_eq_true, ↓reduceIte] at hlen ⊢
        simp only [hlen, ne_eq, not_true_eq_false, ↓reduceIte]
        let decode := fun k => Outcome.ofOption "dense element decode"
          (scalarOfBits ib elt (leNat ((bytes.toArray.extract (k * eb) (k * eb + eb)).toList)))
        obtain ⟨elems, hm⟩ := listMapM_total decode (List.range (shapeSize sh)) (fun k _ => by
          obtain ⟨s, hs, _⟩ := scalarOfBits_conforms (ib := ib) heb
            (leNat ((bytes.toArray.extract (k * eb) (k * eb + eb)).toList))
          exact ⟨s, by simp only [decode, hs, Outcome.ofOption]⟩)
        change ∃ t, ((List.range (shapeSize sh)).mapM decode).bind _ = .ok t ∧ _
        rw [hm, Outcome.bind_ok]
        refine ⟨_, rfl, fun i hi => ?_⟩
        change Idx.InBounds sh i at hi
        have hk : linearIndex sh i < elems.length := by
          rw [listMapM_length decode _ _ hm, List.length_range]
          exact linearIndex_lt _ _ hi
        simp [MerlinVerify.Tensor.canon, hi, List.getElem?_toArray, List.getElem?_eq_getElem hk]
    · cases h
  obtain ⟨t, ht, hd⟩ := htotal
  exact ⟨t, ht, hsafety.2 t ht, fun _ => hd⟩

def denseConstantRule : Rule := fun op fs =>
  if dialectOf op.name = "arith" ∧ op.name = "arith.constant" ∧ fs = [] then
    match op.results, op.attr? "value" with
    | [r], some (.dense ty splat bytes) =>
      match r.ty.staticTensor? with
      | some (sh, e) => if denseOk ty splat bytes sh e then some [⟨.tensor sh e, true⟩] else none
      | none => none
    | _, _ => none
  else none

theorem denseConstantRule_sound (ib : Nat) : RuleSound ib denseConstantRule := by
  intro op fs rs hm h ctx call args hib hargs
  unfold denseConstantRule at h
  split at h
  · next hc =>
    obtain ⟨hd, hn, rfl⟩ := hc
    cases hargs
    split at h
    · split at h
      · split at h
        · next hok =>
          cases h
          obtain ⟨t, ht, hf⟩ := denseTensor_defined (ib := ib) hok
          refine ⟨[.tensor t], ?_, List.Forall₂.cons hf .nil⟩
          rw [opSem_modelled hm]
          simp only [dispatch, Op.blind_name, hd]
          rename_i r ty splat bytes hr ha _ sh e hs
          simp only [Arith.sem, Op.blind_name, hn, Op.blind_attr?, ha, Op.blind_results, hr,
            List.map_cons, List.map_nil, hib, ht, Outcome.bindOp_eq, Outcome.bind_ok, Outcome.pure_eq]
        · cases h
      · cases h
    · cases h
  · cases h

/-- Width-changing integer casts are total only in their legal direction and without
poison-producing overflow/nonnegative promises. -/
def intCastRule : Rule := fun op fs =>
  match fs, op.results with
  | [⟨.scalar (.int w), true⟩], [r] =>
    match r.ty with
    | .int w' =>
      if dialectOf op.name = "arith" ∧
          ((op.name = "arith.extsi" ∧ w < w') ∨ (op.name = "arith.extui" ∧ w < w') ∨
            (op.name = "arith.trunci" ∧ w' < w)) ∧
          overflowFlags op = [] ∧ hasUnit op "nonNeg" = false then
        some [⟨.scalar (.int w'), true⟩]
      else none
    | _ => none
  | _, _ => none

theorem intCastRule_sound (ib : Nat) : RuleSound ib intCastRule := by
  intro op fs rs hm h ctx call args _ hargs
  unfold intCastRule at h
  split at h
  · split at h
    · split at h
      · next hc =>
        obtain ⟨hd, hn, hflags, hnneg⟩ := hc
        cases h
        obtain _ | ⟨hv, hnil⟩ := hargs
        cases hnil
        obtain ⟨x, rfl⟩ := initialized_int hv
        rename_i w r hr _ w' hty
        have hresult : Arith.resultElt op.blind = some (.int w') := by
          simp only [Arith.resultElt, Op.blind_results, hr, List.map_cons, List.map_nil, hty]
        have hf : overflowFlags op.blind = [] := by simpa [overflowFlags, attrEnum?] using hflags
        have hn' : hasUnit op.blind "nonNeg" = false := by simpa [hasUnit] using hnneg
        rw [opSem_modelled hm]
        simp only [dispatch, Op.blind_name, hd]
        rcases hn with ⟨hn, hw⟩ | ⟨hn, hw⟩ | ⟨hn, hw⟩
        all_goals
          simp only [Arith.sem, Op.blind_name, hn, Arith.widthOf, hresult, Ty.bitWidth?,
            Outcome.ofOption, Outcome.bindOp_eq, Outcome.bind_ok, Arith.unOp, Arith.pureN, hf, hn']
          rw [show [.scalar (.some (.bits w x))] = ([.some (.bits w x)] : List Val).map RVal.scalar from rfl,
            liftEager_scalars]
          simp only [Arith.intUn, List.contains_nil,
            Bool.false_and, Bool.false_eq_true, ↓reduceIte, ite_eq_right (Nat.not_le_of_lt hw),
            Outcome.bind_ok, Arith.bv]
          exact ⟨_, rfl, List.Forall₂.cons ⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms],
            by simp [RVal.Defined]⟩ .nil⟩
      · cases h
    · cases h
  · cases h

/-- Rank-zero extraction has no index operands and its sole element is always in bounds.
Positive-rank extraction needs a separate value-dependent bounds proof. -/
def scalarExtractRule : Rule := fun op fs =>
  match fs, op.results with
  | [⟨.tensor [] e, true⟩], [r] =>
    if dialectOf op.name = "tensor" ∧ op.name = "tensor.extract" ∧ r.ty = e then
      some [⟨.scalar e, true⟩]
    else none
  | _, _ => none

theorem scalarExtractRule_sound (ib : Nat) : RuleSound ib scalarExtractRule := by
  intro op fs rs hm h ctx call args _ ha
  unfold scalarExtractRule at h
  split at h
  · split at h
    · next hc =>
      obtain ⟨hd, hn, _⟩ := hc
      cases h
      obtain _ | ⟨ht, hnil⟩ := ha
      cases hnil
      rename_i t
      cases t with
      | scalar _ => cases ht.1
      | tensor t =>
        have hi : Idx.InBounds t.shape [] := by simp [ht.1.1, Idx.InBounds]
        refine ⟨[.scalar (t.get [])], ?_, List.Forall₂.cons ⟨ht.1.2.2 [] (by simp [Idx.InBounds]), ?_⟩ .nil⟩
        · rw [tensor_sem_of hm hd]
          simp [Sem.Ops.Tensor.sem, Op.blind_name, hn, ht.1.1]
        · intro _
          exact ht.2 rfl [] hi
    · cases h
  · cases h

def rules (ib : Nat) : Rule := fun op fs =>
  [constantRule ib, emptyRule, splatRule, intBinaryRule, floatBinaryRule, reshapeRule,
    denseConstantRule, intCastRule, scalarExtractRule].findSome? fun r => r op fs

theorem rules_sound (ib : Nat) : RuleSound ib (rules ib) := by
  intro op fs rs hm hr
  obtain ⟨r, hmem, h⟩ := List.exists_of_findSome?_eq_some hr
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem
  rcases hmem with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact constantRule_sound ib op fs rs hm h
  · exact emptyRule_sound ib op fs rs hm h
  · exact splatRule_sound ib op fs rs hm h
  · exact intBinaryRule_sound ib op fs rs hm h
  · exact floatBinaryRule_sound ib op fs rs hm h
  · exact reshapeRule_sound ib op fs rs hm h
  · exact denseConstantRule_sound ib op fs rs hm h
  · exact intCastRule_sound ib op fs rs hm h
  · exact scalarExtractRule_sound ib op fs rs hm h

def leafBook (ib : Nat) : Book opSem ib where
  rule _ op fs := if unmodelledAttr? op.blind = none ∧ op.regions = [] then rules ib op fs else none
  sound Γ op fs rs h ctx call env args hib _ ha := by
    split at h
    · next hc =>
      obtain ⟨hm, hregs⟩ := hc
      simpa only [hregs, evalRegions] using rules_sound ib op fs rs hm h ctx call args hib ha
    · cases h

/-- Acceptance proves successful, nonpoison results for every typed, nonpoison input. -/
theorem leafCheck_sound {ib : Nat} {m : Module} {f : String} {argTys resTys : List VTy}
    (h : check (leafBook ib) m f argTys resTys = true) :
    FuncDefined opSem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys ∧ ResultsDefined args) m f := check_sound (leafBook ib) h

end MerlinVerify.Check.Definedness
