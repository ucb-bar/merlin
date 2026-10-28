/-
Typing rules and safety proofs for the `tensor` ops (design §4.5, progress).

`tensor.empty` builds a tensor of poison. `tensor.collapse_shape`/`expand_shape` read the source
in row-major order: result element `i` is source element `delinearize src (linearIndex res i)`.
A valid reassociation preserves the element count (`reassocValid_size`), so every such read is
inside the source (`linearIndex_lt`, `delinearize_inBounds`): a reshape never reads past its
operand.
-/
import MerlinVerify.Check.RulesLinalg

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

/-! ### Row-major index arithmetic -/

theorem foldl_mul (l : List Nat) (a : Nat) : l.foldl (· * ·) a = a * l.foldl (· * ·) 1 := by
  induction l generalizing a with
  | nil => simp
  | cons x xs ih =>
    simp only [List.foldl_cons]
    rw [ih (a * x), ih (1 * x), Nat.one_mul, Nat.mul_assoc]

theorem shapeSize_cons (d : Nat) (ds : List Nat) : shapeSize (d :: ds) = d * shapeSize ds := by
  unfold shapeSize
  rw [List.foldl_cons, foldl_mul, Nat.one_mul]

theorem shapeSize_append (a b : List Nat) : shapeSize (a ++ b) = shapeSize a * shapeSize b := by
  unfold shapeSize
  rw [List.foldl_append, foldl_mul]

theorem linearIndex_cons (d : Nat) (ds : List Nat) (x : Nat) (xs : Idx) :
    linearIndex (d :: ds) (x :: xs) = x * shapeSize ds + linearIndex ds xs := rfl

/-- An in-bounds index has a row-major position below the element count. -/
theorem linearIndex_lt : ∀ (s : List Nat) (i : Idx), Idx.InBounds s i → linearIndex s i < shapeSize s
  | [], i, h => by
    rw [(Idx.inBounds_nil_iff i).1 h]
    decide
  | d :: ds, [], h => absurd h (Idx.inBounds_cons_nil d ds)
  | d :: ds, x :: xs, h => by
    obtain ⟨hx, hxs⟩ := (Idx.inBounds_cons_cons d ds x xs).1 h
    have ih := linearIndex_lt ds xs hxs
    rw [linearIndex_cons, shapeSize_cons]
    calc x * shapeSize ds + linearIndex ds xs < x * shapeSize ds + shapeSize ds := by omega
      _ = (x + 1) * shapeSize ds := (Nat.succ_mul x _).symm
      _ ≤ d * shapeSize ds := Nat.mul_le_mul_right _ hx

/-- A row-major position below the element count is the position of an in-bounds index. -/
theorem delinearize_inBounds : ∀ (s : List Nat) (k : Nat), k < shapeSize s →
    Idx.InBounds s (Sem.Ops.Tensor.delinearize s k)
  | [], _, _ => (Idx.inBounds_nil_iff _).2 rfl
  | d :: ds, k, h => by
    rw [shapeSize_cons] at h
    have hS : 0 < shapeSize ds := Nat.pos_of_ne_zero fun h0 => by rw [h0] at h; simp at h
    simp only [Sem.Ops.Tensor.delinearize]
    refine (Idx.inBounds_cons_cons _ _ _ _).2 ⟨?_, delinearize_inBounds ds _ (Nat.mod_lt _ hS)⟩
    exact Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)

theorem map_getD_range (l : List Nat) : (List.range l.length).map (fun i => l.getD i 0) = l := by
  apply List.ext_getElem (by simp)
  intro k _ h₂
  simp [List.getElem?_eq_getElem h₂]

/-- Groups whose element counts are `hi` multiply out to the element count of their union. -/
theorem size_of_groups (L : Nat → Nat) : ∀ (gs : List (List Nat)) (hi : List Nat),
    gs.length = hi.length →
    ((gs.zip hi).all fun (g, h) => shapeSize (g.map L) == h) = true →
    shapeSize hi = shapeSize (gs.flatten.map L)
  | [], [], _, _ => rfl
  | g :: gs, h :: hs, hl, hall => by
    simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true, beq_iff_eq] at hall
    rw [List.flatten_cons, List.map_append, shapeSize_append, shapeSize_cons, ← hall.1,
      size_of_groups L gs hs (by simpa using hl) hall.2]
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl

/-- A valid reassociation preserves the element count. -/
theorem reassocValid_size {groups : List (List Nat)} {lo hi : List Nat}
    (h : Sem.Ops.Tensor.reassocValid groups lo hi = true) : shapeSize hi = shapeSize lo := by
  simp only [Sem.Ops.Tensor.reassocValid, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨hlen, hflat⟩, hall⟩ := h
  rw [size_of_groups (fun i => lo.getD i 0) groups hi hlen hall, hflat, map_getD_range]

/-! ### Rules -/

/-- `tensor.empty`: no operands, a static result tensor. -/
def emptyRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.empty" ∧ argTys = [] then
    match Sem.Ops.Tensor.resultTensorTy op with
    | some (sh, e) => if resTys = [.tensor sh e] then some [] else none
    | none => none
  else none

/-- Reduce a `tensor` op's safety to `Tensor.sem` on the blinded op. -/
theorem tensor_sem_of {ctx : Ctx} {call : CallSem} {op : Op} {args : List RVal}
    {regions : List (Ctx → RegionSem)} (hm : unmodelledAttr? op.blind = none)
    (hd : dialectOf op.name = "tensor") :
    opSem ctx call op args regions =
      Sem.Ops.Tensor.sem ctx (restrictCall (calleeOf op.blind) call) op.blind args regions := by
  rw [opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]

theorem resultTensorTy_blind (op : Op) :
    Sem.Ops.Tensor.resultTensorTy op.blind = Sem.Ops.Tensor.resultTensorTy op := by
  unfold Sem.Ops.Tensor.resultTensorTy
  rw [Op.blind_results]
  rcases op.results with _ | ⟨r, _ | _⟩ <;> rfl

theorem reassociation?_blind (op : Op) :
    Sem.Ops.Tensor.reassociation? op.blind = Sem.Ops.Tensor.reassociation? op := by
  simp only [Sem.Ops.Tensor.reassociation?, Op.blind_attr?]

theorem emptyRule_sound (ib : Nat) : RuleSound ib emptyRule := by
  intro d op argTys resTys sigs hm h
  simp only [emptyRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn, rfl⟩ := hc
    split at h
    · next sh e hr =>
      split at h
      · next hres =>
        subst hres
        cases h
        intro ctx call args regions _ _ hargs _
        cases hargs
        rw [tensor_sem_of hm hd]
        simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, hr]
        exact Safe.of_ok (List.Forall₂.cons (hasTy_full (hasTy_poison _ _)) .nil)
      · cases h
    · cases h
  · cases h

/-- The reassociation check of a reshape from `src` to `res`: the groups index the larger rank. -/
def reshapeValid (op : Op) (groups : List (List Nat)) (src res : List Nat) : Bool :=
  if op.name == "tensor.collapse_shape" then Sem.Ops.Tensor.reassocValid groups src res
  else Sem.Ops.Tensor.reassocValid groups res src

theorem reshapeValid_size {op : Op} {groups : List (List Nat)} {src res : List Nat}
    (h : reshapeValid op groups src res = true) : shapeSize res = shapeSize src := by
  unfold reshapeValid at h
  split at h
  · exact reassocValid_size h
  · exact (reassocValid_size h).symm

/-- `tensor.collapse_shape`/`expand_shape`: one tensor operand whose element type is the result's,
with a reassociation consistent with both shapes. -/
def reshapeRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "tensor" ∧
      (op.name = "tensor.collapse_shape" ∨ op.name = "tensor.expand_shape") then
    match Sem.Ops.Tensor.resultTensorTy op, Sem.Ops.Tensor.reassociation? op, argTys with
    | some (shape, elt), some groups, [.tensor sh e] =>
      if resTys = [.tensor shape elt] ∧ e = elt ∧ reshapeValid op groups sh shape = true then some []
      else none
    | _, _, _ => none
  else none

theorem reshapeRule_sound (ib : Nat) : RuleSound ib reshapeRule := by
  intro d op argTys resTys sigs hm h
  simp only [reshapeRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next shape elt groups sh e hr hg =>
      split at h
      · next hc2 =>
        obtain ⟨rfl, rfl, hvalid⟩ := hc2
        cases h
        intro ctx call args regions _ _ hargs _
        obtain _ | ⟨ht, hnil⟩ := hargs
        cases hnil
        rename_i a
        cases a with
        | scalar _ => exact (ht : False).elim
        | tensor t =>
        obtain ⟨hts, hte, htg⟩ := ht
        subst hts hte
        -- both reshapes preserve the element count
        have hsize := reshapeValid_size hvalid
        rw [tensor_sem_of hm hd]
        have hsem : Sem.Ops.Tensor.sem ctx (restrictCall (calleeOf op.blind) call) op.blind
            [.tensor t] regions =
            .ok [.tensor (Tensor.canon shape t.elt fun i =>
              t.get (Sem.Ops.Tensor.delinearize t.shape (linearIndex shape i)))] := by
          unfold reshapeValid at hvalid
          rcases hn with hn | hn <;>
            simp_all [Sem.Ops.Tensor.sem, Op.blind_name, resultTensorTy_blind, reassociation?_blind]
        rw [hsem]
        refine Safe.of_ok (List.Forall₂.cons ⟨rfl, rfl, fun i hi => ?_⟩ .nil)
        simp only [Tensor.canon, hi, ↓reduceIte]
        exact htg _ (delinearize_inBounds _ _ (hsize ▸ linearIndex_lt _ _ hi))
      · cases h
    · cases h
  · cases h

/-! ### Slices -/

/-- The coordinates of `a` at the non-unit entries of `b` lie below those entries. -/
theorem nonUnit_coords_lt (g : Nat × Nat → Option Nat) (hg : ∀ x s, g (x, s) = if s = 1 then none else some x) :
    ∀ (a b : List Nat), a.length = b.length →
      (∀ d (h₁ : d < a.length) (h₂ : d < b.length), a[d] < b[d]) →
      List.Forall₂ (· < ·) ((a.zip b).filterMap g) (b.filter (· ≠ 1))
  | [], [], _, _ => by simp
  | x :: xs, s :: ss, hl, hlt => by
    have h0 : x < s := hlt 0 (by simp) (by simp)
    have ih := nonUnit_coords_lt g hg xs ss (by simpa using hl)
      (fun d h₁ h₂ => hlt (d + 1) (by simpa using h₁) (by simpa using h₂))
    rw [List.zip_cons_cons, List.filterMap_cons, hg, List.filter_cons]
    by_cases hs : s = 1
    · simp only [hs, ↓reduceIte, ne_eq, not_true_eq_false, decide_false, Bool.false_eq_true]
      exact ih
    · simp only [hs, ↓reduceIte, ne_eq, not_false_eq_true, decide_true]
      exact List.Forall₂.cons h0 ih
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl

/-- Placing coordinates below the non-unit sizes gives an in-bounds index. -/
theorem placeNonUnit_inBounds : ∀ (sizes cs : List Nat) (f : Idx),
    Sem.Ops.Tensor.placeNonUnit sizes cs = some f → List.Forall₂ (· < ·) cs (sizes.filter (· ≠ 1)) →
      Idx.InBounds sizes f
  | [], [], f, h, _ => by
    simp only [Sem.Ops.Tensor.placeNonUnit, Option.some.injEq] at h
    subst h
    exact (Idx.inBounds_nil_iff []).2 rfl
  | [], _ :: _, _, h, _ => by simp [Sem.Ops.Tensor.placeNonUnit] at h
  | s :: ss, cs, f, h, hlt => by
    simp only [Sem.Ops.Tensor.placeNonUnit] at h
    by_cases hs : s = 1
    · simp only [hs, ↓reduceIte] at h
      have hlt' : List.Forall₂ (· < ·) cs (ss.filter (· ≠ 1)) := by
        simpa [List.filter_cons, hs] using hlt
      cases hp : Sem.Ops.Tensor.placeNonUnit ss cs with
      | none => simp [hp] at h
      | some f' =>
        simp only [hp, Option.map_eq_map, Option.map_some, Option.some.injEq] at h
        subst h
        exact (Idx.inBounds_cons_cons _ _ _ _).2 ⟨by omega, placeNonUnit_inBounds ss cs f' hp hlt'⟩
    · simp only [hs, ↓reduceIte] at h
      cases cs with
      | nil => cases h
      | cons c cs' =>
        have hlt' : List.Forall₂ (· < ·) (c :: cs') (s :: ss.filter (· ≠ 1)) := by
          simpa [List.filter_cons, hs] using hlt
        obtain _ | ⟨hc, hrest⟩ := hlt'
        cases hp : Sem.Ops.Tensor.placeNonUnit ss cs' with
        | none => simp [hp] at h
        | some f' =>
          simp only [hp, Option.map_eq_map, Option.map_some, Option.some.injEq] at h
          subst h
          exact (Idx.inBounds_cons_cons _ _ _ _).2 ⟨hc, placeNonUnit_inBounds ss cs' f' hp hrest⟩

/-- A rank-reduced index of a compatible slice expands to an index inside the slice sizes. -/
theorem expandReduced_inBounds {sizes reduced : List Nat} {r f : Idx}
    (hc : Sem.Ops.Tensor.reducedCompatible sizes reduced = true) (hr : Idx.InBounds reduced r)
    (h : Sem.Ops.Tensor.expandReduced sizes reduced r = some f) : Idx.InBounds sizes f := by
  unfold Sem.Ops.Tensor.expandReduced at h
  unfold Sem.Ops.Tensor.reducedCompatible at hc
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
  refine placeNonUnit_inBounds sizes _ f h ?_
  rw [hc.1]
  exact nonUnit_coords_lt _ (by intro _ _; rfl) r reduced hr.1 hr.2

theorem getD_eq_getElem {l : List Nat} {d : Nat} (h : d < l.length) : l.getD d 0 = l[d] := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]

/-- A slice that fits its source: every index inside the slice sizes reads inside the source. -/
theorem slice_index_inBounds {shape offs sizes strides : List Nat} {f : Idx}
    (hsz : sizes.length = shape.length) (hf : Idx.InBounds sizes f)
    (hfit : ((List.range shape.length).any fun d =>
      sizes.getD d 0 ≠ 0 && offs.getD d 0 + (sizes.getD d 0 - 1) * strides.getD d 0 ≥ shape.getD d 0) =
        false) :
    Idx.InBounds shape
      ((List.range shape.length).map fun d => offs.getD d 0 + f.getD d 0 * strides.getD d 0) := by
  refine ⟨by simp, fun d h₁ h₂ => ?_⟩
  have hd : d < shape.length := h₂
  have hfd : d < f.length := by rw [hf.1, hsz]; exact hd
  have hsd : d < sizes.length := by rw [hsz]; exact hd
  have hlt := hf.2 d hfd hsd
  have hno := List.any_eq_false.1 hfit d (List.mem_range.2 hd)
  simp only [getD_eq_getElem hsd, getD_eq_getElem hd, Bool.and_eq_true, decide_eq_true_eq, not_and,
    ne_eq] at hno
  have hfit' := hno (by omega)
  simp only [List.getElem_map, List.getElem_range, getD_eq_getElem hfd]
  have : f[d] * strides.getD d 0 ≤ (sizes[d] - 1) * strides.getD d 0 :=
    Nat.mul_le_mul_right _ (by omega)
  omega

theorem staticArray?_blind (op : Op) (k : String) :
    Sem.Ops.Tensor.staticArray? op.blind k = Sem.Ops.Tensor.staticArray? op k := by
  simp only [Sem.Ops.Tensor.staticArray?, attrDenseArray?, Op.blind_attr?]

/-- `tensor.extract_slice`: a static slice of the source that fits it, with a (possibly rank-reduced)
result shape compatible with the sizes. -/
def extractSliceRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.extract_slice" then
    match Sem.Ops.Tensor.resultTensorTy op, Sem.Ops.Tensor.staticArray? op "static_offsets",
        Sem.Ops.Tensor.staticArray? op "static_sizes", Sem.Ops.Tensor.staticArray? op "static_strides",
        argTys with
    | some (rshape, elt), some offs, some sizes, some strides, [.tensor sh e] =>
      if resTys = [.tensor rshape elt] ∧ e = elt ∧
          ¬(offs.length ≠ sh.length ∨ sizes.length ≠ sh.length ∨ strides.length ≠ sh.length) ∧
          Sem.Ops.Tensor.reducedCompatible sizes rshape = true ∧
          ((List.range sh.length).any fun d => sizes.getD d 0 ≠ 0 &&
            offs.getD d 0 + (sizes.getD d 0 - 1) * strides.getD d 0 ≥ sh.getD d 0) = false then
        some []
      else none
    | _, _, _, _, _ => none
  else none

theorem extractSliceRule_sound (ib : Nat) : RuleSound ib extractSliceRule := by
  intro d op argTys resTys sigs hm h
  simp only [extractSliceRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next rshape elt offs sizes strides sh e hr hoffs hsizes hstrides =>
      split at h
      · next hc2 =>
        obtain ⟨rfl, rfl, hlen, hcomp, hfit⟩ := hc2
        cases h
        intro ctx call args regions _ _ hargs _
        obtain _ | ⟨ht, hnil⟩ := hargs
        cases hnil
        rename_i a
        cases a with
        | scalar _ => exact (ht : False).elim
        | tensor t =>
        obtain ⟨hts, hte, htg⟩ := ht
        subst hts hte
        rw [tensor_sem_of hm hd]
        simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, hr,
          staticArray?_blind, hoffs, hsizes, hstrides, hlen, ↓reduceIte, hcomp, Bool.not_true,
          Bool.false_eq_true, hfit]
        refine Safe.of_ok (List.Forall₂.cons ⟨rfl, rfl, fun r hr' => ?_⟩ .nil)
        simp only [Tensor.canon, hr', ↓reduceIte]
        split
        · next f hf =>
          exact htg _ (slice_index_inBounds (by omega) (expandReduced_inBounds hcomp hr' hf) hfit)
        · exact hasTy_poison _ _
      · cases h
    · cases h
  · cases h

theorem optionMapM_spec {α β : Type} (g : α → Option β) : ∀ (l : List α) (f : List β),
    l.mapM g = some f → f.length = l.length ∧
      ∀ k (h₁ : k < l.length) (h₂ : k < f.length), g l[k] = some f[k]
  | [], f, h => by
    simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
    subst h
    exact ⟨rfl, fun k h₁ => absurd h₁ (Nat.not_lt_zero _)⟩
  | a :: l, f, h => by
    simp only [List.mapM_cons] at h
    cases ha : g a with
    | none => simp [ha] at h
    | some b =>
      cases hl : l.mapM g with
      | none => simp [ha, hl] at h
      | some bs =>
        simp only [ha, hl, Option.pure_def, bind, Option.bind, Option.some.injEq] at h
        subst h
        obtain ⟨hlen, hk⟩ := optionMapM_spec g l bs hl
        refine ⟨by simp [hlen], fun k h₁ h₂ => ?_⟩
        cases k with
        | zero => exact ha
        | succ k => exact hk k (by simpa using h₁) (by simpa using h₂)

/-- `tensor.insert_slice`: a static slice of the destination that fits it, filled from a source of
the same element type whose shape is compatible with the sizes. -/
def insertSliceRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.insert_slice" then
    match Sem.Ops.Tensor.staticArray? op "static_offsets", Sem.Ops.Tensor.staticArray? op "static_sizes",
        Sem.Ops.Tensor.staticArray? op "static_strides", argTys with
    | some offs, some sizes, some strides, [.tensor ssh e, .tensor dsh e'] =>
      if resTys = [.tensor dsh e'] ∧ e = e' ∧
          ¬(offs.length ≠ dsh.length ∨ sizes.length ≠ dsh.length ∨ strides.length ≠ dsh.length) ∧
          Sem.Ops.Tensor.reducedCompatible sizes ssh = true ∧
          ((List.range dsh.length).any fun d => sizes.getD d 0 ≠ 0 &&
            offs.getD d 0 + (sizes.getD d 0 - 1) * strides.getD d 0 ≥ dsh.getD d 0) = false then
        some []
      else none
    | _, _, _, _ => none
  else none

theorem insertSliceRule_sound (ib : Nat) : RuleSound ib insertSliceRule := by
  intro d op argTys resTys sigs hm h
  simp only [insertSliceRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next offs sizes strides ssh e dsh e' hoffs hsizes hstrides =>
      split at h
      · next hc2 =>
        obtain ⟨rfl, rfl, hlen, hcomp, hfit⟩ := hc2
        cases h
        intro ctx call args regions _ _ hargs _
        obtain _ | ⟨hs, _ | ⟨hdst, hnil⟩⟩ := hargs
        cases hnil
        rename_i a b
        cases a with
        | scalar _ => exact (hs : False).elim
        | tensor src =>
        cases b with
        | scalar _ => exact (hdst : False).elim
        | tensor dst =>
        obtain ⟨hss, hse, hsg⟩ := hs
        obtain ⟨hds, hde, hdg⟩ := hdst
        subst hss hse hds
        rw [tensor_sem_of hm hd]
        simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, staticArray?_blind, hoffs, hsizes, hstrides,
          hlen, ↓reduceIte, hcomp, Bool.not_true, Bool.false_eq_true, hfit]
        refine Safe.of_ok (List.Forall₂.cons ⟨rfl, hde, fun i hi => ?_⟩ .nil)
        simp only [Tensor.canon, hi, ↓reduceIte]
        split
        · next j hj =>
          simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hj
          obtain ⟨f, hf, hj⟩ := hj
          obtain ⟨hflen, hfk⟩ := optionMapM_spec _ _ f hf
          simp only [List.length_range] at hflen
          have hsz : sizes.length = dst.shape.length := by omega
          have hlt : ∀ k (h₁ : k < f.length) (h₂ : k < sizes.length), f[k] < sizes[k] := by
            intro k h₁ h₂
            have hk := hfk k (by simpa [hflen] using h₁) h₁
            simp only [List.getElem_range] at hk
            split at hk
            · next hcond =>
              simp only [Option.some.injEq] at hk
              rw [← hk, ← getD_eq_getElem h₂]
              exact hcond.2.2.2
            · cases hk
          have hj' := placeNonUnit_inBounds _ _ j hj (by
            have := hcomp
            simp only [Sem.Ops.Tensor.reducedCompatible, Bool.and_eq_true, beq_iff_eq,
              decide_eq_true_eq] at this
            rw [← this.1]
            exact nonUnit_coords_lt _ (by intro _ _; rfl) f sizes (by omega) hlt)
          exact hsg j hj'
        · exact hdg i hi
      · cases h
    · cases h
  · cases h

/-! ### Concatenation -/

/-- A part of a concatenation along `d`: the result's element type, and the result's extent in every
dimension but `d`. -/
def concatPartOk (shape : List Nat) (elt : Ty) (d : Nat) : VTy → Bool
  | .tensor sh e =>
    decide (e = elt ∧ sh.length = shape.length ∧ ∀ k < sh.length, k ≠ d → sh.getD k 0 = shape.getD k 0)
  | .scalar _ => false

/-- What a typed part of a concatenation provides. -/
def ConcatPart (ib : Nat) (shape : List Nat) (elt : Ty) (d : Nat) (p : Tensor) : Prop :=
  p.elt = elt ∧ p.shape.length = shape.length ∧
    (∀ k < p.shape.length, k ≠ d → p.shape.getD k 0 = shape.getD k 0) ∧
    ∀ j, Idx.InBounds p.shape j → Val.HasTy ib p.elt (p.get j)

/-- Every element of a concatenation reads inside the part it falls in. -/
theorem concatPick_typed {ib : Nat} {shape : List Nat} {elt : Ty} {d : Nat} {i : Idx}
    (hi : Idx.InBounds shape i) :
    ∀ (parts : List Tensor) (off : Nat), off ≤ i.getD d 0 →
      (∀ p ∈ parts, ConcatPart ib shape elt d p) →
      Val.HasTy ib elt (Sem.Ops.Tensor.concatPick d i parts off)
  | [], _, _, _ => hasTy_poison _ _
  | p :: ps, off, hoff, hall => by
    obtain ⟨hpe, hpl, hpk, hpg⟩ := hall p (List.mem_cons_self ..)
    simp only [Sem.Ops.Tensor.concatPick]
    split
    · next hx =>
      rw [← hpe]
      apply hpg
      refine ⟨by rw [List.length_set, hi.1, hpl], fun k h₁ h₂ => ?_⟩
      rw [List.getElem_set]
      split
      · next hkd =>
        subst hkd
        have hdi : d < i.length := by simpa using h₁
        rw [getD_eq_getElem hdi] at hx hoff ⊢
        rw [getD_eq_getElem h₂] at hx
        omega
      · next hkd =>
        have hki : k < i.length := by simpa using h₁
        have hks : k < shape.length := by rw [← hpl]; exact h₂
        have hsame := hpk k h₂ (Ne.symm hkd)
        rw [getD_eq_getElem h₂, getD_eq_getElem hks] at hsame
        rw [hsame]
        exact hi.2 k hki hks
    · next hx =>
      exact concatPick_typed hi ps _ (by omega) (fun q hq => hall q (List.mem_cons_of_mem _ hq))

theorem concat_parts {ib : Nat} {shape : List Nat} {elt : Ty} {d : Nat} :
    ∀ {ts : List Tensor} {tys : List VTy}, HasTys ib (ts.map RVal.tensor) tys →
      tys.all (concatPartOk shape elt d) = true → ∀ p ∈ ts, ConcatPart ib shape elt d p
  | [], [], _, _ => by simp
  | t :: _, ty :: _, .cons ht hrest, hall => by
    simp only [List.all_cons, Bool.and_eq_true] at hall
    intro p hp
    rcases List.mem_cons.1 hp with rfl | hp
    · cases ty with
      | scalar _ => exact (ht : False).elim
      | tensor sh e =>
        obtain ⟨hs, he, hg⟩ := ht
        simp only [concatPartOk, decide_eq_true_eq] at hall
        obtain ⟨⟨rfl, hl, hdims⟩, -⟩ := hall
        subst hs he
        exact ⟨rfl, hl, hdims, hg⟩
    · exact concat_parts hrest hall.2 p hp
  | [], _ :: _, h, _ => by cases h
  | _ :: _, [], h, _ => by cases h

/-- `tensor.concat` along `dim`: parts of the result's element type and extents except along `dim`,
whose extents there sum to the result's. -/
def concatRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.concat" then
    match Sem.Ops.Tensor.resultTensorTy op, attrInt? op "dim" with
    | some (shape, elt), some dd =>
      if resTys = [.tensor shape elt] ∧ argTys ≠ [] ∧
          argTys.all (concatPartOk shape elt dd.toNat) = true ∧
          (argTys.map VTy.shape).foldl (fun acc s => acc + s.getD dd.toNat 0) 0 =
            shape.getD dd.toNat 0 then
        some []
      else none
    | _, _ => none
  else none

theorem concatRule_sound (ib : Nat) : RuleSound ib concatRule := by
  intro d op argTys resTys sigs hm h
  simp only [concatRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next shape elt dd hr hdim =>
      split at h
      · next hc2 =>
        obtain ⟨rfl, hne, hparts, hsum⟩ := hc2
        cases h
        intro ctx call args regions _ _ hargs _
        have hpartsT : ∀ t ∈ argTys, t.isTensor = true := by
          intro t ht
          have := List.all_eq_true.1 hparts t ht
          cases t <;> simp_all [concatPartOk, VTy.isTensor]
        obtain ⟨ts, rfl⟩ := tensors_of_typed hargs hpartsT
        have hshapes : ts.map Tensor.shape = argTys.map VTy.shape := by
          rw [← shapes_typed hargs, List.map_map]; rfl
        have hpart := concat_parts hargs hparts
        cases ts with
        | nil => exact absurd (by simpa using (HasTys.length hargs).symm) hne
        | cons t ts =>
        have hrank : ((t :: ts).any fun p => p.shape.length ≠ shape.length) = false := by
          rw [List.any_eq_false]
          intro p hp
          simp [(hpart p hp).2.1]
        have hext : (t :: ts).foldl (fun acc p => acc + p.shape.getD dd.toNat 0) 0 =
            shape.getD dd.toNat 0 := by
          rw [← hsum, ← hshapes, List.foldl_map]
        rw [tensor_sem_of hm hd]
        simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, hr, attrInt?_blind,
          hdim]
        rw [mapM_tensors _ (fun _ => rfl)]
        simp only [hrank, Bool.false_eq_true, ↓reduceIte, hext, ne_eq, not_true_eq_false]
        refine Safe.of_ok (List.Forall₂.cons ⟨rfl, rfl, fun i hi => ?_⟩ .nil)
        simp only [Tensor.canon, hi, ↓reduceIte]
        exact concatPick_typed hi _ 0 (Nat.zero_le _) hpart
      · cases h
    · cases h
  · cases h

/-! ### Elements -/

/-- `tensor.splat %x`: a scalar of the result's element type. -/
def splatRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.splat" then
    match Sem.Ops.Tensor.resultTensorTy op with
    | some (sh, e) => if argTys = [.scalar e] ∧ resTys = [.tensor sh e] then some [] else none
    | none => none
  else none

theorem splatRule_sound (ib : Nat) : RuleSound ib splatRule := by
  intro d op argTys resTys sigs hm h
  simp only [splatRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next sh e hr =>
      split at h
      · next hc2 =>
        obtain ⟨rfl, rfl⟩ := hc2
        cases h
        intro ctx call args regions _ _ hargs _
        obtain _ | ⟨hx, hnil⟩ := hargs
        cases hnil
        rename_i a
        cases a with
        | tensor _ => exact (hx : False).elim
        | scalar x =>
        rw [tensor_sem_of hm hd]
        simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, hr]
        exact Safe.of_ok (List.Forall₂.cons (hasTy_full hx) .nil)
      · cases h
    · cases h
  · cases h

/-- `tensor.cast` between equal static types. -/
def castRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.cast" then
    match Sem.Ops.Tensor.resultTensorTy op with
    | some (sh, e) => if argTys = [.tensor sh e] ∧ resTys = [.tensor sh e] then some [] else none
    | none => none
  else none

theorem castRule_sound (ib : Nat) : RuleSound ib castRule := by
  intro d op argTys resTys sigs hm h
  simp only [castRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next sh e hr =>
      split at h
      · next hc2 =>
        obtain ⟨rfl, rfl⟩ := hc2
        cases h
        intro ctx call args regions _ _ hargs _
        obtain _ | ⟨ht, hnil⟩ := hargs
        cases hnil
        rename_i a
        cases a with
        | scalar _ => exact (ht : False).elim
        | tensor t =>
        have ht' := ht
        obtain ⟨hs, he, _⟩ := ht'
        rw [tensor_sem_of hm hd]
        simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, hr, hs, he,
          beq_self_eq_true, Bool.and_self, ↓reduceIte]
        exact Safe.of_ok (List.Forall₂.cons (hasTy_canonize (v := .tensor t) ht) .nil)
      · cases h
    · cases h
  · cases h

/-- Integer coordinates inside `shape`, as `tensor.extract`/`insert` check them. -/
theorem coords_inBounds : ∀ (is : List Int) (shape : List Nat), is.length = shape.length →
    ((is.zip shape).all fun (x, s) => 0 ≤ x ∧ x < s) = true → Idx.InBounds shape (is.map Int.toNat)
  | [], [], _, _ => (Idx.inBounds_nil_iff []).2 rfl
  | x :: xs, s :: ss, hl, h => by
    simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨h0, hs⟩, hrest⟩ := h
    simp only [List.map_cons]
    exact (Idx.inBounds_cons_cons _ _ _ _).2
      ⟨by omega, coords_inBounds xs ss (by simpa using hl) hrest⟩
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl

/-- `tensor.extract %t[%i...]`: one `index` per dimension. Out-of-bounds or poison indices are UB. -/
def extractRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.extract" then
    match argTys with
    | .tensor sh e :: idxTys =>
      if idxTys = List.replicate sh.length (.scalar .index) ∧ resTys = [.scalar e] then some []
      else none
    | _ => none
  else none

theorem extractRule_sound (ib : Nat) : RuleSound ib extractRule := by
  intro d op argTys resTys sigs hm h
  simp only [extractRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next sh e idxTys =>
      split at h
      · next hc2 =>
        obtain ⟨-, rfl⟩ := hc2
        cases h
        intro ctx call args regions _ _ hargs _
        obtain _ | ⟨ht, _⟩ := hargs
        rename_i a idxs _
        cases a with
        | scalar _ => exact (ht : False).elim
        | tensor t =>
        obtain ⟨hs, he, hg⟩ := ht
        subst hs he
        rw [tensor_sem_of hm hd]
        simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn]
        split
        · next is _ =>
          split
          · next hin =>
            exact Safe.of_ok (List.Forall₂.cons (hg _ (coords_inBounds is t.shape hin.1 hin.2)) .nil)
          · exact Safe.of_ub
        · exact Safe.of_ub
      · cases h
    · cases h
  · cases h

/-- `tensor.insert %x into %t[%i...]`: a scalar of the element type, one `index` per dimension. -/
def insertRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.insert" then
    match argTys with
    | .scalar e' :: .tensor sh e :: idxTys =>
      if e' = e ∧ idxTys = List.replicate sh.length (.scalar .index) ∧ resTys = [.tensor sh e] then
        some []
      else none
    | _ => none
  else none

theorem insertRule_sound (ib : Nat) : RuleSound ib insertRule := by
  intro d op argTys resTys sigs hm h
  simp only [insertRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next e' sh e idxTys =>
      split at h
      · next hc2 =>
        obtain ⟨rfl, -, rfl⟩ := hc2
        cases h
        intro ctx call args regions _ _ hargs _
        obtain _ | ⟨hx, _ | ⟨ht, _⟩⟩ := hargs
        rename_i a b idxs _
        cases a with
        | tensor _ => exact (hx : False).elim
        | scalar x =>
        cases b with
        | scalar _ => exact (ht : False).elim
        | tensor t =>
        obtain ⟨hs, he, hg⟩ := ht
        rw [tensor_sem_of hm hd]
        simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn]
        split
        · next is _ =>
          split
          · next hin =>
            refine Safe.of_ok (List.Forall₂.cons ⟨hs, he, fun k hk => ?_⟩ .nil)
            simp only [Tensor.canonize, Tensor.canon, Tensor.set, hs, hk, ↓reduceIte]
            split
            · exact hx
            · exact hg k hk
          · exact Safe.of_ub
        · exact Safe.of_ub
      · cases h
    · cases h
  · cases h

/-- `tensor.from_elements`: one scalar of the element type per element, in row-major order. -/
def fromElementsRule : OpRule := fun _ op argTys resTys =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.from_elements" then
    match Sem.Ops.Tensor.resultTensorTy op with
    | some (sh, e) =>
      if argTys = List.replicate (shapeSize sh) (.scalar e) ∧ resTys = [.tensor sh e] then some []
      else none
    | none => none
  else none

theorem fromElementsRule_sound (ib : Nat) : RuleSound ib fromElementsRule := by
  intro d op argTys resTys sigs hm h
  simp only [fromElementsRule] at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next sh e hr =>
      split at h
      · next hc2 =>
        obtain ⟨rfl, rfl⟩ := hc2
        cases h
        intro ctx call args regions _ _ hargs _
        have hrep : List.replicate (shapeSize sh) (VTy.scalar e) =
            (List.replicate (shapeSize sh) e).map (liftTy none) := by simp [liftTy]
        rw [hrep] at hargs
        obtain ⟨vs, rfl, hvs⟩ := hasTys_lift_scalar hargs
        have hlen : vs.length = shapeSize sh := by
          rw [List.Forall₂.length_eq hvs, List.length_replicate]
        rw [tensor_sem_of hm hd]
        simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, hr]
        rw [mapM_scalars _ (fun _ => rfl)]
        simp only [hlen, ↓reduceIte]
        refine Safe.of_ok (List.Forall₂.cons ⟨rfl, rfl, fun i hi => ?_⟩ .nil)
        simp only [Tensor.canon, hi, ↓reduceIte]
        rcases toArray_getD_mem vs (linearIndex sh i) .poison with hp | hp
        · rw [hp]; exact hasTy_poison _ _
        · exact ValsTyped.of_all hvs (fun t ht => (List.eq_of_mem_replicate ht)) _ hp
      · cases h
    · cases h
  · cases h

/-- The `tensor` typing rule. -/
def tensorRule : OpRule :=
  firstRule [emptyRule, reshapeRule, extractSliceRule, insertSliceRule, concatRule, splatRule,
    castRule, extractRule, insertRule, fromElementsRule]

theorem tensorRule_sound (ib : Nat) : RuleSound ib tensorRule := by
  apply firstRule_sound
  simp only [List.forall_mem_cons, List.not_mem_nil, IsEmpty.forall_iff, implies_true, and_true]
  exact ⟨emptyRule_sound ib, reshapeRule_sound ib, extractSliceRule_sound ib,
    insertSliceRule_sound ib, concatRule_sound ib, splatRule_sound ib, castRule_sound ib,
    extractRule_sound ib, insertRule_sound ib, fromElementsRule_sound ib⟩

end MerlinVerify.Check
