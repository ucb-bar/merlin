/-
Typing rules and safety proofs for the `tensor` ops (design §4.5, progress).

`tensor.empty` builds a tensor of poison. `tensor.collapse_shape`/`expand_shape` read the source
in row-major order: result element `i` is source element `delinearize src (linearIndex res i)`.
A valid reassociation preserves the element count (`reassocValid_size`), so every such read is
inside the source (`linearIndex_lt`, `delinearize_inBounds`): a reshape never reads past its
operand.
-/
import MerlinVerify.Check.RulesArith

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

/-- The `tensor` typing rule. -/
def tensorRule : OpRule := firstRule [emptyRule, reshapeRule]

theorem tensorRule_sound (ib : Nat) : RuleSound ib tensorRule := by
  apply firstRule_sound
  simp only [List.forall_mem_cons, List.not_mem_nil, IsEmpty.forall_iff, implies_true, and_true]
  exact ⟨emptyRule_sound ib, reshapeRule_sound ib⟩

end MerlinVerify.Check
