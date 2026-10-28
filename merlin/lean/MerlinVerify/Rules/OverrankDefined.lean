/- Successful and poison-free execution on an explicit inhabited input domain. -/
import MerlinVerify.Rules.OverrankTensor
import MerlinVerify.Sem.Defined
namespace MerlinVerify.Rules.OverrankDefined
open MerlinVerify Sem.Ops.Linalg OverrankIntent OverrankCoordinates

private theorem step_some (M : FloatModel) (kind : FKind) (a b c : Option (FBits kind))
    (ha : a.isSome = true) (hb : b.isSome = true) (hc : c.isSome = true) :
    (stepOpt M kind a b c).isSome = true := by
  cases a <;> cases b <;> cases c <;> simp_all [stepOpt]

private theorem fold_some (M : FloatModel) (kind : FKind) (s : Shape)
    (A B : Idx → Option (FBits kind)) (cell : Idx) (points : List Idx)
    (h : ∀ p ∈ points, (A (leftCoord s p)).isSome = true ∧ (B (rightCoord s p)).isSome = true)
    (acc : Option (FBits kind)) (ha : acc.isSome = true) :
    (points.foldl (fun acc p => if cell = outCoord s p then
      stepOpt M kind (A (leftCoord s p)) (B (rightCoord s p)) acc else acc) acc).isSome = true := by
  induction points generalizing acc with
  | nil => exact ha
  | cons p ps ih =>
    simp only [List.foldl_cons]
    apply ih (fun q hq => h q (by simp [hq]))
    split
    · exact step_some M kind _ _ _ (h p (by simp)).1 (h p (by simp)).2 ha
    · exact ha

/-- No poison on actual output elements, for arbitrary initialized input bit patterns and every
float model. This is separate from both typing and successful execution. -/
theorem intent_clean (M : FloatModel) (kind : FKind) (s : Shape)
    (layout : layoutCheck kind s = true) (A B : Idx → Option (FBits kind))
    (domain : Domain kind s A B) : Clean (tensor M kind s A B) := by
  have g := layout_sound layout
  obtain ⟨hl,hr⟩ : Check.mapInBounds s.ranges (maps s)[0]! s.left = true ∧
      Check.mapInBounds s.ranges (maps s)[1]! s.right = true := by
    cases g.inBounds with
    | cons hl rest => cases rest with
      | cons hr rest => exact ⟨hl,hr⟩
  intro c hc
  have hc' : Idx.InBounds s.out c := hc
  have all (p : Idx) (hp : p ∈ allPoints s.ranges) :
      (A (leftCoord s p)).isSome = true ∧ (B (rightCoord s p)).isSome = true := by
    have hp' := (mem_allPoints _ _).mp hp
    obtain ⟨i,hi,hib⟩ := Check.mapInBounds_sound hl hp'
    have he := index_of_eval hl hp' (eval_left s p hp')
    rw [he] at hi
    have eq := Option.some.inj hi
    subst i
    obtain ⟨j,hj,hjb⟩ := Check.mapInBounds_sound hr hp'
    have he' := index_of_eval hr hp' (eval_right s p hp')
    rw [he'] at hj
    have eq' := Option.some.inj hj
    subst j
    exact ⟨domain.1 _ hib,domain.2 _ hjb⟩
  have hs := fold_some M kind s A B c (allPoints s.ranges) all (some (zero kind)) rfl
  rw [OverrankOrder.trace_cell M kind s A B c hc'] at hs
  change (Tensor.canon s.out (.float kind) (fun p => value kind (cell M kind s A B p))).get c ≠ .poison
  simp only [Tensor.canon, if_pos hc']
  cases he : cell M kind s A B c <;> simp_all [value]

/-- An `.ok` outcome and an explicitly poison-free tensor, not just a NotStuck/type conclusion. -/
theorem successful_clean (ctx : Ctx) (call : CallSem) (env : Env) (kind : FKind) (s : Shape)
    (layout : layoutCheck kind s = true) {actual : Region}
    (body : Check.alphaRegion [] (OverrankBody.region kind) actual = true)
    (A B : Idx → Option (FBits kind)) (domain : Domain kind s A B) :
    genericSem ctx (maps s) (iters s)
      [.tensor (OverrankTensor.floats kind s.left A),.tensor (OverrankTensor.floats kind s.right B)]
      [.tensor (Tensor.full s.out (.float kind) (value kind (some (zero kind))))]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) = .ok [.tensor (tensor ctx.M kind s A B)] ∧
    Clean (tensor ctx.M kind s A B) :=
  ⟨OverrankTensor.generic_sound ctx call env kind s layout body A B,
    intent_clean ctx.M kind s layout A B domain⟩
end MerlinVerify.Rules.OverrankDefined
