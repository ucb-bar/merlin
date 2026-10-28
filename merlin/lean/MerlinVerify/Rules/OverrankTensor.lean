/- Lift the checked scalar body through all broadcasted reduction writes. Stable library lemmas
are reused read-only; this module neither imports policy claims nor changes W8A8. -/
import MerlinVerify.Rules.OverrankBody
import MerlinVerify.Rules.OverrankOrder
import MerlinVerify.Rules.W8A8TensorTrace
namespace MerlinVerify.Rules.OverrankTensor
open MerlinVerify Sem.Ops.Linalg OverrankIntent OverrankCoordinates

def floats (kind : FKind) (shape : List Nat) (get : Idx → Option (FBits kind)) : Tensor :=
  ⟨shape,.float kind,fun p => value kind (get p)⟩
theorem floats_typed (ib : Nat) (kind : FKind) (shape : List Nat) (get : Idx → Option (FBits kind)) :
    Check.RVal.HasTy ib (.tensor (floats kind shape get)) (.tensor shape (.float kind)) := by
  refine ⟨rfl,rfl,?_⟩
  intro p hp
  cases h : get p <;> simp [floats,value,Check.Val.HasTy,Scalar.conforms,h]

/-- Acceptance of geometry and actual scalar region establishes the exact independent tensor
intent. The outcome is successful, including zero K and empty output domains. -/
theorem generic_sound (ctx : Ctx) (call : CallSem) (env : Env) (kind : FKind) (s : Shape)
    (geometry : layoutCheck kind s = true) {actual : Region}
    (body : Check.alphaRegion [] (OverrankBody.region kind) actual = true)
    (A B : Idx → Option (FBits kind)) :
    genericSem ctx (maps s) (iters s)
      [.tensor (floats kind s.left A),.tensor (floats kind s.right B)]
      [.tensor (Tensor.full s.out (.float kind) (value kind (some (zero kind))))]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (tensor ctx.M kind s A B)] := by
  have g := layout_sound geometry
  obtain ⟨hl,hr,ho⟩ : Check.mapInBounds s.ranges (maps s)[0]! s.left = true ∧
      Check.mapInBounds s.ranges (maps s)[1]! s.right = true ∧
      Check.mapInBounds s.ranges (maps s)[2]! s.out = true := by
    cases g.inBounds with
    | cons hl rest => cases rest with
      | cons hr rest => cases rest with
        | cons ho rest => exact ⟨hl,hr,ho⟩
  have hindex (p : Idx) (hp : Idx.InBounds s.ranges p) := index_of_eval ho hp (eval_out s p hp)
  have hleft (p : Idx) (hp : Idx.InBounds s.ranges p) := index_of_eval hl hp (eval_left s p hp)
  have hright (p : Idx) (hp : Idx.InBounds s.ranges p) := index_of_eval hr hp (eval_right s p hp)
  let initial : Idx → Option (FBits kind) := fun p =>
    if Idx.InBounds s.out p then some (zero kind) else none
  have hfull : Tensor.full s.out (.float kind) (value kind (some (zero kind))) =
      W8A8TensorTrace.pack s.out (.float kind) (value kind) initial := by
    simp only [Tensor.full,Tensor.canon,W8A8TensorTrace.pack,initial]
    congr 1
    funext p
    split <;> rfl
  rw [hfull]
  have run := W8A8TensorTrace.generic_sound (ib := ctx.indexBits) (ctx := ctx)
    (inTys := [.tensor s.left (.float kind),.tensor s.right (.float kind)])
    g (.cons (floats_typed _ _ _ A) (.cons (floats_typed _ _ _ B) .nil))
    (value kind) initial (floats_typed _ _ _ initial)
    (by rfl : (maps s).drop 2 = [(maps s)[2]!])
    (fun c => evalRegion Sem.Ops.opSem c call env actual)
    (outCoord s) (fun p acc => stepOpt ctx.M kind (A (leftCoord s p)) (B (rightCoord s p)) acc)
    hindex (by
      intro p hp state
      have a : readAt (.tensor (floats kind s.left A)) (maps s)[0]! p =
          .ok (value kind (A (leftCoord s p))) := by simp only [readAt,floats,hleft p hp]
      have b : readAt (.tensor (floats kind s.right B)) (maps s)[1]! p =
          .ok (value kind (B (rightCoord s p))) := by simp only [readAt,floats,hright p hp]
      have o : readAt (.tensor (W8A8TensorTrace.pack s.out (.float kind) (value kind) state))
          (maps s)[2]! p = .ok (value kind (state (outCoord s p))) := by
        simp only [readAt,W8A8TensorTrace.pack,hindex p hp]
      have bs := OverrankBody.region_sound {ctx with loopIndex := p} call env kind
        (A (leftCoord s p)) (B (rightCoord s p)) (state (outCoord s p))
      change bodyAt ctx _ [(maps s)[0]!, (maps s)[1]!] [(maps s)[2]!] _ _ p = _
      simp only [bodyAt, List.zip_cons_cons, List.zip_nil_left, List.zip_nil_right,
        List.mapM_cons, List.mapM_nil, a, b, o, Outcome.bind_ok, Outcome.bindOp_eq,
        Outcome.pure_eq, List.map_cons, List.map_nil, List.cons_append, List.nil_append,
        OverrankBody.region_transfer call env body, bs]
      rfl)
  rw [run]
  congr 3
  apply Tensor.canon_ext
  intro p hp
  simp only [tensor,W8A8TensorTrace.accumulate_cell,initial,if_pos hp]
  rw [OverrankOrder.trace_cell ctx.M kind s A B p hp]
end MerlinVerify.Rules.OverrankTensor
