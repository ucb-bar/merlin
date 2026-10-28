/- Provisional independent batched-linear intent; NOT MLIR semantics of the invalid named op.
Intent approval is separate. No reassociation, fused operation or source equivalence assumed. -/
import MerlinVerify.Sem.Ops
import MerlinVerify.Check.StructuredRuntime
namespace MerlinVerify.Rules.OverrankIntent
open MerlinVerify MerlinVerify.Sem.Ops.Linalg
structure Shape where
  batch : List Nat
  k : Nat
  n : Nat
  deriving Repr

def Shape.left (s : Shape) := s.batch ++ [s.k]
def Shape.right (s : Shape) := [s.k, s.n]
def Shape.out (s : Shape) := s.batch ++ [s.n]
def Shape.ranges (s : Shape) := s.out ++ [s.k]
def zero (k : FKind) : FBits k := 0
def mulKey : FloatOpKey := Sem.Ops.Arith.floatKey
  (.mk "arith.mulf" [] [] [("fastmath", .enum "arith" "fastmath" "none")] [] [] [])
def addKey : FloatOpKey := Sem.Ops.Arith.floatKey
  (.mk "arith.addf" [] [] [("fastmath", .enum "arith" "fastmath" "none")] [] [] [])
def value (k : FKind) : Option (FBits k) → Val
  | none => .poison
  | some x => .some (.flt k x)
def step (M : FloatModel) (kind : FKind) (a b acc : FBits kind) : FBits kind :=
  M.op addKey kind [acc, M.op mulKey kind [a, b]]
def stepOpt (M : FloatModel) (kind : FKind) (a b acc : Option (FBits kind)) : Option (FBits kind) :=
  do return step M kind (← a) (← b) (← acc)
/-- Increasing k, left-associated, separate multiply/add in the declared format. No widening,
FMA or fast-math. NaNs, infinities, signed zeros and subnormals go through the same float model. -/
def cell (M : FloatModel) (kind : FKind) (s : Shape)
    (A B : Idx → Option (FBits kind)) (output : Idx) : Option (FBits kind) :=
  (List.range s.k).foldl (fun acc k =>
    stepOpt M kind (A (output.take s.batch.length ++ [k]))
      (B [k, output[s.batch.length]?.getD 0]) acc) (some (zero kind))
def tensor (M : FloatModel) (kind : FKind) (s : Shape)
    (A B : Idx → Option (FBits kind)) : Tensor :=
  Tensor.canon s.out (.float kind) (fun p => value kind (cell M kind s A B p))
/-- Defined input domain, inhabited for every shape. -/
def Domain (kind : FKind) (s : Shape) (A B : Idx → Option (FBits kind)) : Prop :=
  (∀ p, Idx.InBounds s.left p → (A p).isSome = true) ∧
  (∀ p, Idx.InBounds s.right p → (B p).isSome = true)
theorem domain_inhabited (kind : FKind) (s : Shape) :
    Domain kind s (fun _ => some (zero kind)) (fun _ => some (zero kind)) := by
  constructor <;> intro p hp <;> rfl
/-- Canonical out-of-bounds poison is never an element. -/
def Clean (t : Tensor) : Prop := ∀ p, Idx.InBounds t.shape p → t.get p ≠ .poison
/-- Geometry maps are implementation obligations, independent of the cell formula. -/
def maps (s : Shape) : List AffineMap :=
  let r := s.batch.length + 1
  let lead := (List.range s.batch.length).map AffineExpr.dim
  [⟨r+1, 0, lead ++ [.dim r]⟩, ⟨r+1, 0, [.dim r, .dim s.batch.length]⟩,
   ⟨r+1, 0, lead ++ [.dim s.batch.length]⟩]
def iters (s : Shape) : List IterKind :=
  List.replicate (s.batch.length + 1) .parallel ++ [.reduction]
def types (kind : FKind) (s : Shape) : List Check.VTy :=
  [.tensor s.left (.float kind), .tensor s.right (.float kind), .tensor s.out (.float kind)]
def layoutCheck (kind : FKind) (s : Shape) : Bool :=
  Check.structuredRanges? (maps s) (iters s).length (types kind s) == some s.ranges
theorem layout_sound {kind : FKind} {s : Shape} (h : layoutCheck kind s = true) :
    Check.StructuredOk (maps s) (iters s).length (types kind s) s.ranges :=
  Check.structuredRanges?_spec (by simpa [layoutCheck, beq_iff_eq] using h)
end MerlinVerify.Rules.OverrankIntent
