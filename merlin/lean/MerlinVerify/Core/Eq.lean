/-
Decidable equality for the AST's mutual/nested types (`Ty`, `Attr`).

Lean's `deriving DecidableEq` does not handle mutual or nested inductives, and checker soundness
needs *proved* equality (alpha-equivalence of ops compares attributes and types), so the boolean
equalities here come with soundness (`beq = true → a = b`) and reflexivity proofs, from which the
`DecidableEq` instances are built.
-/
import MerlinVerify.Core.AST

namespace MerlinVerify

mutual
  def Ty.beq : Ty → Ty → Bool
    | .int a, .int b => a == b
    | .sint a, .sint b => a == b
    | .uint a, .uint b => a == b
    | .index, .index => true
    | .float a, .float b => a == b
    | .tensor s e c, .tensor s' e' c' => s == s' && Ty.beq e e' && Attr.beqOpt c c'
    | .memref s e l m, .memref s' e' l' m' =>
      s == s' && Ty.beq e e' && Attr.beqOpt l l' && Attr.beqOpt m m'
    | .vector s k e, .vector s' k' e' => s == s' && k == k' && Ty.beq e e'
    | .function i r, .function i' r' => Ty.beqList i i' && Ty.beqList r r'
    | .none, .none => true
    | .opaque a, .opaque b => a == b
    | _, _ => false

  def Ty.beqList : List Ty → List Ty → Bool
    | [], [] => true
    | a :: as, b :: bs => Ty.beq a b && Ty.beqList as bs
    | _, _ => false

  def Attr.beq : Attr → Attr → Bool
    | .unit, .unit => true
    | .bool a, .bool b => a == b
    | .int v t, .int v' t' => v == v' && Ty.beq t t'
    | .float v t, .float v' t' => v == v' && Ty.beq t t'
    | .str a, .str b => a == b
    | .type a, .type b => Ty.beq a b
    | .array xs, .array ys => Attr.beqList xs ys
    | .denseArray t v, .denseArray t' v' => Ty.beq t t' && v == v'
    | .dense t s b, .dense t' s' b' => Ty.beq t t' && s == s' && b == b'
    | .affineMap m, .affineMap m' => decide (m = m')
    | .enum a b c, .enum a' b' c' => a == a' && b == b' && c == c'
    | .dict e, .dict e' => Attr.beqEntries e e'
    | .symbolRef r n, .symbolRef r' n' => r == r' && n == n'
    | .opaque a, .opaque b => a == b
    | _, _ => false

  def Attr.beqOpt : Option Attr → Option Attr → Bool
    | none, none => true
    | some a, some b => Attr.beq a b
    | _, _ => false

  def Attr.beqList : List Attr → List Attr → Bool
    | [], [] => true
    | a :: as, b :: bs => Attr.beq a b && Attr.beqList as bs
    | _, _ => false

  def Attr.beqEntries : List (String × Attr) → List (String × Attr) → Bool
    | [], [] => true
    | (k, a) :: as, (k', b) :: bs => k == k' && Attr.beq a b && Attr.beqEntries as bs
    | _, _ => false
end

mutual
  theorem Ty.eq_of_beq : (a b : Ty) → Ty.beq a b = true → a = b
    | .int _, .int _, h | .sint _, .sint _, h | .uint _, .uint _, h | .float _, .float _, h
    | .opaque _, .opaque _, h => by simp_all [Ty.beq]
    | .index, .index, _ | .none, .none, _ => rfl
    | .tensor s e c, .tensor s' e' c', h => by
      simp only [Ty.beq, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨h1, h2⟩, h3⟩ := h
      rw [h1, Ty.eq_of_beq e e' h2, Attr.eq_of_beqOpt c c' h3]
    | .memref s e l m, .memref s' e' l' m', h => by
      simp only [Ty.beq, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
      rw [h1, Ty.eq_of_beq e e' h2, Attr.eq_of_beqOpt l l' h3, Attr.eq_of_beqOpt m m' h4]
    | .vector s k e, .vector s' k' e', h => by
      simp only [Ty.beq, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨h1, h2⟩, h3⟩ := h
      rw [h1, h2, Ty.eq_of_beq e e' h3]
    | .function i r, .function i' r', h => by
      simp only [Ty.beq, Bool.and_eq_true] at h
      rw [Ty.eq_of_beqList i i' h.1, Ty.eq_of_beqList r r' h.2]
    | .int _, .sint _, h | .int _, .uint _, h | .int _, .index, h | .int _, .float _, h
    | .int _, .tensor .., h | .int _, .memref .., h | .int _, .vector .., h | .int _, .function .., h
    | .int _, .none, h | .int _, .opaque _, h
    | .sint _, .int _, h | .sint _, .uint _, h | .sint _, .index, h | .sint _, .float _, h
    | .sint _, .tensor .., h | .sint _, .memref .., h | .sint _, .vector .., h | .sint _, .function .., h
    | .sint _, .none, h | .sint _, .opaque _, h
    | .uint _, .int _, h | .uint _, .sint _, h | .uint _, .index, h | .uint _, .float _, h
    | .uint _, .tensor .., h | .uint _, .memref .., h | .uint _, .vector .., h | .uint _, .function .., h
    | .uint _, .none, h | .uint _, .opaque _, h
    | .index, .int _, h | .index, .sint _, h | .index, .uint _, h | .index, .float _, h
    | .index, .tensor .., h | .index, .memref .., h | .index, .vector .., h | .index, .function .., h
    | .index, .none, h | .index, .opaque _, h
    | .float _, .int _, h | .float _, .sint _, h | .float _, .uint _, h | .float _, .index, h
    | .float _, .tensor .., h | .float _, .memref .., h | .float _, .vector .., h | .float _, .function .., h
    | .float _, .none, h | .float _, .opaque _, h
    | .tensor .., .int _, h | .tensor .., .sint _, h | .tensor .., .uint _, h | .tensor .., .index, h
    | .tensor .., .float _, h | .tensor .., .memref .., h | .tensor .., .vector .., h
    | .tensor .., .function .., h | .tensor .., .none, h | .tensor .., .opaque _, h
    | .memref .., .int _, h | .memref .., .sint _, h | .memref .., .uint _, h | .memref .., .index, h
    | .memref .., .float _, h | .memref .., .tensor .., h | .memref .., .vector .., h
    | .memref .., .function .., h | .memref .., .none, h | .memref .., .opaque _, h
    | .vector .., .int _, h | .vector .., .sint _, h | .vector .., .uint _, h | .vector .., .index, h
    | .vector .., .float _, h | .vector .., .tensor .., h | .vector .., .memref .., h
    | .vector .., .function .., h | .vector .., .none, h | .vector .., .opaque _, h
    | .function .., .int _, h | .function .., .sint _, h | .function .., .uint _, h
    | .function .., .index, h | .function .., .float _, h | .function .., .tensor .., h
    | .function .., .memref .., h | .function .., .vector .., h | .function .., .none, h
    | .function .., .opaque _, h
    | .none, .int _, h | .none, .sint _, h | .none, .uint _, h | .none, .index, h
    | .none, .float _, h | .none, .tensor .., h | .none, .memref .., h | .none, .vector .., h
    | .none, .function .., h | .none, .opaque _, h
    | .opaque _, .int _, h | .opaque _, .sint _, h | .opaque _, .uint _, h | .opaque _, .index, h
    | .opaque _, .float _, h | .opaque _, .tensor .., h | .opaque _, .memref .., h
    | .opaque _, .vector .., h | .opaque _, .function .., h | .opaque _, .none, h => by
      simp [Ty.beq] at h

  theorem Ty.eq_of_beqList : (a b : List Ty) → Ty.beqList a b = true → a = b
    | [], [], _ => rfl
    | x :: xs, y :: ys, h => by
      simp only [Ty.beqList, Bool.and_eq_true] at h
      rw [Ty.eq_of_beq x y h.1, Ty.eq_of_beqList xs ys h.2]
    | [], _ :: _, h | _ :: _, [], h => by simp [Ty.beqList] at h

  theorem Attr.eq_of_beqOpt : (a b : Option Attr) → Attr.beqOpt a b = true → a = b
    | none, none, _ => rfl
    | some x, some y, h => by simp only [Attr.beqOpt] at h; rw [Attr.eq_of_beq x y h]
    | none, some _, h | some _, none, h => by simp [Attr.beqOpt] at h

  theorem Attr.eq_of_beq : (a b : Attr) → Attr.beq a b = true → a = b
    | .unit, .unit, _ => rfl
    | .bool _, .bool _, h | .str _, .str _, h | .opaque _, .opaque _, h => by simp_all [Attr.beq]
    | .int v t, .int v' t', h | .float v t, .float v' t', h => by
      simp only [Attr.beq, Bool.and_eq_true, beq_iff_eq] at h
      rw [h.1, Ty.eq_of_beq t t' h.2]
    | .type a, .type b, h => by simp only [Attr.beq] at h; rw [Ty.eq_of_beq a b h]
    | .array xs, .array ys, h => by simp only [Attr.beq] at h; rw [Attr.eq_of_beqList xs ys h]
    | .denseArray t v, .denseArray t' v', h => by
      simp only [Attr.beq, Bool.and_eq_true, beq_iff_eq] at h
      rw [Ty.eq_of_beq t t' h.1, h.2]
    | .dense t s b, .dense t' s' b', h => by
      simp only [Attr.beq, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨h1, h2⟩, h3⟩ := h
      rw [Ty.eq_of_beq t t' h1, h2, h3]
    | .affineMap m, .affineMap m', h => by simp only [Attr.beq, decide_eq_true_eq] at h; rw [h]
    | .enum a b c, .enum a' b' c', h => by
      simp only [Attr.beq, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨h1, h2⟩, h3⟩ := h
      rw [h1, h2, h3]
    | .dict e, .dict e', h => by simp only [Attr.beq] at h; rw [Attr.eq_of_beqEntries e e' h]
    | .symbolRef r n, .symbolRef r' n', h => by
      simp only [Attr.beq, Bool.and_eq_true, beq_iff_eq] at h
      rw [h.1, h.2]
    | .unit, .bool _, h | .unit, .int .., h | .unit, .float .., h | .unit, .str _, h
    | .unit, .type _, h | .unit, .array _, h | .unit, .denseArray .., h | .unit, .dense .., h
    | .unit, .affineMap _, h | .unit, .enum .., h | .unit, .dict _, h | .unit, .symbolRef .., h
    | .unit, .opaque _, h
    | .bool _, .unit, h | .bool _, .int .., h | .bool _, .float .., h | .bool _, .str _, h
    | .bool _, .type _, h | .bool _, .array _, h | .bool _, .denseArray .., h | .bool _, .dense .., h
    | .bool _, .affineMap _, h | .bool _, .enum .., h | .bool _, .dict _, h | .bool _, .symbolRef .., h
    | .bool _, .opaque _, h
    | .int .., .unit, h | .int .., .bool _, h | .int .., .float .., h | .int .., .str _, h
    | .int .., .type _, h | .int .., .array _, h | .int .., .denseArray .., h | .int .., .dense .., h
    | .int .., .affineMap _, h | .int .., .enum .., h | .int .., .dict _, h | .int .., .symbolRef .., h
    | .int .., .opaque _, h
    | .float .., .unit, h | .float .., .bool _, h | .float .., .int .., h | .float .., .str _, h
    | .float .., .type _, h | .float .., .array _, h | .float .., .denseArray .., h
    | .float .., .dense .., h | .float .., .affineMap _, h | .float .., .enum .., h
    | .float .., .dict _, h | .float .., .symbolRef .., h | .float .., .opaque _, h
    | .str _, .unit, h | .str _, .bool _, h | .str _, .int .., h | .str _, .float .., h
    | .str _, .type _, h | .str _, .array _, h | .str _, .denseArray .., h | .str _, .dense .., h
    | .str _, .affineMap _, h | .str _, .enum .., h | .str _, .dict _, h | .str _, .symbolRef .., h
    | .str _, .opaque _, h
    | .type _, .unit, h | .type _, .bool _, h | .type _, .int .., h | .type _, .float .., h
    | .type _, .str _, h | .type _, .array _, h | .type _, .denseArray .., h | .type _, .dense .., h
    | .type _, .affineMap _, h | .type _, .enum .., h | .type _, .dict _, h
    | .type _, .symbolRef .., h | .type _, .opaque _, h
    | .array _, .unit, h | .array _, .bool _, h | .array _, .int .., h | .array _, .float .., h
    | .array _, .str _, h | .array _, .type _, h | .array _, .denseArray .., h
    | .array _, .dense .., h | .array _, .affineMap _, h | .array _, .enum .., h
    | .array _, .dict _, h | .array _, .symbolRef .., h | .array _, .opaque _, h
    | .denseArray .., .unit, h | .denseArray .., .bool _, h | .denseArray .., .int .., h
    | .denseArray .., .float .., h | .denseArray .., .str _, h | .denseArray .., .type _, h
    | .denseArray .., .array _, h | .denseArray .., .dense .., h | .denseArray .., .affineMap _, h
    | .denseArray .., .enum .., h | .denseArray .., .dict _, h | .denseArray .., .symbolRef .., h
    | .denseArray .., .opaque _, h
    | .dense .., .unit, h | .dense .., .bool _, h | .dense .., .int .., h | .dense .., .float .., h
    | .dense .., .str _, h | .dense .., .type _, h | .dense .., .array _, h
    | .dense .., .denseArray .., h | .dense .., .affineMap _, h | .dense .., .enum .., h
    | .dense .., .dict _, h | .dense .., .symbolRef .., h | .dense .., .opaque _, h
    | .affineMap _, .unit, h | .affineMap _, .bool _, h | .affineMap _, .int .., h
    | .affineMap _, .float .., h | .affineMap _, .str _, h | .affineMap _, .type _, h
    | .affineMap _, .array _, h | .affineMap _, .denseArray .., h | .affineMap _, .dense .., h
    | .affineMap _, .enum .., h | .affineMap _, .dict _, h | .affineMap _, .symbolRef .., h
    | .affineMap _, .opaque _, h
    | .enum .., .unit, h | .enum .., .bool _, h | .enum .., .int .., h | .enum .., .float .., h
    | .enum .., .str _, h | .enum .., .type _, h | .enum .., .array _, h
    | .enum .., .denseArray .., h | .enum .., .dense .., h | .enum .., .affineMap _, h
    | .enum .., .dict _, h | .enum .., .symbolRef .., h | .enum .., .opaque _, h
    | .dict _, .unit, h | .dict _, .bool _, h | .dict _, .int .., h | .dict _, .float .., h
    | .dict _, .str _, h | .dict _, .type _, h | .dict _, .array _, h | .dict _, .denseArray .., h
    | .dict _, .dense .., h | .dict _, .affineMap _, h | .dict _, .enum .., h
    | .dict _, .symbolRef .., h | .dict _, .opaque _, h
    | .symbolRef .., .unit, h | .symbolRef .., .bool _, h | .symbolRef .., .int .., h
    | .symbolRef .., .float .., h | .symbolRef .., .str _, h | .symbolRef .., .type _, h
    | .symbolRef .., .array _, h | .symbolRef .., .denseArray .., h | .symbolRef .., .dense .., h
    | .symbolRef .., .affineMap _, h | .symbolRef .., .enum .., h | .symbolRef .., .dict _, h
    | .symbolRef .., .opaque _, h
    | .opaque _, .unit, h | .opaque _, .bool _, h | .opaque _, .int .., h | .opaque _, .float .., h
    | .opaque _, .str _, h | .opaque _, .type _, h | .opaque _, .array _, h
    | .opaque _, .denseArray .., h | .opaque _, .dense .., h | .opaque _, .affineMap _, h
    | .opaque _, .enum .., h | .opaque _, .dict _, h | .opaque _, .symbolRef .., h => by
      simp [Attr.beq] at h

  theorem Attr.eq_of_beqList : (a b : List Attr) → Attr.beqList a b = true → a = b
    | [], [], _ => rfl
    | x :: xs, y :: ys, h => by
      simp only [Attr.beqList, Bool.and_eq_true] at h
      rw [Attr.eq_of_beq x y h.1, Attr.eq_of_beqList xs ys h.2]
    | [], _ :: _, h | _ :: _, [], h => by simp [Attr.beqList] at h

  theorem Attr.eq_of_beqEntries : (a b : List (String × Attr)) → Attr.beqEntries a b = true → a = b
    | [], [], _ => rfl
    | (k, x) :: xs, (k', y) :: ys, h => by
      simp only [Attr.beqEntries, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨h1, h2⟩, h3⟩ := h
      rw [h1, Attr.eq_of_beq x y h2, Attr.eq_of_beqEntries xs ys h3]
    | [], _ :: _, h | _ :: _, [], h => by simp [Attr.beqEntries] at h
end

mutual
  theorem Ty.beq_refl : (a : Ty) → Ty.beq a a = true
    | .int _ | .sint _ | .uint _ | .float _ | .opaque _ | .index | .none => by simp [Ty.beq]
    | .tensor s e c => by simp [Ty.beq, Ty.beq_refl e, Attr.beqOpt_refl c]
    | .memref s e l m => by simp [Ty.beq, Ty.beq_refl e, Attr.beqOpt_refl l, Attr.beqOpt_refl m]
    | .vector s k e => by simp [Ty.beq, Ty.beq_refl e]
    | .function i r => by simp [Ty.beq, Ty.beqList_refl i, Ty.beqList_refl r]

  theorem Ty.beqList_refl : (a : List Ty) → Ty.beqList a a = true
    | [] => by simp [Ty.beqList]
    | x :: xs => by simp [Ty.beqList, Ty.beq_refl x, Ty.beqList_refl xs]

  theorem Attr.beqOpt_refl : (a : Option Attr) → Attr.beqOpt a a = true
    | none => by simp [Attr.beqOpt]
    | some x => by simp [Attr.beqOpt, Attr.beq_refl x]

  theorem Attr.beq_refl : (a : Attr) → Attr.beq a a = true
    | .unit | .bool _ | .str _ | .opaque _ | .affineMap _ | .enum .. | .symbolRef .. => by
      simp [Attr.beq]
    | .int _ t | .float _ t => by simp [Attr.beq, Ty.beq_refl t]
    | .type t => by simp [Attr.beq, Ty.beq_refl t]
    | .array xs => by simp [Attr.beq, Attr.beqList_refl xs]
    | .denseArray t _ => by simp [Attr.beq, Ty.beq_refl t]
    | .dense t _ _ => by simp [Attr.beq, Ty.beq_refl t]
    | .dict e => by simp [Attr.beq, Attr.beqEntries_refl e]

  theorem Attr.beqList_refl : (a : List Attr) → Attr.beqList a a = true
    | [] => by simp [Attr.beqList]
    | x :: xs => by simp [Attr.beqList, Attr.beq_refl x, Attr.beqList_refl xs]

  theorem Attr.beqEntries_refl : (a : List (String × Attr)) → Attr.beqEntries a a = true
    | [] => by simp [Attr.beqEntries]
    | (_, x) :: xs => by simp [Attr.beqEntries, Attr.beq_refl x, Attr.beqEntries_refl xs]
end

instance : DecidableEq Ty := fun a b =>
  if h : Ty.beq a b = true then isTrue (Ty.eq_of_beq a b h)
  else isFalse (fun e => h (e ▸ Ty.beq_refl a))

instance : DecidableEq Attr := fun a b =>
  if h : Attr.beq a b = true then isTrue (Attr.eq_of_beq a b h)
  else isFalse (fun e => h (e ▸ Attr.beq_refl a))

end MerlinVerify
