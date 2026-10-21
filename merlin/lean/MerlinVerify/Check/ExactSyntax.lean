/- Linear structural equality for complete exported programs. All fields, including SSA ids,
attributes and nested regions, participate. No hash or printer injectivity assumption is used. -/
import MerlinVerify.Core.Eq

namespace MerlinVerify.Check.ExactSyntax

open MerlinVerify

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

mutual
  def operations : List Op → List Op → Bool
    | [], [] => true
    | a :: rest, b :: tail => operation a b && operations rest tail
    | _, _ => false
  def operation : Op → Op → Bool
    | .mk n r o a p regionsA s, .mk n' r' o' a' p' regionsB s' =>
      n == n' && decide (r = r') && o == o' && decide (a = a') && decide (p = p') &&
        regions regionsA regionsB && s == s'
  def regions : List Region → List Region → Bool
    | [], [] => true
    | a :: rest, b :: tail => region a b && regions rest tail
    | _, _ => false
  def region : Region → Region → Bool
    | .mk a, .mk b => blocks a b
  def blocks : List Block → List Block → Bool
    | [], [] => true
    | a :: rest, b :: tail => block a b && blocks rest tail
    | _, _ => false
  def block : Block → Block → Bool
    | .mk args ops, .mk args' ops' => decide (args = args') && operations ops ops'
end

mutual
  theorem operations_sound : (a b : List Op) → operations a b = true → a = b
    | [], [], _ => rfl
    | a :: rest, b :: tail, h => by
      simp only [operations, Bool.and_eq_true] at h
      rw [operation_sound a b h.1, operations_sound rest tail h.2]
    | [], _ :: _, h | _ :: _, [], h => by simp [operations] at h
  theorem operation_sound : (a b : Op) → operation a b = true → a = b
    | .mk n r o a p ra s, .mk n' r' o' a' p' rb s', h => by
      simp only [operation, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      obtain ⟨⟨⟨⟨⟨⟨hn, hr⟩, ho⟩, ha⟩, hp⟩, hregions⟩, hs⟩ := h
      rw [hn, hr, ho, ha, hp, regions_sound ra rb hregions, hs]
  theorem regions_sound : (a b : List Region) → regions a b = true → a = b
    | [], [], _ => rfl
    | a :: rest, b :: tail, h => by
      simp only [regions, Bool.and_eq_true] at h
      rw [region_sound a b h.1, regions_sound rest tail h.2]
    | [], _ :: _, h | _ :: _, [], h => by simp [regions] at h
  theorem region_sound : (a b : Region) → region a b = true → a = b
    | .mk a, .mk b, h => by
      simp only [region] at h
      rw [blocks_sound a b h]
  theorem blocks_sound : (a b : List Block) → blocks a b = true → a = b
    | [], [], _ => rfl
    | a :: rest, b :: tail, h => by
      simp only [blocks, Bool.and_eq_true] at h
      rw [block_sound a b h.1, blocks_sound rest tail h.2]
    | [], _ :: _, h | _ :: _, [], h => by simp [blocks] at h
  theorem block_sound : (a b : Block) → block a b = true → a = b
    | .mk args ops, .mk args' ops', h => by
      simp only [block, Bool.and_eq_true, decide_eq_true_eq] at h
      rw [h.1, operations_sound ops ops' h.2]
end

def modules (a b : Module) : Bool :=
  operations a.ops b.ops && decide (a.attrs = b.attrs) && decide (a.props = b.props)

theorem modules_sound {a b : Module} (h : modules a b = true) : a = b := by
  cases a with
  | mk ops attrs props =>
    cases b with
    | mk ops' attrs' props' =>
      simp only [modules, Bool.and_eq_true, decide_eq_true_eq] at h
      rw [operations_sound ops ops' h.1.1, h.1.2, h.2]

#print axioms modules_sound

end MerlinVerify.Check.ExactSyntax
