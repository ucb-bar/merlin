/- An indexed implementation of `alphaOps`, proved equal to it and installed with `@[csimp]`, so
compiled checkers run it while every definition and theorem keeps referring to `alphaOps`.

`alphaOps` looks every operand up in, and checks every result's freshness against, the list `ρ`,
which grows with each matched operation: quadratic in the length of the compared lists, and each
alias substitution compares a whole block continuation. The index keeps the same `ρ` (prepending
is cheap) beside persistent ordered maps with `ρ.find`'s first-match lookups and its image. They
are persistent so that a region's block arguments extend the index without copying it for the
enclosing operation. -/
import MerlinVerify.Check.Alpha
import Std.Data.TreeMap.Lemmas
import Std.Data.TreeSet.Lemmas

namespace MerlinVerify.Check

open MerlinVerify

structure AlphaIndex where
  fwd : Std.TreeMap ValueId ValueId compare
  img : Std.TreeSet ValueId compare

namespace AlphaIndex

theorem cmp_beq (a b : ValueId) : (compare a b == .eq) = (a == b) := by
  by_cases h : a = b <;> simp [h]

/-- The index of `ps ++ ρ` from the index of `ρ`. Earlier pairs are inserted last, so a lookup
returns the first matching pair, as `List.lookup` does. -/
def push (ix : AlphaIndex) (ps : Rho) : AlphaIndex :=
  { fwd := ps.foldr (fun p m => m.insert p.1 p.2) ix.fwd
    img := ps.foldr (fun p s => s.insert p.2) ix.img }

def Inv (ρ : Rho) (ix : AlphaIndex) : Prop :=
  (∀ x, ix.fwd[x]? = ρ.find x) ∧ (∀ y, ix.img.contains y = ρ.any (fun p => p.2 == y))

theorem push_fwd (ix : AlphaIndex) (ρ : Rho) (h : ∀ x, ix.fwd[x]? = ρ.find x) :
    ∀ (ps : Rho) (x : ValueId), (ix.push ps).fwd[x]? = Rho.find (ps ++ ρ) x
  | [], x => by simpa [push, Rho.find] using h x
  | p :: ps, x => by
    have ih := push_fwd ix ρ h ps x
    simp only [push, List.foldr_cons] at ih ⊢
    rw [Std.TreeMap.getElem?_insert]
    simp only [Rho.find, List.cons_append, List.lookup] at ih ⊢
    by_cases hx : x = p.1
    · subst hx; simp
    · have h1 : (compare p.1 x = .eq) = False := by
        simp only [Nat.compare_eq_eq, eq_iff_iff, iff_false]
        exact fun e => hx e.symm
      have h2 : (x == p.1) = false := beq_eq_false_iff_ne.2 hx
      simp only [h1, h2, ↓reduceIte]
      exact ih

theorem push_img (ix : AlphaIndex) (ρ : Rho) (h : ∀ y, ix.img.contains y = ρ.any (fun p => p.2 == y)) :
    ∀ (ps : Rho) (y : ValueId), (ix.push ps).img.contains y = (ps ++ ρ).any (fun p => p.2 == y)
  | [], y => by simpa [push] using h y
  | p :: ps, y => by
    have ih := push_img ix ρ h ps y
    simp only [push, List.foldr_cons] at ih ⊢
    rw [Std.TreeSet.contains_insert, ih, cmp_beq]
    simp

theorem push_inv {ρ : Rho} {ix : AlphaIndex} (h : Inv ρ ix) (ps : Rho) : Inv (ps ++ ρ) (ix.push ps) :=
  ⟨push_fwd ix ρ h.1 ps, push_img ix ρ h.2 ps⟩

def empty : AlphaIndex := ⟨{}, {}⟩

theorem empty_inv : Inv [] empty :=
  ⟨fun x => by simp [empty, Rho.find], fun y => by simp [empty]⟩

def ofRho (ρ : Rho) : AlphaIndex := empty.push ρ

theorem ofRho_inv (ρ : Rho) : Inv ρ (ofRho ρ) := by
  simpa [ofRho] using push_inv empty_inv ρ

/-- `freshIds` with the image looked up in the index. -/
def fresh (ix : AlphaIndex) (ys : List ValueId) : Bool :=
  decide ys.Nodup && ys.all (fun y => !ix.img.contains y)

theorem all_not_any (ρ : Rho) (y : ValueId) :
    ρ.all (fun p => p.2 != y) = !ρ.any (fun p => p.2 == y) := by
  induction ρ with
  | nil => rfl
  | cons p ρ ih =>
    rw [List.all_cons, List.any_cons, ih]
    cases h : (p.2 == y) <;> simp [bne, h]

theorem fresh_eq {ρ : Rho} {ix : AlphaIndex} (h : Inv ρ ix) (ys : List ValueId) :
    fresh ix ys = freshIds ρ ys := by
  simp only [fresh, freshIds]
  congr 1
  congr 1
  funext y
  rw [h.2 y, all_not_any]

theorem find_eq {ρ : Rho} {ix : AlphaIndex} (h : Inv ρ ix) : (fun x => ix.fwd[x]?) = ρ.find :=
  funext h.1

end AlphaIndex

theorem alphaOp_extends {ρ ρ' : Rho} {s t : Op} (h : alphaOp ρ s t = some ρ') :
    ρ' = (s.results.map (·.id)).zip (t.results.map (·.id)) ++ ρ := by
  cases s with
  | mk nS resS opsS aS pS regsS uS =>
    cases t with
    | mk nT resT opsT aT pT regsT uT =>
      unfold alphaOp at h
      split at h
      · simp only [Option.some.injEq] at h
        exact h.symm
      · cases h

mutual
  def alphaOpsF (ρ : Rho) (ix : AlphaIndex) : List Op → List Op → Bool
    | [], [] => true
    | s :: ss, t :: ts =>
      match alphaOpF ρ ix s t with
      | some (ρ', ix') => alphaOpsF ρ' ix' ss ts
      | none => false
    | _, _ => false

  def alphaOpF (ρ : Rho) (ix : AlphaIndex) : Op → Op → Option (Rho × AlphaIndex)
    | s@(.mk _ resS opsS _ _ regsS _), t@(.mk _ resT opsT _ _ regsT _) =>
      if opHeadEq s t && opsS.mapM (fun x => ix.fwd[x]?) == some opsT && alphaRegionsF ρ ix regsS regsT &&
          ix.fresh (resT.map (·.id))
      then
        let ps := (resS.map (·.id)).zip (resT.map (·.id))
        some (ps ++ ρ, ix.push ps)
      else none

  def alphaRegionsF (ρ : Rho) (ix : AlphaIndex) : List Region → List Region → Bool
    | [], [] => true
    | r :: rs, r' :: rs' => alphaRegionF ρ ix r r' && alphaRegionsF ρ ix rs rs'
    | _, _ => false

  def alphaRegionF (ρ : Rho) (ix : AlphaIndex) : Region → Region → Bool
    | .mk [.mk argsS opsS], .mk [.mk argsT opsT] =>
      let ps := (argsS.map (·.id)).zip (argsT.map (·.id))
      argsS.map (·.ty) == argsT.map (·.ty) && ix.fresh (argsT.map (·.id)) &&
        alphaOpsF (ps ++ ρ) (ix.push ps) opsS opsT
    | .mk [], .mk [] => true
    | _, _ => false
end

mutual
  theorem alphaOpsF_eq : ∀ (ρ : Rho) (ix : AlphaIndex) (ss ts : List Op), ix.Inv ρ →
      alphaOpsF ρ ix ss ts = alphaOps ρ ss ts
    | _, _, [], [], _ => by simp [alphaOpsF, alphaOps]
    | ρ, ix, s :: ss, t :: ts, h => by
      simp only [alphaOpsF, alphaOps]
      rw [alphaOpF_eq ρ ix s t h]
      cases ha : alphaOp ρ s t with
      | none => rfl
      | some ρ' =>
        simp only [Option.map_some]
        have hρ := alphaOp_extends ha
        subst hρ
        exact alphaOpsF_eq _ _ ss ts (AlphaIndex.push_inv h _)
    | _, _, [], _ :: _, _ => by simp [alphaOpsF, alphaOps]
    | _, _, _ :: _, [], _ => by simp [alphaOpsF, alphaOps]

  theorem alphaOpF_eq : ∀ (ρ : Rho) (ix : AlphaIndex) (s t : Op), ix.Inv ρ →
      alphaOpF ρ ix s t = (alphaOp ρ s t).map
        (fun ρ' => (ρ', ix.push ((s.results.map (·.id)).zip (t.results.map (·.id)))))
    | ρ, ix, .mk nS resS opsS aS pS regsS uS, .mk nT resT opsT aT pT regsT uT, h => by
      simp only [alphaOpF, alphaOp, AlphaIndex.find_eq h, AlphaIndex.fresh_eq h,
        alphaRegionsF_eq ρ ix regsS regsT h, Op.results]
      split <;> rfl

  theorem alphaRegionsF_eq : ∀ (ρ : Rho) (ix : AlphaIndex) (rs rs' : List Region), ix.Inv ρ →
      alphaRegionsF ρ ix rs rs' = alphaRegions ρ rs rs'
    | _, _, [], [], _ => by simp [alphaRegionsF, alphaRegions]
    | ρ, ix, r :: rs, r' :: rs', h => by
      simp only [alphaRegionsF, alphaRegions, alphaRegionF_eq ρ ix r r' h,
        alphaRegionsF_eq ρ ix rs rs' h]
    | _, _, [], _ :: _, _ => by simp [alphaRegionsF, alphaRegions]
    | _, _, _ :: _, [], _ => by simp [alphaRegionsF, alphaRegions]

  theorem alphaRegionF_eq : ∀ (ρ : Rho) (ix : AlphaIndex) (r r' : Region), ix.Inv ρ →
      alphaRegionF ρ ix r r' = alphaRegion ρ r r'
    | ρ, ix, .mk [.mk argsS opsS], .mk [.mk argsT opsT], h => by
      simp only [alphaRegionF, alphaRegion, AlphaIndex.fresh_eq h,
        alphaOpsF_eq _ _ opsS opsT (AlphaIndex.push_inv h _)]
    | _, _, .mk [], .mk [], _ => by simp [alphaRegionF, alphaRegion]
    | _, _, .mk [], .mk (_ :: _), _ => by simp [alphaRegionF, alphaRegion]
    | _, _, .mk (_ :: _), .mk [], _ => by simp [alphaRegionF, alphaRegion]
    | _, _, .mk (_ :: _ :: _), .mk [_], _ => by simp [alphaRegionF, alphaRegion]
    | _, _, .mk [_], .mk (_ :: _ :: _), _ => by simp [alphaRegionF, alphaRegion]
    | _, _, .mk (_ :: _ :: _), .mk (_ :: _ :: _), _ => by simp [alphaRegionF, alphaRegion]
end

/-- `alphaOps` with the index built once from `ρ`. -/
def alphaOpsFast (ρ : Rho) (ss ts : List Op) : Bool := alphaOpsF ρ (AlphaIndex.ofRho ρ) ss ts

@[csimp] theorem alphaOps_eq_fast : @alphaOps = @alphaOpsFast := by
  funext ρ ss ts
  exact (alphaOpsF_eq ρ _ ss ts (AlphaIndex.ofRho_inv ρ)).symm

end MerlinVerify.Check
