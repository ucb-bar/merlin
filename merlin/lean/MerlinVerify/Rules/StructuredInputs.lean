/- Exact structured-loop rewriting by equality of body inputs over the checked iteration box.
This proof preserves the actual body, output accesses and lexicographic reduction order. -/
import MerlinVerify.Check.StructuredRuntime

namespace MerlinVerify.Rules.StructuredInputs

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops.Linalg

def reads (ins : List RVal) (maps : List AffineMap) (pt : Idx) : Outcome (List Val) :=
  (ins.zip maps).mapM fun (v, m) => readAt v m pt

theorem foldlM_congr {α β : Type} (f g : β → α → Outcome β) :
    ∀ (xs : List α), (∀ x ∈ xs, ∀ state, f state x = g state x) →
      ∀ state, xs.foldlM f state = xs.foldlM g state
  | [], _, _ => rfl
  | x :: xs, h, state => by
    rw [List.foldlM_cons, List.foldlM_cons, h x (List.mem_cons_self ..) state]
    cases hg : g state x with
    | ub => rfl
    | stuck _ => rfl
    | ok next =>
      exact foldlM_congr f g xs (fun x hx => h x (List.mem_cons_of_mem _ hx)) next

/-- Equality of input reads suffices even for bodies that yield poison, fail, or use floats.
The body receives exactly the same values and loop coordinates, with identical output tensors. -/
theorem loopStep_congr {ctx : Ctx} {insS insT : List RVal} {mapsS mapsT outMaps : List AffineMap}
    {body : Ctx → RegionSem} {pt : Idx}
    (h : reads insS mapsS pt = reads insT mapsT pt) (cur : List Tensor) :
    loopStep ctx insS mapsS outMaps body cur pt =
      loopStep ctx insT mapsT outMaps body cur pt := by
  unfold reads at h
  unfold loopStep bodyAt
  rw [h]

/-- Two independently checked geometries with the same iteration box and output maps execute
identically when every input element presented to the retained body is equal. No float laws,
initialization assumptions or parallelization argument are used. -/
theorem genericSem_congr {ib : Nat} {mapsS mapsT : List AffineMap} {iters : List IterKind}
    {tysS tysT outTys : List VTy} {ranges : List Nat} {insS insT : List RVal} {outs : List Tensor}
    (hs : StructuredOk mapsS iters.length (tysS ++ outTys) ranges)
    (ht : StructuredOk mapsT iters.length (tysT ++ outTys) ranges)
    (hiS : HasTys ib insS tysS) (hiT : HasTys ib insT tysT)
    (ho : HasTys ib (outs.map RVal.tensor) outTys)
    (houts : mapsS.drop insS.length = mapsT.drop insT.length)
    (hreads : ∀ pt, Idx.InBounds ranges pt →
      reads insS (mapsS.take insS.length) pt = reads insT (mapsT.take insT.length) pt)
    (ctx : Ctx) (body : Ctx → RegionSem) :
    genericSem ctx mapsS iters insS (outs.map RVal.tensor) body =
      genericSem ctx mapsT iters insT (outs.map RVal.tensor) body := by
  rw [genericSem_execute hs hiS ho, genericSem_execute ht hiT ho, houts]
  congr 1
  apply foldlM_congr
  intro pt hp cur
  exact loopStep_congr (hreads pt ((mem_allPoints ranges pt).1 hp)) cur

end MerlinVerify.Rules.StructuredInputs
