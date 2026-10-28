/-
Defined outputs, independently of operation support. A defined execution returns successfully
and every observable result element is nonpoison. Input domains must say explicitly whether they
require nonpoison inputs: `HasTys` alone permits poison.
-/
import MerlinVerify.Sem.Program

namespace MerlinVerify

/-- Every observable element of a runtime value is nonpoison. Out-of-bounds tensor elements are
not observable and impose no obligation. -/
def RVal.Defined : RVal → Prop
  | .scalar v => v ≠ .poison
  | .tensor t => ∀ i, Idx.InBounds t.shape i → t.get i ≠ .poison

def ResultsDefined (vs : List RVal) : Prop := ∀ v ∈ vs, v.Defined

/-- On the explicitly stated input domain, execution succeeds with nonpoison results. This excludes
both `ub` and `stuck`; it is stronger than `FuncNotStuck`. -/
def FuncDefined (sem : OpSem) (ctxOk : Ctx → Prop) (D : List RVal → Prop)
    (m : Module) (f : String) : Prop :=
  ∀ ctx, ctxOk ctx → ∀ args, D args →
    ∃ results, denote sem ctx m f args = .ok results ∧ ResultsDefined results

theorem RValRefines.defined {t s : RVal} (hr : RValRefines t s) (hs : s.Defined) : t.Defined := by
  cases t with
  | scalar t =>
    cases s with
    | scalar s =>
      rcases hr with hp | he
      · exact (hs hp).elim
      · simpa [RVal.Defined, he] using hs
    | tensor s => cases hr
  | tensor t =>
    cases s with
    | scalar s => cases hr
    | tensor s =>
      obtain ⟨hshape, _, hvals⟩ := hr
      intro i hi
      have hsi : Idx.InBounds s.shape i := hshape ▸ hi
      rcases hvals i hsi with hp | he
      · exact (hs i hsi hp).elim
      · simpa [he] using hs i hsi

theorem ResultsRefine.defined {t s : List RVal} (hr : ResultsRefine t s)
    (hs : ResultsDefined s) : ResultsDefined t := by
  induction hr with
  | nil => simp [ResultsDefined]
  | cons hv hrest ih =>
    intro v hm
    rcases List.mem_cons.1 hm with rfl | hm
    · exact hv.defined (hs _ (List.mem_cons_self ..))
    · exact ih (fun x hx => hs x (List.mem_cons_of_mem _ hx)) v hm

theorem FuncDefined.notStuck {sem : OpSem} {ctxOk : Ctx → Prop} {D : List RVal → Prop}
    {m : Module} {f : String} (h : FuncDefined sem ctxOk D m f) :
    FuncNotStuck sem ctxOk D m f := by
  intro ctx hc args ha
  obtain ⟨results, he, _⟩ := h ctx hc args ha
  rw [he]
  trivial

/-- Refinement of a defined source cannot introduce UB, stuck execution, or poison results. -/
theorem FuncRefines.defined {sem : OpSem} {ctxOk : Ctx → Prop} {D : List RVal → Prop}
    {t s : Module} {tf sf : String} (hr : FuncRefines sem ctxOk D t tf s sf)
    (hs : FuncDefined sem ctxOk D s sf) : FuncDefined sem ctxOk D t tf := by
  intro ctx hc args ha
  obtain ⟨results, he, hd⟩ := hs ctx hc args ha
  have h := hr ctx hc args ha
  rw [he] at h
  cases ht : denote sem ctx t tf args with
  | ub => simp [ht, Refines] at h
  | stuck r => simp [ht, Refines] at h
  | ok ts => exact ⟨ts, rfl, ResultsRefine.defined (by simpa [ht, Refines] using h) hd⟩

end MerlinVerify
