/- Pointwise relations learned from actual rank-one constructor execution. A relation depends
only on its tensor and selected scalar; unrelated original operands are not stale dependencies. -/
import MerlinVerify.Check.BindingFacts
import MerlinVerify.Rules.FromElementsKernel

namespace MerlinVerify.Check.FromElementsFacts

open MerlinVerify MerlinVerify.Sem.Ops

structure Fact where
  tensor : ValueId
  scalar : ValueId
  coordinate : Nat
  size : Nat
  element : Ty
  deriving DecidableEq

def Holds (f : Fact) (env : Env) : Prop :=
  ∃ t value, env.get? f.tensor = some (.tensor t) ∧ env.get? f.scalar = some (.scalar value) ∧
    t.shape = [f.size] ∧ t.elt = f.element ∧ f.coordinate < f.size ∧ t.get [f.coordinate] = value

def model : BindingFacts.Model Fact where
  dependencies f := [f.tensor, f.scalar]
  holds := Holds
  congr f env env' h hf := by
    obtain ⟨t, value, ht, hv, hs, he, hb, hi⟩ := hf
    exact ⟨t, value, (h f.tensor (by simp)).symm.trans ht,
      (h f.scalar (by simp)).symm.trans hv, hs, he, hb, hi⟩

/-- Indexed traversal allocates one constant-size relation per input without rescanning the
operand list. A self-shadowing result cannot retain the overwritten scalar binding. -/
def learn (s : Op) : List Fact :=
  if s.name = "tensor.from_elements" ∧ unmodelledAttr? s.blind = none ∧
      s.regions = [] ∧ s.successors = [] then
    match s.results, Sem.Ops.Tensor.resultTensorTy s with
    | [result], some ([size], element) =>
      if s.operands.length = size then
        s.operands.zipIdx.filterMap fun (scalar, coordinate) =>
          if scalar ≠ result.id then some ⟨result.id, scalar, coordinate, size, element⟩ else none
      else []
    | _, _ => []
  else []

theorem getAll_lookup {env : Env} : ∀ {ids : List ValueId} {values : List RVal},
    env.getAll ids = .ok values → ∀ {coordinate : Nat} {id : ValueId} {value : RVal},
    ids[coordinate]? = some id → values[coordinate]? = some value → env.get? id = some value
  | [], _, _, _, _, _, h, _ => by simp at h
  | first :: rest, values, h, coordinate, id, value, hid, hv => by
    change (Outcome.ofOption "unbound SSA value" (env.get? first)).bind
      (fun v => (env.getAll rest).bind fun vs => .ok (v :: vs)) = .ok values at h
    cases hf : env.get? first with
    | none => simp [hf, Outcome.ofOption] at h
    | some head =>
      cases ht : env.getAll rest with
      | ub => simp [hf, ht, Outcome.ofOption] at h
      | stuck reason => simp [hf, ht, Outcome.ofOption] at h
      | ok tail =>
        simp only [hf, Outcome.ofOption, Outcome.bind_ok, ht, Outcome.ok.injEq] at h
        subst values
        cases coordinate with
        | zero => simpa using (Option.some.inj hid ▸ Option.some.inj hv ▸ hf)
        | succ coordinate => exact getAll_lookup ht hid hv

theorem learn_sound {s : Op} {f : Fact} (h : f ∈ learn s)
    {ctx : Ctx} {call : CallSem} {env env' : Env}
    (hs : stepOp opSem ctx call env s = .ok env') : Holds f env' := by
  unfold learn at h
  split at h
  · next hc =>
    obtain ⟨hn, hm, hregs, _⟩ := hc
    split at h
    · next result size element hr ht =>
      split at h
      · next hcount =>
        obtain ⟨⟨scalar, coordinate⟩, hpos, hfact⟩ := List.mem_filterMap.1 h
        dsimp only at hfact
        by_cases hne : scalar ≠ result.id
        · rw [ite_eq_left hne] at hfact
          cases hfact
          have hid := List.mem_zipIdx_iff_getElem?.1 hpos
          have hbound : coordinate < size := by
            have hlt := (List.mem_zipIdx' hpos).1
            simpa [hcount] using hlt
          unfold stepOp at hs
          rw [hregs] at hs
          simp only [evalRegions] at hs
          cases hg : env.getAll s.operands with
          | ub => simp [hg] at hs
          | stuck reason => simp [hg] at hs
          | ok args =>
            simp only [hg, Outcome.bind_ok] at hs
            cases hv : opSem ctx call s args [] with
            | ub => simp [hv] at hs
            | stuck reason => simp [hv] at hs
            | ok values =>
              obtain ⟨scalars, hargs, hlen, hvals⟩ := Rules.FromElements.constructor_success hn hm ht hv
              subst args values
              have hindex : coordinate < scalars.length := by omega
              have hp : env.get? scalar = some (.scalar scalars[coordinate]) := by
                apply getAll_lookup hg hid
                simp [List.getElem?_map, List.getElem?_eq_getElem hindex]
              have hpost : env' = (result.id, .tensor (Rules.FromElements.tensor element scalars)) :: env := by
                simpa [hv, Env.bindAll, hr] using hs.symm
              rw [hpost]
              refine ⟨_, scalars[coordinate], ?_, ?_, ?_, rfl, hbound,
                Rules.FromElements.tensor_get hindex⟩
              · simp [Env.get?, List.lookup]
              · simpa [Env.get?, List.lookup, show (scalar == result.id) = false by simpa using hne] using hp
              · simp [Rules.FromElements.tensor, Tensor.canon, hlen]
        · simp [hne] at hfact
      · cases h
    · cases h
  · cases h

def advance (facts : List Fact) (s : Op) : List Fact :=
  learn s ++ BindingFacts.forget model (s.results.map (·.id)) facts

theorem advance_sound {facts : List Fact} {s : Op} {ctx : Ctx} {call : CallSem} {env env' : Env}
    (hf : BindingFacts.Valid model facts env) (hs : stepOp opSem ctx call env s = .ok env') :
    BindingFacts.Valid model (advance facts s) env' := by
  intro f h
  rcases List.mem_append.1 h with hl | hr
  · exact learn_sound hl hs
  · exact BindingFacts.forget_step model hf hs f hr

end MerlinVerify.Check.FromElementsFacts
