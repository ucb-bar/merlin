/- Runtime bridges for the bounded identity-map parallel fusion checker. Pointwise values and
shapes are derived from actual tensor values; syntax checks alone cannot establish these facts. -/
import MerlinVerify.Check.ParallelFusionBody
import MerlinVerify.Check.RulesLinalg
import MerlinVerify.Rules.ParallelFusion

namespace MerlinVerify.Check.ParallelFusionRuntime

open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg
open ParallelFusionBody MerlinVerify.Rules.ParallelFusionScalar

theorem bodyAt_identity {ctx : Ctx} {input init : Tensor} {shape pt : List Nat}
    {body : Ctx → RegionSem} {value : Val}
    (hi : input.shape = shape) (ho : init.shape = shape) (hp : Idx.InBounds shape pt)
    (hb : body {ctx with loopIndex := pt} [.scalar (input.get pt), .scalar (init.get pt)] =
      .ok [.scalar value]) :
    bodyAt ctx [.tensor input] [identityMap shape.length] [identityMap shape.length]
      [init] body pt = .ok [value] := by
  have hri : readAt (.tensor input) (identityMap shape.length) pt = .ok (input.get pt) := by
    simpa [hi] using readAt_identity (t := input) (hi.symm ▸ hp)
  have hro : readAt (.tensor init) (identityMap shape.length) pt = .ok (init.get pt) := by
    simpa [ho] using readAt_identity (t := init) (ho.symm ▸ hp)
  simp [bodyAt, hri, hro, hb, Outcome.bind_ok]

theorem repeated_point {ib w : Nat} {p : Primitive} {inputArg oldArg : Value} {term : Op}
    {input init : Tensor} {shape pt : List Nat}
    (hv : p.Valid w) (hop : p.operation.operands = [inputArg.id, inputArg.id])
    (htn : term.name = "linalg.yield") (hto : term.operands = [p.result.id])
    (hi : input.shape = shape) (ho : init.shape = shape) (hp : Idx.InBounds shape pt)
    (hval : Val.HasTy ib (.int w) (input.get pt)) (ctx : Ctx) (call : CallSem) (env : Env) :
    bodyAt ctx [.tensor input] [identityMap shape.length] [identityMap shape.length] [init]
      (fun nested => evalRegion opSem nested call env
        (.mk [.mk [inputArg, oldArg] [p.operation, term]])) pt =
      .ok [p.kind.value w (input.get pt) (input.get pt)] := by
  apply bodyAt_identity hi ho hp
  obtain ⟨hd, hn, hm, hr, ht, hf, hregs⟩ := hv
  exact repeated_body p.kind hd hn hm hr ht hop hf hregs htn hto hval _ call env

theorem fused_point {ib w : Nat} {p c : Primitive} {inputArg oldArg : Value} {term : Op}
    {input init : Tensor} {shape pt : List Nat}
    (hvP : p.Valid w) (hvC : c.Valid w)
    (hop : p.operation.operands = [inputArg.id, inputArg.id])
    (htn : term.name = "linalg.yield") (hto : term.operands = [c.result.id])
    (hi : input.shape = shape) (ho : init.shape = shape) (hp : Idx.InBounds shape pt)
    (hval : Val.HasTy ib (.int w) (input.get pt)) (ctx : Ctx) (call : CallSem) (env : Env) :
    bodyAt ctx [.tensor input] [identityMap shape.length] [identityMap shape.length] [init]
      (fun nested => evalRegion opSem nested call env
        (.mk [.mk [inputArg, oldArg]
          [p.operation, withOperands c.operation [p.result.id, p.result.id], term]])) pt =
      .ok [c.kind.value w (p.kind.value w (input.get pt) (input.get pt))
        (p.kind.value w (input.get pt) (input.get pt))] := by
  apply bodyAt_identity hi ho hp
  exact fused_body hvP hvC hop htn hto hval _ call env

/-- The complete original generic header is checked against this layout. Region meanings remain
actual closure arguments to opSem; no operation is replaced by an assumed semantic summary. -/
def Header (op : Op) (shape : List Nat) (result : Value) : Prop :=
  op.name = "linalg.generic" ∧ unmodelledAttr? op.blind = none ∧
  attrDenseArray? op "operandSegmentSizes" = some [1, 1] ∧
  attrAffineMaps? op "indexing_maps" = some [identityMap shape.length, identityMap shape.length] ∧
  iterKinds? op = some (List.replicate shape.length .parallel) ∧ op.results = [result]

theorem generic_success {op : Op} {result : Value} {shape : List Nat} {input init output : Tensor}
    {ctx : Ctx} {call : CallSem} {body : Ctx → RegionSem}
    (hh : Header op shape result)
    (hs : genericSem ctx [identityMap shape.length, identityMap shape.length]
      (List.replicate shape.length .parallel) [.tensor input] [.tensor init] body =
        .ok [.tensor output]) :
    opSem ctx call op [.tensor input, .tensor init] [body] = .ok [.tensor output] := by
  obtain ⟨hn, hm, hseg, hmaps, hiters, hr⟩ := hh
  have hd : dialectOf op.name = "linalg" := by rw [hn]; cbv
  have hseg' : attrDenseArray? op.blind "operandSegmentSizes" = some [1, 1] := by
    simpa [attrDenseArray?, Op.blind_attr?] using hseg
  rw [linalg_sem_of hm hd (.inl hn)]
  simp [Linalg.sem, Op.blind_name, hn, splitInsOuts, hseg',
    attrAffineMaps?_blind, hmaps, iterKinds?_blind, hiters, hs, Op.blind_results, hr]

/-- Extract an actual tensor and its element facts from the source environment. -/
theorem tensor_lookup {ib : Nat} {Γ : TEnv} {env : Env} {id : ValueId} {shape : List Nat} {elt : Ty}
    (he : EnvTyped ib Γ env) (hl : Γ.lookup id = some (.tensor shape elt)) :
    ∃ t : Tensor, env.get? id = some (.tensor t) ∧ t.shape = shape ∧ t.elt = elt ∧
      ∀ pt, Idx.InBounds shape pt → Val.HasTy ib elt (t.get pt) := by
  obtain ⟨v, hv, ht⟩ := he id _ hl
  cases v with
  | scalar _ => cases ht
  | tensor t => exact ⟨t, hv, ht⟩

theorem staticTensor_of_vty {ib : Nat} {ty elt : Ty} {shape : List Nat}
    (h : vtyOf ib ty = some (.tensor shape elt)) : ty.staticTensor? = some (shape, elt) := by
  unfold vtyOf at h
  split at h
  · next dims element =>
    split at h
    · cases hm : dims.mapM id with
      | none => simp [hm] at h
      | some actual =>
        simp only [hm, Option.map_some, Option.some.injEq, VTy.tensor.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        simp [Ty.staticTensor?, hm]
    · cases h
  · split at h <;> cases h

end MerlinVerify.Check.ParallelFusionRuntime
