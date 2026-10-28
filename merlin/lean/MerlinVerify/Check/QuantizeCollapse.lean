/- Recover a checked source quantizer from its actual six-operation expansion.
This exact typed normalizer lets the existing region traversal handle nested sites.
Temporary values may not escape into the continuation, including captured uses. -/
import MerlinVerify.Check.StateNormalize
import MerlinVerify.Rules.QuantizeRule

namespace MerlinVerify.Check.QuantizeCollapse

open MerlinVerify MerlinVerify.Sem.Ops
open Rules.QuantizeSite

def identities (Γ : TEnv) : Rho := Γ.map fun p => (p.1, p.1)

theorem identities_find : ∀ {Γ : TEnv} {x y : Nat}, (identities Γ).find x = some y → x = y
  | [], _, _, h => by simp [identities, Rho.find, List.lookup] at h
  | (key, _) :: Γ, x, y, h => by
    simp only [identities, List.map_cons, Rho.find, List.lookup] at h
    split at h
    · next same =>
      have hx : x = key := by simpa using same
      exact hx.trans (Option.some.inj h)
    · exact identities_find h

theorem identities_corr (Γ : TEnv) (env : Env) : Corr (identities Γ) env env := by
  intro x y found
  rw [identities_find found]

def inputTypes (s : Site) (w : Nat) : List VTy :=
  [.tensor s.c.shape (.float .f32), .tensor [] (.float .f32), .tensor [] (.int w)]

def Profile (Γ : TEnv) (s : Site) (w : Nat) : Prop :=
  (w = 32 ∨ w = 64) ∧ s.src.operands.mapM (Γ.lookup ·) = some (inputTypes s w)

instance (Γ : TEnv) (s : Site) (w : Nat) : Decidable (Profile Γ s w) := by
  unfold Profile
  infer_instance

theorem typed_run {ib w : Nat} {s : Site} {args : List RVal} {M : FloatModel}
    (width : w = 32 ∨ w = 64) (typed : HasTys ib args (inputTypes s w)) :
    ∃ data scale zero, args = [.tensor data, .tensor scale, .tensor zero] ∧
      Quantize.validInputs w s.c data scale zero = true ∧
      Quantize.run s.c M args = .ok [.tensor (Quantize.result w s.c M data scale zero)] := by
  cases typed with
  | cons hx rest =>
    cases rest with
    | cons hs rest =>
      cases rest with
      | cons hz rest =>
        cases rest
        rename_i xv sv zv
        cases xv with
        | scalar _ => exact (hx : False).elim
        | tensor data =>
          cases sv with
          | scalar _ => exact (hs : False).elim
          | tensor scale =>
            cases zv with
            | scalar _ => exact (hz : False).elim
            | tensor zero =>
              have valid := Rules.QuantizeSource.validInputs_of_typing hx hs hz width
              refine ⟨data, scale, zero, rfl, valid, ?_⟩
              rcases width with rfl | rfl
              · simp only [Quantize.run, valid, ↓reduceIte]
              · have wrong : Quantize.validInputs 32 s.c data scale zero = false := by
                  simp [Quantize.validInputs, Dequant.tensorTyped, hz.2.1]
                simp only [Quantize.run, wrong, valid, Bool.false_eq_true, ↓reduceIte]

theorem collapse_sound {ib : Nat} {Γ : TEnv} {s : Site} {w : Nat} {tail : List Op}
    (checked : Checks (identities Γ) s) (profile : Profile Γ s w)
    (continuation : alphaOps ((s.rs.id, s.rg.id) :: identities Γ) tail tail = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (typed : EnvTyped ib Γ env) :
    evalOps opSem ctx call env (s.src :: tail) =
      evalOps opSem ctx call env (targets s ++ tail) := by
  obtain ⟨args, fetch, hargs⟩ := typed.getAll _ _ profile.2
  obtain ⟨data, scale, zero, rfl, valid, run⟩ := typed_run profile.1 hargs
  obtain ⟨envT, executed, related⟩ := target_sound checked ctx call env env
    (identities_corr Γ env) w data scale zero fetch valid
  have headers := checked.2.1
  have header (op : Op) (member : op ∈ s.src :: targets s) :
      op.isTerminator = false ∧ unmodelledAttr? op.blind = none := by
    have h := List.all_eq_true.1 headers op member
    simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_true_eq] at h
    exact ⟨h.1.1, h.1.2⟩
  obtain ⟨results, _, _, _, _, _, _, regions, _⟩ := frameCheck_sound checked.1
  have sourceStep : stepOp opSem ctx call env s.src =
      .ok ((s.rs.id, .tensor (Quantize.result w s.c ctx.M data scale zero)) :: env) := by
    apply step_result results fetch
    rw [regions]
    simp only [evalRegions]
    rw [Rules.QuantizeSource.source_sem ctx call _ (header _ (List.mem_cons_self ..)).2 checked.2.2.1]
    exact run
  rw [evalOps_cons_step _ _ _ _ _ _ (header _ (List.mem_cons_self ..)).1,
    sourceStep, Outcome.bind_ok,
    evalOps_append _ _ _ _ _ _ (fun op member => (header op (List.mem_cons_of_mem _ member)).1),
    executed, Outcome.bind_ok]
  exact alphaOps_sound call call _ tail tail ctx _ _ continuation related (fun _ _ => rfl)

def properties : List (String × Attr) :=
  [("output_dtype", .str "int8"), ("quant_max", .int 127 (.int 64)),
   ("quant_min", .int (-128) (.int 64))]

def canonical (s : Op) : Op :=
  .mk s.name s.results s.operands [] properties s.regions s.successors

def canonicalize (s : Op) : Op :=
  match Quantize.config? s with
  | some c =>
    if s.regions = [] ∧ unmodelledAttr? s.blind = none ∧
        unmodelledAttr? (canonical s).blind = none ∧ Quantize.config? (canonical s) = some c then
      canonical s else s
  | none => s

theorem canonicalize_step (s : Op) (ctx : Ctx) (call : CallSem) (env : Env) :
    stepOp opSem ctx call env (canonicalize s) = stepOp opSem ctx call env s := by
  unfold canonicalize
  split
  · next c configured =>
    split
    · next facts =>
      obtain ⟨regions, modelled, modelledQ, configuredQ⟩ := facts
      unfold stepOp
      have sameOperands : (canonical s).operands = s.operands := rfl
      have sameResults : (canonical s).results = s.results := rfl
      have sameRegions : (canonical s).regions = [] := by
        change s.regions = []
        exact regions
      rw [sameOperands, regions, sameRegions]
      simp only [evalRegions]
      simp_rw [Rules.QuantizeSource.source_sem ctx call _ modelled configured,
        Rules.QuantizeSource.source_sem ctx call _ modelledQ configuredQ, sameResults]
    · rfl
  · rfl

def proposal (Γ : TEnv) : List Op → Option Site
  | invEmpty :: inverse :: empty :: lower :: upper :: generic :: rest => do
    let [x, _, zeroId, _] := generic.operands | none
    let [scale, _] := inverse.operands | none
    let src := Op.mk "quant_ext.quantize_per_tensor" generic.results [x, scale, zeroId] [] properties [] []
    Rules.QuantizeRule.propose (identities Γ) [src]
      (invEmpty :: inverse :: empty :: lower :: upper :: generic :: rest)
  | _ => none

def attempt (Γ : TEnv) (ops : List Op) : Option (List Op) := do
  let s ← proposal Γ ops
  let tys ← s.src.operands.mapM (Γ.lookup ·)
  let w := if tys = inputTypes s 32 then 32 else 64
  if decide (Checks (identities Γ) s ∧ Profile Γ s w) &&
      ExactSyntax.operations (ops.take 6) (targets s) &&
      alphaOps ((s.rs.id, s.rg.id) :: identities Γ) (ops.drop 6) (ops.drop 6) then
    some (s.src :: ops.drop 6)
  else none

theorem attempt_sound {ib : Nat} {Γ : TEnv} {ops normalized : List Op}
    (accepted : attempt Γ ops = some normalized) (ctx : Ctx) (call : CallSem) (env : Env)
    (typed : EnvTyped ib Γ env) :
    evalOps opSem ctx call env normalized = evalOps opSem ctx call env ops := by
  unfold attempt at accepted
  cases proposed : proposal Γ ops with
  | none => simp [proposed] at accepted
  | some s =>
    simp only [proposed, bind, Option.bind] at accepted
    cases types : s.src.operands.mapM (Γ.lookup ·) with
    | none => simp [types] at accepted
    | some tys =>
      simp only [types, bind, Option.bind] at accepted
      generalize chosen : (if tys = inputTypes s 32 then 32 else 64) = w at accepted
      split at accepted
      · next facts =>
        simp only [Bool.and_eq_true, decide_eq_true_eq] at facts
        obtain ⟨⟨⟨checked, profile⟩, frame⟩, continuation⟩ := facts
        cases accepted
        have same := ExactSyntax.operations_sound _ _ frame
        rw [← List.take_append_drop 6 ops, same]
        exact collapse_sound checked profile continuation ctx call env typed
      · cases accepted

def finish (Γ : TEnv) (s : Op) (tail : List Op) : List Op :=
  (attempt Γ (canonicalize s :: tail)).getD (canonicalize s :: tail)

theorem finish_sound {ib : Nat} (Γ : TEnv) (s : Op) (tail : List Op)
    (ctx : Ctx) (call : CallSem) (env : Env) (typed : EnvTyped ib Γ env)
    (nonterminal : s.isTerminator = false) :
    evalOps opSem ctx call env (finish Γ s tail) = evalOps opSem ctx call env (s :: tail) := by
  unfold finish
  have term : (canonicalize s).isTerminator = false := by
    unfold canonicalize
    split
    · split
      · simpa only [canonical, Op.isTerminator, Op.name] using nonterminal
      · exact nonterminal
    · exact nonterminal
  have same : evalOps opSem ctx call env (canonicalize s :: tail) =
      evalOps opSem ctx call env (s :: tail) := by
    rw [evalOps_cons_step _ _ _ _ _ _ term,
      evalOps_cons_step _ _ _ _ _ _ nonterminal, canonicalize_step]
  cases h : attempt Γ (canonicalize s :: tail) with
  | none => exact same
  | some normalized => exact (attempt_sound h ctx call env typed).trans same

def family : StateNormalize.Family Unit (fun _ _ => True) where
  initial := ()
  initial_valid _ := trivial
  advance _ _ _ _ := ()
  advance_sound _ _ _ _ _ _ _ _ _ _ _ _ := trivial
  finish _ Γ _ _ s tail := finish Γ s tail
  finish_sound _ Γ _ _ s tail ctx call env nonterminal typed _ _ :=
    finish_sound Γ s tail ctx call env typed nonterminal

def base (ib : Nat) : SplatNormalizer.Normalizer ib := StateNormalize.base family ib []

#print axioms collapse_sound
#print axioms attempt_sound

end MerlinVerify.Check.QuantizeCollapse
