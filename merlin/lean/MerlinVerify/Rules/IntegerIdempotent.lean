/- Exact scalar integer idempotence with checked SSA-use substitution. Structured-region
traversal is supplied by BodyNormalize; this module only normalizes one checked block. -/
import MerlinVerify.Check.BodyNormalize
import MerlinVerify.Rules.Cse
import MerlinVerify.Check.RulesArith

namespace MerlinVerify.Rules.IntegerIdempotent

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def names : List String :=
  ["arith.andi", "arith.ori", "arith.maxsi", "arith.minsi", "arith.maxui", "arith.minui"]

/-- Equal typed scalar operands return precisely that scalar, including poison. Tensor aliases
require a separate representation proof; malformed scalar values are outside this checked domain. -/
theorem meaning {s : Op} {r : Value} {w : Nat} {value : Val}
    (hd : dialectOf s.name = "arith") (hn : s.name ∈ names)
    (hm : unmodelledAttr? s.blind = none) (hr : s.results = [r]) (ht : r.ty = .int w)
    (hf : overflowFlags s = []) (hv : Val.HasTy 0 (.int w) value)
    (ctx : Ctx) (call : CallSem) :
    opSem ctx call s [.scalar value, .scalar value] [] = .ok [.scalar value] := by
  rw [opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  have hf' : overflowFlags s.blind = [] := by simpa [overflowFlags, attrEnum?] using hf
  rcases hasTy_intWidth (ib := 0) (t := .int w) rfl hv with rfl | ⟨x, rfl⟩
  all_goals
    simp only [names, List.mem_cons, List.not_mem_nil, or_false] at hn
    rcases hn with hn | hn | hn | hn | hn | hn
    all_goals simp [Arith.sem, Op.blind_name, hn, Arith.binOp, Arith.pureN,
      Arith.resultElt, Op.blind_results, hr, ht, liftEager, Arith.intBin, Arith.bv]

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

def Legal (Γ : TEnv) (s : Op) (x : ValueId) (r : Value) (w : Nat) : Prop :=
  dialectOf s.name = "arith" ∧ s.name ∈ names ∧ unmodelledAttr? s.blind = none ∧
  s.operands = [x, x] ∧ s.results = [r] ∧ r.ty = .int w ∧
  Γ.lookup x = some (.scalar (.int w)) ∧ s.regions = [] ∧ s.successors = [] ∧ overflowFlags s = []

instance (Γ : TEnv) (s : Op) (x : ValueId) (r : Value) (w : Nat) :
    Decidable (Legal Γ s x r w) := by unfold Legal; infer_instance

/- Substitution is an untrusted syntax proposal. The alpha checker below validates all uses,
including captures, block arguments, rebinding and escaping values before the rewrite is used. -/
mutual
  def substituteOps (oldId newId : ValueId) : List Op → List Op
    | [] => []
    | s :: ss => substituteOp oldId newId s :: substituteOps oldId newId ss
  def substituteOp (oldId newId : ValueId) : Op → Op
    | .mk name results operands attrs props regions successors =>
      .mk name results (operands.map fun x => if x == oldId then newId else x) attrs props
        (substituteRegions oldId newId regions) successors
  def substituteRegions (oldId newId : ValueId) : List Region → List Region
    | [] => []
    | .mk blocks :: rest => .mk (substituteBlocks oldId newId blocks) :: substituteRegions oldId newId rest
  def substituteBlocks (oldId newId : ValueId) : List Block → List Block
    | [] => []
    | .mk args ops :: rest => .mk args (substituteOps oldId newId ops) :: substituteBlocks oldId newId rest
end

def candidate (s : Op) : Option (ValueId × Value × Nat) :=
  match s.operands, s.results with
  | [x, _], [r] => match r.ty with
    | .int w => some (x, r, w)
    | _ => none
  | _, _ => none

def eliminate (Γ : TEnv) (s : Op) (tail : List Op) : Option (List Op) := do
  let (x, r, w) ← candidate s
  let renamed := substituteOps r.id x tail
  if Legal Γ s x r w ∧ alphaOps ((r.id, x) :: SplatNormalize.diagonal Γ) tail renamed = true then
    some renamed
  else none

theorem eliminate_sound {ib : Nat} {Γ : TEnv} {s : Op} {tail renamed : List Op}
    (h : eliminate Γ s tail = some renamed) {ctx : Ctx} {call : CallSem} {env : Env}
    (he : EnvTyped ib Γ env) :
    s.isTerminator = false ∧ ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env renamed = evalOps opSem ctx call env' tail := by
  unfold eliminate at h
  cases hc : candidate s with
  | none => simp [hc] at h
  | some item =>
    obtain ⟨x, r, w⟩ := item
    simp only [hc, bind, Option.bind] at h
    split at h
    · next hchecks =>
      obtain ⟨hl, ha⟩ := hchecks
      obtain ⟨hd, hn, hm, ho, hr, ht, hlookup, hregs, _, hf⟩ := hl
      have hterm : s.isTerminator = false := by
        simp only [names, List.mem_cons, List.not_mem_nil, or_false] at hn
        rcases hn with hn | hn | hn | hn | hn | hn <;> simp [Op.isTerminator, terminatorNames, hn]
      cases h
      obtain ⟨value, hv, hty⟩ := he _ _ hlookup
      cases value with
      | tensor _ => cases hty
      | scalar value =>
        have hwidth : Val.HasTy 0 (.int w) value := by
          change Val.HasTy ib (.int w) value at hty
          cases value with
          | poison => trivial
          | some scalar => cases scalar <;> simp_all [Val.HasTy, Scalar.conforms]
        have hargs : env.getAll s.operands = .ok [.scalar value, .scalar value] := by
          simp [ho, Env.getAll, Outcome.mapM, hv, Outcome.ofOption]
        have hs : stepOp opSem ctx call env s = .ok ((r.id, .scalar value) :: env) := by
          unfold stepOp
          rw [hargs, Outcome.bind_ok, hregs]
          simp only [evalRegions]
          rw [meaning hd hn hm hr ht hf hwidth ctx call]
          simp [Env.bindAll, hr]
        refine ⟨hterm, _, hs, ?_⟩
        exact (alphaOps_sound call call _ tail (substituteOps r.id x tail) ctx _ _ ha
          (Cse.Corr.bindAlias (SplatNormalize.diagonal_refl Γ env) hv) (fun _ _ => rfl)).symm
    · cases h

def normalize (ib : Nat) (Γ : TEnv) : List Op → Option (List Op)
  | [] => some []
  | s :: ss =>
    if s.isTerminator then some (s :: ss) else do
      let _ ← s.operands.mapM (Γ.lookup ·)
      let rs ← s.results.mapM (fun r => vtyOf ib r.ty)
      let tail ← normalize ib ((s.results.map (·.id)).zip rs ++ Γ) ss
      pure ((eliminate Γ s tail).getD (s :: tail))

/-- Every eliminated operation is total on the actual typed source values. Checked alpha
substitution equates the entire continuation, including poison, captures and escaping results. -/
theorem normalize_sound {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize ib Γ ss = some ns →
      EnvTyped ib Γ env → ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss
  | _, [], _, _, _, _, _, h, _, _, _, _ => by simp [checkOps] at h
  | Γ, s :: ss, ns, yields, ctx, call, env, h, hn, he, hib, hd => by
    by_cases hterm : s.isTerminator = true
    · simp only [normalize, hterm, ↓reduceIte, Option.some.injEq] at hn
      subst ns; rfl
    · have ht : s.isTerminator = false := by simpa using hterm
      cases s with
      | mk name results operands attrs props regions successors =>
        simp only [checkOps] at h
        split at h
        · cases h
        · next tys hl =>
          rw [ite_eq_right hterm] at h
          split at h
          · cases h
          · next rs hr =>
            split at h
            · cases h
            · next sigs hR =>
              simp only [Bool.and_eq_true] at h
              obtain ⟨hregs, hrest⟩ := h
              simp only [normalize, ht, Bool.false_eq_true, ↓reduceIte, hl, hr,
                Op.operands, Op.results, bind, Option.bind] at hn
              cases htail : normalize ib ((results.map (·.id)).zip rs ++ Γ) ss with
              | none => simp [htail] at hn
              | some tail =>
                simp only [htail, Option.pure_def, Option.some.injEq] at hn
                subst ns
                have ih : ∀ env', stepOp opSem ctx call env
                    (.mk name results operands attrs props regions successors) = .ok env' →
                    evalOps opSem ctx call env' tail = evalOps opSem ctx call env' ss := by
                  intro env' hs
                  exact normalize_sound R d _ ss tail yields ctx call env' hrest htail
                    (step_envTyped R (op := .mk name results operands attrs props regions successors)
                      hl hr hR hregs hib hd he hs) hib hd
                cases helim : eliminate Γ (.mk name results operands attrs props regions successors) tail with
                | some renamed =>
                  simp only [Option.getD_some]
                  obtain ⟨_, env', hs, heq⟩ := eliminate_sound helim he (ctx := ctx) (call := call)
                  rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
                  exact heq.trans (ih env' hs)
                | none =>
                  simp only [Option.getD_none]
                  rw [evalOps_cons_step _ _ _ _ _ _ ht, evalOps_cons_step _ _ _ _ _ _ ht]
                  cases hs : stepOp opSem ctx call env
                      (.mk name results operands attrs props regions successors) with
                  | ub => rfl
                  | stuck _ => rfl
                  | ok env' => exact ih env' hs

def normalizer (ib : Nat) : SplatNormalizer.Normalizer ib where
  run _ Γ _ ops _ := normalize ib Γ ops
  sound d Γ _ ops normalized yields ctx call env hc hn he _ hi hd :=
    normalize_sound (merlinBook ib) d Γ ops normalized yields ctx call env hc hn he hi hd

end MerlinVerify.Rules.IntegerIdempotent
