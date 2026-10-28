/- Source-side operational typing obligations and composition through SSA and nested regions.
No premise asserts whole-program source/target equivalence. Numerical laws remain arguments.
The generic primitive-typing obligation is explicit: callbacks are used only on typed domains. -/
import MerlinVerify.Rules.ScalarSquareSSA

namespace MerlinVerify.Rules.ScalarSquareComposition

open MerlinVerify MerlinVerify.Check MerlinVerify.Check.ScalarSquare MerlinVerify.Sem.Ops
open ScalarSquareSSA

abbrev CallDomain := String → List RVal → Prop

def CallsAgree (D : CallDomain) (left right : CallSem) : Prop :=
  ∀ name values, D name values → left name values = right name values

def BlockDomain (ib : Nat) (blocks : List Block) (values : List RVal) : Prop :=
  match blocks with
  | [block] => Arguments ib block.args values
  | _ => True

def RegionsAgree (F : Ctx → Prop) (ib : Nat) :
    List Region → List (Ctx → RegionSem) → List (Ctx → RegionSem) → Prop
  | [], [], [] => True
  | .mk blocks :: rs, left :: ls, right :: ts =>
    (∀ ctx, F ctx → ∀ values, BlockDomain ib blocks values → left ctx values = right ctx values) ∧
      RegionsAgree F ib rs ls ts
  | _, _, _ => False

/-- A local, source-primitive typing obligation. It is not source/target program equality:
changing callbacks off their declared domains cannot affect this operation on these operands.
The source callbacks are fixed: their result typing is proved by the operational checker.
Quantifying over arbitrary, ill-typed source callbacks would be false for reductions.
F must include the chosen float model and index width; it may vary the loop index. -/
def PrimitiveTyping (D : CallDomain) (F : Ctx → Prop) (ib : Nat)
    (ctx : Ctx) (op : Op) (args : List RVal) (callS : CallSem)
    (rs : List (Ctx → RegionSem)) : Prop :=
  ∀ callT rt, CallsAgree D callS callT → RegionsAgree F ib op.regions rs rt →
    opSem ctx callS op args rs = opSem ctx callT op args rt

def Numerical (F : Ctx → Prop) : Prop :=
  ∀ ctx, F ctx → ∀ site : Site, ScalarSquareContract.Law ctx.M site.kind site.attrs

theorem primitive_regionless (D : CallDomain) (F : Ctx → Prop) (ib : Nat)
    (ctx : Ctx) (op : Op) (args : List RVal) (noRegions : op.regions = [])
    (noCalls : calleeOf op.blind = none) (left : CallSem) :
    PrimitiveTyping D F ib ctx op args left [] := by
  intro right rt calls related
  rw [noRegions] at related
  cases rt <;> simp only [RegionsAgree] at related
  · exact opSem_call_local (fun name impossible => by rw [noCalls] at impossible; cases impossible)
  all_goals contradiction

theorem primitive_call (D : CallDomain) (F : Ctx → Prop) (ib : Nat)
    (ctx : Ctx) (op : Op) (args : List RVal) (name : String)
    (operation : op.name = "func.call") (callee : calleeOf op.blind = some name)
    (flat : attrSymbol? op.blind "callee" = some name) (typed : D name args) (left : CallSem) (rs : List (Ctx → RegionSem)) :
    PrimitiveTyping D F ib ctx op args left rs := by
  intro right rt calls related
  unfold opSem
  split
  · rfl
  · have hd : dialectOf op.blind.name = "func" := by rw [show op.blind.name = op.name from rfl, operation]; cbv
    simp only [dispatch, hd]
    simp only [Func.sem, show op.blind.name = "func.call" from operation, flat,
      restrictCall, callee, ↓reduceIte, calls name args typed]

mutual
  /-- Obligations mention only execution of the original source and its lexical facts.
  Result typing/facts are required only when a source step and its SSA binding succeed. -/
  def ReadyOps (D : CallDomain) (F : Ctx → Prop) (ib : Nat) (ctx : Ctx) (call : CallSem)
      (scope : Scope) (path : List Nat) (index : Nat) (env : Env) : List Op → Prop
    | [] => True
    | (.mk name results operands attrs props rs successors) :: rest =>
      let op := Op.mk name results operands attrs props rs successors
      let here := path ++ [index]
      if op.isTerminator then True else
        ReadyRegions D F ib call scope here 0 env rs ∧
        ∀ args, env.getAll operands = .ok args →
          ((site? scope here op = none → PrimitiveTyping D F ib ctx op args call (evalRegions opSem call env rs)) ∧
          ∀ values bound,
            opSem ctx call op args (evalRegions opSem call env rs) = .ok values →
            env.bindAll results values = .ok bound →
            ResultFacts ib op values ∧
              ReadyOps D F ib ctx call (bindOp scope op) path (index + 1) bound rest)
    termination_by ops => sizeOf ops
    decreasing_by all_goals simp_wf; all_goals omega
  def ReadyRegions (D : CallDomain) (F : Ctx → Prop) (ib : Nat) (call : CallSem)
      (scope : Scope) (path : List Nat) (index : Nat) (env : Env) : List Region → Prop
    | [] => True
    | .mk bs :: rest =>
      ReadyBlocks D F ib call scope (path ++ [index]) 0 env bs ∧
        ReadyRegions D F ib call scope path (index + 1) env rest
    termination_by rs => sizeOf rs
    decreasing_by all_goals simp_wf; all_goals omega
  def ReadyBlocks (D : CallDomain) (F : Ctx → Prop) (ib : Nat) (call : CallSem)
      (scope : Scope) (path : List Nat) (index : Nat) (env : Env) : List Block → Prop
    | [.mk args ops] =>
      ∀ ctx, F ctx → ∀ values, Arguments ib args values → ∀ bound,
        env.bindAll args values = .ok bound →
        ReadyOps D F ib ctx call (bindArgs scope args) (path ++ [index]) 0 bound ops
    | _ => True
    termination_by bs => sizeOf bs
    decreasing_by all_goals simp_wf; all_goals omega
end

theorem operations_empty (scope : Scope) (path : List Nat) (index : Nat) (ops : List Op) :
    (operations scope path index ops).ops.isEmpty = ops.isEmpty := by
  cases ops with
  | nil => simp [operations]
  | cons op rest => cases op; simp [operations]

theorem eval_cons (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) (op : Op) (rest : List Op) :
    evalOps sem ctx call env (op :: rest) =
      if op.isTerminator then
        if rest.isEmpty then env.getAll op.operands else .stuck "ops after terminator"
      else (env.getAll op.operands).bind fun args =>
        (sem ctx call op args (evalRegions sem call env op.regions)).bind fun values =>
        (env.bindAll op.results values).bind fun next => evalOps sem ctx call next rest := by
  cases op
  rfl

theorem eval_region (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) (r : Region) (values : List RVal) :
    evalRegion sem ctx call env r values = evalBlocks sem ctx call env r.blocks values := by
  cases r
  rfl

mutual
  theorem operations_sound (D : CallDomain) (F : Ctx → Prop) (ib : Nat)
      (numerical : Numerical F) (left right : CallSem) (calls : CallsAgree D left right) :
      ∀ (scope : Scope) (path : List Nat) (index : Nat) (env : Env) (ops : List Op)
        (ctx : Ctx), F ctx → Invariant ib scope env →
        ReadyOps D F ib ctx left scope path index env ops →
        evalOps opSem ctx left env ops =
          evalOps opSem ctx right env (operations scope path index ops).ops
    | _, _, _, _, [], _, _, _, _ => by simp [operations, evalOps]
    | scope, path, index, env, (.mk n results operands attrs props rs successors) :: rest,
        ctx, family, valid, ready => by
      let op := Op.mk n results operands attrs props rs successors
      let here := path ++ [index]
      have opEq : op = .mk n results operands attrs props rs successors := rfl
      by_cases term : op.isTerminator = true
      · have notSite : site? scope here op = none := by
          cases selected : site? scope here op with
          | none => rfl
          | some site =>
            have source := site_source selected
            have : op.isTerminator = false := by rw [← source]; rfl
            simp [this] at term
        rw [operations.eq_2]
        simp only [show site? scope (path ++ [index]) (.mk n results operands attrs props rs successors) = none from notSite]
        simp only [evalOps, Op.isTerminator, Op.name] at term ⊢
        simp only [op, Op.isTerminator, Op.name] at term
        simp only [term, ↓reduceIte, operations_empty, Op.operands]
      · have noTerm : op.isTerminator = false := by simpa using term
        have parts : ReadyRegions D F ib left scope here 0 env rs ∧
            ∀ args, env.getAll operands = .ok args →
              ((site? scope here op = none → PrimitiveTyping D F ib ctx op args left (evalRegions opSem left env rs)) ∧
              ∀ values bound, opSem ctx left op args (evalRegions opSem left env rs) = .ok values →
                env.bindAll results values = .ok bound → ResultFacts ib op values ∧
                  ReadyOps D F ib ctx left (bindOp scope op) path (index + 1) bound rest) := by
          rw [ReadyOps.eq_2] at ready
          simpa only [op, here, noTerm, show (Op.mk n results operands attrs props rs successors).isTerminator = false from noTerm,
            Bool.false_eq_true, ↓reduceIte] using ready
        have nested := regions_sound D F ib numerical left right calls scope here 0 env rs valid parts.1
        cases selected : site? scope here op with
        | none =>
          let target : Op := .mk n results operands attrs props (regions scope here 0 rs).regions successors
          have blind : op.blind = target.blind := rfl
          rw [operations.eq_2]
          try simp only [show site? scope (path ++ [index]) (.mk n results operands attrs props rs successors) = none from selected]
          change evalOps opSem ctx left env (op :: rest) =
            evalOps opSem ctx right env (target :: (operations (bindOp scope op) path (index + 1) rest).ops)
          rw [eval_cons, eval_cons]
          simp only [noTerm, show target.isTerminator = false from noTerm,
            Bool.false_eq_true, ↓reduceIte]
          change ((env.getAll operands).bind fun args =>
              (opSem ctx left op args (evalRegions opSem left env rs)).bind fun values =>
              (env.bindAll results values).bind fun next => evalOps opSem ctx left next rest) =
            ((env.getAll operands).bind fun args =>
              (opSem ctx right target args (evalRegions opSem right env (regions scope here 0 rs).regions)).bind fun values =>
              (env.bindAll results values).bind fun next =>
                evalOps opSem ctx right next (operations (bindOp scope op) path (index + 1) rest).ops)
          cases got : env.getAll operands with
          | ub => rfl
          | stuck _ => rfl
          | ok args =>
            have primitive := (parts.2 args got).1 selected right _ calls nested
            have same : opSem ctx left op args (evalRegions opSem left env rs) =
                opSem ctx right target args (evalRegions opSem right env (regions scope here 0 rs).regions) :=
              primitive.trans (opSem_blind blind)
            simp only [op, target, Op.operands, Op.results, Op.regions, got, Outcome.bind_ok]
            rw [← same]
            cases stepped : opSem ctx left op args (evalRegions opSem left env rs) with
            | ub => rfl
            | stuck _ => rfl
            | ok values =>
              simp only [Outcome.bind_ok]
              cases bound : env.bindAll results values with
              | ub => rfl
              | stuck _ => rfl
              | ok next =>
                have facts := (parts.2 args got).2 values next stepped bound
                exact operations_sound D F ib numerical left right calls _ _ _ next rest ctx family
                  (results_bind valid facts.1 bound) facts.2
        | some site =>
          have source : op = site.source := (site_source selected).symm
          have rsEq : rs = [] := congrArg Op.regions source
          have resEq : results = site.source.results := congrArg Op.results source
          have same := site_step selected valid ctx left right (numerical ctx family site)
          have targetTerm : site.target.isTerminator = false := rfl
          have sourceTerm : site.source.isTerminator = false := rfl
          rw [operations.eq_2]
          try simp only [show site? scope (path ++ [index]) (.mk n results operands attrs props rs successors) = some site from selected]
          change evalOps opSem ctx left env (op :: rest) =
            evalOps opSem ctx right env (site.target :: (operations (bindOp scope op) path (index + 1) rest).ops)
          rw [source]
          rw [eval_cons, eval_cons]
          simp only [sourceTerm, targetTerm, Bool.false_eq_true, ↓reduceIte]
          simp only [show site.target.results = site.source.results from rfl, ← resEq]
          change ((env.getAll site.source.operands).bind fun args =>
              (opSem ctx left site.source args []).bind fun values =>
              (env.bindAll results values).bind fun bound => evalOps opSem ctx left bound rest) =
            ((env.getAll site.target.operands).bind fun args =>
              (opSem ctx right site.target args []).bind fun values =>
              (env.bindAll results values).bind fun bound =>
                evalOps opSem ctx right bound (operations (bindOp scope site.source) path (index + 1) rest).ops)
          have values := site_values selected valid
          obtain ⟨base, baseEq, typed, expEq⟩ := values
          have getSource := get_two env site.base site.exponent base _ baseEq expEq
          have getTarget := get_two env site.base site.base base base baseEq baseEq
          change env.getAll site.source.operands = _ at getSource
          change env.getAll site.target.operands = _ at getTarget
          rw [getSource, getTarget] at same ⊢
          simp only [Outcome.bind_ok] at same ⊢
          rw [← same]
          cases stepped : opSem ctx left site.source
              [base, .scalar (.some (.flt site.kind (ScalarSquareContract.two site.kind)))] [] with
          | ub => rfl
          | stuck _ => rfl
          | ok values =>
            simp only [Outcome.bind_ok]
            cases bound : env.bindAll results values with
            | ub => rfl
            | stuck _ => rfl
            | ok next =>
              have argsEq : env.getAll operands = .ok
                  [base, .scalar (.some (.flt site.kind (ScalarSquareContract.two site.kind)))] := by
                simpa only [← source, op, Op.operands] using getSource
              have stepEq : opSem ctx left op
                  [base, .scalar (.some (.flt site.kind (ScalarSquareContract.two site.kind)))]
                  (evalRegions opSem left env rs) = .ok values := by
                simpa only [source, rsEq, evalRegions] using stepped
              have facts := (parts.2 _ argsEq).2 values next stepEq bound
              simpa only [source, Outcome.bind_ok] using operations_sound D F ib numerical left right calls _ _ _ next rest ctx family
                (results_bind valid facts.1 bound) facts.2
    termination_by _ _ _ _ ops _ _ _ _ => sizeOf ops
    decreasing_by all_goals simp_wf; all_goals omega

  theorem regions_sound (D : CallDomain) (F : Ctx → Prop) (ib : Nat)
      (numerical : Numerical F) (left right : CallSem) (calls : CallsAgree D left right) :
      ∀ (scope : Scope) (path : List Nat) (index : Nat) (env : Env) (rs : List Region),
        Invariant ib scope env → ReadyRegions D F ib left scope path index env rs →
        RegionsAgree F ib rs (evalRegions opSem left env rs)
          (evalRegions opSem right env (regions scope path index rs).regions)
    | _, _, _, _, [], _, _ => by simp [regions, evalRegions, RegionsAgree]
    | scope, path, index, env, .mk bs :: rest, valid, ready => by
      simp only [ReadyRegions] at ready
      simp only [regions, evalRegions, RegionsAgree]
      exact ⟨fun ctx family values typed =>
          blocks_sound D F ib numerical left right calls scope (path ++ [index]) 0 env bs
            ctx family valid ready.1 values typed,
        regions_sound D F ib numerical left right calls scope path (index + 1) env rest valid ready.2⟩
    termination_by _ _ _ _ rs _ _ => sizeOf rs
    decreasing_by all_goals simp_wf; all_goals omega

  theorem blocks_sound (D : CallDomain) (F : Ctx → Prop) (ib : Nat)
      (numerical : Numerical F) (left right : CallSem) (calls : CallsAgree D left right) :
      ∀ (scope : Scope) (path : List Nat) (index : Nat) (env : Env) (bs : List Block)
        (ctx : Ctx), F ctx → Invariant ib scope env →
        ReadyBlocks D F ib left scope path index env bs →
        ∀ values, BlockDomain ib bs values →
        evalBlocks opSem ctx left env bs values =
          evalBlocks opSem ctx right env (blocks scope path index bs).blocks values
    | _, _, _, _, [], _, _, _, _, _, _ => by simp [blocks, evalBlocks]
    | scope, path, index, env, .mk args ops :: rest, ctx, family, valid, ready, values, typed => by
      cases rest with
      | cons block tail =>
        cases block
        simp [blocks, evalBlocks]
      | nil =>
        simp only [ReadyBlocks] at ready
        simp only [BlockDomain, Block.args] at typed
        simp only [blocks, evalBlocks, evalBlock]
        cases bound : env.bindAll args values with
        | ub => rfl
        | stuck _ => rfl
        | ok next =>
          exact operations_sound D F ib numerical left right calls _ _ _ next ops ctx family
            (arguments_bind valid typed bound) (ready ctx family values typed next bound)
    termination_by _ _ _ _ bs _ _ _ _ _ _ => sizeOf bs
    decreasing_by all_goals simp_wf; all_goals omega
end

#print axioms operations_sound
#print axioms regions_sound
#print axioms blocks_sound

end MerlinVerify.Rules.ScalarSquareComposition
