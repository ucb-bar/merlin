/-
G0-FROZEN. The program-level evaluator, generic over the semantics of individual ops.

The evaluator owns SSA environments, regions, blocks, terminators and function calls. The meaning
of each op is supplied by an `OpSem` (the concrete one is `MerlinVerify.Sem.Ops.opSem`). Keeping the
two apart lets block-level theorems (congruence, inlining, DCE) be proved once for every `OpSem`,
while op semantics are filled in independently (design §11.2).

Scope at G0: single-block regions terminated by a terminator op; non-recursive call graphs (calls
are evaluated with fuel `#funcs + 1`, enough for any acyclic call graph). Multi-block regions are
`stuck`.
-/
import MerlinVerify.Sem.Refine
import MerlinVerify.Sem.Float

namespace MerlinVerify

/-- Interpretation of external (declaration-only) symbols, e.g. accelerator calls (design D8). -/
abbrev ExtEnv := String → List RVal → Outcome (List RVal)

/-- The runtime context a program is interpreted in. Claims quantify over it. -/
structure Ctx where
  /-- Interpretation of every float operation (design D5). -/
  M : FloatModel
  /-- Interpretation of external symbols (design D8). -/
  ext : ExtEnv
  /-- Bit width of `index` (from the lowering's `index_lowering.index_bits`). -/
  indexBits : Nat
  /-- Current iteration point inside a structured op's body (read by `linalg.index`). Set by the
  op that evaluates its region per point; empty elsewhere. Not part of any claim's quantifier. -/
  loopIndex : List Nat := []

/-- Meaning of a region: block-argument values ↦ the values its terminator yields. -/
abbrev RegionSem := List RVal → Outcome (List RVal)

/-- Meaning of calling a function symbol. -/
abbrev CallSem := String → List RVal → Outcome (List RVal)

/-- Meaning of one non-terminator op: given the context, calls, the op, its operand values and the
meanings of its regions, produce its result values (one per `op.results`). A region's meaning takes
the context to run it in, so a structured op can set `loopIndex` per iteration point. -/
abbrev OpSem := Ctx → CallSem → Op → List RVal → List (Ctx → RegionSem) → Outcome (List RVal)

/-- Ops that end a block and yield their operands to the enclosing op/function. -/
def terminatorNames : List String :=
  ["func.return", "linalg.yield", "scf.yield", "tensor.yield"]

def Op.isTerminator (op : Op) : Bool := terminatorNames.contains op.name

/-- SSA environment: newest binding first. -/
abbrev Env := List (ValueId × RVal)

def Env.get? (e : Env) (v : ValueId) : Option RVal := e.lookup v

def Env.bindAll (e : Env) (vs : List Value) (rs : List RVal) : Outcome Env :=
  if vs.length = rs.length then .ok ((vs.map (·.id)).zip rs ++ e)
  else .stuck "arity mismatch binding SSA values"

def Env.getAll (e : Env) (ids : List ValueId) : Outcome (List RVal) :=
  Outcome.mapM (fun v => Outcome.ofOption "unbound SSA value" (e.get? v)) ids

mutual
  /-- Run a block's ops in order; the terminator's operand values are the block's result. -/
  def evalOps (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) : List Op → Outcome (List RVal)
    | [] => .stuck "block has no terminator"
    | (op@(.mk _ _ operands _ _ regions _)) :: rest =>
      if op.isTerminator then
        if rest.isEmpty then env.getAll operands
        else .stuck "ops after terminator"
      else
        (env.getAll operands).bind fun args =>
        (sem ctx call op args (evalRegions sem call env regions)).bind fun results =>
        (env.bindAll op.results results).bind fun env' =>
        evalOps sem ctx call env' rest
    termination_by structural ops => ops

  /-- Meanings of a list of regions, closed over the enclosing environment (regions of e.g.
  `linalg.generic` may use values defined above them). -/
  def evalRegions (sem : OpSem) (call : CallSem) (env : Env) :
      List Region → List (Ctx → RegionSem)
    | [] => []
    | r :: rs => (fun ctx => evalRegion sem ctx call env r) :: evalRegions sem call env rs
    termination_by structural rs => rs

  def evalRegion (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) : Region → RegionSem
    | .mk blocks => evalBlocks sem ctx call env blocks

  /-- G0 scope: exactly one block. -/
  def evalBlocks (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) : List Block → RegionSem
    | [b] => evalBlock sem ctx call env b
    | _ => fun _ => .stuck "only single-block regions are supported"

  def evalBlock (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) : Block → RegionSem
    | .mk args ops => fun vals =>
        (env.bindAll args vals).bind fun env' => evalOps sem ctx call env' ops
end

/-- Evaluate function `name` of module `m` with call fuel `fuel`. -/
def evalFunc (sem : OpSem) (ctx : Ctx) (m : Module) : Nat → CallSem
  | 0, _, _ => .stuck "call depth exhausted (recursive call graph?)"
  | fuel + 1, name, args =>
    match m.func? name with
    | none => .stuck s!"unknown function {name}"
    | some f =>
      if f.isDecl then ctx.ext name args
      else evalRegion sem ctx (evalFunc sem ctx m fuel) [] f.body args

/-- The meaning of function `name` of module `m`. Function bodies are isolated from above, so they
start from the empty environment. -/
def denote (sem : OpSem) (ctx : Ctx) (m : Module) (name : String) (args : List RVal) :
    Outcome (List RVal) :=
  evalFunc sem ctx m (m.funcs.length + 1) name args

/-- `tgt.tf ⊑ src.sf` for every admissible context and every input in the domain (design §4.5). -/
def FuncRefines (sem : OpSem) (ctxOk : Ctx → Prop) (D : List RVal → Prop)
    (tgt : Module) (tf : String) (src : Module) (sf : String) : Prop :=
  ∀ ctx, ctxOk ctx → ∀ args, D args →
    Refines ResultsRefine (denote sem ctx tgt tf args) (denote sem ctx src sf args)

/-- Per-boundary claim form (see `RefinesC`): refinement wherever the source is modelled. -/
def FuncRefinesC (sem : OpSem) (ctxOk : Ctx → Prop) (D : List RVal → Prop)
    (tgt : Module) (tf : String) (src : Module) (sf : String) : Prop :=
  ∀ ctx, ctxOk ctx → ∀ args, D args →
    RefinesC ResultsRefine (denote sem ctx tgt tf args) (denote sem ctx src sf args)

/-- The capture is fully modelled on the domain (design §4.5): required of `S0`. -/
def FuncNotStuck (sem : OpSem) (ctxOk : Ctx → Prop) (D : List RVal → Prop)
    (m : Module) (f : String) : Prop :=
  ∀ ctx, ctxOk ctx → ∀ args, D args → (denote sem ctx m f args).NotStuck

theorem FuncRefinesC.trans {sem : OpSem} {ctxOk : Ctx → Prop} {D : List RVal → Prop}
    {a b c : Module} {fa fb fc : String}
    (h₁ : FuncRefinesC sem ctxOk D a fa b fb) (h₂ : FuncRefinesC sem ctxOk D b fb c fc) :
    FuncRefinesC sem ctxOk D a fa c fc :=
  fun ctx hctx args hD =>
    RefinesC.trans (R := ResultsRefine) (fun h h' => ResultsRefine.trans h h') (h₁ ctx hctx args hD)
      (h₂ ctx hctx args hD)

theorem FuncRefinesC.strict {sem : OpSem} {ctxOk : Ctx → Prop} {D : List RVal → Prop}
    {a c : Module} {fa fc : String}
    (h : FuncRefinesC sem ctxOk D a fa c fc) (hs : FuncNotStuck sem ctxOk D c fc) :
    FuncRefines sem ctxOk D a fa c fc :=
  fun ctx hctx args hD => RefinesC.strict (h ctx hctx args hD) (hs ctx hctx args hD)

theorem FuncRefines.trans {sem : OpSem} {ctxOk : Ctx → Prop} {D : List RVal → Prop}
    {a b c : Module} {fa fb fc : String}
    (h₁ : FuncRefines sem ctxOk D a fa b fb) (h₂ : FuncRefines sem ctxOk D b fb c fc) :
    FuncRefines sem ctxOk D a fa c fc :=
  fun ctx hctx args hD =>
    Refines.trans (R := ResultsRefine) (fun h h' => ResultsRefine.trans h h') (h₁ ctx hctx args hD)
      (h₂ ctx hctx args hD)

end MerlinVerify
