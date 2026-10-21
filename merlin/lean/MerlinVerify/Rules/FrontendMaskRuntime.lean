/- Additive, inactive candidate for a bounded tensor/SCF runtime vocabulary.
Protected semantics and locks are unchanged. Poison controls are refused, not reclassified as UB.
This interpretation requires review before pipeline activation. -/
import MerlinVerify.Rules.FrontendBoolContext
import MerlinVerify.Check.Support

namespace MerlinVerify.Rules.FrontendMaskRuntime
open MerlinVerify MerlinVerify.Check

/-- A scalar index must be initialized, have the declared width and be nonnegative. -/
def indexNat? (ib : Nat) : RVal → Option Nat
  | .scalar (.some (.bits w x)) =>
    if 0 < ib ∧ w = ib ∧ 0 ≤ x.toInt then some x.toInt.toNat else none
  | _ => none

/-- Consume exactly one runtime index per dynamic dimension. Static dimensions consume none. -/
def resolveShape (ib : Nat) : List (Option Nat) → List RVal → Option (List Nat)
  | [], [] => some []
  | some n :: dims, args => (resolveShape ib dims args).map (n :: ·)
  | none :: dims, value :: args => do
    let n ← indexNat? ib value
    let rest ← resolveShape ib dims args
    pure (n :: rest)
  | _, _ => none

def empty (ib : Nat) (op : Op) (args : List RVal) : Outcome (List RVal) :=
  match op.results with
  | [{ ty := .tensor dims elt none, .. }] =>
    match resolveShape ib dims args with
    | some shape => .ok [.tensor (Tensor.full shape elt .poison)]
    | none => .stuck "candidate empty: extent is uninitialized, negative, wrong-width or wrong-arity"
  | _ => .stuck "candidate empty: unsupported result type"

/-- Explicit finite iteration budget. Each recursive call consumes one iteration. -/
def loopN (ib : Nat) (body : RegionSem) (step : Nat) : Nat → Nat → List RVal → Outcome (List RVal)
  | 0, _, carried => .ok carried
  | n + 1, iv, carried =>
    (body (.scalar (.some (.bits ib (BitVec.ofNat ib iv))) :: carried)).bind fun next =>
      if next.length = carried.length then loopN ib body step n (iv + step) next
      else .stuck "candidate for: yield arity differs from loop-carried values"

def ifSem (ctx : Ctx) (args : List RVal) (regions : List (Ctx → RegionSem)) : Outcome (List RVal) :=
  match args, regions with
  | [condition], [yes, no] =>
    match condition with
    | .scalar (.some (.bits 1 b)) => (if b = 1 then yes else no) ctx []
    | _ => .stuck "candidate if: condition is not an initialized i1"
  | _, _ => .stuck "candidate if: expected initialized i1 and two branches"

def forSem (limit : Nat) (ctx : Ctx) (args : List RVal)
    (regions : List (Ctx → RegionSem)) : Outcome (List RVal) :=
  match args, regions with
  | lower :: upper :: stride :: carried, [body] =>
    match indexNat? ctx.indexBits lower, indexNat? ctx.indexBits upper, indexNat? ctx.indexBits stride with
    | some lb, some ub, some step =>
      if 0 < ctx.indexBits ∧ 0 < step then
        let count := (ub - lb + step - 1) / step
        if count ≤ limit ∧ lb + count * step < 2 ^ (ctx.indexBits - 1) then
          loopN ctx.indexBits (body ctx) step count lb carried
        else .stuck "candidate for: iteration budget or signed advancement bound exceeded"
      else .stuck "candidate for: index width and step must be positive"
    | _, _, _ => .stuck "candidate for: controls must be initialized nonnegative indices"
  | _, _ => .stuck "candidate for: expected three controls and one body"

/-- New cases are explicit. Every other operation retains the existing interpretation.
The limit is an interpretation parameter, not a trusted hint about a program's trip count. -/
def sem (limit : Nat) : OpSem := fun ctx call op args regions =>
  if Sem.Ops.unmodelledAttr? op.blind = none then
    match Sem.Ops.dialectOf op.blind.name with
    | "scf" =>
      match op.blind.name with
      | "scf.if" => ifSem ctx args regions
      | "scf.for" => forSem limit ctx args regions
      | _ => Sem.Ops.opSem ctx call op args regions
    | "tensor" =>
      if op.blind.name = "tensor.empty" ∧ Sem.Ops.Tensor.resultTensorTy op.blind = none then
        empty ctx.indexBits op.blind args
      else Sem.Ops.opSem ctx call op args regions
    | _ => Sem.Ops.opSem ctx call op args regions
  else Sem.Ops.opSem ctx call op args regions

theorem tensor_eq (limit : Nat) (ctx : Ctx) (call : CallSem) (op : Op)
    (hd : Sem.Ops.dialectOf op.name = "tensor") (hn : op.name ≠ "tensor.empty")
    (args : List RVal) (regions : List (Ctx → RegionSem)) :
    sem limit ctx call op args regions = Sem.Ops.opSem ctx call op args regions := by
  simp only [sem, Op.blind_name, hd]
  split
  · simp [hn]
  · rfl

/-- Structured arithmetic remains delegated to the frozen interpretation. -/
theorem linalg_eq (limit : Nat) (ctx : Ctx) (call : CallSem) (op : Op)
    (hd : Sem.Ops.dialectOf op.name = "linalg") (args : List RVal) (regions : List (Ctx → RegionSem)) :
    sem limit ctx call op args regions = Sem.Ops.opSem ctx call op args regions := by
  simp only [sem, Op.blind_name, hd]
  split <;> rfl

/-- The candidate changes only the declared cases; no floating interpretation is added. -/
theorem arith_eq (limit : Nat) (ctx : Ctx) (call : CallSem) (op : Op)
    (hd : Sem.Ops.dialectOf op.name = "arith") (args : List RVal) (regions : List (Ctx → RegionSem)) :
    sem limit ctx call op args regions = Sem.Ops.opSem ctx call op args regions := by
  simp only [sem, Op.blind_name, hd]
  split <;> rfl

theorem sem_blind {limit : Nat} {ctx : Ctx} {call : CallSem} {op other : Op}
    {args : List RVal} {regions : List (Ctx → RegionSem)} (h : op.blind = other.blind) :
    sem limit ctx call op args regions = sem limit ctx call other args regions := by
  unfold sem
  rw [h, Sem.Ops.opSem_blind h]

private theorem pin_self (ctx : Ctx) (body : Ctx → RegionSem) :
    FrontendBoolContext.pin ctx body ctx = body ctx := by cases ctx; rfl

theorem ifSem_pin (ctx : Ctx) (args : List RVal) (regions : List (Ctx → RegionSem)) :
    ifSem ctx args (regions.map (FrontendBoolContext.pin ctx)) = ifSem ctx args regions := by
  cases args with
  | nil => rfl
  | cons condition rest =>
    cases rest with
    | cons _ _ => rfl
    | nil =>
      cases regions with
      | nil => rfl
      | cons yes rest =>
        cases rest with
        | nil => rfl
        | cons no rest =>
          cases rest with
          | cons _ _ => rfl
          | nil =>
            simp only [List.map_cons, List.map_nil, ifSem]
            split
            · split <;> exact congrFun (pin_self ctx _) []
            · rfl

theorem forSem_pin (limit : Nat) (ctx : Ctx) (args : List RVal) (regions : List (Ctx → RegionSem)) :
    forSem limit ctx args (regions.map (FrontendBoolContext.pin ctx)) = forSem limit ctx args regions := by
  cases args with
  | nil => rfl
  | cons lower rest =>
    cases rest with
    | nil => rfl
    | cons upper rest =>
      cases rest with
      | nil => rfl
      | cons stride carried =>
        cases regions with
        | nil => rfl
        | cons body rest =>
          cases rest with
          | cons _ _ => rfl
          | nil => simp only [List.map_cons, List.map_nil, forSem, pin_self]

theorem sem_pin (limit : Nat) (ctx : Ctx) (call : CallSem) (op : Op) (args : List RVal)
    (regions : List (Ctx → RegionSem)) :
    sem limit ctx call op args (regions.map (FrontendBoolContext.pin ctx)) = sem limit ctx call op args regions := by
  unfold sem
  split
  · split
    · split
      · exact ifSem_pin ctx args regions
      · exact forSem_pin limit ctx args regions
      · exact FrontendBoolContext.opSem_pin ctx call op args regions
    · split
      · rfl
      · exact FrontendBoolContext.opSem_pin ctx call op args regions
    · exact FrontendBoolContext.opSem_pin ctx call op args regions
  · exact FrontendBoolContext.opSem_pin ctx call op args regions

theorem sem_regions (limit : Nat) (ctx : Ctx) (call : CallSem) (op : Op) (args : List RVal)
    (left right : List (Ctx → RegionSem))
    (h : left.map (FrontendBoolContext.pin ctx) = right.map (FrontendBoolContext.pin ctx)) :
    sem limit ctx call op args left = sem limit ctx call op args right := by
  rw [← sem_pin limit ctx call op args left, h, sem_pin]

end MerlinVerify.Rules.FrontendMaskRuntime
