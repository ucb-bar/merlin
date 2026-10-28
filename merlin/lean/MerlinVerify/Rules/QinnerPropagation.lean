/- Propagation alters only a specific metadata field on tensor.empty. It is value-inert in the
frozen semantics, but is deliberately NOT declared inert to the runtime or compiler pipeline. -/
import MerlinVerify.Check.ExactSyntax
import MerlinVerify.Check.Alpha
import MerlinVerify.Check.AlphaFast
set_option maxHeartbeats 2000000
namespace MerlinVerify.Rules.QinnerPropagation
open MerlinVerify Sem.Ops

def dropAttrs (attrs : List (String × Attr)) : List (String × Attr) :=
  attrs.filter (fun a => a.1 != "prov.quant_inner")
def head (op : Op) : Op :=
  .mk op.name op.results op.operands
    (if op.name == "tensor.empty" then dropAttrs op.attrs else op.attrs)
    op.props op.regions op.successors

def scan (attrs : List (String × Attr)) : Option String :=
  attrs.findSome? fun (key,_) => if Spec.Annotations.semanticsFree key then none else some key

theorem scan_drop (attrs : List (String × Attr)) : scan (dropAttrs attrs) = scan attrs := by
  induction attrs with
  | nil => rfl
  | cons pair rest ih =>
    obtain ⟨key,value⟩ := pair
    by_cases eq : key = "prov.quant_inner"
    · subst key
      have free : Spec.Annotations.semanticsFree "prov.quant_inner" = true := by cbv
      simpa [dropAttrs,scan,free] using ih
    · by_cases free : Spec.Annotations.semanticsFree key = true
      · simpa [dropAttrs,scan,eq,free] using ih
      · simp [dropAttrs,scan,eq,free]

theorem head_sound (ctx : Ctx) (call : CallSem) (op : Op) (args : List RVal)
    (regions : List (Ctx → RegionSem)) :
    opSem ctx call (head op) args regions = opSem ctx call op args regions := by
  cases op with
  | mk name results operands attrs props rs succ =>
    by_cases eq : name = "tensor.empty"
    · subst name
      have interpreted : interpretedAttrs "tensor.empty" = [] := by simp [interpretedAttrs]
      have dialect : dialectOf "tensor.empty" = "tensor" := by cbv
      have gate : unmodelledAttr? (head (Op.mk "tensor.empty" results operands attrs props rs succ)).blind =
          unmodelledAttr? (Op.mk "tensor.empty" results operands attrs props rs succ).blind := by
        simp only [head,beq_self_eq_true,↓reduceIte,Op.blind,Op.name,Op.results,Op.attrs,Op.props,
          Op.operands,Op.regions,Op.successors,unmodelledAttr?,interpreted,List.contains_nil, Bool.false_or]
        change scan (props ++ dropAttrs attrs) = scan (props ++ attrs)
        simp only [scan,List.findSome?_append]
        exact congrArg (Option.or (scan props)) (scan_drop attrs)
      unfold opSem
      rw [gate]
      split
      · rfl
      · simp only [head,Op.blind,Op.name,Op.results,Op.operands,Op.attrs,Op.props,Op.regions,
          Op.successors,beq_self_eq_true,↓reduceIte,dispatch,dialect,Tensor.sem,Tensor.resultTensorTy]
    · simp [head,Op.name,Op.results,Op.operands,Op.attrs,Op.props,Op.regions,Op.successors,eq]

mutual
  def operations : List Op → List Op
    | [] => []
    | .mk n r o a p rs s :: rest =>
      .mk n r o (if n == "tensor.empty" then dropAttrs a else a) p (regions rs) s :: operations rest
  def regions : List Region → List Region
    | [] => []
    | .mk bs :: rs => .mk (blocks bs) :: regions rs
  def blocks : List Block → List Block
    | [] => []
    | .mk args ops :: bs => .mk args (operations ops) :: blocks bs
end

def module (m : Module) : Module := {m with ops := operations m.ops}

def check (s t : Module) : Bool := Check.ExactSyntax.modules (module s) (module t)
theorem check_sound {s t : Module} (h : check s t = true) : module s = module t :=
  Check.ExactSyntax.modules_sound h

theorem operations_empty (ops : List Op) : (operations ops).isEmpty = ops.isEmpty := by
  cases ops with
  | nil => simp [operations]
  | cons op rest => cases op; simp [operations]

mutual
  theorem operations_sound (ctx : Ctx) (call : CallSem) (env : Env) : ∀ ops,
      evalOps opSem ctx call env (operations ops) = evalOps opSem ctx call env ops
    | [] => by simp only [operations,evalOps]
    | .mk n r o a p rs s :: rest => by
      simp only [operations,evalOps]
      have hr := regions_sound call env rs
      simp only [Op.isTerminator,Op.name]
      by_cases term : terminatorNames.contains n = true
      · simp only [term, ↓reduceIte, operations_empty]
      · simp only [term, ↓reduceIte]
        rw [hr]
        -- Rewriting the local head uses the proved value-inertness, with regions already equal.
        change (env.getAll o).bind _ = (env.getAll o).bind _
        congr 1
        funext args
        rw [show opSem ctx call (.mk n r o (if n == "tensor.empty" then dropAttrs a else a) p (regions rs) s)
            args (evalRegions opSem call env rs) = opSem ctx call (.mk n r o a p rs s)
            args (evalRegions opSem call env rs) from by
          rw [← head_sound ctx call (.mk n r o a p rs s)]
          apply opSem_blind
          rfl]
        congr 1
        funext values
        congr 1
        funext next
        exact operations_sound ctx call next rest
  theorem regions_sound (call : CallSem) (env : Env) : ∀ rs,
      evalRegions opSem call env (regions rs) = evalRegions opSem call env rs
    | [] => by simp only [regions,evalRegions]
    | .mk bs :: rest => by
      simp only [regions,evalRegions,List.cons.injEq]
      exact ⟨funext fun ctx => funext fun args => blocks_sound ctx call env bs args,
        regions_sound call env rest⟩
  theorem blocks_sound (ctx : Ctx) (call : CallSem) (env : Env) : ∀ bs args,
      evalBlocks opSem ctx call env (blocks bs) args = evalBlocks opSem ctx call env bs args
    | [.mk a ops], args => by
      simp only [blocks,evalBlocks,evalBlock]
      congr 1
      funext next
      exact operations_sound ctx call next ops
    | [], _ => by simp only [blocks,evalBlocks]
    | .mk a ops :: .mk b tail :: bs, args => by
      cases bs <;> simp only [blocks,evalBlocks]
end

def function (f : Func) : Func := {f with body := .mk (blocks f.body.blocks)}

theorem blocks_empty (bs : List Block) : (blocks bs).isEmpty = bs.isEmpty := by
  cases bs with
  | nil => simp [blocks]
  | cons b bs => cases b; simp [blocks]

theorem region_head (rs : List Region) :
    (regions rs).head?.getD (.mk []) = .mk (blocks (rs.headD (.mk [])).blocks) := by
  cases rs with
  | nil => simp [regions,Region.blocks,blocks]
  | cons r rs => cases r; simp [regions,Region.blocks]

theorem asFunc (op : Op) :
    ((operations [op]).head!).asFunc? = op.asFunc?.map function := by
  cases op with
  | mk n r o a p rs s =>
    by_cases name : n = "func.func"
    · subst n
      simp only [operations,List.head!,Op.asFunc?,Op.name,beq_self_eq_true,ne_eq,not_true_eq_false,↓reduceIte]
      have ne : ("func.func" == "tensor.empty") = false := by decide
      simp only [ne,↓reduceIte,Op.attr?,Op.props,Op.attrs]
      cases hp : p.lookup "sym_name" with
      | none =>
        cases ha : a.lookup "sym_name" with
        | none => simp [hp,ha]
        | some attr => cases attr <;> simp [hp,ha,function,Op.regions,region_head,Region.blocks,blocks_empty]
      | some attr => cases attr <;> simp [hp,function,Op.regions,region_head,Region.blocks,blocks_empty]
    · simp [operations,List.head!,Op.asFunc?,Op.name,name]

theorem functions (m : Module) : (module m).funcs = m.funcs.map function := by
  rcases m with ⟨ops,a,p⟩
  change (operations ops).filterMap Op.asFunc? = (ops.filterMap Op.asFunc?).map function
  induction ops with
  | nil => simp [operations]
  | cons op rest ih =>
    cases op with
    | mk n r o a p rs s =>
      have localEq := asFunc (.mk n r o a p rs s)
      simp only [operations,List.head!] at localEq
      simp only [operations,List.filterMap_cons,localEq,ih]
      cases (Op.mk n r o a p rs s).asFunc? <;> rfl

theorem lookup (m : Module) (name : String) :
    (module m).func? name = (m.func? name).map function := by
  unfold Module.func?
  rw [functions]
  induction m.funcs with
  | nil => rfl
  | cons f fs ih =>
    by_cases h : f.name == name <;> simp [List.find?,function,h,ih]

theorem program_preserved (m : Module) (ctx : Ctx) :
    ∀ fuel name args, evalFunc opSem ctx (module m) fuel name args = evalFunc opSem ctx m fuel name args := by
  intro fuel
  induction fuel with
  | zero => intros; rfl
  | succ fuel ih =>
    intro name args
    rw [evalFunc,lookup]
    cases hf : m.func? name with
    | none => simp [evalFunc,hf]
    | some f =>
      simp only [Option.map_some,evalFunc,hf,function]
      split
      · rfl
      · have calls : evalFunc opSem ctx (module m) fuel = evalFunc opSem ctx m fuel :=
          funext fun name => funext fun args => ih name args
        rw [calls]
        cases bodyEq : f.body with
        | mk bs => exact blocks_sound ctx _ [] bs args

theorem denote_preserved (m : Module) (ctx : Ctx) (name : String) (args : List RVal) :
    MerlinVerify.denote opSem ctx (module m) name args = MerlinVerify.denote opSem ctx m name args := by
  have len : (module m).funcs.length = m.funcs.length := by simp [functions]
  simp only [MerlinVerify.denote,len]
  exact program_preserved m ctx _ name args

theorem complete_program {s t : Module} (h : check s t = true)
    (ctx : Ctx) (name : String) (args : List RVal) :
    MerlinVerify.denote opSem ctx s name args = MerlinVerify.denote opSem ctx t name args := by
  rw [← denote_preserved s, ← denote_preserved t,check_sound h]
#print axioms head_sound
#print axioms operations_sound
#print axioms complete_program

end MerlinVerify.Rules.QinnerPropagation
