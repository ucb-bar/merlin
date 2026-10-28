/- An inhabited, successful, poison-free positive lifting example. Whole-model definedness is
not inferred from the general correspondence theorem. No power or floating arithmetic law occurs. -/
import MerlinVerify.Check.QinnerRuntime
namespace MerlinVerify.Rules.QinnerExecution
open MerlinVerify Check QinnerBinding

def ty : Ty := .tensor [some 4] (.float .f32) none

def source : Module := ⟨[.mk "func.func" [] []
  [("function_type",.type (.function [ty] [ty,ty,ty])),("sym_name",.str "forward")] []
  [.mk [.mk [⟨0,ty⟩]
    [.mk "tensor.empty" [⟨1,ty⟩] [] [("prov.quant_inner",.str "left")] [] [] [],
     .mk "tensor.empty" [⟨2,ty⟩] [] [("prov.quant_inner",.str "right")] [] [] [],
     .mk "func.return" [] [2,1,0] [] [] [] []]]] []],[],[]⟩

def target : Module := ⟨[.mk "func.func" [] []
  [("function_type",.type (.function [ty,ty,ty] [ty,ty,ty])),("sym_name",.str "forward")] []
  [.mk [.mk [⟨0,ty⟩,⟨1,ty⟩,⟨2,ty⟩] [.mk "func.return" [] [2,1,0] [] [] [] []]]] []],[],[]⟩

theorem accepted : Qinner.check source target = true := by cbv

theorem target_execution (ctx : Ctx) (x left right : RVal) :
    denote Sem.Ops.opSem ctx target "forward" [x,left,right] = .ok [right,left,x] := by
  simp [target,denote,evalFunc,Module.func?,Module.funcs,Op.asFunc?,Op.attr?,Op.name,Op.regions,
    Region.blocks,evalRegion,evalBlocks,evalBlock,evalOps,Op.isTerminator,terminatorNames,
    Env.bindAll,Env.getAll,Env.get?,Outcome.mapM,Outcome.ofOption,List.lookup,Outcome.bind,
    Op.operands,Op.results,Op.props,Op.attrs]

theorem successful_clean (ctx : Ctx) (x left right : RVal)
    (hx : x.Defined) (hl : left.Defined) (hr : right.Defined) :
    ∃ results, denote Sem.Ops.opSem ctx target "forward" [x,left,right] = .ok results ∧
      ResultsDefined results := by
  refine ⟨[right,left,x],target_execution ctx x left right,?_⟩
  intro value member
  rcases List.mem_cons.mp member with rfl | member
  · exact hr
  rcases List.mem_cons.mp member with rfl | member
  · exact hl
  rcases List.mem_cons.mp member with rfl | member
  · exact hx
  cases member

def zero : RVal := .tensor (Tensor.full [4] (.float .f32) (.some (.flt .f32 0)))

theorem zero_defined : zero.Defined := by
  intro i hi
  change Idx.InBounds [4] i at hi
  simp [Tensor.full,Tensor.canon,hi]

theorem domain_inhabited : ∃ x left right : RVal,
    x.Defined ∧ left.Defined ∧ right.Defined := ⟨zero,zero,zero,zero_defined,zero_defined,zero_defined⟩
#print axioms accepted
#print axioms successful_clean
#print axioms domain_inhabited
end MerlinVerify.Rules.QinnerExecution
