/- Kernel-checked malformed-runtime refusal regressions. These are not MLIR verifier tests. -/
import MerlinVerify.Rules.FrontendMaskExecution

namespace MerlinVerify.Rules.FrontendMaskRefusals
open MerlinVerify FrontendMaskRuntime FrontendMaskProgram

theorem poison_size (ib : Nat) : indexNat? ib (.scalar .poison) = none := rfl

theorem wrong_index_width : indexNat? 64 (.scalar (.some (.bits 32 0))) = none := by cbv

theorem missing_dynamic_size : resolveShape 64 [none, none] [index 0] = none := by cbv

theorem extra_dynamic_size : resolveShape 64 [some 3] [index 0] = none := rfl

theorem zero_width_size : indexNat? 0 (.scalar (.some (.bits 0 0))) = none := by cbv

theorem negative_size : indexNat? 64 (.scalar (.some (.bits 64 (-1)))) = none := by cbv

theorem poison_condition (ctx : Ctx) :
    ifSem ctx [.scalar .poison] [fun _ _ => .ok [], fun _ _ => .ok []] =
      .stuck "candidate if: condition is not an initialized i1" := rfl

theorem wrong_condition_width (ctx : Ctx) :
    ifSem ctx [.scalar (.some (.bits 8 1))] [fun _ _ => .ok [], fun _ _ => .ok []] =
      .stuck "candidate if: condition is not an initialized i1" := rfl

theorem missing_branch (ctx : Ctx) :
    ifSem ctx [.scalar (.some (.bits 1 1))] [fun _ _ => .ok []] =
      .stuck "candidate if: expected initialized i1 and two branches" := rfl

theorem yield_arity : loopN 64 (fun _ => .ok []) 1 1 0 [index 0] =
    .stuck "candidate for: yield arity differs from loop-carried values" := rfl

end MerlinVerify.Rules.FrontendMaskRefusals
