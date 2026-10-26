import Merlin.GraphStateIntegerProgram
import Merlin.GraphStateIntegerTrace
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStateIntegerInstance
open Merlin.GraphState.Scalar Merlin.GraphState.Integer
def registry : List Signature := [{inputs := [(.bits 1),(.bits 1),(.bits 1)],outputs := []}]
namespace F1
def inputs : List Ty := [(.bits 1)]
def outputs : List Ty := [(.bits 1)]
def before : Program := (.pure (.constant 1 (0)) [] (.pure (.constant 1 (-1)) [] (.pure (.constant 1 (0)) [] (.pure (.constant 1 (-1)) [] (.pure (.add 1) [1,0] (.pure (.sub 1) [0,3] (.pure (.mul 1) [0,2] (.pure (.add 1) [6,7] (.pure (.add 1) [0,7] (.pure (.sub 1) [0,8] (.pure (.mul 1) [8,0] (.pure (.mul 1) [0,9] (.opaque 0 [7,6,5] [] (.ret [0]))))))))))))))
theorem beforeTyped : checkTypes registry inputs outputs before = true := by decide
def after : Program := (.pure (.constant 1 (0)) [] (.pure (.constant 1 (-1)) [] (.pure (.constant 1 (0)) [] (.pure (.constant 1 (-1)) [] (.pure (.constant 1 (-1)) [] (.pure (.constant 1 (0)) [] (.pure (.constant 1 (0)) [] (.pure (.mul 1) [5,7] (.pure (.mul 1) [0,6] (.opaque 0 [4,3,2] [] (.ret [0])))))))))))
theorem afterTyped : checkTypes registry inputs outputs after = true := by decide
def first : Code := (.opaque 0 [(.two (.add 1) (.zero (.constant 1 (0))) (.zero (.constant 1 (-1)))),(.two (.sub 1) (.two (.add 1) (.zero (.constant 1 (0))) (.zero (.constant 1 (-1)))) (.zero (.constant 1 (-1)))),(.two (.mul 1) (.two (.sub 1) (.two (.add 1) (.zero (.constant 1 (0))) (.zero (.constant 1 (-1)))) (.zero (.constant 1 (-1)))) (.zero (.constant 1 (-1))))] [] (.ret [(.two (.mul 1) (.two (.mul 1) (.zero (.constant 1 (-1))) (.two (.sub 1) (.two (.add 1) (.two (.add 1) (.zero (.constant 1 (0))) (.var 0)) (.zero (.constant 1 (0)))) (.zero (.constant 1 (0))))) (.zero (.constant 1 (-1))))]))
theorem firstNormalized : normalize registry (vars inputs.length) before = some first := by decide
def second : Code := (.opaque 0 [(.zero (.constant 1 (-1))),(.zero (.constant 1 (0))),(.zero (.constant 1 (0)))] [] (.ret [(.two (.mul 1) (.two (.mul 1) (.zero (.constant 1 (-1))) (.var 0)) (.zero (.constant 1 (-1))))]))
theorem secondNormalized : normalize registry (vars inputs.length) after = some second := by decide
theorem localRewrites : Relation registry inputs outputs first second :=
  (Relation.effect (by decide) (by decide) (ExprsTyped.cons (ExprTyped.two (first := (.bits 1)) (second := (.bits 1)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprsTyped.cons (ExprTyped.two (first := (.bits 1)) (second := (.bits 1)) (by decide) (by decide) (ExprTyped.two (first := (.bits 1)) (second := (.bits 1)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.zero (by decide) (by decide))) (ExprsTyped.cons (ExprTyped.two (first := (.bits 1)) (second := (.bits 1)) (by decide) (by decide) (ExprTyped.two (first := (.bits 1)) (second := (.bits 1)) (by decide) (by decide) (ExprTyped.two (first := (.bits 1)) (second := (.bits 1)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.zero (by decide) (by decide))) ExprsTyped.nil))) (Sames.cons (Same.trans (Same.trans (Same.two (.add 1) (Same.constant _ 1 (0) (0) (by decide)) (Same.constant _ 1 (-1) (1) (by decide))) (Same.fold_add _ 1 (0) (1) (1) (by decide))) (Same.symm (Same.constant _ 1 (-1) (1) (by decide)))) (Sames.cons (Same.trans (Same.trans (Same.two (.sub 1) (Same.trans (Same.two (.add 1) (Same.constant _ 1 (0) (0) (by decide)) (Same.constant _ 1 (-1) (1) (by decide))) (Same.fold_add _ 1 (0) (1) (1) (by decide))) (Same.constant _ 1 (-1) (1) (by decide))) (Same.fold_sub _ 1 (1) (1) (0) (by decide))) (Same.symm (Same.constant _ 1 (0) (0) (by decide)))) (Sames.cons (Same.trans (Same.trans (Same.two (.mul 1) (Same.trans (Same.two (.sub 1) (Same.trans (Same.two (.add 1) (Same.constant _ 1 (0) (0) (by decide)) (Same.constant _ 1 (-1) (1) (by decide))) (Same.fold_add _ 1 (0) (1) (1) (by decide))) (Same.constant _ 1 (-1) (1) (by decide))) (Same.fold_sub _ 1 (1) (1) (0) (by decide))) (Same.constant _ 1 (-1) (1) (by decide))) (Same.fold_mul _ 1 (0) (1) (0) (by decide))) (Same.symm (Same.constant _ 1 (0) (0) (by decide)))) (Sames.nil _)))) (Relation.ret (ExprsTyped.cons (ExprTyped.two (first := (.bits 1)) (second := (.bits 1)) (by decide) (by decide) (ExprTyped.two (first := (.bits 1)) (second := (.bits 1)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.two (first := (.bits 1)) (second := (.bits 1)) (by decide) (by decide) (ExprTyped.two (first := (.bits 1)) (second := (.bits 1)) (by decide) (by decide) (ExprTyped.two (first := (.bits 1)) (second := (.bits 1)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.var (type := (.bits 1)) (by decide))) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.zero (by decide) (by decide)))) (ExprTyped.zero (by decide) (by decide))) ExprsTyped.nil) (Sames.cons (Same.trans (Same.trans (Same.two (.mul 1) (Same.trans (Same.two (.mul 1) (Same.constant _ 1 (-1) (1) (by decide)) (Same.trans (Same.two (.sub 1) (Same.trans (Same.two (.add 1) (Same.trans (Same.two (.add 1) (Same.constant _ 1 (0) (0) (by decide)) (Same.refl _ _)) (Same.zero_add 1 (ExprTyped.var (type := (.bits 1)) (by decide)))) (Same.constant _ 1 (0) (0) (by decide))) (Same.add_zero 1 (ExprTyped.var (type := (.bits 1)) (by decide)))) (Same.constant _ 1 (0) (0) (by decide))) (Same.sub_zero 1 (ExprTyped.var (type := (.bits 1)) (by decide))))) (Same.one_mul 1 (ExprTyped.var (type := (.bits 1)) (by decide)))) (Same.constant _ 1 (-1) (1) (by decide))) (Same.mul_one 1 (ExprTyped.var (type := (.bits 1)) (by decide)))) (Same.symm (Same.trans (Same.two (.mul 1) (Same.trans (Same.two (.mul 1) (Same.constant _ 1 (-1) (1) (by decide)) (Same.refl _ _)) (Same.one_mul 1 (ExprTyped.var (type := (.bits 1)) (by decide)))) (Same.constant _ 1 (-1) (1) (by decide))) (Same.mul_one 1 (ExprTyped.var (type := (.bits 1)) (by decide)))))) (Sames.nil _))))
theorem actual {World Error : Type} (model : Model Int World Error registry)
    (exact : ExactModel model) (laws : LocalTyping model Canonical)
    (values : List Int) (world : World) (entry : Values Canonical inputs values) :
    evalProgram model before values world = evalProgram model after values world :=
  (programs_sound model exact laws firstNormalized secondNormalized localRewrites values world entry).2
theorem reached {World Error : Type} (model : Model Int World Error registry)
    (laws : LocalTyping model Canonical) (values : List Int) (world : World)
    (entry : Values Canonical inputs values) :
    Reached model Canonical before values world ∧ Reached model Canonical after values world :=
  ⟨typed_reached model laws (checkTypes_sound _ _ _ beforeTyped) values world entry,
   typed_reached model laws (checkTypes_sound _ _ _ afterTyped) values world entry⟩
end F1
namespace F2
def inputs : List Ty := [(.bits 1)]
def outputs : List Ty := [(.bits 1)]
def before : Program := (.ret [0])
theorem beforeTyped : checkTypes registry inputs outputs before = true := by decide
def after : Program := (.ret [0])
theorem afterTyped : checkTypes registry inputs outputs after = true := by decide
def first : Code := (.ret [(.var 0)])
theorem firstNormalized : normalize registry (vars inputs.length) before = some first := by decide
def second : Code := (.ret [(.var 0)])
theorem secondNormalized : normalize registry (vars inputs.length) after = some second := by decide
theorem localRewrites : Relation registry inputs outputs first second :=
  (Relation.ret (ExprsTyped.cons (ExprTyped.var (type := (.bits 1)) (by decide)) ExprsTyped.nil) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _)))
theorem actual {World Error : Type} (model : Model Int World Error registry)
    (exact : ExactModel model) (laws : LocalTyping model Canonical)
    (values : List Int) (world : World) (entry : Values Canonical inputs values) :
    evalProgram model before values world = evalProgram model after values world :=
  (programs_sound model exact laws firstNormalized secondNormalized localRewrites values world entry).2
theorem reached {World Error : Type} (model : Model Int World Error registry)
    (laws : LocalTyping model Canonical) (values : List Int) (world : World)
    (entry : Values Canonical inputs values) :
    Reached model Canonical before values world ∧ Reached model Canonical after values world :=
  ⟨typed_reached model laws (checkTypes_sound _ _ _ beforeTyped) values world entry,
   typed_reached model laws (checkTypes_sound _ _ _ afterTyped) values world entry⟩
end F2
def originalEntries : List Merlin.GraphState.Pipeline.Entry := [{inputs := F1.inputs, outputs := F1.outputs, body := F1.before},{inputs := F2.inputs, outputs := F2.outputs, body := F2.before}]
def middleEntries : List Merlin.GraphState.Pipeline.Entry := [{inputs := F1.inputs, outputs := F1.outputs, body := F1.after},{inputs := F2.inputs, outputs := F2.outputs, body := F2.after}]
theorem prefix_actual {World Error : Type} (model : Model Int World Error registry)
    (exact : ExactModel model) (laws : LocalTyping model Canonical) :
    PreservesEntriesCanonical model originalEntries middleEntries :=
  (Merlin.GraphState.Integer.entries_cons model rfl rfl (F1.actual model exact laws) (Merlin.GraphState.Integer.entries_cons model rfl rfl (F2.actual model exact laws) (Merlin.GraphState.Integer.entries_nil model)))
end Merlin.GraphStateIntegerInstance
