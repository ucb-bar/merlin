import Merlin.GraphStateIntegerProgram
import Merlin.GraphStateIntegerTrace
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStateIntegerInstance
open Merlin.GraphState.Scalar Merlin.GraphState.Integer
def registry : List Signature := [{inputs := [(.bits 64),(.bits 64),(.bits 64)],outputs := []}]
namespace F1
def inputs : List Ty := [(.bits 64)]
def outputs : List Ty := [(.bits 64)]
def before : Program := (.pure (.constant 64 (0)) [] (.pure (.constant 64 (1)) [] (.pure (.constant 64 (9223372036854775807)) [] (.pure (.constant 64 (2)) [] (.pure (.add 64) [1,0] (.pure (.sub 64) [0,3] (.pure (.mul 64) [0,2] (.pure (.add 64) [6,7] (.pure (.add 64) [0,7] (.pure (.sub 64) [0,8] (.pure (.mul 64) [8,0] (.pure (.mul 64) [0,9] (.opaque 0 [7,6,5] [] (.ret [0]))))))))))))))
theorem beforeTyped : checkTypes registry inputs outputs before = true := by decide
def after : Program := (.pure (.constant 64 (0)) [] (.pure (.constant 64 (1)) [] (.pure (.constant 64 (9223372036854775807)) [] (.pure (.constant 64 (2)) [] (.pure (.constant 64 (-9223372036854775807)) [] (.pure (.constant 64 (-9223372036854775808)) [] (.pure (.constant 64 (0)) [] (.opaque 0 [2,1,0] [] (.ret [7])))))))))
theorem afterTyped : checkTypes registry inputs outputs after = true := by decide
def first : Code := (.opaque 0 [(.two (.add 64) (.zero (.constant 64 (9223372036854775807))) (.zero (.constant 64 (2)))),(.two (.sub 64) (.two (.add 64) (.zero (.constant 64 (9223372036854775807))) (.zero (.constant 64 (2)))) (.zero (.constant 64 (1)))),(.two (.mul 64) (.two (.sub 64) (.two (.add 64) (.zero (.constant 64 (9223372036854775807))) (.zero (.constant 64 (2)))) (.zero (.constant 64 (1)))) (.zero (.constant 64 (2))))] [] (.ret [(.two (.mul 64) (.two (.mul 64) (.zero (.constant 64 (1))) (.two (.sub 64) (.two (.add 64) (.two (.add 64) (.zero (.constant 64 (0))) (.var 0)) (.zero (.constant 64 (0)))) (.zero (.constant 64 (0))))) (.zero (.constant 64 (1))))]))
theorem firstNormalized : normalize registry (vars inputs.length) before = some first := by decide
def second : Code := (.opaque 0 [(.zero (.constant 64 (-9223372036854775807))),(.zero (.constant 64 (-9223372036854775808))),(.zero (.constant 64 (0)))] [] (.ret [(.var 0)]))
theorem secondNormalized : normalize registry (vars inputs.length) after = some second := by decide
theorem localRewrites : Relation registry inputs outputs first second :=
  (Relation.effect (by decide) (by decide) (ExprsTyped.cons (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprsTyped.cons (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.zero (by decide) (by decide))) (ExprsTyped.cons (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.zero (by decide) (by decide))) ExprsTyped.nil))) (Sames.cons (Same.trans (Same.trans (Same.two (.add 64) (Same.constant _ 64 (9223372036854775807) (9223372036854775807) (by decide)) (Same.constant _ 64 (2) (2) (by decide))) (Same.fold_add _ 64 (9223372036854775807) (2) (9223372036854775809) (by decide))) (Same.symm (Same.constant _ 64 (-9223372036854775807) (9223372036854775809) (by decide)))) (Sames.cons (Same.trans (Same.trans (Same.two (.sub 64) (Same.trans (Same.two (.add 64) (Same.constant _ 64 (9223372036854775807) (9223372036854775807) (by decide)) (Same.constant _ 64 (2) (2) (by decide))) (Same.fold_add _ 64 (9223372036854775807) (2) (9223372036854775809) (by decide))) (Same.constant _ 64 (1) (1) (by decide))) (Same.fold_sub _ 64 (9223372036854775809) (1) (9223372036854775808) (by decide))) (Same.symm (Same.constant _ 64 (-9223372036854775808) (9223372036854775808) (by decide)))) (Sames.cons (Same.trans (Same.trans (Same.two (.mul 64) (Same.trans (Same.two (.sub 64) (Same.trans (Same.two (.add 64) (Same.constant _ 64 (9223372036854775807) (9223372036854775807) (by decide)) (Same.constant _ 64 (2) (2) (by decide))) (Same.fold_add _ 64 (9223372036854775807) (2) (9223372036854775809) (by decide))) (Same.constant _ 64 (1) (1) (by decide))) (Same.fold_sub _ 64 (9223372036854775809) (1) (9223372036854775808) (by decide))) (Same.constant _ 64 (2) (2) (by decide))) (Same.fold_mul _ 64 (9223372036854775808) (2) (0) (by decide))) (Same.symm (Same.constant _ 64 (0) (0) (by decide)))) (Sames.nil _)))) (Relation.ret (ExprsTyped.cons (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.var (type := (.bits 64)) (by decide))) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.zero (by decide) (by decide)))) (ExprTyped.zero (by decide) (by decide))) ExprsTyped.nil) (Sames.cons (Same.trans (Same.trans (Same.two (.mul 64) (Same.trans (Same.two (.mul 64) (Same.constant _ 64 (1) (1) (by decide)) (Same.trans (Same.two (.sub 64) (Same.trans (Same.two (.add 64) (Same.trans (Same.two (.add 64) (Same.constant _ 64 (0) (0) (by decide)) (Same.refl _ _)) (Same.zero_add 64 (ExprTyped.var (type := (.bits 64)) (by decide)))) (Same.constant _ 64 (0) (0) (by decide))) (Same.add_zero 64 (ExprTyped.var (type := (.bits 64)) (by decide)))) (Same.constant _ 64 (0) (0) (by decide))) (Same.sub_zero 64 (ExprTyped.var (type := (.bits 64)) (by decide))))) (Same.one_mul 64 (ExprTyped.var (type := (.bits 64)) (by decide)))) (Same.constant _ 64 (1) (1) (by decide))) (Same.mul_one 64 (ExprTyped.var (type := (.bits 64)) (by decide)))) (Same.symm (Same.refl _ _))) (Sames.nil _))))
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
def inputs : List Ty := [(.bits 64)]
def outputs : List Ty := [(.bits 64)]
def before : Program := (.ret [0])
theorem beforeTyped : checkTypes registry inputs outputs before = true := by decide
def after : Program := (.ret [0])
theorem afterTyped : checkTypes registry inputs outputs after = true := by decide
def first : Code := (.ret [(.var 0)])
theorem firstNormalized : normalize registry (vars inputs.length) before = some first := by decide
def second : Code := (.ret [(.var 0)])
theorem secondNormalized : normalize registry (vars inputs.length) after = some second := by decide
theorem localRewrites : Relation registry inputs outputs first second :=
  (Relation.ret (ExprsTyped.cons (ExprTyped.var (type := (.bits 64)) (by decide)) ExprsTyped.nil) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _)))
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
