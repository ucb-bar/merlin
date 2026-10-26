import Merlin.GraphStateIntegerProgram
import Merlin.GraphStateIntegerTrace
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStateIntegerInstance
open Merlin.GraphState.Scalar Merlin.GraphState.Integer
def registry : List Signature := [{inputs := [(.bits 128),(.bits 128),(.bits 128)],outputs := []}]
namespace F1
def inputs : List Ty := [(.bits 128)]
def outputs : List Ty := [(.bits 128)]
def before : Program := (.pure (.constant 128 (0)) [] (.pure (.constant 128 (1)) [] (.pure (.constant 128 (170141183460469231731687303715884105727)) [] (.pure (.constant 128 (2)) [] (.pure (.add 128) [1,0] (.pure (.sub 128) [0,3] (.pure (.mul 128) [0,2] (.pure (.add 128) [6,7] (.pure (.add 128) [0,7] (.pure (.sub 128) [0,8] (.pure (.mul 128) [8,0] (.pure (.mul 128) [0,9] (.opaque 0 [7,6,5] [] (.ret [0]))))))))))))))
theorem beforeTyped : checkTypes registry inputs outputs before = true := by decide
def after : Program := (.pure (.constant 128 (0)) [] (.pure (.constant 128 (1)) [] (.pure (.constant 128 (170141183460469231731687303715884105727)) [] (.pure (.constant 128 (2)) [] (.pure (.constant 128 (-170141183460469231731687303715884105727)) [] (.pure (.constant 128 (-170141183460469231731687303715884105728)) [] (.pure (.constant 128 (0)) [] (.opaque 0 [2,1,0] [] (.ret [7])))))))))
theorem afterTyped : checkTypes registry inputs outputs after = true := by decide
def first : Code := (.opaque 0 [(.two (.add 128) (.zero (.constant 128 (170141183460469231731687303715884105727))) (.zero (.constant 128 (2)))),(.two (.sub 128) (.two (.add 128) (.zero (.constant 128 (170141183460469231731687303715884105727))) (.zero (.constant 128 (2)))) (.zero (.constant 128 (1)))),(.two (.mul 128) (.two (.sub 128) (.two (.add 128) (.zero (.constant 128 (170141183460469231731687303715884105727))) (.zero (.constant 128 (2)))) (.zero (.constant 128 (1)))) (.zero (.constant 128 (2))))] [] (.ret [(.two (.mul 128) (.two (.mul 128) (.zero (.constant 128 (1))) (.two (.sub 128) (.two (.add 128) (.two (.add 128) (.zero (.constant 128 (0))) (.var 0)) (.zero (.constant 128 (0)))) (.zero (.constant 128 (0))))) (.zero (.constant 128 (1))))]))
theorem firstNormalized : normalize registry (vars inputs.length) before = some first := by decide
def second : Code := (.opaque 0 [(.zero (.constant 128 (-170141183460469231731687303715884105727))),(.zero (.constant 128 (-170141183460469231731687303715884105728))),(.zero (.constant 128 (0)))] [] (.ret [(.var 0)]))
theorem secondNormalized : normalize registry (vars inputs.length) after = some second := by decide
theorem localRewrites : Relation registry inputs outputs first second :=
  (Relation.effect (by decide) (by decide) (ExprsTyped.cons (ExprTyped.two (first := (.bits 128)) (second := (.bits 128)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprsTyped.cons (ExprTyped.two (first := (.bits 128)) (second := (.bits 128)) (by decide) (by decide) (ExprTyped.two (first := (.bits 128)) (second := (.bits 128)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.zero (by decide) (by decide))) (ExprsTyped.cons (ExprTyped.two (first := (.bits 128)) (second := (.bits 128)) (by decide) (by decide) (ExprTyped.two (first := (.bits 128)) (second := (.bits 128)) (by decide) (by decide) (ExprTyped.two (first := (.bits 128)) (second := (.bits 128)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.zero (by decide) (by decide))) ExprsTyped.nil))) (Sames.cons (Same.trans (Same.trans (Same.two (.add 128) (Same.constant _ 128 (170141183460469231731687303715884105727) (170141183460469231731687303715884105727) (by decide)) (Same.constant _ 128 (2) (2) (by decide))) (Same.fold_add _ 128 (170141183460469231731687303715884105727) (2) (170141183460469231731687303715884105729) (by decide))) (Same.symm (Same.constant _ 128 (-170141183460469231731687303715884105727) (170141183460469231731687303715884105729) (by decide)))) (Sames.cons (Same.trans (Same.trans (Same.two (.sub 128) (Same.trans (Same.two (.add 128) (Same.constant _ 128 (170141183460469231731687303715884105727) (170141183460469231731687303715884105727) (by decide)) (Same.constant _ 128 (2) (2) (by decide))) (Same.fold_add _ 128 (170141183460469231731687303715884105727) (2) (170141183460469231731687303715884105729) (by decide))) (Same.constant _ 128 (1) (1) (by decide))) (Same.fold_sub _ 128 (170141183460469231731687303715884105729) (1) (170141183460469231731687303715884105728) (by decide))) (Same.symm (Same.constant _ 128 (-170141183460469231731687303715884105728) (170141183460469231731687303715884105728) (by decide)))) (Sames.cons (Same.trans (Same.trans (Same.two (.mul 128) (Same.trans (Same.two (.sub 128) (Same.trans (Same.two (.add 128) (Same.constant _ 128 (170141183460469231731687303715884105727) (170141183460469231731687303715884105727) (by decide)) (Same.constant _ 128 (2) (2) (by decide))) (Same.fold_add _ 128 (170141183460469231731687303715884105727) (2) (170141183460469231731687303715884105729) (by decide))) (Same.constant _ 128 (1) (1) (by decide))) (Same.fold_sub _ 128 (170141183460469231731687303715884105729) (1) (170141183460469231731687303715884105728) (by decide))) (Same.constant _ 128 (2) (2) (by decide))) (Same.fold_mul _ 128 (170141183460469231731687303715884105728) (2) (0) (by decide))) (Same.symm (Same.constant _ 128 (0) (0) (by decide)))) (Sames.nil _)))) (Relation.ret (ExprsTyped.cons (ExprTyped.two (first := (.bits 128)) (second := (.bits 128)) (by decide) (by decide) (ExprTyped.two (first := (.bits 128)) (second := (.bits 128)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.two (first := (.bits 128)) (second := (.bits 128)) (by decide) (by decide) (ExprTyped.two (first := (.bits 128)) (second := (.bits 128)) (by decide) (by decide) (ExprTyped.two (first := (.bits 128)) (second := (.bits 128)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.var (type := (.bits 128)) (by decide))) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.zero (by decide) (by decide)))) (ExprTyped.zero (by decide) (by decide))) ExprsTyped.nil) (Sames.cons (Same.trans (Same.trans (Same.two (.mul 128) (Same.trans (Same.two (.mul 128) (Same.constant _ 128 (1) (1) (by decide)) (Same.trans (Same.two (.sub 128) (Same.trans (Same.two (.add 128) (Same.trans (Same.two (.add 128) (Same.constant _ 128 (0) (0) (by decide)) (Same.refl _ _)) (Same.zero_add 128 (ExprTyped.var (type := (.bits 128)) (by decide)))) (Same.constant _ 128 (0) (0) (by decide))) (Same.add_zero 128 (ExprTyped.var (type := (.bits 128)) (by decide)))) (Same.constant _ 128 (0) (0) (by decide))) (Same.sub_zero 128 (ExprTyped.var (type := (.bits 128)) (by decide))))) (Same.one_mul 128 (ExprTyped.var (type := (.bits 128)) (by decide)))) (Same.constant _ 128 (1) (1) (by decide))) (Same.mul_one 128 (ExprTyped.var (type := (.bits 128)) (by decide)))) (Same.symm (Same.refl _ _))) (Sames.nil _))))
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
def inputs : List Ty := [(.bits 128)]
def outputs : List Ty := [(.bits 128)]
def before : Program := (.ret [0])
theorem beforeTyped : checkTypes registry inputs outputs before = true := by decide
def after : Program := (.ret [0])
theorem afterTyped : checkTypes registry inputs outputs after = true := by decide
def first : Code := (.ret [(.var 0)])
theorem firstNormalized : normalize registry (vars inputs.length) before = some first := by decide
def second : Code := (.ret [(.var 0)])
theorem secondNormalized : normalize registry (vars inputs.length) after = some second := by decide
theorem localRewrites : Relation registry inputs outputs first second :=
  (Relation.ret (ExprsTyped.cons (ExprTyped.var (type := (.bits 128)) (by decide)) ExprsTyped.nil) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _)))
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
