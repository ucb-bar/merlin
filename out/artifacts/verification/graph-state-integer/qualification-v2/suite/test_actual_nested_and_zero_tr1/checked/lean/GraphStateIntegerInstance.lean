import Merlin.GraphStateIntegerProgram
import Merlin.GraphStateIntegerTrace
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStateIntegerInstance
open Merlin.GraphState.Scalar Merlin.GraphState.Integer
def registry : List Signature := [{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [(.bits 16),(.bits 16),.index],outputs := [(.bits 16),(.bits 16)]},{inputs := [(.bits 16),(.bits 16)],outputs := []}]
namespace F3
def inputs : List Ty := [(.bits 16),(.bits 16)]
def outputs : List Ty := [(.bits 16),(.bits 16)]
def before : Program := (.opaque 0 [] [.index] (.opaque 0 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.pure (.constant 16 (0)) [] (.pure (.constant 16 (1)) [] (.pure (.constant 16 (-3)) [] (.pure (.constant 16 (3)) [] (.pure (.add 16) [1,0] (.loop 8 7 5 [9,10] [(.bits 16),(.bits 16)] (.pure (.add 16) [1,3] (.loop 12 10 9 [0,3] [(.bits 16),(.bits 16)] (.pure (.mul 16) [17,10] (.pure (.sub 16) [2,12] (.opaque 4 [0,1] [] (.opaque 3 [0,4,2] [(.bits 16),(.bits 16)] (.ret [0,1]))))) (.ret [0,1]))) (.opaque 4 [0,1] [] (.ret [1,0]))))))))))))
theorem beforeTyped : checkTypes registry inputs outputs before = true := by decide
def after : Program := (.opaque 0 [] [.index] (.opaque 0 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.pure (.constant 16 (0)) [] (.pure (.constant 16 (1)) [] (.pure (.constant 16 (-3)) [] (.pure (.constant 16 (3)) [] (.pure (.constant 16 (0)) [] (.loop 8 7 5 [9,10] [(.bits 16),(.bits 16)] (.loop 11 9 8 [1,2] [(.bits 16),(.bits 16)] (.opaque 4 [1,16] [] (.opaque 3 [1,2,0] [(.bits 16),(.bits 16)] (.ret [0,1]))) (.ret [0,1])) (.opaque 4 [0,1] [] (.ret [1,0]))))))))))))
theorem afterTyped : checkTypes registry inputs outputs after = true := by decide
def first : Code := (.opaque 0 [] [.index] (.opaque 0 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.loop (.var 3) (.var 2) (.var 0) [(.var 4),(.var 5)] [(.bits 16),(.bits 16)] (.loop (.var 6) (.var 4) (.var 3) [(.two (.add 16) (.var 1) (.two (.add 16) (.zero (.constant 16 (-3))) (.zero (.constant 16 (3))))),(.var 2)] [(.bits 16),(.bits 16)] (.opaque 4 [(.two (.sub 16) (.var 1) (.zero (.constant 16 (0)))),(.two (.mul 16) (.var 11) (.zero (.constant 16 (1))))] [] (.opaque 3 [(.two (.sub 16) (.var 1) (.zero (.constant 16 (0)))),(.var 2),(.var 0)] [(.bits 16),(.bits 16)] (.ret [(.var 0),(.var 1)]))) (.ret [(.var 0),(.var 1)])) (.opaque 4 [(.var 0),(.var 1)] [] (.ret [(.var 1),(.var 0)])))))))
theorem firstNormalized : normalize registry (vars inputs.length) before = some first := by decide
def second : Code := (.opaque 0 [] [.index] (.opaque 0 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.loop (.var 3) (.var 2) (.var 0) [(.var 4),(.var 5)] [(.bits 16),(.bits 16)] (.loop (.var 6) (.var 4) (.var 3) [(.var 1),(.var 2)] [(.bits 16),(.bits 16)] (.opaque 4 [(.var 1),(.var 11)] [] (.opaque 3 [(.var 1),(.var 2),(.var 0)] [(.bits 16),(.bits 16)] (.ret [(.var 0),(.var 1)]))) (.ret [(.var 0),(.var 1)])) (.opaque 4 [(.var 0),(.var 1)] [] (.ret [(.var 1),(.var 0)])))))))
theorem secondNormalized : normalize registry (vars inputs.length) after = some second := by decide
theorem localRewrites : Relation registry inputs outputs first second :=
  (Relation.effect (by decide) (by decide) ExprsTyped.nil (Sames.nil _) (Relation.effect (by decide) (by decide) ExprsTyped.nil (Sames.nil _) (Relation.effect (by decide) (by decide) ExprsTyped.nil (Sames.nil _) (Relation.effect (by decide) (by decide) ExprsTyped.nil (Sames.nil _) (Relation.loop (ExprTyped.var (type := .index) (by decide)) (ExprTyped.var (type := .index) (by decide)) (ExprTyped.var (type := .index) (by decide)) (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) ExprsTyped.nil)) (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _))) (Relation.loop (ExprTyped.var (type := .index) (by decide)) (ExprTyped.var (type := .index) (by decide)) (ExprTyped.var (type := .index) (by decide)) (ExprsTyped.cons (ExprTyped.two (first := (.bits 16)) (second := (.bits 16)) (by decide) (by decide) (ExprTyped.var (type := (.bits 16)) (by decide)) (ExprTyped.two (first := (.bits 16)) (second := (.bits 16)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.zero (by decide) (by decide)))) (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) ExprsTyped.nil)) (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.trans (Same.two (.add 16) (Same.refl _ _) (Same.trans (Same.two (.add 16) (Same.constant _ 16 (-3) (65533) (by decide)) (Same.constant _ 16 (3) (3) (by decide))) (Same.fold_add _ 16 (65533) (3) (0) (by decide)))) (Same.add_zero 16 (ExprTyped.var (type := (.bits 16)) (by decide)))) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _))) (Relation.effect (by decide) (by decide) (ExprsTyped.cons (ExprTyped.two (first := (.bits 16)) (second := (.bits 16)) (by decide) (by decide) (ExprTyped.var (type := (.bits 16)) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprsTyped.cons (ExprTyped.two (first := (.bits 16)) (second := (.bits 16)) (by decide) (by decide) (ExprTyped.var (type := (.bits 16)) (by decide)) (ExprTyped.zero (by decide) (by decide))) ExprsTyped.nil)) (Sames.cons (Same.trans (Same.trans (Same.two (.sub 16) (Same.refl _ _) (Same.constant _ 16 (0) (0) (by decide))) (Same.sub_zero 16 (ExprTyped.var (type := (.bits 16)) (by decide)))) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.trans (Same.two (.mul 16) (Same.refl _ _) (Same.constant _ 16 (1) (1) (by decide))) (Same.mul_one 16 (ExprTyped.var (type := (.bits 16)) (by decide)))) (Same.symm (Same.refl _ _))) (Sames.nil _))) (Relation.effect (by decide) (by decide) (ExprsTyped.cons (ExprTyped.two (first := (.bits 16)) (second := (.bits 16)) (by decide) (by decide) (ExprTyped.var (type := (.bits 16)) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) (ExprsTyped.cons (ExprTyped.var (type := .index) (by decide)) ExprsTyped.nil))) (Sames.cons (Same.trans (Same.trans (Same.two (.sub 16) (Same.refl _ _) (Same.constant _ 16 (0) (0) (by decide))) (Same.sub_zero 16 (ExprTyped.var (type := (.bits 16)) (by decide)))) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _)))) (Relation.ret (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) ExprsTyped.nil)) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _)))))) (Relation.ret (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) ExprsTyped.nil)) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _))))) (Relation.effect (by decide) (by decide) (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) ExprsTyped.nil)) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _))) (Relation.ret (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) ExprsTyped.nil)) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _))))))))))
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
end F3
namespace F4
def inputs : List Ty := [(.bits 16)]
def outputs : List Ty := [(.bits 16)]
def before : Program := (.ret [0])
theorem beforeTyped : checkTypes registry inputs outputs before = true := by decide
def after : Program := (.ret [0])
theorem afterTyped : checkTypes registry inputs outputs after = true := by decide
def first : Code := (.ret [(.var 0)])
theorem firstNormalized : normalize registry (vars inputs.length) before = some first := by decide
def second : Code := (.ret [(.var 0)])
theorem secondNormalized : normalize registry (vars inputs.length) after = some second := by decide
theorem localRewrites : Relation registry inputs outputs first second :=
  (Relation.ret (ExprsTyped.cons (ExprTyped.var (type := (.bits 16)) (by decide)) ExprsTyped.nil) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _)))
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
end F4
def originalEntries : List Merlin.GraphState.Pipeline.Entry := [{inputs := F3.inputs, outputs := F3.outputs, body := F3.before},{inputs := F4.inputs, outputs := F4.outputs, body := F4.before}]
def middleEntries : List Merlin.GraphState.Pipeline.Entry := [{inputs := F3.inputs, outputs := F3.outputs, body := F3.after},{inputs := F4.inputs, outputs := F4.outputs, body := F4.after}]
theorem prefix_actual {World Error : Type} (model : Model Int World Error registry)
    (exact : ExactModel model) (laws : LocalTyping model Canonical) :
    PreservesEntriesCanonical model originalEntries middleEntries :=
  (Merlin.GraphState.Integer.entries_cons model rfl rfl (F3.actual model exact laws) (Merlin.GraphState.Integer.entries_cons model rfl rfl (F4.actual model exact laws) (Merlin.GraphState.Integer.entries_nil model)))
end Merlin.GraphStateIntegerInstance
