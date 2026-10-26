import Merlin.GraphStateIntegerProgram
import Merlin.GraphStateIntegerTrace
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStateIntegerInstance
open Merlin.GraphState.Scalar Merlin.GraphState.Integer
def registry : List Signature := [{inputs := [(.opaque 0),(.opaque 0)],outputs := [(.opaque 0)]},{inputs := [],outputs := [(.opaque 0)]}]
namespace F0
def inputs : List Ty := [(.opaque 0)]
def outputs : List Ty := [(.opaque 0)]
def before : Program := (.opaque 1 [] [(.opaque 0)] (.opaque 0 [1,0] [(.opaque 0)] (.ret [0])))
theorem beforeTyped : checkTypes registry inputs outputs before = true := by decide
def after : Program := (.opaque 1 [] [(.opaque 0)] (.opaque 0 [1,0] [(.opaque 0)] (.ret [0])))
theorem afterTyped : checkTypes registry inputs outputs after = true := by decide
def first : Code := (.opaque 1 [] [(.opaque 0)] (.opaque 0 [(.var 1),(.var 0)] [(.opaque 0)] (.ret [(.var 0)])))
theorem firstNormalized : normalize registry (vars inputs.length) before = some first := by decide
def second : Code := (.opaque 1 [] [(.opaque 0)] (.opaque 0 [(.var 1),(.var 0)] [(.opaque 0)] (.ret [(.var 0)])))
theorem secondNormalized : normalize registry (vars inputs.length) after = some second := by decide
theorem localRewrites : Relation registry inputs outputs first second :=
  (Relation.effect (by decide) (by decide) ExprsTyped.nil (Sames.nil _) (Relation.effect (by decide) (by decide) (ExprsTyped.cons (ExprTyped.var (type := (.opaque 0)) (by decide)) (ExprsTyped.cons (ExprTyped.var (type := (.opaque 0)) (by decide)) ExprsTyped.nil)) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _))) (Relation.ret (ExprsTyped.cons (ExprTyped.var (type := (.opaque 0)) (by decide)) ExprsTyped.nil) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _)))))
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
end F0
def originalEntries : List Merlin.GraphState.Pipeline.Entry := [{inputs := F0.inputs, outputs := F0.outputs, body := F0.before}]
def middleEntries : List Merlin.GraphState.Pipeline.Entry := [{inputs := F0.inputs, outputs := F0.outputs, body := F0.after}]
theorem prefix_actual {World Error : Type} (model : Model Int World Error registry)
    (exact : ExactModel model) (laws : LocalTyping model Canonical) :
    PreservesEntriesCanonical model originalEntries middleEntries :=
  (Merlin.GraphState.Integer.entries_cons model rfl rfl (F0.actual model exact laws) (Merlin.GraphState.Integer.entries_nil model))
end Merlin.GraphStateIntegerInstance
