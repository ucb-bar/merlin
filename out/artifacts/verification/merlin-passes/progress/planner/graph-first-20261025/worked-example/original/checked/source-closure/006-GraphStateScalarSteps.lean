/- Small checked operational steps for finite concrete witnesses. These avoid
duplicating a whole closed execution during kernel conversion. Each primitive/
opaque/loop equation is supplied by a kernel-checked local computation; none is
an assumption about source success or a replacement for the complete evaluator. -/
import Merlin.GraphStateScalarContract

namespace Merlin.GraphState.Scalar
set_option autoImplicit false

theorem pure_execution {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (op : Prim) (arguments : List Nat) (next : Program)
    (environment : List Value) (world : World) (value : Value)
    (computed : model.primitive op (reads model.default environment arguments) = value) :
    evalProgram model (.pure op arguments next) environment world = evalProgram model next (value::environment) world := by
  simp only [evalProgram,computed]

theorem effect_execution {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (identity : Nat) (arguments : List Nat) (outputs : List Ty) (next : Program)
    (environment : List Value) (world : World) (values : List Value) (final : World)
    (computed : opaqueStep model identity (reads model.default environment arguments) world = .ok (values,final)) :
    evalProgram model (.opaque identity arguments outputs next) environment world =
      evalProgram model next (values++environment) final := by
  simp only [evalProgram,computed,Except.bind]

theorem effect_failed {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (identity : Nat) (arguments : List Nat) (outputs : List Ty) (next : Program)
    (environment : List Value) (world : World) (failure : Fault Error × World)
    (computed : opaqueStep model identity (reads model.default environment arguments) world = .error failure) :
    evalProgram model (.opaque identity arguments outputs next) environment world = .error failure := by
  simp only [evalProgram,computed,Except.bind]

theorem loop_done {Value World Error : Type} (count : Nat)
    (body : Int → List Value → World → Outcome Value World Error) (values : List Value) (world : World)
    (arity : values.length = count) : loopEval count body [] values world = .ok (values,world) := by
  simp only [loopEval,ite_eq_left arity]

theorem loop_step {Value World Error : Type} (count : Nat)
    (body : Int → List Value → World → Outcome Value World Error) (index : Int) (later : List Int)
    (values : List Value) (world : World) (next : List Value) (final : World)
    (arity : values.length = count) (computed : body index values world = .ok (next,final)) :
    loopEval count body (index::later) values world = loopEval count body later next final := by
  simp only [loopEval,ite_eq_left arity,computed,Except.bind]

theorem loop_step_failed {Value World Error : Type} (count : Nat)
    (body : Int → List Value → World → Outcome Value World Error) (index : Int) (later : List Int)
    (values : List Value) (world : World) (failure : Fault Error × World)
    (arity : values.length = count) (computed : body index values world = .error failure) :
    loopEval count body (index::later) values world = .error failure := by
  simp only [loopEval,ite_eq_left arity,computed,Except.bind]

theorem loop_execution {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (lower upper stride : Nat) (initial : List Nat)
    (carries : List Ty) (body next : Program) (environment : List Value) (world : World)
    (values : List Value) (final : World)
    (computed : loopEval carries.length
      (fun index carry current => evalProgram model body (model.index index::carry++environment) current)
      (model.iterations (environment[lower]?.getD model.default) (environment[upper]?.getD model.default)
        (environment[stride]?.getD model.default)) (reads model.default environment initial) world = .ok (values,final)) :
    evalProgram model (.loop lower upper stride initial carries body next) environment world =
      evalProgram model next (values++environment) final := by
  simp only [evalProgram,computed,Except.bind]

theorem loop_failed {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (lower upper stride : Nat) (initial : List Nat)
    (carries : List Ty) (body next : Program) (environment : List Value) (world : World) (failure : Fault Error × World)
    (computed : loopEval carries.length
      (fun index carry current => evalProgram model body (model.index index::carry++environment) current)
      (model.iterations (environment[lower]?.getD model.default) (environment[upper]?.getD model.default)
        (environment[stride]?.getD model.default)) (reads model.default environment initial) world = .error failure) :
    evalProgram model (.loop lower upper stride initial carries body next) environment world = .error failure := by
  simp only [evalProgram,computed,Except.bind]

end Merlin.GraphState.Scalar
