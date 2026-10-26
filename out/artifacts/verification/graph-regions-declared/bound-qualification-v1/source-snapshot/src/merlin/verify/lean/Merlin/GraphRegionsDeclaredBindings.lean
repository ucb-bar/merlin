/- Additive strengthening over the immutable 46-case declared-call cut.
Bind source live-in/public output ordering to the independently retained original
ABI windows. These checks do not prove the runtime ownership/packing contract. -/
import Merlin.GraphRegionsDeclaredRaw
import Merlin.GraphRegionsDeclaredProjection

namespace Merlin.GraphRegionsDeclared
open Merlin.GraphRegions

structure ExternalBindings where
  inputNames : List String
  inputs : List Window
  outputNames : List String
  outputs : List Window
  byteOrders : List String
  declaresDisjoint : Bool
  deriving DecidableEq, Repr

def originalBindingsValid (region suffix : List Node) (observations : List Nat)
    (addresses : Nat → Window) (shape : List Nat) (width : Nat)
    (arguments : List TensorArgument) (bindings : ExternalBindings) : Prop :=
  let inputs := (liveIns region).eraseDups
  let outputs := liveOuts region suffix observations
  inputs.map addresses = bindings.inputs ∧ outputs.map addresses = bindings.outputs ∧
  bindings.inputNames.length = inputs.length ∧ bindings.outputNames.length = outputs.length ∧
  arguments.map (·.name) = bindings.inputNames ++ bindings.outputNames ∧
  arguments.map (·.shape) = List.replicate arguments.length shape ∧
  arguments.map (·.dtype) = List.replicate arguments.length (typeTag (.int width)) ∧
  arguments.map (·.access) = List.replicate inputs.length "read" ++ List.replicate outputs.length "write" ∧
  bindings.byteOrders = List.replicate arguments.length "little" ∧ bindings.declaresDisjoint = true

instance (region suffix : List Node) (observations : List Nat) (addresses : Nat → Window)
    (shape : List Nat) (width : Nat) (arguments : List TensorArgument) (bindings : ExternalBindings) :
    Decidable (originalBindingsValid region suffix observations addresses shape width arguments bindings) := by
  unfold originalBindingsValid; infer_instance

def checkOriginalBindings (region suffix : List Node) (observations : List Nat)
    (addresses : Nat → Window) (shape : List Nat) (width : Nat)
    (arguments : List TensorArgument) (bindings : ExternalBindings) : Bool :=
  decide (originalBindingsValid region suffix observations addresses shape width arguments bindings)

/-- Physical initial/public correspondence is recomputed from original ABI
declarations and actual graph liveness. The disjoint flag is a DECLARATION;
runtime disjointness, instruction fidelity and completion remain premises. -/
theorem checkOriginalBindings_sound {region suffix : List Node} {observations : List Nat}
    {addresses : Nat → Window} {shape : List Nat} {width : Nat}
    {arguments : List TensorArgument} {bindings : ExternalBindings}
    (accepted : checkOriginalBindings region suffix observations addresses shape width arguments bindings = true) :
    ((liveIns region).eraseDups).map addresses = bindings.inputs ∧
      (liveOuts region suffix observations).map addresses = bindings.outputs := by
  have valid : originalBindingsValid region suffix observations addresses shape width arguments bindings :=
    of_decide_eq_true accepted
  exact ⟨valid.1, valid.2.1⟩

end Merlin.GraphRegionsDeclared
