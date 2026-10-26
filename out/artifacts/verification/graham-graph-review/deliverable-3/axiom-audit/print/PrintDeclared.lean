-- Graham D3 reviewer-authored audit file: statements and transitive axioms only.
import Check
import Merlin.GraphRegionsDeclaredCountermodels
open Merlin MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared
#check @actual_complete_observers
#print axioms actual_complete_observers
#check @actual_nonzero_witness
#print axioms actual_nonzero_witness
#check @actual_calls_valid
#print axioms actual_calls_valid
#check @actual_kernel_decoded
#print axioms actual_kernel_decoded
#check @actual_generic_binding
#print axioms actual_generic_binding
#check @actual_realization
#print axioms actual_realization
#check @Merlin.GraphRegionsDeclared.derived_local_contract
#print axioms Merlin.GraphRegionsDeclared.derived_local_contract
#check @Merlin.GraphRegionsDeclared.checked_calls_preserve_observers
#print axioms Merlin.GraphRegionsDeclared.checked_calls_preserve_observers
#check @Merlin.GraphRegionsDeclared.sequence_refines
#print axioms Merlin.GraphRegionsDeclared.sequence_refines
#check @Merlin.GraphRegionsDeclared.instruction_refines
#print axioms Merlin.GraphRegionsDeclared.instruction_refines
#check @Merlin.GraphRegions.checkRealization_native
#print axioms Merlin.GraphRegions.checkRealization_native
#check @Merlin.GraphRegions.checkRealization_sound
#print axioms Merlin.GraphRegions.checkRealization_sound
#check @Merlin.GraphRegions.checked_native_contract
#print axioms Merlin.GraphRegions.checked_native_contract
#check @Merlin.GraphRegionsDeclared.nofault_premise_required
#print axioms Merlin.GraphRegionsDeclared.nofault_premise_required
#check @Merlin.GraphRegionsDeclared.completion_premise_required
#print axioms Merlin.GraphRegionsDeclared.completion_premise_required
