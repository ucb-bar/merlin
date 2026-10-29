-- Graham D3 reviewer-authored audit file: statements and transitive axioms only.
import Merlin.GraphLoopsMemorySCFBounds
import Merlin.GraphLoopsMemorySCFWitness
import Merlin.GraphLoopsReuseCountermodels
#check @Merlin.GraphLoopsMemorySCF.EvalPure.frame
#print axioms Merlin.GraphLoopsMemorySCF.EvalPure.frame
#check @Merlin.GraphLoopsMemorySCF.Run.covers
#print axioms Merlin.GraphLoopsMemorySCF.Run.covers
#check @Merlin.GraphLoopsMemorySCF.Run.materializes
#print axioms Merlin.GraphLoopsMemorySCF.Run.materializes
#check @Merlin.GraphLoopsMemorySCF.heap_run_iff_log
#print axioms Merlin.GraphLoopsMemorySCF.heap_run_iff_log
#check @Merlin.GraphLoopsMemorySCF.segmentation_exact_heaps
#print axioms Merlin.GraphLoopsMemorySCF.segmentation_exact_heaps
#check @Merlin.GraphLoopsMemorySCF.materializes_in_bounds
#print axioms Merlin.GraphLoopsMemorySCF.materializes_in_bounds
#check @Merlin.GraphLoopsMemorySCF.related_initialize_logs
#print axioms Merlin.GraphLoopsMemorySCF.related_initialize_logs
#check @Merlin.GraphLoopsMemorySCF.initialization_induced
#print axioms Merlin.GraphLoopsMemorySCF.initialization_induced
#check @Merlin.GraphLoopsMemorySCF.step_forward
#print axioms Merlin.GraphLoopsMemorySCF.step_forward
#check @Merlin.GraphLoopsMemorySCF.step_backward
#print axioms Merlin.GraphLoopsMemorySCF.step_backward
#check @Merlin.GraphLoopsMemorySCF.execution_forward
#print axioms Merlin.GraphLoopsMemorySCF.execution_forward
#check @Merlin.GraphLoopsMemorySCF.execution_backward
#print axioms Merlin.GraphLoopsMemorySCF.execution_backward
#check @Merlin.GraphLoopsMemorySCF.checked_complete_observations
#print axioms Merlin.GraphLoopsMemorySCF.checked_complete_observations
#check @Merlin.GraphLoopsMemorySCF.execution_modes
#print axioms Merlin.GraphLoopsMemorySCF.execution_modes
#check @Merlin.GraphLoopsMemorySCF.checked_reuse_then_segment
#print axioms Merlin.GraphLoopsMemorySCF.checked_reuse_then_segment
#check @Merlin.GraphLoopsMemorySCF.checked_induced_atomic
#print axioms Merlin.GraphLoopsMemorySCF.checked_induced_atomic
#check @Merlin.GraphLoopsMemorySCF.Model.check_sound
#print axioms Merlin.GraphLoopsMemorySCF.Model.check_sound
#check @Merlin.GraphLoopsMemorySCF.Model.checked_chain
#print axioms Merlin.GraphLoopsMemorySCF.Model.checked_chain
#check @Merlin.GraphLoopsMemorySCF.Model.checked_complete
#print axioms Merlin.GraphLoopsMemorySCF.Model.checked_complete
#check @Merlin.GraphLoopsMemorySCF.Model.checked_context
#print axioms Merlin.GraphLoopsMemorySCF.Model.checked_context
#check @Merlin.GraphLoopsMemorySCF.Witness.uniform_heap
#print axioms Merlin.GraphLoopsMemorySCF.Witness.uniform_heap
#check @Merlin.GraphLoopsMemorySCF.Witness.uniform_run_exists
#print axioms Merlin.GraphLoopsMemorySCF.Witness.uniform_run_exists
#check @Merlin.GraphLoopsMemorySCF.Witness.uniform_run_values
#print axioms Merlin.GraphLoopsMemorySCF.Witness.uniform_run_values
#check @Merlin.GraphLoopsMemorySCF.Witness.writes_uniform
#print axioms Merlin.GraphLoopsMemorySCF.Witness.writes_uniform
#check @Merlin.GraphLoopsMemorySCF.Model.checked_outer_control
#print axioms Merlin.GraphLoopsMemorySCF.Model.checked_outer_control
#check @Merlin.GraphLoopsMemorySCF.Model.checked_inner_control
#print axioms Merlin.GraphLoopsMemorySCF.Model.checked_inner_control
#check @Merlin.GraphLoopsMemorySCF.Model.checked_initializer_byte_offset
#print axioms Merlin.GraphLoopsMemorySCF.Model.checked_initializer_byte_offset
#check @Merlin.GraphLoopsReuse.Witness.protocol
#print axioms Merlin.GraphLoopsReuse.Witness.protocol
#check @Merlin.GraphLoopsReuse.Witness.source_inhabited
#print axioms Merlin.GraphLoopsReuse.Witness.source_inhabited
#check @Merlin.GraphLoopsReuse.Witness.target_inhabited
#print axioms Merlin.GraphLoopsReuse.Witness.target_inhabited
#check @Merlin.GraphLoopsReuse.checked_observations
#print axioms Merlin.GraphLoopsReuse.checked_observations
#check @Merlin.GraphLoopsReuse.Model.checked_complete_observations
#print axioms Merlin.GraphLoopsReuse.Model.checked_complete_observations
#check @Merlin.GraphLoopsReuse.Model.checked_context
#print axioms Merlin.GraphLoopsReuse.Model.checked_context
#check @Merlin.GraphLoopsReuse.Countermodels.resource_exhaustion
#print axioms Merlin.GraphLoopsReuse.Countermodels.resource_exhaustion
#check @Merlin.GraphLoopsReuse.Countermodels.pointer_identity
#print axioms Merlin.GraphLoopsReuse.Countermodels.pointer_identity
#check @Merlin.GraphLoopsReuse.Countermodels.allocation_call_budget
#print axioms Merlin.GraphLoopsReuse.Countermodels.allocation_call_budget
#check @Merlin.GraphLoopsReuse.Countermodels.stale_cell
#print axioms Merlin.GraphLoopsReuse.Countermodels.stale_cell
#check @Merlin.GraphLoopsReuse.Countermodels.overlapping_live_values
#print axioms Merlin.GraphLoopsReuse.Countermodels.overlapping_live_values
