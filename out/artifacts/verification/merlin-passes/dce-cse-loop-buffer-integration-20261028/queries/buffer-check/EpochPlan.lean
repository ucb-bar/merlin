import EpochCommands
namespace GraphShapeEpochObligation
set_option autoImplicit false
set_option maxRecDepth 16384
set_option maxHeartbeats 4000000
open Merlin Merlin.GraphLoopsShape
theorem plan_valid : decide plan.Valid = true := by decide

end GraphShapeEpochObligation
