/- Geometry checks are independent of scalar-body matching. -/
import MerlinVerify.Rules.DequantTensor

namespace MerlinVerify.Tests.DequantTensor

open MerlinVerify MerlinVerify.Rules.DequantTensor MerlinVerify.Sem.Ops.Linalg

#guard checkLayout false 0 [] [] (mapsFor false 0 0) []
#guard checkLayout false 0 [2, 3] [] (mapsFor false 2 0) (List.replicate 2 .parallel)
#guard checkLayout true 1 [2, 3] [3] (mapsFor true 2 1) (List.replicate 2 .parallel)
#guard checkLayout true 0 [0, 3] [0] (mapsFor true 2 0) (List.replicate 2 .parallel)
#guard !checkLayout true 2 [2, 3] [] (mapsFor true 2 2) (List.replicate 2 .parallel)
#guard !checkLayout true 1 [2, 3] [2] (mapsFor true 2 1) (List.replicate 2 .parallel)
#guard !checkLayout true 1 [2, 3] [3] (mapsFor true 2 0) (List.replicate 2 .parallel)
#guard !checkLayout false 0 [2, 3] [1] (mapsFor false 2 0) (List.replicate 2 .parallel)
#guard !checkLayout false 0 [2, 3] [] (mapsFor false 2 0) [.parallel, .reduction]
#guard !checkLayout false 0 [2, 3] [] [parameterMap false 2 0, identityMap 2, parameterMap false 2 0, identityMap 2]
  (List.replicate 2 .parallel)
#guard !checkLayout false 0 [2, 3] [] (mapsFor false 2 0).dropLast (List.replicate 2 .parallel)
#guard !checkLayout false 0 [2, 3] []
  ((mapsFor false 2 0).map fun m => { m with numSyms := 1 }) (List.replicate 2 .parallel)

end MerlinVerify.Tests.DequantTensor
