// LinalgToCudaTileText.h — Registration entry point.

#ifndef MERLIN_CODEGEN_CUDATILE_LINALGTOCUDATILETEXT_H
#define MERLIN_CODEGEN_CUDATILE_LINALGTOCUDATILETEXT_H

namespace mlir::iree_compiler {

/// Register the --merlin-linalg-to-cuda-tile-text pass with the MLIR pass
/// infrastructure.  Called from the Merlin IREE compiler plugin.
void registerLinalgToCudaTileTextPass();

} // namespace mlir::iree_compiler

#endif // MERLIN_CODEGEN_CUDATILE_LINALGTOCUDATILETEXT_H
