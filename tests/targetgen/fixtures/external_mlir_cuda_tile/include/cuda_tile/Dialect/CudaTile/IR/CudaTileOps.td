// Synthetic fixture — do not represent real CUDA Tile.

include "mlir/IR/OpBase.td"

def CudaTile_Dialect : Dialect {
  let name = "cuda_tile";
  let cppNamespace = "::cuda_tile";
}

class CudaTile_Op<string mnemonic, list<Trait> traits = []>
    : Op<CudaTile_Dialect, mnemonic, traits>;

def CudaTile_TileMatmulOp : CudaTile_Op<"tile_matmul", []> {
  let summary = "tile-level matmul";
}
