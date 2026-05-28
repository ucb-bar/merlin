cmake_minimum_required(VERSION 3.20)
project(cuda_tile_fixture LANGUAGES CXX)

add_mlir_dialect(CudaTileOps cuda_tile)
mlir_tablegen(CudaTilePasses.h.inc -gen-pass-decls)
add_mlir_library(MLIRCudaTile CudaTileDialect.cpp)
