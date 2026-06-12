# ToyNPU dialect (MLIR/C++)

Complete, idiomatic ODS + C++ for the `toynpu` dialect: real type defs, op defs with
assembly formats and traits, a real `CommitOp::verify()`, and `add_mlir_dialect` wiring.

**Build dependency:** compiling this requires an MLIR/LLVM build (TableGen `mlir-tblgen` and
the MLIR CMake modules), which is not bundled in this scaffold. Point CMake at an MLIR install
(`-DMLIR_DIR=<path>/lib/cmake/mlir`) and add these directories to the build. The code is
written to that contract; the generator does not invoke `mlir-tblgen` itself.
