/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Gemmini.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace merlin::gemmini {

/// Basic dialect to target RISC-V Gemmini extension
/// Gemmini is an accelerator based on systolic array architecture.
///     For more details about Gemmini,
///     please see the [Gemmini repo](https://github.com/ucb-bar/gemmini).
class GemminiDialect : public ::mlir::Dialect {
  explicit GemminiDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~GemminiDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("gemmini");
  }
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::GemminiDialect)
