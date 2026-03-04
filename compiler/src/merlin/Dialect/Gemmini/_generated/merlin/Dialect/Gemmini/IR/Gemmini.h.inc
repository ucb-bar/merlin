/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Gemmini.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace merlin::gemmini {

/// compute accumulated opertion
/// - Local scratchpad address(systolic array single-axis addressed) of A matrix
///     - local scratchpad address(systolic array single-axis addressed) of B matrix
///       (when output-stationary), or D matrix (when weight-stationary)
///     - Number of rows of A matrix
///     - Number of columns of A matrix
///     - Number of rows of B/D matrix
///     - Number of columns of B/D matrix
class ComputeAccumulatedOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// compute operation
/// Explicitly Preloaded
///     - Local scratchpad address(systolic array single-axis addressed) of A matrix
///     - local scratchpad address(systolic array single-axis addressed) of B matrix
///       (when output-stationary), or D matrix (when weight-stationary)
///     - Number of rows of A matrix
///     - Number of columns of A matrix
///     - Number of rows of B/D matrix
///     - Number of columns of B/D matrix
class ComputePreloadedOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// ConfigExOp configures the execute pipeline
/// ConfigExOp configures the execute pipeline.
///     - dataflow: output-stationary (0) or weight-stationary (1) dataflow
///     - sysAct: activation function relu (1) or no activation function (0)
///     - sysShift: the number of bits by which the accumulated result of a matmul
///                 is right-shifted when leaving the systolic array.
///     - sysAccScale: the scalar value by which we scale the accType output of the
///                    accumulator down to inputType values when reading from the
///                    accumulator.
///                    (In the default config, rs1[63:32] is of type float32)
///     - cStride: TODO
///     - aStride: the stride (in scratchpad addresses) by which the rows of A are
///                fed into the systolic array.  "A" in this context refers to the
///                left-hand matrix A in the matmul represented by A * B = C.
///                If this stride is 1, then we feed consecutive rows in the
///                scratchpad, starting from the starting address of A, into the
///                systolic array as the A matrix. If the stride is 2, then we feed
///                every other row into the systolic array instead.
///     - aTranspose: transpose A
///     - bTranspose: transpose B
///     - setOnlyStrides: TODO
class ConfigExOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// Config load operation
/// TODO: consider the attribute type according to Gemmini spec.
class ConfigLdOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// ConfigNormOp configures normalize pipeline
/// ConfigNormOp configures normalize pipeline
///     -qConst: A constant value used for quantization during normalization.
///     -qConstType: Defines the type of the qConst.
///     -setStatsIdOnly: A flag to indicate if only the StatsId should be set.
///     -actMsg: A message related to the normalization activity.
///     -StatsId: An identifier associated with the statistics or metrics of the normalization process.
///     -igeluQb: A parameter related to the IGELU function for quantization. Specifies the 'b' value.
///     -igeluQc: Another parameter related to the IGELU function for quantization. Specifies the 'c' value.
class ConfigNormOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// Config store operation
/// TODO: consider the attribute type according to Gemmini spec.
class ConfigStOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// Flush op
/// Flush operation flushes the TLB.
///     If skip = 1, the current TLB request is skipped.
///     Otherwise, the current TLB request is repeated.
class FlushOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class ComputeAccumulated_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class ComputePreloaded_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class ConfigEX_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class ConfigNorm_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class ConfigSt_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class ConifgLd_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class Flush_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopConvWsConfig1_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopConvWsConfig2_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopConvWsConfig3_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopConvWsConfig4_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopConvWsConfig5_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopConvWsConfig6_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopConvWs_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopWsConfigAddrsAB_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopWsConfigAddrsDC_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopWsConfigBounds_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopWsConfigStridesAB_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopWsConfigStridesDC_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class LoopWs_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class Mvin2_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class Mvin3_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class Mvin_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class Mvout_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

class Preload_IntrOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// Load operation
/// Similar to Mvin
///           Move data from main memory to scratchpad
///       - MemRef to load in.
///         (including DRAM address, number of columns, number of rows)
///       - Local scratchpad or accumulator address.
class Mvin2Op;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// Load operation
/// Similar to Mvin and Mvin2
///           Move data from main memory to scratchpad
///       - MemRef to load in.
///         (including DRAM address, number of columns, number of rows)
///       - Local scratchpad or accumulator address.
class Mvin3Op;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// Load operation
/// Move data from main memory to scratchpad
///     - MemRef to load in.
///       (including DRAM address, number of columns, number of rows)
///     - Local scratchpad or accumulator address.
class MvinOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// Store operation
/// Move data from scratchpad to L2/DRAM
///     - MemRef to store from scratchpad.
///       (including DRAM address, number of columns, number of rows)
///     - Local scratchpad address.
class MvoutOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// Preload matrix in scratchpad
/// - Local scratchpad address of D matrix (when output-stationary),
///       or B matrix (when weight-stationary)
///     - Local scratchpad address of C matrix. If this is set to all high bits,
///       then C will not be written to the scratchpad or accumulator.
///     - Number of rows of D/B matrix.
///     - Number of columns of D/B matrix.
///     - Number of rows of C matrix.
///     - Number of columns of C matrix.
class PreloadOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// Preload zeros in scratchpad
/// - Local scratchpad address.
///     - Number of rows
///     - Number of columns.
class PreloadZerosOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// Print memref value.
class PrintOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// Perform convolution.
/// TODO: consider the attribute type according to Gemmini spec.
class TileConvOp;

} // namespace merlin::gemmini
namespace merlin::gemmini {

/// Perform matrix multiplication.
/// TODO: consider the attribute type according to Gemmini spec.
class TileMatMulOp;

} // namespace merlin::gemmini
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::ComputeAccumulatedOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ComputeAccumulatedOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ComputeAccumulatedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.compute_accumulated", odsAttrs.getContext());
  }

  ComputeAccumulatedOpGenericAdaptorBase(ComputeAccumulatedOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ComputeAccumulatedOpGenericAdaptor : public detail::ComputeAccumulatedOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ComputeAccumulatedOpGenericAdaptorBase;
  static constexpr int odsIndex_aAddr = 0;
  static constexpr int odsIndex_bdAddr = 1;
  static constexpr int odsIndex_aRows = 2;
  static constexpr int odsIndex_aCols = 3;
  static constexpr int odsIndex_bdRows = 4;
  static constexpr int odsIndex_bdCols = 5;
public:
  ComputeAccumulatedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ComputeAccumulatedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ComputeAccumulatedOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ComputeAccumulatedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ComputeAccumulatedOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ComputeAccumulatedOpGenericAdaptor(RangeT values, const ComputeAccumulatedOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ComputeAccumulatedOp, typename = std::enable_if_t<std::is_same_v<LateInst, ComputeAccumulatedOp>>>
  ComputeAccumulatedOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAAddr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getBdAddr() {
    return (*getODSOperands(1).begin());
  }

  ValueT getARows() {
    return (*getODSOperands(2).begin());
  }

  ValueT getACols() {
    return (*getODSOperands(3).begin());
  }

  ValueT getBdRows() {
    return (*getODSOperands(4).begin());
  }

  ValueT getBdCols() {
    return (*getODSOperands(5).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ComputeAccumulatedOpAdaptor : public ComputeAccumulatedOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ComputeAccumulatedOpGenericAdaptor::ComputeAccumulatedOpGenericAdaptor;
  ComputeAccumulatedOpAdaptor(ComputeAccumulatedOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ComputeAccumulatedOp : public ::mlir::Op<ComputeAccumulatedOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ComputeAccumulatedOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ComputeAccumulatedOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_aAddr = 0;
  static constexpr int odsIndex_bdAddr = 1;
  static constexpr int odsIndex_aRows = 2;
  static constexpr int odsIndex_aCols = 3;
  static constexpr int odsIndex_bdRows = 4;
  static constexpr int odsIndex_bdCols = 5;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.compute_accumulated");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getAAddr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getBdAddr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getARows() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getACols() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(3).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getBdRows() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(4).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getBdCols() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(5).begin());
  }

  ::mlir::OpOperand &getAAddrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBdAddrMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getARowsMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getAColsMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBdRowsMutable() {
    auto range = getODSOperandIndexAndLength(4);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBdColsMutable() {
    auto range = getODSOperandIndexAndLength(5);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols);
  static ComputeAccumulatedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols);
  static ComputeAccumulatedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols);
  static ComputeAccumulatedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols);
  static ComputeAccumulatedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ComputeAccumulatedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ComputeAccumulatedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ComputeAccumulatedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ComputeAccumulatedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::ComputeAccumulatedOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::ComputePreloadedOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ComputePreloadedOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ComputePreloadedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.compute_preloaded", odsAttrs.getContext());
  }

  ComputePreloadedOpGenericAdaptorBase(ComputePreloadedOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ComputePreloadedOpGenericAdaptor : public detail::ComputePreloadedOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ComputePreloadedOpGenericAdaptorBase;
  static constexpr int odsIndex_aAddr = 0;
  static constexpr int odsIndex_bdAddr = 1;
  static constexpr int odsIndex_aRows = 2;
  static constexpr int odsIndex_aCols = 3;
  static constexpr int odsIndex_bdRows = 4;
  static constexpr int odsIndex_bdCols = 5;
public:
  ComputePreloadedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ComputePreloadedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ComputePreloadedOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ComputePreloadedOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ComputePreloadedOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ComputePreloadedOpGenericAdaptor(RangeT values, const ComputePreloadedOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ComputePreloadedOp, typename = std::enable_if_t<std::is_same_v<LateInst, ComputePreloadedOp>>>
  ComputePreloadedOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAAddr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getBdAddr() {
    return (*getODSOperands(1).begin());
  }

  ValueT getARows() {
    return (*getODSOperands(2).begin());
  }

  ValueT getACols() {
    return (*getODSOperands(3).begin());
  }

  ValueT getBdRows() {
    return (*getODSOperands(4).begin());
  }

  ValueT getBdCols() {
    return (*getODSOperands(5).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ComputePreloadedOpAdaptor : public ComputePreloadedOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ComputePreloadedOpGenericAdaptor::ComputePreloadedOpGenericAdaptor;
  ComputePreloadedOpAdaptor(ComputePreloadedOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ComputePreloadedOp : public ::mlir::Op<ComputePreloadedOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ComputePreloadedOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ComputePreloadedOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_aAddr = 0;
  static constexpr int odsIndex_bdAddr = 1;
  static constexpr int odsIndex_aRows = 2;
  static constexpr int odsIndex_aCols = 3;
  static constexpr int odsIndex_bdRows = 4;
  static constexpr int odsIndex_bdCols = 5;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.compute_preloaded");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getAAddr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getBdAddr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getARows() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getACols() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(3).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getBdRows() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(4).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getBdCols() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(5).begin());
  }

  ::mlir::OpOperand &getAAddrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBdAddrMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getARowsMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getAColsMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBdRowsMutable() {
    auto range = getODSOperandIndexAndLength(4);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBdColsMutable() {
    auto range = getODSOperandIndexAndLength(5);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols);
  static ComputePreloadedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols);
  static ComputePreloadedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols);
  static ComputePreloadedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols);
  static ComputePreloadedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value aAddr, ::mlir::Value bdAddr, ::mlir::Value aRows, ::mlir::Value aCols, ::mlir::Value bdRows, ::mlir::Value bdCols);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ComputePreloadedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ComputePreloadedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ComputePreloadedOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ComputePreloadedOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::ComputePreloadedOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::ConfigExOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConfigExOpGenericAdaptorBase {
public:
  struct Properties {
    using aStrideTy = ::mlir::IntegerAttr;
    aStrideTy aStride;

    auto getAStride() const {
      auto &propStorage = this->aStride;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setAStride(const ::mlir::IntegerAttr &propValue) {
      this->aStride = propValue;
    }
    using aTransposeTy = ::mlir::BoolAttr;
    aTransposeTy aTranspose;

    auto getATranspose() const {
      auto &propStorage = this->aTranspose;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setATranspose(const ::mlir::BoolAttr &propValue) {
      this->aTranspose = propValue;
    }
    using bTransposeTy = ::mlir::BoolAttr;
    bTransposeTy bTranspose;

    auto getBTranspose() const {
      auto &propStorage = this->bTranspose;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setBTranspose(const ::mlir::BoolAttr &propValue) {
      this->bTranspose = propValue;
    }
    using cStrideTy = ::mlir::IntegerAttr;
    cStrideTy cStride;

    auto getCStride() const {
      auto &propStorage = this->cStride;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setCStride(const ::mlir::IntegerAttr &propValue) {
      this->cStride = propValue;
    }
    using dataflowTy = ::mlir::IntegerAttr;
    dataflowTy dataflow;

    auto getDataflow() const {
      auto &propStorage = this->dataflow;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setDataflow(const ::mlir::IntegerAttr &propValue) {
      this->dataflow = propValue;
    }
    using setOnlyStridesTy = ::mlir::BoolAttr;
    setOnlyStridesTy setOnlyStrides;

    auto getSetOnlyStrides() const {
      auto &propStorage = this->setOnlyStrides;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setSetOnlyStrides(const ::mlir::BoolAttr &propValue) {
      this->setOnlyStrides = propValue;
    }
    using sysAccScaleTy = ::mlir::FloatAttr;
    sysAccScaleTy sysAccScale;

    auto getSysAccScale() const {
      auto &propStorage = this->sysAccScale;
      return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(propStorage);
    }
    void setSysAccScale(const ::mlir::FloatAttr &propValue) {
      this->sysAccScale = propValue;
    }
    using sysActTy = ::mlir::IntegerAttr;
    sysActTy sysAct;

    auto getSysAct() const {
      auto &propStorage = this->sysAct;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setSysAct(const ::mlir::IntegerAttr &propValue) {
      this->sysAct = propValue;
    }
    using sysShiftTy = ::mlir::IntegerAttr;
    sysShiftTy sysShift;

    auto getSysShift() const {
      auto &propStorage = this->sysShift;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setSysShift(const ::mlir::IntegerAttr &propValue) {
      this->sysShift = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.aStride == this->aStride &&
        rhs.aTranspose == this->aTranspose &&
        rhs.bTranspose == this->bTranspose &&
        rhs.cStride == this->cStride &&
        rhs.dataflow == this->dataflow &&
        rhs.setOnlyStrides == this->setOnlyStrides &&
        rhs.sysAccScale == this->sysAccScale &&
        rhs.sysAct == this->sysAct &&
        rhs.sysShift == this->sysShift &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ConfigExOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.config_ex", odsAttrs.getContext());
  }

  ConfigExOpGenericAdaptorBase(ConfigExOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getDataflowAttr();
  uint64_t getDataflow();
  ::mlir::IntegerAttr getSysActAttr();
  uint64_t getSysAct();
  ::mlir::IntegerAttr getSysShiftAttr();
  uint64_t getSysShift();
  ::mlir::FloatAttr getSysAccScaleAttr();
  ::llvm::APFloat getSysAccScale();
  ::mlir::IntegerAttr getCStrideAttr();
  uint64_t getCStride();
  ::mlir::IntegerAttr getAStrideAttr();
  uint64_t getAStride();
  ::mlir::BoolAttr getATransposeAttr();
  bool getATranspose();
  ::mlir::BoolAttr getBTransposeAttr();
  bool getBTranspose();
  ::mlir::BoolAttr getSetOnlyStridesAttr();
  bool getSetOnlyStrides();
};

} // namespace detail
template <typename RangeT>
class ConfigExOpGenericAdaptor : public detail::ConfigExOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConfigExOpGenericAdaptorBase;
public:
  ConfigExOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConfigExOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ConfigExOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConfigExOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConfigExOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConfigExOpGenericAdaptor(RangeT values, const ConfigExOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConfigExOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConfigExOp>>>
  ConfigExOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConfigExOpAdaptor : public ConfigExOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConfigExOpGenericAdaptor::ConfigExOpGenericAdaptor;
  ConfigExOpAdaptor(ConfigExOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConfigExOp : public ::mlir::Op<ConfigExOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConfigExOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConfigExOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("aStride"), ::llvm::StringRef("aTranspose"), ::llvm::StringRef("bTranspose"), ::llvm::StringRef("cStride"), ::llvm::StringRef("dataflow"), ::llvm::StringRef("setOnlyStrides"), ::llvm::StringRef("sysAccScale"), ::llvm::StringRef("sysAct"), ::llvm::StringRef("sysShift")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAStrideAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAStrideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getATransposeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getATransposeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getBTransposeAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getBTransposeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getCStrideAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getCStrideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getDataflowAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getDataflowAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getSetOnlyStridesAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getSetOnlyStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getSysAccScaleAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getSysAccScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getSysActAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getSysActAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getSysShiftAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getSysShiftAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.config_ex");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getDataflowAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().dataflow);
  }

  uint64_t getDataflow();
  ::mlir::IntegerAttr getSysActAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().sysAct);
  }

  uint64_t getSysAct();
  ::mlir::IntegerAttr getSysShiftAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().sysShift);
  }

  uint64_t getSysShift();
  ::mlir::FloatAttr getSysAccScaleAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().sysAccScale);
  }

  ::llvm::APFloat getSysAccScale();
  ::mlir::IntegerAttr getCStrideAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().cStride);
  }

  uint64_t getCStride();
  ::mlir::IntegerAttr getAStrideAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().aStride);
  }

  uint64_t getAStride();
  ::mlir::BoolAttr getATransposeAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().aTranspose);
  }

  bool getATranspose();
  ::mlir::BoolAttr getBTransposeAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().bTranspose);
  }

  bool getBTranspose();
  ::mlir::BoolAttr getSetOnlyStridesAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().setOnlyStrides);
  }

  bool getSetOnlyStrides();
  void setDataflowAttr(::mlir::IntegerAttr attr) {
    getProperties().dataflow = attr;
  }

  void setDataflow(uint64_t attrValue);
  void setSysActAttr(::mlir::IntegerAttr attr) {
    getProperties().sysAct = attr;
  }

  void setSysAct(uint64_t attrValue);
  void setSysShiftAttr(::mlir::IntegerAttr attr) {
    getProperties().sysShift = attr;
  }

  void setSysShift(uint64_t attrValue);
  void setSysAccScaleAttr(::mlir::FloatAttr attr) {
    getProperties().sysAccScale = attr;
  }

  void setSysAccScale(::llvm::APFloat attrValue);
  void setCStrideAttr(::mlir::IntegerAttr attr) {
    getProperties().cStride = attr;
  }

  void setCStride(uint64_t attrValue);
  void setAStrideAttr(::mlir::IntegerAttr attr) {
    getProperties().aStride = attr;
  }

  void setAStride(uint64_t attrValue);
  void setATransposeAttr(::mlir::BoolAttr attr) {
    getProperties().aTranspose = attr;
  }

  void setATranspose(bool attrValue);
  void setBTransposeAttr(::mlir::BoolAttr attr) {
    getProperties().bTranspose = attr;
  }

  void setBTranspose(bool attrValue);
  void setSetOnlyStridesAttr(::mlir::BoolAttr attr) {
    getProperties().setOnlyStrides = attr;
  }

  void setSetOnlyStrides(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr dataflow, ::mlir::IntegerAttr sysAct, ::mlir::IntegerAttr sysShift, ::mlir::FloatAttr sysAccScale, ::mlir::IntegerAttr cStride, ::mlir::IntegerAttr aStride = nullptr, ::mlir::BoolAttr aTranspose = nullptr, ::mlir::BoolAttr bTranspose = nullptr, ::mlir::BoolAttr setOnlyStrides = nullptr);
  static ConfigExOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::IntegerAttr dataflow, ::mlir::IntegerAttr sysAct, ::mlir::IntegerAttr sysShift, ::mlir::FloatAttr sysAccScale, ::mlir::IntegerAttr cStride, ::mlir::IntegerAttr aStride = nullptr, ::mlir::BoolAttr aTranspose = nullptr, ::mlir::BoolAttr bTranspose = nullptr, ::mlir::BoolAttr setOnlyStrides = nullptr);
  static ConfigExOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::IntegerAttr dataflow, ::mlir::IntegerAttr sysAct, ::mlir::IntegerAttr sysShift, ::mlir::FloatAttr sysAccScale, ::mlir::IntegerAttr cStride, ::mlir::IntegerAttr aStride = nullptr, ::mlir::BoolAttr aTranspose = nullptr, ::mlir::BoolAttr bTranspose = nullptr, ::mlir::BoolAttr setOnlyStrides = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr dataflow, ::mlir::IntegerAttr sysAct, ::mlir::IntegerAttr sysShift, ::mlir::FloatAttr sysAccScale, ::mlir::IntegerAttr cStride, ::mlir::IntegerAttr aStride = nullptr, ::mlir::BoolAttr aTranspose = nullptr, ::mlir::BoolAttr bTranspose = nullptr, ::mlir::BoolAttr setOnlyStrides = nullptr);
  static ConfigExOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr dataflow, ::mlir::IntegerAttr sysAct, ::mlir::IntegerAttr sysShift, ::mlir::FloatAttr sysAccScale, ::mlir::IntegerAttr cStride, ::mlir::IntegerAttr aStride = nullptr, ::mlir::BoolAttr aTranspose = nullptr, ::mlir::BoolAttr bTranspose = nullptr, ::mlir::BoolAttr setOnlyStrides = nullptr);
  static ConfigExOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr dataflow, ::mlir::IntegerAttr sysAct, ::mlir::IntegerAttr sysShift, ::mlir::FloatAttr sysAccScale, ::mlir::IntegerAttr cStride, ::mlir::IntegerAttr aStride = nullptr, ::mlir::BoolAttr aTranspose = nullptr, ::mlir::BoolAttr bTranspose = nullptr, ::mlir::BoolAttr setOnlyStrides = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint64_t dataflow, uint64_t sysAct, uint64_t sysShift, ::llvm::APFloat sysAccScale, uint64_t cStride = 1, uint64_t aStride = 1, bool aTranspose = false, bool bTranspose = false, bool setOnlyStrides = false);
  static ConfigExOp create(::mlir::OpBuilder &builder, ::mlir::Location location, uint64_t dataflow, uint64_t sysAct, uint64_t sysShift, ::llvm::APFloat sysAccScale, uint64_t cStride = 1, uint64_t aStride = 1, bool aTranspose = false, bool bTranspose = false, bool setOnlyStrides = false);
  static ConfigExOp create(::mlir::ImplicitLocOpBuilder &builder, uint64_t dataflow, uint64_t sysAct, uint64_t sysShift, ::llvm::APFloat sysAccScale, uint64_t cStride = 1, uint64_t aStride = 1, bool aTranspose = false, bool bTranspose = false, bool setOnlyStrides = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint64_t dataflow, uint64_t sysAct, uint64_t sysShift, ::llvm::APFloat sysAccScale, uint64_t cStride = 1, uint64_t aStride = 1, bool aTranspose = false, bool bTranspose = false, bool setOnlyStrides = false);
  static ConfigExOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint64_t dataflow, uint64_t sysAct, uint64_t sysShift, ::llvm::APFloat sysAccScale, uint64_t cStride = 1, uint64_t aStride = 1, bool aTranspose = false, bool bTranspose = false, bool setOnlyStrides = false);
  static ConfigExOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint64_t dataflow, uint64_t sysAct, uint64_t sysShift, ::llvm::APFloat sysAccScale, uint64_t cStride = 1, uint64_t aStride = 1, bool aTranspose = false, bool bTranspose = false, bool setOnlyStrides = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigExOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigExOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigExOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigExOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void populateDefaultProperties(::mlir::OperationName opName, Properties &properties);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 9 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::ConfigExOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::ConfigLdOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConfigLdOpGenericAdaptorBase {
public:
  struct Properties {
    using block_mvin_strideTy = ::mlir::IntegerAttr;
    block_mvin_strideTy block_mvin_stride;

    auto getBlockMvinStride() const {
      auto &propStorage = this->block_mvin_stride;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setBlockMvinStride(const ::mlir::IntegerAttr &propValue) {
      this->block_mvin_stride = propValue;
    }
    using idTy = ::mlir::IntegerAttr;
    idTy id;

    auto getId() const {
      auto &propStorage = this->id;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setId(const ::mlir::IntegerAttr &propValue) {
      this->id = propValue;
    }
    using pixel_repeatsTy = ::mlir::IntegerAttr;
    pixel_repeatsTy pixel_repeats;

    auto getPixelRepeats() const {
      auto &propStorage = this->pixel_repeats;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setPixelRepeats(const ::mlir::IntegerAttr &propValue) {
      this->pixel_repeats = propValue;
    }
    using scaleTy = ::mlir::FloatAttr;
    scaleTy scale;

    auto getScale() const {
      auto &propStorage = this->scale;
      return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(propStorage);
    }
    void setScale(const ::mlir::FloatAttr &propValue) {
      this->scale = propValue;
    }
    using shrunkTy = ::mlir::BoolAttr;
    shrunkTy shrunk;

    auto getShrunk() const {
      auto &propStorage = this->shrunk;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setShrunk(const ::mlir::BoolAttr &propValue) {
      this->shrunk = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.block_mvin_stride == this->block_mvin_stride &&
        rhs.id == this->id &&
        rhs.pixel_repeats == this->pixel_repeats &&
        rhs.scale == this->scale &&
        rhs.shrunk == this->shrunk &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ConfigLdOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.config_ld", odsAttrs.getContext());
  }

  ConfigLdOpGenericAdaptorBase(ConfigLdOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::FloatAttr getScaleAttr();
  ::llvm::APFloat getScale();
  ::mlir::BoolAttr getShrunkAttr();
  bool getShrunk();
  ::mlir::IntegerAttr getIdAttr();
  uint64_t getId();
  ::mlir::IntegerAttr getBlockMvinStrideAttr();
  uint64_t getBlockMvinStride();
  ::mlir::IntegerAttr getPixelRepeatsAttr();
  uint64_t getPixelRepeats();
};

} // namespace detail
template <typename RangeT>
class ConfigLdOpGenericAdaptor : public detail::ConfigLdOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConfigLdOpGenericAdaptorBase;
  static constexpr int odsIndex_stride = 0;
public:
  ConfigLdOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConfigLdOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ConfigLdOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConfigLdOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConfigLdOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConfigLdOpGenericAdaptor(RangeT values, const ConfigLdOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConfigLdOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConfigLdOp>>>
  ConfigLdOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getStride() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConfigLdOpAdaptor : public ConfigLdOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConfigLdOpGenericAdaptor::ConfigLdOpGenericAdaptor;
  ConfigLdOpAdaptor(ConfigLdOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConfigLdOp : public ::mlir::Op<ConfigLdOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConfigLdOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConfigLdOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_stride = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("block_mvin_stride"), ::llvm::StringRef("id"), ::llvm::StringRef("pixel_repeats"), ::llvm::StringRef("scale"), ::llvm::StringRef("shrunk")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBlockMvinStrideAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBlockMvinStrideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIdAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIdAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getPixelRepeatsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getPixelRepeatsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getScaleAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getShrunkAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getShrunkAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.config_ld");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getStride() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getStrideMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::FloatAttr getScaleAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().scale);
  }

  ::llvm::APFloat getScale();
  ::mlir::BoolAttr getShrunkAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().shrunk);
  }

  bool getShrunk();
  ::mlir::IntegerAttr getIdAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().id);
  }

  uint64_t getId();
  ::mlir::IntegerAttr getBlockMvinStrideAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().block_mvin_stride);
  }

  uint64_t getBlockMvinStride();
  ::mlir::IntegerAttr getPixelRepeatsAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().pixel_repeats);
  }

  uint64_t getPixelRepeats();
  void setScaleAttr(::mlir::FloatAttr attr) {
    getProperties().scale = attr;
  }

  void setScale(::llvm::APFloat attrValue);
  void setShrunkAttr(::mlir::BoolAttr attr) {
    getProperties().shrunk = attr;
  }

  void setShrunk(bool attrValue);
  void setIdAttr(::mlir::IntegerAttr attr) {
    getProperties().id = attr;
  }

  void setId(uint64_t attrValue);
  void setBlockMvinStrideAttr(::mlir::IntegerAttr attr) {
    getProperties().block_mvin_stride = attr;
  }

  void setBlockMvinStride(uint64_t attrValue);
  void setPixelRepeatsAttr(::mlir::IntegerAttr attr) {
    getProperties().pixel_repeats = attr;
  }

  void setPixelRepeats(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value stride, ::mlir::FloatAttr scale, ::mlir::BoolAttr shrunk, ::mlir::IntegerAttr id = nullptr, ::mlir::IntegerAttr block_mvin_stride = nullptr, ::mlir::IntegerAttr pixel_repeats = nullptr);
  static ConfigLdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value stride, ::mlir::FloatAttr scale, ::mlir::BoolAttr shrunk, ::mlir::IntegerAttr id = nullptr, ::mlir::IntegerAttr block_mvin_stride = nullptr, ::mlir::IntegerAttr pixel_repeats = nullptr);
  static ConfigLdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value stride, ::mlir::FloatAttr scale, ::mlir::BoolAttr shrunk, ::mlir::IntegerAttr id = nullptr, ::mlir::IntegerAttr block_mvin_stride = nullptr, ::mlir::IntegerAttr pixel_repeats = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value stride, ::mlir::FloatAttr scale, ::mlir::BoolAttr shrunk, ::mlir::IntegerAttr id = nullptr, ::mlir::IntegerAttr block_mvin_stride = nullptr, ::mlir::IntegerAttr pixel_repeats = nullptr);
  static ConfigLdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value stride, ::mlir::FloatAttr scale, ::mlir::BoolAttr shrunk, ::mlir::IntegerAttr id = nullptr, ::mlir::IntegerAttr block_mvin_stride = nullptr, ::mlir::IntegerAttr pixel_repeats = nullptr);
  static ConfigLdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value stride, ::mlir::FloatAttr scale, ::mlir::BoolAttr shrunk, ::mlir::IntegerAttr id = nullptr, ::mlir::IntegerAttr block_mvin_stride = nullptr, ::mlir::IntegerAttr pixel_repeats = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value stride, ::llvm::APFloat scale, bool shrunk = false, uint64_t id = 0, uint64_t block_mvin_stride = -1, uint64_t pixel_repeats = 1);
  static ConfigLdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value stride, ::llvm::APFloat scale, bool shrunk = false, uint64_t id = 0, uint64_t block_mvin_stride = -1, uint64_t pixel_repeats = 1);
  static ConfigLdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value stride, ::llvm::APFloat scale, bool shrunk = false, uint64_t id = 0, uint64_t block_mvin_stride = -1, uint64_t pixel_repeats = 1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value stride, ::llvm::APFloat scale, bool shrunk = false, uint64_t id = 0, uint64_t block_mvin_stride = -1, uint64_t pixel_repeats = 1);
  static ConfigLdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value stride, ::llvm::APFloat scale, bool shrunk = false, uint64_t id = 0, uint64_t block_mvin_stride = -1, uint64_t pixel_repeats = 1);
  static ConfigLdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value stride, ::llvm::APFloat scale, bool shrunk = false, uint64_t id = 0, uint64_t block_mvin_stride = -1, uint64_t pixel_repeats = 1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigLdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigLdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigLdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigLdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void populateDefaultProperties(::mlir::OperationName opName, Properties &properties);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::ConfigLdOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::ConfigNormOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConfigNormOpGenericAdaptorBase {
public:
  struct Properties {
    using StatsIdTy = ::mlir::IntegerAttr;
    StatsIdTy StatsId;

    auto getStatsId() const {
      auto &propStorage = this->StatsId;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setStatsId(const ::mlir::IntegerAttr &propValue) {
      this->StatsId = propValue;
    }
    using actMsbTy = ::mlir::IntegerAttr;
    actMsbTy actMsb;

    auto getActMsb() const {
      auto &propStorage = this->actMsb;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setActMsb(const ::mlir::IntegerAttr &propValue) {
      this->actMsb = propValue;
    }
    using igeluQbTy = ::mlir::IntegerAttr;
    igeluQbTy igeluQb;

    auto getIgeluQb() const {
      auto &propStorage = this->igeluQb;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setIgeluQb(const ::mlir::IntegerAttr &propValue) {
      this->igeluQb = propValue;
    }
    using igeluQcTy = ::mlir::IntegerAttr;
    igeluQcTy igeluQc;

    auto getIgeluQc() const {
      auto &propStorage = this->igeluQc;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setIgeluQc(const ::mlir::IntegerAttr &propValue) {
      this->igeluQc = propValue;
    }
    using qConstTy = ::mlir::IntegerAttr;
    qConstTy qConst;

    auto getQConst() const {
      auto &propStorage = this->qConst;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setQConst(const ::mlir::IntegerAttr &propValue) {
      this->qConst = propValue;
    }
    using qConstTypeTy = ::mlir::IntegerAttr;
    qConstTypeTy qConstType;

    auto getQConstType() const {
      auto &propStorage = this->qConstType;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setQConstType(const ::mlir::IntegerAttr &propValue) {
      this->qConstType = propValue;
    }
    using setStatsIdOnlyTy = ::mlir::IntegerAttr;
    setStatsIdOnlyTy setStatsIdOnly;

    auto getSetStatsIdOnly() const {
      auto &propStorage = this->setStatsIdOnly;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setSetStatsIdOnly(const ::mlir::IntegerAttr &propValue) {
      this->setStatsIdOnly = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.StatsId == this->StatsId &&
        rhs.actMsb == this->actMsb &&
        rhs.igeluQb == this->igeluQb &&
        rhs.igeluQc == this->igeluQc &&
        rhs.qConst == this->qConst &&
        rhs.qConstType == this->qConstType &&
        rhs.setStatsIdOnly == this->setStatsIdOnly &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ConfigNormOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.config_norm", odsAttrs.getContext());
  }

  ConfigNormOpGenericAdaptorBase(ConfigNormOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getQConstAttr();
  uint64_t getQConst();
  ::mlir::IntegerAttr getQConstTypeAttr();
  uint64_t getQConstType();
  ::mlir::IntegerAttr getSetStatsIdOnlyAttr();
  uint64_t getSetStatsIdOnly();
  ::mlir::IntegerAttr getActMsbAttr();
  uint64_t getActMsb();
  ::mlir::IntegerAttr getStatsIdAttr();
  uint64_t getStatsId();
  ::mlir::IntegerAttr getIgeluQbAttr();
  uint64_t getIgeluQb();
  ::mlir::IntegerAttr getIgeluQcAttr();
  uint64_t getIgeluQc();
};

} // namespace detail
template <typename RangeT>
class ConfigNormOpGenericAdaptor : public detail::ConfigNormOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConfigNormOpGenericAdaptorBase;
public:
  ConfigNormOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConfigNormOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ConfigNormOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConfigNormOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConfigNormOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConfigNormOpGenericAdaptor(RangeT values, const ConfigNormOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConfigNormOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConfigNormOp>>>
  ConfigNormOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConfigNormOpAdaptor : public ConfigNormOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConfigNormOpGenericAdaptor::ConfigNormOpGenericAdaptor;
  ConfigNormOpAdaptor(ConfigNormOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConfigNormOp : public ::mlir::Op<ConfigNormOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConfigNormOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConfigNormOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("StatsId"), ::llvm::StringRef("actMsb"), ::llvm::StringRef("igeluQb"), ::llvm::StringRef("igeluQc"), ::llvm::StringRef("qConst"), ::llvm::StringRef("qConstType"), ::llvm::StringRef("setStatsIdOnly")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getStatsIdAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getStatsIdAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getActMsbAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getActMsbAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getIgeluQbAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getIgeluQbAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getIgeluQcAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getIgeluQcAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getQConstAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getQConstAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getQConstTypeAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getQConstTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getSetStatsIdOnlyAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getSetStatsIdOnlyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.config_norm");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getQConstAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().qConst);
  }

  uint64_t getQConst();
  ::mlir::IntegerAttr getQConstTypeAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().qConstType);
  }

  uint64_t getQConstType();
  ::mlir::IntegerAttr getSetStatsIdOnlyAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().setStatsIdOnly);
  }

  uint64_t getSetStatsIdOnly();
  ::mlir::IntegerAttr getActMsbAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().actMsb);
  }

  uint64_t getActMsb();
  ::mlir::IntegerAttr getStatsIdAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().StatsId);
  }

  uint64_t getStatsId();
  ::mlir::IntegerAttr getIgeluQbAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().igeluQb);
  }

  uint64_t getIgeluQb();
  ::mlir::IntegerAttr getIgeluQcAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().igeluQc);
  }

  uint64_t getIgeluQc();
  void setQConstAttr(::mlir::IntegerAttr attr) {
    getProperties().qConst = attr;
  }

  void setQConst(uint64_t attrValue);
  void setQConstTypeAttr(::mlir::IntegerAttr attr) {
    getProperties().qConstType = attr;
  }

  void setQConstType(uint64_t attrValue);
  void setSetStatsIdOnlyAttr(::mlir::IntegerAttr attr) {
    getProperties().setStatsIdOnly = attr;
  }

  void setSetStatsIdOnly(uint64_t attrValue);
  void setActMsbAttr(::mlir::IntegerAttr attr) {
    getProperties().actMsb = attr;
  }

  void setActMsb(uint64_t attrValue);
  void setStatsIdAttr(::mlir::IntegerAttr attr) {
    getProperties().StatsId = attr;
  }

  void setStatsId(uint64_t attrValue);
  void setIgeluQbAttr(::mlir::IntegerAttr attr) {
    getProperties().igeluQb = attr;
  }

  void setIgeluQb(uint64_t attrValue);
  void setIgeluQcAttr(::mlir::IntegerAttr attr) {
    getProperties().igeluQc = attr;
  }

  void setIgeluQc(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr qConst, ::mlir::IntegerAttr qConstType = nullptr, ::mlir::IntegerAttr setStatsIdOnly = nullptr, ::mlir::IntegerAttr actMsb = nullptr, ::mlir::IntegerAttr StatsId = nullptr, ::mlir::IntegerAttr igeluQb = nullptr, ::mlir::IntegerAttr igeluQc = nullptr);
  static ConfigNormOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::IntegerAttr qConst, ::mlir::IntegerAttr qConstType = nullptr, ::mlir::IntegerAttr setStatsIdOnly = nullptr, ::mlir::IntegerAttr actMsb = nullptr, ::mlir::IntegerAttr StatsId = nullptr, ::mlir::IntegerAttr igeluQb = nullptr, ::mlir::IntegerAttr igeluQc = nullptr);
  static ConfigNormOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::IntegerAttr qConst, ::mlir::IntegerAttr qConstType = nullptr, ::mlir::IntegerAttr setStatsIdOnly = nullptr, ::mlir::IntegerAttr actMsb = nullptr, ::mlir::IntegerAttr StatsId = nullptr, ::mlir::IntegerAttr igeluQb = nullptr, ::mlir::IntegerAttr igeluQc = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr qConst, ::mlir::IntegerAttr qConstType = nullptr, ::mlir::IntegerAttr setStatsIdOnly = nullptr, ::mlir::IntegerAttr actMsb = nullptr, ::mlir::IntegerAttr StatsId = nullptr, ::mlir::IntegerAttr igeluQb = nullptr, ::mlir::IntegerAttr igeluQc = nullptr);
  static ConfigNormOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr qConst, ::mlir::IntegerAttr qConstType = nullptr, ::mlir::IntegerAttr setStatsIdOnly = nullptr, ::mlir::IntegerAttr actMsb = nullptr, ::mlir::IntegerAttr StatsId = nullptr, ::mlir::IntegerAttr igeluQb = nullptr, ::mlir::IntegerAttr igeluQc = nullptr);
  static ConfigNormOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr qConst, ::mlir::IntegerAttr qConstType = nullptr, ::mlir::IntegerAttr setStatsIdOnly = nullptr, ::mlir::IntegerAttr actMsb = nullptr, ::mlir::IntegerAttr StatsId = nullptr, ::mlir::IntegerAttr igeluQb = nullptr, ::mlir::IntegerAttr igeluQc = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint64_t qConst = 0, uint64_t qConstType = 0, uint64_t setStatsIdOnly = 0, uint64_t actMsb = 0, uint64_t StatsId = 0, uint64_t igeluQb = 0, uint64_t igeluQc = 0);
  static ConfigNormOp create(::mlir::OpBuilder &builder, ::mlir::Location location, uint64_t qConst = 0, uint64_t qConstType = 0, uint64_t setStatsIdOnly = 0, uint64_t actMsb = 0, uint64_t StatsId = 0, uint64_t igeluQb = 0, uint64_t igeluQc = 0);
  static ConfigNormOp create(::mlir::ImplicitLocOpBuilder &builder, uint64_t qConst = 0, uint64_t qConstType = 0, uint64_t setStatsIdOnly = 0, uint64_t actMsb = 0, uint64_t StatsId = 0, uint64_t igeluQb = 0, uint64_t igeluQc = 0);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint64_t qConst = 0, uint64_t qConstType = 0, uint64_t setStatsIdOnly = 0, uint64_t actMsb = 0, uint64_t StatsId = 0, uint64_t igeluQb = 0, uint64_t igeluQc = 0);
  static ConfigNormOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint64_t qConst = 0, uint64_t qConstType = 0, uint64_t setStatsIdOnly = 0, uint64_t actMsb = 0, uint64_t StatsId = 0, uint64_t igeluQb = 0, uint64_t igeluQc = 0);
  static ConfigNormOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint64_t qConst = 0, uint64_t qConstType = 0, uint64_t setStatsIdOnly = 0, uint64_t actMsb = 0, uint64_t StatsId = 0, uint64_t igeluQb = 0, uint64_t igeluQc = 0);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigNormOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigNormOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigNormOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigNormOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void populateDefaultProperties(::mlir::OperationName opName, Properties &properties);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 7 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::ConfigNormOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::ConfigStOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConfigStOpGenericAdaptorBase {
public:
  struct Properties {
    using activationTy = ::mlir::IntegerAttr;
    activationTy activation;

    auto getActivation() const {
      auto &propStorage = this->activation;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setActivation(const ::mlir::IntegerAttr &propValue) {
      this->activation = propValue;
    }
    using scaleTy = ::mlir::FloatAttr;
    scaleTy scale;

    auto getScale() const {
      auto &propStorage = this->scale;
      return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(propStorage);
    }
    void setScale(const ::mlir::FloatAttr &propValue) {
      this->scale = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.activation == this->activation &&
        rhs.scale == this->scale &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ConfigStOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.config_st", odsAttrs.getContext());
  }

  ConfigStOpGenericAdaptorBase(ConfigStOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getActivationAttr();
  uint64_t getActivation();
  ::mlir::FloatAttr getScaleAttr();
  ::llvm::APFloat getScale();
};

} // namespace detail
template <typename RangeT>
class ConfigStOpGenericAdaptor : public detail::ConfigStOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConfigStOpGenericAdaptorBase;
  static constexpr int odsIndex_stride = 0;
public:
  ConfigStOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConfigStOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ConfigStOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConfigStOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConfigStOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConfigStOpGenericAdaptor(RangeT values, const ConfigStOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConfigStOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConfigStOp>>>
  ConfigStOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getStride() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConfigStOpAdaptor : public ConfigStOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConfigStOpGenericAdaptor::ConfigStOpGenericAdaptor;
  ConfigStOpAdaptor(ConfigStOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConfigStOp : public ::mlir::Op<ConfigStOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConfigStOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConfigStOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_stride = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("activation"), ::llvm::StringRef("scale")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getActivationAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getActivationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getScaleAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.config_st");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getStride() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getStrideMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getActivationAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().activation);
  }

  uint64_t getActivation();
  ::mlir::FloatAttr getScaleAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().scale);
  }

  ::llvm::APFloat getScale();
  void setActivationAttr(::mlir::IntegerAttr attr) {
    getProperties().activation = attr;
  }

  void setActivation(uint64_t attrValue);
  void setScaleAttr(::mlir::FloatAttr attr) {
    getProperties().scale = attr;
  }

  void setScale(::llvm::APFloat attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value stride, ::mlir::IntegerAttr activation, ::mlir::FloatAttr scale);
  static ConfigStOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value stride, ::mlir::IntegerAttr activation, ::mlir::FloatAttr scale);
  static ConfigStOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value stride, ::mlir::IntegerAttr activation, ::mlir::FloatAttr scale);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value stride, ::mlir::IntegerAttr activation, ::mlir::FloatAttr scale);
  static ConfigStOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value stride, ::mlir::IntegerAttr activation, ::mlir::FloatAttr scale);
  static ConfigStOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value stride, ::mlir::IntegerAttr activation, ::mlir::FloatAttr scale);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value stride, uint64_t activation, ::llvm::APFloat scale);
  static ConfigStOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value stride, uint64_t activation, ::llvm::APFloat scale);
  static ConfigStOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value stride, uint64_t activation, ::llvm::APFloat scale);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value stride, uint64_t activation, ::llvm::APFloat scale);
  static ConfigStOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value stride, uint64_t activation, ::llvm::APFloat scale);
  static ConfigStOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value stride, uint64_t activation, ::llvm::APFloat scale);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigStOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigStOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigStOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigStOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void populateDefaultProperties(::mlir::OperationName opName, Properties &properties);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::ConfigStOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::FlushOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class FlushOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  FlushOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.flush", odsAttrs.getContext());
  }

  FlushOpGenericAdaptorBase(FlushOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class FlushOpGenericAdaptor : public detail::FlushOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::FlushOpGenericAdaptorBase;
  static constexpr int odsIndex_skip = 0;
public:
  FlushOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  FlushOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : FlushOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  FlushOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : FlushOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  FlushOpGenericAdaptor(RangeT values, const FlushOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = FlushOp, typename = std::enable_if_t<std::is_same_v<LateInst, FlushOp>>>
  FlushOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSkip() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class FlushOpAdaptor : public FlushOpGenericAdaptor<::mlir::ValueRange> {
public:
  using FlushOpGenericAdaptor::FlushOpGenericAdaptor;
  FlushOpAdaptor(FlushOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class FlushOp : public ::mlir::Op<FlushOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FlushOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = FlushOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_skip = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.flush");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getSkip() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getSkipMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value skip);
  static FlushOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value skip);
  static FlushOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value skip);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value skip);
  static FlushOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value skip);
  static FlushOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value skip);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static FlushOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static FlushOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static FlushOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static FlushOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::FlushOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::ComputeAccumulated_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ComputeAccumulated_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ComputeAccumulated_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.compute_accumulated", odsAttrs.getContext());
  }

  ComputeAccumulated_IntrOpGenericAdaptorBase(ComputeAccumulated_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ComputeAccumulated_IntrOpGenericAdaptor : public detail::ComputeAccumulated_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ComputeAccumulated_IntrOpGenericAdaptorBase;
public:
  ComputeAccumulated_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ComputeAccumulated_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ComputeAccumulated_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ComputeAccumulated_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ComputeAccumulated_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ComputeAccumulated_IntrOpGenericAdaptor(RangeT values, const ComputeAccumulated_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ComputeAccumulated_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, ComputeAccumulated_IntrOp>>>
  ComputeAccumulated_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ComputeAccumulated_IntrOpAdaptor : public ComputeAccumulated_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ComputeAccumulated_IntrOpGenericAdaptor::ComputeAccumulated_IntrOpGenericAdaptor;
  ComputeAccumulated_IntrOpAdaptor(ComputeAccumulated_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ComputeAccumulated_IntrOp : public ::mlir::Op<ComputeAccumulated_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ComputeAccumulated_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ComputeAccumulated_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.compute_accumulated");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ComputeAccumulated_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ComputeAccumulated_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ComputeAccumulated_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ComputeAccumulated_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ComputeAccumulated_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ComputeAccumulated_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ComputeAccumulated_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ComputeAccumulated_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::ComputeAccumulated_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::ComputePreloaded_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ComputePreloaded_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ComputePreloaded_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.compute_preloaded", odsAttrs.getContext());
  }

  ComputePreloaded_IntrOpGenericAdaptorBase(ComputePreloaded_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ComputePreloaded_IntrOpGenericAdaptor : public detail::ComputePreloaded_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ComputePreloaded_IntrOpGenericAdaptorBase;
public:
  ComputePreloaded_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ComputePreloaded_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ComputePreloaded_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ComputePreloaded_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ComputePreloaded_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ComputePreloaded_IntrOpGenericAdaptor(RangeT values, const ComputePreloaded_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ComputePreloaded_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, ComputePreloaded_IntrOp>>>
  ComputePreloaded_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ComputePreloaded_IntrOpAdaptor : public ComputePreloaded_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ComputePreloaded_IntrOpGenericAdaptor::ComputePreloaded_IntrOpGenericAdaptor;
  ComputePreloaded_IntrOpAdaptor(ComputePreloaded_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ComputePreloaded_IntrOp : public ::mlir::Op<ComputePreloaded_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ComputePreloaded_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ComputePreloaded_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.compute_preloaded");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ComputePreloaded_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ComputePreloaded_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ComputePreloaded_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ComputePreloaded_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ComputePreloaded_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ComputePreloaded_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ComputePreloaded_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ComputePreloaded_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::ComputePreloaded_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::ConfigEX_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConfigEX_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ConfigEX_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.config_ex", odsAttrs.getContext());
  }

  ConfigEX_IntrOpGenericAdaptorBase(ConfigEX_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ConfigEX_IntrOpGenericAdaptor : public detail::ConfigEX_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConfigEX_IntrOpGenericAdaptorBase;
public:
  ConfigEX_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConfigEX_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ConfigEX_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConfigEX_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConfigEX_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConfigEX_IntrOpGenericAdaptor(RangeT values, const ConfigEX_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConfigEX_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConfigEX_IntrOp>>>
  ConfigEX_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConfigEX_IntrOpAdaptor : public ConfigEX_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConfigEX_IntrOpGenericAdaptor::ConfigEX_IntrOpGenericAdaptor;
  ConfigEX_IntrOpAdaptor(ConfigEX_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConfigEX_IntrOp : public ::mlir::Op<ConfigEX_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConfigEX_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConfigEX_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.config_ex");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConfigEX_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConfigEX_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConfigEX_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConfigEX_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigEX_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigEX_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigEX_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigEX_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::ConfigEX_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::ConfigNorm_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConfigNorm_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ConfigNorm_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.config_norm", odsAttrs.getContext());
  }

  ConfigNorm_IntrOpGenericAdaptorBase(ConfigNorm_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ConfigNorm_IntrOpGenericAdaptor : public detail::ConfigNorm_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConfigNorm_IntrOpGenericAdaptorBase;
public:
  ConfigNorm_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConfigNorm_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ConfigNorm_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConfigNorm_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConfigNorm_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConfigNorm_IntrOpGenericAdaptor(RangeT values, const ConfigNorm_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConfigNorm_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConfigNorm_IntrOp>>>
  ConfigNorm_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConfigNorm_IntrOpAdaptor : public ConfigNorm_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConfigNorm_IntrOpGenericAdaptor::ConfigNorm_IntrOpGenericAdaptor;
  ConfigNorm_IntrOpAdaptor(ConfigNorm_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConfigNorm_IntrOp : public ::mlir::Op<ConfigNorm_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConfigNorm_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConfigNorm_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.config_norm");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConfigNorm_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConfigNorm_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConfigNorm_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConfigNorm_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigNorm_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigNorm_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigNorm_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigNorm_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::ConfigNorm_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::ConfigSt_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConfigSt_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ConfigSt_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.config_st", odsAttrs.getContext());
  }

  ConfigSt_IntrOpGenericAdaptorBase(ConfigSt_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ConfigSt_IntrOpGenericAdaptor : public detail::ConfigSt_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConfigSt_IntrOpGenericAdaptorBase;
public:
  ConfigSt_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConfigSt_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ConfigSt_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConfigSt_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConfigSt_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConfigSt_IntrOpGenericAdaptor(RangeT values, const ConfigSt_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConfigSt_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConfigSt_IntrOp>>>
  ConfigSt_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConfigSt_IntrOpAdaptor : public ConfigSt_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConfigSt_IntrOpGenericAdaptor::ConfigSt_IntrOpGenericAdaptor;
  ConfigSt_IntrOpAdaptor(ConfigSt_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConfigSt_IntrOp : public ::mlir::Op<ConfigSt_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConfigSt_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConfigSt_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.config_st");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConfigSt_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConfigSt_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConfigSt_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConfigSt_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigSt_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigSt_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigSt_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigSt_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::ConfigSt_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::ConifgLd_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConifgLd_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ConifgLd_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.config_ld", odsAttrs.getContext());
  }

  ConifgLd_IntrOpGenericAdaptorBase(ConifgLd_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ConifgLd_IntrOpGenericAdaptor : public detail::ConifgLd_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConifgLd_IntrOpGenericAdaptorBase;
public:
  ConifgLd_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConifgLd_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ConifgLd_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConifgLd_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConifgLd_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConifgLd_IntrOpGenericAdaptor(RangeT values, const ConifgLd_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConifgLd_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConifgLd_IntrOp>>>
  ConifgLd_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConifgLd_IntrOpAdaptor : public ConifgLd_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConifgLd_IntrOpGenericAdaptor::ConifgLd_IntrOpGenericAdaptor;
  ConifgLd_IntrOpAdaptor(ConifgLd_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConifgLd_IntrOp : public ::mlir::Op<ConifgLd_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConifgLd_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConifgLd_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.config_ld");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConifgLd_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConifgLd_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConifgLd_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static ConifgLd_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConifgLd_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConifgLd_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConifgLd_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConifgLd_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::ConifgLd_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::Flush_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class Flush_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  Flush_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.flush", odsAttrs.getContext());
  }

  Flush_IntrOpGenericAdaptorBase(Flush_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class Flush_IntrOpGenericAdaptor : public detail::Flush_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::Flush_IntrOpGenericAdaptorBase;
public:
  Flush_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  Flush_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : Flush_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  Flush_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : Flush_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  Flush_IntrOpGenericAdaptor(RangeT values, const Flush_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = Flush_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, Flush_IntrOp>>>
  Flush_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class Flush_IntrOpAdaptor : public Flush_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using Flush_IntrOpGenericAdaptor::Flush_IntrOpGenericAdaptor;
  Flush_IntrOpAdaptor(Flush_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class Flush_IntrOp : public ::mlir::Op<Flush_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Flush_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = Flush_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.flush");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Flush_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Flush_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Flush_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Flush_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Flush_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Flush_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Flush_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Flush_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::Flush_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopConvWsConfig1_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopConvWsConfig1_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopConvWsConfig1_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_conv_ws_config1", odsAttrs.getContext());
  }

  LoopConvWsConfig1_IntrOpGenericAdaptorBase(LoopConvWsConfig1_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopConvWsConfig1_IntrOpGenericAdaptor : public detail::LoopConvWsConfig1_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopConvWsConfig1_IntrOpGenericAdaptorBase;
public:
  LoopConvWsConfig1_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopConvWsConfig1_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopConvWsConfig1_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopConvWsConfig1_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopConvWsConfig1_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopConvWsConfig1_IntrOpGenericAdaptor(RangeT values, const LoopConvWsConfig1_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopConvWsConfig1_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopConvWsConfig1_IntrOp>>>
  LoopConvWsConfig1_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopConvWsConfig1_IntrOpAdaptor : public LoopConvWsConfig1_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopConvWsConfig1_IntrOpGenericAdaptor::LoopConvWsConfig1_IntrOpGenericAdaptor;
  LoopConvWsConfig1_IntrOpAdaptor(LoopConvWsConfig1_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopConvWsConfig1_IntrOp : public ::mlir::Op<LoopConvWsConfig1_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopConvWsConfig1_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopConvWsConfig1_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_conv_ws_config1");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig1_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig1_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig1_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig1_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWsConfig1_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWsConfig1_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWsConfig1_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWsConfig1_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopConvWsConfig1_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopConvWsConfig2_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopConvWsConfig2_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopConvWsConfig2_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_conv_ws_config2", odsAttrs.getContext());
  }

  LoopConvWsConfig2_IntrOpGenericAdaptorBase(LoopConvWsConfig2_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopConvWsConfig2_IntrOpGenericAdaptor : public detail::LoopConvWsConfig2_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopConvWsConfig2_IntrOpGenericAdaptorBase;
public:
  LoopConvWsConfig2_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopConvWsConfig2_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopConvWsConfig2_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopConvWsConfig2_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopConvWsConfig2_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopConvWsConfig2_IntrOpGenericAdaptor(RangeT values, const LoopConvWsConfig2_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopConvWsConfig2_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopConvWsConfig2_IntrOp>>>
  LoopConvWsConfig2_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopConvWsConfig2_IntrOpAdaptor : public LoopConvWsConfig2_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopConvWsConfig2_IntrOpGenericAdaptor::LoopConvWsConfig2_IntrOpGenericAdaptor;
  LoopConvWsConfig2_IntrOpAdaptor(LoopConvWsConfig2_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopConvWsConfig2_IntrOp : public ::mlir::Op<LoopConvWsConfig2_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopConvWsConfig2_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopConvWsConfig2_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_conv_ws_config2");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig2_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig2_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig2_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig2_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWsConfig2_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWsConfig2_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWsConfig2_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWsConfig2_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopConvWsConfig2_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopConvWsConfig3_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopConvWsConfig3_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopConvWsConfig3_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_conv_ws_config3", odsAttrs.getContext());
  }

  LoopConvWsConfig3_IntrOpGenericAdaptorBase(LoopConvWsConfig3_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopConvWsConfig3_IntrOpGenericAdaptor : public detail::LoopConvWsConfig3_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopConvWsConfig3_IntrOpGenericAdaptorBase;
public:
  LoopConvWsConfig3_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopConvWsConfig3_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopConvWsConfig3_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopConvWsConfig3_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopConvWsConfig3_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopConvWsConfig3_IntrOpGenericAdaptor(RangeT values, const LoopConvWsConfig3_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopConvWsConfig3_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopConvWsConfig3_IntrOp>>>
  LoopConvWsConfig3_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopConvWsConfig3_IntrOpAdaptor : public LoopConvWsConfig3_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopConvWsConfig3_IntrOpGenericAdaptor::LoopConvWsConfig3_IntrOpGenericAdaptor;
  LoopConvWsConfig3_IntrOpAdaptor(LoopConvWsConfig3_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopConvWsConfig3_IntrOp : public ::mlir::Op<LoopConvWsConfig3_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopConvWsConfig3_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopConvWsConfig3_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_conv_ws_config3");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig3_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig3_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig3_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig3_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWsConfig3_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWsConfig3_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWsConfig3_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWsConfig3_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopConvWsConfig3_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopConvWsConfig4_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopConvWsConfig4_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopConvWsConfig4_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_conv_ws_config4", odsAttrs.getContext());
  }

  LoopConvWsConfig4_IntrOpGenericAdaptorBase(LoopConvWsConfig4_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopConvWsConfig4_IntrOpGenericAdaptor : public detail::LoopConvWsConfig4_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopConvWsConfig4_IntrOpGenericAdaptorBase;
public:
  LoopConvWsConfig4_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopConvWsConfig4_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopConvWsConfig4_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopConvWsConfig4_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopConvWsConfig4_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopConvWsConfig4_IntrOpGenericAdaptor(RangeT values, const LoopConvWsConfig4_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopConvWsConfig4_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopConvWsConfig4_IntrOp>>>
  LoopConvWsConfig4_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopConvWsConfig4_IntrOpAdaptor : public LoopConvWsConfig4_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopConvWsConfig4_IntrOpGenericAdaptor::LoopConvWsConfig4_IntrOpGenericAdaptor;
  LoopConvWsConfig4_IntrOpAdaptor(LoopConvWsConfig4_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopConvWsConfig4_IntrOp : public ::mlir::Op<LoopConvWsConfig4_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopConvWsConfig4_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopConvWsConfig4_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_conv_ws_config4");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig4_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig4_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig4_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig4_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWsConfig4_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWsConfig4_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWsConfig4_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWsConfig4_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopConvWsConfig4_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopConvWsConfig5_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopConvWsConfig5_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopConvWsConfig5_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_conv_ws_config5", odsAttrs.getContext());
  }

  LoopConvWsConfig5_IntrOpGenericAdaptorBase(LoopConvWsConfig5_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopConvWsConfig5_IntrOpGenericAdaptor : public detail::LoopConvWsConfig5_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopConvWsConfig5_IntrOpGenericAdaptorBase;
public:
  LoopConvWsConfig5_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopConvWsConfig5_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopConvWsConfig5_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopConvWsConfig5_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopConvWsConfig5_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopConvWsConfig5_IntrOpGenericAdaptor(RangeT values, const LoopConvWsConfig5_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopConvWsConfig5_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopConvWsConfig5_IntrOp>>>
  LoopConvWsConfig5_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopConvWsConfig5_IntrOpAdaptor : public LoopConvWsConfig5_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopConvWsConfig5_IntrOpGenericAdaptor::LoopConvWsConfig5_IntrOpGenericAdaptor;
  LoopConvWsConfig5_IntrOpAdaptor(LoopConvWsConfig5_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopConvWsConfig5_IntrOp : public ::mlir::Op<LoopConvWsConfig5_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopConvWsConfig5_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopConvWsConfig5_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_conv_ws_config5");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig5_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig5_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig5_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig5_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWsConfig5_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWsConfig5_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWsConfig5_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWsConfig5_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopConvWsConfig5_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopConvWsConfig6_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopConvWsConfig6_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopConvWsConfig6_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_conv_ws_config6", odsAttrs.getContext());
  }

  LoopConvWsConfig6_IntrOpGenericAdaptorBase(LoopConvWsConfig6_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopConvWsConfig6_IntrOpGenericAdaptor : public detail::LoopConvWsConfig6_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopConvWsConfig6_IntrOpGenericAdaptorBase;
public:
  LoopConvWsConfig6_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopConvWsConfig6_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopConvWsConfig6_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopConvWsConfig6_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopConvWsConfig6_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopConvWsConfig6_IntrOpGenericAdaptor(RangeT values, const LoopConvWsConfig6_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopConvWsConfig6_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopConvWsConfig6_IntrOp>>>
  LoopConvWsConfig6_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopConvWsConfig6_IntrOpAdaptor : public LoopConvWsConfig6_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopConvWsConfig6_IntrOpGenericAdaptor::LoopConvWsConfig6_IntrOpGenericAdaptor;
  LoopConvWsConfig6_IntrOpAdaptor(LoopConvWsConfig6_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopConvWsConfig6_IntrOp : public ::mlir::Op<LoopConvWsConfig6_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopConvWsConfig6_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopConvWsConfig6_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_conv_ws_config6");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig6_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig6_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig6_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWsConfig6_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWsConfig6_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWsConfig6_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWsConfig6_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWsConfig6_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopConvWsConfig6_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopConvWs_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopConvWs_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopConvWs_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_conv_ws", odsAttrs.getContext());
  }

  LoopConvWs_IntrOpGenericAdaptorBase(LoopConvWs_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopConvWs_IntrOpGenericAdaptor : public detail::LoopConvWs_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopConvWs_IntrOpGenericAdaptorBase;
public:
  LoopConvWs_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopConvWs_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopConvWs_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopConvWs_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopConvWs_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopConvWs_IntrOpGenericAdaptor(RangeT values, const LoopConvWs_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopConvWs_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopConvWs_IntrOp>>>
  LoopConvWs_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopConvWs_IntrOpAdaptor : public LoopConvWs_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopConvWs_IntrOpGenericAdaptor::LoopConvWs_IntrOpGenericAdaptor;
  LoopConvWs_IntrOpAdaptor(LoopConvWs_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopConvWs_IntrOp : public ::mlir::Op<LoopConvWs_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopConvWs_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopConvWs_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_conv_ws");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWs_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWs_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWs_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopConvWs_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWs_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopConvWs_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWs_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopConvWs_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopConvWs_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopWsConfigAddrsAB_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopWsConfigAddrsAB_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopWsConfigAddrsAB_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_ws_config_addrs_ab", odsAttrs.getContext());
  }

  LoopWsConfigAddrsAB_IntrOpGenericAdaptorBase(LoopWsConfigAddrsAB_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopWsConfigAddrsAB_IntrOpGenericAdaptor : public detail::LoopWsConfigAddrsAB_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopWsConfigAddrsAB_IntrOpGenericAdaptorBase;
public:
  LoopWsConfigAddrsAB_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopWsConfigAddrsAB_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopWsConfigAddrsAB_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopWsConfigAddrsAB_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopWsConfigAddrsAB_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopWsConfigAddrsAB_IntrOpGenericAdaptor(RangeT values, const LoopWsConfigAddrsAB_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopWsConfigAddrsAB_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopWsConfigAddrsAB_IntrOp>>>
  LoopWsConfigAddrsAB_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopWsConfigAddrsAB_IntrOpAdaptor : public LoopWsConfigAddrsAB_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopWsConfigAddrsAB_IntrOpGenericAdaptor::LoopWsConfigAddrsAB_IntrOpGenericAdaptor;
  LoopWsConfigAddrsAB_IntrOpAdaptor(LoopWsConfigAddrsAB_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopWsConfigAddrsAB_IntrOp : public ::mlir::Op<LoopWsConfigAddrsAB_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopWsConfigAddrsAB_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopWsConfigAddrsAB_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_ws_config_addrs_ab");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigAddrsAB_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigAddrsAB_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigAddrsAB_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigAddrsAB_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopWsConfigAddrsAB_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopWsConfigAddrsAB_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopWsConfigAddrsAB_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopWsConfigAddrsAB_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopWsConfigAddrsAB_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopWsConfigAddrsDC_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopWsConfigAddrsDC_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopWsConfigAddrsDC_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_ws_config_addrs_dc", odsAttrs.getContext());
  }

  LoopWsConfigAddrsDC_IntrOpGenericAdaptorBase(LoopWsConfigAddrsDC_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopWsConfigAddrsDC_IntrOpGenericAdaptor : public detail::LoopWsConfigAddrsDC_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopWsConfigAddrsDC_IntrOpGenericAdaptorBase;
public:
  LoopWsConfigAddrsDC_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopWsConfigAddrsDC_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopWsConfigAddrsDC_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopWsConfigAddrsDC_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopWsConfigAddrsDC_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopWsConfigAddrsDC_IntrOpGenericAdaptor(RangeT values, const LoopWsConfigAddrsDC_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopWsConfigAddrsDC_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopWsConfigAddrsDC_IntrOp>>>
  LoopWsConfigAddrsDC_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopWsConfigAddrsDC_IntrOpAdaptor : public LoopWsConfigAddrsDC_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopWsConfigAddrsDC_IntrOpGenericAdaptor::LoopWsConfigAddrsDC_IntrOpGenericAdaptor;
  LoopWsConfigAddrsDC_IntrOpAdaptor(LoopWsConfigAddrsDC_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopWsConfigAddrsDC_IntrOp : public ::mlir::Op<LoopWsConfigAddrsDC_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopWsConfigAddrsDC_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopWsConfigAddrsDC_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_ws_config_addrs_dc");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigAddrsDC_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigAddrsDC_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigAddrsDC_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigAddrsDC_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopWsConfigAddrsDC_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopWsConfigAddrsDC_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopWsConfigAddrsDC_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopWsConfigAddrsDC_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopWsConfigAddrsDC_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopWsConfigBounds_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopWsConfigBounds_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopWsConfigBounds_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_ws_config_bounds", odsAttrs.getContext());
  }

  LoopWsConfigBounds_IntrOpGenericAdaptorBase(LoopWsConfigBounds_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopWsConfigBounds_IntrOpGenericAdaptor : public detail::LoopWsConfigBounds_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopWsConfigBounds_IntrOpGenericAdaptorBase;
public:
  LoopWsConfigBounds_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopWsConfigBounds_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopWsConfigBounds_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopWsConfigBounds_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopWsConfigBounds_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopWsConfigBounds_IntrOpGenericAdaptor(RangeT values, const LoopWsConfigBounds_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopWsConfigBounds_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopWsConfigBounds_IntrOp>>>
  LoopWsConfigBounds_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopWsConfigBounds_IntrOpAdaptor : public LoopWsConfigBounds_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopWsConfigBounds_IntrOpGenericAdaptor::LoopWsConfigBounds_IntrOpGenericAdaptor;
  LoopWsConfigBounds_IntrOpAdaptor(LoopWsConfigBounds_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopWsConfigBounds_IntrOp : public ::mlir::Op<LoopWsConfigBounds_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopWsConfigBounds_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopWsConfigBounds_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_ws_config_bounds");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigBounds_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigBounds_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigBounds_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigBounds_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopWsConfigBounds_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopWsConfigBounds_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopWsConfigBounds_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopWsConfigBounds_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopWsConfigBounds_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopWsConfigStridesAB_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopWsConfigStridesAB_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopWsConfigStridesAB_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_ws_config_strides_ab", odsAttrs.getContext());
  }

  LoopWsConfigStridesAB_IntrOpGenericAdaptorBase(LoopWsConfigStridesAB_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopWsConfigStridesAB_IntrOpGenericAdaptor : public detail::LoopWsConfigStridesAB_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopWsConfigStridesAB_IntrOpGenericAdaptorBase;
public:
  LoopWsConfigStridesAB_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopWsConfigStridesAB_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopWsConfigStridesAB_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopWsConfigStridesAB_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopWsConfigStridesAB_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopWsConfigStridesAB_IntrOpGenericAdaptor(RangeT values, const LoopWsConfigStridesAB_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopWsConfigStridesAB_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopWsConfigStridesAB_IntrOp>>>
  LoopWsConfigStridesAB_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopWsConfigStridesAB_IntrOpAdaptor : public LoopWsConfigStridesAB_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopWsConfigStridesAB_IntrOpGenericAdaptor::LoopWsConfigStridesAB_IntrOpGenericAdaptor;
  LoopWsConfigStridesAB_IntrOpAdaptor(LoopWsConfigStridesAB_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopWsConfigStridesAB_IntrOp : public ::mlir::Op<LoopWsConfigStridesAB_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopWsConfigStridesAB_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopWsConfigStridesAB_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_ws_config_strides_ab");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigStridesAB_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigStridesAB_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigStridesAB_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigStridesAB_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopWsConfigStridesAB_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopWsConfigStridesAB_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopWsConfigStridesAB_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopWsConfigStridesAB_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopWsConfigStridesAB_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopWsConfigStridesDC_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopWsConfigStridesDC_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopWsConfigStridesDC_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_ws_config_strides_dc", odsAttrs.getContext());
  }

  LoopWsConfigStridesDC_IntrOpGenericAdaptorBase(LoopWsConfigStridesDC_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopWsConfigStridesDC_IntrOpGenericAdaptor : public detail::LoopWsConfigStridesDC_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopWsConfigStridesDC_IntrOpGenericAdaptorBase;
public:
  LoopWsConfigStridesDC_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopWsConfigStridesDC_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopWsConfigStridesDC_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopWsConfigStridesDC_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopWsConfigStridesDC_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopWsConfigStridesDC_IntrOpGenericAdaptor(RangeT values, const LoopWsConfigStridesDC_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopWsConfigStridesDC_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopWsConfigStridesDC_IntrOp>>>
  LoopWsConfigStridesDC_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopWsConfigStridesDC_IntrOpAdaptor : public LoopWsConfigStridesDC_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopWsConfigStridesDC_IntrOpGenericAdaptor::LoopWsConfigStridesDC_IntrOpGenericAdaptor;
  LoopWsConfigStridesDC_IntrOpAdaptor(LoopWsConfigStridesDC_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopWsConfigStridesDC_IntrOp : public ::mlir::Op<LoopWsConfigStridesDC_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopWsConfigStridesDC_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopWsConfigStridesDC_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_ws_config_strides_dc");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigStridesDC_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigStridesDC_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigStridesDC_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWsConfigStridesDC_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopWsConfigStridesDC_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopWsConfigStridesDC_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopWsConfigStridesDC_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopWsConfigStridesDC_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopWsConfigStridesDC_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::LoopWs_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoopWs_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoopWs_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.loop_ws", odsAttrs.getContext());
  }

  LoopWs_IntrOpGenericAdaptorBase(LoopWs_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoopWs_IntrOpGenericAdaptor : public detail::LoopWs_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoopWs_IntrOpGenericAdaptorBase;
public:
  LoopWs_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoopWs_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoopWs_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoopWs_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoopWs_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoopWs_IntrOpGenericAdaptor(RangeT values, const LoopWs_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoopWs_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoopWs_IntrOp>>>
  LoopWs_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoopWs_IntrOpAdaptor : public LoopWs_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoopWs_IntrOpGenericAdaptor::LoopWs_IntrOpGenericAdaptor;
  LoopWs_IntrOpAdaptor(LoopWs_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoopWs_IntrOp : public ::mlir::Op<LoopWs_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopWs_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoopWs_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.loop_ws");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWs_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWs_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWs_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static LoopWs_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopWs_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoopWs_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopWs_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoopWs_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::LoopWs_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::Mvin2_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class Mvin2_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  Mvin2_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.mvin2", odsAttrs.getContext());
  }

  Mvin2_IntrOpGenericAdaptorBase(Mvin2_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class Mvin2_IntrOpGenericAdaptor : public detail::Mvin2_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::Mvin2_IntrOpGenericAdaptorBase;
public:
  Mvin2_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  Mvin2_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : Mvin2_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  Mvin2_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : Mvin2_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  Mvin2_IntrOpGenericAdaptor(RangeT values, const Mvin2_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = Mvin2_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, Mvin2_IntrOp>>>
  Mvin2_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class Mvin2_IntrOpAdaptor : public Mvin2_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using Mvin2_IntrOpGenericAdaptor::Mvin2_IntrOpGenericAdaptor;
  Mvin2_IntrOpAdaptor(Mvin2_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class Mvin2_IntrOp : public ::mlir::Op<Mvin2_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Mvin2_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = Mvin2_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.mvin2");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvin2_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvin2_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvin2_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvin2_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Mvin2_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Mvin2_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Mvin2_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Mvin2_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::Mvin2_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::Mvin3_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class Mvin3_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  Mvin3_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.mvin3", odsAttrs.getContext());
  }

  Mvin3_IntrOpGenericAdaptorBase(Mvin3_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class Mvin3_IntrOpGenericAdaptor : public detail::Mvin3_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::Mvin3_IntrOpGenericAdaptorBase;
public:
  Mvin3_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  Mvin3_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : Mvin3_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  Mvin3_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : Mvin3_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  Mvin3_IntrOpGenericAdaptor(RangeT values, const Mvin3_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = Mvin3_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, Mvin3_IntrOp>>>
  Mvin3_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class Mvin3_IntrOpAdaptor : public Mvin3_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using Mvin3_IntrOpGenericAdaptor::Mvin3_IntrOpGenericAdaptor;
  Mvin3_IntrOpAdaptor(Mvin3_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class Mvin3_IntrOp : public ::mlir::Op<Mvin3_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Mvin3_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = Mvin3_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.mvin3");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvin3_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvin3_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvin3_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvin3_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Mvin3_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Mvin3_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Mvin3_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Mvin3_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::Mvin3_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::Mvin_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class Mvin_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  Mvin_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.mvin", odsAttrs.getContext());
  }

  Mvin_IntrOpGenericAdaptorBase(Mvin_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class Mvin_IntrOpGenericAdaptor : public detail::Mvin_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::Mvin_IntrOpGenericAdaptorBase;
public:
  Mvin_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  Mvin_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : Mvin_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  Mvin_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : Mvin_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  Mvin_IntrOpGenericAdaptor(RangeT values, const Mvin_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = Mvin_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, Mvin_IntrOp>>>
  Mvin_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class Mvin_IntrOpAdaptor : public Mvin_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using Mvin_IntrOpGenericAdaptor::Mvin_IntrOpGenericAdaptor;
  Mvin_IntrOpAdaptor(Mvin_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class Mvin_IntrOp : public ::mlir::Op<Mvin_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Mvin_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = Mvin_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.mvin");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvin_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvin_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvin_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvin_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Mvin_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Mvin_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Mvin_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Mvin_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::Mvin_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::Mvout_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class Mvout_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  Mvout_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.mvout", odsAttrs.getContext());
  }

  Mvout_IntrOpGenericAdaptorBase(Mvout_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class Mvout_IntrOpGenericAdaptor : public detail::Mvout_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::Mvout_IntrOpGenericAdaptorBase;
public:
  Mvout_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  Mvout_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : Mvout_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  Mvout_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : Mvout_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  Mvout_IntrOpGenericAdaptor(RangeT values, const Mvout_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = Mvout_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, Mvout_IntrOp>>>
  Mvout_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class Mvout_IntrOpAdaptor : public Mvout_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using Mvout_IntrOpGenericAdaptor::Mvout_IntrOpGenericAdaptor;
  Mvout_IntrOpAdaptor(Mvout_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class Mvout_IntrOp : public ::mlir::Op<Mvout_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Mvout_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = Mvout_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.mvout");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvout_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvout_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvout_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Mvout_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Mvout_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Mvout_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Mvout_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Mvout_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::Mvout_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::Preload_IntrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class Preload_IntrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  Preload_IntrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.intr.preload", odsAttrs.getContext());
  }

  Preload_IntrOpGenericAdaptorBase(Preload_IntrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class Preload_IntrOpGenericAdaptor : public detail::Preload_IntrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::Preload_IntrOpGenericAdaptorBase;
public:
  Preload_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  Preload_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : Preload_IntrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  Preload_IntrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : Preload_IntrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  Preload_IntrOpGenericAdaptor(RangeT values, const Preload_IntrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = Preload_IntrOp, typename = std::enable_if_t<std::is_same_v<LateInst, Preload_IntrOp>>>
  Preload_IntrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class Preload_IntrOpAdaptor : public Preload_IntrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using Preload_IntrOpGenericAdaptor::Preload_IntrOpGenericAdaptor;
  Preload_IntrOpAdaptor(Preload_IntrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class Preload_IntrOp : public ::mlir::Op<Preload_IntrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Preload_IntrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = Preload_IntrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.intr.preload");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Preload_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Preload_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Preload_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static Preload_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Preload_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Preload_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Preload_IntrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Preload_IntrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::Preload_IntrOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::Mvin2Op declarations
//===----------------------------------------------------------------------===//

namespace detail {

class Mvin2OpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  Mvin2OpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.mvin2", odsAttrs.getContext());
  }

  Mvin2OpGenericAdaptorBase(Mvin2Op op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class Mvin2OpGenericAdaptor : public detail::Mvin2OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::Mvin2OpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
  static constexpr int odsIndex_addr = 1;
public:
  Mvin2OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  Mvin2OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : Mvin2OpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  Mvin2OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : Mvin2OpGenericAdaptor(values, attrs, Properties{}, {}) {}

  Mvin2OpGenericAdaptor(RangeT values, const Mvin2OpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = Mvin2Op, typename = std::enable_if_t<std::is_same_v<LateInst, Mvin2Op>>>
  Mvin2OpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getAddr() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class Mvin2OpAdaptor : public Mvin2OpGenericAdaptor<::mlir::ValueRange> {
public:
  using Mvin2OpGenericAdaptor::Mvin2OpGenericAdaptor;
  Mvin2OpAdaptor(Mvin2Op op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class Mvin2Op : public ::mlir::Op<Mvin2Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Mvin2OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = Mvin2OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static constexpr int odsIndex_addr = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.mvin2");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getAddr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getAddrMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value addr);
  static Mvin2Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::Value addr);
  static Mvin2Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::Value addr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value addr);
  static Mvin2Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value addr);
  static Mvin2Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value addr);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Mvin2Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Mvin2Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Mvin2Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Mvin2Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::Mvin2Op)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::Mvin3Op declarations
//===----------------------------------------------------------------------===//

namespace detail {

class Mvin3OpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  Mvin3OpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.mvin3", odsAttrs.getContext());
  }

  Mvin3OpGenericAdaptorBase(Mvin3Op op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class Mvin3OpGenericAdaptor : public detail::Mvin3OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::Mvin3OpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
  static constexpr int odsIndex_addr = 1;
public:
  Mvin3OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  Mvin3OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : Mvin3OpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  Mvin3OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : Mvin3OpGenericAdaptor(values, attrs, Properties{}, {}) {}

  Mvin3OpGenericAdaptor(RangeT values, const Mvin3OpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = Mvin3Op, typename = std::enable_if_t<std::is_same_v<LateInst, Mvin3Op>>>
  Mvin3OpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getAddr() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class Mvin3OpAdaptor : public Mvin3OpGenericAdaptor<::mlir::ValueRange> {
public:
  using Mvin3OpGenericAdaptor::Mvin3OpGenericAdaptor;
  Mvin3OpAdaptor(Mvin3Op op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class Mvin3Op : public ::mlir::Op<Mvin3Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Mvin3OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = Mvin3OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static constexpr int odsIndex_addr = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.mvin3");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getAddr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getAddrMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value addr);
  static Mvin3Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::Value addr);
  static Mvin3Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::Value addr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value addr);
  static Mvin3Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value addr);
  static Mvin3Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value addr);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Mvin3Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Mvin3Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Mvin3Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Mvin3Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::Mvin3Op)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::MvinOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class MvinOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  MvinOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.mvin", odsAttrs.getContext());
  }

  MvinOpGenericAdaptorBase(MvinOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class MvinOpGenericAdaptor : public detail::MvinOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MvinOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
  static constexpr int odsIndex_addr = 1;
public:
  MvinOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MvinOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : MvinOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  MvinOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : MvinOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  MvinOpGenericAdaptor(RangeT values, const MvinOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = MvinOp, typename = std::enable_if_t<std::is_same_v<LateInst, MvinOp>>>
  MvinOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getAddr() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MvinOpAdaptor : public MvinOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MvinOpGenericAdaptor::MvinOpGenericAdaptor;
  MvinOpAdaptor(MvinOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class MvinOp : public ::mlir::Op<MvinOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MvinOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MvinOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static constexpr int odsIndex_addr = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.mvin");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getAddr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getAddrMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value addr);
  static MvinOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::Value addr);
  static MvinOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::Value addr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value addr);
  static MvinOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value addr);
  static MvinOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value addr);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MvinOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MvinOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static MvinOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static MvinOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::MvinOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::MvoutOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class MvoutOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  MvoutOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.mvout", odsAttrs.getContext());
  }

  MvoutOpGenericAdaptorBase(MvoutOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class MvoutOpGenericAdaptor : public detail::MvoutOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MvoutOpGenericAdaptorBase;
  static constexpr int odsIndex_output = 0;
  static constexpr int odsIndex_addr = 1;
public:
  MvoutOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MvoutOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : MvoutOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  MvoutOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : MvoutOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  MvoutOpGenericAdaptor(RangeT values, const MvoutOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = MvoutOp, typename = std::enable_if_t<std::is_same_v<LateInst, MvoutOp>>>
  MvoutOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getOutput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getAddr() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MvoutOpAdaptor : public MvoutOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MvoutOpGenericAdaptor::MvoutOpGenericAdaptor;
  MvoutOpAdaptor(MvoutOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class MvoutOp : public ::mlir::Op<MvoutOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MvoutOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MvoutOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_output = 0;
  static constexpr int odsIndex_addr = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.mvout");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getOutput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getAddr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getOutputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getAddrMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value output, ::mlir::Value addr);
  static MvoutOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value output, ::mlir::Value addr);
  static MvoutOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value output, ::mlir::Value addr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value output, ::mlir::Value addr);
  static MvoutOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value output, ::mlir::Value addr);
  static MvoutOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value output, ::mlir::Value addr);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MvoutOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MvoutOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static MvoutOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static MvoutOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::MvoutOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::PreloadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class PreloadOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  PreloadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.preload", odsAttrs.getContext());
  }

  PreloadOpGenericAdaptorBase(PreloadOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class PreloadOpGenericAdaptor : public detail::PreloadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PreloadOpGenericAdaptorBase;
  static constexpr int odsIndex_bdAddr = 0;
  static constexpr int odsIndex_cAddr = 1;
  static constexpr int odsIndex_bdRows = 2;
  static constexpr int odsIndex_bdCols = 3;
  static constexpr int odsIndex_cRows = 4;
  static constexpr int odsIndex_cCols = 5;
public:
  PreloadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  PreloadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : PreloadOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  PreloadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : PreloadOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  PreloadOpGenericAdaptor(RangeT values, const PreloadOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = PreloadOp, typename = std::enable_if_t<std::is_same_v<LateInst, PreloadOp>>>
  PreloadOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getBdAddr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getCAddr() {
    return (*getODSOperands(1).begin());
  }

  ValueT getBdRows() {
    return (*getODSOperands(2).begin());
  }

  ValueT getBdCols() {
    return (*getODSOperands(3).begin());
  }

  ValueT getCRows() {
    return (*getODSOperands(4).begin());
  }

  ValueT getCCols() {
    return (*getODSOperands(5).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class PreloadOpAdaptor : public PreloadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PreloadOpGenericAdaptor::PreloadOpGenericAdaptor;
  PreloadOpAdaptor(PreloadOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class PreloadOp : public ::mlir::Op<PreloadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PreloadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PreloadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_bdAddr = 0;
  static constexpr int odsIndex_cAddr = 1;
  static constexpr int odsIndex_bdRows = 2;
  static constexpr int odsIndex_bdCols = 3;
  static constexpr int odsIndex_cRows = 4;
  static constexpr int odsIndex_cCols = 5;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.preload");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getBdAddr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getCAddr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getBdRows() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getBdCols() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(3).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getCRows() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(4).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getCCols() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(5).begin());
  }

  ::mlir::OpOperand &getBdAddrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCAddrMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBdRowsMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBdColsMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCRowsMutable() {
    auto range = getODSOperandIndexAndLength(4);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCColsMutable() {
    auto range = getODSOperandIndexAndLength(5);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value bdAddr, ::mlir::Value cAddr, ::mlir::Value bdRows, ::mlir::Value bdCols, ::mlir::Value cRows, ::mlir::Value cCols);
  static PreloadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value bdAddr, ::mlir::Value cAddr, ::mlir::Value bdRows, ::mlir::Value bdCols, ::mlir::Value cRows, ::mlir::Value cCols);
  static PreloadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value bdAddr, ::mlir::Value cAddr, ::mlir::Value bdRows, ::mlir::Value bdCols, ::mlir::Value cRows, ::mlir::Value cCols);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value bdAddr, ::mlir::Value cAddr, ::mlir::Value bdRows, ::mlir::Value bdCols, ::mlir::Value cRows, ::mlir::Value cCols);
  static PreloadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value bdAddr, ::mlir::Value cAddr, ::mlir::Value bdRows, ::mlir::Value bdCols, ::mlir::Value cRows, ::mlir::Value cCols);
  static PreloadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value bdAddr, ::mlir::Value cAddr, ::mlir::Value bdRows, ::mlir::Value bdCols, ::mlir::Value cRows, ::mlir::Value cCols);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PreloadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PreloadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static PreloadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static PreloadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::PreloadOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::PreloadZerosOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class PreloadZerosOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  PreloadZerosOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.preload_zeros", odsAttrs.getContext());
  }

  PreloadZerosOpGenericAdaptorBase(PreloadZerosOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class PreloadZerosOpGenericAdaptor : public detail::PreloadZerosOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PreloadZerosOpGenericAdaptorBase;
  static constexpr int odsIndex_addr = 0;
  static constexpr int odsIndex_cRows = 1;
  static constexpr int odsIndex_cCols = 2;
public:
  PreloadZerosOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  PreloadZerosOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : PreloadZerosOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  PreloadZerosOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : PreloadZerosOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  PreloadZerosOpGenericAdaptor(RangeT values, const PreloadZerosOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = PreloadZerosOp, typename = std::enable_if_t<std::is_same_v<LateInst, PreloadZerosOp>>>
  PreloadZerosOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAddr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getCRows() {
    return (*getODSOperands(1).begin());
  }

  ValueT getCCols() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class PreloadZerosOpAdaptor : public PreloadZerosOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PreloadZerosOpGenericAdaptor::PreloadZerosOpGenericAdaptor;
  PreloadZerosOpAdaptor(PreloadZerosOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class PreloadZerosOp : public ::mlir::Op<PreloadZerosOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PreloadZerosOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PreloadZerosOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_addr = 0;
  static constexpr int odsIndex_cRows = 1;
  static constexpr int odsIndex_cCols = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.preload_zeros");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getAddr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getCRows() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getCCols() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getAddrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCRowsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCColsMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value addr, ::mlir::Value cRows, ::mlir::Value cCols);
  static PreloadZerosOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value addr, ::mlir::Value cRows, ::mlir::Value cCols);
  static PreloadZerosOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value addr, ::mlir::Value cRows, ::mlir::Value cCols);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, ::mlir::Value cRows, ::mlir::Value cCols);
  static PreloadZerosOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value addr, ::mlir::Value cRows, ::mlir::Value cCols);
  static PreloadZerosOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value addr, ::mlir::Value cRows, ::mlir::Value cCols);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PreloadZerosOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PreloadZerosOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static PreloadZerosOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static PreloadZerosOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::PreloadZerosOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::PrintOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class PrintOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  PrintOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.print", odsAttrs.getContext());
  }

  PrintOpGenericAdaptorBase(PrintOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class PrintOpGenericAdaptor : public detail::PrintOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PrintOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  PrintOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  PrintOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : PrintOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  PrintOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : PrintOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  PrintOpGenericAdaptor(RangeT values, const PrintOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = PrintOp, typename = std::enable_if_t<std::is_same_v<LateInst, PrintOp>>>
  PrintOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class PrintOpAdaptor : public PrintOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PrintOpGenericAdaptor::PrintOpGenericAdaptor;
  PrintOpAdaptor(PrintOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class PrintOp : public ::mlir::Op<PrintOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PrintOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PrintOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.print");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input);
  static PrintOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input);
  static PrintOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static PrintOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static PrintOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PrintOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PrintOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static PrintOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static PrintOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::PrintOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::TileConvOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class TileConvOpGenericAdaptorBase {
public:
  struct Properties {
    using actTy = ::mlir::IntegerAttr;
    actTy act;

    auto getAct() const {
      auto &propStorage = this->act;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setAct(const ::mlir::IntegerAttr &propValue) {
      this->act = propValue;
    }
    using inputDilationTy = ::mlir::IntegerAttr;
    inputDilationTy inputDilation;

    auto getInputDilation() const {
      auto &propStorage = this->inputDilation;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setInputDilation(const ::mlir::IntegerAttr &propValue) {
      this->inputDilation = propValue;
    }
    using kernelDilationTy = ::mlir::IntegerAttr;
    kernelDilationTy kernelDilation;

    auto getKernelDilation() const {
      auto &propStorage = this->kernelDilation;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setKernelDilation(const ::mlir::IntegerAttr &propValue) {
      this->kernelDilation = propValue;
    }
    using paddingTy = ::mlir::IntegerAttr;
    paddingTy padding;

    auto getPadding() const {
      auto &propStorage = this->padding;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setPadding(const ::mlir::IntegerAttr &propValue) {
      this->padding = propValue;
    }
    using poolPaddingTy = ::mlir::IntegerAttr;
    poolPaddingTy poolPadding;

    auto getPoolPadding() const {
      auto &propStorage = this->poolPadding;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setPoolPadding(const ::mlir::IntegerAttr &propValue) {
      this->poolPadding = propValue;
    }
    using poolSizeTy = ::mlir::IntegerAttr;
    poolSizeTy poolSize;

    auto getPoolSize() const {
      auto &propStorage = this->poolSize;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setPoolSize(const ::mlir::IntegerAttr &propValue) {
      this->poolSize = propValue;
    }
    using poolStrideTy = ::mlir::IntegerAttr;
    poolStrideTy poolStride;

    auto getPoolStride() const {
      auto &propStorage = this->poolStride;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setPoolStride(const ::mlir::IntegerAttr &propValue) {
      this->poolStride = propValue;
    }
    using scaleTy = ::mlir::FloatAttr;
    scaleTy scale;

    auto getScale() const {
      auto &propStorage = this->scale;
      return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(propStorage);
    }
    void setScale(const ::mlir::FloatAttr &propValue) {
      this->scale = propValue;
    }
    using strideTy = ::mlir::IntegerAttr;
    strideTy stride;

    auto getStride() const {
      auto &propStorage = this->stride;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setStride(const ::mlir::IntegerAttr &propValue) {
      this->stride = propValue;
    }
    using transInput3120Ty = ::mlir::BoolAttr;
    transInput3120Ty transInput3120;

    auto getTransInput3120() const {
      auto &propStorage = this->transInput3120;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setTransInput3120(const ::mlir::BoolAttr &propValue) {
      this->transInput3120 = propValue;
    }
    using transOutput1203Ty = ::mlir::BoolAttr;
    transOutput1203Ty transOutput1203;

    auto getTransOutput1203() const {
      auto &propStorage = this->transOutput1203;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setTransOutput1203(const ::mlir::BoolAttr &propValue) {
      this->transOutput1203 = propValue;
    }
    using transWeight0132Ty = ::mlir::BoolAttr;
    transWeight0132Ty transWeight0132;

    auto getTransWeight0132() const {
      auto &propStorage = this->transWeight0132;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setTransWeight0132(const ::mlir::BoolAttr &propValue) {
      this->transWeight0132 = propValue;
    }
    using transWeight1203Ty = ::mlir::BoolAttr;
    transWeight1203Ty transWeight1203;

    auto getTransWeight1203() const {
      auto &propStorage = this->transWeight1203;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setTransWeight1203(const ::mlir::BoolAttr &propValue) {
      this->transWeight1203 = propValue;
    }
    using wrot180Ty = ::mlir::BoolAttr;
    wrot180Ty wrot180;

    auto getWrot180() const {
      auto &propStorage = this->wrot180;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setWrot180(const ::mlir::BoolAttr &propValue) {
      this->wrot180 = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.act == this->act &&
        rhs.inputDilation == this->inputDilation &&
        rhs.kernelDilation == this->kernelDilation &&
        rhs.padding == this->padding &&
        rhs.poolPadding == this->poolPadding &&
        rhs.poolSize == this->poolSize &&
        rhs.poolStride == this->poolStride &&
        rhs.scale == this->scale &&
        rhs.stride == this->stride &&
        rhs.transInput3120 == this->transInput3120 &&
        rhs.transOutput1203 == this->transOutput1203 &&
        rhs.transWeight0132 == this->transWeight0132 &&
        rhs.transWeight1203 == this->transWeight1203 &&
        rhs.wrot180 == this->wrot180 &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  TileConvOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.tile_conv", odsAttrs.getContext());
  }

  TileConvOpGenericAdaptorBase(TileConvOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::FloatAttr getScaleAttr();
  ::llvm::APFloat getScale();
  ::mlir::IntegerAttr getStrideAttr();
  uint64_t getStride();
  ::mlir::IntegerAttr getInputDilationAttr();
  uint64_t getInputDilation();
  ::mlir::IntegerAttr getKernelDilationAttr();
  uint64_t getKernelDilation();
  ::mlir::IntegerAttr getPaddingAttr();
  uint64_t getPadding();
  ::mlir::BoolAttr getWrot180Attr();
  bool getWrot180();
  ::mlir::BoolAttr getTransOutput1203Attr();
  bool getTransOutput1203();
  ::mlir::BoolAttr getTransInput3120Attr();
  bool getTransInput3120();
  ::mlir::BoolAttr getTransWeight1203Attr();
  bool getTransWeight1203();
  ::mlir::BoolAttr getTransWeight0132Attr();
  bool getTransWeight0132();
  ::mlir::IntegerAttr getActAttr();
  uint64_t getAct();
  ::mlir::IntegerAttr getPoolSizeAttr();
  uint64_t getPoolSize();
  ::mlir::IntegerAttr getPoolStrideAttr();
  uint64_t getPoolStride();
  ::mlir::IntegerAttr getPoolPaddingAttr();
  uint64_t getPoolPadding();
};

} // namespace detail
template <typename RangeT>
class TileConvOpGenericAdaptor : public detail::TileConvOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TileConvOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
  static constexpr int odsIndex_weights = 1;
  static constexpr int odsIndex_bias = 2;
  static constexpr int odsIndex_output = 3;
  static constexpr int odsIndex_outRowDim = 4;
  static constexpr int odsIndex_outColDim = 5;
  static constexpr int odsIndex_kernelDim = 6;
public:
  TileConvOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TileConvOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TileConvOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  TileConvOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : TileConvOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  TileConvOpGenericAdaptor(RangeT values, const TileConvOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TileConvOp, typename = std::enable_if_t<std::is_same_v<LateInst, TileConvOp>>>
  TileConvOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  ValueT getWeights() {
    return (*getODSOperands(1).begin());
  }

  ValueT getBias() {
    return (*getODSOperands(2).begin());
  }

  ValueT getOutput() {
    return (*getODSOperands(3).begin());
  }

  ValueT getOutRowDim() {
    return (*getODSOperands(4).begin());
  }

  ValueT getOutColDim() {
    return (*getODSOperands(5).begin());
  }

  ValueT getKernelDim() {
    return (*getODSOperands(6).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TileConvOpAdaptor : public TileConvOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TileConvOpGenericAdaptor::TileConvOpGenericAdaptor;
  TileConvOpAdaptor(TileConvOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TileConvOp : public ::mlir::Op<TileConvOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<7>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TileConvOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TileConvOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  static constexpr int odsIndex_weights = 1;
  static constexpr int odsIndex_bias = 2;
  static constexpr int odsIndex_output = 3;
  static constexpr int odsIndex_outRowDim = 4;
  static constexpr int odsIndex_outColDim = 5;
  static constexpr int odsIndex_kernelDim = 6;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("act"), ::llvm::StringRef("inputDilation"), ::llvm::StringRef("kernelDilation"), ::llvm::StringRef("padding"), ::llvm::StringRef("poolPadding"), ::llvm::StringRef("poolSize"), ::llvm::StringRef("poolStride"), ::llvm::StringRef("scale"), ::llvm::StringRef("stride"), ::llvm::StringRef("transInput3120"), ::llvm::StringRef("transOutput1203"), ::llvm::StringRef("transWeight0132"), ::llvm::StringRef("transWeight1203"), ::llvm::StringRef("wrot180")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getActAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getActAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getInputDilationAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getInputDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getKernelDilationAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getKernelDilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getPaddingAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getPoolPaddingAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getPoolPaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getPoolSizeAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getPoolSizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getPoolStrideAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getPoolStrideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getScaleAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getStrideAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getStrideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getTransInput3120AttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getTransInput3120AttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getTransOutput1203AttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getTransOutput1203AttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  ::mlir::StringAttr getTransWeight0132AttrName() {
    return getAttributeNameForIndex(11);
  }

  static ::mlir::StringAttr getTransWeight0132AttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 11);
  }

  ::mlir::StringAttr getTransWeight1203AttrName() {
    return getAttributeNameForIndex(12);
  }

  static ::mlir::StringAttr getTransWeight1203AttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 12);
  }

  ::mlir::StringAttr getWrot180AttrName() {
    return getAttributeNameForIndex(13);
  }

  static ::mlir::StringAttr getWrot180AttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 13);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.tile_conv");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::MemRefType> getWeights() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::MemRefType> getBias() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::MemRefType> getOutput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(3).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getOutRowDim() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(4).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getOutColDim() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(5).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getKernelDim() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(6).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getWeightsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBiasMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getOutputMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getOutRowDimMutable() {
    auto range = getODSOperandIndexAndLength(4);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getOutColDimMutable() {
    auto range = getODSOperandIndexAndLength(5);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getKernelDimMutable() {
    auto range = getODSOperandIndexAndLength(6);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::FloatAttr getScaleAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().scale);
  }

  ::llvm::APFloat getScale();
  ::mlir::IntegerAttr getStrideAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().stride);
  }

  uint64_t getStride();
  ::mlir::IntegerAttr getInputDilationAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().inputDilation);
  }

  uint64_t getInputDilation();
  ::mlir::IntegerAttr getKernelDilationAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().kernelDilation);
  }

  uint64_t getKernelDilation();
  ::mlir::IntegerAttr getPaddingAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().padding);
  }

  uint64_t getPadding();
  ::mlir::BoolAttr getWrot180Attr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().wrot180);
  }

  bool getWrot180();
  ::mlir::BoolAttr getTransOutput1203Attr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().transOutput1203);
  }

  bool getTransOutput1203();
  ::mlir::BoolAttr getTransInput3120Attr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().transInput3120);
  }

  bool getTransInput3120();
  ::mlir::BoolAttr getTransWeight1203Attr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().transWeight1203);
  }

  bool getTransWeight1203();
  ::mlir::BoolAttr getTransWeight0132Attr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().transWeight0132);
  }

  bool getTransWeight0132();
  ::mlir::IntegerAttr getActAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().act);
  }

  uint64_t getAct();
  ::mlir::IntegerAttr getPoolSizeAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().poolSize);
  }

  uint64_t getPoolSize();
  ::mlir::IntegerAttr getPoolStrideAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().poolStride);
  }

  uint64_t getPoolStride();
  ::mlir::IntegerAttr getPoolPaddingAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().poolPadding);
  }

  uint64_t getPoolPadding();
  void setScaleAttr(::mlir::FloatAttr attr) {
    getProperties().scale = attr;
  }

  void setScale(::llvm::APFloat attrValue);
  void setStrideAttr(::mlir::IntegerAttr attr) {
    getProperties().stride = attr;
  }

  void setStride(uint64_t attrValue);
  void setInputDilationAttr(::mlir::IntegerAttr attr) {
    getProperties().inputDilation = attr;
  }

  void setInputDilation(uint64_t attrValue);
  void setKernelDilationAttr(::mlir::IntegerAttr attr) {
    getProperties().kernelDilation = attr;
  }

  void setKernelDilation(uint64_t attrValue);
  void setPaddingAttr(::mlir::IntegerAttr attr) {
    getProperties().padding = attr;
  }

  void setPadding(uint64_t attrValue);
  void setWrot180Attr(::mlir::BoolAttr attr) {
    getProperties().wrot180 = attr;
  }

  void setWrot180(bool attrValue);
  void setTransOutput1203Attr(::mlir::BoolAttr attr) {
    getProperties().transOutput1203 = attr;
  }

  void setTransOutput1203(bool attrValue);
  void setTransInput3120Attr(::mlir::BoolAttr attr) {
    getProperties().transInput3120 = attr;
  }

  void setTransInput3120(bool attrValue);
  void setTransWeight1203Attr(::mlir::BoolAttr attr) {
    getProperties().transWeight1203 = attr;
  }

  void setTransWeight1203(bool attrValue);
  void setTransWeight0132Attr(::mlir::BoolAttr attr) {
    getProperties().transWeight0132 = attr;
  }

  void setTransWeight0132(bool attrValue);
  void setActAttr(::mlir::IntegerAttr attr) {
    getProperties().act = attr;
  }

  void setAct(uint64_t attrValue);
  void setPoolSizeAttr(::mlir::IntegerAttr attr) {
    getProperties().poolSize = attr;
  }

  void setPoolSize(uint64_t attrValue);
  void setPoolStrideAttr(::mlir::IntegerAttr attr) {
    getProperties().poolStride = attr;
  }

  void setPoolStride(uint64_t attrValue);
  void setPoolPaddingAttr(::mlir::IntegerAttr attr) {
    getProperties().poolPadding = attr;
  }

  void setPoolPadding(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::mlir::FloatAttr scale, ::mlir::IntegerAttr stride, ::mlir::IntegerAttr inputDilation = nullptr, ::mlir::IntegerAttr kernelDilation = nullptr, ::mlir::IntegerAttr padding = nullptr, ::mlir::BoolAttr wrot180 = nullptr, ::mlir::BoolAttr transOutput1203 = nullptr, ::mlir::BoolAttr transInput3120 = nullptr, ::mlir::BoolAttr transWeight1203 = nullptr, ::mlir::BoolAttr transWeight0132 = nullptr, ::mlir::IntegerAttr act = nullptr, ::mlir::IntegerAttr poolSize = nullptr, ::mlir::IntegerAttr poolStride = nullptr, ::mlir::IntegerAttr poolPadding = nullptr);
  static TileConvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::mlir::FloatAttr scale, ::mlir::IntegerAttr stride, ::mlir::IntegerAttr inputDilation = nullptr, ::mlir::IntegerAttr kernelDilation = nullptr, ::mlir::IntegerAttr padding = nullptr, ::mlir::BoolAttr wrot180 = nullptr, ::mlir::BoolAttr transOutput1203 = nullptr, ::mlir::BoolAttr transInput3120 = nullptr, ::mlir::BoolAttr transWeight1203 = nullptr, ::mlir::BoolAttr transWeight0132 = nullptr, ::mlir::IntegerAttr act = nullptr, ::mlir::IntegerAttr poolSize = nullptr, ::mlir::IntegerAttr poolStride = nullptr, ::mlir::IntegerAttr poolPadding = nullptr);
  static TileConvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::mlir::FloatAttr scale, ::mlir::IntegerAttr stride, ::mlir::IntegerAttr inputDilation = nullptr, ::mlir::IntegerAttr kernelDilation = nullptr, ::mlir::IntegerAttr padding = nullptr, ::mlir::BoolAttr wrot180 = nullptr, ::mlir::BoolAttr transOutput1203 = nullptr, ::mlir::BoolAttr transInput3120 = nullptr, ::mlir::BoolAttr transWeight1203 = nullptr, ::mlir::BoolAttr transWeight0132 = nullptr, ::mlir::IntegerAttr act = nullptr, ::mlir::IntegerAttr poolSize = nullptr, ::mlir::IntegerAttr poolStride = nullptr, ::mlir::IntegerAttr poolPadding = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::mlir::FloatAttr scale, ::mlir::IntegerAttr stride, ::mlir::IntegerAttr inputDilation = nullptr, ::mlir::IntegerAttr kernelDilation = nullptr, ::mlir::IntegerAttr padding = nullptr, ::mlir::BoolAttr wrot180 = nullptr, ::mlir::BoolAttr transOutput1203 = nullptr, ::mlir::BoolAttr transInput3120 = nullptr, ::mlir::BoolAttr transWeight1203 = nullptr, ::mlir::BoolAttr transWeight0132 = nullptr, ::mlir::IntegerAttr act = nullptr, ::mlir::IntegerAttr poolSize = nullptr, ::mlir::IntegerAttr poolStride = nullptr, ::mlir::IntegerAttr poolPadding = nullptr);
  static TileConvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::mlir::FloatAttr scale, ::mlir::IntegerAttr stride, ::mlir::IntegerAttr inputDilation = nullptr, ::mlir::IntegerAttr kernelDilation = nullptr, ::mlir::IntegerAttr padding = nullptr, ::mlir::BoolAttr wrot180 = nullptr, ::mlir::BoolAttr transOutput1203 = nullptr, ::mlir::BoolAttr transInput3120 = nullptr, ::mlir::BoolAttr transWeight1203 = nullptr, ::mlir::BoolAttr transWeight0132 = nullptr, ::mlir::IntegerAttr act = nullptr, ::mlir::IntegerAttr poolSize = nullptr, ::mlir::IntegerAttr poolStride = nullptr, ::mlir::IntegerAttr poolPadding = nullptr);
  static TileConvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::mlir::FloatAttr scale, ::mlir::IntegerAttr stride, ::mlir::IntegerAttr inputDilation = nullptr, ::mlir::IntegerAttr kernelDilation = nullptr, ::mlir::IntegerAttr padding = nullptr, ::mlir::BoolAttr wrot180 = nullptr, ::mlir::BoolAttr transOutput1203 = nullptr, ::mlir::BoolAttr transInput3120 = nullptr, ::mlir::BoolAttr transWeight1203 = nullptr, ::mlir::BoolAttr transWeight0132 = nullptr, ::mlir::IntegerAttr act = nullptr, ::mlir::IntegerAttr poolSize = nullptr, ::mlir::IntegerAttr poolStride = nullptr, ::mlir::IntegerAttr poolPadding = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::llvm::APFloat scale, uint64_t stride = 1, uint64_t inputDilation = 1, uint64_t kernelDilation = 1, uint64_t padding = 0, bool wrot180 = false, bool transOutput1203 = false, bool transInput3120 = false, bool transWeight1203 = false, bool transWeight0132 = false, uint64_t act = 0, uint64_t poolSize = 0, uint64_t poolStride = 0, uint64_t poolPadding = 0);
  static TileConvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::llvm::APFloat scale, uint64_t stride = 1, uint64_t inputDilation = 1, uint64_t kernelDilation = 1, uint64_t padding = 0, bool wrot180 = false, bool transOutput1203 = false, bool transInput3120 = false, bool transWeight1203 = false, bool transWeight0132 = false, uint64_t act = 0, uint64_t poolSize = 0, uint64_t poolStride = 0, uint64_t poolPadding = 0);
  static TileConvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::llvm::APFloat scale, uint64_t stride = 1, uint64_t inputDilation = 1, uint64_t kernelDilation = 1, uint64_t padding = 0, bool wrot180 = false, bool transOutput1203 = false, bool transInput3120 = false, bool transWeight1203 = false, bool transWeight0132 = false, uint64_t act = 0, uint64_t poolSize = 0, uint64_t poolStride = 0, uint64_t poolPadding = 0);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::llvm::APFloat scale, uint64_t stride = 1, uint64_t inputDilation = 1, uint64_t kernelDilation = 1, uint64_t padding = 0, bool wrot180 = false, bool transOutput1203 = false, bool transInput3120 = false, bool transWeight1203 = false, bool transWeight0132 = false, uint64_t act = 0, uint64_t poolSize = 0, uint64_t poolStride = 0, uint64_t poolPadding = 0);
  static TileConvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::llvm::APFloat scale, uint64_t stride = 1, uint64_t inputDilation = 1, uint64_t kernelDilation = 1, uint64_t padding = 0, bool wrot180 = false, bool transOutput1203 = false, bool transInput3120 = false, bool transWeight1203 = false, bool transWeight0132 = false, uint64_t act = 0, uint64_t poolSize = 0, uint64_t poolStride = 0, uint64_t poolPadding = 0);
  static TileConvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weights, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value outRowDim, ::mlir::Value outColDim, ::mlir::Value kernelDim, ::llvm::APFloat scale, uint64_t stride = 1, uint64_t inputDilation = 1, uint64_t kernelDilation = 1, uint64_t padding = 0, bool wrot180 = false, bool transOutput1203 = false, bool transInput3120 = false, bool transWeight1203 = false, bool transWeight0132 = false, uint64_t act = 0, uint64_t poolSize = 0, uint64_t poolStride = 0, uint64_t poolPadding = 0);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TileConvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TileConvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TileConvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TileConvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void populateDefaultProperties(::mlir::OperationName opName, Properties &properties);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 14 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::TileConvOp)

namespace merlin::gemmini {


//===----------------------------------------------------------------------===//
// ::merlin::gemmini::TileMatMulOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class TileMatMulOpGenericAdaptorBase {
public:
  struct Properties {
    using aScaleFactorTy = ::mlir::FloatAttr;
    aScaleFactorTy aScaleFactor;

    auto getAScaleFactor() const {
      auto &propStorage = this->aScaleFactor;
      return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(propStorage);
    }
    void setAScaleFactor(const ::mlir::FloatAttr &propValue) {
      this->aScaleFactor = propValue;
    }
    using aTransposeTy = ::mlir::BoolAttr;
    aTransposeTy aTranspose;

    auto getATranspose() const {
      auto &propStorage = this->aTranspose;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setATranspose(const ::mlir::BoolAttr &propValue) {
      this->aTranspose = propValue;
    }
    using accScaleTy = ::mlir::FloatAttr;
    accScaleTy accScale;

    auto getAccScale() const {
      auto &propStorage = this->accScale;
      return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(propStorage);
    }
    void setAccScale(const ::mlir::FloatAttr &propValue) {
      this->accScale = propValue;
    }
    using actTy = ::mlir::IntegerAttr;
    actTy act;

    auto getAct() const {
      auto &propStorage = this->act;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setAct(const ::mlir::IntegerAttr &propValue) {
      this->act = propValue;
    }
    using bScaleFactorTy = ::mlir::FloatAttr;
    bScaleFactorTy bScaleFactor;

    auto getBScaleFactor() const {
      auto &propStorage = this->bScaleFactor;
      return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(propStorage);
    }
    void setBScaleFactor(const ::mlir::FloatAttr &propValue) {
      this->bScaleFactor = propValue;
    }
    using bTransposeTy = ::mlir::BoolAttr;
    bTransposeTy bTranspose;

    auto getBTranspose() const {
      auto &propStorage = this->bTranspose;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setBTranspose(const ::mlir::BoolAttr &propValue) {
      this->bTranspose = propValue;
    }
    using bertScaleTy = ::mlir::FloatAttr;
    bertScaleTy bertScale;

    auto getBertScale() const {
      auto &propStorage = this->bertScale;
      return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(propStorage);
    }
    void setBertScale(const ::mlir::FloatAttr &propValue) {
      this->bertScale = propValue;
    }
    using dScaleFactorTy = ::mlir::FloatAttr;
    dScaleFactorTy dScaleFactor;

    auto getDScaleFactor() const {
      auto &propStorage = this->dScaleFactor;
      return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(propStorage);
    }
    void setDScaleFactor(const ::mlir::FloatAttr &propValue) {
      this->dScaleFactor = propValue;
    }
    using dataflowTy = ::mlir::IntegerAttr;
    dataflowTy dataflow;

    auto getDataflow() const {
      auto &propStorage = this->dataflow;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setDataflow(const ::mlir::IntegerAttr &propValue) {
      this->dataflow = propValue;
    }
    using fullCTy = ::mlir::BoolAttr;
    fullCTy fullC;

    auto getFullC() const {
      auto &propStorage = this->fullC;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setFullC(const ::mlir::BoolAttr &propValue) {
      this->fullC = propValue;
    }
    using lowDTy = ::mlir::BoolAttr;
    lowDTy lowD;

    auto getLowD() const {
      auto &propStorage = this->lowD;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setLowD(const ::mlir::BoolAttr &propValue) {
      this->lowD = propValue;
    }
    using repeatingBiasTy = ::mlir::BoolAttr;
    repeatingBiasTy repeatingBias;

    auto getRepeatingBias() const {
      auto &propStorage = this->repeatingBias;
      return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(propStorage);
    }
    void setRepeatingBias(const ::mlir::BoolAttr &propValue) {
      this->repeatingBias = propValue;
    }
    using weightATy = ::mlir::IntegerAttr;
    weightATy weightA;

    auto getWeightA() const {
      auto &propStorage = this->weightA;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setWeightA(const ::mlir::IntegerAttr &propValue) {
      this->weightA = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.aScaleFactor == this->aScaleFactor &&
        rhs.aTranspose == this->aTranspose &&
        rhs.accScale == this->accScale &&
        rhs.act == this->act &&
        rhs.bScaleFactor == this->bScaleFactor &&
        rhs.bTranspose == this->bTranspose &&
        rhs.bertScale == this->bertScale &&
        rhs.dScaleFactor == this->dScaleFactor &&
        rhs.dataflow == this->dataflow &&
        rhs.fullC == this->fullC &&
        rhs.lowD == this->lowD &&
        rhs.repeatingBias == this->repeatingBias &&
        rhs.weightA == this->weightA &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  TileMatMulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("gemmini.tile_matmul", odsAttrs.getContext());
  }

  TileMatMulOpGenericAdaptorBase(TileMatMulOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::FloatAttr getAScaleFactorAttr();
  ::llvm::APFloat getAScaleFactor();
  ::mlir::FloatAttr getBScaleFactorAttr();
  ::llvm::APFloat getBScaleFactor();
  ::mlir::FloatAttr getDScaleFactorAttr();
  ::llvm::APFloat getDScaleFactor();
  ::mlir::IntegerAttr getActAttr();
  uint64_t getAct();
  ::mlir::FloatAttr getAccScaleAttr();
  ::llvm::APFloat getAccScale();
  ::mlir::FloatAttr getBertScaleAttr();
  ::llvm::APFloat getBertScale();
  ::mlir::BoolAttr getRepeatingBiasAttr();
  bool getRepeatingBias();
  ::mlir::BoolAttr getATransposeAttr();
  bool getATranspose();
  ::mlir::BoolAttr getBTransposeAttr();
  bool getBTranspose();
  ::mlir::BoolAttr getFullCAttr();
  bool getFullC();
  ::mlir::BoolAttr getLowDAttr();
  bool getLowD();
  ::mlir::IntegerAttr getWeightAAttr();
  uint64_t getWeightA();
  ::mlir::IntegerAttr getDataflowAttr();
  uint64_t getDataflow();
};

} // namespace detail
template <typename RangeT>
class TileMatMulOpGenericAdaptor : public detail::TileMatMulOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TileMatMulOpGenericAdaptorBase;
  static constexpr int odsIndex_aArray = 0;
  static constexpr int odsIndex_bArray = 1;
  static constexpr int odsIndex_cArray = 2;
  static constexpr int odsIndex_dArray = 3;
public:
  TileMatMulOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TileMatMulOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TileMatMulOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  TileMatMulOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : TileMatMulOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  TileMatMulOpGenericAdaptor(RangeT values, const TileMatMulOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TileMatMulOp, typename = std::enable_if_t<std::is_same_v<LateInst, TileMatMulOp>>>
  TileMatMulOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAArray() {
    return (*getODSOperands(0).begin());
  }

  ValueT getBArray() {
    return (*getODSOperands(1).begin());
  }

  ValueT getCArray() {
    return (*getODSOperands(2).begin());
  }

  ValueT getDArray() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TileMatMulOpAdaptor : public TileMatMulOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TileMatMulOpGenericAdaptor::TileMatMulOpGenericAdaptor;
  TileMatMulOpAdaptor(TileMatMulOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TileMatMulOp : public ::mlir::Op<TileMatMulOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TileMatMulOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TileMatMulOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_aArray = 0;
  static constexpr int odsIndex_bArray = 1;
  static constexpr int odsIndex_cArray = 2;
  static constexpr int odsIndex_dArray = 3;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("aScaleFactor"), ::llvm::StringRef("aTranspose"), ::llvm::StringRef("accScale"), ::llvm::StringRef("act"), ::llvm::StringRef("bScaleFactor"), ::llvm::StringRef("bTranspose"), ::llvm::StringRef("bertScale"), ::llvm::StringRef("dScaleFactor"), ::llvm::StringRef("dataflow"), ::llvm::StringRef("fullC"), ::llvm::StringRef("lowD"), ::llvm::StringRef("repeatingBias"), ::llvm::StringRef("weightA")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAScaleFactorAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAScaleFactorAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getATransposeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getATransposeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getAccScaleAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getAccScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getActAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getActAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getBScaleFactorAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getBScaleFactorAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getBTransposeAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getBTransposeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getBertScaleAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getBertScaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getDScaleFactorAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getDScaleFactorAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getDataflowAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getDataflowAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getFullCAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getFullCAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getLowDAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getLowDAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  ::mlir::StringAttr getRepeatingBiasAttrName() {
    return getAttributeNameForIndex(11);
  }

  static ::mlir::StringAttr getRepeatingBiasAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 11);
  }

  ::mlir::StringAttr getWeightAAttrName() {
    return getAttributeNameForIndex(12);
  }

  static ::mlir::StringAttr getWeightAAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 12);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("gemmini.tile_matmul");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getAArray() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::MemRefType> getBArray() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::MemRefType> getCArray() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::MemRefType> getDArray() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(3).begin());
  }

  ::mlir::OpOperand &getAArrayMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getBArrayMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCArrayMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDArrayMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::FloatAttr getAScaleFactorAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().aScaleFactor);
  }

  ::llvm::APFloat getAScaleFactor();
  ::mlir::FloatAttr getBScaleFactorAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().bScaleFactor);
  }

  ::llvm::APFloat getBScaleFactor();
  ::mlir::FloatAttr getDScaleFactorAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().dScaleFactor);
  }

  ::llvm::APFloat getDScaleFactor();
  ::mlir::IntegerAttr getActAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().act);
  }

  uint64_t getAct();
  ::mlir::FloatAttr getAccScaleAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().accScale);
  }

  ::llvm::APFloat getAccScale();
  ::mlir::FloatAttr getBertScaleAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::FloatAttr>(getProperties().bertScale);
  }

  ::llvm::APFloat getBertScale();
  ::mlir::BoolAttr getRepeatingBiasAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().repeatingBias);
  }

  bool getRepeatingBias();
  ::mlir::BoolAttr getATransposeAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().aTranspose);
  }

  bool getATranspose();
  ::mlir::BoolAttr getBTransposeAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().bTranspose);
  }

  bool getBTranspose();
  ::mlir::BoolAttr getFullCAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().fullC);
  }

  bool getFullC();
  ::mlir::BoolAttr getLowDAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().lowD);
  }

  bool getLowD();
  ::mlir::IntegerAttr getWeightAAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().weightA);
  }

  uint64_t getWeightA();
  ::mlir::IntegerAttr getDataflowAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().dataflow);
  }

  uint64_t getDataflow();
  void setAScaleFactorAttr(::mlir::FloatAttr attr) {
    getProperties().aScaleFactor = attr;
  }

  void setAScaleFactor(::llvm::APFloat attrValue);
  void setBScaleFactorAttr(::mlir::FloatAttr attr) {
    getProperties().bScaleFactor = attr;
  }

  void setBScaleFactor(::llvm::APFloat attrValue);
  void setDScaleFactorAttr(::mlir::FloatAttr attr) {
    getProperties().dScaleFactor = attr;
  }

  void setDScaleFactor(::llvm::APFloat attrValue);
  void setActAttr(::mlir::IntegerAttr attr) {
    getProperties().act = attr;
  }

  void setAct(uint64_t attrValue);
  void setAccScaleAttr(::mlir::FloatAttr attr) {
    getProperties().accScale = attr;
  }

  void setAccScale(::llvm::APFloat attrValue);
  void setBertScaleAttr(::mlir::FloatAttr attr) {
    getProperties().bertScale = attr;
  }

  void setBertScale(::llvm::APFloat attrValue);
  void setRepeatingBiasAttr(::mlir::BoolAttr attr) {
    getProperties().repeatingBias = attr;
  }

  void setRepeatingBias(bool attrValue);
  void setATransposeAttr(::mlir::BoolAttr attr) {
    getProperties().aTranspose = attr;
  }

  void setATranspose(bool attrValue);
  void setBTransposeAttr(::mlir::BoolAttr attr) {
    getProperties().bTranspose = attr;
  }

  void setBTranspose(bool attrValue);
  void setFullCAttr(::mlir::BoolAttr attr) {
    getProperties().fullC = attr;
  }

  void setFullC(bool attrValue);
  void setLowDAttr(::mlir::BoolAttr attr) {
    getProperties().lowD = attr;
  }

  void setLowD(bool attrValue);
  void setWeightAAttr(::mlir::IntegerAttr attr) {
    getProperties().weightA = attr;
  }

  void setWeightA(uint64_t attrValue);
  void setDataflowAttr(::mlir::IntegerAttr attr) {
    getProperties().dataflow = attr;
  }

  void setDataflow(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::mlir::FloatAttr aScaleFactor, ::mlir::FloatAttr bScaleFactor, ::mlir::FloatAttr dScaleFactor, ::mlir::IntegerAttr act, ::mlir::FloatAttr accScale, ::mlir::FloatAttr bertScale, ::mlir::BoolAttr repeatingBias, ::mlir::BoolAttr aTranspose = nullptr, ::mlir::BoolAttr bTranspose = nullptr, ::mlir::BoolAttr fullC = nullptr, ::mlir::BoolAttr lowD = nullptr, ::mlir::IntegerAttr weightA = nullptr, ::mlir::IntegerAttr dataflow = nullptr);
  static TileMatMulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::mlir::FloatAttr aScaleFactor, ::mlir::FloatAttr bScaleFactor, ::mlir::FloatAttr dScaleFactor, ::mlir::IntegerAttr act, ::mlir::FloatAttr accScale, ::mlir::FloatAttr bertScale, ::mlir::BoolAttr repeatingBias, ::mlir::BoolAttr aTranspose = nullptr, ::mlir::BoolAttr bTranspose = nullptr, ::mlir::BoolAttr fullC = nullptr, ::mlir::BoolAttr lowD = nullptr, ::mlir::IntegerAttr weightA = nullptr, ::mlir::IntegerAttr dataflow = nullptr);
  static TileMatMulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::mlir::FloatAttr aScaleFactor, ::mlir::FloatAttr bScaleFactor, ::mlir::FloatAttr dScaleFactor, ::mlir::IntegerAttr act, ::mlir::FloatAttr accScale, ::mlir::FloatAttr bertScale, ::mlir::BoolAttr repeatingBias, ::mlir::BoolAttr aTranspose = nullptr, ::mlir::BoolAttr bTranspose = nullptr, ::mlir::BoolAttr fullC = nullptr, ::mlir::BoolAttr lowD = nullptr, ::mlir::IntegerAttr weightA = nullptr, ::mlir::IntegerAttr dataflow = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::mlir::FloatAttr aScaleFactor, ::mlir::FloatAttr bScaleFactor, ::mlir::FloatAttr dScaleFactor, ::mlir::IntegerAttr act, ::mlir::FloatAttr accScale, ::mlir::FloatAttr bertScale, ::mlir::BoolAttr repeatingBias, ::mlir::BoolAttr aTranspose = nullptr, ::mlir::BoolAttr bTranspose = nullptr, ::mlir::BoolAttr fullC = nullptr, ::mlir::BoolAttr lowD = nullptr, ::mlir::IntegerAttr weightA = nullptr, ::mlir::IntegerAttr dataflow = nullptr);
  static TileMatMulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::mlir::FloatAttr aScaleFactor, ::mlir::FloatAttr bScaleFactor, ::mlir::FloatAttr dScaleFactor, ::mlir::IntegerAttr act, ::mlir::FloatAttr accScale, ::mlir::FloatAttr bertScale, ::mlir::BoolAttr repeatingBias, ::mlir::BoolAttr aTranspose = nullptr, ::mlir::BoolAttr bTranspose = nullptr, ::mlir::BoolAttr fullC = nullptr, ::mlir::BoolAttr lowD = nullptr, ::mlir::IntegerAttr weightA = nullptr, ::mlir::IntegerAttr dataflow = nullptr);
  static TileMatMulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::mlir::FloatAttr aScaleFactor, ::mlir::FloatAttr bScaleFactor, ::mlir::FloatAttr dScaleFactor, ::mlir::IntegerAttr act, ::mlir::FloatAttr accScale, ::mlir::FloatAttr bertScale, ::mlir::BoolAttr repeatingBias, ::mlir::BoolAttr aTranspose = nullptr, ::mlir::BoolAttr bTranspose = nullptr, ::mlir::BoolAttr fullC = nullptr, ::mlir::BoolAttr lowD = nullptr, ::mlir::IntegerAttr weightA = nullptr, ::mlir::IntegerAttr dataflow = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::llvm::APFloat aScaleFactor, ::llvm::APFloat bScaleFactor, ::llvm::APFloat dScaleFactor, uint64_t act, ::llvm::APFloat accScale, ::llvm::APFloat bertScale, bool repeatingBias = false, bool aTranspose = false, bool bTranspose = false, bool fullC = false, bool lowD = false, uint64_t weightA = 0, uint64_t dataflow = 1);
  static TileMatMulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::llvm::APFloat aScaleFactor, ::llvm::APFloat bScaleFactor, ::llvm::APFloat dScaleFactor, uint64_t act, ::llvm::APFloat accScale, ::llvm::APFloat bertScale, bool repeatingBias = false, bool aTranspose = false, bool bTranspose = false, bool fullC = false, bool lowD = false, uint64_t weightA = 0, uint64_t dataflow = 1);
  static TileMatMulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::llvm::APFloat aScaleFactor, ::llvm::APFloat bScaleFactor, ::llvm::APFloat dScaleFactor, uint64_t act, ::llvm::APFloat accScale, ::llvm::APFloat bertScale, bool repeatingBias = false, bool aTranspose = false, bool bTranspose = false, bool fullC = false, bool lowD = false, uint64_t weightA = 0, uint64_t dataflow = 1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::llvm::APFloat aScaleFactor, ::llvm::APFloat bScaleFactor, ::llvm::APFloat dScaleFactor, uint64_t act, ::llvm::APFloat accScale, ::llvm::APFloat bertScale, bool repeatingBias = false, bool aTranspose = false, bool bTranspose = false, bool fullC = false, bool lowD = false, uint64_t weightA = 0, uint64_t dataflow = 1);
  static TileMatMulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::llvm::APFloat aScaleFactor, ::llvm::APFloat bScaleFactor, ::llvm::APFloat dScaleFactor, uint64_t act, ::llvm::APFloat accScale, ::llvm::APFloat bertScale, bool repeatingBias = false, bool aTranspose = false, bool bTranspose = false, bool fullC = false, bool lowD = false, uint64_t weightA = 0, uint64_t dataflow = 1);
  static TileMatMulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value aArray, ::mlir::Value bArray, ::mlir::Value cArray, ::mlir::Value dArray, ::llvm::APFloat aScaleFactor, ::llvm::APFloat bScaleFactor, ::llvm::APFloat dScaleFactor, uint64_t act, ::llvm::APFloat accScale, ::llvm::APFloat bertScale, bool repeatingBias = false, bool aTranspose = false, bool bTranspose = false, bool fullC = false, bool lowD = false, uint64_t weightA = 0, uint64_t dataflow = 1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TileMatMulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TileMatMulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TileMatMulOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TileMatMulOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void populateDefaultProperties(::mlir::OperationName opName, Properties &properties);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 13 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace merlin::gemmini
MLIR_DECLARE_EXPLICIT_TYPE_ID(::merlin::gemmini::TileMatMulOp)


#endif // GET_OP_CLASSES

