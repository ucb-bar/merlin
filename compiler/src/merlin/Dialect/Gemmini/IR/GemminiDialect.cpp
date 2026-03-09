#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiDialect.h"

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiAttrs.h"
#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::iree_compiler::Gemmini;

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiDialect.cpp.inc"

void GemminiDialect::initialize() {
	addAttributes<
#define GET_ATTRDEF_LIST
#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiAttrs.cpp.inc"
		>();

	addOperations<
#define GET_OP_LIST
#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.cpp.inc"
		>();
}

Attribute GemminiDialect::parseAttribute(
	DialectAsmParser &parser, Type type) const {
	StringRef attrTag;
	if (failed(parser.parseKeyword(&attrTag))) {
		return {};
	}

	if (attrTag == "dataflow") {
		if (failed(parser.parseLess()))
			return {};

		StringRef valueTag;
		if (failed(parser.parseKeyword(&valueTag)))
			return {};

		if (failed(parser.parseGreater()))
			return {};

		auto maybeValue = symbolizeDataflow(valueTag);
		if (!maybeValue) {
			parser.emitError(parser.getNameLoc())
				<< "expected one of [os, ws] for Gemmini dataflow mode";
			return {};
		}

		return DataflowAttr::get(getContext(), *maybeValue);
	}

	parser.emitError(parser.getNameLoc()) << "unknown gemmini attribute";
	return {};
}

void GemminiDialect::printAttribute(
	Attribute attr, DialectAsmPrinter &os) const {
	if (auto dataflow = dyn_cast<DataflowAttr>(attr)) {
		os << "dataflow<" << stringifyDataflow(dataflow.getValue()) << ">";
		return;
	}
	llvm_unreachable("unknown gemmini attribute");
}
