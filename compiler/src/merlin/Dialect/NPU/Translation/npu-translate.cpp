#include "compiler/src/merlin/Dialect/NPU/Translation/TranslateToTextISA.h"

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include <cstdlib>

int main(int argc, char **argv) {
	mlir::iree_compiler::NPU::registerToTextISATranslation();

	return mlir::failed(
			   mlir::mlirTranslateMain(argc, argv, "NPU translation driver\n"))
		? EXIT_FAILURE
		: EXIT_SUCCESS;
}
