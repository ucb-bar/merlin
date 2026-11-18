# Capture preparation

These modules prepare selected source programs, run the external model2MLIR
conversion API, and record exact artifacts and framework observations. Actual
framework conversion remains in model2MLIR; do not copy its converters here.
Merlin numerical intent and admission remain compiler quantization policy.

Workers may execute by direct file path in an isolated framework interpreter.
Resolve their own selected Merlin source root and retain the exact portable
TorchAO source roster. They must not require an independently installed Merlin
or xDSL merely to load the recipe entry or state observer.

Preserve existing recipe/calibration arithmetic, input/output types, all-output
references, provenance and source identity checks. Structural relocation changes
source hashes and requires fresh qualification; old receipts are not rewritten.
