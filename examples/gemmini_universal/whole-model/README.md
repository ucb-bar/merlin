# Whole-model lowering inspection

The generic [lowering inspection guide](../../../docs/guides/model_lowering.md)
shows how to record intermediate MLIR and separate tensor sidecars from text.
For a real capture, run:

```sh
merlin lower /absolute/capture/model.mlir \
  --out /configured/out/build/model-lowering/universal-001 \
  --ir-audit both
```

Pass only sidecars that actually exist. The capture itself is produced by
[model2MLIR](../../../docs/guides/model2mlir.md), not by a duplicate example
frontend. Inspection is not an executable Universal compiler package.

The [published Gemmini compiler smoke](../../gemmini/README.md#use-the-published-compiler-without-merlin)
targets a different configuration. It must not be presented as Universal
whole-model correctness or performance. Qualify a separate Universal compiler,
RTL tools, simulator and hardware path before making that claim.
