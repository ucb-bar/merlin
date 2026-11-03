# Optional attention runtime provider

The policy and vision graphs compile and run CPU attention by default. Their ordinary fixture source lists need no atlas-mlir checkout, registry header, or registry implementation.

To inject an attention backend, define `SMOL_ATTENTION_RUNTIME=1` when compiling both graph sources, add the atlas-mlir `runtime` include directory, and link that checkout's `runtime/smol_attention.c`. The sole authoritative callback ABI is `atlas-mlir/runtime/smol_attention.h`; Merlin does not duplicate it. Configure the registry with `smol_attention_configure(callback, context)` before graph execution. A null callback, including the initial state and `smol_attention_configure(NULL, NULL)`, selects the same CPU path. A nonzero callback result fails the graph through its ordinary error path. The registry is process-global and intended for sequential graph execution.

The atlas-mlir full-host builder supplies the opt-in macro and runtime dependency. Other launch/build adapters must supply all three together: macro, include path, and source/object. Defining the macro without linking the provider is a link error.

Run the dependency-free graph build and CPU control, then the explicit runtime control, from a Merlin checkout:

```sh
python -m pytest merlin/tests/infra/test_smol_attention_registry.py -v
ATLAS_MLIR_ROOT=/absolute/path/to/atlas-mlir python -m pytest merlin/tests/infra/test_smol_attention_registry.py -v
```

These controls compile with strict warnings and reject unresolved shared-library symbols. They exercise CPU attention for all three policy kinds and a vision layer, successful registered callbacks for all four kinds, callback failure, and callback reset. They use finite zero tensors to isolate routing; the configured policy-block, policy-graph, and vision replay fixtures remain the checks against actual captured arithmetic and model weights.
