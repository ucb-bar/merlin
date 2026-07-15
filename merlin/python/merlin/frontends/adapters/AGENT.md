# frontends/adapters

One module per ingestible model source, each conforming to the `FrontendAdapter` protocol in
`merlin.frontends.registry` (module-level `NAME`, `can_handle(source)`, `ingest(source, *, model,
variant, **kw) -> CaptureBundle`). A new frontend framework is added as one file here plus one entry
in the registry — nothing else in the pipeline changes.

- `m2m.py` — the reference/catch-all adapter: PyTorch / torchAO models captured by model2MLIR
  (including HuggingFace checkpoints via its per-model loaders).
- `gguf.py` — lifts a `.gguf` checkpoint into the `quant_ext` dialect (the reader lands with the P1
  vertical slice); GGML types map onto `merlin.common.quant_formats` via `QuantFormat.ggml_type`.

Adapters produce a framework-neutral `CaptureBundle`; the format each weight carries is described by
the target-agnostic quant-format registry, never by the adapter — so adding a quantization format
needs no new adapter.
