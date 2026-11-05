# Qwen kernel campaign

See [local integration](docs/local-integration.md) for the current repository split, offline random full-model launch bench, shared kernel APIs, and qualification limits.

For the actual pretrained-model export, graph-derived PyTorch regions and model-shape library benchmark, use [full_model/README.md](full_model/README.md). The workflow below is the earlier manually authored microbenchmark campaign. Imported numerical reports are historical evidence from Jeremy Mills' source commits; this path-rewritten replay does not reproduce their measurements.

These are ordinary PyTorch subgraphs of `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, the Qwen2 model in [Radiance PR #1](https://github.com/ucb-bar/radiance-kernels/pull/1). See the [results and recommendations](docs/qwen-results.md) for measured cycles, numerical error, and the exact boundaries of each workload. The follow-up [roofline analysis](docs/qwen-roofline.md) includes per-resource lower bounds, default/faster-memory comparisons, raw reports and reproduction.

Use the compiler wheel or a separate compiler checkout selected by `--atlas-mlir-root`, `ATLAS_MLIR_ROOT`, or the sibling `atlas-mlir` directory. `--atlas-bin` or `ATLAS_OOT_BIN_DIR` selects built `atlas-opt` and `atlas-emit` tools. Run commands from the Merlin repository:

```sh
python3 examples/atlas/model_bringup/qwen/run.py --setup
python3 examples/atlas/model_bringup/qwen/run.py
```

Setup downloads pinned Merlin and NPU-model sources and creates three isolated Python environments under `out/qwen-deps`: capture, Merlin selection, and numerical simulation. It does not modify sibling checkouts. Cargo's locked bridge dependencies are fetched during setup; snapshot construction is offline. The second command creates a fresh `atlas-out/qwen` campaign. Use `-o DIR` for subsequent runs. Existing results are never overwritten. Use `--panel signed -o atlas-out/qwen-signed` to repeat with bounded signed inputs. Both panels use magnitudes in [0.03125,0.15625), within the selected FP8 input domain; they do not cover arbitrary pretrained activations.

For already prepared environments, override `--capture-python`, `--merlin-python`, `--model-python`, `--merlin-root`, and `--model-root`. The current Merlin checkout is the default frontend source. Explicit local source roots are accepted, and `provenance.json` records observed revisions and compiler hashes separately from the historical pins. The setup action remains an optional network-dependent convenience; prepared local roots do not require it.

The campaign captures all workloads, then compiles supported graphs through Merlin and Atlas and executes their emitted instructions on the NPU model. Full-width source graphs that are unsupported, or whose search times out, are recorded in `summary.json`; they do not get fabricated performance numbers. Unexpected failures in the supported attention or SwiGLU graph stop the run. The K=1536 projection has a 60-second search budget in the repeatable campaign; the report also records an earlier failed 300-second experiment.

Each compiled workload retains:

- The original and normalized Linalg, argument manifest, tensor inputs and f32 reference outputs.
- Precision policy, source hash, Merlin request and ABI, verified native artifacts.
- Original unscheduled, boundary-DMA-only, and scheduled machine programs.
- Re-encoded instruction checks, model traces, bit-exact output/DRAM guard checks,
  engine busy cycles and error against PyTorch in `performance.json`.

The input values and weights are seeded synthetic tensors. The full Qwen architecture uses hidden width 1536, FFN width 8960, and 128-channel heads. The attention experiment uses all 128 head channels in a 32-token unmasked cache window; the SwiGLU experiment is an interior 32-channel partition. The K=128, 256 and 512 projection experiments are reduction partitions, not complete 1536-wide projections. No token throughput, pretrained accuracy, or full-model execution is claimed.

`kernels.py` also contains full-width RMSNorm/QKV, residual MLP, RoPE and GQA PyTorch definitions for extending the campaign. The selected Atlas numerical policy remains diagnostic; in particular its reciprocal domain is [1,32].

## Integration provenance

This replay derives from `a92d1732dd436e2b99637046eaa2a151f3407de0`, `63759308bdc33587679ff1349d00cd3fa4a616a6`, and `5884524e3cf85027e08734526c1e3dbf0a093686` by Jeremy Mills. Model workflows, reports, and tests moved into this directory. Generic capture, normalization, the `atlasc` entrypoint, precision lowering, compiler support files, and compiler tests remain in atlas-mlir.

The current kernel library provides D128 attention with GQA and streamed RMSNorm at width 1536. `full_model/mapping.yaml` requests epsilon `1.0e-6`. Qualification must still check Qwen's causal masks, 12 query heads and 2 KV heads from export, cached-decode lengths, RoPE, QKV biases, and precision boundaries. Historical reports retain their original no-variant observations and measurements.

Run the relocated semantic and extraction regressions with `python -m unittest discover -s examples/atlas/model_bringup/qwen/tests -v`. PyTorch semantics run independently of the optional torch-mlir capture extra. Missing torch-mlir produces explicit skipped capture tests, leaving fresh Linalg capture PENDING.
