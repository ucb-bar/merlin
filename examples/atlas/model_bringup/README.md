# Model bringup

The TinyLlama, SmolVLA, and ResNet runtime, builders, tests, and runbooks are copied here from atlas-mlir. Capture providers remain in [comparisons](../comparisons/README.md), compiler and generic host ABI sources remain in atlas-mlir, and kernel implementations remain in atlas-kernels at the selection pin.

The [bringup guide](docs/bringing-up-a-model.md), [Tiny reproduction notes](tools/tiny_full/REPRODUCING.md), and [required source joins](test/FAITHFULNESS.md) use historical source-relative model paths through a verified external execution view. The view copies each file from its explicit canonical owner and checks canonical and staged hashes. It must be outside all three repositories.

## Build

Build `atlas-opt` and `atlas-emit` in the canonical compiler checkout following the [compiler Build section](https://github.com/ucb-bar/atlas-mlir/blob/787ad51/README.md#build). Compiler hardware guidance remains in [compiler hardware status](https://github.com/ucb-bar/atlas-mlir/blob/787ad51/docs/hardware-status.md). These compiler documents and binaries are not supplied by the execution view.

From this directory, set physical absolute paths to the canonical repositories, the original47 model checkout, the interpreter with NumPy and PyYAML, and fresh external outputs:

```sh
export ATLAS_ROOT=/absolute/path/to/atlas-mlir
export MERLIN_ROOT=/absolute/path/to/Merlin
export KERNELS_ROOT=/absolute/path/to/atlas-kernels
export MODEL_ROOT=/absolute/path/to/original47
export PYTHON=/absolute/path/to/model-venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
export VIEW=/absolute/path/out/model-bringup-view
export RESULTS_ROOT=/absolute/path/out/model-bringup-results
export MODEL_OUTPUT_ROOT=/absolute/path/out/model-bringup-outputs
mkdir -p "$MODEL_OUTPUT_ROOT"
export ATLAS_MLIR_BIN="$ATLAS_ROOT/build/bin"
make view MODE=relocated PYTHON="$PYTHON" ATLAS_ROOT="$ATLAS_ROOT" MERLIN_ROOT="$MERLIN_ROOT" KERNELS_ROOT="$KERNELS_ROOT" VIEW="$VIEW"
make check MODE=relocated PYTHON="$PYTHON" ATLAS_ROOT="$ATLAS_ROOT" MERLIN_ROOT="$MERLIN_ROOT" KERNELS_ROOT="$KERNELS_ROOT" MODEL_ROOT="$MODEL_ROOT" VIEW="$VIEW" RESULTS_ROOT="$RESULTS_ROOT"
```

`check` runs the complete model suite and the required source joins separately, retains both receipts even on failure, and fails on skipped required checks. Before running it, supply the authenticated `SMOL_REPLAY`, `SMOL_KERNEL_ASSETS`, and `TINY_FULL_CAPTURE` inputs described in [required source joins](test/FAITHFULNESS.md). Those three variables do not replace the separate fixture inputs required by individual test modules.

Keep `MODEL_OUTPUT_ROOT` physical and outside the view and all three canonical repositories. For historical model-relative commands, export `ATLAS="$VIEW"`, `ATLAS_SOURCE_ROOT="$VIEW"`, and `ATLAS_MLIR_ROOT="$VIEW"`, and use `cd "$VIEW"`; the bringup guide names this execution root `ATLAS`. Keep `ATLAS_ROOT` pointing to the canonical compiler checkout and `ATLAS_MLIR_BIN` pointing to its built binaries. Capture commands explicitly run from Merlin or atlas-kernels where the guide says so.

[source-map.json](source-map.json) records explicit ownership and dependency paths. The adjacent external view inventory records canonical and staged SHA-256 values. [test-manifest.json](test-manifest.json) drives execution of the moved suites and the two extracted host-bit cases. Split qualification and refusal receipts remain under the implementation workspace's `out/`.

## Qualification status

The pure additive move acceptance criterion is separate move-equivalence evidence: identical membership and leaf outcomes, every baseline PASS preserved, and no new failure, skip, error, or timeout, alongside source identity, ELF identity, and census equality. This criterion applies to the pure move only. The strict raw suite result remains FAIL; equality of an uncompleted timeout does not establish passing tests. Required source joins remain required, with the retained receipt reporting 43/43 and zero skips. A compiler cut needs fresh compiler, build, identity, census, and move-equivalence gates.

The retained baseline and additive receipts agree on 67 modules, 479 discovered leaf IDs, 63 successful modules, and 472 reported executions. The following four modules retain unavailable coverage; none is silently whitelisted, and strict integration still requires their inputs and completed results.

| Module | Coverage status and exact pre-existing reason |
| --- | --- |
| `test_host_iss_gdb_buffered` | PENDING: sandbox socket operation denied; the module timed out, and its seven discovered cases have no completed per-case receipts. |
| `test_host_iss_smol` | PENDING: three `AdmissionTests` cases lack the authenticated seven-launch `SMOL_BLOCK_BUNDLE`; attempted fixture skips remain raw failures. |
| `test_tiny_attention_export` | PENDING: `AttentionExportTest.test_eight_launch_cli_original_numerics_remain_diagnostic` lacks the matching eight-launch diagnostic capture and model fixtures; its attempted skip remains a raw failure. A full/native capture cannot substitute for this fixture. |
| `test_tiny_full_export` | PENDING qualification: `FullExportTest.test_vector_codec_selected_rtl_boundaries_and_actual_weight` retains the raw error/FAIL from the separate `47b8599` versus `9e6313` model-pin refusal. This is not a missing-fixture failure. |

The moved runbooks have deliberate path and ownership edits required by the reviews: canonical compiler documentation links, execution-view setup, Merlin source-join and model commands, and external generated-output paths. Their original bytes and the exception hashes are retained in the implementation workspace's `out/docs-original/` and `out/review2-docs/MOVE-AUDIT.json`; these documentation exceptions are not byte-identical copies. Pinned producers, builders, and the original identity manifest remain unchanged.

## Model integration notes moved from compiler documentation

For the supported complete-graph build route, see [Complete ResNet50 host graph](docs/resnet-full-host.md). The graph builder enables `mstatus.FS`, initializes `fcsr` to nearest-even, reserves a 64 KiB stack and checks ELF assets and load segments. A linked ELF does not establish host execution, DMA reachability or cache coherence on a selected SoC.

`tools/attention_timing_contract.py` validates attention metadata geometry, descriptor mask extents, count consistency, padding, and any supplied contract against the staged bytes. Tiny's v2 exporter and the Smol host builder invoke this check. Existing hosts copy those immutable templates and patch only operand spans. This validates generator inputs; the caller still owes the invocation-wide immutability assumption.

Contracts cannot suppress another timing rule. The 45 remaining frozen attention programs all end with `dma.wait ch1; delay 128; ecall`. Supplying their exact descriptor/count images resolves their DMA ownership and capacity rejections, but all 45 then reject DELAY-before-halt. The explicit `atlas-kernels` halt variant at `ed70fb5` removes the dead terminal delay without changing the default generator. All 45 regenerated halt programs pass as `contract-verified` with these exact metadata contracts; their words differ from the frozen kit. Without contracts, their metadata loads remain unconstrained. The frozen streams remain `stream-verified`.
