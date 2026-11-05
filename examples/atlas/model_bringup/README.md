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

The retained baseline and additive receipts agree on 67 modules, 479 discovered leaf IDs, 63 successful modules, and 472 reported executions. Fixture recovery subsequently completed the three modules below: 31 cases pass with zero skips through a fresh verified view. The complete 67-module rerun remains PENDING; these targeted results do not replace the historical raw suite receipt.

| Recovered module | Completed coverage |
| --- | --- |
| `test_host_iss_smol` | PASS: 15 cases, including all three real seven-launch bundle admission and mutation cases. The metadata adapter preserves the original contract, build receipt, ELF, and payload bytes. |
| `test_tiny_attention_export` | PASS: seven cases, including all eight original47 launches, exact full arenas, RTL numerics, and at least 45 model-source pins per launch. This is the original `SYNTHETIC_EXPORT_CLI_ONLY_NOT_A_MODEL_CAPTURE` diagnostic; it still fails complete-workload admission as required. |
| `test_tiny_full_export` | PASS: nine cases with the separately selected `9e6313` model, the capture-bound codec, and the actual captured q_proj weight. The retained native capture still requires original47 backfill for complete numerical admission. |

The following module retains unavailable sandbox coverage and remains registered.

| Module | Coverage status and exact pre-existing reason |
| --- | --- |
| `test_host_iss_gdb_buffered` | PENDING: sandbox socket operation denied; the module timed out, and its seven discovered cases have no completed per-case receipts. |

### Recovered fixture inputs

[fixture-pins.json](fixture-pins.json) records the original receipt, contract, ELF, payload, codec, and model identities. Fixture availability is an external prerequisite: the stacked PRs alone do not supply these exact bytes, and no durable download location is recorded here. Before running the recovered modules, request the exact hash-pinned minimal closures from the artifact custodian, supplying this pin file: Smol needs all 310 files listed in its original `SHA256SUMS`, that unchanged manifest, and the historical trust receipt; the attention diagnostic needs all 42 entries under `tiny_attention.files`; the Tiny codec test needs the original 465-call capture receipt, its `parameters/000-B.bin` weight, and the archived codec. The model checkouts at the recorded revisions are separate prerequisites. If the custodian cannot supply matching bytes, fixture-dependent reproduction remains PENDING. Keep all fixture paths physical, without symlink components, and compare their SHA-256 values with the pin file before use.

The Smol adapter verifies the complete source archive before selecting its 39 inherited files, so supplying only those 39 files with the original 310-entry manifest is insufficient.

Smol's recorded overlay metadata hashes describe the recovery placement. A different physical archive root changes `storage.json` and `SHA256SUMS`; the original contract and inherited payload hashes remain binding.

Setting both `SMOL_BLOCK_ARCHIVE` and `SMOL_BLOCK_TRUST` also enables a relocated-root adapter regression within the Smol admission case; bundle admission without those variables remains supported.

The original Smol archive predates the reader's `atlas.block_bundle.v1` metadata. After building the view, the adapter derives that metadata from the pinned historical contract and trust receipt, checks every inherited byte, and invokes the unchanged bundle and ELF admission checks. It writes an overlay rather than copying the 34,487,482-byte fixture closure:

```sh
export SMOL_BLOCK_ARCHIVE=/absolute/path/to/selected-layer1-block-bundle-v1
export SMOL_BLOCK_TRUST=/absolute/path/to/historical-smol-inputs/manifest.json
export SMOL_BLOCK_BUNDLE="$MODEL_OUTPUT_ROOT/smol-block-bundle"
"$PYTHON" "$VIEW/test/prepare_smol_fixture.py" \
  --source "$SMOL_BLOCK_ARCHIVE" --trust-receipt "$SMOL_BLOCK_TRUST" \
  --view "$VIEW" --out "$SMOL_BLOCK_BUNDLE" \
  --report "$MODEL_OUTPUT_ROOT/smol-fixture.json"
```

The recovered diagnostic needs its 35 payload files, unchanged `receipt.json`, and six exact source files under `source-root/`, retaining their paths relative to the recorded source prefix. `TINY_ATTENTION_RELOCATE` passes the exporter's existing relocation option; every relocated source must still match the original receipt hash. Tiny's codec test separately accepts `TINY_FULL_CODEC` for the archived physical source file and verifies its captured SHA-256 before import. The current Merlin codec is a different revision and cannot substitute.

```sh
export TINY_ATTENTION_FIXTURE=/absolute/path/to/recovered-eight-launch-diagnostic
export TINY_ATTENTION_CAPTURE="$TINY_ATTENTION_FIXTURE/receipt.json"
export TINY_ATTENTION_RELOCATE="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["tiny_attention"]["capture_source_prefix"])' "$MERLIN_ROOT/examples/atlas/model_bringup/fixture-pins.json")=$TINY_ATTENTION_FIXTURE/source-root"
export TINY_FULL_CAPTURE=/absolute/path/to/original-465-call-capture/receipt.json
export TINY_FULL_CODEC=/absolute/path/to/captured/tiny_fp8_pack.py
export TINY_CODEC_MODEL_ROOT=/absolute/path/to/npu-model-at-9e6313
for module in test_host_iss_smol test_tiny_attention_export; do
  "$PYTHON" "$MERLIN_ROOT/examples/atlas/model_bringup/run_tests.py" tests \
    --_test "test/$module.py" --view "$VIEW" --atlas "$ATLAS_ROOT" \
    --merlin "$MERLIN_ROOT" --kernels "$KERNELS_ROOT" --model-root "$MODEL_ROOT" \
    --out "$RESULTS_ROOT/$module.json"
done
"$PYTHON" "$MERLIN_ROOT/examples/atlas/model_bringup/run_tests.py" tests \
  --_test test/test_tiny_full_export.py --view "$VIEW" --atlas "$ATLAS_ROOT" \
  --merlin "$MERLIN_ROOT" --kernels "$KERNELS_ROOT" --model-root "$TINY_CODEC_MODEL_ROOT" \
  --out "$RESULTS_ROOT/test_tiny_full_export.json"
```

Use `MODEL_ROOT` at `47b8599` for the first two commands and `TINY_CODEC_MODEL_ROOT` at `9e6313bddecb8ba936206eb347f5cd4e5335282c` for the codec module. Run these modules separately because the producers retain distinct model pins. The default single-model suite invocation cannot satisfy both pins. These fixtures establish only the coverage in the table; full-model numerical execution, the sandbox socket module, QEMU, RTL, and FireSim remain separate gates.

The moved runbooks have deliberate path and ownership edits required by the reviews: canonical compiler documentation links, execution-view setup, Merlin source-join and model commands, and external generated-output paths. Their original bytes and the exception hashes are retained in the implementation workspace's `out/docs-original/` and `out/review2-docs/MOVE-AUDIT.json`; these documentation exceptions are not byte-identical copies. Pinned producers, builders, and the original identity manifest remain unchanged.

## Model integration notes moved from compiler documentation

For the supported complete-graph build route, see [Complete ResNet50 host graph](docs/resnet-full-host.md). The graph builder enables `mstatus.FS`, initializes `fcsr` to nearest-even, reserves a 64 KiB stack and checks ELF assets and load segments. A linked ELF does not establish host execution, DMA reachability or cache coherence on a selected SoC.

`tools/attention_timing_contract.py` validates attention metadata geometry, descriptor mask extents, count consistency, padding, and any supplied contract against the staged bytes. Tiny's v2 exporter and the Smol host builder invoke this check. Existing hosts copy those immutable templates and patch only operand spans. This validates generator inputs; the caller still owes the invocation-wide immutability assumption.

Contracts cannot suppress another timing rule. The 45 remaining frozen attention programs all end with `dma.wait ch1; delay 128; ecall`. Supplying their exact descriptor/count images resolves their DMA ownership and capacity rejections, but all 45 then reject DELAY-before-halt. The explicit `atlas-kernels` halt variant at `ed70fb5` removes the dead terminal delay without changing the default generator. All 45 regenerated halt programs pass as `contract-verified` with these exact metadata contracts; their words differ from the frozen kit. Without contracts, their metadata loads remain unconstrained. The frozen streams remain `stream-verified`.
