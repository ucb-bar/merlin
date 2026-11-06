# RV64 host execution with a model-backed Atlas device

`tools/host_iss/run.py` executes a host ELF under QEMU `virt` and intercepts the actual instructions in `atlas_host_direct_read32` and `atlas_host_direct_write32`. QEMU stops while one `npu_model.Core` executes each Atlas program; host arithmetic runs between starts. Use [full-model contracts](full-model-contracts.md) for complete Tiny and Smol hosts and their segment inputs.

## Inputs

| Workload | Bundle schema | Atlas launches |
| --- | --- | ---: |
| `resnet` | `atlas.full_resnet_bundle.v3` | 54 |
| `tiny` | `atlas.block_bundle.v1` | 24 |
| `smol` | `atlas.block_bundle.v1` | 7 |
| `tiny-full` | `atlas.full_bundle.v1` | 840 |
| `smol-full`, projection-only profile | `atlas.full_bundle.v1` | 2,715 |
| `smol-full`, attention-inclusive full policy | `atlas.full_bundle.v1` | 2,915 |

The Smol projection-only profile launches only the 2,715 projection panels. The attention-inclusive full policy adds 200 attention launches, for 2,915 Atlas launches.

Supply the matching `host/host.elf`, `SHA256SUMS`, provenance, workload contract, programs, weights, initial images, and expected outputs. Inputs are external fixed-capture fixtures. The [ResNet route](resnet-full-host.md) includes bundle preparation; [Smol inputs](smol-full-host.md) include the required phase archive. Checksums bind content, so retain an independently trusted archive or checksum digest. The reader checks listed members, safe physical paths, source hashes, geometry, ELF mappings, and embedded payloads before starting QEMU.

Use model revision `9e6313bddecb8ba936206eb347f5cd4e5335282c`, its Python 3.14 environment, Git, and `qemu-system-riscv64`. Model source must stay unchanged during a run. The bridge selects the `curve` backend and records its parameters and digest. Set numerical-library thread counts to one.

## Run

Use [model bringup setup](../README.md#build), keeping `ATLAS_ROOT` and `ATLAS_MLIR_BIN` on the canonical compiler; export `ATLAS="$VIEW"`, `ATLAS_SOURCE_ROOT="$VIEW"`, and `ATLAS_MLIR_ROOT="$VIEW"`, and use `cd "$VIEW"`.

```sh
"$MODEL_PYTHON" -B tools/host_iss/run.py \
  --workload resnet --elf "$BUNDLE/host/host.elf" \
  --bundle-root "$BUNDLE" --model-root "$NPU_MODEL_ROOT" \
  --qemu "$QEMU_RISCV64" --out "$MODEL_OUTPUT_ROOT/resnet-iss" --wall-seconds 21500
```

Select another workload with `--workload`. The wall limit includes input verification and execution; full models require an explicit limit. Default per-start model cycle caps are 20,000,000 for ResNet, 200,000,000 for Tiny, and 100,000,000 for Smol. `--kernel-cycle-cap` sets a positive alternative. Runtime assertions require ordinary Python; `-O` is rejected.

QEMU uses one TCG CPU, 8 GiB guest RAM, no BIOS, and a local GDB socket. Atlas staging is at `0x100000000`, IMEM at `0x20000..0x3ffff`, and control/status at `0x40000`, with START at `+24`. For UART profiles and the explicit SiFive bridge, see [result output](full-model-contracts.md#result-output).

Each START checks actual guest-written instructions and the complete input image. The device checks exact DMA totals, output bytes, preserved inputs, guards, drained completion, and invocation markers before copying its actual buffer back. Registers, VMEM, accumulators, DRAM, and counters persist between starts. Completion checks the workload mailbox, trap state, tohost, launch ledger, and final buffers: 1,000 logits for ResNet, outputs and caches for block workloads, and full inference or policy state for full models.

Successful execution writes `PASS_RV64_HOST_WITH_MODEL_DEVICE` in `receipt.json`. Retained outputs include input/tool/source hashes, per-kernel summaries, actual guest readbacks, `uart.log`, and QEMU diagnostics. Failures retain the first failing boundary. Owned QEMU processes are reaped on exit or deadline.

## Independent SoC kernel checks

`tools/chipyard/block_kernels.py` builds LOADMEM ELFs from `atlas.block_soc_kernels.v1` manifests. Each entry declares a unique `id`, program and initial/expected DRAM paths and hashes, aligned `dram_base`, and disjoint output ranges. Images cover the complete region and preserve every byte outside the outputs. Use the complete inventory when reporting block coverage: 24 kernels for Tiny or seven for Smol.

```sh
python3 tools/chipyard/block_kernels.py build \
  --manifest "$BLOCK_SOC_MANIFEST" --cc "$RISCV_CC" --out "$MODEL_OUTPUT_ROOT/block-build"
python3 tools/chipyard/block_kernels.py run \
  --build-root "$MODEL_OUTPUT_ROOT/block-build" --out "$MODEL_OUTPUT_ROOT/block-runs" --jobs 1 \
  --simulator "$EE290_SIMULATOR" --dramsim-ini "$DRAMSIM_INI" \
  --max-cycles 2000000000 --wall-seconds 172800
python3 tools/chipyard/block_kernels.py results \
  --build-root "$MODEL_OUTPUT_ROOT/block-build" --runs-root "$MODEL_OUTPUT_ROOT/block-runs" \
  --out "$MODEL_OUTPUT_ROOT/block-results.json"
```

Provide a prebuilt simulator and its library/license environment. `--jobs` is bounded at 18 per invocation; allocate resources and enforce the aggregate limit. `--resume` reuses only completed exact runs with matching ELF/status records. Results require every listed kernel, matching ELF hashes, zero process exits, no timeout, exact output/preservation checks, and completion markers.

`build --fast-check` selects stable-span volatile XOR/OR checks with byte prefixes/tails and a failure-only byte scan. A disappearing mismatch fails; fast logs count bytes, while default logs retain mixed word/byte units. The flag binds `runtime/atlas_host_check.h`; `ATLAS_KERNEL_CHECK_TIMING` measures the check interval using the guest counter.

## Checks and limits

Run `PYTHONPATH=tools:test python -m unittest discover -s test -p 'test_host_iss*.py'` for CPU-only protocol and contract checks. Most use synthetic payloads and an in-memory transport. Set `SMOL_BLOCK_BUNDLE` to a qualified selected-layer bundle to run three admission tests against its real ELF and payloads; mutations stay in memory. Numerical runs require actual host instructions and device execution; linking, static checks, and UART PASS records alone do not qualify a model. QEMU plus the instruction model establishes functional integration, while Chipyard interconnect behavior, FPGA execution, platform timing, and task accuracy require separate evidence.
