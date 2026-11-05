# Model ELF handoff to FPGA and FireSim

`tools/atlas_model_handoff.py` copies an unchanged bare-metal ELF and creates a FireSim baremetal workload description and memory-footprint manifest. The manifest is `PREPARATION_ONLY`; numerical evidence comes from separate live QEMU/model or SoC receipts. Full TinyLlama and SmolVLA handoffs require the full-model ELF and its final footprint.

The default profile, `runtime/ee290-model-platform.json`, matches the `EE290GenericIOSimConfig` software contract: uncached IMEM at `0x20000`, CSR at `0x40000`, VMEM base `0x20000000`, system DRAM at `0x80000000` for 64 GiB, and HTIF completion. The staging arena at `0x100000000` lies within system DRAM. Hosts do not access VMEM directly; the profile specifies no host VMEM aperture size.

The ELF must define distinct, aligned, writable eight-byte `tohost` and `fromhost` symbols. Success uses `tohost=1`; failure uses `(exit_code << 1) | 1`. Observation symbols must be writable loaded memory outside the DMA arena. The arena must avoid executable host memory, IMEM, and CSRs, and fit one 32-bit DMA window; high addresses use the runtime's 64-bit host pointers.

The manifest distinguishes each ELF `PT_LOAD` file size from its memory size. Loaders must copy file bytes and zero the remaining extent, including NOLOAD DMA segments and BSS. Stack storage counts when reserved in loaded RAM.

Generate a package from a completed host build, with output paths outside this checkout:

```sh
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$ATLAS_SESSION/readiness"
python3 -B tools/atlas_model_handoff.py \
  --elf "$MODEL_ELF" --name "$MODEL_NAME" \
  --dma-base 0x100000000 --dma-bytes "$MODEL_DMA_BYTES" \
  --observe-symbol "$MODEL_MAILBOX" \
  --out "$ATLAS_SESSION/readiness/packages/$MODEL_NAME"
```

The fresh output directory contains `host.elf`, `handoff.json`, `workload.json`, and the exact selected `platform.json`. The manifest records platform, compiled host-layout, and ELF hashes; entry point; file/zero-fill ranges; HTIF and observation addresses; and resident host/DMA bytes, counting overlapping NOLOAD storage once. The ELF embeds its runtime assets, so external capture files are omitted.

Select another profile with `--platform "$MODEL_PLATFORM"`. Changed IMEM/CSR/VMEM or staging ABI requires a matching rebuilt host layout. Supply a generated map with `--soc-memmap "$TARGET_MEMMAP" --target-identity "$TARGET_ID"`. Checks cover every loaded byte's permissions, uncached read/write IMEM and CSR access, and host-visible read/write DMA coverage. The manifest distinguishes a selected platform contract from a generated target map and records/copies those inputs. A supplied identity is a label; target execution must establish accelerator DMA connectivity.

Before a target run, compare the packaged ELF hash with the admitted host and preserve its numerical receipts. Check load/zero-fill ranges against target memory and loader address width. Bind the map to the prebuilt bitstream or AGFI, driver, and Atlas RTL revision, and qualify cache/DMA coherence. A fence alone supplies no cache maintenance. Different cache hooks or address maps require a rebuilt, reviewed platform interface that preserves admitted program and payload bytes. A hardware pass requires target completion and every embedded output/preserved-byte check. Packaging supplies neither hardware execution nor target qualification.

## UART TSI

Run the matching prebuilt UART TSI loader with the selected target and serial device:

```sh
"$UART_TSI" +tty="$UART_DEVICE" +baudrate=921600 +selfcheck \
  "$MODEL_PACKAGE/host.elf"
```

The baud rate comes from `EE290BringupHostConfig`; use the actual target's rate if different. `+selfcheck` reads back ELF load chunks byte by byte. Loading uses TSI/FESVR and HTIF exit, with no simulator `+loadmem` bypass. Zero exit requires target HTIF completion; external bit-exact comparisons need separate evidence. Retain loader output and the model mailbox/results.

## FireSim

Stage the package into an existing matching FireSim deployment:

```sh
install -D "$MODEL_PACKAGE/host.elf" \
  "$FIRESIM_DEPLOY/workloads/$MODEL_NAME/host.elf"
install -D "$MODEL_PACKAGE/workload.json" \
  "$FIRESIM_DEPLOY/workloads/$MODEL_NAME.json"
cd "$FIRESIM_DEPLOY"
./firesim runworkload -c "$RUNTIME_YAML" -a "$HWDB_YAML" -r "$BUILD_RECIPES_YAML"
```

Set `workload.workload_name` to `$MODEL_NAME.json`, select the matching Atlas-capable `target_config.default_hw_config`, and retain `workload.terminate_on_completion: true`. Generated JSON sets `common_rootfs: null` to prevent derived filesystem selection. FireSim resolves the boot binary under `workloads/<benchmark_name>/`, generates `+prog0`, and collects `uartlog`. The prebuilt driver, bitstream, and named memory-region bridge must match the map, including the high DMA arena.
