# Native FASED integration into gSIM SimDRAM

`mm_fased_t` implements the existing Chipyard `mm_t` functional-memory interface.
The selected gSIM `SimDRAM` blackbox now instantiates that class in an explicitly
built emulator. The TSI ELF loader, terminal dump and CPU/accelerator requests
share the original backing mapping. No source checkout, deployed hardware or
campaign simulator selection is changed by this build.

The actual FASED core determines admission, scheduling and response cycles.
The adapter supplies functional bytes and per-ID response queues; it contains
no latency formula, fitted coefficient or substitute scheduler. Request sizes,
geometry and bounds are checked against the compiled contract. Unsupported
locked or wrapping bursts fail explicitly. Aligned INCR transfers and one-beat
FIXED transfers are supported; unaligned addresses are currently rejected.

## Build and check

First run `build_reference.py` as described in README.md with a selection that
matches the target SimDRAM geometry. Then:

```sh
python3 build_tools/gsim/reference/build_mm.py \
  --native-build "$NATIVE_OUTPUT" \
  --runtime-settings "$EXPLICIT_RUNTIME_SETTINGS" \
  --vendor-csrc "$TESTCHIPIP_CSRC" --riscv-include "$RISCV_INCLUDE" \
  --verilator-include "$VERILATOR_INCLUDE" --output "$ADAPTER_OUTPUT" \
  --harness-receipt "$SELECTED_GSIM_BUILD_RECEIPT"
python3 build_tools/gsim/reference/probe_mm.py \
  --reference "$ADAPTER_OUTPUT/verilator-mm" \
  --candidate "$ADAPTER_OUTPUT/gsim-mm" \
  --selection "$ADAPTER_OUTPUT/selection.json" \
  --receipt "$ADAPTER_OUTPUT/integration-receipt.json" \
  --output "$ADAPTER_OUTPUT/probe"
"$RISCV_GCC" -march=rv64imac_zicsr -mabi=lp64 -nostdlib -nostartfiles \
  -T build_tools/gsim/reference/memory_smoke.ld \
  build_tools/gsim/reference/memory_smoke.S -o "$ADAPTER_OUTPUT/memory-smoke.elf"
timeout 60 "$ADAPTER_OUTPUT/emulator-fased" \
  "$ADAPTER_OUTPUT/memory-smoke.elf" +max-cycles=200000
```

The ELF must precede emulator plusargs because the retained FESVR parser treats
the first positional argument as its program. All outputs belong in a task-local
output directory. Provision dependencies separately and use a task-local
`TMPDIR`. The integration receipt pins copied blackbox source, reused native
objects, generated core, explicit runtime settings, geometry and output binaries.
The build accepts the selected original receipt and byte-checks its source and
binary before copying/patching. It does not rebuild the entire target.

## Functional and cycle contract

At each SimDRAM tick, the adapter consumes the ready/valid state already
advertised to the target. It evaluates the native core at clock low, advances
exactly one rising edge, selects the next functional egress payload and settles
outputs without another edge. It refuses a configuration whose ready or
response signals change with newly presented inputs in a way incompatible with
the registered SimDRAM interface. This is checked dynamically rather than
assuming every FASED configuration has the same boundary behavior.

The reset boundary advertises no readiness. Its first release tick cannot
consume an unacknowledged request. Reset discards outstanding transactions and
resets timing state while retaining backing memory. Reads snapshot addressed
byte lanes at AR acceptance; writes update only enabled byte lanes at W
acceptance. Simultaneous accepted reads observe bytes before same-cycle writes,
a deterministic policy for otherwise unordered cross-channel races. Programs
requiring a specific read-after-write order must synchronize their accesses.
Per-ID FIFO queues preserve response order within each ID while allowing the
native scheduler to order different IDs. Payloads remain stable until handshake;
read beats advance only on RVALID && RREADY and release on LAST. W can be
presented before AW and is held until the native interface accepts it; the
adapter never invents an AW or accepts unmatched W data.

## Measured acceptance

On the 2026-10-23 selected build:

- 40 transactions completed: 28 reads and 12 writes, including readback.
- Peak outstanding counts were 10 reads and 3 writes, with repeated IDs.
- 95 R and 56 B backpressure cycles; six cycles of W-before-AW presentation.
- Partial and zero strobes; 1-, 2-, 4- and 8-byte transfers; burst LAST checks.
- Verilator and gSIM matched all meaningful outputs across 756 target cycles.
- Invalid geometry, size, WLAST and out-of-lane strobes were rejected.
- The linked whole-SoC gSIM emulator executed the public CPU ELF, checked byte/
  halfword updates and 64 cache lines, and exited successfully at 77,346 cycles.

That last count includes loader/boot and harness work. It is a functional smoke
result, **not a performance calibration or benchmark measurement**.

## Explicit deployment gap

The local build selects `firesim.configs.FRFCFS16GBQuadRankLLC4MB`, 35-bit addresses,
4-bit IDs, 64-bit data, base `0x80000000`, 16 GiB and 64-byte maximum transfers.
`id_reuse` and `max_flight` edge metadata remain unspecified in this selection.
All selected native source files, tools, runtime settings and generated outputs
are pinned. Requested settings and effective register-width values are both
recorded; upstream default queue depth 8 occupies a 3-bit register and becomes
0 at that boundary. The adapter reproduces this assignment without changing RTL.

Read-only inspection of the local FPGA wrapper and AWS SSM document resolves
only a call to a manager-local script. It does not reveal the deployed bitstream
source identity, runtime override file or AXI edge limits. No manager was
started and no command was sent to it. Consequently **deployed FireSim
equivalence remains NOT_ESTABLISHED**. The linked native memory integration is
implemented and functionally demonstrated, but whole-system timing must remain
provisional until deployment identities and the independent per-case timing
gates pass. Do not activate it as a qualified model based on these probes.

Curated evidence: `experiments/reference-data/fased-simdram-20261023/`.
Full task-local artifacts: `out/artifacts/fased-simdram-20261023/`.
