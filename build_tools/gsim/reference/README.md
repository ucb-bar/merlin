# Native FASED boundary reference

This harness instantiates FireSim's **existing** `MemModelKey` timing core. It
builds independently through Verilator and gSIM, then checks every meaningful
output on every target cycle. It is a local CPU reference, not a FireSim launch.

The standalone wrapper flattens boundary IO without changing timing logic. The
CIRCT conversion preserves assertions. `expand_else_when.py` only expands
indentation syntax and makes diagnostic verification labels unique; the
Verilator build uses the original emitted Verilog as its independent path.
Both native drivers sample outputs before each rising edge and reject input
values wider than their ports. gSIM's internal debug step count is not a target
cycle count.

## Reproduce

Provision a **task-local editable copy** of the pinned FireSim `sim` tree with
its pinned Scala submodules and sbt dependencies. Do not point this script at
the deployed hardware checkout. Supply tools explicitly; no PATH discovery,
FPGA access, or downloads happen in these scripts. Dependencies may need to be
provisioned before the sbt build. Use a task-local `TMPDIR`.

```sh
python3 build_tools/gsim/reference/build_reference.py \
  --project "$LOCAL_FIRESIM_SIM" --selection "$SELECTION_JSON" \
  --output "$NATIVE_OUTPUT" --java "$JAVA_BIN" --firtool "$FIRTOOL_BIN" \
  --circt-translate "$CIRCT_TRANSLATE_BIN" --gsim "$GSIM_BIN" \
  --verilator "$VERILATOR_BIN" --cxx "$CXX_BIN" --jobs 2
python3 build_tools/gsim/reference/probe_boundary.py \
  --reference "$NATIVE_OUTPUT/verilator/VFasedTimingStandalone" \
  --candidate "$NATIVE_OUTPUT/gsim-driver/run" \
  --ports "$NATIVE_OUTPUT/verilator-driver/ports.json" \
  --settings "$NATIVE_OUTPUT/rtl/runtime-defaults.conf" \
  --build-receipt "$NATIVE_OUTPUT/build-receipt.json" \
  --output "$NATIVE_OUTPUT/probe"
merlin-timing memory-boundary-compare \
  --reference "$NATIVE_OUTPUT/probe/reference.jsonl" \
  --candidate "$NATIVE_OUTPUT/probe/candidate.jsonl" \
  --output "$NATIVE_OUTPUT/probe/comparison.json"
```

`SELECTION_JSON` uses `merlin.memory_reference_selection.v1`, with the Config
class and explicit AXI geometry. For a qualified selection, pin the actual
configuration, timing/DRAM/LLC source, build receipt, and **all runtime overrides**
using `merlin-timing memory-reference-prepare`. Emitted defaults alone do not
establish the deployed configuration. The retained prototype selection has no
deployment source receipt and deliberately remains unqualified.

Build receipts record source/tool/output hashes and exact commands. The canary
reopens all build outputs and binds traces to that receipt, effective runtime
settings, initial state, scope and exact stimulus. It explicitly discloses
runtime values truncated by the existing hardware register widths rather than
silently altering the RTL to make them fit.

## Delivered acceptance and remaining integration

The retained probe completed 11 transactions over 312 post-reset target cycles:
read, write, readback, 1/8-beat requests, several addresses/IDs, repeated cache
line access and R/B backpressure. All returned IDs/data/LAST are checked. Both
engines match with zero meaningful signal mismatches. Invalid response payloads
are masked by VALID; ready signals, counters and other meaningful outputs are
compared. Eight reset cycles precede the contiguous comparison window.

The functional byte store and egress adapter in `probe_boundary.py` support this
**one-outstanding-transaction**, aligned 64-bit INCR, full-strobe scope. They are
not yet wired into gSIM's production SimDRAM/mm_magic_t interface. This evidence
does not cover multiple outstanding requests, reordering, interleaved IDs,
partial writes, target full-SoC behavior, or equivalence to deployed FireSim.
Those are explicit remaining gates before replacement/activation; no timing
coefficient or production model selection changes here.

Curated results and compressed exact traces are in
`experiments/reference-data/fased-boundary-20261023/`. Full local build outputs
and unsanitized receipt are under `out/artifacts/memory-reference-native-20261023/`.
The source/tool hashes in curated provenance permit review without publishing
machine-specific paths or native executables.

## Subsequent production-interface integration

The standalone scope above is retained historical evidence. The production
`mm_t`/SimDRAM integration, concurrent/partial-write probes and actual CPU ELF
execution are now implemented separately; see [PRODUCTION.md](PRODUCTION.md).
They close the functional-interface gap while retaining the independent
**deployed-FireSim equivalence** and whole-system timing qualification gates.
