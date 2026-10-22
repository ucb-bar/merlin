# Bounded BYOC validation — 2026-10-20

The final C1 backend passes the three native smoke workloads on **both Spike
and the existing GemminiRocketConfig Verilator simulator**. Each positive guest
executes twice and checks exact INT32 outputs, primitive counts, output and workspace
overrun guards, input preservation, input rejection and the stack canary.
The matmul guest also passes a separate pinned vendor OS-adapter practice check.
The wrong-oracle control passes on both engines: application numerical failure,
guest 22, and the expected engine-specific termination are observed. An additional
input-corruption control must reach guest 26 on both engines.

The committed [validation.json](validation.json) binds final sources, toolchain,
generated objects/ELFs, engine identities, receipts and observed resource limits.
Detailed build commands, partitioned Relax, **actual fused** semantic TensorIR,
replayed schedule traces, exported graph/object, guest and logs are retained in
`out/artifacts/verification/gemmini/tvm/byoc-20261020/preservation-{tails,bmm,conv}/`.
The [host report](../../../../../out/artifacts/verification/gemmini/tvm/byoc-20261020/host-preservation/report.json)
records seven passing regression methods, including parameterized subcases.

| Workload, including CPU work | Graph calls / workspace | Tile updates per inference | Spike graph cycles, calls 1 / 2 | RTL graph cycles, calls 1 / 2 | RTL wall time |
| --- | --- | --- | --- | --- | --- |
| INT8 `[3,17] @ [17,5]`, INT32 bias and ReLU; worst-case −128 operands | 1 / 0 B | 2 | 2938 / 2938 | 4644 / 4175 | 23.29 s |
| INT8 `[2,2,17] @ [2,17,3]`, INT32 bias and ReLU, batch slicing/concat | 5 / 192 B | 4 | 5837 / 5837 | 8747 / 8197 | 21.31 s |
| INT8 NCHW `[1,2,2,2]`, OIHW `[3,2,3,3]`, pad1, INT32 bias/ReLU, packing/restoration | 3 / 192 B | 2 | 4871 / 4871 | 7707 / 7109 | 21.54 s |

Graph cycles are guest **mcycle** differences around `model_run`. Two calls are
shown as observed; no statistical speedup, timing distribution or trained search
winner is inferred. Spike's values are functional instruction-model diagnostics,
not Rocket performance. RTL uses the existing 256-MiB **magic-memory** model,
without DRAMSim. These are not FireSim timing measurements.

The vendor OS **raw matmul** practice check observes 954 Spike cycles and 2127
RTL cycles. It excludes bias/ReLU and uses its own tiling/transfer strategy;
the table's BYOC entries include the full graph. These are different boundaries
and do not establish a comparative speedup. The new intrinsic deliberately pays
extra accumulator DMA and fences to implement composable `C += A @ B` semantics.

Every final run has a 1-GiB cgroup cap, no swap and a one-CPU quota. Guest runs
have 45-second wall deadlines and RTL has a 200,000-cycle ceiling. The positive
RTL runs peak at approximately **25–53 MiB** cgroup memory; final Spike runs peak
at approximately **180–359 MiB**. The host regression suite peaks below 123 MiB and
finishes in about four seconds. Recorded memory events report no OOM kills.
Services run serially; no hardware, LLVM or TVM C++ rebuild was performed.

The final wrong-oracle RTL control completes in 16.65 seconds, prints
`BYOC_NUMERICAL_FAILURE`, decodes TSI guest 22 and returns direct host status
255. Its [receipt](../../../../../out/artifacts/verification/gemmini/tvm/byoc-20261020/preservation-tails/rtl-bad-oracle.json)
and [log](../../../../../out/artifacts/verification/gemmini/tvm/byoc-20261020/preservation-tails/rtl-bad-oracle.log)
retain those separate facts. The Spike negative control returns 22 with the same
application failure marker. Positive RTL logs include both the application PASS
and TestDriver `$finish`.

The original input-preservation check was invalid: the C compiler pooled the
constant `input` and `input_copy` arrays onto the same address, so comparing them
could not detect corruption. The corrected guest uses writable input and volatile
runtime reads compared with an independently host-computed 64-bit FNV digest.
All three cases were rebuilt and passed both engines again. Their arithmetic
operator objects are byte-identical to the original fused runs. The new
`--bad-input` control deliberately changes one input byte after numerical checks:
both engines print `BYOC_INPUT_PRESERVATION_FAILURE`, Spike returns 26, and RTL
decodes guest 26 with direct host status 255 in 22.45 seconds. Earlier numerical,
guard and instruction evidence remains separate; earlier input-preservation
claims are superseded. The previous machine-readable summary is retained as
`out/artifacts/verification/gemmini/tvm/byoc-20261020/validation-before-preservation-fix.json`.

Final linked ELFs have no unresolved symbols and pass source-derived forbidden
hardware-loop audits. `XCUSTOM_ACC=3` selects opcode **0x7b**, not 0x0b. Each build
assembles a standalone forbidden-loop instruction and requires the audit to
detect it before accepting the production ELF. The selected hardware lineage
and existing runtime are recorded by the retained
[hardware binding](../../../../../out/artifacts/verification/gemmini/tvm/rtl-blocks-20261020T050040Z/hardware-binding.json).
That binding does not newly authenticate the full simulator/toolchain source
closure or prove equality with a FireSim image.

Development attempts remain preserved. The initial 16×16 staging design hit the
exporter's hidden-workspace rejection at the default 1024-byte threshold. The
final schedule uses 16×8 staging and ordinary TVM compaction instead. Early guests
needed freestanding memset/memcpy support, and `rdcycle` trapped under the pinned
Spike ISA selection; the final machine-mode diagnostic uses `mcycle`. An early
negative-log parser missed whitespace around the TSI exit code; final controls
use the actual decoded failure. The audit's early hardcoded opcode was replaced
with the header-derived opcode and its injected instruction control. Intermediate
host scheduling probes also exposed TVM's positive-seed requirement and the need
for a public global symbol when independently executing a TIR module; both are
covered by the final passing runs. Superseded attempts do not count as passes.

The result qualifies these small integer workloads and the implementation's
stated contracts. [SUPPORT.md](SUPPORT.md) explains the remaining operation
limits. Full-model execution, complete ATen import coverage, arbitrary dtypes,
general strided/overlapping aliases, FireSim timing and model quality remain
outside this validation.
