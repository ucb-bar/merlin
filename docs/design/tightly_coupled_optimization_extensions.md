# Additional opportunities within the tightly coupled accelerator class

This short companion to the [optimization catalog](joint_optimization_catalog.md)
keeps the hardware class broader than the selected Gemmini interface. These are
optional hypotheses: the agent chooses whether, how and when to explore them.
They prescribe no implementation, constants, schedule or pass sequence.

A mechanism needs additional capabilities beyond “CPU-issued” or “tightly coupled.”
The current Gemmini run does not receive these as active optimization directions.
Unknown means discovery is needed; not exposed means the selected interface does
not grant it; unsupported refers only to the selected execution contract.

| Opportunity | Additional prerequisites | Possible compiler layer | Current selected interface |
| --- | --- | --- | --- |
| Pass shared host tensor storage directly and omit staging/cache-maintenance copies when ownership permits | accelerator participates in declared host coherence domain; shared buffer accesses have specified acquire/release visibility | Memory-effect/ownership dialect with coherence-domain attribute; Zero-copy legality pass; Shared-buffer ABI adapter | **unknown**: The selected memory_model and hardware_promises are empty. DMA readback named coherent_dump_v1 is an observation transport, not a CPU/accelerator coherence proof. |
| Yield the host while work runs and resume from a completion interrupt | per-job completion interrupts; interrupt delivery and acknowledgement ABI; race-free sleep/wakeup protocol | Runtime completion-event abstraction; Interrupt-backed wait adapter; Host blocking versus polling policy | **not_exposed**: Selected harness exposes entry and completion function calls, with no interrupt registration, event or wait ABI. This does not establish absence of interrupt wiring. |
| Compile reusable command lists for autonomous execution, amortizing CPU instruction issue | device fetches commands from memory; command-list encoding and bounds; submission doorbell and list completion | Command-list dialect/backend; Relocation and pointer patching pass; List submission/completion runtime adapter | **not_exposed**: The selected endpoint is inline_asm_insn. No device-fetch command-list ABI is granted; recorded software command_buffer.json is not evidence of autonomous hardware execution. |
| Fuse wider pointwise/reduction epilogues into vector execution without scalar host roundtrips | independently programmable vector ALU; lane masks and accumulator-to-vector transport; supported elementwise numeric formats | Vector-epilogue dialect; Contraction-to-vector ownership/transport adapter; Mask-aware fusion and register allocation | **not_exposed**: Selected compute_units declares a systolic int8 unit; existing composed elementwise/reduction support does not expose an independent vector instruction set. |
| Select floating matrix instructions directly instead of integer quantization or host decomposition | floating operand and accumulator modes with exact arithmetic contract; format conversion and scale behavior specified | Floating contraction type/rounding dialect; Format-aware instruction selection; Overflow/underflow and error-budget proof interface | **unsupported**: Selected contraction runtime is explicitly i8xi8_i32 and the exposed matrix dtype is int8. FP scale fields for readout do not authorize floating matrix multiplication. This is unsupported in this selected interface; broader hardware possibility remains unproven. |
| Transfer sparse/noncontiguous tensor regions without scalar packing when descriptor setup is cheaper | hardware transfer descriptor supports indexed/segmented addresses; bounded segment count and alignment; defined fault/completion behavior | Segmented transfer dialect; Descriptor construction/coalescing pass; Scatter/gather legality adapter | **not_exposed**: Selected movement support declares copy; no indexed descriptor ABI or scatter/gather semantics are exposed. Ordinary strided matrix loads are insufficient evidence for arbitrary scatter/gather. |

The [structured companion](tightly_coupled_optimization_extensions.json) records
contract evidence and relevant legality/measurement questions. Named compiler
layers are proposed extension points, not claims of existing implementations.

Keep richer mixed-precision, VPU and asynchronous-runtime catalogs separate and
compose them only when the target/runtime actually has those capabilities. The
breadth of a catalog does not establish compiler generalization: that requires
other target adapters and experiments. Disclose the catalog as compiler assistance
in comparisons. Private implementations, chosen schedules, tuning constants and
reference performance scores remain excluded from the automatic loop.
