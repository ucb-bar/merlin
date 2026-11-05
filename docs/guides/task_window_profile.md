# Diagnostic task windows

`merlin_experiments.phase2.task_window_profile.instrument_task_windows(text, cb,
symbol_prefix="profile_case0")` returns `lowered_mlir_text`, `c_support`,
`completion_statement`, `dump_statement` and a manifest. Compile the returned
LLVM through the ordinary object path, link its C support against the existing
`htif.h` runtime, and add the completion statement immediately after the existing
completion call. Dump after ending the whole-kernel counter window.

Each declared task gets one timestamp before its first emitted instruction.
The parsed CFG must show that the marker dominates the task and all returns,
and its block belongs to no cycle. Disjoint, absent or unsupported task layouts
refuse. Original emitted lines and source metadata remain byte-identical.

The final marker closes the last task window. Adjacent `mcycle` and `minstret`
samples give modulo-64-bit deltas; require every hit count to equal one and the
invalid count to equal zero. Console output uses the existing signed decimal
HTIF formatter; normal positive raw values must be below 2^63. These windows
include hook overhead and any register-allocation changes from instrumentation.
Kernel prologue/call overhead before the first mark remains outside task windows.
They provide diagnostic CPU costs, with source effects pending.
