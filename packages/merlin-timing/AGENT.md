# AGENT.md — packages/merlin-timing

This independent distribution owns the canonical C++17 streaming engine and
stdlib Python model/calibration tooling. It imports no Merlin compiler, xDSL,
NumPy or target provider. ISA decoding, hardware facts, timing parameters and
target scheduling policy remain outside the generic engine.

Provider source packaging and build orchestration are shared here. Target
repositories supply source bytes, an entrypoint and configuration facts; do not
add target names, file rosters, opcodes or model coefficients to the builder.

`include/merlin_timing/engine.hpp` is canonical; the wheel stages those exact
bytes in its portable source payload. Do not maintain another implementation.
Caller-supplied prices, model estimates and observed hardware cycles retain
separate provenance and qualification. Unknown timing cannot become zero cost.

Tests use synthetic prices and bounded native CPU execution only. Build products
belong in a caller-selected temporary/build directory, never beside source.
