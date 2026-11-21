# Compile and run programs

These examples use an existing compiler or provider. To build a new target
compiler with an agent, use [compiler generation](../compiler-generation/README.md).

| Example | What it exercises |
| --- | --- |
| [Quickstart](quickstart/README.md) | Compile checked-in MLIR to portable scalar C, use the same Python service, and optionally run on native Linux. |
| [Native host](native-host/README.md) | Capture, inspect LLVM lowering and compare a model on the host. |
| [Compiler APIs](compiler-api) | Program construction, composition, fusion proposals and scheduling-contract examples. |
| [Triton](triton/README.md) | Small frontend kernels and their existing compile/run walkthrough. |
| [Kodiak](kodiak) and [gemmelos](gemmelos) | Board-specific invocation scripts using the shared board catalog. |
| [FireSim modelblaster](firesim_modelblaster) | Prepare and collect existing guest model runs; FireSim provisioning stays external. |

Start with the quickstart for the smallest set of prerequisites. Target examples
need the selected external compiler, runtime and hardware inputs documented there.
Source programs stay here; generated products go under `out/`.
