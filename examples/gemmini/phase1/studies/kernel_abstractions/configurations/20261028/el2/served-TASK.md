# Compiler construction ablation

Build one general compiler in submission/ for every selected public xs capsule. Start with an empty target backend. Preserve the admitted source semantics, precision, fusion boundaries and every live output. Implement the manifest and required entrypoints from the common backend contract.

The selected compiler_library supplies C++ OOT scaffolding generators and their support dependencies only.
Language/tooling environment: C++/MLIR.
Declare language: cpp and a real C++/MLIR build block in manifest.yaml.
Read AUTHOR_PROFILE.json for the exact API inventory. Import supplied components with submission/compiler_library on the Python module search path.

Raw hardware, ordering and reference evidence and the independent numerical/ISA/timing judge are common across profiles. Use only the granted inputs; hidden data and prior implementations are withheld. Production decisions derive from semantics, shapes and hardware facts, never case names.

Use python agent_selfcheck.py for focused numerical and timing checks, immutable variants for alternatives and the supplied request/attach interfaces. Check qa/verdict.json source identities. A model estimate is provisional; unknown checks remain unknown. Keep working within the selected clock and helper/CPU limits. Freeze the compiler before final offline evaluation.


## Selected public scope and runtime limits

- `PUBLIC_CASES.json` lists the actual selected public case IDs and their suites: 109 core, 40 scalability, 3 programs; 152 public cases in total. Every listed case retains its declared numerical, placement, output and required evaluation obligations.
- Authoring budget: 4 hours (14400 seconds), followed by at most 900 seconds for the selected final drain; total run ceiling 15300 seconds. Provider waiting and retries count within the original clock.
- The author and its helpers share the assigned 15-CPU pool. Preparation, compilation and simulation share that same 15-CPU pool; helpers receive no additional CPUs. At most 8 helper agents may be active concurrently.
- `mlir_oot_backend_contract.yaml` defines the submitted manifest, build and entrypoint contract. Read `interface_grammar.md`, `interface_dialect_contract.yaml`, `command_buffer_abi.yaml`, `target_dialect_contract.yaml`, `target_contract.yaml`, `schemas/` and `isa/` for the public source, output, ABI and instruction declarations. The common LLVM/MLIR installation is exposed under `install/`; `MERLIN_MLIR_INSTALL` selects the installation prefix when supplied to an authored build.
- `AUTHOR_PROFILE.json` records the exact selected author tools and source boundary. Use only that selection and the granted public inputs. This runtime block adds no mandatory tool call, discovery sequence, construction order or optimization policy.
- Source checkpoints are scheduled at 0, 30, 60, 120, 240 minutes from authoring start; the harness records actual capture times and source identities. Final functional and physical evaluation is deferred until authoring freezes. Private transfer sources and results are unavailable during authoring; private whole-model campaigns are outside this selected study. On-demand development feedback remains provisional.

## Selected public software declaration

Read `/work/merlin/tasks/el4-grader-throughput-20261024/merlin/out/build/agent-workspaces/phase1/gemmini/merlin_assisted/xs-el123-20261028-r2-el2/workspace/public_declarations/software-spec.yaml` for the original selected software semantics (SHA-256 `3c1b8aa1c77775e1c0e590af871f05e038743d7000a99860fb12843f7053372f`). This is a readable data declaration; The original backend tree remains withheld. This identical-byte alias grants no additional implementation or qualification.
