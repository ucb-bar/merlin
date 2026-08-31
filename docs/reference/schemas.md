# Schema reference

_Generated from `merlin/schemas/*.schema.yaml` by `build_tools/scripts/gen_schema_docs.py` — do not edit by hand; run the generator._

The schemas are the cross-workstream coordination contract (see [Contracts](contracts.md)). Each is the source of truth for one artifact type.

| Schema | Title | Purpose |
|---|---|---|
| `abstraction_candidate` | Abstraction Candidate | A candidate HW/SW abstraction surfaced from kernels or design pressure (e.g. resident_packed_tensor, accumulator_commit). Owned by kernel-mining; consumed by… |
| `baseline_cost` | Baseline Cost Breakdown | A measured/modelled latency breakdown for a workload, decomposed into named cost components, each tagged with its evidence source. The baseline input to the… |
| `command_buffer` | Command Buffer | The Merlin-owned, target-independent command-buffer format. A command buffer is an ordered list of opaque commands plus resource/handle tables and a list of… |
| `compilation_strategy` | Compilation Strategy | A first-class, hashable description of ONE way of compiling a workload: the contract assumptions made, schedule policies applied, interface features exposed,… |
| `cpu_coupling` | CPU Coupling Measurements | Measured host/runtime coupling overhead for a workload, under different dispatch regimes (op-level vs batched command buffer). Consumed by merlin-dse-guidanc… |
| `design_pressure` | Design Pressure Report | Per-workload measurable pressures extracted at one or more compiler cut points (reuse, lifetimes, layout conversions, dispatch counts, intermediate bytes). O… |
| `dialect_plan` | Dialect Plan | A plan for a target dialect derived from a target_contract: which ops and types to expose, how they lower, and what tests to generate. Owned by TargetGen. |
| `dialect_requirement` | Dialect Requirement | What a *validated interface candidate* requires from a target dialect (L6): the ops, types and verifier conditions a target must provide to implement the abs… |
| `dse_axis_triage` | DSE Axis Triage | The key DSE-guidance output: a ranking of accelerator DSE axes by how much of the measured/trace-derived target gap each axis can actually close for a given… |
| `dse_result` | DSE Result | Results of comparing variants (baseline / software_visible / hardware_managed / oracle) for a candidate feature, with measurable cost-model parameters. Owned… |
| `evidence_report` | Evidence Report Index | Machine-readable index of the source evidence TargetGen collected for a target: the files discovered (docs, Scala/Chisel, examples) with short filename/first… |
| `exploitability_report` | Exploitability Report | How much of the oracle benefit a compiler can actually capture for a feature across a parameter sweep (the 'compiler exploitability' of an abstraction). Owne… |
| `interface_candidate` | Interface Candidate | A concrete target-independent interface abstraction proposed for the interface dialect, with the design pressure that justifies it. Owned by design-pressure/… |
| `kernel_record` | Kernel Record | Normalized record of a single mined kernel from an external source. Owned by the kernel-mining workstream; consumed by design-pressure/DSE. |
| `llvm_extension_plan` | LLVM Extension Plan | Records whether and how a target needs LLVM-project changes. The default posture is out-of-tree (TableGen fragments / intrinsic headers / runtime calls) with… |
| `llvm_requirement` | LLVM Requirement | Whether a mined abstraction needs LLVM-project (backend/MC) support — the L8 record. Emitted by kernel mining per interface candidate, feeding TargetGen's ll… |
| `metrics` | Metrics | The Merlin-owned common metrics schema. All backends and target adapters normalize their raw counters into these common names so results are comparable acros… |
| `paper_run_result` | Frozen Compiler Paper Run Result | One lifecycle-complete cell in the K1 paper matrix. A cell distinguishes build, execution, correctness, task quality, session latency, peak memory, exact exe… |
| `paper_study` | Frozen Compiler Paper Study | Versioned input contract for a holdout-safe compiler comparison. Development workloads and paper models are disjoint; the compiler policy is frozen before an… |
| `policy_rule` | Policy Rule | A compiler heuristic distilled from kernel evidence: when a condition holds, take these scheduling actions. Owned by kernel-mining; consumed by DSE and (late… |
| `quant_format` | Quantization Format | Describes a numeric / quantization FORMAT structurally and target-agnostically: its element encoding (bit width, and exponent/mantissa split for floats), sub… |
| `runtime_adapter_plan` | Runtime Adapter Plan | Describes how a target implements the Merlin-owned runtime abstraction. Merlin owns the runtime ABI, command-buffer schema, event/handle/metrics model; the t… |
| `runtime_candidate` | Runtime Candidate | A runtime-level abstraction surfaced from kernels whose bottleneck is launch/dispatch/state rather than schedule (e.g. command-buffer batching for many tiny… |
| `rvv_package_manifest` | RVV Target-Package Manifest | Provenance + identity for one isolated RVV codegen package under artifacts/targets/rvv/<run_id>/. RVV is a transform-dialect SCHEDULE + cflags, not a residen… |
| `rvv_result` | RVV Experiment Result | One certify_rvv run of (RVV package x workload) across coupled targets. Records the K-ladder verdict, correctness gate, per-target cycles (with cycle_accurat… |
| `search_space` | Search Space | Declarative description of a search over candidate compiler artifacts: which candidate type, which method (grid \| evolutionary \| map_elites), the parameter s… |
| `session_contract` | Stateful Inference Session Contract | Capture-owned mapping from semantic inference steps to the compiled forward ABI. Declares stage semantics, output-to-input state carry, and per-observation i… |
| `target_contract` | Target Contract | Describes a hardware/software target: its capabilities, the obligations it places on the compiler, and the promises the hardware/runtime makes. Owned by the… |
| `target_source_manifest` | Target Source Manifest | Records the inputs TargetGen was pointed at for a target: local source directories, files, Scala roots, example directories, plus optional URLs and a branch/… |
| `temporal_workload_metadata` | Temporal Workload Metadata | A small multi-rate wrapper around an existing workload region. Captures the temporal structure that flat captures hide: the K-step denoise/action loop, the a… |
| `trace` | Trace | The Merlin-owned trace-event stream format. Backends emit ordered trace markers (e.g. resident_hit, eviction, dispatch boundaries) so runs can be inspected a… |
| `workload_region` | Workload Region | A small, analyzable workload region: ops, tensors, shapes, dtypes, reuse, and lifetimes. The shared input format for design-pressure analysis. Owned by desig… |
| `zephyr_plan` | Zephyr Plan | Describes the Zephyr runtime-backend scaffold to generate for a target: the module, the devicetree binding, Kconfig symbols, the driver API surface, and whic… |
