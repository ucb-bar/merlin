# Phase 1 compiler author guidance

Read this with the selected public software declaration, target contract and the
shared `command_buffer_abi.yaml` / `mlir_oot_backend_contract.yaml`. The selected
bytes define admission and numerical semantics; this general checklist explains
how to meet them across programs and IR spellings. It supplies no additional
hardware forms or operation-specific lowering recipes. Ordinary runs read the
public software data leaf directly; `read_scheduling_contracts` is available only
when the selected shared-pass compiler library grants it.

## What counts as device work

1. **The persisted command buffer is your claim.** The offload census and
   command-buffer evaluation read `--emit-command-buffer`, not your ELF; the
   host-compute audit joins that buffer to lowered host LLVM. Put every unit task
   in `commands`, including literal materialization and internal helper opcodes.
   Moving commands into another `params` field loses their device accounting.
   Basis: command-buffer ABI `commands`; census `programs` / `instruction_use`.
2. **Never leave a program silent.** Finish with commands and declared placement,
   declared host placement, or `declined` with a reason. No commands, placement
   or decline means `silent`: it fails full accounting and is audited as an
   accelerator claim. Declaring host work makes it visible; it does not satisfy
   required accelerator coverage. Basis: census `all_accounted`, `silent`,
   `host_only`, `declined`; mandatory accelerator coverage.
3. **Declare every region in `params.lane_placement`.** A unit lane's name contains
   `mesh`, `accel`, `unit` or `device`. Name a host lane `host`; other names are
   read as host too. Basis: census `placement_undeclared`, `unit_regions` and
   `host_regions`.
4. **Movement and fills are device work.** Use the declared data path: DMA
   loads/stores, stride-0 source-row replay, zero load from DRAM address 0, and
   accumulator readout. Respect each form's width, stride and tail constraints.
   A compile-time-known result still needs device production and accounting.
   Basis: software `operations.movement`, `transfer_contracts.operand_load` and
   `transfer_contracts.accumulator_readout`; target movement capabilities.
5. **Floating storage follows its declared numerical model.** Where
   `numerical_semantics.component_domain.floating_storage_arithmetic` selects
   the quantized model, quantize with the declared calibration and scale policy,
   evaluate exactly, and round at the declared byte readout. An IEEE f32 host
   island satisfies neither that result nor device work. Preserve separately
   declared exact movement/identity and readout semantics. Basis: software
   `numerical_semantics`, including `calibration_domain` and `readout`.
6. **Declare an unsupported stage.** Place it on the host with its reason, or
   decline explicitly. A missing compiler lowering is not evidence that hardware
   cannot express the work. Basis: backend contract `host_compute_boundary`,
   route declaration quality, and mandatory accelerator coverage.

## Host work budget

7. **Budget host work per output element.** The published default limits are
   1.0 total arithmetic operations (integer, float and conversion), with at most
   0.25 float/conversion operations, and 0.5 bytes of host scalar payload per
   output byte. Fixed prologues can exceed the budget on small outputs: check
   rank 0, one element, and shapes smaller than a tile row. Address math is not
   free: only independently proved address/control subsets are excluded; raw
   counts remain reported. Basis: `host_compute_audit.Budget`,
   `arithmetic_per_element`, `value_arithmetic_per_element`, `payload_ratio`.
8. **Keep scalar data memory out of accelerator kernels.** Avoid host
   `load`/`store` and data `alloca`. Use the zero-load path for zeros; keep
   nonzero constant rows in read-only module data and DMA them, replaying outer
   rows at stride 0 where declared. Host addresses, branches and command issue
   still need to remain within the audit. Basis: host audit scalar payload and
   software `operations.movement` / `transfer_contracts`.
9. **Derive coordinates from nested induction variables or constants.** Hoist
   tile clipping bounds or use zero rows. Flattened per-element coordinate
   recovery with `udiv` / `urem` / `select` is charged unless independently
   proved excluded. Basis: host audit arithmetic and address-loop checks.
10. **Issue commands per tile or row block.** Avoid per-element command issue;
    aim well below one command per 16 output elements. This is an authoring
    target, not a grading exemption or a fixed hardware tile size. Derive legal
    geometry from the selected facts. Basis: host audit command-scale budget.
11. **Keep staging on the device data path.** Do not materialize an im2col
    matrix through DRAM round-trips. Gather source taps into scratchpad with
    declared strided loads and accumulate blocks in the accumulator, rather
    than storing a staged matrix back out. Basis: software `operations.contraction`
    and target transfer capabilities; host audit scalar payload.
12. **Produce widened results without per-element host casts.** A `sext`/cast
    loop is value arithmetic. If no declared device form produces the required
    width, decline with the actual limitation. Basis: host audit
    `value_arithmetic_per_element`; selected operation dtypes and readout width.

## Recognize the frontend's IR

13. **Handle linalg-on-tensors as well as `merlin_iface`.** Static
    `tensor.extract_slice`, `insert_slice`, `concat`, `collapse_shape` and
    `expand_shape` can carry the same movement semantics. A rank-reducing slice
    followed by reshape is a strided copy; derive its bounds and layout from IR.
    Basis: software `operations.movement` forms, ranks, layouts and tails.
14. **Recognize indexed movement structurally.** A parallel `linalg.generic`
    yielding only `tensor.extract` of an input at indices affine in `linalg.index`
    is the movement row's strided read. Derive offsets and per-axis strides.
    Basis: software `operations.movement`, including `as_strided` and its bounds.
15. **Recognize gated reductions structurally.** A sum gated by a `linalg.index`
    comparison and select against literal zero is the `operations.reduction_prefix_sum`
    row, along the actual reduced axis. That row declares the triangular 0/1
    contraction or accumulate-on-load realization, not a new scan instruction.
    Preserve its exact accumulator and bound obligations. Basis: software
    `operations.reduction_prefix_sum` and its evidence.
16. **Recognize contractions across carriers.** Products feeding sum reductions
    over shifted slices are contraction work. Do not leave their elementwise
    products on the host. Basis: software `operations.contraction`; mandatory
    accelerator coverage and host-compute audit.
17. **Own raw-interface intermediates internally.** Mark them `role=intermediate`,
    keep them out of the external `kernel_abi` pointer list, and own their storage
    in an internal global rather than a data `alloca`. Declare the program's
    `outputs` so internal destinations are not published as results. Basis:
    command-buffer tensor roles and backend contract logical input/output order.

## Self-check before declaring done

18. **Census every spelling and every program.** Check interface forms, captured
    linalg, constant-result programs (`arith.constant`, `tensor.splat`,
    `linalg.fill`), and programs with no inputs. Run
    `python agent_selfcheck.py --submission submission --offload-census` when
    the full `--capsules all` result lacks a complete census. Fix every `silent`
    row and every unintended `host_only` row; read `placement_undeclared` and
    `host_compute_vetoed` as well as the pass count. Basis: self-check census
    `programs`, `all_accounted` and host-compute findings. Emit-only accounting
    does not replace required numerical or RTL checks.
19. **Read failure detail as well as category.** A refusal labelled
    "compiler authoring gap" still needs its detail read before choosing a fix.
    A pending, unavailable or unknown check is not a pass. Basis: public
    `qa/verdict.json` failure plane/category/detail and self-check results.

## Maintaining this document

These are public authored task inputs. Preserve experimental contracts when
editing them. The runner archives the served `TASK.md`; generated-prompt modes
use shared rendering and point here for this checklist. Never edit an archived
run or sealed payload in place.
