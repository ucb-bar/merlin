# Fresh successor launch

`el4-successor.yaml` is a prospective template. It selects an empty compiler,
3 hours of authoring and a separate final hour for strict capsule certification,
freeze and publication. Preparation time is recorded separately. The CPU
allocation is `0-5,24-31`: author `4,5`, compute `0-3,24-31`. The ordinary
reservation owner checks availability and restores host CPU masks on exit.
External full-model validation has its own owner and starts after the capsule
owner terminates. The original 120-second compilation limit and 1 MiB program
bound remain requirements of the selected resource policy.

Keep archived configurations, releases, source snapshots and receipts unchanged.
A passing V26 candidate is neither an input nor a prerequisite for this release.
Infrastructure preparation uses selected tools/runtime evidence and accepts an
empty candidate. It does not turn unknown mandatory Phase 0 obligations into a
pass or qualify a compiler.

## Prepare and review the source and corpus

Select one clean Merlin revision, compatible installed Python dependencies,
source-produced target facts, capability contract, board/runtime selections,
public capture sources and an independent hidden cohort. Use the ordinary
[Phase 0 setup and release gates](../../../../docs/guides/phase0_phase1_setup.md).
The supplied definition must select the new recipe and source inputs explicitly:

```sh
merlin experiment inspect "$SPEC" --phase 0 --run-dir "$P0"
merlin experiment preflight "$SPEC" --phase 0 --run-dir "$P0"
merlin experiment run "$SPEC" --phase 0 --run-dir "$P0"
merlin experiment corpus coverage "$P0" --spec "$CONFORMANCE"
```

Resolve required unknowns and inspect omissions before sealing. New generated
corpus counts come from this run, not a retained 153-case denominator.

Export the generic shared SDK from the selected installed sources. This exports
no target compiler, prior candidate or handwritten target schedule. The tool
registry supplies the selected EL4 tool source membership; staging source bytes
does not grant extra tools:

```python
from pathlib import Path
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release
from merlin.targetgen.tool_registry import arm_tools

prepare_shared_source_release(
    Path(SDK_SOURCE), python_source_root=Path(REPO) / "src",
    license_directory=Path(REPO), tools=arm_tools("merlin_rtlchecks"),
    source_binding=True,
)
```

The export records actual dependency versions, license/notice and source hashes.
Its three declared namespace replacements remain separate from original host
source identity. Select a Python environment with all actual dependencies,
including `z3-solver`; no fake distribution version establishes readiness.
The launch selects any separate dependency directory through
`PYTHON_DEPENDENCIES` and pins its inventory through
`PYTHON_DEPENDENCY_SELECTION`. Include those same selected bytes during release
preparation. An inventory identifies files; native SDK admission remains required.

`fire-aligned-successor.yaml` records the prospective release call. The equivalent
ordinary API lets the operator retain the exact selected policies and pins:

```python
from pathlib import Path
from merlin_experiments.corpus import release

release.prepare(
    Path(P0), Path(RELEASE), generated_only=True, capsule_scope="accelerator",
    selected_levels=("EL4",), selected_variants=("public_v0",),
    phase1_policy_descriptor=Path(PHASE1_POLICY_DESCRIPTOR),
    lean_toolchain_manifest=Path(LEAN_TOOLCHAIN_MANIFEST), host_index_bits=64,
    kernel_timing_selection=(Path(KERNEL_TIMING_CONTRACT), KERNEL_TIMING_SHA256),
    common_substrate="merlin.common_compiler_substrate.v6",
    python_source_root=Path(SDK_SOURCE),
)
```

Use `private_baseline=Path(PRIVATE_HIDDEN_ROOT)` only when the selected source
requires an independently bound hidden cohort. Inspect the prepared release,
review its actual digest, then call ordinary `corpus seal` with that digest,
reviewer and review note. No preparation helper seals on the operator's behalf.
Omitting the new substrate/source options preserves historical V3 preparation.

The selected bundle's `compiler_tool_profile` publishes the exact generic API,
implementation and automatic-checker inventories and source digests. The normal
native session independently verifies that inventory and source coverage. A new
SDK export, corpus seal or inert scaffold alone proves no runtime or compiler
qualification. Preserve the public host projection/instruction selections needed
by model-containing public capsules even though full-model workers are deferred.

For prospective gSIM correspondence use the generic entry point with explicit
target-owned selection data:

```sh
"$PYTHON" -m merlin_experiments.phase1.gsim_alignment \
  --selection "$REPO/examples/gemmini/phase1/launch/gsim-alignment.yaml" \
  --contract "$TARGET_CONTRACT" --rtl-facts "$RTL_FACTS" \
  --chipyard-root "$CHIPYARD" --hardware-pins "$HARDWARE_PINS" \
  --board-catalog "$BOARD_CATALOG" --gsim-dts "$GSIM_DTS"
```

This structurally parses DTS, joins the selected board and source pins, and
retains independent elaboration/build/correspondence checks. Its result grants
neither chip equivalence nor timing transfer. The archived no-selection API and
its target-specific audit remain available unchanged.

## Bind and check the prospective launch

After the release is sealed, prepare a new private host-projection spec at
`$PREP/private-models.yaml`, instruction selection/approval, selected input bundle,
LLVM/MLIR toolchain and runtime files. Select all machine-dependent paths through
variables; the operator's candidate inventory is not an admission certificate.

```sh
"$PYTHON" "$REPO/examples/gemmini/phase1/launch/prepare_successor.py" \
  --task-root "$TASK_ROOT" --prep "$PREP" --run-id "$FRESH_RUN_ID" \
  --output "$BOUND" \
  --var REPO="$REPO" --var PYTHON="$PYTHON" --var RELEASE="$RELEASE" \
  --var PHASE0_RUN="$P0" --var SDK_SOURCE="$SDK_SOURCE" \
  --var PYTHON_DEPENDENCIES="$PYTHON_DEPENDENCIES" \
  --var PYTHON_DEPENDENCY_SELECTION="$PYTHON_DEPENDENCY_SELECTION" \
  --var NUMERICS_BUNDLE="$NUMERICS_BUNDLE" --var CHIPYARD="$CHIPYARD" \
  --var SPIKE_RUNTIME="$SPIKE_RUNTIME" --var GSIM_EMULATOR="$GSIM_EMULATOR" \
  --var TIMING_PROFILE="$TIMING_PROFILE" --var RTL_FACTS="$RTL_FACTS" \
  --var MODEL2MLIR="$MODEL2MLIR" --var MLIR_TOOLCHAIN="$MLIR_TOOLCHAIN" \
  --var INSTRUCTION_SELECTION="$INSTRUCTION_SELECTION" --var ORACLE_TIMING="$ORACLE_TIMING" \
  --var KERNEL_TIMING_CONTRACT="$KERNEL_TIMING_CONTRACT" \
  --var LEAN_TOOLCHAIN_MANIFEST="$LEAN_TOOLCHAIN_MANIFEST" \
  --var PHASE1_POLICY_DESCRIPTOR="$PHASE1_POLICY_DESCRIPTOR"
"$PYTHON" "$REPO/examples/gemmini/phase1/launch/launch_phase1.py" \
  --config "$BOUND/config.yaml" --task-root "$TASK_ROOT" \
  --prep "$PREP" --run-id "$FRESH_RUN_ID"
```

The binder refuses a dirty source checkout, reused configuration directories or run IDs, missing bytes
or missing V6 profile. It pins current selected files and retains the actual
existing seal's review metadata. `configuration_bound` is not `ready`: the second
command independently checks launch inputs without starting an author. The
normal session still owns source/SDK/tool admissions and strict public/hidden,
resource and final certification checks. FP-quality report-only policy never
stands in for exact selected quantized-reference checks.

For the authorized actual launch, wrap that second command (with `--run`) in
`resource_reservation.py --cpus 0-5,24-31 --max-seconds 15300 --unit UNIQUE
--state STATE --log LOG -- COMMAND`. The outer 900-second preparation allowance
does not extend the controller's 14,400-second clock. Use a fresh alphanumeric
`merlinphase1...` unit name. No external model or FPGA job starts through this
capsule launcher. Terminal status, logs, resource restoration and the frozen
handoff remain the operator's durable outputs.
