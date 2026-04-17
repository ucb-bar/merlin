from __future__ import annotations

from collections import OrderedDict

from .model import (
    EvidenceItem,
    PatchSurface,
    SupportPlan,
    TargetCapabilities,
    TaskNode,
    VerificationManifest,
    VerificationStage,
)

PRIMARY_INTEGRATION_ORDER = [
    "runtime_hal",
    "structured_text_isa",
    "post_global_plugin",
    "llvm_ukernel",
]

COMMON_EVIDENCE = [
    EvidenceItem(
        "repo_path",
        "docs/architecture/plugin_and_patch_model.md",
        "Merlin extension and patch policy",
    ),
    EvidenceItem(
        "repo_path",
        "docs/reference/cli.md",
        "Current CLI surface and user-facing command model",
    ),
    EvidenceItem(
        "repo_path", "tools/compile.py", "Current compile-target view that the planner must eventually derive"
    ),
]

INTEGRATION_EVIDENCE = {
    "post_global_plugin": [
        EvidenceItem(
            "repo_path",
            "docs/how_to/add_compiler_dialect_plugin.md",
            "Documented post-global-optimization compiler-plugin workflow",
        ),
        EvidenceItem(
            "repo_path",
            "third_party/iree_bar/compiler/src/iree/compiler/PluginAPI/Client.h",
            "IREE pipeline extension seam used by Merlin plugins",
        ),
    ],
    "llvm_ukernel": [
        EvidenceItem(
            "repo_path",
            "docs/dev_blog/2026-03-13-riscv-mmt4d-ukernel-workstream.md",
            "Current ukernel, tiling, and RISC-V codegen workstream",
        ),
        EvidenceItem(
            "repo_path",
            "third_party/iree_bar/compiler/src/iree/compiler/Codegen/ExternalInterfaces/CPUEncodingExternalModels.cpp",
            "Current CPU encoding and tile model extension point",
        ),
    ],
    "structured_text_isa": [
        EvidenceItem(
            "repo_path",
            "compiler/src/merlin/Dialect/NPU/README.md",
            "Existing staged kernel/schedule/ISA design in Merlin",
        ),
        EvidenceItem(
            "repo_path",
            "docs/dev_blog/2026-03-11-npu-dialect-e2e.md",
            "Current NPU end-to-end validation and simulator flow",
        ),
    ],
    "runtime_hal": [
        EvidenceItem(
            "repo_path",
            "docs/how_to/add_runtime_hal_driver.md",
            "Documented HAL driver integration workflow",
        ),
        EvidenceItem(
            "repo_path",
            "runtime/src/iree/hal/drivers/radiance/README.md",
            "Current runtime-first backend example in Merlin",
        ),
    ],
}


def build_support_plan(capabilities: TargetCapabilities) -> SupportPlan:
    target_families = _classify_target_families(capabilities)
    integration_styles = _derive_integration_styles(capabilities)
    primary_integration = next(style for style in PRIMARY_INTEGRATION_ORDER if style in integration_styles)
    required_layers = _derive_required_layers(capabilities, integration_styles)
    patch_surfaces = _derive_patch_surfaces(capabilities, integration_styles)
    verification_manifest = build_verification_manifest(capabilities, integration_styles)
    task_graph = build_task_graph(capabilities, integration_styles)

    return SupportPlan(
        schema_version=1,
        target=capabilities.identity.name,
        display_name=capabilities.identity.display_name,
        vendor=capabilities.identity.vendor,
        maturity=capabilities.identity.maturity,
        target_families=target_families,
        integration_styles=integration_styles,
        primary_integration=primary_integration,
        required_layers=required_layers,
        derived_artifacts=_derive_artifacts(capabilities),
        patch_surfaces=patch_surfaces,
        deployment_profile=capabilities.deployment.name if capabilities.deployment else None,
        verification_manifest=verification_manifest,
        task_node_ids=[task.id for task in task_graph],
    )


def build_task_graph(
    capabilities: TargetCapabilities,
    integration_styles: list[str] | None = None,
) -> list[TaskNode]:
    styles = integration_styles or _derive_integration_styles(capabilities)
    tasks: list[TaskNode] = [
        TaskNode(
            id="review_target_spec",
            title="Review canonical capability and deployment inputs",
            phase="planning",
            family="shared",
            depends_on=[],
            evidence=_merge_evidence(capabilities, []),
            actions=[
                "Validate the capability spec and optional deployment overlay "
                "against Merlin's canonical TargetGen schema.",
                "Check the declared execution model, ISA exposure, memory "
                "model, numeric contract, and verification resources for "
                "internal consistency.",
            ],
            write_scope=[
                "build/generated/targetgen/<target>/support_plan.json",
                "build/generated/targetgen/<target>/verification_manifest.json",
            ],
            acceptance_checks=[
                "Schema validation passes with no unknown or contradictory fields.",
                "The target can be classified without introducing ad hoc target-specific rules.",
            ],
            **_execution_contract(
                capabilities,
                family="shared",
                phase="planning",
                task_id="review_target_spec",
                write_scope=[
                    "build/generated/targetgen/<target>/support_plan.json",
                    "build/generated/targetgen/<target>/verification_manifest.json",
                ],
            ),
        ),
        TaskNode(
            id="derive_compile_view",
            title="Derive compile-target and profile view",
            phase="planning",
            family="shared",
            depends_on=["review_target_spec"],
            evidence=_merge_evidence(
                capabilities,
                [
                    EvidenceItem(
                        "repo_path",
                        "docs/how_to/add_compile_target.md",
                        "Current compile-target YAML schema",
                    ),
                    EvidenceItem(
                        "repo_path",
                        "models",
                        "Current handwritten compile target views",
                    ),
                ],
            ),
            actions=[
                "Translate canonical capabilities into the compile-target shape " "used by tools/compile.py.",
                "Record the build profile, compile target name, and compile hw "
                "mode that should be derived from the planner output.",
            ],
            write_scope=[
                "build/generated/targetgen/<target>/compile_view.yaml",
                "models/<target>.yaml (derived in a later adoption phase)",
            ],
            acceptance_checks=[
                "Generated compile view matches current handwritten target " "YAMLs for exemplar targets.",
                "No compile-only fact is stored exclusively outside the "
                "canonical capability spec or deployment overlay.",
            ],
            **_execution_contract(
                capabilities,
                family="shared",
                phase="planning",
                task_id="derive_compile_view",
                write_scope=[
                    "build/generated/targetgen/<target>/compile_view.yaml",
                    "models/<target>.yaml (derived in a later adoption phase)",
                ],
            ),
        ),
    ]

    if capabilities.deployment is not None:
        tasks.append(
            TaskNode(
                id="derive_deployment_overlay",
                title="Derive deployment and hardware-backend view",
                phase="deployment",
                family="shared",
                depends_on=["review_target_spec"],
                evidence=_merge_evidence(
                    capabilities,
                    [
                        EvidenceItem(
                            "repo_path",
                            "docs/hardware_backends/overview.md",
                            "Current deployment recipe model in Merlin",
                        ),
                        EvidenceItem(
                            "repo_path",
                            "build_tools/hardware",
                            "Current handwritten hardware recipes",
                        ),
                    ],
                ),
                actions=[
                    "Translate deployment overlay data into the current "
                    "hardware-recipe shape used by merlin chipyard and board "
                    "flows.",
                    "Preserve Chipyard pins, board runtime assumptions, and "
                    "simulator metadata as deployment-only facts.",
                ],
                write_scope=[
                    "build/generated/targetgen/<target>/deployment_view.yaml",
                    "build_tools/hardware/<recipe>.yaml (derived in a later adoption phase)",
                ],
                acceptance_checks=[
                    "Deployment view preserves all required pins and execution metadata from current recipes.",
                    "Deployment-only facts are not leaked into the canonical capability spec.",
                ],
                **_execution_contract(
                    capabilities,
                    family="shared",
                    phase="deployment",
                    task_id="derive_deployment_overlay",
                    write_scope=[
                        "build/generated/targetgen/<target>/deployment_view.yaml",
                        "build_tools/hardware/<recipe>.yaml (derived in a later adoption phase)",
                    ],
                ),
            )
        )

    previous = "derive_compile_view"
    for style in styles:
        task_id = f"implement_{style}"
        tasks.append(
            TaskNode(
                id=task_id,
                title=_task_title(style),
                phase=_task_phase(style),
                family=style,
                depends_on=[previous],
                evidence=_merge_evidence(capabilities, INTEGRATION_EVIDENCE[style]),
                actions=_task_actions(style),
                write_scope=_task_write_scope(style),
                acceptance_checks=_task_acceptance(capabilities, style),
                **_execution_contract(
                    capabilities,
                    family=style,
                    phase=_task_phase(style),
                    task_id=task_id,
                    write_scope=_task_write_scope(style),
                ),
            )
        )
        previous = task_id

    tasks.append(
        TaskNode(
            id="define_verification_ladder",
            title="Define the generated verification ladder",
            phase="verification",
            family="shared",
            depends_on=[previous],
            evidence=_merge_evidence(
                capabilities,
                [
                    EvidenceItem(
                        "repo_path",
                        "docs/hardware_backends/compatibility_matrix.md",
                        "Current backend maturity and deployment combinations",
                    ),
                ],
            ),
            actions=[
                "Generate a staged verification manifest from spec sanity "
                "through hardware and performance validation.",
                "Attach family-specific smoke, lowering, runtime, " "simulator, and parity checks to the manifest.",
            ],
            write_scope=[
                "build/generated/targetgen/<target>/verification_manifest.json",
                "build/generated/targetgen/<target>/task_graph.json",
            ],
            acceptance_checks=[
                "The verification ladder includes only stages justified by "
                "declared resources and current Merlin families.",
                "The manifest exposes an explicit target maturity level "
                "instead of a binary supported/unsupported status.",
            ],
            **_execution_contract(
                capabilities,
                family="shared",
                phase="verification",
                task_id="define_verification_ladder",
                write_scope=[
                    "build/generated/targetgen/<target>/verification_manifest.json",
                    "build/generated/targetgen/<target>/task_graph.json",
                ],
            ),
        )
    )
    return tasks


def build_verification_manifest(
    capabilities: TargetCapabilities, integration_styles: list[str]
) -> VerificationManifest:
    stages: list[VerificationStage] = [
        VerificationStage(
            0, "spec_validation", "ready", "Canonical capability and deployment inputs are schema-validated."
        ),
        VerificationStage(
            1, "planner_validation", "ready", "Support-plan and task-graph generation are deterministic."
        ),
        VerificationStage(
            2,
            "compiler_structural",
            "ready"
            if any(
                style in integration_styles for style in ("post_global_plugin", "structured_text_isa", "llvm_ukernel")
            )
            else "not_applicable",
            "Compiler-facing targets require structural registration and pass " "visibility checks.",
        ),
        VerificationStage(
            3,
            "lowering_codegen",
            "ready"
            if any(
                style in integration_styles for style in ("post_global_plugin", "structured_text_isa", "llvm_ukernel")
            )
            else "not_applicable",
            "Lowering and codegen checks apply to compiler-facing targets.",
        ),
        VerificationStage(
            4,
            "runtime_smoke",
            "ready" if capabilities.runtime.required or capabilities.deployment is not None else "planned",
            "Runtime and deployment-backed targets need smoke validation " "before numeric checks.",
        ),
        VerificationStage(
            5,
            "numerical_parity",
            "ready" if capabilities.verification.golden_model.get("available") else "planned",
            "Numerical parity is available when a golden model or software " "reference exists.",
        ),
        VerificationStage(
            6,
            "simulator_or_hardware",
            "ready"
            if capabilities.verification.simulator.get("available")
            or capabilities.verification.rtl.get("available")
            or "board" in capabilities.platform.environments
            else "planned",
            "Execution validation depends on simulator, RTL, or hardware " "availability.",
        ),
        VerificationStage(
            7,
            "performance_characterization",
            "ready" if capabilities.verification.perf_counters.get("available") else "planned",
            "Performance characterization requires counters or other timing " "resources.",
        ),
    ]
    return VerificationManifest(
        target=capabilities.identity.name,
        maturity=capabilities.identity.maturity,
        stages=stages,
    )


def render_explain_text(
    capabilities: TargetCapabilities,
    support_plan: SupportPlan,
    task_graph: list[TaskNode],
) -> str:
    lines = [
        f"Target: {capabilities.identity.display_name} ({capabilities.identity.name})",
        f"Vendor: {capabilities.identity.vendor}",
        f"Maturity: {capabilities.identity.maturity}",
        f"Access model: {capabilities.access.model}",
        f"SDK requirements: {', '.join(capabilities.access.sdk_requirements) or 'none'}",
        f"Credential requirements: {capabilities.access.credential_requirements}",
        f"Families: {', '.join(support_plan.target_families)}",
        f"Integrations: {', '.join(support_plan.integration_styles)}",
        f"Primary integration: {support_plan.primary_integration}",
        f"Required layers: {', '.join(support_plan.required_layers)}",
        f"Deployment overlay: {support_plan.deployment_profile or 'none'}",
        "",
        "Derived artifacts:",
    ]
    lines.extend(f"- {artifact}" for artifact in support_plan.derived_artifacts)
    lines.append("")
    lines.append("Planned task order:")
    lines.extend(f"- {task.title} [{task.family}]" for task in task_graph)
    return "\n".join(lines)


def _classify_target_families(capabilities: TargetCapabilities) -> list[str]:
    kind = capabilities.execution.kind
    if kind == "vector_cpu_extension":
        return ["rvv_cpu_extension"]
    if kind == "matrix_coprocessor":
        return ["riscv_vendor_matrix_extension"]
    if kind == "rocc_accelerator":
        return ["rocc_accelerator"]
    if kind == "structured_npu":
        return ["structured_npu_text_isa"]
    if kind == "simt_gpu":
        return ["simt_gpu_hal"]
    if kind == "mixed":
        return ["mixed_cpu_accelerator"]
    return ["cpu_only"]


def _derive_integration_styles(capabilities: TargetCapabilities) -> list[str]:
    styles: list[str] = []
    if capabilities.execution.compiler_recovery_stage == "post_global_optimization":
        styles.append("post_global_plugin")
    if capabilities.execution.kind == "structured_npu" or capabilities.isa.exposure.kind in {
        "text_isa",
        "custom_exporter",
    }:
        styles.append("structured_text_isa")
    if (
        capabilities.runtime.required
        or capabilities.execution.kind == "simt_gpu"
        or capabilities.isa.exposure.kind == "hal_runtime"
    ):
        styles.append("runtime_hal")
    if (
        capabilities.isa.exposure.kind in {"llvm_intrinsics", "llvm_backend_features"}
        or capabilities.execution.compiler_recovery_stage == "llvmcpu_codegen"
    ):
        styles.append("llvm_ukernel")
    deduped: OrderedDict[str, None] = OrderedDict()
    for style in styles:
        deduped[style] = None
    return list(deduped.keys()) or ["llvm_ukernel"]


def _derive_required_layers(
    capabilities: TargetCapabilities,
    integration_styles: list[str],
) -> list[str]:
    layers = ["canonical_spec", "planner", "task_graph", "verification_manifest"]
    if capabilities.deployment is not None:
        layers.append("deployment_overlay")
    if "post_global_plugin" in integration_styles:
        layers.extend(["merlin_compiler_plugin", "target_dialect"])
    if "structured_text_isa" in integration_styles:
        layers.extend(["schedule_and_isa_export"])
    if "runtime_hal" in integration_styles:
        layers.extend(["runtime_hal_driver"])
    if "llvm_ukernel" in integration_styles:
        layers.extend(["llvm_backend_or_intrinsics", "ukernel_tile_model"])
    return list(OrderedDict((layer, None) for layer in layers).keys())


def _derive_artifacts(capabilities: TargetCapabilities) -> list[str]:
    artifacts = [
        "support_plan.json",
        "task_graph.json",
        "verification_manifest.json",
        "execution_bundle.json",
        "execution_state.json",
        "briefs/",
        "compile_view.yaml",
    ]
    if capabilities.deployment is not None:
        artifacts.append("deployment_view.yaml")
    return artifacts


def _derive_patch_surfaces(
    capabilities: TargetCapabilities,
    integration_styles: list[str],
) -> list[PatchSurface]:
    surfaces: list[PatchSurface] = []
    if "llvm_ukernel" in integration_styles:
        reasons: list[str] = []
        if capabilities.isa.exposure.needs_llvm_backend_changes:
            reasons.append("backend_feature_bits_or_instruction_patterns")
        if capabilities.isa.exposure.needs_new_intrinsics:
            reasons.append("llvm_intrinsic_surface")
        if capabilities.isa.exposure.needs_new_feature_bits:
            reasons.append("target_feature_bits")
        if reasons:
            surfaces.append(PatchSurface("llvm", reasons, "agentic_contextual_edit"))
        if capabilities.tiles.preferred_tiles or capabilities.memory.packing:
            surfaces.append(
                PatchSurface(
                    "iree",
                    ["cpu_encoding_models", "ukernel_tile_selection", "pack_unpack_selection"],
                    "agentic_contextual_edit",
                )
            )
    if capabilities.deployment is not None and capabilities.deployment.mode == "bare-metal":
        surfaces.append(
            PatchSurface(
                "iree_runtime",
                ["bare_metal_runtime_validation"],
                "agentic_contextual_edit",
            )
        )
    return surfaces


def _merge_evidence(
    capabilities: TargetCapabilities,
    extra: list[EvidenceItem],
) -> list[EvidenceItem]:
    items = list(COMMON_EVIDENCE)
    items.extend(extra)
    items.extend(
        EvidenceItem(
            "repo_path",
            ref,
            "Target-specific reference from the canonical capability spec",
        )
        for ref in capabilities.references
    )
    if capabilities.deployment and capabilities.deployment.hardware_recipe:
        items.append(
            EvidenceItem(
                "repo_path",
                capabilities.deployment.hardware_recipe,
                "Current handwritten deployment recipe corresponding to this overlay",
            )
        )
    return list(OrderedDict(((item.kind, item.value, item.reason), item) for item in items).values())


def _task_title(style: str) -> str:
    return {
        "post_global_plugin": "Recover native semantics after global optimization",
        "structured_text_isa": "Stage target-specific schedule and ISA export",
        "runtime_hal": "Define the runtime/HAL backend contract",
        "llvm_ukernel": "Define ISA exposure, ukernel, and tile-model work",
    }[style]


def _task_phase(style: str) -> str:
    return {
        "post_global_plugin": "compiler",
        "structured_text_isa": "compiler",
        "runtime_hal": "runtime",
        "llvm_ukernel": "compiler",
    }[style]


def _task_actions(style: str) -> list[str]:
    if style == "post_global_plugin":
        return [
            "Use the post-global-optimization seam to recover accelerator "
            "semantics from normalized linalg/tensor/arithmetic IR.",
            "Define or refine the target dialect, recovery passes, and "
            "fallback boundaries for non-native operations.",
        ]
    if style == "structured_text_isa":
        return [
            "Define the staged kernel, schedule, and ISA export pipeline " "required by the target.",
            "Attach verifier and memory-planning responsibilities to explicit "
            "stages instead of mixing them into late codegen.",
        ]
    if style == "runtime_hal":
        return [
            "Specify the driver registration, device model, transport/backend "
            "seam, and executable contract for the runtime target.",
            "Separate queue, synchronization, and dispatch responsibilities " "from compiler-only work items.",
        ]
    return [
        "Translate ISA exposure into the required LLVM intrinsics, backend "
        "features, instruction patterns, and ukernel tile decisions.",
        "Define layout, packing, unpacking, and datatype legality work from "
        "the canonical hardware geometry and memory model.",
    ]


def _task_write_scope(style: str) -> list[str]:
    if style == "post_global_plugin":
        return [
            "compiler/plugins/target/<Target>/",
            "compiler/src/merlin/Dialect/<Target>/",
        ]
    if style == "structured_text_isa":
        return [
            "compiler/src/merlin/Dialect/<Target>/",
            "compiler/src/merlin/Translation/<Target>/",
        ]
    if style == "runtime_hal":
        return [
            "runtime/src/iree/hal/drivers/<target>/",
            "runtime/src/iree/hal/drivers/<target>/registration/",
        ]
    return [
        "third_party/iree_bar/compiler/src/iree/compiler/Codegen/",
        "third_party/iree_bar/runtime/src/iree/builtins/ukernel/",
        "third_party/iree_bar/third_party/llvm-project/llvm/lib/Target/RISCV/",
        "third_party/iree_bar/third_party/llvm-project/llvm/include/llvm/IR/",
    ]


def _task_acceptance(capabilities: TargetCapabilities, style: str) -> list[str]:
    if style == "post_global_plugin":
        return [
            "Recovered target ops appear immediately after the global optimization pipeline on representative models.",
            "Unsupported patterns fall back cleanly to the generic downstream flow.",
        ]
    if style == "structured_text_isa":
        return [
            "Kernel, schedule, and ISA stages are individually verifiable.",
            "The exported ISA form remains compatible with the declared simulator or text-ISA consumer.",
        ]
    if style == "runtime_hal":
        return [
            "Driver registration and device creation are validated independently from dispatch execution.",
            "Runtime responsibilities are explicit enough to generate a smoke-test ladder.",
        ]
    checks = [
        "Lowering strategy and tile selection are explainable from the canonical geometry and memory model.",
        "LLVM exposure matches the declared ISA state model and register constraints.",
    ]
    if capabilities.isa.register_constraints.even_alignment_required:
        checks.append("Legality checks cover even-alignment or grouped-register constraints.")
    return checks


def _execution_contract(
    capabilities: TargetCapabilities,
    *,
    family: str,
    phase: str,
    task_id: str,
    write_scope: list[str],
) -> dict[str, object]:
    repo_root = _task_repo_root(capabilities, family)
    execution_adapter = _task_execution_adapter(capabilities, family)
    return {
        "repo_root": repo_root,
        "execution_adapter": execution_adapter,
        "mutation_policy": _task_mutation_policy(family, execution_adapter),
        "artifacts_in": _task_artifacts_in(capabilities),
        "artifacts_out": [*write_scope, f"build/generated/targetgen/<target>/task_states/{task_id}.json"],
        "validation_commands": _task_validation_commands(capabilities, family, phase),
        "credential_requirements": _task_credential_requirements(capabilities, family),
        "handoff_contract": _task_handoff_contract(family),
    }


def _task_repo_root(capabilities: TargetCapabilities, family: str) -> str:
    if family == "llvm_ukernel":
        if capabilities.isa.exposure.needs_llvm_backend_changes:
            return "third_party/iree_bar/third_party/llvm-project"
        return "third_party/iree_bar"
    return "."


def _task_execution_adapter(capabilities: TargetCapabilities, family: str) -> str:
    if family == "runtime_hal":
        return "runtime_hal"
    if family == "llvm_ukernel":
        if capabilities.isa.exposure.needs_llvm_backend_changes:
            return "llvm_submodule"
        return "iree_submodule"
    return "merlin_local"


def _task_mutation_policy(family: str, execution_adapter: str) -> str:
    if family == "shared":
        return "planner_generated_only"
    if execution_adapter == "llvm_submodule":
        return "llvm_submodule_edit"
    if execution_adapter == "iree_submodule":
        return "iree_submodule_edit"
    if execution_adapter == "runtime_hal":
        return "runtime_driver_edit"
    return "merlin_local_edit"


def _task_artifacts_in(capabilities: TargetCapabilities) -> list[str]:
    artifacts = [capabilities.capability_path]
    if capabilities.deployment is not None:
        artifacts.append(capabilities.deployment.source_path)
    return artifacts


def _task_validation_commands(
    capabilities: TargetCapabilities,
    family: str,
    phase: str,
) -> list[str]:
    commands = [
        "conda run -n merlin-dev uv run pytest tests/targetgen/test_targetgen.py",
    ]
    if phase == "planning":
        commands.append(
            "conda run -n merlin-dev uv run tools/merlin.py targetgen plan <capability> [--overlay <overlay>]"
        )
    if family in {"post_global_plugin", "structured_text_isa", "llvm_ukernel"}:
        commands.append("conda run -n merlin-dev uv run tools/merlin.py build --profile vanilla --cmake-target install")
    if family == "runtime_hal":
        commands.append("conda run -n merlin-dev uv run tools/merlin.py build --profile vanilla --cmake-target install")
        for gate in capabilities.access.verification_gates:
            commands.append(f"# runtime verification gate: {gate}")
    return commands


def _task_credential_requirements(capabilities: TargetCapabilities, family: str) -> str:
    if family == "runtime_hal" or capabilities.access.model == "cloud_service":
        return capabilities.access.credential_requirements
    return "none"


def _task_handoff_contract(family: str) -> list[str]:
    contract = [
        "Summarize the evidence reviewed before any mutation.",
        "Report the exact files touched or confirmed unchanged.",
        "Record validation results or blockers before marking the task complete.",
    ]
    if family in {"post_global_plugin", "structured_text_isa"}:
        contract.append("Call out dialect, pass, or exporter seams affected by the change.")
    if family == "llvm_ukernel":
        contract.append("Call out LLVM, IREE, and ukernel surfaces that must stay in sync.")
    if family == "runtime_hal":
        contract.append("Call out device, transport, executable, and synchronization responsibilities separately.")
    return contract
