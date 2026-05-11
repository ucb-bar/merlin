from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TargetIdentity:
    name: str
    display_name: str
    vendor: str
    maturity: str


@dataclass(slots=True)
class PlatformCapabilities:
    host_isa: str
    operating_systems: list[str]
    environments: list[str]


@dataclass(slots=True)
class ExecutionModel:
    kind: str
    attachment: str
    submission_model: str
    compiler_recovery_stage: str


@dataclass(slots=True)
class ISAExposure:
    kind: str
    needs_llvm_backend_changes: bool
    needs_new_intrinsics: bool
    needs_new_feature_bits: bool


@dataclass(slots=True)
class RegisterConstraints:
    accumulator_register_file: str | None
    even_alignment_required: bool
    vl_dependent: bool


@dataclass(slots=True)
class ISAContract:
    base: str
    features: list[str]
    exposure: ISAExposure
    state_model: str
    register_constraints: RegisterConstraints


@dataclass(slots=True)
class OperationSpec:
    name: str
    native: bool
    type_triples: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OperationCoverage:
    compute: list[OperationSpec]
    movement: list[str]
    synchronization: list[str]


@dataclass(slots=True)
class TileModel:
    compute_array_kind: str | None
    vector_vlen_bits: int | None
    vector_dlen_bits: int | None
    native_tile_options: list[list[int]]
    preferred_tiles: list[list[int]]


@dataclass(slots=True)
class MemoryModel:
    spaces: list[dict[str, Any]]
    preferred_layouts: dict[str, str]
    packing: dict[str, Any]


@dataclass(slots=True)
class NumericContract:
    legal_type_triples: list[str]
    quantization: dict[str, Any]
    rounding_modes: list[str]
    saturation: bool


@dataclass(slots=True)
class RuntimeContract:
    required: bool
    executable_format: str
    driver_backend_id: str | None
    uri_schemes: list[str]
    synchronization: dict[str, Any]


@dataclass(slots=True)
class VerificationContract:
    golden_model: dict[str, Any]
    simulator: dict[str, Any]
    rtl: dict[str, Any]
    perf_counters: dict[str, Any]
    tolerances: dict[str, Any]


@dataclass(slots=True)
class AccessModel:
    model: str
    sdk_requirements: list[str]
    credential_requirements: str
    availability_class: str
    verification_gates: list[str]


@dataclass(slots=True)
class DeploymentProfile:
    name: str
    mode: str
    build_profile: str
    compile_target: str | None
    compile_hw: str | None
    hardware_recipe: str | None
    chipyard: dict[str, Any]
    runtime: dict[str, Any]
    extra: dict[str, Any]
    source_path: str


@dataclass(slots=True)
class TargetCapabilities:
    schema_version: int
    identity: TargetIdentity
    platform: PlatformCapabilities
    execution: ExecutionModel
    isa: ISAContract
    operations: OperationCoverage
    tiles: TileModel
    memory: MemoryModel
    numeric: NumericContract
    runtime: RuntimeContract
    verification: VerificationContract
    access: AccessModel
    references: list[str]
    capability_path: str
    deployment: DeploymentProfile | None = None


@dataclass(slots=True)
class EvidenceItem:
    kind: str
    value: str
    reason: str


@dataclass(slots=True)
class PatchSurface:
    surface: str
    reasons: list[str]
    execution_mode: str


@dataclass(slots=True)
class TaskNode:
    id: str
    title: str
    phase: str
    family: str
    depends_on: list[str]
    evidence: list[EvidenceItem]
    actions: list[str]
    write_scope: list[str]
    acceptance_checks: list[str]
    repo_root: str
    execution_adapter: str
    mutation_policy: str
    artifacts_in: list[str]
    artifacts_out: list[str]
    validation_commands: list[str]
    credential_requirements: str
    handoff_contract: list[str]


@dataclass(slots=True)
class VerificationStage:
    level: int
    name: str
    status: str
    rationale: str


@dataclass(slots=True)
class VerificationManifest:
    target: str
    maturity: str
    stages: list[VerificationStage]


@dataclass(slots=True)
class SupportPlan:
    schema_version: int
    target: str
    display_name: str
    vendor: str
    maturity: str
    target_families: list[str]
    integration_styles: list[str]
    primary_integration: str
    required_layers: list[str]
    derived_artifacts: list[str]
    patch_surfaces: list[PatchSurface]
    deployment_profile: str | None
    verification_manifest: VerificationManifest
    task_node_ids: list[str]


@dataclass(slots=True)
class PromptFragment:
    source: str
    scope: str
    section: str
    merge_mode: str


@dataclass(slots=True)
class PromptSection:
    name: str
    heading: str
    content: str
    sources: list[str]


@dataclass(slots=True)
class ResolvedPromptBundle:
    backend: str
    tool_profile: str
    agent: str | None
    sections: list[PromptSection]
    fragments: list[PromptFragment]
    flattened_prompt: str


@dataclass(slots=True)
class TaskExecutionState:
    status: str
    repo_root: str
    execution_adapter: str
    mutation_policy: str
    artifacts_in: list[str]
    artifacts_out: list[str]
    validation_commands: list[str]
    credential_requirements: str
    handoff_contract: list[str]
    current_stage: str | None = None
    attempt_count: int = 0
    opened_at: str | None = None
    updated_at: str | None = None
    open_question_id: str | None = None
    response_file: str | None = None
    proposal_summary: str | None = None
    branch_gate_required: bool = False
    live_mutation_enabled: bool = False
    logs: list[str] = field(default_factory=list)
    validation_results: list[str] = field(default_factory=list)
    diff_summary: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExecutionTask:
    id: str
    title: str
    phase: str
    family: str
    depends_on: list[str]
    tool_profile: str
    prompt: str
    prompt_bundle: ResolvedPromptBundle
    evidence: list[EvidenceItem]
    write_scope: list[str]
    acceptance_checks: list[str]
    repo_root: str
    execution_adapter: str
    mutation_policy: str
    artifacts_in: list[str]
    artifacts_out: list[str]
    validation_commands: list[str]
    credential_requirements: str
    handoff_contract: list[str]
    execution_state: TaskExecutionState
    prompt_packet: str | None = None
    response_packet: str | None = None


@dataclass(slots=True)
class ExecutionBundle:
    target: str
    workflow: str
    mode: str
    prompt_backend: str
    agent: str | None
    tasks: list[ExecutionTask]
    prompts_dir: str | None = None
    provider_config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionState:
    target: str
    workflow: str
    prompt_backend: str
    task_order: list[str]
    tasks: dict[str, TaskExecutionState]
    current_task: str | None = None
    current_stage: str | None = None
    live_mutation_enabled: bool = False
    open_question_id: str | None = None


@dataclass(slots=True)
class OperatorOption:
    id: str
    label: str
    description: str


@dataclass(slots=True)
class OperatorRequest:
    id: str
    target: str
    task_id: str
    stage: str
    question: str
    options: list[OperatorOption]
    recommended_option: str
    status: str
    selected_option: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class GeneratedScaffoldFile:
    repo_path: str
    staged_path: str
    category: str
    rationale: str
    task_ids: list[str]


@dataclass(slots=True)
class GenerationBundle:
    target: str
    primary_integration: str
    integration_styles: list[str]
    generated_root: str
    files: list[GeneratedScaffoldFile]


@dataclass(slots=True)
class MutationCandidate:
    repo_path: str
    staged_path: str
    action: str
    rationale: str
    task_ids: list[str]
    mutation_policy: str


@dataclass(slots=True)
class MutationBundle:
    target: str
    branch_name: str
    worktree_root: str
    generated_root: str
    proposed_root: str
    candidates: list[MutationCandidate]
    blocking_gates: list[str]
    next_steps: list[str]


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
