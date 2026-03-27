from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import targetgen_cmd  # noqa: E402
from raycp import service as ray_service  # noqa: E402
from targetgen import (  # noqa: E402
    build_execution_bundle,
    build_execution_state,
    build_support_plan,
    build_task_graph,
    load_capability_spec,
    load_deployment_overlay,
    load_provider_config,
    render_prompt_packets,
)

EXAMPLES = [
    (
        "spacemit_x60_xsmtvdot",
        "board_linux",
        "llvm_ukernel",
        {"rvv_cpu_extension"},
        {"llvm_ukernel"},
    ),
    (
        "saturn_opu_v128",
        "firesim_u250",
        "llvm_ukernel",
        {"riscv_vendor_matrix_extension"},
        {"llvm_ukernel"},
    ),
    (
        "gemmini_mx",
        "firesim_u250",
        "post_global_plugin",
        {"rocc_accelerator"},
        {"post_global_plugin", "llvm_ukernel"},
    ),
    (
        "npu_ucb",
        "simulator_local",
        "structured_text_isa",
        {"structured_npu_text_isa"},
        {"post_global_plugin", "structured_text_isa"},
    ),
    (
        "radiance_muon",
        "runtime_driver_local",
        "runtime_hal",
        {"simt_gpu_hal"},
        {"runtime_hal"},
    ),
]

EXTERNAL_EXAMPLES = [
    (
        "nvidia_vulkan_ada",
        "desktop_local",
        "runtime_hal",
        {"simt_gpu_hal"},
        {"runtime_hal"},
    ),
    (
        "qualcomm_adreno_vulkan",
        "android_lab_device",
        "runtime_hal",
        {"simt_gpu_hal"},
        {"runtime_hal"},
    ),
    (
        "nvidia_cuda_sm89",
        "desktop_local",
        "runtime_hal",
        {"simt_gpu_hal"},
        {"runtime_hal"},
    ),
    (
        "qualcomm_qnn_htp_snapdragon8elite",
        "ai_hub_cloud",
        "runtime_hal",
        {"mixed_cpu_accelerator"},
        {"runtime_hal"},
    ),
    (
        "ara_rvv_vlen256",
        "simulator_local",
        "llvm_ukernel",
        {"rvv_cpu_extension"},
        {"llvm_ukernel"},
    ),
    (
        "vortex_gpgpu_u250",
        "fpga_local",
        "runtime_hal",
        {"simt_gpu_hal"},
        {"runtime_hal", "structured_text_isa"},
    ),
]


@pytest.mark.parametrize(
    ("target_name", "overlay_name", "primary_integration", "families", "integrations"),
    EXAMPLES,
)
def test_example_specs_classify(
    target_name,
    overlay_name,
    primary_integration,
    families,
    integrations,
):
    capabilities = _load_example(target_name, overlay_name)
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    bundle = build_execution_bundle(capabilities, support_plan, task_graph)

    assert support_plan.primary_integration == primary_integration
    assert families.issubset(set(support_plan.target_families))
    assert integrations.issubset(set(support_plan.integration_styles))
    assert support_plan.task_node_ids
    assert bundle.tasks
    assert any(
        stage.name == "spec_validation" and stage.status == "ready"
        for stage in support_plan.verification_manifest.stages
    )


def test_invalid_capability_spec_missing_target_name(tmp_path: Path):
    bad_spec = tmp_path / "bad.yaml"
    bad_spec.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "target:",
                "  display_name: Missing Name",
                "  vendor: Example",
                "  maturity: experimental",
                "platform:",
                "  host_isa: riscv64",
                "  operating_systems: [linux]",
                "  environments: [board]",
                "execution_model:",
                "  kind: vector_cpu_extension",
                "  attachment: cpu_extension",
                "  submission_model: inline_cpu_instructions",
                "  compiler_recovery_stage: llvmcpu_codegen",
                "isa:",
                "  base: riscv64",
                "  features: [+v]",
                "  exposure:",
                "    kind: llvm_intrinsics",
                "    needs_llvm_backend_changes: true",
                "    needs_new_intrinsics: true",
                "    needs_new_feature_bits: true",
                "  state_model:",
                "    kind: explicit_ssa",
                "  register_constraints:",
                "    even_alignment_required: false",
                "    vl_dependent: true",
                "operations:",
                "  compute:",
                "    - name: matmul",
                "      native: true",
                "  movement: [dma_in]",
                "  synchronization: [fence]",
                "geometry:",
                "  compute_array: {kind: vector}",
                "  vector: {vlen_bits: 256, dlen_bits: 256}",
                "  native_tile_options: [[8, 8, 8]]",
                "  preferred_tiles: [[8, 8, 8]]",
                "memory:",
                "  spaces: [{name: global, kind: dram}]",
                "  preferred_layouts: {lhs: row_major, rhs: row_major, result: row_major}",
                "  packing: {requires_pack: false, requires_unpack: false}",
                "numeric:",
                "  legal_type_triples: [i8*i8->i32]",
                "  quantization: {}",
                "  rounding_modes: [rne]",
                "  saturation: true",
                "runtime:",
                "  required: false",
                "  executable_format: llvm_cpu_vmfb",
                "  driver: {backend_id: null, uri_schemes: []}",
                "  synchronization: {semaphores: false, timelines: false}",
                "verification:",
                "  golden_model: {available: true, kind: cpu_reference}",
                "  simulator: {available: false, kind: none}",
                "  rtl: {available: false, kind: none}",
                "  perf_counters: {available: false, kind: none}",
                "  tolerances: {}",
                "access:",
                "  model: local_device",
                "  sdk_requirements: [iree_llvmcpu]",
                "  credential_requirements: none",
                "  availability_class: sdk_install_required",
                "  verification_gates: [compile_only]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target.name"):
        load_capability_spec(bad_spec)


def test_cli_plan_and_orchestrate_write_outputs(tmp_path: Path):
    parser = argparse.ArgumentParser()
    targetgen_cmd.setup_parser(parser)
    cap_path = REPO_ROOT / "target_specs" / "examples" / "saturn_opu_v128" / "capability.yaml"
    overlay_path = REPO_ROOT / "target_specs" / "examples" / "saturn_opu_v128" / "overlays" / "firesim_u250.yaml"

    plan_args = parser.parse_args(
        [
            "plan",
            str(cap_path),
            "--overlay",
            str(overlay_path),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert targetgen_cmd.main(plan_args) == 0

    target_dir = tmp_path / "saturn_opu_v128"
    assert (target_dir / "support_plan.json").exists()
    assert (target_dir / "task_graph.json").exists()
    assert (target_dir / "verification_manifest.json").exists()

    orchestrate_args = parser.parse_args(
        [
            "orchestrate",
            str(cap_path),
            "--overlay",
            str(overlay_path),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert targetgen_cmd.main(orchestrate_args) == 0
    assert (target_dir / "execution_bundle.json").exists()
    assert (target_dir / "execution_state.json").exists()
    briefs = sorted((target_dir / "briefs").glob("*.md"))
    assert briefs
    prompt_packets = sorted((target_dir / "prompts").glob("prompt_*.md"))
    assert prompt_packets
    task_states = sorted((target_dir / "task_states").glob("*.json"))
    assert task_states

    bundle = json.loads((target_dir / "execution_bundle.json").read_text(encoding="utf-8"))
    execution_state = json.loads((target_dir / "execution_state.json").read_text(encoding="utf-8"))
    assert bundle["workflow"] == "evidence_first_target_enablement"
    assert bundle["prompt_backend"] == "manualllm"
    assert bundle["tasks"][0]["prompt_packet"] == "prompt_001.md"
    assert execution_state["task_order"]
    assert execution_state["tasks"]["review_target_spec"]["status"] == "planned"


def test_prompt_bundle_resolves_sections_and_tool_profiles():
    capabilities = _load_example("npu_ucb", "simulator_local")
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    bundle = build_execution_bundle(capabilities, support_plan, task_graph)

    assert bundle.prompt_backend == "manualllm"
    plugin_task = next(task for task in bundle.tasks if task.family == "post_global_plugin")
    isa_task = next(task for task in bundle.tasks if task.family == "structured_text_isa")
    assert plugin_task.tool_profile == "mlir_agent"
    assert isa_task.tool_profile == "mlir_agent"
    section_names = [section.name for section in plugin_task.prompt_bundle.sections]
    assert "goal" in section_names
    assert "response_contract" in section_names
    assert "kernel, schedule, and ISA layers distinct" in isa_task.prompt


def test_target_and_overlay_prompt_overrides_apply():
    capabilities = _load_example("saturn_opu_v128", "firesim_u250")
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    bundle = build_execution_bundle(capabilities, support_plan, task_graph)
    review_task = next(task for task in bundle.tasks if task.id == "review_target_spec")

    assert "implicit machine-state assumptions explicit" in review_task.prompt
    assert "FireSim on U250" in review_task.prompt


def test_provider_backend_uses_mlir_agent_configs(tmp_path: Path):
    parser = argparse.ArgumentParser()
    targetgen_cmd.setup_parser(parser)
    cap_path = REPO_ROOT / "target_specs" / "examples" / "radiance_muon" / "capability.yaml"
    overlay_path = REPO_ROOT / "target_specs" / "examples" / "radiance_muon" / "overlays" / "runtime_driver_local.yaml"
    args = parser.parse_args(
        [
            "orchestrate",
            str(cap_path),
            "--overlay",
            str(overlay_path),
            "--out-dir",
            str(tmp_path),
            "--prompt-backend",
            "provider",
            "--agent",
            "codex",
        ]
    )

    assert targetgen_cmd.main(args) == 0
    target_dir = tmp_path / "radiance_muon"
    provider_backend = json.loads((target_dir / "provider_backend.json").read_text(encoding="utf-8"))
    assert provider_backend["model"]
    assert provider_backend["api_key_env"]


def test_render_prompt_packets_assigns_manual_llm_filenames():
    capabilities = _load_example("gemmini_mx", "firesim_u250")
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    bundle = build_execution_bundle(capabilities, support_plan, task_graph)
    prompt_packets = render_prompt_packets(bundle)

    assert "prompt_001.md" in prompt_packets
    assert bundle.tasks[0].prompt_packet == "prompt_001.md"
    assert bundle.tasks[0].response_packet == "prompt_001.response.md"
    assert "Expected Response File" in prompt_packets["prompt_001.md"]


def test_load_provider_config_reuses_mlir_agent_agent_configs():
    config = load_provider_config("codex")

    assert config["model"]
    assert config["api_key_env"]


def test_execution_bundle_includes_agentic_task_metadata():
    capabilities = _load_example("saturn_opu_v128", "firesim_u250")
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    bundle = build_execution_bundle(capabilities, support_plan, task_graph)
    execution_state = build_execution_state(bundle)
    llvm_task = next(task for task in bundle.tasks if task.family == "llvm_ukernel")

    assert llvm_task.execution_adapter == "llvm_submodule"
    assert llvm_task.repo_root == "third_party/iree_bar/third_party/llvm-project"
    assert llvm_task.execution_state.status == "planned"
    assert execution_state.tasks["implement_llvm_ukernel"].execution_adapter == "llvm_submodule"


def test_cli_execute_status_and_answer_runtime_gate(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    parser = argparse.ArgumentParser()
    targetgen_cmd.setup_parser(parser)
    cap_path = REPO_ROOT / "target_specs" / "examples" / "nvidia_vulkan_ada" / "capability.yaml"
    overlay_path = REPO_ROOT / "target_specs" / "examples" / "nvidia_vulkan_ada" / "overlays" / "desktop_local.yaml"
    target_dir = tmp_path / "nvidia_vulkan_ada"

    execute_args = parser.parse_args(
        [
            "execute",
            str(cap_path),
            "--overlay",
            str(overlay_path),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert targetgen_cmd.main(execute_args) == 0

    execution_state = json.loads((target_dir / "execution_state.json").read_text(encoding="utf-8"))
    assert execution_state["current_stage"] == "device"
    assert execution_state["open_question_id"]
    request_path = target_dir / "operator_requests" / f"{execution_state['open_question_id']}.json"
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert request_payload["stage"] == "device"
    assert request_payload["status"] == "open"

    status_args = parser.parse_args(
        [
            "status",
            "--target-dir",
            str(target_dir),
        ]
    )
    assert targetgen_cmd.main(status_args) == 0
    status_output = capsys.readouterr().out
    assert "Current stage: device" in status_output
    assert "Operator requests:" in status_output

    answer_args = parser.parse_args(
        [
            "answer",
            "--target-dir",
            str(target_dir),
            "--question-id",
            request_payload["id"],
            "--choice",
            "device_available",
        ]
    )
    assert targetgen_cmd.main(answer_args) == 0
    answered_payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert answered_payload["status"] == "answered"
    assert answered_payload["selected_option"] == "device_available"


def test_cli_execute_resume_from_dir_ingests_response_and_hits_branch_gate(tmp_path: Path):
    parser = argparse.ArgumentParser()
    targetgen_cmd.setup_parser(parser)
    cap_path = REPO_ROOT / "target_specs" / "examples" / "nvidia_vulkan_ada" / "capability.yaml"
    overlay_path = REPO_ROOT / "target_specs" / "examples" / "nvidia_vulkan_ada" / "overlays" / "desktop_local.yaml"
    target_dir = _prepare_runtime_hal_prompt_state(
        parser=parser,
        tmp_path=tmp_path,
        capability_path=cap_path,
        overlay_path=overlay_path,
    )

    resume_args = parser.parse_args(
        [
            "execute",
            "--from-dir",
            str(target_dir),
            "--resume",
        ]
    )
    assert targetgen_cmd.main(resume_args) == 0

    execution_state = json.loads((target_dir / "execution_state.json").read_text(encoding="utf-8"))
    awaiting_response = [
        task_id for task_id, payload in execution_state["tasks"].items() if payload["status"] == "awaiting_response"
    ]
    assert awaiting_response
    bundle_payload = json.loads((target_dir / "execution_bundle.json").read_text(encoding="utf-8"))
    task_payload = next(task for task in bundle_payload["tasks"] if task["id"] == awaiting_response[0])
    response_path = target_dir / "prompts" / task_payload["response_packet"]
    response_path.write_text(
        "# Proposal\n\n- Reviewed the runtime HAL seams.\n- Keep execution paused until mutation is approved.\n",
        encoding="utf-8",
    )

    assert targetgen_cmd.main(resume_args) == 0
    execution_state = json.loads((target_dir / "execution_state.json").read_text(encoding="utf-8"))
    assert execution_state["current_stage"] == "branch_gate"
    assert execution_state["open_question_id"]
    branch_request = json.loads(
        (target_dir / "operator_requests" / f"{execution_state['open_question_id']}.json").read_text(encoding="utf-8")
    )
    assert branch_request["stage"] == "branch_gate"
    assert branch_request["status"] == "open"
    task_state = execution_state["tasks"][task_payload["id"]]
    assert task_state["proposal_summary"]
    assert task_state["response_file"] == str(response_path)


def test_cli_execute_provider_backend_pauses_with_provider_request(tmp_path: Path):
    parser = argparse.ArgumentParser()
    targetgen_cmd.setup_parser(parser)
    cap_path = REPO_ROOT / "target_specs" / "examples" / "radiance_muon" / "capability.yaml"
    overlay_path = REPO_ROOT / "target_specs" / "examples" / "radiance_muon" / "overlays" / "runtime_driver_local.yaml"
    target_dir = tmp_path / "radiance_muon"

    execute_args = parser.parse_args(
        [
            "execute",
            str(cap_path),
            "--overlay",
            str(overlay_path),
            "--out-dir",
            str(tmp_path),
            "--prompt-backend",
            "provider",
            "--agent",
            "codex",
        ]
    )
    assert targetgen_cmd.main(execute_args) == 0

    execution_state = json.loads((target_dir / "execution_state.json").read_text(encoding="utf-8"))
    assert execution_state["current_stage"] == "provider_disabled"
    request_payload = json.loads(
        (target_dir / "operator_requests" / f"{execution_state['open_question_id']}.json").read_text(encoding="utf-8")
    )
    assert request_payload["stage"] == "provider_disabled"
    assert request_payload["recommended_option"] == "switch_to_manualllm"


def test_cli_execute_ray_engine_creates_merlin_run_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    parser = argparse.ArgumentParser()
    targetgen_cmd.setup_parser(parser)
    monkeypatch.setattr(ray_service.shutil, "which", lambda _: None)
    cap_path = REPO_ROOT / "target_specs" / "examples" / "nvidia_vulkan_ada" / "capability.yaml"
    overlay_path = REPO_ROOT / "target_specs" / "examples" / "nvidia_vulkan_ada" / "overlays" / "desktop_local.yaml"
    ray_state_root = tmp_path / "ray-state"
    target_dir = tmp_path / "nvidia_vulkan_ada"

    execute_args = parser.parse_args(
        [
            "execute",
            str(cap_path),
            "--overlay",
            str(overlay_path),
            "--out-dir",
            str(tmp_path),
            "--engine",
            "ray",
            "--ray-state-root",
            str(ray_state_root),
        ]
    )

    assert targetgen_cmd.main(execute_args) == 0
    assert (target_dir / "execution_bundle.json").exists()
    assert (target_dir / "execution_state.json").exists()
    run_dirs = sorted((ray_state_root / "runs").glob("*"))
    assert run_dirs
    run_record = json.loads((run_dirs[0] / "run_record.json").read_text(encoding="utf-8"))
    assert run_record["target"] == "nvidia_vulkan_ada"
    assert run_record["workflow"] == "targetgen_execute"
    assert run_record["status"] == "blocked"
    assert "--engine" in run_record["command"]
    assert "local" in run_record["command"]


@pytest.mark.parametrize(
    ("target_name", "overlay_name", "primary_integration", "families", "integrations"),
    EXTERNAL_EXAMPLES,
)
def test_external_example_specs_classify(
    target_name,
    overlay_name,
    primary_integration,
    families,
    integrations,
):
    capabilities = _load_example(target_name, overlay_name)
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    bundle = build_execution_bundle(capabilities, support_plan, task_graph)

    assert support_plan.primary_integration == primary_integration
    assert families.issubset(set(support_plan.target_families))
    assert integrations.issubset(set(support_plan.integration_styles))
    assert capabilities.access.model
    assert capabilities.access.verification_gates
    assert bundle.tasks


def test_cloud_service_target_requires_credentials_in_runtime_tasks():
    capabilities = _load_example("qualcomm_qnn_htp_snapdragon8elite", "ai_hub_cloud")
    support_plan = build_support_plan(capabilities)
    task_graph = build_task_graph(capabilities, support_plan.integration_styles)
    runtime_task = next(task for task in task_graph if task.family == "runtime_hal")

    assert runtime_task.credential_requirements == "aihub_api_token"


def _load_example(target_name: str, overlay_name: str):
    base = REPO_ROOT / "target_specs" / "examples" / target_name
    capabilities = load_capability_spec(base / "capability.yaml")
    capabilities.deployment = load_deployment_overlay(base / "overlays" / f"{overlay_name}.yaml")
    return capabilities


def _prepare_runtime_hal_prompt_state(
    *,
    parser: argparse.ArgumentParser,
    tmp_path: Path,
    capability_path: Path,
    overlay_path: Path,
) -> Path:
    target_dir = tmp_path / capability_path.parent.name
    execute_args = parser.parse_args(
        [
            "execute",
            str(capability_path),
            "--overlay",
            str(overlay_path),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert targetgen_cmd.main(execute_args) == 0
    state_payload = json.loads((target_dir / "execution_state.json").read_text(encoding="utf-8"))
    question_id = state_payload["open_question_id"]
    answer_args = parser.parse_args(
        [
            "answer",
            "--target-dir",
            str(target_dir),
            "--question-id",
            question_id,
            "--choice",
            "device_available",
        ]
    )
    assert targetgen_cmd.main(answer_args) == 0
    return target_dir
