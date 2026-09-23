"""Historical V2 roster frozen from 50f5cba1d; never recaptured through a current builder."""

from merlin_experiments.phase2 import contracts as C

V2_IDENTITIES = (
    "controller/global",
    "python/merlin.targetgen.sandbox.toolchain",
    "python/merlin.common.access",
    "python/merlin.common.digest",
    "python/merlin_experiments.source_snapshot",
    "python/merlin.targetgen.sandbox.answer_surfaces",
    "python/merlin.targetgen.sandbox.bwrap",
    "python/merlin.targetgen.sandbox.build_dependencies",
    "python/merlin.targetgen.sandbox.executable_dependencies",
    "python/merlin.targetgen.contract.build_service",
    "python/merlin.targetgen.contract.build_recipe",
    "python/merlin.perf.compiler_plan_evidence",
    "python/merlin.perf.task_cfg_evidence",
    "python/merlin.perf.task_instruction_evidence",
    "python/merlin.perf.task_route_presence",
    "python/merlin.perf.storage_encoding",
    "python/merlin.perf.structural_transitions",
    "python/merlin.perf.physical_transition_evidence",
    "python/merlin.perf.external_objective",
    "python/merlin.perf.model_placement",
    "python/merlin.perf.model_macs",
    "python/merlin.runtime.storage_binding",
    "python/merlin.runtime.prepack_authority",
    "python/merlin.runtime.captured_constants",
    "python/merlin.frontends.argument_identity",
    "python/merlin.perf.host_cfg_activity",
    "python/merlin.perf.analysis_worker",
    "python/merlin.perf.isolated_probe_provider",
    "python/merlin.perf.primitive_probe",
    "python/merlin.perf.instruction_motif",
    "python/merlin.perf.probe_relevance",
    "python/merlin.perf.structural_delta",
    "python/merlin.perf.completion_delta",
    "python/merlin.perf.context_probe",
    "python/merlin.perf.context_program",
    "python/merlin.perf.controlled_context_provider",
    "python/merlin.perf.fixed_work_context",
    "python/merlin.perf.paired_context_provider",
    "python/merlin.perf.static_imports",
    "python/merlin.perf.compiler_edit_scope",
    "python/merlin.perf.agent_guidance",
    "python/merlin.perf.agent_guidance_vocabulary",
    "python/merlin.kernels.cca_contract",
    "python/merlin.perf.host_region_qualifier",
    "python/merlin.perf.host_source_witness",
    "python/merlin.perf.source_convolution_witness",
    "python/merlin.perf.source_convolution_preparation",
    "python/merlin.perf.source_program_pair",
    "python/merlin.perf.source_contraction_witness",
    "python/merlin.perf.source_contraction_preparation",
    "python/merlin.perf.source_program_pair_provider",
    "python/merlin.perf.source_initializer_elision",
    "python/merlin.targetgen.conv_geometry",
    "python/merlin.targetgen.capsule_golden",
    "python/merlin.runtime.commandbuffer",
    "python/merlin.runtime.tensor",
    "python/merlin.perf.host_physical_transition_qualifier",
    "python/merlin.perf.native_host_witness_runner",
    "python/merlin.perf.mechanism_probe",
    "python/merlin.perf.historical_reference",
    "python/merlin.perf.harvest",
    "python/merlin.perf.work_volume",
    "python/merlin.perf.execution_policy",
    "python/merlin.targetgen.rocc.decode",
    "python/merlin.common.source_membership",
    "resource/contract/gate_phases.yaml",
    "resource/contract/hardware_pins.yaml",
    "resource/contract/schemas/manifest.schema.json",
    "resource/contract/schemas/command_buffer.schema.json",
    "python/merlin_experiments.phase2",
)
V2_CONTENT_SHA256 = "f02dfa9715518aacbbf36b1e5b86d0b8de8515bd98d05b298fd5677f524886ec"


def historical_v2(root):
    identities, sources = {}, {}
    for index, identity in enumerate(V2_IDENTITIES):
        path = root / ("phase2/__init__.py" if identity == "python/merlin_experiments.phase2" else f"owners/{index}.py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("# historical v2 fixture: " + identity + "\n").encode())
        identities[identity] = str(path)
        sources[str(path)] = C.sha256_file(path)
    record = {
        "schema": "global_host_verification_policy_v2",
        "identities": identities,
        "sources": sources,
        "closures": {
            "merlin_experiments.phase2": {
                "root": str(root / "phase2"),
                "members": {"__init__.py": "python/merlin_experiments.phase2"},
            }
        },
        "sha256": V2_CONTENT_SHA256,
    }
    record["location_sha256"] = C.document_sha256({key: record[key] for key in ("sources", "identities", "closures")})
    return record
