"""Opt-in complete genuine-capture public baseline; never restores an approval.

MERLIN_TEST_GEMMINI_PARAMETER_CHAIN_SELECTION names a closed operator JSON
accepted by parameter_chain_execution_inputs.read_selection. Run in a fresh pytest
process with the normal selected compiler tools. No capture/replay/CAS copy occurs.
This quiet test establishes complete functional outputs and end-of-call input
bytes only: dynamic accelerator counts, timing, resources and formal tiers remain
unknown/pending without their separate original owners.
"""

import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path

import numpy as np
import packing_execution_inputs as P
import parameter_chain_execution_inputs as E
import parameter_chain_reference as R
import pytest
import yaml

pytestmark = pytest.mark.target("gemmini")
SELECTION_ENV = "MERLIN_TEST_GEMMINI_PARAMETER_CHAIN_SELECTION"


def test_original_public_parameter_chain_baseline(tmp_path, monkeypatch):
    path = os.environ.get(SELECTION_ENV)
    if not path:
        pytest.skip(f"set {SELECTION_ENV}; no target/capture/admission qualification established")
    run_baseline(tmp_path, monkeypatch, E.read_selection(path))


def run_baseline(tmp_path, monkeypatch, selected):
    from merlin_experiments.phase0.capture_execution_attestation import require_verified_execution
    from merlin_experiments.phase1.feedback import private_instruction_coordinator as C
    from merlin_experiments.phase1.feedback import private_instruction_declaration as D

    from merlin.common import invocation_record as I
    from merlin.compile.baremetal_model import compile_saved_model, read_full_outputs
    from merlin.compile.model_execution_inputs import selected_execution_firrtl, strict_tree_sha256
    from merlin.compile.route_before_build import plan_before_build
    from merlin.llvmlower import optional_passes
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe
    from merlin.llvmlower.device_offload import SIDECAR_NAME
    from merlin.runtime.input_observation import revalidate_observation
    from merlin.targetgen import capsule_source, target_registry
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    execution = selected.execution
    quiet_path = Path(execution["quiet_contract"]["path"])
    quiet = yaml.safe_load(quiet_path.read_bytes())
    target, block = quiet["name"], quiet["runner"]["chipyard_rocc"]
    for key, value in dict(
        MERLIN_RTL_FACTS=execution["facts"]["path"],
        MERLIN_TARGET_CONTRACT=str(quiet_path),
        MERLIN_SPIKE=block["engines"]["spike"]["binary"]["path"],
        MERLIN_RISCV_GCC=block["toolchain"]["compiler"]["path"],
        MERLIN_OUT_ROOT=str(tmp_path / "out"),
    ).items():
        monkeypatch.setenv(key, value)
    controls = {
        key: os.environ.get(key)
        for key in (
            optional_passes.ENV,
            *(r.switch for r in optional_passes.REGISTRY if r.switch),
            "MERLIN_COMPILER_DRIVER",
            "MERLIN_COMPILER_PYTHON",
            "MERLIN_MLIR_INSTALL",
            "MERLIN_CLANG",
            "MERLIN_RTL_FACTS",
            "MERLIN_TARGET_CONTRACT",
            "MERLIN_SPIKE",
            "MERLIN_RISCV_GCC",
            "MERLIN_OUT_ROOT",
        )
    }

    def readers():
        return (
            readers,
            run_baseline,
            E.prepare,
            E.read_selection,
            E.recheck,
            E._file,
            R.public_reference,
            R.ordered_reference,
            R.exact_outputs,
            P.pin,
            P.canonical,
            P.actual_observation,
            P.recheck_artifacts,
            P.read_selection,
            P.recheck_selection,
            I.verify,
            compile_saved_model,
            read_full_outputs,
            selected_execution_firrtl,
            strict_tree_sha256,
            plan_before_build,
            verify_completed_recipe,
            revalidate_observation,
            capsule_source.model_op_demands_checked,
            require_verified_execution,
            C.prepare,
            D.read,
            D.Declaration.verify,
            C.PreparedInstructionSelection.verify,
            C.PreparedInstructionSelection.require_facts,
            C.PreparedInstructionSelection.record,
            LinkedElfAdmissionService.verify,
            LinkedElfAdmissionService._verify,
            LinkedElfAdmissionService.evaluate,
            LinkedElfAdmissionService.revalidate,
            LinkedElfAdmissionService._revalidate_report,
        )

    reader, reader_code, functions = readers, readers.__code__, readers()
    frozen = tuple((f, f.__code__) for f in functions)
    pins = [P.pin(Path(inspect.getsourcefile(f)).resolve()) for f in functions]
    intake = None
    callbacks = None
    result_snapshot = None
    package = tmp_path / "original-public-compiler"

    def current():
        assert readers is reader and reader.__code__ is reader_code and reader() == functions
        assert all(f.__code__ is code for f, code in frozen)
        P.recheck_artifacts(pins)
        E.recheck(selected)
        assert controls == {key: os.environ.get(key) for key in controls}
        if intake is not None:
            assert (intake.current, intake.current.__code__, intake.verify, intake.verify.__code__) == callbacks
            intake.current()
            intake.verify()
            intake.bound.verify(intake.bundle)
        if result_snapshot is not None:
            assert json.dumps(result, sort_keys=True) == result_snapshot
        if package.exists():
            assert strict_tree_sha256(package) == {"sha256": P.PUBLIC_COMPILER_SHA, "n_files": 16}

    current()
    intake = E.prepare(selected, tmp_path / "genuine-intake", monkeypatch)
    callbacks = intake.current, intake.current.__code__, intake.verify, intake.verify.__code__
    # The intake deliberately selects no support provider. Preserve ordinary pass defaults.
    assert os.environ["MERLIN_TARGET_PATH"] == str(tmp_path / "genuine-intake/no-providers")
    current()
    source = (intake.bundle / "model.mlir").read_text()
    n = R.geometry(source)
    demands = capsule_source.model_op_demands_checked(source, "i8", source_kind="captured_provenance_v1")
    contractions = [d for d in demands if d.family == "contraction"]
    assert len(demands) == 45 and len(contractions) == 3
    assert all(
        (d.op, d.m, d.n, d.k, d.in_fmt, d.weight_fmt) == ("int_matmul", 1, n, n, "i8", "i8") for d in contractions
    )
    current()
    prepared = C.prepare(
        D.read(Path(execution["instruction_declaration"]["path"]), target=target),
        target=target,
        output=tmp_path / "live-admission",
    )
    current()
    with target_registry.observed_contract(target, quiet, source_path=quiet_path):
        facts = selected_execution_firrtl(execution["facts"]["path"], target=target)
    binding = {"public_effective": facts, "private_input": facts}
    prepared.require_facts(binding)
    current()
    shutil.copytree(execution["public_compiler"]["path"], package)
    current()
    with target_registry.observed_contract(target, quiet, source_path=quiet_path):
        placement = plan_before_build(
            target,
            source,
            datapath="i8",
            device_package=str(package),
            capture=intake.bundle,
            granularity="contraction",
            linked_elf_admission=prepared.service,
            source_kind="captured_provenance_v1",
        )
        current()
        device = placement.get("device_routing")
        assert device is not None and device.linked_elf_admission is prepared.service
        result = compile_saved_model(
            capture=intake.bundle,
            package=execution["host_package"]["path"],
            board_catalog=execution["board_catalog"]["path"],
            board=execution["board"],
            dts=execution["dts"]["path"],
            output=tmp_path / "out/baseline",
            target=target,
            run="spike",
            arena_mb=16,
            timeout_s=300,
            readback="full",
            tolerance=None,
            respect_captured_quantization_scope=True,
            host_llvm_bitcode=True,
            readonly_input_indices=(0, 1, 2, 3),
            max_readonly_input_bytes=4 << 20,
            device=device,
            materialized_inputs=intake.bound,
        )
    result_snapshot = json.dumps(result, sort_keys=True)
    current()
    assert result["status"] == "verified_complete_output"
    elf = Path(result["output"]["elf"])
    elf_pin, recipe = P.pin(elf), P.pin(elf.parent / "compilation_recipe.json")
    verify_completed_recipe(Path(recipe["path"]), executable=elf, expected_recipe_sha256=recipe["sha256"])
    sidecar_pin = P.pin(elf.parent / SIDECAR_NAME)
    sidecar = json.loads(Path(sidecar_pin["path"]).read_bytes())
    assert sidecar["granularity"] == "contraction" and len(sidecar["routed"]) == 3
    assert len({r["source_operation_ordinal"] for r in sidecar["routed"]}) == 3
    assert all(
        (r["parallel"], r["reduction"], r["dtypes"]) == ([1, n], [n], ["i8", "i8", "i32"]) for r in sidecar["routed"]
    )
    policy_path, policy_call = P.actual_observation(elf.parent, "device_final_linked_elf_policy")
    admission = json.loads(Path(policy_call["stdout"]["path"]).read_bytes())
    assert policy_call["inputs"] == [elf_pin]
    assert prepared.service.revalidate(elf=elf, result=admission, target=target) == "accepted"
    quiet_path, quiet_call = P.actual_observation(elf.parent, "selected_rocc_process")
    assert quiet_call["inputs"] == [elf_pin] and quiet_call["returncode"] == 0
    assert policy_call["finished_ns"] <= quiet_call["started_ns"]
    observation = result["output"]["readonly_input_observation"]
    roster = revalidate_observation(observation["artifacts"])["arguments"]
    assert [r["original_index"] for r in roster] == [0, 1, 2, 3]
    assert [r["bytes"] for r in roster] == [n * n] * 3 + [n]
    originals = [intake.weights[f"w{i}"] for i in range(3)] + [intake.x]
    assert [a["sha256"] for a in observation["arguments"]] == [
        hashlib.sha256(a.tobytes()).hexdigest() for a in originals
    ]
    actual, _ = read_full_outputs(
        Path(quiet_call["stdout"]["path"]).read_bytes(),
        intake.bundle,
        result["output"]["build_hash"],
        readonly_input_observation=observation["artifacts"],
    )
    R.exact_outputs(actual, intake.expected)
    actual_path = tmp_path / "actual-outputs.npz"
    np.savez(actual_path, wide=actual[0], state=actual[1])
    actual_pin = P.pin(actual_path)
    prepared_record, materialized_record = prepared.record(), intake.bound.record()
    prepared.require_facts(binding)
    assert prepared.service.revalidate(elf=elf, result=admission, target=target) == "accepted"
    require_verified_execution(intake.attestation)
    current()
    revalidate_observation(observation["artifacts"])
    verify_completed_recipe(Path(recipe["path"]), executable=elf, expected_recipe_sha256=recipe["sha256"])
    P.recheck_artifacts(
        [*pins, elf_pin, recipe, sidecar_pin, actual_pin], invocations=[P.pin(policy_path), P.pin(quiet_path)]
    )
    report = dict(
        scope="one ordinary quiet full-source functional baseline, no formal admission or optimized candidate",
        selection=selected.pin,
        original_intake=intake.record,
        original_admission=prepared_record,
        materialization=materialized_record,
        source_kind=placement.get("source_kind", "captured_provenance_v1"),
        source=P.pin(intake.bundle / "model.mlir"),
        demand_count=len(demands),
        contractions=len(contractions),
        elf=elf_pin,
        recipe=recipe,
        sidecar=sidecar_pin,
        quiet_invocation=P.pin(quiet_path),
        policy_invocation=P.pin(policy_path),
        actual_outputs=actual_pin,
        output_values=sum(a.size for a in actual),
        readonly_bytes=sum(r["bytes"] for r in roster),
        readonly_observation=observation,
        host_artifacts=result["output"]["host_artifacts"],
        default_pass_controls=controls,
        compilation=result["compilation"],
        fixed_input_prefixes=intake.prefixes,
        dynamic_custom_executions="UNKNOWN; quiet only",
        cost="UNKNOWN",
        resource_admission="OPEN",
        formal_tiers={t: "PENDING_NOT_GRADED" for t in E.REQUIRED_TIERS},
    )
    (tmp_path / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
