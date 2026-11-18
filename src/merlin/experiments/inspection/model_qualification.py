"""Deterministic model checks against a selected OOT compiler or the existing native host route.

This evaluator never derives capsules or tunes a compiler. Compiler acceptance,
numerical execution and application workflow validation are separate observations.
Historical captures remain useful diagnostics, not newly certified source evidence.
Native-host-only checks preserve finite input scope and never stand in for accelerator or RVV execution.
"""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import yaml

from merlin.common.digest import hash_tree, sha256_file
from merlin.experiments.runs.execution import without_unsealed_board_catalog

# ---- model_qualification -----------------------------------------------------------------------


def _worker(request: dict, root: Path) -> dict:
    from merlin.experiments.inspection.qualification_workflow import (
        _command_observation,
        _compiler_check,
        _ir_observation,
        _json,
        _plain,
        _selected_source_certifier,
    )

    if request.get("native_host_only"):
        return _native_host_worker(request, root)
    from merlin.development.targetgen.packaging.loading import load_package

    package = load_package(request["package"])
    if package.target != request["target"]:
        raise ValueError("selected compiler package names a different target")
    workflow = request["workflow"]
    deadline = time.monotonic() + request["timeout_seconds"]
    accounting = {"status": "not_measured", "reason": "no immutable selected target evidence bundle supplied"}
    selection = None
    if request.get("evidence_bundle"):
        from merlin.development.targetgen.applications.demand_inventory import application_operation_inventory
        from merlin.development.targetgen.applications.operation_accounting import build_operation_accounting
        from merlin.development.testgen.evidence.evidence import load_exported_evidence

        selection = load_exported_evidence(request["evidence_bundle"])
        if selection.target != request["target"]:
            raise ValueError("selected evidence belongs to a different target")
        applications = {
            item["name"]: application_operation_inventory(
                Path(item["bundle"]) / "model.mlir",
                request["target"],
                capability_contract=selection.contract,
                workload_id=request["workload_id"],
                workload_role="validation",
            )
            for item in workflow["programs"]
        }
        inventory = {
            "schema_version": 2,
            "applications": applications,
            "status": "inventoried",
            "coverage_status": "unverified",
            "n_operations": sum(app["n_operations"] for app in applications.values()),
            "basis": "held-out validation observation only; never a capsule derivation input",
        }
        traces, catalogs = {}, {}
        for program in workflow["programs"]:
            for filename, destination in (("frontend-trace.json", traces), ("pytorch-opset.json", catalogs)):
                member = Path(program["bundle"]) / filename
                if member.is_file():
                    destination[program["name"]] = json.loads(_plain(member).read_bytes())
        accounting = build_operation_accounting(
            inventory,
            selection.software_spec,
            capability_contract=selection.contract,
            framework_catalogs=catalogs,
            frontend_traces=traces,
            application_graphs={label: app["operation_graph"] for label, app in inventory["applications"].items()},
            host_capabilities=getattr(selection, "host_capabilities", None),
        )
        _json(root / "operation-accounting.json", accounting)
    observations, native_lowerings = [], []
    for index, program in enumerate(workflow["programs"]):
        output = root / f"program-{index:03d}"
        output.mkdir()
        input_ir = _ir_observation((Path(program["bundle"]) / "model.mlir").read_text())
        if request.get("lower_native"):
            from merlin.compiler.lower.model.lower import lower_model_file

            bundle = Path(program["bundle"])
            try:
                from merlin.development.targetgen.applications.inventory import verify_capture_receipt

                if verify_capture_receipt(bundle / "model.mlir")["status"] != "verified_materialized":
                    raise ValueError("full native lowering requires a verified materialized capture receipt")
                if not input_ir.get("operation_counts", {}).get("func.func"):
                    raise ValueError("full native lowering requires a parsed program entrypoint, not an empty module")
                lowered = lower_model_file(
                    bundle / "model.mlir",
                    output / "native",
                    targets=(),
                    ir_audit="both",
                    audit_sidecars=tuple(
                        bundle / name
                        for name in ("weights.safetensors", "weights.safetensors.manifest.json", "capture_receipt.json")
                        if (bundle / name).is_file()
                    ),
                )
                native_lowerings.append(
                    {
                        "program": program["name"],
                        "status": "llvm_ir_emitted",
                        "model_sha256": program["model_sha256"],
                        "llvm_ir": str(lowered.ll_path.relative_to(root)),
                        "llvm_ir_sha256": sha256_file(lowered.ll_path),
                        "audit_index": str(lowered.audit_index.relative_to(root)),
                        "qualification": "complete program lowering only; not numerical execution",
                    }
                )
            except Exception as exc:
                native_lowerings.append(
                    {
                        "program": program["name"],
                        "status": "failed",
                        "model_sha256": program["model_sha256"],
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        for name in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("model qualification exhausted its declared budget")
            try:
                completed = _compiler_check(
                    package,
                    Path(program["bundle"]) / "model.mlir",
                    name,
                    output,
                    timeout=min(remaining, request["stage_timeout_seconds"]),
                )
                (output / f"{name}.stdout").write_text(completed.stdout)
                (output / f"{name}.stderr").write_text(completed.stderr)
                emitted = _ir_observation(completed.stdout)
                row = {
                    "program": program["name"],
                    "entrypoint": name,
                    "returncode": completed.returncode,
                    "status": "accepted" if completed.returncode == 0 else "refused",
                    "input_ir": input_ir,
                    "emitted_ir": emitted,
                    "canonical_ir_changed": (
                        emitted["canonical_sha256"] != input_ir["canonical_sha256"]
                        if "canonical_sha256" in emitted and "canonical_sha256" in input_ir
                        else None
                    ),
                }
                if name == "emit_command_buffer":
                    row["command_buffer"] = _command_observation(output / "compiler-scratch/commands.json")
                    if completed.returncode == 0:
                        row["status"] = row["command_buffer"]["status"]
                elif completed.returncode == 0 and name == "lower_interface_to_target":
                    row["status"] = (
                        "unchanged"
                        if row["canonical_ir_changed"] is False
                        else "transformed_unverified"
                        if row["canonical_ir_changed"] is True
                        else "uninspectable"
                    )
                elif completed.returncode == 0 and name == "lower_target_to_llvm":
                    prior = next(
                        (
                            item
                            for item in observations
                            if item["program"] == program["name"] and item["entrypoint"] == "emit_command_buffer"
                        ),
                        None,
                    )
                    emitted_commands = ((prior or {}).get("command_buffer") or {}).get("status") == "emitted"
                    row["status"] = (
                        "llvm_artifact_unverified"
                        if emitted_commands and emitted.get("status") == "parsed"
                        else "unqualified"
                    )
                observations.append(row)
            except Exception as exc:
                observations.append(
                    {
                        "program": program["name"],
                        "entrypoint": name,
                        "status": "unavailable",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    runtime = {"status": "not_requested", "target_executed": False}
    if request["execute"]:
        if len(workflow["programs"]) != 1:
            runtime = {
                "status": "unavailable",
                "target_executed": False,
                "reason": "the existing mesh dispatch runtime does not execute multi-program session contracts",
            }
        else:
            # This is the existing execution route, not a second interpreter or
            # target compiler. Its fallback counters must remain visible.
            from merlin.development.targetgen.packaging import loading as package_runtime
            from merlin.runtime.dispatch.whole_model import run_model

            original_entrypoint = package_runtime.run_entrypoint
            original_certify = package_runtime.certify
            certify_was_explicit = "certify" in vars(package_runtime)
            selected_certification = contextvars.ContextVar("selected_model_certification", default=False)
            certify_patched = False
            try:

                def isolated_entrypoint(pkg, name, input_mlir, output_json=None, *, timeout=600, **_kwargs):
                    if not selected_certification.get():
                        raise ValueError("compiler entrypoint ran outside selected certification")
                    completed = _compiler_check(
                        pkg,
                        Path(input_mlir),
                        name,
                        root / "runtime",
                        timeout=min(timeout, max(0.01, deadline - time.monotonic())),
                    )
                    if output_json is not None:
                        generated = root / "runtime/compiler-scratch/commands.json"
                        if generated.is_file() and not generated.is_symlink():
                            Path(output_json).write_bytes(generated.read_bytes())
                    return completed

                (root / "runtime").mkdir()
                package_runtime.certify = _selected_source_certifier(
                    package.directory, original_certify, selected_certification
                )
                certify_patched = True
                package_runtime.run_entrypoint = isolated_entrypoint
                with ExitStack() as context:
                    if selection is not None:
                        from merlin.target.registry.identity import observed_contract
                        from merlin.target.rtl.facts import observed_facts

                        context.enter_context(observed_contract(request["target"], selection.contract))
                        facts_path = (
                            Path(request["evidence_bundle"]) / "hardware/circt/facts.json"
                            if selection.raw_facts
                            else None
                        )
                        context.enter_context(
                            observed_facts(
                                request["target"],
                                selection.refreshed_facts if selection.raw_facts else {},
                                facts_path,
                            )
                        )
                    else:
                        # The evaluator must not regenerate RTL evidence or mix
                        # ambient name-keyed caches into a numerical observation.
                        raise ValueError("target execution requires an explicit saved Phase 0 evidence bundle")
                    result = run_model(
                        request["bundle"],
                        root / "runtime",
                        kernel_backend="mesh",
                        mesh_target=request["target"],
                        mesh_package=request["package"],
                        numeric_policy=request.get("numeric_policy"),
                    )
                runtime = {
                    key: value for key, value in result.items() if key not in {"output", "outputs", "golden", "goldens"}
                }
                runtime["status"] = "numerically_matched" if result.get("ok") is True else "not_matched"
                runtime["target_executed"] = int(result.get("mesh_ran", 0)) > 0
            except Exception as exc:
                (root / "runtime/exception.txt").write_text(traceback.format_exc())
                runtime = {
                    "status": "unavailable",
                    "target_executed": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "diagnostic": "runtime/exception.txt",
                }
            finally:
                package_runtime.run_entrypoint = original_entrypoint
                if certify_patched:
                    if certify_was_explicit:
                        package_runtime.certify = original_certify
                    else:
                        del package_runtime.certify
    from merlin.development.backend.qualification.full_models.audits import summarize_model_routes

    model_routes = summarize_model_routes(workflow, accounting, observations, native_lowerings)
    return {
        "workflow": workflow,
        "accounting": accounting,
        "compiler_observations": observations,
        "model_routes": model_routes,
        "runtime": runtime,
        "native_lowerings": native_lowerings,
        "full_native_lowering_verified": bool(native_lowerings)
        and all(row["status"] == "llvm_ir_emitted" for row in native_lowerings),
        "whole_workload_validation_verified": False,
        "qualification": "diagnostic observations; no compiler certificate or application accuracy claim",
        "cleanup_scope": (
            "bounded process groups and PID-isolated compiler subprocesses; "
            "arbitrary worker-loss guardian reaping is not verified"
        ),
    }


def _native_compiler_identity() -> dict:
    """Pin the real existing host pipeline and selected tools, not a fictitious RVV package."""
    import merlin
    from merlin.compiler.lower.model import codegen, toolchain
    from merlin.experiments.inspection.qualification_workflow import _plain

    interpreter = toolchain.compiler_python()
    probe = subprocess.run(
        [
            str(interpreter),
            "-c",
            "import importlib.util; print(importlib.util.find_spec('torch_mlir').submodule_search_locations[0])",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    upstream = _plain(Path(probe.stdout.strip()), directory=True)
    tools = {}
    for name, value in {
        "compiler_python": interpreter,
        "clang": toolchain.clang(),
        "mlir_translate": toolchain.mlir_translate(),
        "system_cc": shutil.which("cc"),
        "worker_python": sys.executable,
    }.items():
        if not value:
            raise ValueError(f"native host tool is unavailable: {name}")
        path = Path(value).absolute()
        resolved = path.resolve(strict=True)
        tools[name] = {"path": str(path), "resolved_path": str(resolved), "sha256": sha256_file(resolved)}
    payload = _plain(Path(merlin.__file__).parent, directory=True)
    return {
        "route": "merlin.compiler.lower.model.lower_model_file:host",
        "core_payload": {"path": str(payload), **hash_tree(payload)},
        "upstream_mlir": {"path": str(upstream), **hash_tree(upstream)},
        "producer": {"path": __file__, "sha256": sha256_file(__file__)},
        "runtime_c": {"path": str(codegen.mlir_runtime_c()), "sha256": sha256_file(codegen.mlir_runtime_c())},
        "tools": tools,
        "platform": {"machine": platform.machine(), "system": platform.system(), "libc": list(platform.libc_ver())},
        "qualification": "selected compiler/tool bytes; not an RVV package or complete OS closure",
    }


def _native_host_worker(request: dict, root: Path) -> dict:
    """Run one saved FP32 program with existing manifest, lowering and ciface owners."""
    import numpy as np

    from merlin.compiler.ir.mlir_query import forward_signature
    from merlin.compiler.lower.model.lower import lower_model_file
    from merlin.compiler.lower.model.oracle import HostModel
    from merlin.development.targetgen.applications.inventory import verify_capture_receipt
    from merlin.runtime.dispatch.whole_model import resolve_forward_args

    workflow = request["workflow"]
    if len(workflow["programs"]) != 1 or workflow["session"]:
        raise ValueError("native-host-only requires one standalone program, not a recurrent session")
    program = workflow["programs"][0]
    bundle = Path(program["bundle"])
    receipt = verify_capture_receipt(bundle / "model.mlir")
    if receipt["status"] != "verified_materialized":
        raise ValueError("native host execution requires a verified materialized capture receipt")
    metadata = json.loads((bundle / "meta.json").read_bytes())
    inputs, outputs = forward_signature(bundle / "model.mlir")
    expected_abi = [{"shape": shape, "dtype": dtype} for shape, dtype in outputs]
    if len(outputs) != 1 or outputs[0][1] != "f32" or metadata.get("output_abi") != expected_abi:
        raise ValueError("native host golden.npy qualification requires one exact FP32 result ABI")
    if len(inputs) >= 1024 or any(any(dim <= 0 for dim in shape) for shape, _ in inputs + outputs):
        raise ValueError("native host qualification requires bounded static tensor arguments")
    arguments = resolve_forward_args(bundle)
    if len(arguments) != len(inputs):
        raise ValueError("resolved argument count differs from the captured function ABI")
    storage = {"f32": "float32", "i64": "int64", "i32": "int32", "i8": "int8", "i1": "bool"}
    for value, (shape, dtype) in zip(arguments, inputs, strict=True):
        if list(value.shape) != shape or str(value.dtype) != storage.get(dtype):
            raise ValueError("resolved argument shape/precision differs from the captured ABI")
    golden = np.load(bundle / "golden.npy", allow_pickle=False)
    if list(golden.shape) != outputs[0][0] or str(golden.dtype) != "float32" or not np.isfinite(golden).all():
        raise ValueError("saved eager reference differs from the qualified FP32 output ABI")
    output = root / "program-000"
    output.mkdir()
    lowered = lower_model_file(
        bundle / "model.mlir",
        output / "native",
        targets=("host",),
        ir_audit="both",
        audit_sidecars=tuple(bundle / name for name in workflow["members"]),
    )
    observed = np.zeros(outputs[0][0], dtype=np.float32)
    before = [hashlib.sha256(value.tobytes()).hexdigest() for value in arguments]
    host = HostModel.load(str(lowered.host_so))
    host(
        [(value.ctypes.data, list(value.shape)) for value in arguments] + [(observed.ctypes.data, list(observed.shape))]
    )
    if before != [hashlib.sha256(value.tobytes()).hexdigest() for value in arguments]:
        raise ValueError("native program modified its captured input/weight buffers")
    policy = request["numeric_policy"]
    finite = bool(np.isfinite(observed).all())
    matched = bool(finite and np.allclose(observed, golden, atol=policy["atol"], rtol=policy["rtol"], equal_nan=False))
    np.save(output / "output.npy", observed, allow_pickle=False)
    row = {
        "program": program["name"],
        "status": "numerically_matched" if matched else "not_matched",
        "model_sha256": program["model_sha256"],
        "llvm_ir": str(lowered.ll_path.relative_to(root)),
        "llvm_ir_sha256": sha256_file(lowered.ll_path),
        "host_shared": str(lowered.host_so.relative_to(root)),
        "host_shared_sha256": sha256_file(lowered.host_so),
        "audit_index": str(lowered.audit_index.relative_to(root)),
        "numeric_policy": policy,
        "max_absolute_error": float(np.max(np.abs(observed.astype(np.float64) - golden))) if finite else None,
        "nonfinite_output_elements": int(np.count_nonzero(~np.isfinite(observed))),
        "input_abi": [{"shape": shape, "dtype": dtype} for shape, dtype in inputs],
        "output_abi": expected_abi,
        "input_buffer_sha256": before,
        "capture_receipt": receipt,
        "qualification": (
            "this exact saved program, inputs and reference only; not individual-op or all-shape support"
        ),
    }
    return {
        "workflow": workflow,
        "native_lowerings": [row],
        "compiler_observations": [],
        "runtime": {"status": row["status"], "target_executed": False, "native_host_executed": True},
        "full_native_lowering_verified": True,
        "native_host_numerical_verified": matched,
        "whole_workload_validation_verified": False,
        "qualification": "finite standalone native-host check; no accelerator, RVV or application accuracy claim",
    }


def qualify(
    *,
    bundle: Path,
    package: Path | None,
    target: str | None,
    output: Path,
    execute: bool = False,
    timeout_seconds: int = 120,
    stage_timeout_seconds: int = 30,
    memory_gib: int = 24,
    cpu_count: int = 2,
    numeric_policy: dict | None = None,
    evidence_bundle: Path | None = None,
    workload_id: str | None = None,
    lower_native: bool = False,
    native_host_only: bool = False,
) -> dict:
    """Generate a fresh, immutable receipt using existing process/timeout tooling."""
    from merlin.development.backend.providers.execution import stream_stamped
    from merlin.experiments.inspection.qualification_workflow import MODULE, SCHEMA, _json, _plain, inspect_workflow

    bundle = _plain(bundle, directory=True)
    if native_host_only:
        if package is not None or execute or evidence_bundle is not None or target is not None:
            raise ValueError("native-host-only cannot select an OOT package, target, execution or target evidence")
        if (
            not isinstance(numeric_policy, dict)
            or set(numeric_policy) != {"atol", "rtol"}
            or any(
                type(value) not in (float, int) or not math.isfinite(value) or value < 0
                for value in numeric_policy.values()
            )
        ):
            raise ValueError("native-host-only requires explicit finite nonnegative atol/rtol")
    else:
        package = _plain(package, directory=True)
    if min(timeout_seconds, stage_timeout_seconds, memory_gib, cpu_count) <= 0:
        raise ValueError("qualification resource budgets must be positive")
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise ValueError("qualification output must be a fresh non-symlink path")
    if package is not None:
        for path in package.rglob("*"):
            if path.is_symlink():
                raise ValueError("compiler package contains a symlink")
    before = _native_compiler_identity() if native_host_only else hash_tree(package)
    support_root = None
    support_before = None
    if execute and target is not None:
        from merlin.target.registry.identity import resolve

        support_root = resolve(target).external_root
        if support_root is not None:
            support_root = _plain(support_root, directory=True)
            if any(path.is_symlink() for path in support_root.rglob("*")):
                raise ValueError("selected OOT support provider contains a symlink")
            support_before = {"path": str(support_root), **hash_tree(support_root)}
    evidence_before = None
    if evidence_bundle is not None:
        evidence_bundle = _plain(evidence_bundle, directory=True)
        from merlin.development.testgen.evidence.evidence import load_exported_evidence

        load_exported_evidence(evidence_bundle)
        for path in evidence_bundle.rglob("*"):
            if path.is_symlink():
                raise ValueError("selected target evidence contains a symlink")
        evidence_before = hash_tree(evidence_bundle)
    workflow = inspect_workflow(bundle)
    output.mkdir(parents=True, mode=0o700)
    request = {
        "bundle": str(bundle),
        "package": str(package) if package else None,
        "target": target,
        "execute": execute,
        "lower_native": lower_native,
        "native_host_only": native_host_only,
        "workload_id": workload_id or bundle.name,
        "timeout_seconds": timeout_seconds,
        "stage_timeout_seconds": stage_timeout_seconds,
        "numeric_policy": numeric_policy,
        "workflow": workflow,
        "evidence_bundle": str(evidence_bundle) if evidence_bundle else None,
        "evidence_identity": evidence_before,
        "support_provider": support_before,
        "certification_output_root": str(output / "certification-output") if execute else None,
        "tool_environment": {
            name: os.environ[name]
            for name in (
                "MERLIN_M2M_DIR",
                "MERLIN_M2M_PYTHON",
                "MERLIN_M2M_VENV",
                "MERLIN_COMPILER_PYTHON",
                "MERLIN_COMPILER_VENV",
                "MERLIN_MLC_DIR",
                "MERLIN_TARGET_PATH",
                "MERLIN_CHIPYARD",
                "MERLIN_MESH_SIM",
                "MERLIN_REQUIRED_RTL_ENGINE",
                "MERLIN_CLANG",
                "MERLIN_MLIR_OPT",
                "MERLIN_MLIR_TRANSLATE",
            )
            if name in os.environ
        },
    }
    _json(output / "request.json", request)
    cpus = sorted(os.sched_getaffinity(0))[:cpu_count]
    command = [
        "prlimit",
        f"--as={memory_gib * 1024**3}",
        "--cpu=" + str(timeout_seconds * cpu_count + 5),
        "taskset",
        "--cpu-list",
        ",".join(str(cpu) for cpu in cpus),
        sys.executable,
        "-m",
        MODULE,
        "--worker",
        str(output),
    ]
    environment = without_unsealed_board_catalog(os.environ)
    worker_tmp = output / "worker-tmp"
    worker_tmp.mkdir(mode=0o700)
    environment["TMPDIR"] = str(worker_tmp)
    if execute:
        # The K-ladder's shape-keyed run IDs are stable, so a global out/runs
        # root would silently reuse or overwrite another qualification's files.
        environment["MERLIN_OUT_ROOT"] = request["certification_output_root"]
    environment["PYTHONPATH"] = os.pathsep.join(
        str(Path(value or ".").absolute()) for value in environment.get("PYTHONPATH", "").split(os.pathsep)
    )
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS=str(cpu_count),
        OPENBLAS_NUM_THREADS=str(cpu_count),
        MKL_NUM_THREADS=str(cpu_count),
    )
    state, error, returncode = "completed", None, None
    try:
        returncode = stream_stamped(
            command,
            cwd=output,
            transcript=output / "worker.transcript",
            stderr_path=output / "worker.stderr",
            raw_path=output / "worker.stdout",
            timeout=timeout_seconds,
            env=environment,
        )
        if returncode != 0:
            state = "worker_failed"
    except subprocess.TimeoutExpired:
        state = "budget_exhausted"
    except Exception as exc:
        state, error = "unavailable", f"{type(exc).__name__}: {exc}"
    after = _native_compiler_identity() if native_host_only else hash_tree(package)
    if before != after:
        state, error = "inputs_changed", "selected compiler package bytes changed during qualification"
    if support_root is not None and support_before != {"path": str(support_root), **hash_tree(support_root)}:
        state, error = "inputs_changed", "selected OOT support provider bytes changed during qualification"
    if workflow != inspect_workflow(bundle):
        state, error = "inputs_changed", "selected model/session input bytes changed during qualification"
    if evidence_bundle is not None and evidence_before != hash_tree(evidence_bundle):
        state, error = "inputs_changed", "selected target evidence bytes changed during qualification"
    result_path = output / "observations.json"
    observations = json.loads(result_path.read_bytes()) if result_path.is_file() else None
    guide = [
        "# Model qualification artifacts",
        "",
        f"Observation status: `{state}`. This is not a compiler certificate or application-accuracy result.",
        "",
        "- [request.json](request.json): selected inputs, complete program roster and resource limits.",
        "- [observations.json](observations.json): native lowering, OOT compiler decisions and runtime status.",
        "- [qualification.json](qualification.json): immutable input/output identities and scope.",
        "",
        "## Complete-program lowering",
        "",
        "| Program | Result | Audited stages |",
        "| --- | --- | --- |",
    ]
    for row in (observations or {}).get("native_lowerings", []):
        audit = row.get("audit_index")
        guide.append(
            f"| {row['program']} | {row['status']} | "
            + (f"[index]({audit})" if audit else "Not completed; inspect observations")
            + " |"
        )
    if native_host_only:
        guide += [
            "",
            "## Finite native-host numerical check",
            "",
            "This mode compiles and executes one standalone captured program on this CPU using the",
            "existing manifest/weight ABI and HostModel runner. Inspect `output.npy`, explicit atol/rtol,",
            "the saved eager reference, compiler/tool byte identities and read-only input checks.",
            "A match applies only to these exact inputs and weights. It does not establish individual-op",
            "support, other shapes/precisions, RVV deployment, accelerator offload or application accuracy.",
            "No immutable historical host compiler package is edited or implicitly selected.",
        ]
    guide += [
        "",
        "`llvm_ir_emitted` means the complete declared program reached LLVM IR; it does not mean",
        "accelerator offload, executable generation or numerical execution passed. Inspect each audit's",
        "exact stages, compact inspection-only views and weight-sidecar identities.",
        "",
        "`observations.json` also has one `model_routes` row per declared program. It joins the exact",
        "capture identity, selected operation ledger, native-host LLVM output and OOT command-buffer",
        "route. The route status distinguishes observed decline from unresolved or unverified",
        "candidates; `whole_model_compiler_verified` remains false without a complete execution witness.",
        "",
        "Compiler return codes alone are insufficient: inspect command-buffer declines, empty command lists,",
        "unchanged IR and unsupported operations. Eager goldens are reference inputs, not target results.",
        "A full checkpoint with bounded/synthetic inputs is not an application-accuracy evaluation.",
        "",
    ]
    (output / "README.md").write_text("\n".join(guide), encoding="utf-8")
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError("qualification output contains a symlink")
        if path.is_file():
            _plain(path)
            files[path.relative_to(output).as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    receipt = {
        "schema": SCHEMA,
        "status": state,
        "error": error,
        "worker_returncode": returncode,
        "target": target,
        "compiler": {"path": str(package) if package else None, **before},
        "request": request,
        "observations": observations,
        "artifacts": files,
        "resources": {"wall_seconds": timeout_seconds, "memory_gib": memory_gib, "cpus": cpus},
        "held_out_policy": "evaluation only; no capsule derivation, tuning or model-specific compiler changes",
        "native_host_numerical_verified": state == "completed"
        and bool((observations or {}).get("native_host_numerical_verified")),
        "whole_workload_validation_verified": False,
    }
    _json(output / "qualification.json", receipt)
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    for path in sorted((p for p in output.rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o500)
    output.chmod(0o500)
    return receipt


def main(argv=None) -> int:
    from merlin.experiments.inspection.qualification_workflow import _json, _plain

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--workload-id", help="validation workload identity, defaulting to capture directory name")
    parser.add_argument(
        "--evidence-bundle",
        type=Path,
        help="saved Phase 0 target evidence for operator/support accounting; never recaptured",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true", help="also attempt the existing single-model mesh runtime")
    parser.add_argument(
        "--lower-native",
        action="store_true",
        help="audit every declared complete program through the shared LLVM IR lowering path",
    )
    parser.add_argument(
        "--native-host-only",
        action="store_true",
        help="execute one receipt-bound FP32 program on this CPU; no OOT package or accelerator",
    )
    parser.add_argument("--atol", type=float, help="explicit absolute tolerance for native-host-only")
    parser.add_argument("--rtol", type=float, help="explicit relative tolerance for native-host-only")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--stage-timeout-seconds", type=int, default=30)
    parser.add_argument("--memory-gib", type=int, default=24)
    parser.add_argument("--cpu-count", type=int, default=2)
    args = parser.parse_args(argv)
    if args.worker:
        root = _plain(args.worker, directory=True)
        _json(root / "observations.json", _worker(json.loads((root / "request.json").read_bytes()), root))
        return 0
    if args.native_host_only and not all((args.bundle, args.out)):
        parser.error("native-host-only requires --bundle and --out")
    if not args.native_host_only and not all((args.bundle, args.package, args.target, args.out)):
        parser.error("--bundle, --package, --target and --out are required")
    if args.native_host_only and (args.atol is None or args.rtol is None):
        parser.error("native-host-only requires explicit --atol and --rtol")
    result = qualify(
        bundle=args.bundle,
        package=args.package,
        target=args.target,
        output=args.out,
        execute=args.execute,
        timeout_seconds=args.timeout_seconds,
        stage_timeout_seconds=args.stage_timeout_seconds,
        memory_gib=args.memory_gib,
        cpu_count=args.cpu_count,
        evidence_bundle=args.evidence_bundle,
        workload_id=args.workload_id,
        lower_native=args.lower_native,
        native_host_only=args.native_host_only,
        numeric_policy={"atol": args.atol, "rtol": args.rtol} if args.native_host_only else None,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt": str(args.out / "qualification.json"),
                "native_host_numerical_verified": result["native_host_numerical_verified"],
                "whole_workload_validation_verified": False,
            }
        )
    )
    return (
        0
        if result["status"] == "completed" and (not args.native_host_only or result["native_host_numerical_verified"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
