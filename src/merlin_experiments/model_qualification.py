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

from merlin.common.digest import sha256_file
from merlin.common.tree_hash import hash_tree
from merlin_experiments.measured_launch import without_unsealed_board_catalog

SCHEMA = "merlin.model_qualification.v1"
MODULE = "merlin_experiments.model_qualification"
_MAX_IR_OBSERVATION_BYTES = 256 * 1024 * 1024


def _plain(path: Path, *, directory: bool = False) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f"qualification input traverses a symlink: {path}")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"qualification input is absent: {path}")
    return path


def _json(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _selected_source_certifier(selected: Path, certify, active: contextvars.ContextVar[bool]):
    """Guard the source passed to certification, not its verified build copy."""

    def checked(package_dir, *args, **kwargs):
        if Path(package_dir).resolve(strict=True) != selected.resolve(strict=True):
            raise ValueError("mesh runtime selected a different compiler package")
        token = active.set(True)
        try:
            return certify(package_dir, *args, **kwargs)
        finally:
            active.reset(token)

    return checked


def _ir_observation(text: str) -> dict:
    """Inspect actual emitted IR, rather than treating a zero exit as lowering."""
    digest = hashlib.sha256()
    size = 0
    for offset in range(0, len(text), 4 * 1024 * 1024):
        chunk = text[offset : offset + 4 * 1024 * 1024].encode()
        digest.update(chunk)
        size += len(chunk)
    result = {"sha256": digest.hexdigest(), "bytes": size}
    if size > _MAX_IR_OBSERVATION_BYTES:
        result.update(
            status="not_parsed",
            reason=(
                f"emitted IR exceeds the {_MAX_IR_OBSERVATION_BYTES}-byte in-memory verification limit; "
                "the saved artifact remains available for separate inspection"
            ),
        )
        return result

    from xdsl.printer import Printer

    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir

    try:
        normalized, normalization = normalize_capture_mlir(text)
        module = mq.parse(normalized)
        names = Counter(mq.op_name(op) for op in module.walk())
        dialects = Counter()
        for name, count in names.items():
            dialects[name.partition(".")[0]] += count
        canonical = io.StringIO()
        Printer(stream=canonical, print_generic_format=True).print_op(module)
        result.update(
            status="parsed",
            operation_counts=dict(sorted(names.items())),
            dialect_counts=dict(sorted(dialects.items())),
            canonical_sha256=hashlib.sha256(canonical.getvalue().encode()).hexdigest(),
            normalization=normalization,
        )
        try:
            module.verify()
            result["structural_verification"] = "passed"
        except Exception as exc:
            result["structural_verification"] = "failed"
            result["verification_error"] = f"{type(exc).__name__}: {exc}"
        result["verification_scope"] = (
            "shared registered xDSL grammar; unregistered OOT operations are not independently verified"
        )
    except Exception as exc:
        result.update(status="not_parsed", reason=f"{type(exc).__name__}: {exc}")
    return result


def _command_observation(path: Path) -> dict:
    if not path.exists():
        return {"status": "not_emitted"}
    from merlin.runtime.route_partition import route_of

    try:
        document = json.loads(_plain(path).read_bytes())
    except json.JSONDecodeError as exc:
        return {"status": "invalid", "reason": f"command buffer is not JSON: {exc}"}
    if not isinstance(document, dict):
        return {"status": "invalid", "reason": "command buffer is not a mapping"}
    commands = document.get("commands")
    verdict = route_of(document)
    status = {"A": "emitted", "D": "declined", "!": "invalid"}[verdict.route]
    return {
        "status": status,
        "route": verdict.route,
        "violations": verdict.kinds,
        "commands_count": len(commands) if isinstance(commands, list) else None,
        "declined": document.get("declined"),
        "qualification": "payload route only; commands are not proven complete, executable or numerically correct",
    }


def inspect_workflow(bundle: Path) -> dict:
    """Inventory the actual program roster; never upgrade a diagnostic stage to a session."""
    bundle = _plain(bundle, directory=True)
    session_path = bundle / "session_contract.yaml"
    session = yaml.safe_load(_plain(session_path).read_bytes()) if session_path.exists() else {}
    if not isinstance(session, dict):
        raise ValueError("model session contract must be a mapping")
    programs = []
    if session_path.exists() and session.get("version") not in (1, 2):
        raise ValueError("model session contract must have version 1 or 2")
    if session.get("version") == 1 and session.get("programs"):
        raise ValueError("a multi-program session cannot become one forward")
    if session.get("version") == 2:
        roster = session.get("programs")
        if not isinstance(roster, list) or not roster:
            raise ValueError("multi-program session has no declared programs")
        for item in roster:
            relative = Path(item["bundle"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("session program escapes its bundle")
            root = _plain(bundle / relative, directory=True)
            programs.append({"name": item["name"], "bundle": str(root), "steps": item.get("steps")})
        if len({item["name"] for item in programs}) != len(programs):
            raise ValueError("session program names are not unique")
    else:
        programs.append({"name": "forward", "bundle": str(bundle), "steps": session.get("steps", 1)})
    members = {}
    for program in programs:
        root = Path(program["bundle"])
        model = _plain(root / "model.mlir")
        program["model_sha256"] = sha256_file(model)
        missing = []
        required = {
            "model.mlir",
            "weights.safetensors",
            "weights.safetensors.manifest.json",
            "inputs.npz",
            "input_order.json",
            "golden.npy",
        }
        sidecars = {
            "extra.npz",
            "goldens.npz",
            "output_order.json",
            "golden_w8a8.npy",
            "session_inputs.npz",
            "session_goldens.npz",
            "session_quality_fp32.npz",
            "frontend-trace.json",
            "pytorch-opset.json",
            "capture_receipt.json",
            "meta.json",
        }
        for name in sorted(required | sidecars):
            path = root / name
            if not path.exists():
                if name in required:
                    missing.append(name)
                continue
            _plain(path)
            members[path.relative_to(bundle).as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        program["missing_runtime_inputs"] = missing
    if session_path.exists():
        members["session_contract.yaml"] = {"sha256": sha256_file(session_path), "bytes": session_path.stat().st_size}
    session_abi = {"status": "not_applicable", "bindings": 0}
    session_capture = {"status": "not_applicable", "source_closure_verified": False}
    if session.get("version") == 2:
        # The shared session loader validates ordered programs, recurrent child
        # contracts and the actual typed MLIR ABIs at every cross-program edge.
        # A name-only roster check silently misses a changed cache or action
        # tensor even when each program still lowers in isolation.
        from merlin.llvmlower.session_bundle import load as load_session

        try:
            selected = load_session(bundle)
            if selected.program_names != tuple(program["name"] for program in programs):
                raise ValueError("validated session differs from selected program roster")
            session_abi = {"status": "verified_structure", "bindings": len(selected.bindings)}
        except Exception as exc:
            session_abi = {"status": "unverified", "bindings": None, "reason": f"{type(exc).__name__}: {exc}"}

        receipt_path = bundle / "session-receipt.json"
        if receipt_path.exists():
            _plain(receipt_path)
            members["session-receipt.json"] = {
                "sha256": sha256_file(receipt_path),
                "bytes": receipt_path.stat().st_size,
            }
            try:
                receipt = json.loads(receipt_path.read_bytes())
                rows = receipt.get("programs") if isinstance(receipt, dict) else None
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("schema") != "merlin.model_session_capture.v1"
                    or receipt.get("session_contract_sha256") != members["session_contract.yaml"]["sha256"]
                    or not isinstance(rows, list)
                    or len(rows) != len(programs)
                ):
                    raise ValueError("session receipt schema, contract or program count differs")
                for program, row in zip(programs, rows, strict=True):
                    relative = Path(program["bundle"]).relative_to(bundle) / "capture_receipt.json"
                    stage_receipt = members.get(relative.as_posix())
                    if (
                        not isinstance(row, dict)
                        or row.get("name") != program["name"]
                        or row.get("receipt_sha256") != (stage_receipt or {}).get("sha256")
                        or row.get("ok") is not True
                        or type(row.get("opaque")) is not int
                        or row["opaque"] != 0
                        or not isinstance(row.get("materialized_abi"), dict)
                        or row["materialized_abi"].get("complete") is not True
                    ):
                        raise ValueError(f"session receipt does not bind valid stage {program['name']}")
                session_capture = {
                    "status": "producer_receipts_bound",
                    "receipt_sha256": members["session-receipt.json"]["sha256"],
                    "source_closure_verified": False,
                    "qualification": (
                        "producer-stated stage receipts; not an independent source or numerical attestation"
                    ),
                }
            except (ValueError, TypeError, KeyError) as exc:
                session_capture = {
                    "status": "unverified",
                    "source_closure_verified": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        else:
            session_capture = {
                "status": "unverified",
                "source_closure_verified": False,
                "reason": "session-receipt.json is absent",
            }
    provenance = session.get("provenance") or {}
    blockers = []
    if provenance.get("full_checkpoint") is not True:
        blockers.append("full pretrained checkpoint provenance is not established")
    if provenance.get("synthetic_inputs") is not False or not provenance.get("input_source"):
        blockers.append("attributed non-synthetic application inputs are not established")
    if not session:
        blockers.append("no end-to-end application session contract; only a single forward is available")
    elif session.get("version") == 2 and session_abi["status"] != "verified_structure":
        blockers.append("cross-program typed ABI and binding structure are unverified")
    if session.get("version") == 2 and session_capture["status"] != "producer_receipts_bound":
        blockers.append("session producer receipts do not bind all selected programs")
    return {
        "programs": programs,
        "session": session,
        "members": members,
        "session_abi": session_abi,
        "session_capture": session_capture,
        "application_validation_blockers": blockers,
        "scope": "declared session roster and artifact identities; not checkpoint/source closure certification",
    }


def _compiler_check(package, model: Path, name: str, output: Path, *, timeout: float):
    """Use the existing bounded candidate executor and PID-isolated sandbox base."""
    from merlin.common.paths import repo_root
    from merlin.llvmlower.toolchain import clang_for
    from merlin.perf.analysis_worker import run_sandboxed_entrypoint
    from merlin.targetgen.sandbox.bwrap import base_argv
    from merlin.targetgen.sandbox.preflight import require_working_sandbox

    require_working_sandbox(
        context="model compiler requires network-isolated bwrap",
        network_isolation=True,
    )

    scratch = output / "compiler-scratch"
    scratch.mkdir(exist_ok=True)
    prefix = base_argv(scratch, {})
    # Match the operator-selected LLVM toolchain used by the agent and the
    # ordinary package runner.  The base sandbox hides /scratch*, so PATH alone
    # cannot make an installed compiler's mlir-translate visible here.
    selected_clang = clang_for(repo_root(), os.environ)
    llvm_bin = None
    if selected_clang.is_absolute() and selected_clang.is_file():
        candidate = selected_clang.parent
        if (candidate / "mlir-translate").is_file():
            llvm_bin = candidate
            if not candidate.is_relative_to(Path("/usr")):
                install = candidate.parent
                prefix += ["--ro-bind", str(install), str(install)]
    tool_path = ":".join(
        str(path) for path in (Path(sys.executable).parent, llvm_bin, Path("/usr/bin"), Path("/bin")) if path
    )
    # This evaluator is not an agent launch: credentials and user-home state
    # have no role in compiler execution. Only its explicit package and MLIR
    # input are exposed; numerical answers/weights remain outside the sandbox.
    prefix += [
        "--clearenv",
        "--unshare-net",
        "--setenv",
        "PATH",
        tool_path,
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
    ]
    if llvm_bin is not None:
        prefix += ["--setenv", "MERLIN_CLANG", str(selected_clang)]
    user_state = Path.home() / ".claude"
    if user_state.exists():
        prefix += ["--tmpfs", str(user_state)]
    import sysconfig

    # Bind the selected compiler interpreter and its Python environment,
    # retaining venv spelling rather than rebasing its executable symlink.
    for path in sorted({Path(sys.prefix), Path(sys.base_prefix), Path(sysconfig.get_path("stdlib"))}):
        prefix += ["--ro-bind", str(path), str(path)]
    executable = Path(sys.executable)
    if executable.is_symlink():
        target = executable.readlink()
        target = target if target.is_absolute() else executable.parent / target
        # UV venvs may retain an unversioned installation alias while
        # sys.base_prefix names its versioned destination. Bind that selected
        # alias too; otherwise env/python falls through to system Python.
        alias = target.parent.parent
        if alias != Path(sys.base_prefix):
            prefix += ["--ro-bind", str(alias.resolve(strict=True)), str(alias)]
    prefix += ["--ro-bind", str(package.directory), str(package.directory), "--ro-bind", str(model), str(model)]
    return run_sandboxed_entrypoint(
        package,
        name,
        model,
        scratch / "commands.json" if name == "emit_command_buffer" else None,
        sandbox={"package_path": str(package.directory), "command_prefix": prefix},
        timeout_s=timeout,
    )


def _worker(request: dict, root: Path) -> dict:
    if request.get("native_host_only"):
        return _native_host_worker(request, root)
    from merlin.targetgen.package_runtime import load_package

    package = load_package(request["package"])
    if package.target != request["target"]:
        raise ValueError("selected compiler package names a different target")
    workflow = request["workflow"]
    deadline = time.monotonic() + request["timeout_seconds"]
    accounting = {"status": "not_measured", "reason": "no immutable selected target evidence bundle supplied"}
    selection = None
    if request.get("evidence_bundle"):
        from merlin.targetgen.application_inventory import application_operation_inventory
        from merlin.targetgen.operation_accounting import build_operation_accounting

        from .phase0.evidence import load_exported_evidence

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
            from merlin.llvmlower.lower import lower_model_file

            bundle = Path(program["bundle"])
            try:
                from merlin.targetgen.application_inventory import verify_capture_receipt

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
            from merlin.runtime.dispatch_runtime import run_model
            from merlin.targetgen import package_runtime

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
                        from merlin.targetgen.rtl.facts import observed_facts
                        from merlin.targetgen.target_registry import observed_contract

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
    from .phase1.model_routes import summarize_model_routes

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
    from merlin.llvmlower import codegen, toolchain

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
        "route": "merlin.llvmlower.lower_model_file:host",
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

    from merlin.common.mlir_query import forward_signature
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model_file
    from merlin.runtime.dispatch_runtime import resolve_forward_args
    from merlin.targetgen.application_inventory import verify_capture_receipt

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
    from merlin.common.arrival_stamp import stream_stamped

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
        from merlin.targetgen.target_registry import resolve

        support_root = resolve(target).external_root
        if support_root is not None:
            support_root = _plain(support_root, directory=True)
            if any(path.is_symlink() for path in support_root.rglob("*")):
                raise ValueError("selected OOT support provider contains a symlink")
            support_before = {"path": str(support_root), **hash_tree(support_root)}
    evidence_before = None
    if evidence_bundle is not None:
        evidence_bundle = _plain(evidence_bundle, directory=True)
        from .phase0.evidence import load_exported_evidence

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
