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

SCHEMA = "merlin.model_qualification.v1"


MODULE = "merlin.experiments.inspection.qualification_workflow"


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

    from merlin.compiler.frontend.patterns.capture_normalization import normalize_capture_mlir
    from merlin.compiler.ir import mlir_query as mq

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
    from merlin.evaluation.route_admission import route_of

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
        from merlin.compiler.lower.c.session import load as load_session

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
    from merlin.analysis.optimization.probes.analysis_worker import run_sandboxed_entrypoint
    from merlin.common.paths import repo_root
    from merlin.compiler.lower.model.toolchain import clang_for
    from merlin.development.targetgen.sandbox.grants import base_argv
    from merlin.development.targetgen.sandbox.preflight import require_working_sandbox

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
