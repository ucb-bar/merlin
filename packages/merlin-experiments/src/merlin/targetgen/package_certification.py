"""Trusted OOT certification and AET recording, owned by optional experiments.

Compiler package invocation remains in core package_runtime. Access through that
module preserves legacy oot_runner monkeypatches and error identities.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from . import package_runtime as _runtime
from .package_records import payload_inventory


def _cert_artifact_identity(paths: _runtime.RunPaths, run_id: str) -> dict[str, Any]:
    """Content-address the exact compiler/oracle artifacts produced by one certification.

    Model grading reuses shape-keyed run directories, so a pathname is not an immutable identity.  The
    ordered digest below binds the dispatch evidence to the command buffer, lowered LLVM, object, ELF and
    decoded trace bytes that actually existed when the cert returned.  Missing artifacts are explicit;
    callers decide which set is mandatory for their endpoint.
    """
    candidates = {
        "input_interface": paths.generated / "input.interface.mlir",
        "command_buffer": paths.generated / "command_buffer.json",
        "lowered_llvm": paths.generated / "lowered.llvm.mlir",
        "kernel_object": paths.generated / "kernel.o",
        "package_kernel_elf": paths.generated / "package_kernel.elf",
        "instruction_trace": paths.generated / "instruction_trace.json",
    }
    artifacts: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name, path in candidates.items():
        if not path.is_file():
            missing.append(name)
            continue
        data = path.read_bytes()
        artifacts[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    canonical = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
    return {
        "version": 1,
        "run_id": run_id,
        "content_sha256": hashlib.sha256(canonical).hexdigest(),
        "artifacts": artifacts,
        "missing": missing,
    }


def certify(
    package_dir: str | Path,
    interface_mlir: str | Path,
    *,
    runs_root: str | Path,
    run_id: str,
    simulator: str = "spike",
    contract: str | Path | None = None,
    seed: int = 0,
    timeout: int = 600,
    target: str | None = None,
    inputs: dict | None = None,
    require_accelerator_trace: bool = False,
) -> dict[str, Any]:
    """Run the K-ladder for one (package, interface input) and record an aet run dir.

    Returns the results dict (also written as results.yaml). Never raises for a package/gate
    failure — those are recorded as status: fail with a plane-routed FailureRecord; only an
    internal harness bug raises.

    ``target`` labels the run; when omitted it is derived from the package manifest's ``target``
    field, so the runner is target-agnostic rather than hardcoded to the reference target.
    """
    from ..runtime.backends import base as _bk
    from ..runtime.reference import outputs_match, reference_outputs
    from ..runtime.simulator import simulate
    from .provenance import toolchain_shas

    interface_mlir = Path(interface_mlir)
    rung = interface_mlir.stem.split(".")[0]
    if target is None:
        target = _runtime._package_target(package_dir)

    spec = _runtime.RunSpec(
        project="merlin",
        suite=_runtime.SUITE,
        method=f"{run_id}",
        seed=seed,
        run_id=run_id,
        project_root=Path(runs_root),
        tracking_mode="local",
        target=target,
        dtype="i8xi8_i32",
        benchmark=rung,
    )
    paths = _runtime.RunPaths.from_spec(spec, run_id)
    if paths.run_path.resolve().is_relative_to(Path(package_dir).resolve()):
        raise ValueError("certification run directory must be outside the package payload")
    for dd in (paths.run_path, paths.logs, paths.artifacts_dir, paths.generated, paths.contracts):
        dd.mkdir(parents=True, exist_ok=True)

    entry = {
        "parse": "skipped",
        "lower_interface_to_target": "skipped",
        "emit_command_buffer": "skipped",
        "lower_target_to_llvm": "skipped",
    }
    semantic = {"reference_outputs_vs_simulate": "skipped"}
    oracle = {
        "kind": "none",
        "engine": simulator,
        "derived_from_rtl": False,
        "cycle_accurate": False,
        "result": "skipped",
        "cycles": None,
    }
    oracle_outputs: dict | None = None  # the mesh's actual output values (for in-process callers)
    trace_check = {
        "required": bool(require_accelerator_trace),
        "status": "not_required",
        "drives_accelerator": None,
        "n_instructions": 0,
    }
    artifact_identity: dict[str, Any] = {}
    package_input_identity: dict[str, Any] | None = None
    source_inventory = None
    execution_inventory = None
    source_root = Path(package_dir).absolute()

    def verify_inputs() -> None:
        if (
            payload_inventory(source_root) != source_inventory
            or payload_inventory(pkg.directory) != execution_inventory
        ):
            raise _runtime.CertFailure(
                "package_identity",
                _runtime.FailureCategory.PROTOCOL_VIOLATION,
                "package source or built execution copy changed during certification",
            )

    def invoke(name: str, input_path: Path, output_path: Path | None = None):
        verify_inputs()
        try:
            return _runtime.run_entrypoint(pkg, name, input_path, output_path, timeout=timeout, write_bytecode=False)
        finally:
            verify_inputs()

    artifacts_recorded: dict[str, bool] = {}
    failure: dict[str, Any] | None = None
    status = "pass"
    cb: dict[str, Any] | None = None
    shas = toolchain_shas(target)

    # input artifact
    inp = paths.generated / "input.interface.mlir"
    inp.write_text(interface_mlir.read_text(encoding="utf-8"), encoding="utf-8")

    try:
        # K0/K1: load + validate manifest, integrity scan, build if needed
        pkg = _runtime.load_package(package_dir, contract=contract)
        _runtime.integrity_scan(pkg)
        compiler_package_id = pkg.manifest.get("package_id")
        source_inventory = payload_inventory(source_root)
        execution_parent = Path(tempfile.mkdtemp(prefix="compiler-execution-", dir=paths.artifacts_dir))
        execution_root = execution_parent / "package"
        shutil.copytree(source_root, execution_root)
        if payload_inventory(execution_root) != source_inventory or payload_inventory(source_root) != source_inventory:
            raise _runtime.CertFailure(
                "package_identity",
                _runtime.FailureCategory.PROTOCOL_VIOLATION,
                "package changed while preparing execution copy",
            )
        pkg = _runtime.load_package(execution_root, contract=contract)
        _runtime.build_package(pkg)
        execution_inventory = payload_inventory(execution_root)
        verify_inputs()
        if not pkg.tool.exists():
            raise _runtime.CertFailure(
                "build", _runtime.FailureCategory.ELABORATION_ERROR, f"package tool not found after build: {pkg.tool}"
            )

        # Resolve the package's runtime backend from the run's target (not a name literal) — the
        # runner is target-agnostic, so a package for any registered target reaches its own backend
        # helpers. Done AFTER load_package so a broken/unknown package fails closed on the manifest
        # (K0) rather than raising here; an unregistered target is a fail-closed contract violation.
        try:
            gem = _bk.get_backend(target)
        except KeyError as e:
            raise _runtime.CertFailure(
                "contract",
                _runtime.FailureCategory.STRUCTURAL_INVARIANT_VIOLATION,
                f"package declares target {target!r} with no registered backend",
            ) from e

        # K2: parse
        p = invoke("parse", inp)
        if p.returncode != 0:
            raise _runtime.CertFailure(
                "runner_invocation",
                _runtime.FailureCategory.TOOL_CRASH,
                f"parse entrypoint exited {p.returncode}: {p.stderr[-500:]}",
            )
        entry["parse"] = "pass"

        # K3: lower_interface_to_target -> non-empty MLIR
        p = invoke("lower_interface_to_target", inp)
        if p.returncode != 0 or not p.stdout.strip():
            raise _runtime.CertFailure(
                "codegen",
                _runtime.FailureCategory.ELABORATION_ERROR,
                f"lower_interface_to_target failed (rc={p.returncode}): {p.stderr[-500:]}",
            )
        target_path = paths.generated / "lowered.target.mlir"
        target_path.write_text(p.stdout, encoding="utf-8")
        entry["lower_interface_to_target"] = "pass"

        # K4: emit_command_buffer -> schema-valid command_buffer.json
        cb_path = paths.generated / "command_buffer.json"
        p = invoke("emit_command_buffer", inp, cb_path)
        if p.returncode != 0 or not cb_path.exists():
            raise _runtime.CertFailure(
                "artifact_class",
                _runtime.FailureCategory.STRUCTURAL_INVARIANT_VIOLATION,
                f"emit_command_buffer produced no command_buffer.json (rc={p.returncode}): {p.stderr[-500:]}",
            )
        try:
            cb = json.loads(cb_path.read_text(encoding="utf-8"))
            _runtime.schemas.validate_command_buffer(cb, contract=contract)
            # Schema validity alone does not bind this artifact back to the input program.  In
            # particular, a backend must not narrow an interface output and ask the oracle to decode
            # the smaller container.  Import locally to avoid the capsule-common/runner module cycle.
            from .capsule_common import validate_interface_tensor_dtypes

            validate_interface_tensor_dtypes(cb, inp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, _runtime.schemas.ContractViolation) as e:
            raise _runtime.CertFailure(
                "abi_schema", _runtime.FailureCategory.PROTOCOL_VIOLATION, f"command_buffer.json invalid: {e}"
            ) from e
        entry["emit_command_buffer"] = "pass"

        # K5 (L0): reference == simulate, always — over the SAME (optionally injected) operands
        ref = reference_outputs(cb, inputs)
        sim = simulate(cb, inputs)["outputs"]
        if not outputs_match(ref, sim):
            raise _runtime.CertFailure(
                "command_buffer_semantics",
                _runtime.FailureCategory.FUNCTIONAL_MISMATCH,
                "reference_outputs(cb) != simulate(cb): the emitted command buffer is not internally consistent",
            )
        semantic["reference_outputs_vs_simulate"] = "pass"

        # K6: lower_target_to_llvm -> compile to object/ELF
        p = invoke("lower_target_to_llvm", inp)
        if p.returncode != 0 or not p.stdout.strip():
            raise _runtime.CertFailure(
                "codegen",
                _runtime.FailureCategory.ELABORATION_ERROR,
                f"lower_target_to_llvm failed (rc={p.returncode}): {p.stderr[-500:]}",
            )
        llvm_path = paths.generated / "lowered.llvm.mlir"
        llvm_path.write_text(p.stdout, encoding="utf-8")
        entry["lower_target_to_llvm"] = "pass"

        # A whole-model mesh certificate is stronger than the generic OOT K-ladder: the exact LLVM it
        # is about must contain an instruction claimed by the target accelerator decoder.  Verilator
        # executing an RV64 program is not sufficient by itself -- a CPU-only kernel can compute the
        # injected operands correctly and never touch the accelerator.  Decode runner-side, persist the
        # full trace, and fail before the oracle if the artifact does not drive the accelerator.
        if require_accelerator_trace:
            from . import trace_check as _tck
            from .rocc import decode as _rd

            try:
                trace = _rd.decode_text(p.stdout, source=str(llvm_path), target=target)
                trace_path = paths.generated / "instruction_trace.json"
                trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
                _runtime.schemas.validate(trace, "instruction_trace", contract=contract)
            except Exception as exc:  # noqa: BLE001 -- decode/schema failure is absence of proof
                # Decoder and schema failures are harness-visible absence of proof, never an internal
                # crash/pass.
                raise _runtime.CertFailure(
                    "trace_check",
                    _runtime.FailureCategory.PROTOCOL_VIOLATION,
                    f"exact lowered LLVM could not be decoded as an accelerator trace: "
                    f"{type(exc).__name__}: {str(exc)[-500:]}",
                ) from exc
            drives = bool(_tck.drives_accelerator(trace))
            ins = trace.get("instructions", []) if isinstance(trace, dict) else []
            trace_check = {
                "required": True,
                "status": "pass" if drives else "fail",
                "drives_accelerator": drives,
                "n_instructions": len(ins),
                "classes": sorted({str(i.get("class")) for i in ins if isinstance(i, dict) and i.get("class")}),
            }
            if not drives:
                raise _runtime.CertFailure(
                    "trace_check",
                    _runtime.FailureCategory.PROTOCOL_VIOLATION,
                    "exact lowered LLVM emitted no decoded accelerator instruction; "
                    "a CPU-only program cannot certify a model mesh dispatch",
                )

        from merlin.llvmlower import toolchain as llvm_tc

        if llvm_tc.available():
            try:
                # target= so the object is built for the ISA this target's own harness recipe
                # declares; a default march is the other half of an ELF built for a different core.
                obj = _runtime.oot_compile.llvm_mlir_to_object(p.stdout, paths.generated, target=target)
                artifacts_recorded["object"] = obj.exists()
            except Exception as e:
                raise _runtime.CertFailure(
                    "codegen",
                    _runtime.FailureCategory.ELABORATION_ERROR,
                    f"compile of lowered LLVM to RV64 object failed: {str(e)[-800:]}",
                ) from e
        else:
            artifacts_recorded["object"] = False  # toolchain absent; K6 compile deferred

        # K7/K8: oracle (skip-if-unavailable)
        if gem.available(simulator):
            try:
                # The SAME operands the reference and the simulator were given. Without this the
                # device materialized every leaf from its name while K5 above compared reference and
                # simulate over the INJECTED values, so any caller injecting real operands failed the
                # three-way gate by construction -- and the failure was attributed to the target.
                res = _runtime.oot_compile.run_on_oracle(
                    cb,
                    p.stdout,
                    simulator=simulator,
                    target=target,
                    workdir=paths.generated,
                    timeout=timeout,
                    inputs=inputs,
                )
            except Exception as e:
                raise _runtime.CertFailure(
                    "oracle_rtl",
                    _runtime.FailureCategory.TOOL_CRASH,
                    f"oracle {simulator} invocation failed: {str(e)[-800:]}",
                ) from e
            ok = outputs_match(res["outputs"], ref) and outputs_match(res["outputs"], sim)
            oracle_outputs = res["outputs"]  # what the mesh actually produced (bit-exact == ref when ok)
            oracle = {
                "kind": res["oracle"].get("kind"),
                "engine": simulator,
                "derived_from_rtl": res["oracle"].get("derived_from_rtl", False),
                "cycle_accurate": simulator in _runtime._CYCLE_ACCURATE_SIMULATORS and ok,
                "result": "pass" if ok else "fail",
                "cycles": res.get("cycles"),
            }
            if res.get("console") is not None:
                cpath = paths.artifacts_dir / "console.log"
                cpath.write_text(res["console"], encoding="utf-8")
            if not ok:
                raise _runtime.CertFailure(
                    "oracle_rtl",
                    _runtime.FailureCategory.FUNCTIONAL_MISMATCH,
                    f"oracle {simulator} output != reference == simulate (three-way bit-exact gate)",
                )
        else:
            oracle["result"] = "skipped"
            oracle["kind"] = f"{simulator}_unavailable"

        artifact_identity = _runtime._cert_artifact_identity(paths, run_id)
        if require_accelerator_trace and oracle.get("result") == "pass":
            required_identity = {
                "input_interface",
                "command_buffer",
                "lowered_llvm",
                "kernel_object",
                "package_kernel_elf",
                "instruction_trace",
            }
            missing_identity = sorted(required_identity - set(artifact_identity.get("artifacts", {})))
            if missing_identity:
                raise _runtime.CertFailure(
                    "artifact_identity",
                    _runtime.FailureCategory.PROTOCOL_VIOLATION,
                    "successful model mesh cert is missing exact artifact identity for " + ", ".join(missing_identity),
                )
            trace_check["artifact_sha256"] = artifact_identity["artifacts"]["instruction_trace"]["sha256"]

        verify_inputs()
        if pkg.tool.resolve().is_relative_to(pkg.directory.resolve()):
            package_input_identity = {
                "version": 1,
                "scope": "package-payload",
                "external_dependency_closure": "not-attested",
                "compiler_package_id": compiler_package_id,
                "source": source_inventory,
                "execution": execution_inventory,
                "tool": pkg.tool.resolve().relative_to(pkg.directory.resolve()).as_posix(),
                "execution_path": str(pkg.directory),
                "environment": {
                    "toolchain_shas": shas,
                    "python": sys.version,
                    "executable": sys.executable,
                    "python_bytecode": "disabled",
                },
            }

    except _runtime.CertFailure as cf:
        status = "fail"
        failure = {"plane": cf.plane, "category": cf.category.value, "detail": cf.detail}
    except Exception as e:  # pragma: no cover - internal harness bug
        status = "error"
        failure = {
            "plane": "runner_internal",
            "category": _runtime.FailureCategory.RUNNER_CRASH.value,
            "detail": f"{type(e).__name__}: {e}",
        }

    if not artifact_identity:
        artifact_identity = _runtime._cert_artifact_identity(paths, run_id)

    _runtime._record(
        paths,
        run_id,
        rung,
        simulator,
        status,
        cb,
        shas,
        oracle,
        entry,
        semantic,
        artifacts_recorded,
        failure,
        seed,
        target,
    )

    results = {
        "status": status,
        "artifact_type": "mlir_oot_target_backend",
        "target": target,
        "rung": rung,
        "run_id": run_id,
        "contract": {
            "version": _runtime.CONTRACT_VERSION,
            "package_valid": failure is None or (failure.get("plane") not in ("contract",)),
        },
        "entrypoints": entry,
        "semantic_checks": semantic,
        "oracle": oracle,
        "trace_check": trace_check,
        "artifact_identity": artifact_identity,
        "package_input_identity": package_input_identity,
        "artifacts_recorded": artifacts_recorded,
        "failure": failure,
    }
    (paths.run_path / "results.yaml").write_text(yaml.safe_dump(results, sort_keys=False), encoding="utf-8")
    try:
        _runtime.schemas.validate(results, "result", contract=contract)
    except _runtime.schemas.ContractViolation as e:  # pragma: no cover - shape bug
        sys.stderr.write(f"WARNING: results.yaml self-validation failed: {e}\n")
    # expose the mesh's actual outputs to in-process callers (NOT persisted to results.yaml / validated),
    # so a whole-model executor can thread a matmul layer's real on-mesh result to the next layer.
    results["oracle_outputs"] = oracle_outputs
    return results


def _record(
    paths: _runtime.RunPaths,
    run_id: str,
    rung: str,
    simulator: str,
    status: str,
    cb: dict | None,
    shas: dict,
    oracle: dict,
    entry: dict,
    semantic: dict,
    artifacts_recorded: dict,
    failure: dict | None,
    seed: int,
    target: str,
) -> None:
    """Write the run_manifest + artifact records + FailureRecord (the attributable ledger)."""
    cycle_accurate = simulator in _runtime._CYCLE_ACCURATE_SIMULATORS and oracle.get("result") == "pass"
    manifest = {
        "schema_version": "1.0",
        "project": "merlin",
        "suite": _runtime.SUITE,
        "method": run_id,
        "seed": seed,
        "run_id": run_id,
        "target": target,
        "benchmark": rung,
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "status": status,
        "codegen_backend": "oot_package",
        "metadata": {
            "oracle": {"kind": oracle.get("kind"), "derived_from_rtl": oracle.get("derived_from_rtl", False)},
            "toolchain_shas": shas,
            "cycle_accurate": cycle_accurate,
            "cycles": oracle.get("cycles"),
            "contract_version": _runtime.CONTRACT_VERSION,
            "entrypoints": entry,
            "semantic_checks": semantic,
        },
    }
    (paths.run_path / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    logger = _runtime.EvalRunLogger.start(
        project="merlin",
        suite=_runtime.SUITE,
        target=target,
        method=run_id,
        seed=seed,
        run_id=run_id,
        run_path=paths.run_path,
        tracking_mode="local",
    )
    logger.log_params(
        {
            "rung": rung,
            "simulator": simulator,
            "oracle_kind": oracle.get("kind"),
            "derived_from_rtl": oracle.get("derived_from_rtl", False),
            "cycle_accurate": cycle_accurate,
            **{f"sha.{k}": v for k, v in shas.items()},
        }
    )
    logger.log_metrics({"correct": int(status == "pass"), "cycles": int(oracle.get("cycles") or 0)})
    logger.log_event("oot.certify", {"rung": rung, "simulator": simulator, "status": status})

    store = _runtime.ArtifactStore(paths.run_path, run_id)
    origin_map = [
        (paths.generated / "input.interface.mlir", _runtime.ArtifactOrigin.GENERATED, "interface_mlir"),
        (paths.generated / "lowered.target.mlir", _runtime.ArtifactOrigin.COMPILER_GENERATED, "target_mlir"),
        (paths.generated / "command_buffer.json", _runtime.ArtifactOrigin.COMPILER_GENERATED, "command_buffer"),
        (paths.generated / "lowered.llvm.mlir", _runtime.ArtifactOrigin.COMPILER_GENERATED, "llvm_ir"),
        (paths.generated / "kernel.o", _runtime.ArtifactOrigin.COMPILER_GENERATED, "object"),
        (paths.generated / "package_kernel.elf", _runtime.ArtifactOrigin.COMPILER_GENERATED, "executable"),
        (paths.generated / "instruction_trace.json", _runtime.ArtifactOrigin.COMPILER_GENERATED, "instruction_trace"),
        (paths.artifacts_dir / "console.log", _runtime.ArtifactOrigin.ORACLE_OUTPUT, "log"),
    ]
    for p, origin, kind in origin_map:
        if p.exists():
            store.record(p, origin, kind=kind)

    if failure is not None:
        fr = _runtime.FailureRecord(
            category=_runtime.FailureCategory(failure["category"]),
            detail=failure["detail"],
            failure_id=f"{run_id}-{failure['plane']}",
            likely_cause=failure["plane"],
        )
        (paths.logs / "failures.jsonl").write_text(
            json.dumps(dataclasses.asdict(fr), default=str) + "\n", encoding="utf-8"
        )

    logger.finish(status="pass" if status == "pass" else "fail")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Out-of-tree target-backend package runner")
    ap.add_argument("--contract", default="merlin/contract")
    ap.add_argument("--package", required=True)
    ap.add_argument("--input", required=True, help="path to an *.interface.mlir")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--simulator", default="spike", choices=["spike", "gsim", "verilator"])
    ap.add_argument("--runs-root", help="run root; defaults to the declared package target's contract runs")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args(argv)

    if args.runs_root is None:
        from merlin.common.paths import runs_root

        target = _runtime._package_target(args.package)
        if Path(target).name != target or target in {".", ".."}:
            raise ValueError("package target must be one safe path component")
        args.runs_root = str(runs_root(target, "contract"))

    results = _runtime.certify(
        args.package,
        args.input,
        runs_root=args.runs_root,
        run_id=args.run_id,
        simulator=args.simulator,
        contract=args.contract,
        timeout=args.timeout,
    )
    print(yaml.safe_dump(results, sort_keys=False))
    return 0 if results["status"] == "pass" else 1
