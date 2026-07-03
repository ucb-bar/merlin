"""Generic out-of-tree target-backend package runner (experiment ABI v0.1).

Hooks ANY contract-satisfying package into Merlin through the subprocess + file boundary, runs the
K-ladder certification flow, and records each run through the same aet substrate as
``merlin.targetgen.eval.gemmini_suite`` (RunSpec / RunPaths / EvalRunLogger / ArtifactStore / FailureRecord).

A package is invoked ONLY via its CLI entrypoints (it is never imported). Non-exempt packages are
integrity-scanned (no harness imports). Every gate failure is fail-closed and plane-routed.

CLI:
    python -m merlin.targetgen.oot_runner --contract merlin/contract \\
        --package artifacts/targets/gemmini/merlin_native_v0 \\
        --input merlin/contract/examples/g0_matmul.interface.mlir \\
        --run-id contract_smoke_g0 [--simulator spike|verilator] [--runs-root runs/gemmini_contract]
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from aet.core.artifact_store import ArtifactOrigin, ArtifactStore
from aet.core.failures import FailureCategory, FailureRecord
from aet.core.run_paths import RunPaths
from aet.core.run_spec import RunSpec
from aet.tracking import EvalRunLogger

from .contract import schemas
from .contract import compile as oot_compile

SUITE = "gemmini-contract"
CONTRACT_VERSION = "0.1"

# Forbidden substrings in a non-exempt package's tool sources (integrity scan; see
# merlin/contract/integrity_policy.md).
_FORBIDDEN = (
    "merlin.runtime.reference", "merlin.runtime.simulator", "reference_outputs",
    "import merlin", "from merlin",
)
_SRC_SUFFIXES = (".py", ".cpp", ".cc", ".h", ".hpp", ".td", ".sh")


class CertFailure(Exception):
    """A gate failed. Carries the plane + FailureCategory for fail-closed recording."""

    def __init__(self, plane: str, category: FailureCategory, detail: str):
        super().__init__(detail)
        self.plane = plane
        self.category = category
        self.detail = detail


# --------------------------------------------------------------------------- package model


@dataclasses.dataclass
class Package:
    directory: Path
    manifest: dict[str, Any]
    tool: Path                       # resolved entrypoint tool path

    @property
    def target(self) -> str:
        return self.manifest.get("target", "unknown")

    @property
    def package_id(self) -> str:
        return self.manifest.get("package_id", self.directory.name)

    @property
    def language(self) -> str:
        return self.manifest.get("language", "unknown")

    @property
    def integrity_exempt(self) -> bool:
        return bool(self.manifest.get("integrity_exempt", False))


def load_package(package_dir: str | Path, *, contract: str | Path | None = None) -> Package:
    """Load + validate a package manifest (fail-closed). Resolves the entrypoint tool path."""
    d = Path(package_dir)
    man_path = d / "manifest.yaml"
    if not man_path.is_file():
        raise CertFailure("contract", FailureCategory.STRUCTURAL_INVARIANT_VIOLATION,
                          f"no manifest.yaml in package {d}")
    manifest = yaml.safe_load(man_path.read_text(encoding="utf-8"))
    try:
        schemas.validate_manifest(manifest, contract=contract)
    except schemas.ContractViolation as e:
        raise CertFailure("contract", FailureCategory.STRUCTURAL_INVARIANT_VIOLATION, str(e)) from e
    # tool path: build.tool_output if a build block is declared, else entrypoints.tool
    build = manifest.get("build")
    tool_rel = build["tool_output"] if build else manifest["entrypoints"]["tool"]
    tool = (d / tool_rel).resolve()
    return Package(directory=d, manifest=manifest, tool=tool)


def build_package(pkg: Package, *, timeout: int = 1800) -> None:
    """If the manifest declares a build block (C++ packages), run configure + build."""
    build = pkg.manifest.get("build")
    if not build:
        return
    from .contract import toolchain as mlir_tc
    subst = {"{package}": str(pkg.directory.resolve()),
             "{mlir_dir}": str(mlir_tc.mlir_cmake_dir()),
             "{llvm_dir}": str(mlir_tc.mlir_install() / "lib" / "cmake" / "llvm")}
    for key in ("configure", "command"):
        argv = build.get(key)
        if not argv:
            continue
        resolved = []
        for a in argv:
            for k, v in subst.items():
                a = a.replace(k, v)
            resolved.append(a)
        argv = resolved
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise CertFailure("build", FailureCategory.ELABORATION_ERROR,
                              f"package build step {key} failed:\n{proc.stderr[-2000:]}")


def integrity_scan(pkg: Package) -> None:
    """Reject a non-exempt package whose tool sources import the harness / read the reference."""
    if pkg.integrity_exempt:
        return
    for src in pkg.directory.rglob("*"):
        if not src.is_file() or src.suffix not in _SRC_SUFFIXES:
            continue
        if "build" in src.parts:        # skip generated build trees
            continue
        text = src.read_text(encoding="utf-8", errors="ignore")
        for needle in _FORBIDDEN:
            if needle in text:
                raise CertFailure("integrity", FailureCategory.FORBIDDEN_PATTERN,
                                  f"integrity violation in {src.name}: contains {needle!r} "
                                  f"(a non-exempt package must not import the harness/reference)")


def _resolve_argv(pkg: Package, name: str, input_mlir: Path, output_json: Path | None) -> list[str]:
    template = pkg.manifest["commands"][name]["argv"]
    out: list[str] = []
    for tok in template:
        tok = tok.replace("{tool}", str(pkg.tool))
        tok = tok.replace("{input_mlir}", str(input_mlir))
        if output_json is not None:
            tok = tok.replace("{output_json}", str(output_json))
        out.append(tok)
    return out


def run_entrypoint(pkg: Package, name: str, input_mlir: Path,
                   output_json: Path | None = None, *, timeout: int = 600) -> subprocess.CompletedProcess:
    """Invoke one entrypoint as a subprocess (never imports the package)."""
    argv = _resolve_argv(pkg, name, input_mlir, output_json)
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------------- certification


def certify(package_dir: str | Path, interface_mlir: str | Path, *, runs_root: str | Path,
            run_id: str, simulator: str = "spike", contract: str | Path | None = None,
            seed: int = 0, timeout: int = 600) -> dict[str, Any]:
    """Run the K-ladder for one (package, interface input) and record an aet run dir.

    Returns the results dict (also written as results.yaml). Never raises for a package/gate
    failure — those are recorded as status: fail with a plane-routed FailureRecord; only an
    internal harness bug raises.
    """
    from ..runtime.backends import gemmini as gem
    from ..runtime.reference import reference_outputs, outputs_match
    from ..runtime.simulator import simulate
    from ..eval.gemmini_suite import toolchain_shas

    interface_mlir = Path(interface_mlir)
    rung = interface_mlir.stem.split(".")[0]

    spec = RunSpec(project="merlin", suite=SUITE, method=f"{run_id}", seed=seed, run_id=run_id,
                   project_root=Path(runs_root), tracking_mode="local", target="gemmini",
                   dtype="i8xi8_i32", benchmark=rung)
    paths = RunPaths.from_spec(spec, run_id)
    for dd in (paths.run_path, paths.logs, paths.artifacts_dir, paths.generated, paths.contracts):
        dd.mkdir(parents=True, exist_ok=True)

    entry = {"parse": "skipped", "lower_interface_to_target": "skipped",
             "emit_command_buffer": "skipped", "lower_target_to_llvm": "skipped"}
    semantic = {"reference_outputs_vs_simulate": "skipped"}
    oracle = {"kind": "none", "derived_from_rtl": False, "cycle_accurate": False,
              "result": "skipped", "cycles": None}
    artifacts_recorded: dict[str, bool] = {}
    failure: dict[str, Any] | None = None
    status = "pass"
    cb: dict[str, Any] | None = None
    shas = toolchain_shas()

    # input artifact
    inp = paths.generated / "input.interface.mlir"
    inp.write_text(interface_mlir.read_text(encoding="utf-8"), encoding="utf-8")

    try:
        # K0/K1: load + validate manifest, integrity scan, build if needed
        pkg = load_package(package_dir, contract=contract)
        integrity_scan(pkg)
        build_package(pkg)
        if not pkg.tool.exists():
            raise CertFailure("build", FailureCategory.ELABORATION_ERROR,
                              f"package tool not found after build: {pkg.tool}")

        # K2: parse
        p = run_entrypoint(pkg, "parse", inp, timeout=timeout)
        if p.returncode != 0:
            raise CertFailure("runner_invocation", FailureCategory.TOOL_CRASH,
                              f"parse entrypoint exited {p.returncode}: {p.stderr[-500:]}")
        entry["parse"] = "pass"

        # K3: lower_interface_to_target -> non-empty MLIR
        p = run_entrypoint(pkg, "lower_interface_to_target", inp, timeout=timeout)
        if p.returncode != 0 or not p.stdout.strip():
            raise CertFailure("codegen", FailureCategory.ELABORATION_ERROR,
                              f"lower_interface_to_target failed (rc={p.returncode}): {p.stderr[-500:]}")
        target_path = paths.generated / "lowered.target.mlir"
        target_path.write_text(p.stdout, encoding="utf-8")
        entry["lower_interface_to_target"] = "pass"

        # K4: emit_command_buffer -> schema-valid command_buffer.json
        cb_path = paths.generated / "command_buffer.json"
        p = run_entrypoint(pkg, "emit_command_buffer", inp, cb_path, timeout=timeout)
        if p.returncode != 0 or not cb_path.exists():
            raise CertFailure("artifact_class", FailureCategory.STRUCTURAL_INVARIANT_VIOLATION,
                              f"emit_command_buffer produced no command_buffer.json "
                              f"(rc={p.returncode}): {p.stderr[-500:]}")
        try:
            cb = json.loads(cb_path.read_text(encoding="utf-8"))
            schemas.validate_command_buffer(cb, contract=contract)
        except (json.JSONDecodeError, schemas.ContractViolation) as e:
            raise CertFailure("abi_schema", FailureCategory.PROTOCOL_VIOLATION,
                              f"command_buffer.json invalid: {e}") from e
        entry["emit_command_buffer"] = "pass"

        # K5 (L0): reference == simulate, always
        ref = reference_outputs(cb)
        sim = simulate(cb)["outputs"]
        if not outputs_match(ref, sim):
            raise CertFailure("command_buffer_semantics", FailureCategory.FUNCTIONAL_MISMATCH,
                              "reference_outputs(cb) != simulate(cb): the emitted command buffer "
                              "is not internally consistent")
        semantic["reference_outputs_vs_simulate"] = "pass"

        # K6: lower_target_to_llvm -> compile to object/ELF
        p = run_entrypoint(pkg, "lower_target_to_llvm", inp, timeout=timeout)
        if p.returncode != 0 or not p.stdout.strip():
            raise CertFailure("codegen", FailureCategory.ELABORATION_ERROR,
                              f"lower_target_to_llvm failed (rc={p.returncode}): {p.stderr[-500:]}")
        llvm_path = paths.generated / "lowered.llvm.mlir"
        llvm_path.write_text(p.stdout, encoding="utf-8")
        entry["lower_target_to_llvm"] = "pass"

        from merlin.llvmlower import toolchain as llvm_tc
        if llvm_tc.available():
            try:
                obj = oot_compile.llvm_mlir_to_object(p.stdout, paths.generated)
                artifacts_recorded["object"] = obj.exists()
            except Exception as e:
                raise CertFailure("codegen", FailureCategory.ELABORATION_ERROR,
                                  f"compile of lowered LLVM to RV64 object failed: {str(e)[-800:]}") from e
        else:
            artifacts_recorded["object"] = False  # toolchain absent; K6 compile deferred

        # K7/K8: oracle (skip-if-unavailable)
        if gem.available(simulator):
            try:
                res = oot_compile.run_on_oracle(cb, p.stdout, simulator=simulator,
                                                workdir=paths.generated, timeout=timeout)
            except Exception as e:
                raise CertFailure("oracle_rtl", FailureCategory.TOOL_CRASH,
                                  f"oracle {simulator} invocation failed: {str(e)[-800:]}") from e
            ok = outputs_match(res["outputs"], ref) and outputs_match(res["outputs"], sim)
            oracle = {"kind": res["oracle"].get("kind"),
                      "derived_from_rtl": res["oracle"].get("derived_from_rtl", False),
                      "cycle_accurate": simulator == "verilator" and ok,
                      "result": "pass" if ok else "fail", "cycles": res.get("cycles")}
            if res.get("console") is not None:
                cpath = paths.artifacts_dir / "console.log"
                cpath.write_text(res["console"], encoding="utf-8")
            if not ok:
                raise CertFailure("oracle_rtl", FailureCategory.FUNCTIONAL_MISMATCH,
                                  f"oracle {simulator} output != reference == simulate "
                                  f"(three-way bit-exact gate)")
        else:
            oracle["result"] = "skipped"
            oracle["kind"] = f"{simulator}_unavailable"

    except CertFailure as cf:
        status = "fail"
        failure = {"plane": cf.plane, "category": cf.category.value, "detail": cf.detail}
    except Exception as e:  # pragma: no cover - internal harness bug
        status = "error"
        failure = {"plane": "runner_internal", "category": FailureCategory.RUNNER_CRASH.value,
                   "detail": f"{type(e).__name__}: {e}"}

    _record(paths, run_id, rung, simulator, status, cb, shas, oracle, entry, semantic,
            artifacts_recorded, failure, seed)

    results = {
        "status": status, "artifact_type": "mlir_oot_target_backend", "target": "gemmini",
        "rung": rung, "run_id": run_id,
        "contract": {"version": CONTRACT_VERSION, "package_valid": failure is None or
                     (failure.get("plane") not in ("contract",))},
        "entrypoints": entry, "semantic_checks": semantic, "oracle": oracle,
        "artifacts_recorded": artifacts_recorded, "failure": failure,
    }
    (paths.run_path / "results.yaml").write_text(yaml.safe_dump(results, sort_keys=False),
                                                 encoding="utf-8")
    try:
        schemas.validate(results, "result", contract=contract)
    except schemas.ContractViolation as e:  # pragma: no cover - shape bug
        sys.stderr.write(f"WARNING: results.yaml self-validation failed: {e}\n")
    return results


def _record(paths: RunPaths, run_id: str, rung: str, simulator: str, status: str,
            cb: dict | None, shas: dict, oracle: dict, entry: dict, semantic: dict,
            artifacts_recorded: dict, failure: dict | None, seed: int) -> None:
    """Write the run_manifest + artifact records + FailureRecord (the attributable ledger)."""
    cycle_accurate = simulator == "verilator" and oracle.get("result") == "pass"
    manifest = {
        "schema_version": "1.0", "project": "merlin", "suite": SUITE, "method": run_id,
        "seed": seed, "run_id": run_id, "target": "gemmini", "benchmark": rung,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "status": status,
        "codegen_backend": "oot_package",
        "metadata": {
            "oracle": {"kind": oracle.get("kind"), "derived_from_rtl": oracle.get("derived_from_rtl", False)},
            "toolchain_shas": shas,
            "cycle_accurate": cycle_accurate,
            "cycles": oracle.get("cycles"),
            "contract_version": CONTRACT_VERSION,
            "entrypoints": entry, "semantic_checks": semantic,
        },
    }
    (paths.run_path / "run_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    logger = EvalRunLogger.start(project="merlin", suite=SUITE, target="gemmini",
                                 method=run_id, seed=seed, run_id=run_id,
                                 run_path=paths.run_path, tracking_mode="local")
    logger.log_params({"rung": rung, "simulator": simulator,
                       "oracle_kind": oracle.get("kind"),
                       "derived_from_rtl": oracle.get("derived_from_rtl", False),
                       "cycle_accurate": cycle_accurate,
                       **{f"sha.{k}": v for k, v in shas.items()}})
    logger.log_metrics({"correct": int(status == "pass"),
                        "cycles": int(oracle.get("cycles") or 0)})
    logger.log_event("oot.certify", {"rung": rung, "simulator": simulator, "status": status})

    store = ArtifactStore(paths.run_path, run_id)
    origin_map = [
        (paths.generated / "input.interface.mlir", ArtifactOrigin.GENERATED, "interface_mlir"),
        (paths.generated / "lowered.target.mlir", ArtifactOrigin.COMPILER_GENERATED, "target_mlir"),
        (paths.generated / "command_buffer.json", ArtifactOrigin.COMPILER_GENERATED, "command_buffer"),
        (paths.generated / "lowered.llvm.mlir", ArtifactOrigin.COMPILER_GENERATED, "llvm_ir"),
        (paths.generated / "kernel.o", ArtifactOrigin.COMPILER_GENERATED, "object"),
        (paths.artifacts_dir / "console.log", ArtifactOrigin.ORACLE_OUTPUT, "log"),
    ]
    for p, origin, kind in origin_map:
        if p.exists():
            store.record(p, origin, kind=kind)

    if failure is not None:
        fr = FailureRecord(
            category=FailureCategory(failure["category"]),
            detail=failure["detail"], failure_id=f"{run_id}-{failure['plane']}",
            likely_cause=failure["plane"])
        (paths.logs / "failures.jsonl").write_text(
            json.dumps(dataclasses.asdict(fr), default=str) + "\n", encoding="utf-8")

    logger.finish(status="pass" if status == "pass" else "fail")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Out-of-tree target-backend package runner")
    ap.add_argument("--contract", default="merlin/contract")
    ap.add_argument("--package", required=True)
    ap.add_argument("--input", required=True, help="path to an *.interface.mlir")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--simulator", default="spike", choices=["spike", "verilator"])
    ap.add_argument("--runs-root", default="runs/gemmini_contract")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args(argv)

    results = certify(args.package, args.input, runs_root=args.runs_root, run_id=args.run_id,
                      simulator=args.simulator, contract=args.contract, timeout=args.timeout)
    print(yaml.safe_dump(results, sort_keys=False))
    return 0 if results["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
