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

SUITE = "gemmini-contract"  # target-ok: aet suite identity of the gemmini reference contract cert flow
# The target under certification is DERIVED per run from the package's manifest ``target`` field (via
# ``_package_target`` / the ``target=`` param of ``certify``), never hardcoded — the runner is
# target-agnostic and threads that value through the run record / logger calls below. The SUITE label
# above is the fixed aet identity of the gemmini reference contract suite and is intentionally kept.
DEFAULT_TARGET = "unknown"  # fallback only when a package manifest declares no ``target`` field
CONTRACT_VERSION = "0.1"

# Cycle-accurate RTL SIMULATOR tools — a property of the simulator TOOL, not of any target. A tier
# graded by one of these carries a cycle-accurate cert; a functional tier (spike / the arc coarse
# model) does not. Extensible as data: a new cycle-accurate sim adds its tool name here. These are
# simulator tool names, never target names, so no target is baked by keying on the set.
_CYCLE_ACCURATE_SIMULATORS = frozenset({"verilator", "vcs"})

# Reference/oracle-ACCESS markers forbidden in a non-exempt package's tool sources (integrity scan; see
# merlin/contract/integrity_policy.md). These are specific dotted paths — matched as substrings across
# ALL languages because they name the actual reference/oracle surface, not a common word.
_FORBIDDEN = (
    "merlin.runtime.reference", "merlin.runtime.simulator", "reference_outputs",
)
# A Python package importing the harness itself is caught STRUCTURALLY (AST), not by a substring: the old
# ``"from merlin" in text`` check false-flagged prose — a docstring "Lowering from merlin_iface …" or a
# comment — as an import. Only a real ``import merlin`` / ``from merlin[.…] import`` statement counts.
_SRC_SUFFIXES = (".py", ".cpp", ".cc", ".h", ".hpp", ".td", ".sh")


# The one merlin module a submission MAY import: the PUBLIC input-dialect grammar. The benchmark's input
# format is the fixed public contract ("Reading the contract bundle — grammar, schemas" is fair game), and
# the shipped oot_starterkit tells agents to parse the input via this typed dialect rather than regex-scrape
# it. Importing it is "using the interface", not reading the ANSWER — so it is exempt while every other
# merlin import (the reference/simulator/lowering/oracle that COMPUTES the expected result) stays forbidden.
_INPUT_DIALECT_EXEMPT = "merlin.xdsl_dialects.interface"


def _is_input_dialect(mod: str) -> bool:
    return mod == _INPUT_DIALECT_EXEMPT or mod.startswith(_INPUT_DIALECT_EXEMPT + ".")


def _py_imports_merlin(text: str) -> str | None:
    """Return the offending module name iff ``text`` (Python source) actually imports a NON-EXEMPT ``merlin``
    harness module, else None. AST-based: docstrings, comments, and unrelated names like ``merlin_iface``
    never match. The public input dialect (:data:`_INPUT_DIALECT_EXEMPT`) is allowed (using the interface,
    not reading the answer). Unparseable source returns None — a syntax error is the build gate's job."""
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if (a.name == "merlin" or a.name.startswith("merlin.")) and not _is_input_dialect(a.name):
                    return a.name
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                continue
            mod = node.module or ""
            if not (mod == "merlin" or mod.startswith("merlin.")):
                continue
            if _is_input_dialect(mod):
                continue
            # `from merlin.xdsl_dialects import interface` resolves each imported name to its FQN; allow only
            # when EVERY name is the exempt input dialect, else flag the module (e.g. a `lowering` sibling).
            fqns = [f"{mod}.{a.name}" for a in node.names]
            if fqns and all(_is_input_dialect(f) for f in fqns):
                continue
            return mod
    return None


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


def usable_cmake() -> str:
    """The first cmake on PATH that actually RUNS, falling back to known-good locations.

    Sourcing the chipyard/Vitis environment — which the FireSim path requires — prepends
    Xilinx's bundled toolchain to PATH, and that ships a cmake 3.3.2 linked against a
    `libidn.so.11` no current distro has. ``shutil.which`` finds it, every configure step then
    dies with a loader error, and nothing in the message mentions Xilinx or PATH. So probe
    ``--version`` rather than trusting the first hit, and prefer a system cmake if the winner
    is broken. Returns "cmake" if nothing works, so the caller still fails with cmake's own
    error rather than ours.
    """
    import os
    import shutil

    seen: list[str] = []
    for cand in (shutil.which("cmake"), "/usr/bin/cmake", "/usr/local/bin/cmake"):
        if not cand or cand in seen or not os.access(cand, os.X_OK):
            continue
        seen.append(cand)
        try:
            if subprocess.run([cand, "--version"], capture_output=True,
                              timeout=30).returncode == 0:
                return cand
        except (OSError, subprocess.SubprocessError):
            continue
    return "cmake"


_usable_cmake = usable_cmake


def build_package(pkg: Package, *, timeout: int = 1800) -> None:
    """If the manifest declares a build block (C++ packages), run configure + build.

    Runs each step FROM THE PACKAGE ROOT (cwd=pkg.directory) with the toolchain env exported, so a
    natural manifest using RELATIVE paths (e.g. `bash build.sh`, `cmake --build mlir_oot/build`) or
    toolchain env vars ($CM / $MLIR_DIR / $LLVM_DIR / $MERLIN_CLANG) builds correctly in the graded
    copy — not only manifests that hard-code absolute {package}/{mlir_dir} placeholders. The grade copies
    the package WITHOUT the build/ tree, so a CLEAN configure must be possible (a `configure` step, or a
    self-configuring `command`). A step may be a list argv (preferred) or a shell string."""
    build = pkg.manifest.get("build")
    if not build:
        return
    import os, shlex, shutil
    from .contract import toolchain as mlir_tc
    mlir_dir = str(mlir_tc.mlir_cmake_dir())
    llvm_dir = str(mlir_tc.mlir_install() / "lib" / "cmake" / "llvm")
    subst = {"{package}": str(pkg.directory.resolve()),
             "{mlir_dir}": mlir_dir, "{llvm_dir}": llvm_dir}
    # toolchain locations the manifest may reference by env var. These are TOOLCHAIN paths, not answers —
    # the reference manifest gets the same values via {mlir_dir}/{llvm_dir} placeholders, so this is parity.
    cmake = _usable_cmake()
    env = dict(os.environ)
    env.setdefault("MLIR_DIR", mlir_dir)
    env.setdefault("LLVM_DIR", llvm_dir)
    env.setdefault("CM", cmake)
    env.setdefault("CMAKE", cmake)
    # The broker rebuilds the package fresh in an env that may lack libidn.so.11 (a conda-era SONAME no
    # current distro ships), which some cmake/git/curl in the package's build.sh dlopens -> the build dies
    # before it even configures. The agent sandbox already gets the .compat_lib libidn.so.11->.12 shim; the
    # broker rebuild must too, or a C++ package that builds fine for the agent fails only at grading. Same
    # single shim dir, so this is parity, not a new capability.
    from .sandbox.toolchain import COMPAT_LIB
    if os.path.isdir(COMPAT_LIB):
        env["LD_LIBRARY_PATH"] = COMPAT_LIB + (os.pathsep + env["LD_LIBRARY_PATH"]
                                               if env.get("LD_LIBRARY_PATH") else "")
    # A manifest is free to spell the step as a bare `cmake` (the reference ones do), and a
    # shell-string step can name it anywhere in a pipeline, so exporting $CM is not enough —
    # put the working cmake's directory FIRST on the child's PATH.
    if os.path.dirname(cmake) and cmake != shutil.which("cmake"):
        env["PATH"] = os.path.dirname(cmake) + os.pathsep + env.get("PATH", "")

    def _resolve(a: str) -> str:
        for k, v in subst.items():
            a = a.replace(k, v)
        return os.path.expandvars(a)   # expand $CM / $MLIR_DIR / ... from env

    for key in ("configure", "command"):
        step = build.get(key)
        if not step:
            continue
        if isinstance(step, str):
            argv = shlex.split(_resolve(step))
        else:
            argv = [_resolve(a) for a in step]
        # run FROM the package dir so relative manifest paths resolve against the package root
        proc = subprocess.run(argv, cwd=str(pkg.directory), env=env,
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            tail = (proc.stdout or "")[-800:] + (proc.stderr or "")[-2000:]
            raise CertFailure("build", FailureCategory.ELABORATION_ERROR,
                              f"package build step {key} failed (rc={proc.returncode}):\n"
                              f"$ {' '.join(argv)}\n{tail}")


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
        for needle in _FORBIDDEN:                       # reference/oracle-access dotted paths (any lang)
            if needle in text:
                raise CertFailure("integrity", FailureCategory.FORBIDDEN_PATTERN,
                                  f"integrity violation in {src.name}: contains {needle!r} "
                                  f"(a non-exempt package must not read the reference/oracle)")
        if src.suffix == ".py":                         # real merlin-harness import (AST, not substring)
            mod = _py_imports_merlin(text)
            if mod is not None:
                raise CertFailure("integrity", FailureCategory.FORBIDDEN_PATTERN,
                                  f"integrity violation in {src.name}: imports {mod!r} "
                                  f"(a non-exempt package must not import the harness/reference)")


# The 4th entrypoint was renamed lower_target_to_llvm -> emit_target_artifact (it emits the target's
# codegen artifact, which for a SIMT/other target is not LLVM). Either name resolves to whichever the
# package's manifest declares, so old and new packages both work.
_ENTRYPOINT_ALIASES = {"emit_target_artifact": "lower_target_to_llvm",
                       "lower_target_to_llvm": "emit_target_artifact"}


def _resolve_argv(pkg: Package, name: str, input_mlir: Path, output_json: Path | None) -> list[str]:
    commands = pkg.manifest["commands"]
    if name not in commands and _ENTRYPOINT_ALIASES.get(name) in commands:
        name = _ENTRYPOINT_ALIASES[name]      # back-compat: package declares the other spelling
    template = commands[name]["argv"]
    out: list[str] = []
    for tok in template:
        tok = tok.replace("{tool}", str(pkg.tool))
        tok = tok.replace("{input_mlir}", str(input_mlir))
        if output_json is not None:
            tok = tok.replace("{output_json}", str(output_json))
        # Robustness: a package may reference its OWN tool by a bare/relative path (e.g. ``atlas-opt``,
        # ``./atlas-opt``, ``submission/<tool>``) instead of the ``{tool}`` placeholder. Steps run with
        # cwd=pkg.directory, so a "submission/"-prefixed or otherwise-misrooted reference does not resolve
        # and the run fails on a manifest path-format nit rather than the compiler logic. If a token names
        # the SAME file as the declared tool (basename match) but does not exist as written from the package
        # root, rewrite it to the absolute tool path. Never touches {input_mlir}/{output_json} (different
        # basenames) or a real, correctly-rooted sibling reference (those exist, so are left as-is).
        if tok != str(pkg.tool) and Path(tok).name == Path(str(pkg.tool)).name \
                and not (pkg.directory / tok).exists():
            tok = str(pkg.tool)
        out.append(tok)
    return out


def _needs_interpreter(pkg: Package, argv: list[str]) -> bool:
    """True iff argv[0] is the package's own Python tool declared as a bare path without the execute
    bit — a very common shape (``language: python`` + ``argv: ["{tool}", ...]`` written 0644). Exec'ing
    such a path directly raises PermissionError; run it through the interpreter instead. Only fires when
    argv[0] IS the tool (not an interpreter the package already prepended) and the tool is a real,
    non-executable ``.py``/py-shebang file — so a compiled binary or a chmod+x script is untouched."""
    if not argv or argv[0] != str(pkg.tool):
        return False
    tool = pkg.tool
    try:
        if not tool.is_file() or (tool.stat().st_mode & 0o111):
            return False  # missing, or already executable — leave as-is
    except OSError:
        return False
    if pkg.language.lower() == "python" or tool.suffix == ".py":
        return True
    try:
        return tool.open("rb").readline(2) == b"#!" and b"python" in tool.open("rb").readline(128)
    except OSError:
        return False


def run_entrypoint(pkg: Package, name: str, input_mlir: Path,
                   output_json: Path | None = None, *, timeout: int = 600) -> subprocess.CompletedProcess:
    """Invoke one entrypoint as a subprocess (never imports the package).

    Runs FROM THE PACKAGE ROOT, like the build steps and like :func:`_resolve_argv` already documents
    ("Steps run with cwd=pkg.directory"). This used to inherit the CALLER's cwd, which made an
    entrypoint's exit status depend on who invoked it: the self-check runs from the workspace root
    (where a ``submission/``-prefixed path happens to resolve) while the grader runs from elsewhere
    (where it does not). The same submission then self-reported passes and graded 0 — the agent spent
    a round optimising against a signal that could not predict its own grade. An explicit cwd makes a
    misrooted path fail identically in both, so the feedback is truthful and early.

    Paths are absolutised first, so pinning the cwd cannot break a caller that passed them relative.
    """
    input_mlir = Path(input_mlir).resolve()
    output_json = Path(output_json).resolve() if output_json is not None else None
    argv = _resolve_argv(pkg, name, input_mlir, output_json)
    if _needs_interpreter(pkg, argv):
        argv = [sys.executable, *argv]
    return subprocess.run(argv, cwd=str(pkg.directory), capture_output=True, text=True,
                          timeout=timeout)


# --------------------------------------------------------------------------- certification


def _package_target(package_dir: str | Path, default: str = DEFAULT_TARGET) -> str:
    """Peek a package's declared ``target`` from its manifest, fail-lenient.

    The run identity (RunSpec / run record / logger) must be fixed BEFORE the package is
    load-validated, so read the manifest ``target`` directly rather than through ``load_package``
    (which raises on any contract violation). A package that can't even be peeked still gets a run
    dir and a fail-closed record from ``load_package`` inside ``certify``.
    """
    man_path = Path(package_dir) / "manifest.yaml"
    try:
        manifest = yaml.safe_load(man_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return default
    if isinstance(manifest, dict) and manifest.get("target"):
        return str(manifest["target"])
    return default


def certify(package_dir: str | Path, interface_mlir: str | Path, *, runs_root: str | Path,
            run_id: str, simulator: str = "spike", contract: str | Path | None = None,
            seed: int = 0, timeout: int = 600, target: str | None = None,
            inputs: dict | None = None) -> dict[str, Any]:
    """Run the K-ladder for one (package, interface input) and record an aet run dir.

    Returns the results dict (also written as results.yaml). Never raises for a package/gate
    failure — those are recorded as status: fail with a plane-routed FailureRecord; only an
    internal harness bug raises.

    ``target`` labels the run; when omitted it is derived from the package manifest's ``target``
    field, so the runner is target-agnostic rather than hardcoded to the reference target.
    """
    from ..runtime.backends import base as _bk
    from ..runtime.reference import reference_outputs, outputs_match
    from ..runtime.simulator import simulate
    from .provenance import toolchain_shas

    interface_mlir = Path(interface_mlir)
    rung = interface_mlir.stem.split(".")[0]
    if target is None:
        target = _package_target(package_dir)

    spec = RunSpec(project="merlin", suite=SUITE, method=f"{run_id}", seed=seed, run_id=run_id,
                   project_root=Path(runs_root), tracking_mode="local", target=target,
                   dtype="i8xi8_i32", benchmark=rung)
    paths = RunPaths.from_spec(spec, run_id)
    for dd in (paths.run_path, paths.logs, paths.artifacts_dir, paths.generated, paths.contracts):
        dd.mkdir(parents=True, exist_ok=True)

    entry = {"parse": "skipped", "lower_interface_to_target": "skipped",
             "emit_command_buffer": "skipped", "lower_target_to_llvm": "skipped"}
    semantic = {"reference_outputs_vs_simulate": "skipped"}
    oracle = {"kind": "none", "derived_from_rtl": False, "cycle_accurate": False,
              "result": "skipped", "cycles": None}
    oracle_outputs: dict | None = None      # the mesh's actual output values (for in-process callers)
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
        pkg = load_package(package_dir, contract=contract)
        integrity_scan(pkg)
        build_package(pkg)
        if not pkg.tool.exists():
            raise CertFailure("build", FailureCategory.ELABORATION_ERROR,
                              f"package tool not found after build: {pkg.tool}")

        # Resolve the package's runtime backend from the run's target (not a name literal) — the
        # runner is target-agnostic, so a package for any registered target reaches its own backend
        # helpers. Done AFTER load_package so a broken/unknown package fails closed on the manifest
        # (K0) rather than raising here; an unregistered target is a fail-closed contract violation.
        try:
            gem = _bk.get_backend(target)
        except KeyError as e:
            raise CertFailure("contract", FailureCategory.STRUCTURAL_INVARIANT_VIOLATION,
                              f"package declares target {target!r} with no registered backend") from e

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

        # K5 (L0): reference == simulate, always — over the SAME (optionally injected) operands
        ref = reference_outputs(cb, inputs)
        sim = simulate(cb, inputs)["outputs"]
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
                res = oot_compile.run_on_oracle(cb, p.stdout, simulator=simulator, target=target,
                                                workdir=paths.generated, timeout=timeout)
            except Exception as e:
                raise CertFailure("oracle_rtl", FailureCategory.TOOL_CRASH,
                                  f"oracle {simulator} invocation failed: {str(e)[-800:]}") from e
            ok = outputs_match(res["outputs"], ref) and outputs_match(res["outputs"], sim)
            oracle_outputs = res["outputs"]      # what the mesh actually produced (bit-exact == ref when ok)
            oracle = {"kind": res["oracle"].get("kind"),
                      "derived_from_rtl": res["oracle"].get("derived_from_rtl", False),
                      "cycle_accurate": simulator in _CYCLE_ACCURATE_SIMULATORS and ok,
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
            artifacts_recorded, failure, seed, target)

    results = {
        "status": status, "artifact_type": "mlir_oot_target_backend", "target": target,
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
    # expose the mesh's actual outputs to in-process callers (NOT persisted to results.yaml / validated),
    # so a whole-model executor can thread a matmul layer's real on-mesh result to the next layer.
    results["oracle_outputs"] = oracle_outputs
    return results


def _record(paths: RunPaths, run_id: str, rung: str, simulator: str, status: str,
            cb: dict | None, shas: dict, oracle: dict, entry: dict, semantic: dict,
            artifacts_recorded: dict, failure: dict | None, seed: int, target: str) -> None:
    """Write the run_manifest + artifact records + FailureRecord (the attributable ledger)."""
    cycle_accurate = simulator in _CYCLE_ACCURATE_SIMULATORS and oracle.get("result") == "pass"
    manifest = {
        "schema_version": "1.0", "project": "merlin", "suite": SUITE, "method": run_id,
        "seed": seed, "run_id": run_id, "target": target, "benchmark": rung,
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

    logger = EvalRunLogger.start(project="merlin", suite=SUITE, target=target,
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
    ap.add_argument("--runs-root", default="out/runs/gemmini_contract")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args(argv)

    results = certify(args.package, args.input, runs_root=args.runs_root, run_id=args.run_id,
                      simulator=args.simulator, contract=args.contract, timeout=args.timeout)
    print(yaml.safe_dump(results, sort_keys=False))
    return 0 if results["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
