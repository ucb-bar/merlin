"""OOT package loading, integrity and process invocation, independent of evaluation telemetry.

The legacy ``oot_runner`` name aliases this module so class identities and monkeypatches
remain stable. Certification is provided lazily by the optional experiments distribution.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from merlin.common.access import PUBLIC_INPUT_MODULE, is_harness_module, is_public_input_module

from .contract import compile as oot_compile  # noqa: F401 -- legacy evaluator monkeypatch seam
from .contract import schemas

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
_CYCLE_ACCURATE_SIMULATORS = frozenset({"gsim", "verilator", "vcs"})

# A package root IS the submission directory, so an argv token rooted at ``submission/`` is doubly
# rooted. Only these prefixes are eligible for the strip below; every other token is left untouched.
_SUBMISSION_PREFIXES = ("./submission/", "submission/")

# Reference/oracle-ACCESS markers forbidden in a non-exempt package's tool sources (integrity scan; see
# merlin/contract/integrity_policy.md). These are specific dotted paths — matched as substrings across
# ALL languages because they name the actual reference/oracle surface, not a common word.
_FORBIDDEN = (
    "merlin.runtime.reference",
    "merlin.runtime.simulator",
    "reference_outputs",
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
_INPUT_DIALECT_EXEMPT = PUBLIC_INPUT_MODULE


def _is_input_dialect(mod: str) -> bool:
    return is_public_input_module(mod)


def _py_imports_merlin(text: str) -> str | None:
    """Return the offending module name iff source imports a non-exempt core/extension harness module.
    The historical function name remains compatible. AST-based: docstrings, comments and ``merlin_iface``
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
                if is_harness_module(a.name) and not _is_input_dialect(a.name):
                    return a.name
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                continue
            mod = node.module or ""
            if not is_harness_module(mod):
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

    def __init__(self, plane: str, category, detail: str):
        super().__init__(detail)
        self.plane = plane
        self._category = category
        self.detail = detail

    @property
    def category(self):
        """Canonical AET enum when available; a value-compatible token in core-only installs."""
        if not isinstance(self._category, str) or hasattr(self._category, "value"):
            return self._category
        try:
            from aet.core.failures import FailureCategory
        except ModuleNotFoundError as exc:
            if exc.name != "aet" and not (exc.name or "").startswith("aet."):
                raise
            return _CategoryToken(self._category)
        return FailureCategory(self._category)

    @category.setter
    def category(self, value):
        self._category = value


class _CategoryToken(str):
    """An error's transport token, not a second failure taxonomy."""

    @property
    def value(self):
        return str(self)


#: The plane for a failure that is NOT about the graded artifact at all -- the harness could not put a
#: declared input in front of it. It is deliberately NOT one of the submission planes (schema, parse,
#: build, integrity, contract, oracle_*, ...) so that no reader, report or brief can mistake it for a
#: verdict on the submission.
INFRASTRUCTURE_PLANE = "infrastructure"


class InfraCategory(str, Enum):  # noqa: UP042 -- preserve the existing failure enum identity/behavior
    """Categories for :class:`InfraFailure`.

    ``aet``'s :class:`~aet.core.failures.FailureCategory` enumerates ways a SUBMISSION can be wrong --
    every member names something the graded artifact did (a syntax error, a numeric mismatch, a protocol
    violation). A harness that could not stage its own inputs has done nothing of the kind, and borrowing
    one of those names to say so is what this class exists to stop.
    """

    #: The staged capsule cohort is not on disk: never materialized, or collected mid-grade.
    COHORT_NOT_MATERIALIZED = "cohort_not_materialized"

    #: The device readback was REFUSED on structural grounds before any value was compared --
    #: a whole residue class of words came back exactly zero while the rest carried data, which
    #: is a transport defect and not an arithmetic result (see
    #: :mod:`merlin.common.readback_integrity`). The kernel may be perfectly correct; nothing
    #: about it was measured, because the bytes that came back are not its output.
    READBACK_TRANSPORT_REFUSED = "readback_transport_refused"

    def __str__(self) -> str:
        # The three recorders in this repo serialize a category differently -- `cf.category.value`
        # (oot_runner), `str(cf.category)` (capsule_grade) and a `hasattr(..., "value")` probe
        # (capsule_runner). Making __str__ agree with .value keeps the recorded string identical
        # whichever one writes the row, instead of leaking "InfraCategory.COHORT_NOT_MATERIALIZED"
        # into one report and the honest token into another.
        return self.value


class InfraFailure(CertFailure):
    """The HARNESS failed, not the submission -- a declared input was missing, so nothing was measured.

    A subclass of :class:`CertFailure` on purpose: every existing recorder already catches CertFailure
    and writes ``plane``/``category``/``detail``, so an infrastructure fault is recorded honestly through
    the paths that already exist, while a caller that wants to treat it specially (and the per-capsule
    status mapping in ``capsule_runner`` does) can catch this narrower type first.

    Why it exists. A grade resolves the per-target cohort symlink to a concrete staging dir once and then
    reads capsules out of it for the whole grade; when a sibling materialization collected that dir, the
    missing interface MLIR was raised as ``schema / structural_invariant_violation``, and an official
    round-0 verdict recorded 31 of 33 capsules as structurally invalid SUBMISSIONS -- for a package that
    scored 33/34 minutes earlier, with ``gradeable: True`` asserting the number was a real measurement.
    The number was then handed to the next round as the agent's own failure history. A harness fault that
    can wear a verdict's clothes is worse than a crash, because it gets believed and cited.
    """


class BackendDeclined(Exception):
    """The backend STATED that it does not handle this capsule, instead of emitting a wrong program.

    A backend that cannot lower a shape has two ways to say so, and only one of them is legible. It can
    emit a program that writes nothing -- which arrives at the grader as an output full of zeros,
    indistinguishable from arithmetic that ran and was wrong -- or it can decline. Measured: one
    submission chained twelve shape-keyed builders with ``or`` and fell through to a bare terminator, so
    twelve capsules failed as "your artifact does not compute the declared operation" when the artifact
    had never been written. Nothing in the round feedback could say "you declined these shapes", because
    nothing could tell the two apart, and the agent iterated on arithmetic it had not emitted.

    This is the same "decline rather than guess" contract the routing/cost-model layer already uses
    (:mod:`merlin.targetgen.routing`): declining is a legitimate, reportable answer. It is NOT a pass --
    a declined capsule stays in the denominator, uncertified -- but it is not a numeric failure either,
    and the difference is what an agent needs to act on.
    """

    def __init__(self, reason: str, *, shape=None, op: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.shape = list(shape) if shape is not None else None
        self.op = op

    def to_dict(self) -> dict:
        d = {"reason": self.reason}
        if self.shape is not None:
            d["shape"] = self.shape
        if self.op:
            d["op"] = self.op
        return d


# --------------------------------------------------------------------------- package model


@dataclasses.dataclass
class Package:
    directory: Path
    manifest: dict[str, Any]
    tool: Path  # resolved entrypoint tool path
    #: What the package classifier said about ``compiler`` when this package was loaded. Carried so a
    #: recorder can write the verdict down at ``report`` phase, where it blocks nothing: a gate whose
    #: answer is computed and then dropped is indistinguishable from one that never ran.
    compiler_capability: Any = None

    @property
    def provider(self):
        """Optional declared provider metadata; never capability, permission, or trust."""
        from .providers import read_provider

        return read_provider(self.directory)

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


#: The gate name this loader's capability check ships under. Its phase is declared in
#: ``merlin/contract/gate_phases.yaml``, never as a literal here -- this is the LIVE certification
#: path, and a new refusal that lands blocking is indistinguishable from a regression to every agent
#: session already in flight.
CAPABILITY_GATE = "package_capability"


def load_package(package_dir: str | Path, *, contract: str | Path | None = None) -> Package:
    """Load + validate a package manifest (fail-closed). Resolves the entrypoint tool path.

    Also ASKS the package classifier whether this directory provides the ``compiler`` capability at
    all, so "this is not a compiler package" is answered once, by name, instead of surfacing as
    whichever field happened to be missing first. That verdict rides on the returned
    :class:`Package`; whether it REFUSES is decided by :data:`CAPABILITY_GATE`'s declared phase, and
    when it refuses it raises the existing :class:`CertFailure` with the plane and category this
    function already used -- no caller sees a new exception type.
    """
    from ..perf.gate_phase import blocks as _blocks
    from ..perf.gate_phase import configured_phase as _configured_phase
    from . import package as _package

    d = Path(package_dir)
    capability = _package.capability(d, "compiler")
    if _blocks(
        _configured_phase(CAPABILITY_GATE), "absent" if not capability.provided else "provided", failing=("absent",)
    ):
        raise CertFailure("contract", "structural_invariant_violation", capability.explain(d))
    man_path = d / "manifest.yaml"
    if not man_path.is_file():
        raise CertFailure("contract", "structural_invariant_violation", f"no manifest.yaml in package {d}")
    # The manifest is AGENT-AUTHORED, so a syntax error in it is an ordinary grading outcome, not an
    # internal error. Unguarded, `yaml.safe_load` raised straight through `capsule_grade.grade` and
    # killed the whole run: one agent wrote `author:(opencode agent)` (no space after the colon) and a
    # 12-round campaign arm died on round 1 with a ScannerError traceback the agent never saw. The
    # schema check below was already wrapped; the PARSE has to be too, or a malformed file is a lost
    # run instead of feedback the agent can act on.
    try:
        manifest = yaml.safe_load(man_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise CertFailure("contract", "structural_invariant_violation", f"manifest.yaml is not valid YAML: {e}") from e
    if not isinstance(manifest, dict):
        raise CertFailure(
            "contract",
            "structural_invariant_violation",
            f"manifest.yaml must be a mapping, got {type(manifest).__name__}",
        )
    try:
        schemas.validate_manifest(manifest, contract=contract)
    except schemas.ContractViolation as e:
        raise CertFailure("contract", "structural_invariant_violation", str(e)) from e
    # tool path: build.tool_output if a build block is declared, else entrypoints.tool
    build = manifest.get("build")
    tool_rel = build["tool_output"] if build else manifest["entrypoints"]["tool"]
    tool = (d / tool_rel).resolve()
    return Package(directory=d, manifest=manifest, tool=tool, compiler_capability=capability)


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
            if subprocess.run([cand, "--version"], capture_output=True, timeout=30).returncode == 0:
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
    import os
    import shlex
    import shutil

    from .contract import toolchain as mlir_tc

    mlir_dir = str(mlir_tc.mlir_cmake_dir())
    llvm_dir = str(mlir_tc.mlir_install() / "lib" / "cmake" / "llvm")
    subst = {"{package}": str(pkg.directory.resolve()), "{mlir_dir}": mlir_dir, "{llvm_dir}": llvm_dir}
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
    from merlin.common.paths import compat_lib_dir

    COMPAT_LIB = str(compat_lib_dir())
    if os.path.isdir(COMPAT_LIB):
        env["LD_LIBRARY_PATH"] = COMPAT_LIB + (
            os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
    # A manifest is free to spell the step as a bare `cmake` (the reference ones do), and a
    # shell-string step can name it anywhere in a pipeline, so exporting $CM is not enough —
    # put the working cmake's directory FIRST on the child's PATH.
    if os.path.dirname(cmake) and cmake != shutil.which("cmake"):
        env["PATH"] = os.path.dirname(cmake) + os.pathsep + env.get("PATH", "")

    def _resolve(a: str) -> str:
        for k, v in subst.items():
            a = a.replace(k, v)
        return os.path.expandvars(a)  # expand $CM / $MLIR_DIR / ... from env

    for key in ("configure", "command"):
        step = build.get(key)
        if not step:
            continue
        if isinstance(step, str):
            argv = shlex.split(_resolve(step))
        else:
            argv = [_resolve(a) for a in step]
        # run FROM the package dir so relative manifest paths resolve against the package root
        proc = subprocess.run(argv, cwd=str(pkg.directory), env=env, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            tail = (proc.stdout or "")[-800:] + (proc.stderr or "")[-2000:]
            raise CertFailure(
                "build",
                "elaboration_error",
                f"package build step {key} failed (rc={proc.returncode}):\n$ {' '.join(argv)}\n{tail}",
            )


def integrity_scan(pkg: Package, *, additional_forbidden: tuple[str, ...] = ()) -> None:
    """Reject harness/reference access, with optional host-owned stricter markers.

    Supplemental markers belong to one invocation and cannot remove the default
    oracle markers or structural import restrictions. Never read them from a
    candidate manifest or mutate the process-wide defaults for an experiment.
    """
    if pkg.integrity_exempt:
        return
    for src in pkg.directory.rglob("*"):
        if not src.is_file() or src.suffix not in _SRC_SUFFIXES:
            continue
        if "build" in src.relative_to(pkg.directory).parts:  # skip the package's generated build trees
            continue
        text = src.read_text(encoding="utf-8", errors="ignore")
        for needle in (*_FORBIDDEN, *additional_forbidden):  # oracle-access paths (any lang)
            if needle in text:
                raise CertFailure(
                    "integrity",
                    "forbidden_pattern",
                    f"integrity violation in {src.name}: contains {needle!r} "
                    f"(a non-exempt package must not read the reference/oracle)",
                )
        if src.suffix == ".py":  # real merlin-harness import (AST, not substring)
            mod = _py_imports_merlin(text)
            if mod is not None:
                raise CertFailure(
                    "integrity",
                    "forbidden_pattern",
                    f"integrity violation in {src.name}: imports {mod!r} "
                    f"(a non-exempt package must not import the harness/reference)",
                )


# The 4th entrypoint was renamed lower_target_to_llvm -> emit_target_artifact (it emits the target's
# codegen artifact, which for a SIMT/other target is not LLVM). Either name resolves to whichever the
# package's manifest declares, so old and new packages both work.
_ENTRYPOINT_ALIASES = {"emit_target_artifact": "lower_target_to_llvm", "lower_target_to_llvm": "emit_target_artifact"}


def analysis_emission_entrypoints(pkg: Package) -> tuple[str, ...]:
    """Commands needed to obtain both artifacts for host-owned analysis.

    The four experiment ABI commands remain mandatory.  A package may additionally expose
    ``emit_analysis_bundle`` to produce the command buffer at ``{output_json}`` and the target
    artifact on stdout in one compiler process.  Analysis feature-detects that optimization and
    otherwise preserves the two-command protocol for existing packages.
    """
    manifest = getattr(pkg, "manifest", {}) or {}
    commands = manifest.get("commands") or {}
    if "emit_analysis_bundle" in commands:
        return ("emit_analysis_bundle",)
    return ("emit_command_buffer", "lower_target_to_llvm")


def _resolve_argv(pkg: Package, name: str, input_mlir: Path, output_json: Path | None) -> list[str]:
    commands = pkg.manifest["commands"]
    if name not in commands and _ENTRYPOINT_ALIASES.get(name) in commands:
        name = _ENTRYPOINT_ALIASES[name]  # back-compat: package declares the other spelling
    template = commands[name]["argv"]
    # Substituted I/O paths must be ABSOLUTE. Entrypoints run with cwd=pkg.directory, so a caller that
    # passes a repo-relative capsule path would otherwise hand the package a path that cannot resolve
    # from where it actually runs -- trading one misrooting for another.
    input_mlir = Path(input_mlir).resolve()
    output_json = Path(output_json).resolve() if output_json is not None else None
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
        if tok != str(pkg.tool) and Path(tok).name == Path(str(pkg.tool)).name and not (pkg.directory / tok).exists():
            tok = str(pkg.tool)
        # The rescue above only fires for the DECLARED tool's own basename, so it cannot help a package
        # that declares a separate script per command (a shape the schema permits) -- and it is dead
        # entirely when `entrypoints.tool` names an interpreter rather than a script, since no script
        # basename can match `python`. The package root IS the submission directory, so a token rooted
        # at `submission/` is unambiguously double-rooted; strip it when, and only when, the remainder
        # names a real file under the package root.
        elif tok.startswith(_SUBMISSION_PREFIXES):
            _root = getattr(pkg, "directory", None)
            if _root is not None and not (Path(_root) / tok).exists():
                for _pfx in _SUBMISSION_PREFIXES:
                    if tok.startswith(_pfx) and (Path(_root) / tok[len(_pfx) :]).exists():
                        tok = tok[len(_pfx) :]
                        break
        out.append(tok)
    # Fail CLOSED on a placeholder the runner does not substitute. Left alone, an unknown token reaches
    # the package verbatim and surfaces as FileNotFoundError: '{input_json}' from inside the submission's
    # own traceback -- indistinguishable from the package being broken. Measured: a model invented
    # {input_json} for a chained-JSON pipeline, and every capsule reported a cryptic missing file instead
    # of "that placeholder does not exist, here are the ones that do".
    _known = ("{tool}", "{input_mlir}", "{output_json}")
    for tok in out:
        lo = tok.find("{")
        if lo == -1:
            continue
        hi = tok.find("}", lo)
        if hi == -1:
            continue
        raise CertFailure(
            "contract",
            "structural_invariant_violation",
            f"manifest command {name!r}: unsubstituted placeholder {tok[lo : hi + 1]!r} in argv. "
            f"The runner substitutes only {', '.join(_known)}. Every stage receives the interface MLIR "
            f"as {{input_mlir}}; stages are not chained through intermediate JSON.",
        )
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


def run_entrypoint(
    pkg: Package,
    name: str,
    input_mlir: Path,
    output_json: Path | None = None,
    *,
    timeout: int = 600,
    write_bytecode: bool = True,
) -> subprocess.CompletedProcess:
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
    # cwd=pkg.directory is the CONTRACT, not a convenience: _resolve_argv documents "steps run with
    # cwd=pkg.directory" and build_package already honours it, but this path did not pass cwd at all, so
    # every relative argv token resolved against the grader's process CWD (the repo root). A package that
    # declared its entrypoints package-relative -- exactly what the contract describes -- failed every
    # capsule at `parse` with "no such file", naming a file that was present in the submission.
    env = dict(os.environ)
    if not write_bytecode:
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    if "PYTHONPATH" in env:
        # Imported compiler helpers belong to the caller's declared environment. Preserve their
        # meaning when the entrypoint moves into its package; relative import roots otherwise
        # silently rebase and a development-successful compiler fails in the production runner.
        env["PYTHONPATH"] = os.pathsep.join(
            str(Path(value or ".").resolve()) for value in env["PYTHONPATH"].split(os.pathsep)
        )
    return subprocess.run(argv, cwd=str(pkg.directory), env=env, capture_output=True, text=True, timeout=timeout)


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


_CERTIFICATION_EXPORTS = frozenset({"certify", "_record", "_cert_artifact_identity", "main"})
_AET_EXPORTS = {
    "ArtifactOrigin": "aet.core.artifact_store",
    "ArtifactStore": "aet.core.artifact_store",
    "FailureCategory": "aet.core.failures",
    "FailureRecord": "aet.core.failures",
    "RunPaths": "aet.core.run_paths",
    "RunSpec": "aet.core.run_spec",
    "EvalRunLogger": "aet.tracking",
}


def __getattr__(name):
    """Resolve old evaluation attributes only when the caller requests evaluation."""
    import importlib

    if name in _CERTIFICATION_EXPORTS:
        try:
            import merlin.targetgen.package_certification as module
        except ModuleNotFoundError as exc:
            if exc.name != "merlin.targetgen.package_certification":
                raise
            raise ModuleNotFoundError(
                "OOT certification requires the optional merlin-experiments distribution; "
                "install packages/merlin-experiments alongside core"
            ) from exc
        return getattr(module, name)
    if name in _AET_EXPORTS:
        return getattr(importlib.import_module(_AET_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
