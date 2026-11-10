"""Repo-root and well-known directory resolution.

Small, dependency-light helpers so the rest of the package never hard-codes layout
assumptions. Honors ``MERLIN_REPO_ROOT`` for installed/relocated checkouts; otherwise
resolves the repo root relative to this source file.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root.

    Honors ``MERLIN_REPO_ROOT``; otherwise finds the checkout by its project markers.
    Installed distributions without a checkout use the caller's writable work directory.
    """
    env = os.environ.get("MERLIN_REPO_ROOT")
    if env:
        return Path(env)
    root = checkout_root()
    return root if root is not None else work_dir()


def checkout_root() -> Path | None:
    """Find this implementation's source project, never an enclosing wheel-install host."""
    module = Path(__file__).resolve()
    for parent in module.parents:
        source_files = (parent / "src/merlin/common/paths.py",)
        if (
            (parent / "pyproject.toml").is_file()
            and (parent / "build_tools" / "package_resources.json").is_file()
            and any(source.resolve() == module for source in source_files)
        ):
            return parent
    return None


def python_source_dir() -> Path:
    """Import root for the active core, in a checkout or installed distribution."""
    return Path(__file__).resolve().parents[2]


def module_source_path(module: str) -> Path:
    """Find a core or optional module without importing its implementation.

    Use this for helper scripts/resources owned by an installed distribution, not a
    guessed checkout depth. Normal Python package resolution owns module identity.
    """
    import importlib.util

    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin or spec.origin in {"built-in", "frozen"}:
        raise FileNotFoundError(f"Python source module is unavailable: {module}")
    return Path(spec.origin)


def python_import_roots() -> tuple[Path, ...]:
    """Core and optional checkout source roots for isolated child interpreters.

    A child using another Python environment does not inherit editable installs.
    Deduplicate compatibility symlinks and retain canonical core precedence.
    """
    roots = [python_source_dir()]
    checkout = checkout_root()
    if checkout is not None:
        from merlin.common.access import PYTHON_SOURCE_ROOTS

        for source in PYTHON_SOURCE_ROOTS:
            package = checkout / source.path
            if package.is_dir():
                parent = package.resolve().parent
                if parent not in roots:
                    roots.append(parent)
    return tuple(roots)


def compat_lib_dir() -> Path:
    """Shared compatibility libraries supplied to compiler builds and agent sandboxes."""
    return repo_root() / ".compat_lib"


def runs_root(target: str, suite: str) -> Path:
    """Canonical suite run root: out/runs/<target>/<suite>, honoring MERLIN_OUT_ROOT."""
    return runs_dir() / target / suite


def phase_suite(phase: int | str) -> str:
    """The run suite of a phase: ``phase0``, ``phase1`` or ``phase2`` (the suite IS the phase)."""
    name = str(phase)
    if name not in ("0", "1", "2"):
        raise ValueError(f"phase must be 0, 1 or 2, got {phase!r}")
    return f"phase{name}"


def phase_runs_root(target: str, phase: int | str) -> Path:
    """Canonical phase run root: out/runs/<target>/phase<N>, honoring MERLIN_OUT_ROOT."""
    return runs_root(target, phase_suite(phase))


GENERATION_SUITES = {"0": "test-generation", "1": "backend-generation", "2": "backend-generation/optimization"}


def generation_run_roots(target: str, engine: int | str) -> tuple[Path, Path]:
    """Named run root, followed by the historical root retained for record discovery."""
    return runs_root(target, GENERATION_SUITES[str(engine)]), phase_runs_root(target, engine)


# Logical grant identities in archived protocols remain stable. Only a selected
# current-layout source owner translates them to the new physical locations.
_GRANT_PREFIXES = (
    (
        "merlin/experiments/reference-data/joint-public-plan-library",
        "experiments/inputs/joint-public-plan-library",
    ),
    ("merlin/tests/data/cca_asm", "experiments/inputs/cca-asm"),
    ("merlin/tests/data/opu_shapes", "experiments/inputs/opu-shapes"),  # target-ok: archived path identity
    ("merlin/contract/capsules", "experiments/corpora"),
    ("contract/capsules", "experiments/corpora"),
    ("merlin/contract", "src/merlin/resources/contracts"),
    ("contract", "src/merlin/resources/contracts"),
    ("merlin/schemas", "src/merlin/resources/schemas"),
    ("schemas", "src/merlin/resources/schemas"),
    ("merlin/prompts", "src/merlin/resources/prompts"),
    ("prompts", "src/merlin/resources/prompts"),
    ("merlin/runtime", "src/merlin/runtime/native"),
    ("runtime", "src/merlin/runtime/native"),
    ("merlin/targets", "experiments/inputs/target-inputs"),
    ("targets", "experiments/inputs/target-inputs"),
    ("merlin/tests", "tests"),
    ("merlin/benchmarks", "examples/workloads/benchmarks"),
    ("benchmarks", "examples/workloads/benchmarks"),
    ("merlin/experiments", "experiments/studies"),
    *(
        ("experiments/" + name, "experiments/studies/" + name)
        for name in (
            "agent_bench",
            "agent_recipe_select_v0",
            "agentic_report",
            "capsule_bench",
            "cpu_host_compiler_v0",
            "dataset_accuracy",
            "gemmini_cert",  # target-ok: archived path identity
            "gemmini_perf_bench",  # target-ok: archived path identity
            "gsim_wholemodel",
            "kernel_policy",
            "llm_kernel_vs_compiler_v0",
            "muon_perf_bench_v0",  # target-ok: archived path identity
            "performance_contract",
            "scheduled_backend_v0",
            "targetgen_evals",
            "voyager_h2h",
        )
    ),
)


def _layout_version(owner: Path) -> int | None:
    """Read the selected owner's layout declaration, never infer it from path existence."""
    import json

    manifest = owner / "build_tools/package_resources.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("version")


def resolve_grant(rel: str, *, root: Path | None = None) -> Path:
    """Resolve a logical grant within its explicitly selected source owner.

    Current owners declare layout version 2. Their old resource identities map
    explicitly; current project-relative paths remain literal. Archived version-1
    owners keep their original layout and documented shorthand. A missing path
    never redirects a grant to another owner or changes its meaning.
    """
    owner = Path(root) if root is not None else repo_root()
    logical = Path(rel).as_posix()
    version = _layout_version(owner)
    if version == 2:
        # Public benchmark specs have one package-owned authority; captures and
        # executable study inputs remain outside the installed package.
        benchmark = next(
            (logical[len(prefix) :] for prefix in ("merlin/benchmarks/", "benchmarks/") if logical.startswith(prefix)),
            None,
        )
        if benchmark is not None:
            import json

            document = json.loads((owner / "build_tools/package_resources.json").read_text(encoding="utf-8"))
            public = "src/merlin/resources/benchmarks/" + benchmark
            if public in document["files"]:
                return owner / public
        for old, new in _GRANT_PREFIXES:
            if logical == old or logical.startswith(old + "/"):
                return owner / (new + logical[len(old) :])
    elif version == 1 and logical.split("/", 1)[0] in {
        "experiments",
        "contract",
        "schemas",
        "prompts",
        "runtime",
        "targets",
        "benchmarks",
        "tests",
    }:
        return owner / "merlin" / logical
    return owner / rel


def _package_dir() -> Path:
    """Resource owner selected by the implementation or an explicit project root."""
    selected = os.environ.get("MERLIN_REPO_ROOT")
    return Path(selected) / "src/merlin" if selected else Path(__file__).resolve().parents[1]


def data_path(*parts: str) -> Path:
    """Resolve read-only resources identically in editable and installed packages.

    The implementation's package owns data. Writable work/output directories and
    the caller's CWD never select compiler resources. Optional experiment inputs
    remain unbundled and use their separately named project helpers.
    """
    rel = Path(*parts)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"resource path must be package-relative: {rel}")
    selected = os.environ.get("MERLIN_REPO_ROOT")
    if selected and _layout_version(Path(selected)) == 1:
        return Path(selected).joinpath("merlin", *parts)
    kind, *tail = rel.parts
    if kind == "runtime":
        return _package_dir().joinpath("runtime", "native", *tail)
    if kind in {"contract", "contracts"}:
        if tail[:1] == ["capsules"]:
            return capsules_dir().joinpath(*tail[1:])
        return _package_dir().joinpath("resources", "contracts", *tail)
    if kind == "targets":
        return targets_dir().joinpath(*tail)
    if kind == "experiments":
        return repo_root().joinpath("experiments", "studies", *tail)
    return _package_dir().joinpath("resources", kind, *tail)


def schemas_dir() -> Path:
    """Reviewed schemas, honoring ``MERLIN_SCHEMAS_DIR``."""
    return Path(os.environ["MERLIN_SCHEMAS_DIR"]) if os.environ.get("MERLIN_SCHEMAS_DIR") else data_path("schemas")


def contract_dir() -> Path:
    """Reviewed shared contracts, honoring the selected frozen ``MERLIN_CONTRACT_DIR``."""
    return Path(os.environ["MERLIN_CONTRACT_DIR"]) if os.environ.get("MERLIN_CONTRACT_DIR") else data_path("contract")


def prompts_dir() -> Path:
    """Agent prompt resources, honoring ``MERLIN_PROMPTS_DIR``."""
    return Path(os.environ["MERLIN_PROMPTS_DIR"]) if os.environ.get("MERLIN_PROMPTS_DIR") else data_path("prompts")


def bench_dir() -> Path:
    """Public benchmark specifications; ``MERLIN_BENCH_DIR`` selects an explicit owner."""
    return Path(os.environ["MERLIN_BENCH_DIR"]) if os.environ.get("MERLIN_BENCH_DIR") else data_path("benchmarks")


def study_inputs_dir() -> Path:
    """Unbundled benchmark captures and programs, honoring ``MERLIN_BENCH_DIR``.

    A wheel needs an explicit project root or benchmark override to access study
    inputs. Public benchmark specifications continue to come from :func:`bench_dir`.
    """
    override = os.environ.get("MERLIN_BENCH_DIR")
    if override:
        return Path(override)
    owner = os.environ.get("MERLIN_REPO_ROOT") or checkout_root()
    if owner is None:
        raise FileNotFoundError("benchmark study inputs require MERLIN_BENCH_DIR or MERLIN_REPO_ROOT")
    return Path(owner) / "examples/workloads/benchmarks"


def capsules_dir() -> Path:
    """Unbundled historical corpus inputs belonging to an explicitly selected project."""
    owner = os.environ.get("MERLIN_REPO_ROOT") or checkout_root()
    return Path(owner) / "experiments/corpora" if owner else _package_dir() / "resources/corpora"


def targets_dir() -> Path:
    """Optional target inputs, never bundled; honor ``MERLIN_TARGETS_DIR``."""
    override = os.environ.get("MERLIN_TARGETS_DIR")
    if override:
        return Path(override)
    owner = os.environ.get("MERLIN_REPO_ROOT") or checkout_root()
    return Path(owner) / "experiments/inputs/target-inputs" if owner else _package_dir() / "resources/targets"


def runtime_dir() -> Path:
    """Native runtime sources, honoring ``MERLIN_RUNTIME_DIR``; external tools compile them."""
    return Path(os.environ["MERLIN_RUNTIME_DIR"]) if os.environ.get("MERLIN_RUNTIME_DIR") else data_path("runtime")


# --- Generated-output roots. All generated products live under a single top-level ``out/`` with
#     three subdirs (runs/ artifacts/ build/). These helpers are the SINGLE source of truth for the
#     root names — callers must never hard-code the literal strings. Honors ``MERLIN_OUT_ROOT`` for
#     relocated/installed checkouts (mirrors ``MERLIN_REPO_ROOT``). ---
#: The single generated-output root name. Callers never spell it; they call the helpers below.
_OUT_ROOT_NAME = "out"


def work_dir() -> Path:
    """Return the writable WORK root for ephemeral scratch (``tmp/`` build scratch, calibration
    intermediates, external baseline checkouts) — ``<repo>`` in-repo, honoring ``MERLIN_WORK_DIR``.

    Distinct from ``out_dir()``: ``out/`` holds durable/curated products + runs (redirected by
    ``MERLIN_OUT_ROOT``); ``work_dir()`` holds throwaway scratch. An installed wheel with no checkout
    uses the current working directory, never the installation's ``site-packages`` directory.
    Set ``MERLIN_WORK_DIR`` to choose a different writable root."""
    env = os.environ.get("MERLIN_WORK_DIR")
    if env:
        return Path(env)
    explicit = os.environ.get("MERLIN_REPO_ROOT")
    return Path(explicit) if explicit else (checkout_root() or Path.cwd())


def out_dir() -> Path:
    """Return the generated-output root ``<repo>/out`` (honors ``MERLIN_OUT_ROOT``)."""
    env = os.environ.get("MERLIN_OUT_ROOT")
    if env:
        return Path(env)
    return work_dir() / _OUT_ROOT_NAME


def tracked_out_dir() -> Path:
    """The REPO-anchored ``<repo>/out`` — deliberately ignoring ``MERLIN_OUT_ROOT``.

    ``out/`` is the generated-output root and is redirectable, but a few files under it are TRACKED,
    reviewed inputs rather than output: the generated targets' ``contracts/`` (their capability
    manifest and residual). Resolving those through the redirectable root means a run with
    ``MERLIN_OUT_ROOT`` pointed elsewhere — a test, a worktree, a relocated checkout — cannot see a
    declaration that is committed to the repository, and a policy declaration that disappears when an
    env var moves is not a fact about the target.

    Use :func:`out_dir` for everything generated. Use this ONLY to read a tracked file under ``out/``.
    """
    return repo_root() / _OUT_ROOT_NAME


def runs_dir() -> Path:
    """Return ``<repo>/out/runs`` (aet-managed experiment runs)."""
    return out_dir() / "runs"


def artifacts_dir() -> Path:
    """Return ``<repo>/out/artifacts`` (versioned products, measurements, caches, recaptures)."""
    return out_dir() / "artifacts"


def build_dir() -> Path:
    """Return ``<repo>/out/build`` (compiled trees, baseline toolchains, OOT codegen)."""
    return out_dir() / "build"


def tracked_build_dir() -> Path:
    """The REPO-anchored ``<repo>/out/build`` — deliberately ignoring ``MERLIN_OUT_ROOT``.

    :func:`build_dir` is right for anything a run BUILDS. It is wrong for a tool a run merely USES:
    a simulator binary is a machine asset, installed once, not a per-run artifact. Resolving one
    through the redirectable root means a run whose ``project_root`` is a per-invocation temp
    directory looks for it under that temp directory, where it can never be — measured: 0 of 2295
    such roots on this host held an engine home, so an elaborated-RTL cert silently took the slow
    engine every time, and installing the fast one could not change that.

    Use :func:`build_dir` for build output. Use this to FIND an installed tool.
    """
    return tracked_out_dir() / "build"


# --- External, machine-specific dependency locations (chipyard, toolchains, boards, sibling
#     repos). Repo-INTERNAL paths never go here — the repo finds itself via repo_root(). These
#     differ per machine, so they live in a gitignored ``.env`` at the repo root. Copy
#     ``.env.example`` -> ``.env`` and edit. Resolve in code with ``ext_path('chipyard')``. ---


@functools.lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    """Parse ``<repo>/.env`` (``KEY=VALUE`` lines, ``#`` comments) into a dict. Cached."""
    return read_dotenv(repo_root() / ".env")


def read_dotenv(p: Path) -> dict[str, str]:
    """Parse one ``.env`` file the way :func:`_dotenv` parses this checkout's; absent means empty."""
    out: dict[str, str] = {}
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def env(key: str, default: str | None = None) -> str | None:
    """Resolve a machine-specific environment var, honoring the gitignored ``.env`` as a real config
    source (not only ``ext_path``). Precedence: the process environment WINS, then ``<repo>/.env``,
    then ``default``. This mirrors how ``llvmlower.toolchain`` already resolves the m2m/clang vars — now
    shared so the spike / Zephyr / board guards honor ``.env`` too, instead of only reading
    ``os.environ`` (the reason those capabilities looked unavailable unless the caller exported the vars
    by hand). Per-process and read-only: it NEVER mutates ``os.environ`` (safe on a shared host)."""
    return os.environ.get(key) or _dotenv().get(key) or default


def target_env_name(target: str, what: str) -> str:
    """The per-target environment variable name ``MERLIN_<TARGET>_<WHAT>``.

    One spelling for every per-target override (a target's Verilator binary is
    ``MERLIN_<TARGET>_VERILATOR``, its VCS simv ``MERLIN_<TARGET>_SIMV``), DERIVED from the target name
    so a newly registered target gets its variable with no edit to shared code. It is the same convention
    ``build_tools/scripts/check_repro_env.py`` and the sandbox toolchain already spell inline.
    """
    if not target or not what:
        raise ValueError(f"target_env_name needs a target and a suffix, got {target!r}, {what!r}")
    return f"MERLIN_{target.upper()}_{what.upper()}"


class ExternalPathUnset(KeyError):
    """``MERLIN_EXT_<NAME>`` is not configured: this host does not have that external checkout.

    A ``KeyError`` so every existing ``except KeyError`` is unchanged. Its own type is what lets a reader
    whose contract is "nothing derived when the source is absent" tell an absent checkout apart from a
    ``KeyError`` raised by a defect somewhere inside the code that would have read it.
    """


def ext_path(name: str) -> Path:
    """Resolve an external, machine-specific dependency location by short key.

    Reads ``MERLIN_EXT_<NAME_UPPERCASE>`` from the process environment (wins) or from the
    gitignored ``<repo>/.env``. Raises :class:`ExternalPathUnset` (a ``KeyError``) if unset (copy
    ``.env.example`` -> ``.env``). Example: ``ext_path('chipyard')`` -> reads ``MERLIN_EXT_CHIPYARD``.
    """
    key = f"MERLIN_EXT_{name.upper()}"
    val = os.environ.get(key) or _dotenv().get(key)
    if not val:
        known = sorted(k[len("MERLIN_EXT_") :].lower() for k in _dotenv() if k.startswith("MERLIN_EXT_"))
        raise ExternalPathUnset(f"external path {name!r} unset — set {key} in .env (copy .env.example). Known: {known}")
    return Path(val)


def is_external_path_unset(exc: BaseException | None) -> bool:
    """Whether ``exc`` is, or was explicitly raised FROM, an unconfigured external checkout.

    The lookup's error used to escape from deep inside an extractor, crashing readers documented to
    report "unavailable" -- only on a host without the checkout. Only explicit ``raise ... from`` causes
    are followed: a declaration naming a checkout this host lacks wraps the lookup that way, whereas an
    error a defect raises merely *while* handling one is implicit context and is not absence.
    """
    for _ in range(16):  # a cause chain is short; the bound only guards a pathological cycle
        if exc is None:
            return False
        if isinstance(exc, ExternalPathUnset):
            return True
        exc = exc.__cause__
    return False
