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
        source_files = (
            parent / "src/merlin/common/paths.py",
            parent / "merlin/python/merlin/common/paths.py",
        )
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


def merlin_dir() -> Path:
    """Return ``<repo>/merlin``."""
    return repo_root() / "merlin"


def resolve_grant(rel: str) -> Path:
    """Resolve a bundle-convention grant path string to an absolute host path.

    Grant paths in target descriptors / bundle manifests are repo-root-relative by convention, with
    one documented shorthand: a leading ``experiments/...`` (and other in-``merlin/`` trees) "resolves
    under ``merlin/``" — i.e. the ``merlin/`` prefix may be elided. This resolver honors that: it prefers
    ``<repo>/<rel>`` when that exists, else falls back to ``<repo>/merlin/<rel>`` when THAT exists, else
    returns ``<repo>/<rel>`` (the caller's existence check then treats it as missing). Keeps the sandbox
    binder (``bwrap``) and the workspace assembler in lockstep so a granted path is never silently
    dropped by one but honored by the other.
    """
    root = repo_root() / rel
    if root.exists():
        return root
    under_merlin = merlin_dir() / rel
    if under_merlin.exists():
        return under_merlin
    return root


def data_path(*parts: str) -> Path:
    """Resolve bundled read-only package data (``schemas/``, ``prompts/``, …).

    Prefers the in-repo canonical tree (``<repo>/merlin/<parts>``) when a checkout is present, so
    dev workflows and the repo linters always read the live files. Falls back to the copy bundled
    into an installed wheel (``merlin/_data/<parts>`` via ``importlib.resources``) so ``pip install
    merlin`` works outside a checkout. Per-class env overrides (e.g. ``MERLIN_SCHEMAS_DIR``) are
    applied by the class-specific wrappers below, not here.
    """
    rel = Path(*parts)
    cand = merlin_dir() / rel
    if cand.exists():
        return cand
    try:
        import importlib.resources as _ir

        base = _ir.files("merlin").joinpath("_data", *rel.parts)
        # normally-installed (unzipped) wheel -> a real filesystem path
        return Path(str(base))
    except (ModuleNotFoundError, FileNotFoundError, TypeError, NotADirectoryError):
        return cand  # nonexistent repo path -> callers raise a clear FileNotFoundError


def schemas_dir() -> Path:
    """Return the schemas dir (``<repo>/merlin/schemas`` in-repo, bundled ``_data/schemas`` in a
    wheel). Honors ``MERLIN_SCHEMAS_DIR``."""
    env = os.environ.get("MERLIN_SCHEMAS_DIR")
    if env:
        return Path(env)
    return data_path("schemas")


def prompts_dir() -> Path:
    """Return the agent-prompt dir (``<repo>/merlin/prompts`` in-repo, bundled ``_data/prompts`` in a
    wheel). Honors ``MERLIN_PROMPTS_DIR``."""
    env = os.environ.get("MERLIN_PROMPTS_DIR")
    if env:
        return Path(env)
    return data_path("prompts")


def bench_dir() -> Path:
    """Return the benchmarks corpus ROOT (``<repo>/merlin/benchmarks`` in-repo). Honors
    ``MERLIN_BENCH_DIR`` (which is the benchmarks *root*, not a subdir). In an installed wheel this
    resolves to the bundled ``_data/benchmarks`` — the LIGHT specs only (workload region YAMLs,
    accuracy/cycle/dispatch tables, region maps); the heavy ``recaptures*`` capture corpora are NOT
    bundled, so point ``MERLIN_BENCH_DIR`` at a checkout to reach them from a wheel."""
    env = os.environ.get("MERLIN_BENCH_DIR")
    if env:
        return Path(env)
    return data_path("benchmarks")


def targets_dir() -> Path:
    """Return the optional target-input directory.

    A checkout may retain legacy ``merlin/targets`` inputs, but the installed core does not
    bundle target contracts or facts. Set ``MERLIN_TARGETS_DIR`` to a selected out-of-tree
    target's directory when a caller needs those authored inputs.
    """
    env = os.environ.get("MERLIN_TARGETS_DIR")
    if env:
        return Path(env)
    return data_path("targets")


def runtime_dir() -> Path:
    """Return the C runtime substrate dir (``<repo>/merlin/runtime`` in-repo, bundled
    ``_data/runtime`` in a wheel) — the ``c/``, ``abi/``, ``baremetal/`` sources the whole-model
    compile paths read at use time. Honors ``MERLIN_RUNTIME_DIR``. Compiling them still needs an
    external toolchain (the ``[board]`` extra / ``MERLIN_*`` tools); bundling only makes the sources
    resolve so the failure is the actionable 'toolchain unavailable', not 'source not found'."""
    env = os.environ.get("MERLIN_RUNTIME_DIR")
    if env:
        return Path(env)
    return data_path("runtime")


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
    out: dict[str, str] = {}
    p = repo_root() / ".env"
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


def ext_path(name: str) -> Path:
    """Resolve an external, machine-specific dependency location by short key.

    Reads ``MERLIN_EXT_<NAME_UPPERCASE>`` from the process environment (wins) or from the
    gitignored ``<repo>/.env``. Raises ``KeyError`` if unset (copy ``.env.example`` -> ``.env``).
    Example: ``ext_path('chipyard')`` -> reads ``MERLIN_EXT_CHIPYARD``.
    """
    key = f"MERLIN_EXT_{name.upper()}"
    val = os.environ.get(key) or _dotenv().get(key)
    if not val:
        known = sorted(k[len("MERLIN_EXT_") :].lower() for k in _dotenv() if k.startswith("MERLIN_EXT_"))
        raise KeyError(f"external path {name!r} unset — set {key} in .env (copy .env.example). Known: {known}")
    return Path(val)
