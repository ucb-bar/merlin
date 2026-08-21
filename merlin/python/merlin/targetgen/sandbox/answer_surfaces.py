"""The ANSWER SURFACE — every file/dir the agent-under-test must NOT be able to read, DERIVED.

The isolation sandbox is deny-by-default (see :mod:`merlin.targetgen.sandbox.bwrap`): all of
``/scratch*`` is tmpfs-masked and only the legit toolchain + the arm's declared inputs are bound back.
But a few answer surfaces are RE-EXPOSED by a broad legit bind (the goldens/hidden sit under the
allowed ``merlin/contract/`` tree; the experimenter memory sits under the bound ``~/.claude``), so they
must be explicitly re-masked on top. This module ENUMERATES that set from the target's declarative
descriptor + a single DECLARED oracle/grader registry — never a per-target hand-list — so a new target
gets a correct, complete mask from its ``target_experiment.yaml`` with zero copied policy.

The set has six origins, all derived:
  * ``golden``        — every ``golden.yaml`` under the declared capsule corpus + its sibling corpora
  * ``hidden``        — the hidden-capsule dir (the corpus's ``hidden/`` sibling)
  * ``prior_backend`` — the reference exemplars the descriptor's ``answer_surfaces.prior_backends`` names
  * ``oracle``        — the reference/simulator/runtime-backend modules (the DECLARED registry below)
  * ``grader``        — the decoder/grader/golden-gen modules (the DECLARED registry below)
  * ``memory``        — the experimenter's ``~/.claude`` memory for THIS repo (derived from the repo path)

:func:`coverage_gap` is the drift/cheat guard: given a built bwrap argv it returns the surfaces that are
NOT masked (the historical cheat gap — a hard-coded path that left the memory dir unmasked — is exactly
this class of bug, and this assertion closes it).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root
from merlin.targetgen.target_experiment import TargetExperiment

# --------------------------------------------------------------------------- the DECLARED registry
# The oracle + grader modules are a PROPERTY OF THE HARNESS, not of any target — one declared constant,
# consumed by BOTH the filesystem mask (below) and the transcript audit (audit_tokens). Repo-relative.
# Reading the oracle gives a route to the true reference/simulator; reading the grader lets the agent
# reverse-engineer the decoder/golden-gen instead of deriving from public facts. Neither arm may see them.
ORACLE_MODULES: tuple[str, ...] = (
    "merlin/python/merlin/runtime/reference.py",     # the numerical reference oracle
    "merlin/python/merlin/runtime/simulator.py",     # the functional simulator oracle
    "merlin/python/merlin/runtime/backends",         # the callable oracle backends (a route to the oracle)
    # The per-target oracle adapters (L2/L3/L4 routing) + the RTL-model bridge they call. Reading these
    # hands the agent the DRAM ABI (base/layout/stacking) and readback convention it is supposed to DERIVE
    # from the public contract + RTL facts — the arm-4 answer_access leak this registry must close.
    "merlin/python/merlin/targetgen/program_oracle.py",   # external_backend program oracle (atlas L2/L3/L4)
    "merlin/python/merlin/targetgen/muon_oracles.py",     # SIMT/Muon oracle adapters (radiance)
    "merlin/python/merlin/targetgen/heavy_oracles.py",    # heavy (cycle-accurate) oracle adapters
    "merlin/python/merlin/targetgen/rtl/mlc_bridge.py",   # mlc arc cosim + DRAM readback (the oracle bridge)
)
GRADER_MODULES: tuple[str, ...] = (
    "merlin/python/merlin/targetgen/rocc_decode.py",     # raw command-trace decoder (grader internal)
    "merlin/python/merlin/targetgen/trace_check.py",     # trace gate
    "merlin/python/merlin/targetgen/capsule_grade.py",   # the grader
    "merlin/python/merlin/targetgen/capsule_golden.py",  # golden generation
    "merlin/python/merlin/targetgen/capsule_runner.py",  # the tier runner
    "merlin/python/merlin/targetgen/capsule_dram.py",    # the DRAM preload/layout (input/output ABI the oracle expects)
    "merlin/python/merlin/targetgen/oot_runner.py",       # the OOT build+grade driver
    "merlin/python/merlin/targetgen/coverage_report.py",  # coverage grading
)
# Oracle-callable helper SUBPATHS that live INSIDE otherwise-allowed authoring tool dirs (the merlin-arm
# leak): reading them gives a callable route to the oracle. These are relative fragments, matched by the
# transcript audit (they are excised from the workspace copy by the deny-wins sub-path logic, not a
# separate filesystem mask). Declared here so there is ONE source of oracle identity.
ORACLE_CALLABLE_SUBPATHS: tuple[str, ...] = (
    "runtime_adapter", "xdsl_dialects/lowering", "lowering/pipeline")


@dataclass(frozen=True)
class AnswerSurface:
    """One answer-bearing path the sandbox must hide, with how it is masked."""
    label: str          # human label for diagnostics
    path: Path          # absolute host path
    kind: str           # "file" -> /dev/null overlay ; "dir" -> tmpfs
    origin: str         # golden | hidden | prior_backend | oracle | grader | memory | example


def claude_config_dir() -> Path:
    """The experimenter's Claude Code config dir — ``$CLAUDE_CONFIG_DIR``, else ``~/.claude``.

    The ROOT has to be derived too, not just the project slug. This module already knew that a stale
    hard-coded slug was the past cheat gap, and derived it from the repo — but it hard-coded the root as
    ``~/.claude``, which is the same bug one level up. On a host that relocates the config
    (``CLAUDE_CONFIG_DIR=/scratch/.../claude_config_dir/.claude``) the mask pointed at a path that does
    not even exist, so the REAL memory was masked only by the blanket ``/scratch`` tmpfs — and would have
    been exposed the moment anything bound the real config dir back in (which crediting the agent's
    ``claude`` CLI requires).
    """
    env = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    return Path(env) if env else Path(os.path.expanduser("~/.claude"))


def _slug(path: Path) -> str:
    """Claude Code's project-dir slug: the absolute path with ``/`` replaced by ``-``."""
    return str(path).replace("/", "-")


def experimenter_memory_dirs() -> list[Path]:
    """EVERY Claude Code memory dir that could hold notes about this repo.

    A session rooted at a PARENT of the repo gets its own project slug, and its memory is just as much
    an answer surface — here the workspace root and the repo each have one. Masking only the repo's slug
    leaves the sibling readable. Only existing dirs are returned; the caller masks what is there.
    """
    root = repo_root()
    projects = claude_config_dir() / "projects"
    out: list[Path] = []
    for base in (root, *root.parents):
        mem = projects / _slug(base) / "memory"
        if mem.is_dir():
            out.append(mem)
        if str(base) == "/":
            break
    return out


def experimenter_memory_dir() -> Path:
    """The memory dir for THIS repo's own slug (back-compat; prefer :func:`experimenter_memory_dirs`)."""
    return claude_config_dir() / "projects" / _slug(repo_root()) / "memory"


def golden_files(te: TargetExperiment) -> list[Path]:
    """Every golden/answer file the agent must not read. Globbed RECURSIVELY (any extension, any nesting
    depth) from the DECLARED corpus + its sibling corpora AND the whole capsule tree — so a run never sees
    a DEEPER-nested capsule's or ANOTHER target's golden (both escaped the old one-level ``*/golden.yaml``
    glob) — plus every example expected-output. Generic: no hard-coded ``isa/layers/model_slices`` list and
    no per-capsule literal. This is the parity-preserving replacement for the old hand-rolled
    ``answer_files()``."""
    files: list[Path] = []
    root = repo_root()
    corpora = [te.capsule_corpus] if te.capsule_corpus else []
    corpora += [root / rel.rstrip("/") for rel in te.corpus_siblings()]
    # The full capsule tree, in addition to the declared corpus: masks nested + other-target goldens the
    # declared one-level glob would miss (e.g. capsules/<target>/isa/<capsule>/golden.yaml). rglob only
    # matches files literally named ``golden.*`` — capsule INPUTS (interface MLIR, spec) stay visible.
    corpora.append(root / "merlin/contract/capsules")
    for corpus in corpora:
        if corpus and corpus.is_dir():
            files += sorted(corpus.rglob("golden.*"))
    # Every example expected-output (``expected_command_buffer_g0/g1/g2…`` and any ``expected_*`` artifact),
    # not just the single ``_g0`` literal that used to be masked.
    ex_dir = root / "merlin/contract/examples"
    if ex_dir.is_dir():
        files += sorted(ex_dir.rglob("expected_*"))
    # de-dup while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for f in files:
        if f.is_file() and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def answer_surfaces(te: TargetExperiment) -> list[AnswerSurface]:
    """The COMPLETE derived answer-surface set for one target — the single source the sandbox masks and
    the coverage guard checks. Only surfaces that actually exist on this host are returned (a masked
    non-existent path is a no-op); the coverage guard therefore checks a real, achievable set."""
    root = repo_root()
    out: list[AnswerSurface] = []

    examples_dir = root / "merlin/contract/examples"
    for g in golden_files(te):
        origin = "example" if examples_dir in g.parents else "golden"
        out.append(AnswerSurface(f"{origin}:{g.relative_to(root)}", g, "file", origin))

    hidden_rel = te.hidden_corpus()
    if hidden_rel:
        hp = root / hidden_rel.rstrip("/")
        if hp.is_dir():
            out.append(AnswerSurface("hidden-capsules", hp, "dir", "hidden"))

    tgt_root = artifacts_dir() / "targets" / te.target
    for name in te.prior_backends:
        bp = tgt_root / name
        if bp.exists():
            out.append(AnswerSurface(f"prior-backend:{name}", bp, "dir", "prior_backend"))

    for rel in ORACLE_MODULES:
        p = root / rel
        if p.exists():
            out.append(AnswerSurface(f"oracle:{Path(rel).name}", p,
                                     "dir" if p.is_dir() else "file", "oracle"))
    for rel in GRADER_MODULES:
        p = root / rel
        if p.exists():
            out.append(AnswerSurface(f"grader:{Path(rel).name}", p,
                                     "dir" if p.is_dir() else "file", "grader"))

    # Every memory dir, not just the repo's own slug — a session rooted at a parent directory has its
    # own slug and its own notes, and masking one while leaving the other readable is not a mask.
    for mem in experimenter_memory_dirs():
        out.append(AnswerSurface(f"experimenter-memory:{mem.parent.name}", mem, "dir", "memory"))

    return out


# --------------------------------------------------------------------------- transcript-audit tokens
def audit_tokens(te: TargetExperiment) -> dict[str, tuple[str, ...]]:
    """The path-fragment tokens the transcript audit flags as answer/grader/oracle READS — DERIVED from
    the same declared registry + descriptor as the filesystem mask, so there is one source of truth (no
    parallel hand-list to drift). ``answer`` = goldens/hidden/oracle-modules/prior-backends/grader-private;
    ``grader`` = grader-module stems; ``oracle_subpath`` = the oracle-callable helper subpaths."""
    answer: list[str] = ["golden.yaml", "expected_command_buffer"]
    hidden_rel = te.hidden_corpus()
    if hidden_rel:
        # e.g. "capsules/hidden" — the trailing two path components identify the hidden set
        answer.append("/".join(Path(hidden_rel.rstrip("/")).parts[-2:]))
    for rel in ORACLE_MODULES:
        # "merlin/runtime/reference" etc. — drop the merlin/python prefix + the .py suffix
        frag = rel[len("merlin/python/"):] if rel.startswith("merlin/python/") else rel
        answer.append(frag[:-3] if frag.endswith(".py") else frag)
    answer += list(te.prior_backends)
    answer.append("grader_private")
    grader = tuple(Path(rel).stem for rel in GRADER_MODULES)
    return {"answer": tuple(dict.fromkeys(answer)), "grader": grader,
            "oracle_subpath": ORACLE_CALLABLE_SUBPATHS}
