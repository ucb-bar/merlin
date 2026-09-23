"""The ANSWER SURFACE — every file/dir the agent-under-test must NOT be able to read, DERIVED.

The isolation sandbox is deny-by-default (see :mod:`merlin.targetgen.sandbox.bwrap`): all of
``/scratch*`` is tmpfs-masked and only the legit toolchain + the arm's declared inputs are bound back.
But a few answer surfaces are RE-EXPOSED by a broad legit bind (the goldens/hidden sit under the
allowed ``merlin/contract/`` tree; the experimenter memory sits under the bound ``~/.claude``), so they
must be explicitly re-masked on top. This module ENUMERATES that set from the target's declarative
descriptor + a single DECLARED oracle/grader registry — never a per-target hand-list — so a new target
gets a correct, complete mask from its ``target_experiment.yaml`` with zero copied policy.

The set has nine origins, all derived:
  * ``golden``        — every golden and expected grading-coverage file under the capsule corpora
  * ``weight``        — externalized model weights (the private inputs from which model goldens derive)
  * ``hidden``        — the hidden-capsule dir (the corpus's ``hidden/`` sibling) + the holdout
                        SPECIFICATION sidecars (``capsules/profiles/*.hidden.yaml``)
  * ``prior_backend`` — the reference exemplars the descriptor's ``answer_surfaces.prior_backends`` names
  * ``oracle``        — the reference/simulator/runtime-backend modules (the DECLARED registry below)
  * ``grader``        — the decoder/grader/golden-gen modules (the DECLARED registry below)
  * ``memory``        — the experimenter's ``~/.claude`` memory for THIS repo (derived from the repo path)
  * ``recovery_key``  — every minted known-answer recovery KEY, and every path a key DECLARES as the
                        material it was minted from (:mod:`merlin.perf.recovery`)
  * ``backend``       — the target's own backend package, whose ``contracts/`` sub-paths a bundle may
                        grant back and whose derivations it may not

WHY ``recovery_key`` AND ``backend`` ARE HERE. The known-answer recovery benchmark scores how much of a
reduction a loop re-authored unaided, against a key holding the steps that achieved it. That key, and
the patches it was minted from, are the most concentrated answer this repo owns: a loop that reads
either does not need to author anything. A fairness rule that is only written down is not one, so the
key declares its own source material and this module masks both — the same shape as the oracle
registry, so a benchmark minted tomorrow is covered the day its key exists rather than the day someone
remembers to add it. The backend package is the sibling case: it holds the target's hand-authored
reference derivations, and only the declared ``contracts/`` grants (which are DEEPER paths, so the
deny-wins longest-prefix rule lets them through) are ever meant to reach an arm.

:func:`coverage_gap` is the drift/cheat guard: given a built bwrap argv it returns the surfaces that are
NOT masked (the historical cheat gap — a hard-coded path that left the memory dir unmasked — is exactly
this class of bug, and this assertion closes it).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from merlin.common.access import (
    AUDIT_ADVISORY_KINDS as AUDIT_ADVISORY_KINDS,
)
from merlin.common.access import (
    AUDIT_VIOLATION_KINDS as AUDIT_VIOLATION_KINDS,
)
from merlin.common.access import (
    KEY_ENV,
    KEY_FILENAME,
    KEY_TOPIC,
    KEY_TOPIC_FOLDED,
    MODULE_ACCESS,
    contract_resource_roots,
    declared_modules,
    legacy_module_paths,
    module_locations,
)
from merlin.common.access import (
    audit_hit_is_violation as audit_hit_is_violation,
)
from merlin.common.access import (
    module_matches as module_matches,
)
from merlin.common.access import (
    module_name_for as module_name_for,
)
from merlin.common.paths import artifacts_dir, repo_root
from merlin.targetgen.target_experiment import TargetExperiment

# --------------------------------------------------------------------------- the DECLARED registry
# The oracle + grader modules are a PROPERTY OF THE HARNESS, not of any target — one declared constant,
# consumed by BOTH the filesystem mask (below) and the transcript audit (audit_tokens). Repo-relative.
# Reading the oracle gives a route to the true reference/simulator; reading the grader lets the agent
# reverse-engineer the decoder/golden-gen instead of deriving from public facts. Neither arm may see them.
# Compatibility views only. Identity, source locations and new namespace aliases live in access.py;
# the physical mask below must not use these historical path strings as its resolver.
ORACLE_MODULES: tuple[str, ...] = legacy_module_paths("oracle")
GRADER_MODULES: tuple[str, ...] = legacy_module_paths("grader")
# Oracle-callable helper SUBPATHS that live INSIDE otherwise-allowed authoring tool dirs (the merlin-arm
# leak): reading them gives a callable route to the oracle. These are relative fragments, matched by the
# transcript audit (they are excised from the workspace copy by the deny-wins sub-path logic, not a
# separate filesystem mask). Declared here so there is ONE source of oracle identity.
ORACLE_CALLABLE_SUBPATHS: tuple[str, ...] = ("runtime_adapter", "xdsl_dialects/lowering/pipeline")

#: The ONE sub-path of a target's own backend package that a bundle may legitimately grant back.
#:
#: The package is denied as a blanket precisely so that only this sub-tree reaches an arm: the arm is
#: REQUIRED to derive from its target's RTL facts and contract, and those live here, while the
#: hand-authored derivations beside them are the answer. The split already existed in this module
#: twice -- ``audit_tokens`` excludes it when tokenising the package's children, and the binder resolves
#: registry-owned grants by the same tail -- so naming it once is a de-duplication, not a new policy.
#:
#: It is a property of how a target package is SHAPED, not of any particular target, which is why it is
#: keyed by surface ORIGIN below: a target added tomorrow inherits it without an edit here.
PACKAGE_CONTRACT_SUBDIR = "contracts"

#: Per-ORIGIN sub-paths a bundle may grant back out of an otherwise-denied surface. Every origin absent
#: from this table is absolutely withheld: no grant, at any depth, reaches inside it. Adding an entry is
#: a reviewed edit, which is the point -- see :class:`AnswerSurface.grantable`.
GRANTABLE_SUBPATHS_BY_ORIGIN: dict[str, tuple[str, ...]] = {"backend": (PACKAGE_CONTRACT_SUBDIR,)}


@dataclass(frozen=True)
class AnswerSurface:
    """One answer-bearing path the sandbox must hide, with how it is masked."""

    label: str  # human label for diagnostics
    path: Path  # absolute host path
    kind: str  # "file" -> /dev/null overlay ; "dir" -> tmpfs
    origin: str  # golden | weight | hidden | prior_backend | oracle | grader | memory | example
    #: Sub-paths of this surface a bundle MAY grant back, relative to ``path``. Empty for every surface
    #: that holds an answer outright -- a golden, a weight, a hidden capsule, an oracle or grader module,
    #: the experimenter's memory -- which is to say: for almost all of them.
    #:
    #: WHY THIS IS A PROPERTY OF THE SURFACE AND NOT OF THE BUNDLE. A containment rule in which a deeper
    #: grant simply out-ranks a broader deny hands the decision to whoever writes the bundle, and that is
    #: exactly how an answer key gets re-admitted: someone grants a subdirectory for a good reason and
    #: silently re-opens what the deny existed to hide, with nothing anywhere recording that it happened.
    #: Declaring the exemption HERE means a bundle can only reach into a surface where the surface itself
    #: says a way in exists, so widening containment takes an edit to this file rather than to a manifest.
    grantable: tuple[str, ...] = ()


def experimenter_memory_dir() -> Path:
    """The experimenter's Claude Code memory dir for THIS repo. Claude Code slugifies the project path
    by replacing ``/`` with ``-``; deriving it from the CURRENT repo (never hard-coding) is what keeps
    the mask honest across repo moves — a stale hard-coded slug is precisely the past cheat gap."""
    return Path(os.path.expanduser(f"~/.claude/projects/{str(repo_root()).replace('/', '-')}/memory"))


def golden_files(te: TargetExperiment) -> list[Path]:
    """Every golden/expected-grading file the agent must not read. Globbed recursively (any nesting
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
    # matches only grading artifacts — capsule INPUTS (loader, linalg/interface MLIR, spec) stay visible.
    corpora.extend(contract_resource_roots(root, "capsules"))
    for corpus in corpora:
        if corpus and corpus.is_dir():
            files += sorted(corpus.rglob("golden.*"))
            # Instruction-coverage expectations are used by the grader in the same way as numerical
            # goldens. Exposing them tells the agent the exact command mix it must synthesize, so they
            # are an answer surface even though their filename does not begin with ``golden``.
            files += sorted(corpus.rglob("expected_instruction_coverage.yaml"))
    # Every example expected-output (``expected_command_buffer_g0/g1/g2…`` and any ``expected_*`` artifact),
    # not just the single ``_g0`` literal that used to be masked.
    for ex_dir in contract_resource_roots(root, "examples"):
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


def weight_files(te: TargetExperiment) -> list[Path]:
    """Every externalized capsule-weight file the agent must not read.

    Whole-model weights remain available to the operator-side grader and are copied into the immutable
    bundle snapshot, but exposing their values to the bring-up agent exposes the private model instance
    from which its withheld golden was computed.  Sweep the complete capsule tree, just as
    :func:`golden_files` does, so broad contract grants cannot leak another target's model weights.
    Both the safetensors blob and an optional manifest are covered by suffix, with no per-model list.
    """
    files: list[Path] = []
    for caps_root in contract_resource_roots(repo_root(), "capsules"):
        if caps_root.is_dir():
            files.extend(caps_root.rglob("*.safetensors"))
            files.extend(caps_root.rglob("*.safetensors.manifest.json"))
    return sorted({path for path in files if path.is_file()})


def recovery_key_files() -> list[Path]:
    """Every minted known-answer recovery KEY on this host.

    A key holds the step-by-step decomposition of a reduction a directed session already achieved:
    which transformation removed which operations, in what order, under what correctness gate. It is
    the most concentrated answer this repo owns, because it is not a destination but a PLAN — a loop
    that reads one does not need to author anything.

    Derived by walking the benchmark's own product topic rather than from a hand-list, so a key minted
    tomorrow is masked the day it exists. Persistent identity is declared in the dependency-free
    access registry and shared with the producer. No optional scorer import is needed: uninstalling
    a research distribution must not make already-minted keys disappear from the withheld set.
    """
    roots = [artifacts_dir() / KEY_TOPIC, artifacts_dir().joinpath(*KEY_TOPIC_FOLDED)]
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found += sorted(root.rglob(KEY_FILENAME))
    # A key the operator has pointed the benchmark at, wherever it lives. The scorer reads THAT file,
    # so masking only the topic roots would hide the key nobody is using and expose the one that is.
    override = os.environ.get(KEY_ENV)
    if override and Path(override).is_file():
        found.append(Path(override))
    return sorted({p.resolve() for p in found if p.is_file()})


def recovery_source_dirs() -> list[Path]:
    """The material every minted key was DERIVED FROM, as each key declares it.

    Masking the key alone would be theatre: the key is a summary of patches, base packages and
    replay work-dirs that are still on disk, and those hold the transformation itself rather than its
    arithmetic. So a key states its own source paths under ``provenance.answer_surfaces`` (repo-root
    relative), and they are masked with it. The declaration lives in the key rather than here for the
    same reason the oracle registry lives in one place: a second benchmark must not require editing
    this module.

    A key that declares nothing contributes nothing, and that is visible in the coverage report rather
    than silent — an undeclared source is a gap in the BENCHMARK, which is where it should be fixed.
    """
    root = repo_root()
    out: list[Path] = []
    for key in recovery_key_files():
        try:
            import yaml

            document = yaml.safe_load(key.read_text())
        except Exception:  # noqa: BLE001 — an unreadable key still gets masked as a file above
            continue
        if not isinstance(document, dict):
            continue
        declared = (document.get("provenance") or {}).get("answer_surfaces") or ()
        for rel in declared:
            path = Path(str(rel))
            resolved = path if path.is_absolute() else root / path
            if resolved.exists():
                out.append(resolved)
    return sorted({p for p in out})


def prior_backend_exemptions(te: TargetExperiment) -> tuple[str, ...]:
    """Package names under the target's own ``artifacts/targets/<target>/`` an arm may legitimately read.

    The target's codegen-package dir is masked WHOLESALE (see :func:`answer_surfaces`), so letting one
    package through is an explicit, reviewed line in the descriptor rather than the default. The same
    shape as :data:`GRANTABLE_SUBPATHS_BY_ORIGIN`: a way IN exists only where something declares one.

    Read straight from the descriptor document because ``TargetExperiment`` does not carry the field —
    the same thing :func:`merlin.targetgen.sandbox.resolve_kind` does for the target contract. FAIL
    CLOSED: an unreadable or malformed descriptor yields NO exemption, so a broken descriptor withholds
    more rather than less. A non-string entry is dropped for the same reason.
    """
    try:
        import yaml

        doc = yaml.safe_load(Path(te.path).read_text()) or {}
        declared = (doc.get("answer_surfaces") or {}).get("prior_backend_exemptions") or ()
    except Exception:  # noqa: BLE001 — no readable descriptor -> no exemption, never a blanket one
        return ()
    if isinstance(declared, str) or not isinstance(declared, (list, tuple)):
        return ()
    return tuple(str(name) for name in declared if isinstance(name, str) and name.strip())


def backend_package_dir(te: TargetExperiment) -> Path | None:
    """The target's own backend package — hand-authored reference derivations, not an arm's input.

    DERIVED from the descriptor (``te.backend_package``, which a target may redeclare), never a path
    literal. Masked as a directory, which is safe precisely because the sandbox resolves mounts by
    LONGEST PREFIX with deny breaking a tie: the ``contracts/rtl_facts`` and ``contracts/irdl`` grants
    an arm legitimately receives are deeper paths and survive this mask, while the derivations beside
    them do not. That is the property worth having — the grant list says what an arm may read, and
    this says that everything else in the package is not merely ungranted but denied.
    """
    rel = te.backend_package
    if not rel:
        return None
    path = repo_root() / str(rel).rstrip("/")
    return path if path.is_dir() else None


def _evicted_oracle_modules_result() -> tuple[list[Path], str | None]:
    """``(paths, failure_reason)`` for the registry-driven eviction sweep.

    Reference-target BACKENDS + sim-oracles evicted to their own packages (OV11) are oracle ROUTES too
    — the SIMT cyclotron oracle now lives in the muon package (``muon_oracles`` inside its backend), and a
    reference backend is the 'answer' codegen. DERIVE their host paths from the plugin registry (the same
    discovery the runtime uses) rather than a per-target literal, so the mask FOLLOWS the eviction instead
    of the now-stale in-tree paths in ORACLE_MODULES.

    The sweep is best-effort by necessity — the registry is optional and a malformed target package must
    not break sandbox assembly — but "best-effort" was spelled as a bare ``except: pass``, which is the
    same silent no-op this module now refuses everywhere else. Measured on this host: an unrelated syntax
    error in a module the registry imports made the sweep return ZERO paths, so every evicted oracle route
    went unmasked and nothing anywhere said so. The failure is therefore RETURNED, so
    :func:`dropped_declarations` can record it as UNKNOWN rather than let it read as "nothing to mask".
    """
    paths: list[Path] = []
    try:
        from merlin.runtime.backends import base as _bk

        for key in ("backend", "sim_oracle"):
            for _name, p in _bk._oot_plugin_modules(key):
                if p.exists():
                    paths.append(p)
    except Exception as exc:  # noqa: BLE001 — recorded as UNKNOWN, never silently treated as "none"
        return paths, f"{type(exc).__name__}: {exc}"
    return paths, None


def _evicted_oracle_modules() -> list[Path]:
    """The evicted oracle/backend routes discoverable right now (see :func:`_evicted_oracle_modules_result`)."""
    return _evicted_oracle_modules_result()[0]


# ------------------------------------------------------------------ declarations that matched NOTHING
#: Why a declared answer-surface rule contributed no mask. Each is a different operator action: a stale
#: literal is fixed by correcting the declaration, an undiscoverable registry by repairing the import.
DROP_PATH_ABSENT = "declared_path_absent"
DROP_DISCOVERY_FAILED = "discovery_failed"


@dataclass(frozen=True)
class DroppedDeclaration:
    """A required path rule or plugin discovery that contributed no mask.

    Physical masks only cover existing paths, so a stale required declaration could otherwise make
    ``coverage_gap`` pass vacuously. Optional logical module identities are tracked separately by
    ``merlin.common.access`` and may legitimately lack an installed implementation.
    """

    origin: str  # oracle | grader | prior_backend | evicted_oracle — which rule class dropped
    declared: str  # exactly what was declared, as written
    reason: str  # DROP_PATH_ABSENT | DROP_DISCOVERY_FAILED (+ detail)

    def describe(self) -> str:
        return f"{self.origin}:{self.declared} ({self.reason})"

    def as_record(self) -> dict[str, str]:
        return {"origin": self.origin, "declared": self.declared, "reason": self.reason}


def dropped_declarations(te: TargetExperiment) -> list[DroppedDeclaration]:
    """Every required path declaration or plugin discovery that contributed no mask.

    Fail-closed companion to :func:`answer_surfaces`: that function answers "what is masked", this one
    answers "what did somebody declare that is not". A caller making a fairness claim must check BOTH —
    an empty :func:`coverage_gap` over a set that silently lost a rule proves nothing about the rule.

    Logical module identities are not required paths: optional research and historical OOT modules
    legitimately have no local implementation. Their physical copies are masked through
    ``MODULE_ACCESS`` and ``module_locations`` wherever installed; ``unresolved_modules`` is the
    explicit migration diagnostic for absent implementations. A derived sweep that finds nothing
    (no goldens, no weights, no minted recovery key) is likewise not a drop.
    """
    out: list[DroppedDeclaration] = []
    _, failure = _evicted_oracle_modules_result()
    if failure is not None:
        out.append(
            DroppedDeclaration(
                "evicted_oracle", "plugin registry (backend, sim_oracle)", f"{DROP_DISCOVERY_FAILED}: {failure}"
            )
        )
    # The descriptor's named prior backends. These no longer drive the mask (the target's whole
    # codegen-package dir is masked wholesale — see `answer_surfaces`), but a name that resolves to
    # nothing is still a stale declaration somebody is reading as policy, so it is reported.
    tgt_root = artifacts_dir() / "targets" / str(getattr(te, "target", "") or "")
    # ``getattr``: a descriptor object that carries no prior-backend list declares none, which drops
    # nothing. Absence of a declaration is not a dropped declaration.
    for name in getattr(te, "prior_backends", ()) or ():
        if not (tgt_root / name).exists():
            out.append(DroppedDeclaration("prior_backend", str(name), DROP_PATH_ABSENT))
    for name in prior_backend_exemptions(te):
        if not (tgt_root / name).exists():
            out.append(DroppedDeclaration("prior_backend_exemption", str(name), DROP_PATH_ABSENT))
    return out


class DroppedDeclarations(RuntimeError):
    """Raised when a caller requires that every declared deny rule is actually in force."""

    def __init__(self, dropped: list[DroppedDeclaration], context: str = "") -> None:
        prefix = f"{context}: " if context else ""
        super().__init__(
            prefix + "declared answer-surface rules matched nothing: " + "; ".join(d.describe() for d in dropped)
        )
        self.dropped = dropped


def require_declarations_in_force(te: TargetExperiment, *, context: str = "") -> None:
    """Raise unless every DECLARED answer-surface rule resolved to something maskable."""
    dropped = dropped_declarations(te)
    if dropped:
        raise DroppedDeclarations(dropped, context)


def _support_package_dirs() -> list[Path]:
    """Selected host-support ownership, including siblings of backend plugins.

    Read metadata only: importing a backend is neither necessary nor sufficient
    to find its private conformance, build helpers and tools. Invalid discovery
    propagates: an unknown support mask must not become an empty mask. Candidate
    compiler and host-schedule packages are not support providers.
    """
    from merlin.targetgen import target_registry
    from merlin.targetgen.providers import ProviderRole

    roots: set[Path] = set()
    for name in target_registry.all_targets():
        provider = target_registry.resolve(name).provider
        if provider is not None and provider.role == ProviderRole.SUPPORT:
            roots.add(provider.root)
    return sorted(roots)


def _backend_package_dirs(te: TargetExperiment) -> list[Path]:
    # Keep the descriptor-owned location too: a historical or separately named
    # reference package remains private even when an OOT provider is selected.
    roots = set(_support_package_dirs())
    declared = backend_package_dir(te)
    if declared is not None:
        roots.add(declared)
    return sorted(roots)


def answer_surfaces(te: TargetExperiment) -> list[AnswerSurface]:
    """The COMPLETE derived answer-surface set for one target — the single source the sandbox masks and
    the coverage guard checks. Only surfaces that actually exist on this host are returned (a masked
    non-existent path is a no-op); the coverage guard therefore checks a real, achievable set."""
    root = repo_root()
    out: list[AnswerSurface] = []

    def shown_path(path: Path) -> Path:
        return path.relative_to(root) if path.is_relative_to(root) else path

    examples_dirs = contract_resource_roots(root, "examples")
    for g in golden_files(te):
        origin = "example" if any(p in g.parents for p in examples_dirs) else "golden"
        out.append(AnswerSurface(f"{origin}:{shown_path(g)}", g, "file", origin))
    for weights in weight_files(te):
        out.append(AnswerSurface(f"weight:{shown_path(weights)}", weights, "file", "weight"))

    # Mask EVERY hidden-capsule dir under the capsule tree, not only THIS target's declared one. The bundle
    # grants the frozen ABI (``merlin/contract/``) broadly, which re-exposes the SHARED
    # ``capsules/hidden`` set and any OTHER target's ``<t>/hidden`` — a radiance run could otherwise read
    # the shared/atlas hidden capsules (a held-out answer surface; the ``CANARY_HIDDEN`` marker caught
    # exactly this). Mirrors :func:`golden_files`' whole-tree sweep that masks cross-target/nested goldens.
    #
    # Measured on saturn_opu before the fix: the hidden GOLDENS were masked (they are enumerated
    # file-by-file above, so answer VALUES never leaked), but the hidden capsule DIRECTORIES stayed
    # listable and their ``capsule.yaml`` inputs readable from all three merlin-family bundles and not
    # from raw_baseline — so the held-out set was enumerable for three of four arms, which both weakens
    # the hidden grade as a generalization check and makes a merlin-vs-baseline hidden comparison
    # asymmetric. Derived by walking, so a new target's hidden dir is covered the day it appears.
    hidden_dirs: set[Path] = set()
    hidden_rel = te.hidden_corpus()
    if hidden_rel:
        hidden_dirs.add(root / hidden_rel.rstrip("/"))
    caps_roots = contract_resource_roots(root, "capsules")
    for caps_root in caps_roots:
        if caps_root.is_dir():
            hidden_dirs.update(d for d in caps_root.rglob("hidden") if d.is_dir())
    for hp in sorted(hidden_dirs):
        if hp.is_dir():
            out.append(AnswerSurface(f"hidden-capsules:{shown_path(hp)}", hp, "dir", "hidden"))

    # The holdout SPECIFICATION, not just its directory. A capsule's op + dtype + exact shape is an
    # answer to the generalization question even when the golden values stay masked, and the spec used
    # to live in the tracked ``profiles/<target>.yaml`` inside the same broad ``merlin/contract/`` grant.
    # It now lives in an untracked ``profiles/<target>.hidden.yaml`` sidecar; masking it here as well
    # means a checkout that still carries one (or a target that grows a new sidecar) is covered without
    # relying on .gitignore. Derived by glob, never a per-target hand-list.
    for caps_root in caps_roots:
        prof_dir = caps_root / "profiles"
        if prof_dir.is_dir():
            for hp in sorted(prof_dir.glob("*.hidden.yaml")):
                out.append(AnswerSurface(f"hidden-spec:{shown_path(hp)}", hp, "file", "hidden"))

    # THE TARGET'S OWN CODEGEN-PACKAGE DIR, WHOLESALE — not the descriptor's list of names.
    #
    # The descriptor used to enumerate which finished backends to hide, and enumeration is what drifted:
    # measured on gemmini, `answer_surfaces.prior_backends` named 4 of the 40 packages present under
    # `out/artifacts/targets/gemmini/`, so 36 finished backends — including a hand-authored `hand_v0` —
    # were covered by no answer surface at all. Nothing in the guard could see that, because an
    # enumeration is vacuously complete with respect to itself. A package is minted by
    # `merlin-rvv-mine` / `merlin-rvv-autotune` / `merlin-targetgen` whenever anyone runs one, so any
    # hand-list is stale the next time a tool runs, not the next time somebody edits the descriptor.
    #
    # Masking the DIRECTORY inverts the default: every package under the target's own dir is withheld the
    # day it is minted, and letting one through takes a declared exemption (below). Only THIS target's
    # subdir is masked: another target's packages are legitimately granted as cross-target baselines (the
    # gemmini descriptor grants `out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8`), and they are
    # not an answer to this target's question.
    tgt_root = artifacts_dir() / "targets" / te.target
    if tgt_root.is_dir():
        exemptions = prior_backend_exemptions(te)
        out.append(
            AnswerSurface(
                f"prior-backends:{te.target}",
                tgt_root,
                "dir",
                "prior_backend",
                grantable=exemptions,
            )
        )
    # The named backends are still emitted individually, so a descriptor that names one keeps its own
    # label in the coverage report. `apply_answer_masks` skips any surface the wholesale mask above
    # already hides, so these are records, not redundant mounts.
    for name in te.prior_backends:
        bp = tgt_root / name
        if bp.exists():
            out.append(AnswerSurface(f"prior-backend:{name}", bp, "dir", "prior_backend"))

    for item in MODULE_ACCESS:
        for p in module_locations(root, item):
            out.append(AnswerSurface(f"{item.origin}:{p.name}", p, "dir" if p.is_dir() else "file", item.origin))
    for p in _evicted_oracle_modules():  # OV11: oracle/backend routes relocated to target packages
        out.append(AnswerSurface(f"oracle:{p.name}", p, "dir" if p.is_dir() else "file", "oracle"))

    # The known-answer recovery benchmark's key, and the material it was minted from. A benchmark that
    # scores how much of a reduction an agent re-authored UNAIDED is worthless the moment the agent can
    # read the reduction; nothing else in this file is as directly self-defeating to leak.
    for kp in recovery_key_files():
        out.append(AnswerSurface(f"recovery-key:{kp.name}", kp, "file", "recovery_key"))
    for sp in recovery_source_dirs():
        label = sp.relative_to(root) if root in sp.parents else sp
        out.append(AnswerSurface(f"recovery-source:{label}", sp, "dir" if sp.is_dir() else "file", "recovery_key"))

    # The target's own backend package. Its `contracts/` grants are deeper paths and survive this mask
    # (longest-prefix, deny-breaks-ties); its hand-authored derivations do not.
    for bp in _backend_package_dirs(te):
        out.append(
            AnswerSurface(
                f"backend-package:{shown_path(bp)}",
                bp,
                "dir",
                "backend",
                grantable=GRANTABLE_SUBPATHS_BY_ORIGIN.get("backend", ()),
            )
        )

    mem = experimenter_memory_dir()
    if mem.is_dir():
        out.append(AnswerSurface("experimenter-memory", mem, "dir", "memory"))

    return out


# ------------------------------------------------------------------ the audit's hit VOCABULARY
# The transcript audit records every suspicious event, but only some of them mean the agent actually
# SAW withheld content. Both the audit that produces the hits and every downstream gate that consumes
# them (the perf campaign's fail-closed boundary) must agree on which is which, so the vocabulary is
# declared ONCE here rather than duplicated per consumer -- a gate that re-derives the split by hand
# is exactly how "no hits at all" became the accidental bar and disqualified conformant rounds.
#
# ADVISORY -- the protection WORKED (or nothing was read at all); these are recorded for visibility:
#   blocked_probe   the mask returned nothing / an error, so no withheld bytes reached the agent
#   recon_probe     a path-LISTING search that surfaced no answer path (filenames, not content)
#   owned_read      the read resolves inside this run's agent-authored submission tree
#   granted_read    the read target is a file the arm's own bundle GRANTS
#   pattern_mention the withheld token appeared as a search PATTERN, not as a path being read
# VIOLATION -- withheld content reached the agent, or agent code routes to the oracle:
#   path_read       a content read of a withheld path that returned data
#   oracle_use      agent-authored code imports/calls a denied oracle module
# Identity-preserving reexports above keep existing audit producers/consumers on the same
# core-owned vocabulary even when the optional scorer or sandbox package is not installed.


# ------------------------------------------------------- oracle IDENTITY as importable module names
def declared_oracle_modules() -> tuple[str, ...]:
    """The DECLARED oracle registry as dotted module names -- the harness-level identity of "the
    oracle", independent of any bundle. Importing one of these is oracle USE for EVERY arm, even if
    some bundle's grant list were to name it."""
    return declared_modules("oracle")


# --------------------------------------------------------------------------- transcript-audit tokens
def audit_tokens(te: TargetExperiment) -> dict[str, tuple[str, ...]]:
    """The path-fragment tokens the transcript audit flags as answer/grader/oracle READS — DERIVED from
    the same declared registry + descriptor as the filesystem mask, so there is one source of truth (no
    parallel hand-list to drift). ``answer`` = goldens/weights/hidden/oracle-modules/prior-backends/
    grader-private; ``grader`` = grader-module stems; ``oracle_subpath`` = the oracle-callable helper
    subpaths."""
    answer: list[str] = [
        "golden.yaml",
        "expected_command_buffer",
        "expected_instruction_coverage.yaml",
    ]
    # A token for EVERY hidden-capsule dir (this target's + the shared one + any other target's), matching
    # the filesystem mask above — the trailing two path components identify each hidden set (e.g.
    # "radiance/hidden", "capsules/hidden", "atlas/hidden"). A read of any is an answer surface.
    _hidden_rels = []
    if te.hidden_corpus():
        _hidden_rels.append(te.hidden_corpus().rstrip("/"))
    _caps_roots = contract_resource_roots(repo_root(), "capsules")
    for _caps in _caps_roots:
        if _caps.is_dir():
            _hidden_rels += [d.as_posix() for d in _caps.rglob("hidden") if d.is_dir()]
    for _hr in _hidden_rels:
        answer.append("/".join(Path(_hr).parts[-2:]))
    # ...and the holdout SPECIFICATION sidecars, matching the filesystem mask. The ".hidden.yaml"
    # suffix identifies every one of them, so a new target's sidecar is covered the day it appears.
    if any((_caps / "profiles").is_dir() for _caps in _caps_roots):
        answer.append(".hidden.yaml")
    # Weight filenames are legitimately DECLARED in agent-visible capsule YAML/MLIR. A bare
    # ``.safetensors`` token would therefore accuse an agent that greps its public interface of reading
    # the private blob. Qualify by category/capsule/file: a real path read still matches, while a public
    # declaration containing only ``capsule.weights.safetensors`` does not.
    for _weights in weight_files(te):
        answer.append("/".join(_weights.parts[-3:]))
    answer.extend(module.replace(".", "/") for module in declared_oracle_modules())
    # The target's whole codegen-package dir, matching the wholesale filesystem mask, plus the names the
    # descriptor happens to list. The token is the FULL repo-relative fragment ("out/artifacts/targets/
    # <target>") rather than "targets/<target>": the short form substring-matches the arm's own task tree
    # (merlin/experiments/capsule_bench/targets/<target>/...), which would accuse every agent that reads
    # the task it was given — the cry-wolf failure this token vocabulary is explicitly shaped to avoid.
    _pkg_root = artifacts_dir() / "targets" / te.target
    try:
        answer.append(_pkg_root.relative_to(repo_root()).as_posix())
    except ValueError:  # out root relocated outside the repo (MERLIN_OUT_ROOT) — use the absolute path
        answer.append(str(_pkg_root))
    answer += list(te.prior_backends)
    # Evicted oracle/backend routes, tokenised as "<target>/<route>" rather than by bare stem. Most of
    # these live in a directory literally named `backend`, so a stem token is the generic word "backend"
    # — which substring-matches innocuous granted paths (mlir_oot_backend_contract.yaml, any *_backend_*
    # doc) and flags them as answer reads. A transcript audit that cries wolf on the contract the agent is
    # required to read is worse than no audit: the next real leak reads as more noise.
    for _p in _evicted_oracle_modules():
        _stem = _p.stem if not _p.name.endswith(".py") else _p.name[:-3]
        answer.append(f"{_p.parent.name}/{_stem}" if _p.parent.name else _stem)
    answer.append("grader_private")
    grader_names = [module.rsplit(".", 1)[-1] for module in declared_modules("grader")]
    # Directory identities are importable packages, not only script filenames.
    # Keep the bare stem for exact relative operands and add their qualified
    # import/path spellings without relying on currently installed contents.
    for item in MODULE_ACCESS:
        if item.origin == "grader" and item.directory:
            for module in item.modules:
                grader_names.extend((module, module.replace(".", "/")))
    grader = tuple(dict.fromkeys(grader_names))
    # The recovery benchmark's key, by its own filename (specific enough to be a token, and owned by
    # the benchmark module so there is one spelling), and every source tree a key declares, by the two
    # trailing components that identify it. A transcript that READS one of these is a leak of the plan.
    answer.append(KEY_FILENAME)
    for _src in recovery_source_dirs():
        answer.append("/".join(_src.parts[-2:]))
    # The target's backend package, EXCLUDING the `contracts/` sub-tree an arm is legitimately granted.
    # Tokenising the package as a whole would accuse every arm-4 agent of reading its own RTL facts,
    # which is the cry-wolf failure `test_audit_token_precision` exists to stop. Derived by walking the
    # package's own children, so a new sibling dir of derivations is covered without an edit here.
    for _bp in _backend_package_dirs(te):
        for _child in sorted(_bp.iterdir()):
            if _child.name != PACKAGE_CONTRACT_SUBDIR:
                answer.append(f"{_bp.name}/{_child.name}")
    return {"answer": tuple(dict.fromkeys(answer)), "grader": grader, "oracle_subpath": ORACLE_CALLABLE_SUBPATHS}
