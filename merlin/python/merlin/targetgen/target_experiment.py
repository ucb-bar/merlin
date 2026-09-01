"""Load a declarative per-target experiment descriptor — the target-parameterized replacement for the
gemmini-hardcoded experiment setup.

Per the derive-first rule, the hardware FACTS (ISA/opcode set, memory map, mesh DIM, arc model) are
DERIVED from the RTL by mlc (``rtl_backend.target_profile`` / ``mlc_bridge``), never hand-written. What a
run genuinely cannot derive — which RTL repo, which hardware-spec files every arm gets, which capsule
corpus to grade on, how the simulator runs — is the irreducible SETUP, declared in a small YAML
descriptor (``target_experiment.yaml`` beside the experiment). A new accelerator drops its own descriptor
and registers its RTL with mlc; no per-target code.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import repo_root


def _safe_relative(value: str, *, field: str) -> Path:
    """Parse one descriptor path without letting it escape the repository/package it names."""
    path = Path(str(value))
    if path.is_absolute() or not path.parts or any(part == ".." for part in path.parts):
        raise ValueError(f"host_lane.{field} must be a non-empty repo-relative path, got {value!r}")
    return path


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


@dataclass(frozen=True)
class HostLane:
    """Descriptor-owned identity of the frozen scalar/vector compiler used beside an accelerator.

    This is infrastructure, not the submitted accelerator package.  ``package`` identifies the exact
    artifact grading must pass to ``compile_rvv``; ``read_only`` and ``deny_modification`` describe the
    corresponding agent grant boundary.
    """

    description: str
    repo_canonical: str
    branch: str
    commit: str
    package: str
    requires_paths: tuple[str, ...]
    read_only: tuple[str, ...]
    deny_modification: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any, *, descriptor: Path) -> "HostLane | None":
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(
                f"{descriptor}: `host_lane` must be a mapping, got {type(value).__name__}")
        required = ("description", "repo_canonical", "branch", "commit", "package",
                    "requires_paths", "read_only", "deny_modification")
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"{descriptor}: host_lane is missing required field(s) {missing}")

        def paths(name: str) -> tuple[str, ...]:
            raw = value[name]
            if not isinstance(raw, (list, tuple)) or any(not isinstance(path, str) for path in raw):
                raise ValueError(f"{descriptor}: host_lane.{name} must be a list of paths")
            return tuple(raw)

        return cls(
            description=str(value["description"]),
            repo_canonical=str(value["repo_canonical"]),
            branch=str(value["branch"]),
            commit=str(value["commit"]),
            package=str(value["package"]),
            requires_paths=paths("requires_paths"),
            read_only=paths("read_only"),
            deny_modification=paths("deny_modification"),
        )

    def resolve(self, *, root: Path | None = None, descriptor: Path | None = None) \
            -> tuple[Path, dict[str, Any]]:
        """Validate, load and identify the exact package grading is allowed to use.

        The content digest uses the same tree-hash implementation as bundle locks.  It therefore joins
        the descriptor-selected path to the bytes the experiment granted at launch and gives every model
        result an independently checkable host-compiler identity.
        """
        root = (root or repo_root()).resolve()
        package_rel = _safe_relative(self.package, field="package")
        read_only = tuple(_safe_relative(path, field="read_only") for path in self.read_only)
        denied = tuple(_safe_relative(path, field="deny_modification")
                       for path in self.deny_modification)
        if not read_only:
            raise ValueError("host_lane grants no read-only path; its package is not pinned")
        if not self.requires_paths:
            raise ValueError("host_lane.requires_paths is empty; package content is not pinned")
        if not any(_is_within(package_rel, grant) for grant in read_only):
            raise ValueError(
                f"host_lane package {self.package!r} is outside every read-only grant; the agent and "
                "grader would not share the declared compiler")
        masked_by = [str(path) for path in denied if _is_within(package_rel, path)]
        if masked_by:
            raise ValueError(
                f"host_lane package {self.package!r} is masked by denied path(s) {masked_by}")

        package_lexical = root / package_rel
        try:
            package = package_lexical.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"host_lane package {self.package!r} is missing or unreadable") from exc
        if not _is_within(package, root) or not package.is_dir():
            raise ValueError(
                f"host_lane package {self.package!r} does not resolve to a directory inside {root}")

        # A symlink would let a path that looks pinned consume bytes outside the hashed/granted tree.
        # The sandbox snapshot rejects the same shape, so grading and agent visibility stay congruent.
        relative = package_lexical.relative_to(root)
        prefixes = (root.joinpath(*relative.parts[:i]) for i in range(1, len(relative.parts) + 1))
        symlinks = [path for path in prefixes if path.is_symlink()]
        symlinks += [path for path in package.rglob("*") if path.is_symlink()]
        if symlinks:
            raise ValueError(f"host_lane package contains a symlink: {symlinks[0]}")

        required_paths: list[str] = []
        missing_required: list[str] = []
        for raw in self.requires_paths:
            rel = _safe_relative(raw, field="requires_paths")
            required_paths.append(rel.as_posix())
            if not (package / rel).exists():
                missing_required.append(rel.as_posix())
        if missing_required:
            raise ValueError(
                f"host_lane package {self.package!r} is missing required path(s) {missing_required}")

        # The real loader is part of validation: presence alone is insufficient if the manifest/knobs
        # are malformed or if knobs redirect the schedule outside the package whose digest we record.
        from ..mining.registry import load_rvv_package
        loaded = load_rvv_package(package)
        schedule_rel = _safe_relative(
            str(loaded.knobs.get("schedule_file", "schedule.mlir")), field="schedule_file")
        schedule = package / schedule_rel
        if not schedule.is_file() or schedule.is_symlink():
            raise ValueError(
                f"host_lane schedule {schedule_rel.as_posix()!r} is not a regular in-package file")
        try:
            schedule.resolve(strict=True).relative_to(package)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"host_lane schedule {schedule_rel.as_posix()!r} escapes the pinned package") from exc

        from merlin.benchharness import hash_tree
        hashed = hash_tree(package)
        digest = hashed.get("sha256")
        if not hashed.get("present") or not digest or int(hashed.get("n_files") or 0) < 1:
            raise ValueError(f"host_lane package {self.package!r} has no hashable content")
        identity = {
            "descriptor": str(descriptor) if descriptor else None,
            "package": package_rel.as_posix(),
            "resolved_package": str(package),
            "package_sha256": str(digest),
            "n_files": int(hashed["n_files"]),
            "required_paths": required_paths,
            "read_only_grants": [path.as_posix() for path in read_only],
            "repo_canonical": self.repo_canonical,
            "branch": self.branch,
            "commit": self.commit,
            "target": loaded.name,
            "run_id": loaded.run_id,
            "dtype_strategy": loaded.dtype_strategy,
            "schedule_file": schedule_rel.as_posix(),
        }
        return package, identity


@dataclass(frozen=True)
class TargetExperiment:
    """The declarative SETUP for one target's experiment (derivable facts are NOT here)."""
    target: str
    isa_headers: tuple[str, ...]       # shared hardware-spec headers (bundle-convention path STRINGS)
    hwbringup_set: str | None          # shared RTL/ISA/README/example set (bundle-convention path STRING)
    # OPTIONAL declarative setup: the curated baremetal C harness (linker/crt/headers, NO kernels) an
    # agent's compiler needs — only chipyard-sim targets have one; arc/cyclotron targets omit it. A path
    # relative to the experiment dir. Genuinely per-target setup, so declared (not derived).
    curated_harness: str | None
    capsule_corpus: Path               # the corpus the arms author against + are graded on (resolved)
    sim_via: str                       # how the simulator runs (e.g. "chipyard")
    rtl_via: str                       # how RTL facts are obtained (e.g. "mlc" — DERIVED, not declared)
    # OPTIONAL: where the accelerator's RTL lives (a local path or a URL). When set, the descriptor
    # itself points at the RTL so onboarding can validate the pointer + wire mlc discovery at it, rather
    # than ASSUMING the RTL was separately registered with mlc. None (the default) keeps the legacy
    # contract: the RTL is already registered with mlc under ``target``. Additive + backward-compatible.
    rtl_repo: str | None
    # Prior backends / reference exemplars the agent must NOT read/copy (an experiment CHOICE, so
    # declared, not derived). Names under ``artifacts/targets/<target>/``.
    prior_backends: tuple[str, ...]
    path: Path                         # the descriptor file this came from
    # OPTIONAL: the KNOWN-GOOD self-contained model program the pre-flight runs end-to-end through the
    # grading oracle (assemble→cosim→readback, compared bit-exact to its own golden) to prove the oracle
    # produces a correct verdict BEFORE a paid run — not just that ``arc_available`` is True. Genuinely
    # per-target SETUP (which shipped validation program to smoke), so declared, not derived. Only an
    # ``external_backend`` (self-hosted-ISA program-oracle) target needs one; others leave it None.
    preflight_smoke_program: str | None = None
    # OPTIONAL: repo-relative dir of the BACKEND package whose ``contracts/`` hold this target's
    # rtl_facts / irdl pins, when it is NOT ``merlin/targets/<target>``. An experiment target can be
    # served by a differently-named core package; leaving that to be inferred from the target name
    # yields bundle grants pointing at paths that cannot exist (a CIRCT arm granted nothing while the
    # manifest claims otherwise). Declared, never inferred; default preserves same-name targets.
    backend_package_dir: str | None = None
    # OPTIONAL: the contract the descriptor DECLARES as this target's capability manifest, repo-root
    # relative. It was parsed and thrown away before — no field held it — so every descriptor's
    # ``hardware_spec.target_contract`` was dead data, and what the tooling actually read was whatever
    # ``target_registry.resolve(target)`` found by name. That was invisible in both directions: for one
    # target the two paths resolve to DIFFERENT contracts (one naming its fp8 datapaths, the other
    # carrying the fail-closed ``unnamed_float_datapaths`` derivation), and for another the registry
    # resolves NOTHING while the declaration is right there, which is why its STARTER_PROMPT.md silently
    # failed to render. Kept as a declaration, deliberately NOT as an override: see
    # :func:`declared_vs_resolved_contract`.
    declared_contract: str | None = None
    # OPTIONAL: capsule DIRECTORY NAMES this experiment withholds from the PUBLIC graded set. An
    # experiment CHOICE about scope (which capsules a paid agentic loop is scored on), so declared per
    # target, never inferred — the library reads it as data and knows nothing about any target's corpus.
    #
    # Why the knob exists: a whole-model capsule costs one oracle invocation per matmul layer, and the
    # cost is the MODEL's, not the compiler's. Measured on radiance: 15 layers ~= 45 min, and the four
    # model capsules together are 297 layers ~= 15 h per arm per ROUND, which makes a 12-round A/B
    # unreachable while adding nothing the first model has not already shown. Withholding is honest only
    # because it is visible: the excluded names land in the run's own manifest and the denominator moves
    # with them. It is NOT a way to drop capsules a submission fails — see the fail-closed check in
    # :func:`~merlin.targetgen.contract.materialize.materialize_public_capsules`, which refuses an
    # exclusion that matches no capsule so a typo cannot quietly widen the set back open.
    graded_exclude: tuple[str, ...] = ()
    # OPTIONAL: which DISCOVERED memory group is this device's on-chip OPERAND store, given as the name
    # prefix its sibling banks share. Only the LABEL is declared; the capacity itself stays RTL-derived
    # (mlc's discovered depth x row_bytes, summed over the group). This exists because mlc classifies a
    # memory map for some targets and refuses for others -- it discovers atlas's 39 SRAMs and then raises
    # "no memory map discovered", so the ``capacity_fit`` contract obligation was undecidable there while
    # being enforced on gemmini. Guessing the operand store from the bank list is not safe (this device's
    # instruction memory is LARGER than its operand file, so "the biggest one" picks IMEM), so the
    # descriptor names it and the bytes are still read out of the RTL.
    operand_store: str | None = None
    # OPTIONAL for legacy/minimal descriptors. Targeted whole-model grading requires it and fails closed
    # when absent; keeping None loadable lets descriptor tooling report the omission instead of making
    # unrelated onboarding helpers crash while parsing an otherwise useful partial descriptor.
    host_lane: HostLane | None = None
    # OPTIONAL: which BOARD this target's host lane compiles for, a key of ``merlin.runtime.boards``.
    #
    # Nothing declared it anywhere, and the omission was silent in the worst way: ``system_for`` returns
    # ``System(host=None, ...)`` unless a board is passed, ``place.host_units(None)`` then synthesizes
    # only a scalar host, and every placement that should have landed on the host VECTOR lane landed on
    # a scalar one instead -- with no error, because "no board" and "a board with no vector unit" were
    # the same value. The only caller that had a host at all was a test, which hardcoded one.
    #
    # Declared rather than derived because the board is a fact about the SoC the accelerator sits in,
    # not about the accelerator: two targets can share an RTL and be brought up on different boards.
    # Left None where the evidence does not name one -- ``system_for_experiment`` then reports the
    # omission instead of fabricating a host, since a made-up host is worse than an absent one.
    host_board: str | None = None

    def resolve_host_lane(self, *, root: Path | None = None) -> tuple[Path, dict[str, Any]]:
        if self.host_lane is None:
            raise ValueError(f"{self.path}: targeted whole-model grading requires a host_lane declaration")
        return self.host_lane.resolve(root=root, descriptor=self.path)

    def declared_contract_path(self) -> Path | None:
        """The declared contract as an absolute path, if the descriptor names one that exists."""
        if not self.declared_contract:
            return None
        p = repo_root() / self.declared_contract
        return p if p.is_file() else None

    @property
    def exp_name(self) -> str:
        """The experiment dir path RELATIVE TO ``merlin/experiments`` (e.g.
        ``capsule_bench/targets/gemmini``) — for the exp-scoped bundle paths. In the target-neutral layout
        each target lives under ``capsule_bench/targets/<target>/``; this returns that full relative path
        (not just the leaf) so ``experiments/{exp_name}/...`` reconstructs the real location. Falls back to
        the bare dir name when the descriptor is not under an ``experiments/`` root."""
        d = self.path.parent
        for anc in d.parents:
            if anc.name == "experiments":
                return str(d.relative_to(anc))
        return d.name

    # DERIVED target-specific paths (bundle-convention strings) — from the backend package, never
    # hand-listed.
    @property
    def backend_package(self) -> str:
        """Repo-relative dir of the target's BACKEND package, where its contracts/ live.

        Defaults to ``merlin/targets/<target>`` — true whenever the experiment target and the backend
        package share a name (gemmini, atlas, ...). It is NOT universally true: an experiment target may
        be served by a differently-named core package (a SIMT experiment served by its core's package),
        and assuming otherwise silently produces grant paths that can never exist — a bundle that
        *looks* like it hands the CIRCT arm its RTL facts while handing it nothing. So the mapping is a
        DECLARED fact (``backend_package`` in the descriptor) whenever it differs, never inferred."""
        d = self.backend_package_dir
        return str(d).rstrip("/") if d else f"merlin/targets/{self.target}"

    @property
    def rtl_facts_pin(self) -> str:
        return f"{self.backend_package}/contracts/rtl_facts/"

    @property
    def irdl_pin(self) -> str:
        return f"{self.backend_package}/contracts/irdl/"

    def corpus_rel(self) -> str:
        """The capsule corpus as a repo-root-relative string (bundle convention)."""
        return str(self.capsule_corpus.relative_to(repo_root())) + "/"

    def corpus_siblings(self) -> list[str]:
        """Sibling capsule CATEGORIES that actually EXIST beside the primary corpus (e.g. layers/
        model_slices) — globbed, not a hardcoded gemmini taxonomy. Repo-root-relative strings.

        A sibling category holds capsule dirs DIRECTLY (``d/*/capsule.yaml``). A subdir that instead holds
        its OWN categories (``d/*/*/capsule.yaml``) is a different TARGET's corpus that merely nests under
        the same parent (e.g. ``capsules/atlas/`` beside gemmini's ``capsules/isa``) — it is NOT a sibling
        of this corpus and must be excluded, or a target's capsules leak into another target's set."""
        parent = self.capsule_corpus.parent
        out = []
        for d in sorted(parent.iterdir()) if parent.is_dir() else []:
            if (d.is_dir() and d != self.capsule_corpus and d.name != "hidden"
                    and not d.name.startswith(("_", "."))          # skip __pycache__/dotdirs, not corpora
                    and next(d.glob("*/capsule.yaml"), None) is not None):  # a CATEGORY, not a nested corpus
                out.append(str(d.relative_to(repo_root())) + "/")
        return out

    def hidden_corpus(self) -> str | None:
        """The hidden-capsule deny path (sibling ``hidden/`` of the corpus), if present."""
        h = self.capsule_corpus.parent / "hidden"
        return str(h.relative_to(repo_root())) + "/" if h.is_dir() else None

    def graded_roots(self) -> list[Path]:
        """Every root the PUBLIC/dev grade must read: the primary corpus plus its sibling categories.

        This exists because "the corpus" and "what gets graded" had drifted apart. The descriptor names
        one directory (``.../isa``), but a target's capsules are split by kind across sibling categories,
        so grading the named directory alone scores a subset — for this repo's targets, 8 of 20 for one
        and 21 of 28 for another — while reporting a clean denominator.

        The obvious alternative, grading their shared parent, is wrong in the other direction: that
        parent also holds other targets' corpora, which would leak foreign capsules into the suite.
        :meth:`corpus_siblings` already draws that line structurally (a CATEGORY holds capsule dirs
        directly; a nested corpus holds categories), so this is just its resolved form.
        """
        return [self.capsule_corpus, *(repo_root() / s for s in self.corpus_siblings())]

    def hidden_roots(self) -> list[Path]:
        """The roots the HIDDEN grade must read — empty when the target ships no hidden capsules.

        Empty is a real answer and the caller must not paper over it: grading the public roots with the
        ``hidden`` label matches nothing and yields a 0/0 "pass" that never ran.
        """
        h = self.hidden_corpus()
        return [repo_root() / h] if h else []


def load_target_experiment(descriptor: str | Path) -> TargetExperiment:
    """Load + validate a ``target_experiment.yaml`` descriptor. Shared-spec paths are kept as the bundle-
    convention STRINGS (so the governance check compares like-for-like); the capsule corpus is resolved."""
    p = Path(descriptor)
    doc = yaml.safe_load(p.read_text())
    if not isinstance(doc, dict) or not doc.get("target"):
        raise ValueError(f"{p}: not a target-experiment descriptor (missing 'target')")
    root = repo_root()
    hw = doc.get("hardware_spec") or {}
    return TargetExperiment(
        target=str(doc["target"]),
        isa_headers=tuple(hw.get("isa_headers") or []),
        hwbringup_set=hw.get("hwbringup_set"),
        curated_harness=hw.get("curated_harness"),
        capsule_corpus=root / doc["capsule_corpus"] if doc.get("capsule_corpus") else None,
        sim_via=str((doc.get("toolchain") or {}).get("sim_via", "")),
        rtl_via=str((doc.get("rtl") or {}).get("via", "mlc")),
        rtl_repo=(lambda r: str(r) if r else None)((doc.get("rtl") or {}).get("repo")),
        operand_store=(lambda v: str(v) if v else None)((doc.get("rtl") or {}).get("operand_store")),
        prior_backends=tuple((doc.get("answer_surfaces") or {}).get("prior_backends") or ()),
        path=p,
        preflight_smoke_program=(lambda s: str(s) if s else None)(
            (doc.get("preflight") or {}).get("smoke_program")),
        declared_contract=(lambda s: str(s) if s else None)(hw.get("target_contract")),
        backend_package_dir=(lambda s: str(s) if s else None)(doc.get("backend_package_dir")),
        graded_exclude=tuple(str(x) for x in ((doc.get("grading") or {}).get("exclude_capsules") or ())),
        host_lane=HostLane.from_mapping(doc.get("host_lane"), descriptor=p),
        host_board=(lambda v: str(v) if v else None)((doc.get("host") or {}).get("board")),
    )


def declared_vs_resolved_contract(te: TargetExperiment) -> tuple[Path | None, Path | None, str]:
    """``(declared, resolved, verdict)`` for this target's capability contract.

    Verdicts: ``"agree"`` (same file, or nothing declared and one resolves), ``"declared_only"`` (the
    registry resolves nothing — the declaration is the only contract there is), ``"stale_declaration"``
    (the declared path does not exist but the registry resolves one), ``"mismatch"`` (both exist and are
    DIFFERENT files), or ``"none"`` (no contract at all).

    A mismatch is reported, never resolved here. Picking a winner would silently change what an agent is
    told the hardware is — the two saturn_opu contracts disagree on whether its fp8 datapaths are named
    (``fp8_e4m3``/``fp8_e5m2``) or honestly unnamed (``float8`` + ``unnamed_float_datapaths``), and that is
    the difference between a target that declares two formats and one that admits its RTL does not name
    them. Which is authoritative is a call for whoever owns the contract, so this surfaces it and the
    readiness gate fails on it.
    """
    from . import target_registry
    declared = te.declared_contract_path()
    try:
        resolved = target_registry.resolve(te.target).contract_path
        resolved = resolved if resolved and Path(resolved).is_file() else None
    except Exception:  # noqa: BLE001 — an unresolvable target is one of the answers
        resolved = None
    if declared and resolved:
        same = Path(declared).resolve() == Path(resolved).resolve()
        return declared, resolved, "agree" if same else "mismatch"
    if declared:
        return declared, None, "declared_only"
    if resolved:
        return None, resolved, "stale_declaration" if te.declared_contract else "agree"
    return None, None, "none"


def shared_spec_paths(te: TargetExperiment) -> set[str]:
    """The shared hardware-spec path strings the descriptor makes authoritative — the ISA headers + the
    hwbringup set EVERY arm's bundle must grant (a constant input, not assistance)."""
    paths = set(te.isa_headers)
    if te.hwbringup_set:
        paths.add(te.hwbringup_set)
    return paths


def bundles_match_descriptor(te: TargetExperiment, manifest_paths) -> list[str]:
    """Governance: the descriptor is the single source of truth for the shared hardware spec. Return the
    drift — for each bundle manifest, the shared-spec paths it fails to grant in ``allowed``. Empty list
    means every arm's bundle is consistent with the descriptor (so a run for this target is honest)."""
    required = shared_spec_paths(te)
    drift: list[str] = []
    for mp in manifest_paths:
        doc = yaml.safe_load(Path(mp).read_text())
        allowed = {e.get("path") for e in (doc.get("allowed") or []) if isinstance(e, dict)}
        missing = required - allowed
        if missing:
            drift.append(f"{Path(mp).parent.name}: missing shared-spec {sorted(missing)}")
    return drift


# --------------------------------------------------------------- derived ABI readout-bit convention
# The RoCC accumulator-readout bits are NOT magic hex: every one is a fixed function of ``addr_len`` plus
# the accelerator's 3-flag-bit accumulator-address convention (the top three bits of the DIM-relative
# scratchpad/accumulator address field) + the universal IEEE-754 float32(1.0) scale literal. The
# convention is grounded in the target's own ISA header address arithmetic (gemmini.h:
# ``D_sp_addr_start = 1 << (ADDR_LEN-1)``; ``C_sp_addr_start = (3 << (ADDR_LEN-2)) | (full_C << (ADDR_LEN-3))``;
# accumulate cleared via ``&= ~(1 << (ADDR_LEN-2))``), so the five constants are re-derived here rather than
# hand-declared. This retires the ``readout_bits`` hex block from the contract; the residue is only the
# {addr_len anchor + the 3-flag-bit role assignment} convention, which now lives in code (documented), not
# as opaque literals. Cross-checked byte-identical against the frozen hex by test_encoding_manifest.
_F32_ONE_BITS = struct.unpack("<I", struct.pack("<f", 1.0))[0]  # 0x3F800000 — IEEE-754 float32(1.0)


def derived_readout_bits(addr_len: int) -> dict[str, int]:
    """The RoCC accumulator-readout bit constants DERIVED from ``addr_len`` + the 3-flag-bit accumulator-
    address convention + float32(1.0). Byte-identical to the former hand-declared ``readout_bits`` hex:

      * ``acc_i8``     = ``1 << (addr_len-1)`` — accumulator-select / scaled-i8 readout base (bit 31)
      * ``acc_accum``  = ``1 << (addr_len-2)`` — accumulate-onto (vs overwrite) (bit 30)
      * ``full_c_bit`` = ``1 << (addr_len-3)`` — full-i32 (vs scaled-i8) readout width (bit 29)
      * ``c_acc``      = ``acc_i8 | full_c_bit`` — full-i32 accumulator readout base (0xA0000000)
      * ``f1``         = float32(1.0) bits (0x3F800000) — the identity acc_scale
    """
    acc_i8 = 1 << (addr_len - 1)
    acc_accum = 1 << (addr_len - 2)
    full_c_bit = 1 << (addr_len - 3)
    return {"f1": _F32_ONE_BITS, "c_acc": acc_i8 | full_c_bit,
            "acc_i8": acc_i8, "acc_accum": acc_accum, "full_c_bit": full_c_bit}


# --------------------------------------------------------------------------- capability manifest
@dataclass(frozen=True)
class CapabilityManifest:
    """The per-target capability model that drives GENERATION — a human-reviewed cache derived from RTL
    facts + the designer's docs (the committed ``target_contract.yaml``), NOT hand-invented for merlin.

    It resolves the target's PRIMARY compute-unit ``kind`` (the unit not embedded in another) and, via
    the family registry, the generation defaults (codegen endpoint, RTL tiers, perf fields, whether an
    op->``.insn`` encoding derivation + trace gate apply). Any default may be overridden by an optional
    ``runner``/``endpoint_kind`` block in the contract. Core generators consult this by ``kind`` so they
    never branch on a target name."""
    target: str
    kind: str                      # primary compute-unit kind (systolic|simt|vector|scalar)
    endpoint_kind: str             # inline_asm_insn (default) | upstream_target | external_backend | command_buffer
    suite: str
    dtype: str                     # run-identity dtype token (e.g. i8xi8_i32, f32)
    fourth_output_name: str | None # None -> the runner derives it from endpoint_kind
    tier_sim: dict                 # tier -> sim name (empty -> family/arc default)
    rtl_tiers: tuple[str, ...]
    perf_fields: tuple[str, ...]
    trace_gate: str | None         # trace-gate plugin name (e.g. "rocc_insn") or None
    force_match_policy: dict | None  # optional oracle output-equality override (float target -> {compare,atol})
    encoding_required: bool
    encoding: dict                 # the ABI encoding surface RTL can't ground (readout_bits/semantic_class/...)
    contract: dict                 # the full target_contract.yaml (for consumers that need more)


def _primary_kind(units) -> str:
    """The kind of the target's primary compute unit = the one NOT contained by any other."""
    contained = {c for u in units for c in u.contains}
    primary = [u for u in units if u.name not in contained]
    return (primary[0] if primary else units[0]).kind


def _derived_dtype_token(units) -> str:
    """A run-identity dtype token DERIVED from the primary compute unit's first accumulate rule
    (``<in>x<weight>_<acc>``). Replaces the former gemmini ``i8xi8_i32`` fail-open default so a target
    that omits ``runner.dtype`` (e.g. an mx target) is labeled by its OWN datapath, never mislabeled as
    gemmini int8. Falls back to ``"unknown"`` (fail-closed, surfaced in the run label) if no rule."""
    for u in units:
        if u.accumulate:
            a = u.accumulate[0]
            if a.inp and a.acc:
                return f"{a.inp}x{a.weight or a.inp}_{a.acc}"
    return "unknown"


def load_capability_manifest(target: str, *,
                             contract_path: str | Path | None = None) -> CapabilityManifest:
    """Load a target's capability manifest from its committed ``target_contract.yaml`` + fill the family
    defaults. Raises if the target has no contract or no compute_units (fail-closed: no fabricated kind).

    ``contract_path`` reads that file instead of asking the registry. It exists for the case where the
    registry resolves NOTHING and a descriptor names the contract explicitly — the alternative there is
    not "use the resolved one", it is "render no prompt at all", which is what used to happen. It is not
    a general override: when the registry does resolve a contract, callers pass nothing and any
    disagreement with the declaration is reported by :func:`declared_vs_resolved_contract`."""
    from . import families, compute_units, target_registry   # lazy: avoid import-order cycles
    if contract_path is not None:
        contract = yaml.safe_load(Path(contract_path).read_text(encoding="utf-8"))
    else:
        contract = target_registry.resolve(target).load_contract()
    units = compute_units.compute_units(contract)
    if not units:
        raise ValueError(f"{target}: target_contract has no compute_units — cannot derive a kind")
    kind = _primary_kind(units)
    prof = families.family_profile(kind)
    runner = contract.get("runner") or {}
    endpoint = contract.get("endpoint_kind") or prof.endpoint_kind_default
    if endpoint not in families.ENDPOINT_KINDS:
        raise ValueError(f"{target}: endpoint_kind {endpoint!r} not in {families.ENDPOINT_KINDS}")
    encoding = dict(contract.get("encoding") or {})
    # ``readout_bits`` is DERIVED, not declared: synthesize it from ``addr_len`` + the 3-flag-bit
    # accumulator-address convention (see derived_readout_bits) when the contract omits it. A contract
    # that still carries an explicit block is honored as-is (back-compat) — but the derivation makes the
    # magic hex unnecessary, so the gemmini contract no longer declares it.
    if "readout_bits" not in encoding and encoding.get("addr_len") is not None:
        encoding["readout_bits"] = derived_readout_bits(int(encoding["addr_len"]))
    return CapabilityManifest(
        target=target, kind=kind, endpoint_kind=endpoint,
        suite=runner.get("suite") or f"{target}-capsule-bench",
        dtype=runner.get("dtype") or _derived_dtype_token(units),
        fourth_output_name=runner.get("fourth_output_name"),
        tier_sim=dict(runner.get("tier_sim") or {}),
        rtl_tiers=tuple(runner.get("rtl_tiers") or prof.default_rtl_tiers),
        perf_fields=tuple(runner.get("perf_fields") or prof.perf_fields),
        # The RoCC-.insn trace gate applies ONLY to an inline_asm_insn (RoCC) endpoint — it decodes a
        # host `.insn` stream from lowered.llvm.mlir. A self-hosted-ISA (external_backend, emits kernel.S)
        # or ISA-less (command_buffer) target has no such stream, so it defaults to no trace gate (unless
        # the contract explicitly declares one). Keys on the endpoint, never a target name.
        trace_gate=runner.get("trace_gate",
                              prof.trace_gate if endpoint == "inline_asm_insn" else None),
        # Optional oracle output-equality override (a float target declares {compare: float, atol: ...}
        # so its oracle comparison is tolerant regardless of the per-capsule numeric_policy). None ->
        # the capsule's own numeric_policy governs (integer capsules -> exact).
        force_match_policy=runner.get("force_match_policy"),
        encoding_required=prof.encoding_required,
        encoding=encoding,
        contract=contract)
