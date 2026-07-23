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

from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.common.paths import repo_root


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
    # Prior backends / reference exemplars the agent must NOT read/copy (an experiment CHOICE, so
    # declared, not derived). Names under ``artifacts/targets/<target>/``.
    prior_backends: tuple[str, ...]
    path: Path                         # the descriptor file this came from

    @property
    def exp_name(self) -> str:
        """The experiment directory name (e.g. ``gemmini_capsule_bench_v0``) — for the exp-scoped paths."""
        return self.path.parent.name

    # DERIVED target-specific paths (bundle-convention strings) — from ``target``, never hand-listed.
    @property
    def rtl_facts_pin(self) -> str:
        return f"merlin/targets/{self.target}/contracts/rtl_facts/"

    @property
    def irdl_pin(self) -> str:
        return f"merlin/targets/{self.target}/contracts/irdl/"

    def corpus_rel(self) -> str:
        """The capsule corpus as a repo-root-relative string (bundle convention)."""
        return str(self.capsule_corpus.relative_to(repo_root())) + "/"

    def corpus_siblings(self) -> list[str]:
        """Sibling corpora that actually EXIST beside the primary corpus (e.g. layers/model_slices) —
        globbed, not a hardcoded gemmini taxonomy. Repo-root-relative strings."""
        parent = self.capsule_corpus.parent
        out = []
        for d in sorted(parent.iterdir()) if parent.is_dir() else []:
            if (d.is_dir() and d != self.capsule_corpus and d.name != "hidden"
                    and not d.name.startswith(("_", "."))):   # skip __pycache__/dotdirs, not corpora
                out.append(str(d.relative_to(repo_root())) + "/")
        return out

    def hidden_corpus(self) -> str | None:
        """The hidden-capsule deny path (sibling ``hidden/`` of the corpus), if present."""
        h = self.capsule_corpus.parent / "hidden"
        return str(h.relative_to(repo_root())) + "/" if h.is_dir() else None


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
        prior_backends=tuple((doc.get("answer_surfaces") or {}).get("prior_backends") or ()),
        path=p,
    )


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
    endpoint_kind: str             # inline_asm_insn (default, fork-free) | upstream_target | external_backend
    suite: str
    dtype: str                     # run-identity dtype token (e.g. i8xi8_i32, f32)
    fourth_output_name: str | None # None -> the runner derives it from endpoint_kind
    tier_sim: dict                 # tier -> sim name (empty -> family/arc default)
    rtl_tiers: tuple[str, ...]
    perf_fields: tuple[str, ...]
    trace_gate: str | None         # trace-gate plugin name (e.g. "rocc_insn") or None
    encoding_required: bool
    encoding: dict                 # the ABI encoding surface RTL can't ground (readout_bits/semantic_class/...)
    contract: dict                 # the full target_contract.yaml (for consumers that need more)


def _primary_kind(units) -> str:
    """The kind of the target's primary compute unit = the one NOT contained by any other."""
    contained = {c for u in units for c in u.contains}
    primary = [u for u in units if u.name not in contained]
    return (primary[0] if primary else units[0]).kind


def load_capability_manifest(target: str) -> CapabilityManifest:
    """Load a target's capability manifest from its committed ``target_contract.yaml`` + fill the family
    defaults. Raises if the target has no contract or no compute_units (fail-closed: no fabricated kind)."""
    from . import families, compute_units, target_registry   # lazy: avoid import-order cycles
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
    return CapabilityManifest(
        target=target, kind=kind, endpoint_kind=endpoint,
        suite=runner.get("suite") or f"{target}-capsule-bench",
        dtype=runner.get("dtype") or "i8xi8_i32",
        fourth_output_name=runner.get("fourth_output_name"),
        tier_sim=dict(runner.get("tier_sim") or {}),
        rtl_tiers=tuple(runner.get("rtl_tiers") or prof.default_rtl_tiers),
        perf_fields=tuple(runner.get("perf_fields") or prof.perf_fields),
        trace_gate=runner.get("trace_gate", prof.trace_gate),
        encoding_required=prof.encoding_required,
        encoding=dict(contract.get("encoding") or {}),
        contract=contract)
