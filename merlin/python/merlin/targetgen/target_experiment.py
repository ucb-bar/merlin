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
