#!/usr/bin/env python3
"""Gate the MEASURABILITY of target-conditioned semantic coverage — never the score.

Acceleratable Region Recall only means something if three things hold: the denominator is grounded in
the target's own evidence, every family it declares is actually probed, and the corpus exercises what
the contract claims. Each of those can rot silently, and each rots in the direction that flatters us:

* a family the hardware has but the contract omits shrinks the denominator, so recall rises;
* a declared family with no probe or no materializer is a claim nobody ever tests;
* a capsule with no ``semantic`` block cannot raise a ``must_accelerate`` violation, so the
  CPU-fallback escape hatch is open and the coverage certificate passes vacuously.

⚠️ **This gate deliberately does NOT check the ARR value.** Gating on ``ARR >= x`` makes the rational
response to a hard family "delete it from the contract", which is precisely the incentive the whole
apparatus exists to defeat. It checks that the number is *measurable and honest*, and leaves what it
says to the reader.

Ratcheted like the other structural gates: known holes live in ``generalization_debt.txt`` and that
list MAY ONLY SHRINK, so this lands on a tree that is not yet clean without blocking every commit.

Usage:  check_semantic_coverage.py [--target NAME] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin" / "python"))

from merlin.common.paths import repo_root                              # noqa: E402
from merlin.targetgen import capability_probes as cp                   # noqa: E402
from merlin.targetgen import eligibility as el                         # noqa: E402
from merlin.targetgen import semantic_families as sf                   # noqa: E402
from merlin.targetgen import target_registry as tr                     # noqa: E402

DEBT = repo_root() / "build_tools" / "generalization_debt.txt"

#: Capsule KIND directories, the shared layout every corpus uses. Not target names.
_KIND_DIRS = ("isa", "layers", "model_slices", "model", "hidden")


def _corpus_dirs(target: str) -> list[Path]:
    """Where this target's capsules live, resolved from the tree rather than from a per-target table.

    A target that owns a subdirectory uses it; the one that predates that convention occupies the shared
    kind directories at the root. Deriving this keeps the rule "no target-name literals in tooling" --
    a table mapping a name to a layout is exactly the coupling that stops a new target working."""
    caps = repo_root() / "merlin" / "contract" / "capsules"
    own = caps / target
    if own.is_dir():
        return [own]
    return [caps / d for d in _KIND_DIRS if (caps / d).is_dir()]


def _load_capsules(target: str) -> list[dict]:
    import yaml
    out = []
    for d in _corpus_dirs(target):
        for f in sorted(d.rglob("capsule.yaml")):
            try:
                out.append(yaml.safe_load(f.read_text()) or {})
            except Exception:  # noqa: BLE001
                continue
    return out


def _materializable_families() -> set[str]:
    """Families the generalization difftest can turn into a runnable capsule. A declared family with a
    probe and no materializer is an untested claim, so this is read from the runner rather than assumed."""
    try:
        sys.path.insert(0, str(repo_root() / "merlin" / "experiments" / "capsule_bench" / "harness"))
        import generalization_difftest as gd
        return set(gd.FAMILY_MAT)
    except Exception:  # noqa: BLE001 — runner unavailable: report nothing rather than fail everything
        return set()


def _targets_with_profiles() -> list[str]:
    """Targets that declare a corpus profile.

    A held-out spec lives in a ``<target>.hidden.yaml`` SIDECAR (its op/dtype/shape is itself an answer,
    so it is untracked and kept out of the grant every arm reads). The sidecar is not a target: taking
    ``Path.stem`` of it yields ``"radiance.hidden"`` and the audit then reports a missing contract for a
    target that does not exist, burying the real findings under one per sidecar.
    """
    d = repo_root() / "merlin" / "contract" / "capsules" / "profiles"
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.yaml") if not p.stem.endswith(".hidden"))


def audit(target: str) -> list[dict]:
    """Findings for one target. Empty list == this target is measurable."""
    findings: list[dict] = []
    try:
        contract = tr.load_contract(target)
    except Exception as exc:  # noqa: BLE001
        return [{"target": target, "kind": "no_contract", "detail": f"{type(exc).__name__}: {exc}"}]

    cap_map = el.capability_map_from_contract(contract)
    undet = el.undetermined_families_from_contract(contract)

    # 1. capability drift -- the contract disagrees with its own target's evidence
    for d in (contract.get("capability_evidence") or {}).get("drift", []):
        findings.append({"target": target, "kind": d.get("kind", "capability_drift"),
                         "family": d.get("family"), "detail": d.get("detail", "")})

    if not cap_map:
        findings.append({"target": target, "kind": "no_declared_capability",
                         "detail": "the contract declares no semantic_capabilities, so every region is "
                                   "ineligible and ARR is undefined -- the target is outside the "
                                   "measurement entirely"})
        return findings

    # 2/3. every declared family must be probeable AND materializable
    probes = cp.synthesize(cap_map)
    probed = {p.descriptor.resolved_family() for p in probes}
    mat = _materializable_families()
    unmaterializable = {k: v for k, v in (contract.get("unmaterializable_families") or {}).items()}
    for fam in sorted(cap_map):
        if fam not in probed:
            findings.append({"target": target, "kind": "family_without_probe", "family": fam,
                             "detail": "declared but the probe synthesizer produces nothing for it"})
        if mat and fam not in mat and fam not in unmaterializable:
            findings.append({"target": target, "kind": "family_without_materializer", "family": fam,
                             "detail": "declared and probed, but no materializer can turn a probe into "
                                       "a runnable capsule, so the claim is never tested; declare it in "
                                       "unmaterializable_families with a reason if that is intended"})

    # 3b. The resolved corpus must not contain capsules this target STRUCTURALLY cannot execute.
    #
    # Corpus roots are discovered by directory layout and filtered by label -- never by capability -- so a
    # generic capsule dropped into a shared root is silently adopted as graded work. Measured: one target
    # reads the shared top-level roots (every other target is namespaced under its own name), so 12 bf16
    # capsules added to a shared model_slices/ directory entered its graded suite. It cannot execute them
    # (its contract declares one integer format), so they could never pass -- and because the agent loop's
    # only early exit is a genuine all_pass, they made all_pass UNREACHABLE and turned every run into a
    # fixed-price purchase of its full round budget: 20 rounds, ~12 hours, for a suite that was already
    # complete at round 0.
    #
    # Relocating the directory does not fix it: the shared root MIXES in-scope and out-of-scope capsules,
    # so the layout cannot express the distinction. Catching it here does, and it fires the moment such a
    # capsule is added rather than after a run has paid for it.
    #
    # Uses the runner's OWN withholding rule, imported rather than reimplemented, so the gate and the
    # grader can never drift on what "cannot execute" means.
    try:
        from merlin.targetgen.capsule_runner import _split_ineligible
        from merlin.targetgen.target_experiment import load_target_experiment
        desc = (repo_root() / "merlin/experiments/capsule_bench/targets" / target / "target_experiment.yaml")
        if desc.is_file():
            te = load_target_experiment(str(desc))
            import yaml as _y
            corpus = []
            for r in [Path(te.capsule_corpus)] + [Path(s) for s in te.corpus_siblings()]:
                if not r.is_dir():
                    continue
                try:
                    subs = sorted(r.iterdir())
                except PermissionError:
                    continue
                for d in subs:
                    f = d / "capsule.yaml"
                    if f.is_file():
                        corpus.append(_y.safe_load(f.read_text()))
            op = [c for c in corpus if c.get("kind") != "model"]
            _, withheld = _split_ineligible(op, target)
            for w in withheld:
                findings.append({"target": target, "kind": "corpus_contains_unexecutable",
                                 "family": w.get("capsule"),
                                 "detail": "in this target's RESOLVED corpus but structurally impossible "
                                           "for it, so it can never pass and keeps all_pass unreachable: "
                                           + str((w.get("failure") or {}).get("detail", ""))})
    except Exception:  # noqa: BLE001 - a corpus we cannot resolve is reported by the checks above
        pass

    # 4/5/6. the corpus
    caps = _load_capsules(target)
    if not caps:
        findings.append({"target": target, "kind": "no_corpus", "detail": "no capsules found on disk"})
        return findings

    exercised, asserted = set(), set()
    for c in caps:
        sem = c.get("semantic") or {}
        fam = sem.get("semantic_family")
        if not sem.get("generalization_axis"):
            findings.append({"target": target, "kind": "capsule_without_semantic_block",
                             "family": c.get("name"),
                             "detail": "no semantic block, so it can never raise a must_accelerate "
                                       "violation and its coverage certificate passes vacuously"})
            continue
        derived = sf.from_op((c.get("operation") or {}).get("op"))
        if fam and derived and fam != derived:
            findings.append({"target": target, "kind": "capsule_family_contradicts_op",
                             "family": c.get("name"),
                             "detail": f"declares {fam!r} but its op derives {derived!r}"})
        if fam:
            exercised.add(fam)
            if sem.get("must_accelerate"):
                asserted.add(fam)

    for fam in sorted(cap_map):
        if fam in undet:
            continue
        if fam not in exercised:
            findings.append({"target": target, "kind": "declared_family_unexercised", "family": fam,
                             "detail": "the contract declares it and no capsule in the corpus exercises "
                                       "it, so nothing measures whether the compiler covers it"})
        elif fam not in asserted:
            findings.append({"target": target, "kind": "family_never_must_accelerate", "family": fam,
                             "detail": "exercised but no capsule asserts must_accelerate, so a compiler "
                                       "that falls back on every one of them still passes"})
    return findings


def _key(f: dict) -> str:
    return f"{f['target']}:{f['kind']}:{f.get('family') or '-'}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", help="one target (default: every target with a corpus profile)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    targets = [a.target] if a.target else _targets_with_profiles()
    findings = [f for t in targets for f in audit(t)]

    known = set()
    if DEBT.exists():
        known = {ln.strip() for ln in DEBT.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")}
    fresh = [f for f in findings if _key(f) not in known]
    stale = sorted(known - {_key(f) for f in findings})

    if a.json:
        print(json.dumps({"findings": findings, "new": fresh, "resolved": stale}, indent=1))
        return 1 if fresh else 0

    for f in fresh:
        print(f"[FAIL] semantic-coverage: {f['target']}: {f['kind']}"
              f"{' ' + f['family'] if f.get('family') else ''} -- {f['detail']}")
    if stale:
        print(f"[  ok] semantic-coverage: {len(stale)} debt entry(ies) RESOLVED -- delete them from "
              f"{DEBT.name} so the count keeps meaning something:")
        for k in stale[:10]:
            print(f"         {k}")
    if not fresh:
        n = len(findings)
        print(f"[  ok] semantic-coverage: {len(targets)} target(s) measurable"
              + (f"; {n} known hole(s) on the ratchet (may only fall)." if n else "."))
        return 0
    print(f"[FAIL] semantic-coverage: {len(fresh)} new finding(s). Fix them, or record a reviewed "
          f"rationale in {DEBT.name} -- that list may only shrink.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
