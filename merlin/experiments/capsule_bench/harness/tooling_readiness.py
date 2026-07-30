"""Zero-generation tooling-readiness gate — prove every tool an arm advertises actually produces REAL
output for a target, WITHOUT running a single agent round.

The A/B arms promise the agent a toolset (arm-3 = the CCA compiler-modification spine + xDSL kit;
arm-4 = all of that PLUS the RTL fact/FileCheck surface). If any of those tools silently returns nothing
for the target (as the CCA seam menu did for every non-RVV target before it was wired), a run burns
tokens producing an unusable result and we only find out after. This gate exercises each advertised tool
directly and asserts it yields real output — so a target's readiness is verified statically, before spend.

Fully TARGET-AGNOSTIC: the only input is the target name; every capability + sample is DERIVED (descriptor,
manifest, mlc facts, the target's own capsule corpus). No target literals, no stored facts. It also
enforces the arm contract structurally: arm-4's tool grants ⊇ arm-3's (the delta is exactly the RTL
surface). Run it for a target (``--target atlas``) or import ``readiness(target)``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def _ok(name: str, cond: bool, evidence: str) -> dict:
    return {"check": name, "ok": bool(cond), "evidence": evidence}


def _seam_menu_checks(target: str) -> list[dict]:
    """The CCA seam menu the assisted arms are told to call — must enumerate real modifiable sections."""
    from merlin.kernels import cca_contract as CC, action_catalog as AC
    out = []
    lax = sorted(CC.leverable_axes(target))
    out.append(_ok("cca.leverable_axes non-empty", bool(lax), f"axes={lax}"))
    for ax in lax:
        lad = AC.escalation_ladder(ax, target)
        out.append(_ok(f"cca.escalation_ladder[{ax}] non-empty", bool(lad),
                       f"{len(lad)} rows: {[r['action_class'] for r in lad]}"))
    b = CC.check_bijection(target)
    out.append(_ok("cca.check_bijection clean", not (b.orphan_fields or b.orphan_routes or b.ladder_errors),
                   f"orphan_fields={b.orphan_fields} orphan_routes={b.orphan_routes} errors={b.ladder_errors}"))
    return out


def _derived_lever_checks(target: str) -> list[dict]:
    from merlin.targetgen import rtl_backend as RB
    prof = RB.target_profile(target)
    lev = RB.derived_levers(prof)
    return [_ok("rtl_backend.derived_levers non-empty", bool(lev),
                f"dim={prof.dim} has_mesh={prof.has_mesh} has_accumulator={prof.has_accumulator} levers={lev}")]


def _rtl_fact_checks(target: str) -> list[dict]:
    from merlin.targetgen import rtl_check_runner as RUN
    f = RUN.load_facts(target)
    facts = (f or {}).get("facts", f) or {}
    n = len(facts.get("interfaces") or []) + len(facts.get("arrays") or [])
    return [_ok("rtl facts derivable", n > 0, f"{n} interface/array facts")]


def _rtl_check_checks(target: str) -> list[dict]:
    """The arm-4 RTL FileCheck surface: FileCheck present + endpoint-appropriate checks compile + (for a
    self-hosted target) the ISA-def decode signatures derive. Sample capsule from the target's OWN corpus."""
    import yaml
    from merlin.targetgen import rtl_check_runner as RUN, rtl_check_compiler as CCk
    from merlin.targetgen.target_experiment import load_target_experiment
    out = [_ok("FileCheck binary present", RUN.find_filecheck() is not None, str(RUN.find_filecheck()))]
    desc = C.EXP.parent / target / "target_experiment.yaml"
    corpus = load_target_experiment(desc).capsule_corpus if desc.is_file() else None
    cap_p = next(iter(sorted((corpus or Path("/nonexistent")).rglob("capsule.yaml"))), None) if corpus else None
    if cap_p is None:
        out.append(_ok("rtl checks compile", False, "no capsule in the target corpus"))
        return out
    cap = yaml.safe_load(cap_p.read_text())
    facts = RUN.load_facts(target)
    checks = CCk.compile_checks(facts, cap, target)
    endpoint_check = checks.get("kernel") or checks.get("trace") or checks.get("dialect")
    out.append(_ok("rtl checks compile (endpoint-appropriate)", bool(endpoint_check),
                   f"kernel={bool(checks.get('kernel'))} trace={bool(checks.get('trace'))} "
                   f"dialect={bool(checks.get('dialect'))} (capsule={cap_p.parent.name})"))
    return out


def _oracle_checks(target: str) -> list[dict]:
    from merlin.targetgen import capsule_runner as CR
    ad = CR.oracle_adapters(target)
    kinds = {k: getattr(v, "__qualname__", str(v)).split(".")[0] for k, v in ad.items()}
    return [_ok("oracle adapters resolve to the target's endpoint oracle", bool(ad), f"{kinds}")]


def _arm_superset_check(target: str) -> list[dict]:
    """Structural arm contract: arm-4's granted tools ⊇ arm-3's (the delta is exactly the RTL surface)."""
    import yaml
    b = C.EXP.parent / target / "input_bundles"
    def allow(arm):
        p = b / arm / "input_bundle_manifest.yaml"
        return {a["path"] for a in (yaml.safe_load(p.read_text()).get("allowed") or [])} if p.is_file() else None
    a3, a4 = allow("merlin_assisted_hwbringup_v0"), allow("merlin_assisted_rtlchecks_hwbringup_v0")
    if a3 is None or a4 is None:
        return [_ok("arm-4 ⊇ arm-3 tool grants", False, "a bundle manifest is absent")]
    missing = a3 - a4
    return [_ok("arm-4 ⊇ arm-3 tool grants", not missing,
                f"arm4 adds {sorted(a4 - a3)}; arm4 missing-from-arm3 {sorted(missing)}")]


# arm -> the capability groups that arm advertises (arm-4 is arm-3 ∪ the RTL surface).
_ARM3 = ("seam_menu", "derived_levers")
_ARM4 = _ARM3 + ("rtl_facts", "rtl_checks", "oracle")
_GROUPS = {"seam_menu": _seam_menu_checks, "derived_levers": _derived_lever_checks,
           "rtl_facts": _rtl_fact_checks, "rtl_checks": _rtl_check_checks, "oracle": _oracle_checks}


def readiness(target: str, arm: str = "merlin_assisted_rtlchecks") -> dict:
    """Run the readiness checks for ``target`` at the given arm's tool level (default arm-4 = the full
    superset). Returns {target, arm, checks:[...], ok}. Zero generation."""
    groups = _ARM4 if "rtlchecks" in arm else _ARM3
    checks: list[dict] = []
    for g in groups:
        try:
            checks += _GROUPS[g](target)
        except Exception as e:  # noqa: BLE001 — a dead tool is a readiness FAIL, reported not raised
            checks.append(_ok(f"{g} runnable", False, f"{type(e).__name__}: {str(e)[:160]}"))
    checks += _arm_superset_check(target)
    return {"target": target, "arm": arm, "checks": checks, "ok": all(c["ok"] for c in checks)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--arm", default="merlin_assisted_rtlchecks")
    a = ap.parse_args(argv)
    rep = readiness(a.target, a.arm)
    print(f"=== tooling readiness: target={rep['target']} arm={rep['arm']} ===")
    for c in rep["checks"]:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}: {c['evidence']}")
    print(f"\n  readiness: {'READY' if rep['ok'] else 'NOT READY'}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
