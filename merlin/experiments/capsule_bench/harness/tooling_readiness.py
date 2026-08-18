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


def _is_mesh_target(target: str) -> bool:
    """True iff the target is a systolic/RoCC MESH target (derives a mesh + RoCC funct/array facts). A SIMT
    target (e.g. radiance) legitimately derives NONE of these — its dataflow levers and RoCC facts are
    empty BY DESIGN (cf. test_cross_target: radiance's derived_levers/legal_opcodes are asserted empty). So
    the 'systolic facts non-empty' readiness checks are n/a for it, not failures. Best-effort; a profile
    that cannot be built (no RTL facts) reads as non-mesh (the systolic checks then degrade to n/a)."""
    try:
        from merlin.targetgen import rtl_backend as RB
        return bool(RB.target_profile(target).has_mesh)
    except Exception:  # noqa: BLE001 — no profile -> treat as non-mesh (systolic checks become n/a)
        return False


def _seam_menu_checks(target: str) -> list[dict]:
    """The CCA seam menu the assisted arms are told to call — must enumerate real modifiable sections."""
    from merlin.kernels import cca_contract as CC, action_catalog as AC
    out = []
    lax = sorted(CC.leverable_axes(target))
    if _is_mesh_target(target):
        out.append(_ok("cca.leverable_axes non-empty", bool(lax), f"axes={lax}"))
    else:
        out.append(_ok("cca.leverable_axes", True,
                       f"n/a (SIMT/non-mesh target: no systolic dataflow axes by design); axes={lax}"))
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
    if not prof.has_mesh:                                    # SIMT/non-mesh: no systolic dataflow levers
        return [_ok("rtl_backend.derived_levers", True,
                    f"n/a (non-mesh target: no systolic dataflow levers by design) dim={prof.dim} levers={lev}")]
    return [_ok("rtl_backend.derived_levers non-empty", bool(lev),
                f"dim={prof.dim} has_mesh={prof.has_mesh} has_accumulator={prof.has_accumulator} levers={lev}")]


def _rtl_fact_checks(target: str) -> list[dict]:
    from merlin.targetgen import rtl_check_runner as RUN
    f = RUN.load_facts(target)
    facts = (f or {}).get("facts", f) or {}
    n = len(facts.get("interfaces") or []) + len(facts.get("arrays") or [])
    if not _is_mesh_target(target):                         # SIMT/non-mesh: no RoCC funct/mesh facts to derive
        return [_ok("rtl facts derivable", True,
                    f"n/a (non-mesh target: no RoCC funct/array facts by design) {n} interface/array facts")]
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
    endpoint_check = checks.get("kernel") or checks.get("trace")
    out.append(_ok("rtl checks compile (endpoint-appropriate)", bool(endpoint_check),
                   f"kernel={bool(checks.get('kernel'))} trace={bool(checks.get('trace'))} "
                   f"(capsule={cap_p.parent.name})"))
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


def submission_language_ok(submission_dir, arm: str) -> tuple[bool, str]:
    """ENFORCE the arm language contract on an EMITTED submission: a merlin arm (arm-3/arm-4) must build
    its dialect with the xDSL/Python kit — NOT a hand C++/TableGen backend. Returns (ok, reason). Reads
    the submission's manifest.yaml: for a merlin arm it fails if ``language: cpp`` or a build block that
    compiles a C++ tool (cmake / mlir-tblgen / a *-opt binary) is present. The C++ arms are exempt (that
    is their mandated method). Pure + target-agnostic — the grader/driver calls this to reject a
    non-compliant round with an actionable reason instead of grading a forbidden backend."""
    import yaml
    from pathlib import Path
    if "merlin_assisted" not in arm:               # only the merlin (xDSL) arms are constrained
        return True, "not a merlin arm (no xDSL mandate)"
    mpath = Path(submission_dir) / "mlir_oot" / "manifest.yaml"
    if not mpath.is_file():
        mpath = next(iter(sorted(Path(submission_dir).rglob("manifest.yaml"))), None)
    if not mpath or not mpath.is_file():
        return False, "no manifest.yaml in submission"
    m = yaml.safe_load(mpath.read_text()) or {}
    lang = str(m.get("language", "")).strip().lower()
    if lang in ("cpp", "c++", "cxx"):
        return False, f"merlin arm must use xDSL/Python, manifest declares language: {lang}"
    build = m.get("build") or {}
    blob = " ".join(str(v) for v in (build.values() if isinstance(build, dict) else [build])).lower()
    for marker in ("cmake", "mlir-tblgen", "tblgen", "clang++", "g++"):
        if marker in blob:
            return False, f"merlin arm must use xDSL/Python, build block invokes {marker!r} (C++ toolchain)"
    return True, f"xDSL/Python (language={lang or 'python'})"


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
