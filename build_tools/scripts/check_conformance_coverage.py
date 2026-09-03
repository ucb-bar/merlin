#!/usr/bin/env python3
"""Gate: does a target's capsule corpus cover the coverage requirement DERIVED for it?

The requirement comes from :mod:`merlin.targetgen.conformance` — the intersection of what the target's
capability manifest admits with what the real target-models' captures contain, expressed in
``cert_capsule_cover``'s ``(semantic_family, dtype, tile_alignment)`` cells. This script reports which
required cells no capsule exercises.

An uncovered cell is NOT a failing capsule. It means the corpus contains no capsule that would exercise
that family/dtype/alignment at all, so a submission's silence there is unmeasured rather than correct —
which is the failure mode a pass-rate cannot express. Radiance measured 21 of 56 covered when this was
first run, while its headline scorecard read 36/39.

Modes, mirroring the other gates in this directory:

  --target NAME        audit one target (repeatable); default: every target with a conformance spec
  --spec PATH          compare against a tracked spec instead of re-deriving (drift check)
  --write PATH         regenerate the spec (this is how the tracked spec is produced)
  --json               machine-readable
  --ratchet PATH       pre-existing debt that MAY ONLY SHRINK; unlisted new gaps fail
  --fail-on-uncovered  exit non-zero when any non-ratcheted cell is uncovered (default: report only)

Three axes are measured. The ``(semantic_family, dtype, tile_alignment)`` cells say WHAT the corpus
computes; the COMPOSITION axis (:mod:`merlin.targetgen.boundary`) says how the work is assembled --
``A``, ``A->A``, ``H->A->H``, ``A->H->A``, ``routing``, ``H``. They are reported side by side and never
crossed: a cross product would demand cells no real model presents. Composition debt is ratcheted under a
``composition:`` prefix so the axes cannot collide in one flat ratchet file. The MEMORY-MAPPING axis
(:mod:`merlin.targetgen.memory_regime`) says which regime the program puts the target's on-chip operand
store in -- ``fits_double`` / ``fits_single`` / ``fits_on_reuse`` / ``spills`` -- because a corpus whose
capsules all fit the store many times over cannot detect a memory-mapping failure at all.

Reporting-only by default because the derivation is new and the corpus predates it: turning a 35-cell gap
into a hard failure on day one would only teach everyone to pass `--no-verify`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2] / "merlin" / "python",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from merlin.common.paths import artifacts_dir, repo_root  # noqa: E402
from merlin.targetgen import conformance as CF  # noqa: E402


def _captures(model_root: Path | None = None, *, include_claim_models: bool = False) -> dict[str, Path]:
    """The captured model bundles that may DERIVE the requirement.

    The bundle store is the one place a capture is guaranteed to be the SAME IR the grader compiles
    (``_ensure_bundle`` writes it), so deriving from it cannot drift from what is actually graded.

    ⚠️ THE CLAIM MODELS ARE HELD OUT. Both callers feed this to ``conformance.spec``, i.e. to
    requirement derivation, and the requirement decides what the synthesized corpus contains. Coverage
    is then reported over captured models -- so a capture doing both jobs means the corpus was built
    from the model it is said to generalize to. This used to return EVERY bundle, the four claim models
    included, and lstmnetvit was already in both roles. The split is declared in
    ``merlin/contract/claim_models.yaml`` and applied by ``merlin.targetgen.claim_models``.

    Matching runs on the bundle's RAW directory name, before the label is prettified: the matcher works
    on token boundaries and the prettified label erases the tokens it needs.

    ``include_claim_models=True`` returns the unfiltered set. It exists for the disjointness gate, which
    has to derive the requirement BOTH ways to check that holding the claim models out costs no cell --
    never for producing a requirement.
    """
    from merlin.targetgen import claim_models as CM

    root = model_root or (artifacts_dir() / "recaptures")
    if not root.is_dir():
        return {}
    out = {}
    for d in sorted(root.iterdir()):
        m = d / "model.mlir"
        if not m.is_file():
            continue
        if not include_claim_models and CM.is_claim_bundle(d.name):
            continue
        out[d.name.replace("_fp32_consistent", "").replace("_consistent", "")] = m
    return out


def _target_experiment(target: str) -> Path | None:
    """The descriptor for ``target``, found by DIRECTORY NAME or by the name it DECLARES.

    The directory-only lookup is why `saturn_opu_mxv256d128` and `saturn_opu_mxv256d128_rvv` reported
    `no_target_experiment` and this gate then exited 0 for both: their descriptors live in directories
    named `saturn_opu`/`saturn_opu_rvv` and declare the configuration-qualified name in their own
    ``target:``. Asked about the name everything else resolves by, the gate found nothing and called it
    clean. Directory first (cheap, and the common case), then the declared name.
    """
    root = repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets"
    cand = root / target / "target_experiment.yaml"
    if cand.is_file():
        return cand
    if not root.is_dir():
        return None
    import yaml
    for desc in sorted(root.glob("*/target_experiment.yaml")):
        try:
            doc = yaml.safe_load(desc.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if str(doc.get("target") or "") == target:
            return desc
    return None


def _contract_target(target: str) -> str:
    """The target name its descriptor DECLARES, falling back to the directory name.

    A descriptor sits in a directory that need not match the target its contract is registered under
    (a configuration-qualified name beside a short directory). Resolving a contract by the directory
    name then finds nothing, which is why some targets had no derived requirement at all.
    """
    from merlin.targetgen.target_experiment import load_target_experiment

    desc = _target_experiment(target)
    if desc is None:
        return target
    try:
        return str(getattr(load_target_experiment(desc), "target", "") or target)
    except Exception:                              # noqa: BLE001 -- unreadable descriptor: use the dir
        return target


def audit(target: str, *, spec_path: Path | None = None) -> dict:
    """Derive (or load) the requirement and measure the corpus against it."""
    from merlin.targetgen.target_experiment import load_target_experiment

    desc = _target_experiment(target)
    if desc is None:
        return {"target": target, "status": "no_target_experiment",
                "detail": f"no target_experiment.yaml for {target!r}"}
    te = load_target_experiment(desc)
    # THE DIRECTORY NAME IS NOT ALWAYS THE TARGET NAME. A descriptor declares the target its contract is
    # registered under, and for some targets that differs from the directory the descriptor sits in
    # (a configuration-qualified name beside a short directory). Deriving against the directory name
    # then fails to resolve any contract at all, which is why those targets had no requirement -- not
    # because none could be derived, but because nobody was asking about the right name.
    contract_target = _contract_target(target)
    roots = list(te.graded_roots())
    exclude = set(getattr(te, "graded_exclude", ()) or ())

    caps = _captures()
    if spec_path and spec_path.is_file():
        import yaml
        doc = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        origin = f"tracked spec {spec_path}"
    else:
        doc = CF.spec(contract_target, caps)
        origin = ("derived now" if contract_target == target
                  else f"derived now against contract target {contract_target!r}")
    if not doc.get("cells"):
        return {"target": target, "status": "no_requirement", "spec_origin": origin,
                "detail": ("nothing was derived: no capability manifest resolved, or no captured model "
                           "was readable. This is 'we do not know', never 'nothing is required'."),
                "captures_available": sorted(caps)}

    tile = (doc.get("boundaries") or {}).get("tile_edge")
    gap = CF.uncovered(doc, roots, labels={"public", "dev"}, tile_dim=tile, exclude=exclude)
    by_cell = {c["cell"]: c for c in doc["cells"]}
    return {
        "target": target,
        "status": "ok",
        "spec_origin": origin,
        "graded_roots": [str(Path(r).name) for r in roots],
        "graded_exclude": sorted(exclude),
        "captures_used": (doc.get("diagnostics") or {}).get("captures_read", sorted(caps)),
        "tile_edge": tile,
        "n_required": gap["n_required"],
        "n_covered": gap["n_covered"],
        "uncovered": [{"cell": c, "basis": by_cell.get(c, {}).get("basis"),
                       "admitted_by": by_cell.get(c, {}).get("admitted_by", [])}
                      for c in gap["uncovered"]],
        "corpus_cells_not_required": gap["extra_cells"],
        "composition": gap.get("composition") or {"status": "not_measured"},
        "memory_mapping": gap.get("memory_mapping") or {"status": "not_measured"},
        "host_only": gap.get("host_only") or {"status": "not_measured"},
        "diagnostics": doc.get("diagnostics") or {},
    }


def _debt(target: str, item: str, axis: str = "cell") -> str:
    """A ratchet entry, SCOPED TO ITS TARGET.

    A bare cell name would let one target's accepted debt silently excuse another's: `contraction/f32/
    aligned` is a real gap on more than one target here, and a flat entry forgives every one of them at
    once. The axis tag keeps a composition shape and a coverage cell from ever colliding in one file.
    """
    return f"{target} {axis}:{item}"


def _load_ratchet(p: Path | None) -> set[str]:
    if not p or not p.is_file():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--spec", type=Path, default=None)
    ap.add_argument("--write", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ratchet", type=Path, default=None)
    ap.add_argument("--fail-on-uncovered", action="store_true")
    ap.add_argument("--fail-on-unverifiable", action="store_true",
                    help="exit 2 when a target could not be audited at all")
    a = ap.parse_args(argv)

    # DEFAULT TARGET SET IS DISCOVERED, not named: every target that already has a tracked conformance
    # spec. A hardcoded default here would make this gate silently about one target forever, which is the
    # overfitting the whole module exists to prevent (and the no-target-name gate rightly rejects it).
    targets = a.target or sorted(
        p.stem for p in (repo_root() / "merlin" / "contract" / "capsules" / "conformance").glob("*.yaml"))
    if not targets:
        print("no --target given and no tracked conformance spec found under "
              "merlin/contract/capsules/conformance/; pass --target NAME (with --write to create one)",
              file=sys.stderr)
        return 2

    if a.write:
        import yaml
        if len(targets) != 1:
            print("--write takes exactly one --target", file=sys.stderr)
            return 2
        # Same resolution as `audit`: derive against the target its DESCRIPTOR declares, which is not
        # always the directory the descriptor sits in. Two call sites resolved this independently and
        # only one of them was right, so a target whose names differ produced a requirement from
        # `audit` and a crash from `--write` -- the path that actually creates the file.
        doc = CF.spec(_contract_target(targets[0]), _captures())
        a.write.parent.mkdir(parents=True, exist_ok=True)
        a.write.write_text(
            "# DERIVED — regenerate with:\n"
            f"#   build_tools/scripts/check_conformance_coverage.py --target {targets[0]} "
            f"--write {a.write.relative_to(repo_root()) if a.write.is_absolute() else a.write}\n"
            "# Do not hand-edit: the point of this file is that it is evidence, not authorship.\n"
            + yaml.safe_dump(doc, sort_keys=False, width=100), encoding="utf-8")
        print(f"wrote {a.write} — {len(doc['cells'])} required cell(s)")
        return 0

    ratchet = _load_ratchet(a.ratchet)
    reports = [audit(t, spec_path=a.spec) for t in targets]
    if a.json:
        print(json.dumps(reports, indent=2))
    else:
        for r in reports:
            if r["status"] != "ok":
                print(f"== {r['target']}: {r['status']} — {r.get('detail', '')}")
                continue
            print(f"== {r['target']}  ({r['spec_origin']})")
            print(f"   captures used : {r['captures_used']}")
            print(f"   tile edge     : {r['tile_edge']}")
            print(f"   covered       : {r['n_covered']} / {r['n_required']} required cell(s)")
            new = [u for u in r["uncovered"] if _debt(r["target"], u["cell"]) not in ratchet]
            if r["uncovered"]:
                print(f"   UNCOVERED     : {len(r['uncovered'])}"
                      + (f" ({len(new)} not in the ratchet)" if ratchet else ""))
                for u in r["uncovered"]:
                    mark = " " if _debt(r["target"], u["cell"]) in ratchet else "*"
                    print(f"     {mark} {u['cell']:34s} basis={u['basis']} by={u['admitted_by']}")
            comp = r.get("composition") or {}
            if comp.get("status") == "ok":
                print(f"   composition   : {comp['n_covered']} / {comp['n_required']} required shape(s)")
                for kind in comp["uncovered"]:
                    mark = " " if _debt(r["target"], kind, "composition") in ratchet else "*"
                    print(f"     {mark} {kind:34s} no capsule assembles work this way")
                thin = [(k, v) for k, v in sorted((comp.get("covered_by") or {}).items()) if len(v) == 1]
                for kind, names in thin:
                    print(f"       {kind:32s} covered by ONE capsule ({names[0]}) — a single point of "
                          f"evidence for a whole composition shape")
                for kind in (comp.get("covered_only_incidentally") or []):
                    print(f"       {kind:32s} covered only INCIDENTALLY — every capsule containing it "
                          f"is named for a different shape, so nothing is built to prove it")
                if comp.get("unreadable_capsules"):
                    # TWO DIFFERENT FACTS, and they license different actions. "We could not read this
                    # capsule" is a defect in the capsule -- fix the capsule. "This target's seam cannot
                    # be emitted by any path here" is a defect in the toolchain -- build the transport,
                    # or stop claiming the shape. Printing both under one word sent readers to the wrong
                    # one.
                    und = {n: w for n, w in comp["unreadable_capsules"].items() if "undeterminable" in w}
                    bad = {n: w for n, w in comp["unreadable_capsules"].items() if n not in und}
                    if bad:
                        print(f"   UNREADABLE    : {len(bad)} capsule(s) whose composition could not "
                              f"be determined")
                        for name, why in sorted(bad.items()):
                            print(f"     ? {name:32s} {why}")
                    if und:
                        print(f"   UNBUILDABLE   : {len(und)} capsule(s) are accelerator-eligible on a "
                              f"target whose host/device seam no path in this repo can emit, so their "
                              f"composition is UNDETERMINABLE -- not covered, and not a capsule defect")
                        for name, why in sorted(und.items()):
                            print(f"     ? {name:32s} {why}")
            elif comp:
                print(f"   composition   : {comp.get('status')} — {comp.get('detail', '')}")
            mem = r.get("memory_mapping") or {}
            if mem.get("status") == "ok":
                print(f"   memory regime : {mem['n_covered']} / {mem['n_required']} required regime(s)"
                      f"   (operand store {mem.get('capacity_rows')} rows)")
                counts = mem.get("region_counts") or {}
                total = sum(counts.values()) or 1
                for kind in mem["uncovered"]:
                    mark = " " if _debt(r["target"], kind, "memory") in ratchet else "*"
                    n = counts.get(kind, 0)
                    print(f"     {mark} {kind:34s} no capsule reaches it; {n} real region(s) "
                          f"({100.0 * n / total:.1f}% of what the captures contain) do")
                lw = mem.get("largest_working_set") or {}
                if lw.get("name"):
                    print(f"       largest capsule working set: {lw['name']} at "
                          f"{100.0 * float(lw.get('fraction_of_capacity') or 0):.2f}% of capacity")
            elif mem:
                print(f"   memory regime : {mem.get('status')} — {mem.get('detail', '')}")
            ho = r.get("host_only") or {}
            if ho.get("status") == "ok":
                print(f"   host-only lane: {ho['n_covered']} / {ho['n_required']} family/families the "
                      f"hardware must NOT accelerate")
                for fam in ho["uncovered"]:
                    mark = " " if _debt(r["target"], fam, "host_only") in ratchet else "*"
                    print(f"     {mark} {fam:34s} no capsule proves it lands on the host lane")
                for fam, names in sorted((ho.get("covered_by") or {}).items()):
                    print(f"       {fam:32s} proven host-only by {names}")
            elif ho.get("status") == "undeterminable":
                print(f"   host-only lane: UNDETERMINABLE — {ho.get('detail', '')}")
            elif ho:
                print(f"   host-only lane: {ho.get('status')} — {ho.get('detail', '')}")
            if r["corpus_cells_not_required"]:
                print(f"   corpus cells not in the requirement: {r['corpus_cells_not_required']}")
                print("     (a cell the hardware does not admit for that family — e.g. an int8 movement "
                      "capsule on a target whose movement datapath is float-only — is INTENTIONAL: it is "
                      "what forces the compiler off the accelerator path)")
            for n in (r["diagnostics"].get("notes") or []):
                print(f"   note: {n}")

    bad = [_debt(r["target"], u["cell"]) for r in reports if r["status"] == "ok"
           for u in r["uncovered"] if _debt(r["target"], u["cell"]) not in ratchet]
    # Composition gaps carry a `composition` axis tag so a shape and a cell can never collide in one flat
    # file, and so a reader of the ratchet can see which axis each debt belongs to.
    bad += [_debt(r["target"], k, "composition") for r in reports if r["status"] == "ok"
            for k in ((r.get("composition") or {}).get("uncovered") or [])
            if _debt(r["target"], k, "composition") not in ratchet]
    bad += [_debt(r["target"], k, "memory") for r in reports if r["status"] == "ok"
            for k in ((r.get("memory_mapping") or {}).get("uncovered") or [])
            if _debt(r["target"], k, "memory") not in ratchet]
    # The negative lane carries its own axis tag for the same reason the others do: a family name and a
    # composition shape must never collide in one flat ratchet file.
    bad += [_debt(r["target"], k, "host_only") for r in reports if r["status"] == "ok"
            for k in ((r.get("host_only") or {}).get("uncovered") or [])
            if _debt(r["target"], k, "host_only") not in ratchet]
    # ⚠️ A TARGET THAT COULD NOT BE AUDITED HAS ESTABLISHED NOTHING. Every `bad` list above filters on
    # `status == "ok"`, so a target whose descriptor, contract or corpus could not be resolved
    # contributes no debt and the gate returns 0 -- reporting success for a question it never asked.
    # Measured: `saturn_opu_mxv256d128` and `..._rvv` reported `no_target_experiment` and this gate
    # exited 0 for BOTH; with the descriptor found by its declared name they owe 5 uncovered items.
    # This repo has now paid for that shape five times, so it is spelled 2 ("cannot decide"), never 0.
    unrunnable = [r for r in reports if r["status"] != "ok"]
    if unrunnable:
        print(f"\n  COULD NOT AUDIT ({len(unrunnable)}) — these establish NOTHING, they are not clean:",
              file=sys.stderr)
        for r in unrunnable:
            print(f"    ? {r['target']:28s} {r['status']}: {r.get('detail', '')}", file=sys.stderr)

    if bad and a.fail_on_uncovered:
        print(f"\nFAIL: {len(bad)} required cell(s) uncovered and not ratcheted", file=sys.stderr)
        return 1
    if unrunnable and (a.fail_on_uncovered or a.fail_on_unverifiable):
        print(f"\nCANNOT DECIDE: {len(unrunnable)} target(s) could not be audited at all.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
