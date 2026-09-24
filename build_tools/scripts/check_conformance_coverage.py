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
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2] / "merlin" / "python",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from merlin.common.paths import artifacts_dir, repo_root  # noqa: E402
from merlin.targetgen import conformance as CF  # noqa: E402


class ClaimModelInApplications(Exception):
    """A target named a CLAIM model as one of its applications.

    Raised rather than filtered. Silently dropping the entry would leave a target declaring five
    applications and deriving from four, with the difference visible nowhere -- and the thing being
    dropped is precisely the circularity `claim_models.yaml` exists to prevent, so it must be loud.
    """


def _applications(te) -> dict:
    """The target's DECLARED applications, as ``{label: model.mlir}``.

    A target says which exported models it is FOR via ``workload_spec.applications``. Unlike
    ``_captures`` below, which globs the whole recapture store and is therefore the same evidence for
    every target, this is per-target BY CONSTRUCTION: it is the set the target's own descriptor names.

    TWO SPELLINGS, because they answer different questions. A **list** names bundles in the recapture
    store, and is what a target should normally declare: the descriptor states which models it is for,
    the statement is reviewable in the diff, and it does not depend on a generated directory happening
    to contain the right subdirectories. A **string** is a directory of ingested bundles, which is what
    ``ingest_applications.py`` produces and is kept working for it.

    ⚠️ A CLAIM MODEL MAY NEVER APPEAR HERE. The application axis is a derivation source -- it decides
    which capsules exist -- so admitting a claim model would build the corpus from the model it is then
    said to generalize to, which is the exact circularity `claim_models.yaml` forbids. Checked here,
    where the set is named, rather than downstream where the bundles have already become shapes and the
    model they came from is no longer visible.
    """
    from merlin.common.artifacts import recaptures_dir
    from merlin.common.paths import repo_root
    from merlin.targetgen import claim_models as CM

    spec = dict(getattr(te, "workload_spec", None) or {})
    declared = spec.get("applications")
    if not declared:
        return {}

    if isinstance(declared, (list, tuple)):
        store = Path(recaptures_dir())
        found, missing = {}, []
        for name in declared:
            d = store / str(name)
            if (d / "model.mlir").is_file():
                found[d.name] = d / "model.mlir"
            else:
                missing.append(str(name))
        if missing:
            raise FileNotFoundError(
                f"declared application capture(s) missing model.mlir in the recapture store: {missing}; "
                "fresh Phase 0 derivation cannot use a smaller application set than the descriptor declares"
            )
    else:
        root = Path(declared)
        if not root.is_absolute():
            root = repo_root() / root
        if not root.is_dir():
            raise FileNotFoundError(f"declared application capture directory is missing: {root}")
        found = {d.name: d / "model.mlir" for d in sorted(root.iterdir()) if (d / "model.mlir").is_file()}
        if not found:
            raise FileNotFoundError(f"declared application capture directory has no model.mlir bundles: {root}")

    offending = sorted(
        n for n, path in found.items() if CM.is_claim_bundle(n) or CM.is_claim_bundle(path.resolve().parent.name)
    )
    if offending:
        raise ClaimModelInApplications(
            f"{offending} named as application(s), but they are CLAIM models. The application axis "
            f"decides which capsules exist, so reading a claim model here would build the corpus from "
            f"the model it is then said to generalize to. Claim models: {list(CM.claim_models())}"
        )
    return found


def _certifying_members(te) -> int:
    """How many graded capsules DEMAND a cycle-accurate tier, i.e. how many the budget is split over.

    Counted from the corpus rather than from the budget, so the derivation below is not circular: a
    member demands certification because of what its own acceptance block declares, which no budget
    input can change.
    """
    import yaml

    deepest = 0
    for root in list(te.graded_roots() or ()):
        for p in sorted(Path(root).glob("*/capsule.yaml")):
            try:
                cap = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            tiers = [str(t) for t in (cap.get("required_oracle_tiers") or ())]
            cap_to = str(cap.get("max_oracle_tier") or "")
            if cap_to and cap_to in tiers:
                tiers = tiers[: tiers.index(cap_to) + 1]
            # A cycle-accurate rung is the one a certification budget pays for. Read as "deeper than the
            # loop tier" from the capsule's own declared ladder rather than by naming a rung, so a
            # target whose ladder is spelled differently is counted the same way.
            if len(tiers) > 3:
                deepest += 1
    return deepest


def _cert_budget_s(te) -> float | None:
    """The seconds a single cycle-accurate certification may take, DERIVED where possible.

    A POLICY number rather than a hardware fact -- how much simulator time this experiment will
    spend -- but for a long time it was an unsourced one: `300.0`, written in two places and traceable
    to nobody. It is the single input that sets `cert_affordability.max_elements`, which decides how
    large a capsule may be, which decides that the whole performance corpus sits in a residency regime
    almost no real work occupies. A number with that much downstream reach should be a consequence of
    something, not a constant.

    So a target may declare the quantity it actually controls -- the total certification wall-clock it
    will spend on one grade -- and the per-capsule figure follows from dividing it by the members that
    demand certification. That makes the budget a consequence of corpus size and round cadence, both
    deliberate choices, and it moves when either does.

    A directly declared `cert_budget_s` still wins, and is reported as declared: an explicit number is
    a stronger statement than a derivation, and quietly overriding it would make the field decorative.
    """
    spec = dict(getattr(te, "workload_spec", None) or {})
    value = spec.get("cert_budget_s")
    try:
        if value:
            return float(value)
    except (TypeError, ValueError):
        return None
    total = spec.get("cert_wall_clock_budget_s")
    try:
        total = float(total) if total else None
    except (TypeError, ValueError):
        total = None
    if not total:
        return None
    members = _certifying_members(te)
    # FAIL CLOSED. No certifying member means the division has no denominator, and inventing one would
    # hand every capsule the whole grade's budget -- the most permissive possible ceiling, from the
    # least evidence.
    return (total / members) if members else None


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
    except Exception:  # noqa: BLE001 -- unreadable descriptor: use the dir
        return target


def audit(target: str, *, spec_path: Path | None = None) -> dict:
    """Derive (or load) the requirement and measure the corpus against it."""
    from merlin.targetgen.target_experiment import load_target_experiment

    desc = _target_experiment(target)
    if desc is None:
        return {
            "target": target,
            "status": "no_target_experiment",
            "detail": f"no target_experiment.yaml for {target!r}",
        }
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
        app_demands = doc.get("application_demands") or {}
        sidecar = app_demands.get("sidecar")
        if sidecar:
            from merlin_experiments.phase0.profiles import application_inventory_path

            try:
                inventory_path = application_inventory_path(spec_path)
                if inventory_path is None:
                    raise ValueError("declared application sidecar is absent")
                full = json.loads(inventory_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                return {"target": target, "status": "unverifiable_application_inventory", "detail": str(exc)}
            digest = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if digest != app_demands.get("full_inventory_sha256"):
                return {
                    "target": target,
                    "status": "unverifiable_application_inventory",
                    "detail": f"generated application sidecar digest mismatch: {inventory_path}",
                }
    else:
        # The corpus is handed in as a SECOND evidence source for the negative lane: it is what the
        # grader will run, and it can present work the captures never do (a bf16 contraction on an
        # int8-only array). See `conformance.corpus_presented_pairs`.
        from merlin_experiments.corpus.admission import conformance_spec

        try:
            doc = conformance_spec(
                contract_target,
                caps,
                applications=_applications(te),
                corpus_roots=roots,
                cert_budget_s=_cert_budget_s(te),
            )
        except (FileNotFoundError, ValueError) as exc:
            return {
                "target": target,
                "status": "unverifiable_application_capture",
                "detail": str(exc),
            }
        origin = (
            "derived now" if contract_target == target else f"derived now against contract target {contract_target!r}"
        )
    if not doc.get("cells"):
        return {
            "target": target,
            "status": "no_requirement",
            "spec_origin": origin,
            "detail": (
                "nothing was derived: no capability manifest resolved, or no captured model "
                "was readable. This is 'we do not know', never 'nothing is required'."
            ),
            "captures_available": sorted(caps),
        }

    tile = (doc.get("boundaries") or {}).get("tile_edge")
    gap = CF.uncovered(doc, roots, labels={"public", "dev"}, tile_dim=tile, exclude=exclude)
    by_cell = {c["cell"]: c for c in doc["cells"]}
    return {
        "target": target,
        "status": "ok",
        "applications_declared": bool((dict(getattr(te, "workload_spec", None) or {})).get("applications")),
        "spec_origin": origin,
        "graded_roots": [str(Path(r).name) for r in roots],
        "graded_exclude": sorted(exclude),
        "captures_used": (doc.get("diagnostics") or {}).get("captures_read", sorted(caps)),
        "tile_edge": tile,
        "n_required": gap["n_required"],
        "n_covered": gap["n_covered"],
        "uncovered": [
            {
                "cell": c,
                "basis": by_cell.get(c, {}).get("basis"),
                "admitted_by": by_cell.get(c, {}).get("admitted_by", []),
            }
            for c in gap["uncovered"]
        ],
        "corpus_cells_not_required": gap["extra_cells"],
        "application_demands": gap.get("application_demands") or {"coverage_status": "unverified"},
        "composition": gap.get("composition") or {"status": "not_measured"},
        "memory_mapping": gap.get("memory_mapping") or {"status": "not_measured"},
        "shape_geometry": gap.get("shape_geometry") or {"status": "not_measured"},
        "host_only": gap.get("host_only") or {"status": "not_measured"},
        # THE THREE AXES THAT WERE DERIVED AND NEVER PRINTED. Measuring an axis and not reporting it
        # is only a little better than not measuring it: the number exists, nobody reads it, and the
        # gap stays invisible for exactly as long.
        "host_lane": gap.get("host_lane") or {"status": "not_measured"},
        "epilogue": gap.get("epilogue") or {"status": "not_measured"},
        "conv_geometry": gap.get("conv_geometry") or {"status": "not_measured"},
        # ...AND THE THREE THAT WERE NEVER CARRIED OUT OF HERE AT ALL. `uncovered()` computes a scope
        # gap (the adjacency chains real captures present), a group gap (stage combinations) and a
        # carried-state gap, and none of the three was a key of this dict -- so no printer could show
        # them and `uncovered_debt` could not reach them however it was written. Adding them here is
        # what makes their entries in `AXES` mean anything.
        "scope": gap.get("scope") or {"status": "not_measured"},
        "groups": gap.get("groups") or {"status": "not_measured"},
        "carried_state": gap.get("carried_state") or {"status": "not_measured"},
        # HOW MUCH CORPUS IS BUYING HOW MUCH REQUIREMENT. Nothing reported the ratio, so corpus growth
        # was invisible: a corpus can double without covering one more obligation, and the only signal
        # would have been the wall-clock of a grade. Certification cost is floor-dominated -- 56-68% of
        # a member's seconds are the per-member intercept on the engines measured here -- so the price
        # of an entry sweep is driven by member COUNT rather than member size, and this is the number
        # that tracks it. Reported, never gated: a low ratio is not a defect (an edge case the
        # requirement cannot express is a legitimate capsule), and gating on it would be satisfiable by
        # deleting capsules, which is the trap `check_semantic_coverage` already documents for ARR.
        "corpus_pressure": _corpus_pressure(gap, roots),
        # WHAT THE BUDGET COSTS WHEN YOU MULTIPLY IT OUT. The per-capsule figure is the one anybody
        # declares and the total is the one anybody would actually notice, and nothing reported the
        # total -- so a per-capsule number nobody sourced set a grade-long wall-clock nobody chose.
        # Reported beside the pressure ratio because they answer one question together: how much does
        # this corpus cost to stand behind.
        "certification": _certification_cost(te),
        "diagnostics": doc.get("diagnostics") or {},
    }


def _certification_cost(te) -> dict:
    """``{members, budget_s, implied_total_s, source}`` -- the budget multiplied out.

    ``source`` says which way round it was resolved: a directly declared per-capsule figure, or one
    derived from a declared total. Both are reported, so a reader can tell a number somebody chose from
    a number that fell out of one.
    """
    spec = dict(getattr(te, "workload_spec", None) or {})
    members = _certifying_members(te)
    budget = _cert_budget_s(te)
    source = (
        "declared per capsule"
        if spec.get("cert_budget_s")
        else ("derived from a declared grade total" if spec.get("cert_wall_clock_budget_s") else "inherited default")
    )
    return {
        "members": members,
        "budget_s": budget,
        "implied_total_s": round(members * budget, 1) if (budget and members) else None,
        "source": source,
    }


def _corpus_pressure(gap: dict, roots) -> dict:
    """``{obligations, capsules, capsules_per_obligation}`` -- how much corpus buys how much requirement.

    Obligations are summed across EVERY axis rather than counting cells alone, because a cell is one of
    eleven things the requirement asks for and the other ten are what the corpus is mostly spending
    itself on. Capsules are counted from the graded roots, the same trees the cover is measured over.

    Neither number is a verdict. The ratio is a reading: it says whether the corpus is growing faster
    than the requirement it exists to evidence, which is exactly the question nobody could answer while
    the count of either was unreported.
    """
    obligations = int(gap.get("n_required") or 0)
    for key, _tag in AXES:
        axis = gap.get(key)
        if isinstance(axis, dict):
            obligations += int(axis.get("n_required") or 0)
    capsules = 0
    for root in roots or ():
        capsules += sum(1 for _ in Path(root).glob("*/capsule.yaml"))
    return {
        "obligations": obligations,
        "capsules": capsules,
        "capsules_per_obligation": round(capsules / obligations, 2) if obligations else None,
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


#: Every axis besides cells that :func:`merlin.targetgen.conformance.uncovered` measures, as
#: ``(key in the report, tag in the ratchet)``.
#:
#: THE TABLE IS THE VERDICT. `uncovered_debt` iterates it rather than repeating a list comprehension
#: per axis, because the repetition is what the bug was made of: an axis gets measured, gets printed
#: with the same `*` un-ratcheted marker as the rest, and is never accumulated -- so the gate prints a
#: violation and returns 0. That happened to `shape_geometry` once, was fixed for that one axis, and
#: had happened again to `host_lane`, `epilogue` and `conv_geometry` by the time anyone looked; `scope`,
#: `groups` and `carried_state` were measured and never even carried out of `audit`. Adding an axis to
#: the requirement now means adding one line here, and `test_coverage_gates_can_fail` asserts this table
#: names every axis `uncovered()` can return, so forgetting is a test failure rather than silence.
#:
#: The tag is what a ratchet line is keyed on, so it is part of the file format and may not be renamed
#: without rewriting `conformance_ratchet.txt`. It differs from the report key where the shorter word
#: was already in the file (`memory_mapping` -> `memory`, `shape_geometry` -> `geometry`).
AXES: tuple[tuple[str, str], ...] = (
    ("composition", "composition"),
    ("memory_mapping", "memory"),
    ("host_only", "host_only"),
    ("shape_geometry", "geometry"),
    ("host_lane", "host_lane"),
    ("epilogue", "epilogue"),
    ("conv_geometry", "conv_geometry"),
    ("scope", "scope"),
    ("groups", "groups"),
    ("carried_state", "carried_state"),
)


def uncovered_debt(reports: list[dict], ratchet: set) -> list[str]:
    """Every un-ratcheted coverage gap, across EVERY measured axis, as ratchet keys.

    Pulled out of ``main()`` because axes silently went missing there -- see :data:`AXES` for the
    history. A single function over a single table is what lets a test assert that an uncovered gap on
    each axis reaches the verdict.

    Cells are the one axis handled separately: their ``uncovered`` entries are dicts carrying the basis
    and what admitted them, where every other axis reports a bare string.
    """
    bad = [
        _debt(r["target"], u["cell"])
        for r in reports
        if r["status"] == "ok"
        for u in r["uncovered"]
        if _debt(r["target"], u["cell"]) not in ratchet
    ]
    for key, tag in AXES:
        bad += [
            _debt(r["target"], k, tag)
            for r in reports
            if r["status"] == "ok"
            for k in ((r.get(key) or {}).get("uncovered") or [])
            if _debt(r["target"], k, tag) not in ratchet
        ]
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--spec", type=Path, default=None)
    ap.add_argument("--write", type=Path, default=None)
    ap.add_argument(
        "--inventory-out",
        type=Path,
        default=None,
        help="write an exact diagnostic application inventory even when derivation is incomplete",
    )
    ap.add_argument(
        "--application-capture",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="explicit capture for --inventory-out only; repeatable, diagnostic and never a conformance spec",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ratchet", type=Path, default=None)
    ap.add_argument("--fail-on-uncovered", action="store_true")
    ap.add_argument(
        "--fail-on-unverifiable", action="store_true", help="exit 2 when a target could not be audited at all"
    )
    a = ap.parse_args(argv)

    # DEFAULT TARGET SET IS DISCOVERED, not named: every target that already has a tracked conformance
    # spec. A hardcoded default here would make this gate silently about one target forever, which is the
    # overfitting the whole module exists to prevent (and the no-target-name gate rightly rejects it).
    from merlin.targetgen.corpora import conformance_reference_dir

    targets = a.target or sorted(p.stem for p in conformance_reference_dir().glob("*.yaml"))
    if not targets:
        print(
            "no --target given and no tracked conformance spec found under "
            "experiments/reference-data/phase0/conformance/; pass --target NAME "
            "(with --write to create a reviewed reference)",
            file=sys.stderr,
        )
        return 2

    if a.inventory_out:
        if len(targets) != 1 or a.write or a.spec:
            print(
                "--inventory-out takes exactly one --target and cannot be combined with --write or --spec",
                file=sys.stderr,
            )
            return 2
        from merlin.targetgen.application_inventory import application_demand_inventory
        from merlin.targetgen.corpora import descriptor_path
        from merlin.targetgen.target_experiment import load_target_experiment

        try:
            if a.application_capture:
                paths = {}
                for item in a.application_capture:
                    label, separator, path = item.partition("=")
                    if not separator or not label or not path or label in paths:
                        raise ValueError(f"invalid or duplicate --application-capture {item!r}; expected LABEL=PATH")
                    paths[label] = Path(path)
            else:
                paths = _applications(load_target_experiment(descriptor_path(targets[0])))
            if not paths:
                raise ValueError("no declared application captures to inventory")
            full = application_demand_inventory(paths, _contract_target(targets[0]), detailed=True)
        except (FileNotFoundError, ValueError) as exc:
            print(f"cannot inventory declared applications: {exc}", file=sys.stderr)
            return 2
        a.inventory_out.parent.mkdir(parents=True, exist_ok=True)
        a.inventory_out.write_text(json.dumps(full, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"wrote diagnostic inventory {a.inventory_out}: {full['n_operations']} parsed op(s), "
            f"status={full['status']} (not a conformance spec)"
        )
        return 2 if full["status"] != "inventoried" else 0
    if a.application_capture:
        print("--application-capture requires --inventory-out", file=sys.stderr)
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
        # Imported here rather than at module scope, matching `_contract_target`'s own local import:
        # this script runs before merlin is necessarily importable in every environment it is used in.
        from merlin.targetgen.corpora import descriptor_path
        from merlin.targetgen.target_experiment import load_target_experiment

        _te = load_target_experiment(descriptor_path(targets[0]))
        # SAME EVIDENCE AS `audit`, corpus included. `--write` is the path that creates the tracked
        # requirement, and omitting the corpus here produced a spec that named fewer negative-lane
        # pairs than the one `audit` derives from the very same tree -- the two disagreeing about the
        # requirement is worse than either being wrong.
        from merlin_experiments.corpus.admission import conformance_spec

        try:
            _app_paths = _applications(_te)
            doc = conformance_spec(
                _contract_target(targets[0]),
                _captures(),
                applications=_app_paths,
                corpus_roots=list(_te.graded_roots()),
                cert_budget_s=_cert_budget_s(_te),
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"cannot write a derived requirement: {exc}", file=sys.stderr)
            return 2
        if (doc.get("application_demands") or {}).get("status") == "incomplete":
            print("cannot write a derived requirement: declared application inventory is incomplete", file=sys.stderr)
            return 2
        sidecar_path = None
        full_inventory = None
        if _app_paths:
            from merlin.targetgen.application_inventory import application_demand_inventory

            full_inventory = application_demand_inventory(_app_paths, _contract_target(targets[0]), detailed=True)
            digest = hashlib.sha256(
                json.dumps(full_inventory, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if digest != doc["application_demands"]["full_inventory_sha256"]:
                print(
                    "cannot write a derived requirement: application inventory changed during derivation",
                    file=sys.stderr,
                )
                return 2
            sidecar_path = a.write.with_name(f"{a.write.stem}.application-demands.json")
            doc["application_demands"]["sidecar"] = sidecar_path.name
        a.write.parent.mkdir(parents=True, exist_ok=True)
        if sidecar_path is not None:
            sidecar_path.write_text(json.dumps(full_inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_label = str(a.write)
        try:
            write_label = str(a.write.resolve().relative_to(repo_root()))
        except ValueError:
            pass
        a.write.write_text(
            "# DERIVED — regenerate with:\n"
            f"#   build_tools/scripts/check_conformance_coverage.py --target {targets[0]} "
            f"--write {write_label}\n"
            "# Do not hand-edit: the point of this file is that it is evidence, not authorship.\n"
            + yaml.safe_dump(doc, sort_keys=False, width=100),
            encoding="utf-8",
        )
        print(f"wrote {a.write} — {len(doc['cells'])} required cell(s)")
        if sidecar_path is not None:
            print(f"wrote generated application inventory {sidecar_path}")
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
            app = r.get("application_demands") or {}
            if r.get("applications_declared"):
                print(
                    f"   applications  : {app.get('status', 'not_measured')} inventory, "
                    f"{app.get('n_operations', 0)} parsed op(s); "
                    f"operation/capsule coverage {app.get('coverage_status', 'unverified')}"
                )
            new = [u for u in r["uncovered"] if _debt(r["target"], u["cell"]) not in ratchet]
            if r["uncovered"]:
                print(
                    f"   UNCOVERED     : {len(r['uncovered'])}"
                    + (f" ({len(new)} not in the ratchet)" if ratchet else "")
                )
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
                    print(
                        f"       {kind:32s} covered by ONE capsule ({names[0]}) — a single point of "
                        f"evidence for a whole composition shape"
                    )
                for kind in comp.get("covered_only_incidentally") or []:
                    print(
                        f"       {kind:32s} covered only INCIDENTALLY — every capsule containing it "
                        f"is named for a different shape, so nothing is built to prove it"
                    )
                if comp.get("unreadable_capsules"):
                    # TWO DIFFERENT FACTS, and they license different actions. "We could not read this
                    # capsule" is a defect in the capsule -- fix the capsule. "This target's seam cannot
                    # be emitted by any path here" is a defect in the toolchain -- build the transport,
                    # or stop claiming the shape. Printing both under one word sent readers to the wrong
                    # one.
                    und = {n: w for n, w in comp["unreadable_capsules"].items() if "undeterminable" in w}
                    bad = {n: w for n, w in comp["unreadable_capsules"].items() if n not in und}
                    if bad:
                        print(f"   UNREADABLE    : {len(bad)} capsule(s) whose composition could not be determined")
                        for name, why in sorted(bad.items()):
                            print(f"     ? {name:32s} {why}")
                    if und:
                        print(
                            f"   UNBUILDABLE   : {len(und)} capsule(s) are accelerator-eligible on a "
                            f"target whose host/device seam no path in this repo can emit, so their "
                            f"composition is UNDETERMINABLE -- not covered, and not a capsule defect"
                        )
                        for name, why in sorted(und.items()):
                            print(f"     ? {name:32s} {why}")
            elif comp:
                print(f"   composition   : {comp.get('status')} — {comp.get('detail', '')}")
            geo = r.get("shape_geometry") or {}
            if geo.get("status") == "ok":
                print(
                    f"   shape geometry: {geo['n_covered']} / {geo['n_required']} aspect-ratio "
                    f"class(es) real models present"
                )
                for kind in geo["uncovered"]:
                    mark = " " if _debt(r["target"], kind, "geometry") in ratchet else "*"
                    print(f"     {mark} {kind:34s} no capsule reproduces this aspect ratio")
                if geo.get("mac_fraction_uncovered"):
                    print(
                        f"       {100.0 * float(geo['mac_fraction_uncovered']):.1f}% of real "
                        f"contraction MAC work sits in an untested aspect ratio"
                    )
            elif geo:
                print(f"   shape geometry: {geo.get('status')} — {geo.get('detail', '')}")
            mem = r.get("memory_mapping") or {}
            if mem.get("status") == "ok":
                print(
                    f"   memory regime : {mem['n_covered']} / {mem['n_required']} required regime(s)"
                    f"   (operand store {mem.get('capacity_rows')} rows)"
                )
                counts = mem.get("region_counts") or {}
                total = sum(counts.values()) or 1
                for kind in mem["uncovered"]:
                    mark = " " if _debt(r["target"], kind, "memory") in ratchet else "*"
                    n = counts.get(kind, 0)
                    print(
                        f"     {mark} {kind:34s} no capsule reaches it; {n} real region(s) "
                        f"({100.0 * n / total:.1f}% of what the captures contain) do"
                    )
                lw = mem.get("largest_working_set") or {}
                if lw.get("name"):
                    print(
                        f"       largest capsule working set: {lw['name']} at "
                        f"{100.0 * float(lw.get('fraction_of_capacity') or 0):.2f}% of capacity"
                    )
            elif mem:
                print(f"   memory regime : {mem.get('status')} — {mem.get('detail', '')}")
            ho = r.get("host_only") or {}
            if ho.get("status") == "ok":
                print(
                    f"   host-only lane: {ho['n_covered']} / {ho['n_required']} family/families the "
                    f"hardware must NOT accelerate"
                )
                for fam in ho["uncovered"]:
                    mark = " " if _debt(r["target"], fam, "host_only") in ratchet else "*"
                    print(f"     {mark} {fam:34s} no capsule proves it lands on the host lane")
                for fam, names in sorted((ho.get("covered_by") or {}).items()):
                    print(f"       {fam:32s} proven host-only by {names}")
            elif ho.get("status") == "undeterminable":
                print(f"   host-only lane: UNDETERMINABLE — {ho.get('detail', '')}")
            elif ho:
                print(f"   host-only lane: {ho.get('status')} — {ho.get('detail', '')}")
            for label, key, noun in (
                ("host lane    ", "host_lane", "(family, dtype) pair(s) the compiler must place on the HOST"),
                ("epilogue     ", "epilogue", "fusable stage(s)"),
                ("conv window  ", "conv_geometry", "convolution window(s) real captures contain"),
                ("scope        ", "scope", "adjacency chain(s) real captures present"),
                ("groups       ", "groups", "stage combination(s) real captures present"),
                ("carried state", "carried_state", "carried-state behaviour(s) the target declares"),
            ):
                ax = r.get(key) or {}
                if ax.get("status") != "ok":
                    print(f"   {label}: {ax.get('status', 'missing')} — {ax.get('detail', '')}")
                    continue
                print(f"   {label}: {ax['n_covered']} / {ax['n_required']} {noun}")
                for miss in ax.get("uncovered") or ():
                    mark = " " if _debt(r["target"], miss, key) in ratchet else "*"
                    print(f"     {mark} {miss}")
                # A stage evidenced ONLY standalone says the lowering exists and the FUSION is not
                # tested, which is a different remedy from the reverse; one number cannot carry both.
                if ax.get("standalone_only"):
                    print(f"       evidenced standalone only (fusion untested): {ax['standalone_only']}")
                if ax.get("entry_tensor_only"):
                    print(
                        f"       evidenced only by a whole model's ENTRY tensor, which is not an "
                        f"operand dtype: {ax['entry_tensor_only']}"
                    )
                if ax.get("covered_only_incidentally"):
                    print(
                        f"       covered only INCIDENTALLY (no capsule is named for it): "
                        f"{ax['covered_only_incidentally']}"
                    )
            cc = r.get("certification") or {}
            if cc.get("implied_total_s"):
                print(
                    f"   certification  : {cc['members']} member(s) demand it at {cc['budget_s']:.0f}s each "
                    f"= {cc['implied_total_s'] / 3600.0:.1f}h per grade  ({cc['source']})"
                )
            cp = r.get("corpus_pressure") or {}
            if cp.get("obligations"):
                print(
                    f"   corpus pressure: {cp['capsules']} capsule(s) for {cp['obligations']} "
                    f"obligation(s) across every axis  ({cp['capsules_per_obligation']}x)"
                )
            if r["corpus_cells_not_required"]:
                print(f"   corpus cells not in the requirement: {r['corpus_cells_not_required']}")
                print(
                    "     (a cell the hardware does not admit for that family — e.g. an int8 movement "
                    "capsule on a target whose movement datapath is float-only — is INTENTIONAL: it is "
                    "what forces the compiler off the accelerator path)"
                )
            for n in r["diagnostics"].get("notes") or []:
                print(f"   note: {n}")

    bad = uncovered_debt(reports, ratchet)

    # ⚠️ A TARGET THAT COULD NOT BE AUDITED HAS ESTABLISHED NOTHING. Every `bad` list above filters on
    # `status == "ok"`, so a target whose descriptor, contract or corpus could not be resolved
    # contributes no debt and the gate returns 0 -- reporting success for a question it never asked.
    # Measured: `saturn_opu_mxv256d128` and `..._rvv` reported `no_target_experiment` and this gate
    # exited 0 for BOTH; with the descriptor found by its declared name they owe 5 uncovered items.
    # This repo has now paid for that shape five times, so it is spelled 2 ("cannot decide"), never 0.
    unrunnable = [r for r in reports if r["status"] != "ok"]
    incomplete_applications = [
        r for r in reports if r["status"] == "ok" and (r.get("application_demands") or {}).get("status") == "incomplete"
    ]
    unverified_applications = [
        r
        for r in reports
        if r["status"] == "ok"
        and r.get("applications_declared")
        and (r.get("application_demands") or {}).get("status") in {None, "not_measured", "incomplete"}
    ]
    if incomplete_applications:
        print(
            f"\n  INCOMPLETE APPLICATION INVENTORY ({len(incomplete_applications)}) "
            "— cell coverage does not close this debt:",
            file=sys.stderr,
        )
        for r in incomplete_applications:
            print(f"    ? {r['target']:28s} application operations remain unclassified", file=sys.stderr)
    if unrunnable:
        print(
            f"\n  COULD NOT AUDIT ({len(unrunnable)}) — these establish NOTHING, they are not clean:", file=sys.stderr
        )
        for r in unrunnable:
            print(f"    ? {r['target']:28s} {r['status']}: {r.get('detail', '')}", file=sys.stderr)

    if bad and a.fail_on_uncovered:
        print(f"\nFAIL: {len(bad)} required cell(s) uncovered and not ratcheted", file=sys.stderr)
        return 1
    if (unrunnable or incomplete_applications or (a.fail_on_unverifiable and unverified_applications)) and (
        a.fail_on_uncovered or a.fail_on_unverifiable
    ):
        print(
            f"\nCANNOT DECIDE: {len(unrunnable)} target(s) could not be audited and "
            f"{len(incomplete_applications)} application inventory/inventories were incomplete; "
            f"{len(unverified_applications)} declared application inventory/inventories are missing or incomplete.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
