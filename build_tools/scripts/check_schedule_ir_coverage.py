#!/usr/bin/env python3
"""Gate: every scheduling-IR capability axis is declared, and its status is MEASURED not asserted.

"Our IR is more general than X" is the kind of claim this repo has learned it cannot check. Stated in
prose it is an instrument that cannot fail: a capability that quietly stopped being expressible reads
exactly like one that never was. `merlin/contract/schedule_ir_coverage.yaml` declares the claim one axis
at a time; this gate re-derives each axis' STATUS from the live tree and refuses a register that has
drifted from it in EITHER direction -- a primitive with no row, or a row naming a primitive that is gone.

WHAT IT DELIBERATELY DOES NOT GATE: the coverage ratio. Gating "expressed / total >= x" makes the
rational response to a hard axis deleting its row, which is precisely the incentive the register exists
to defeat. This checks the number is MEASURABLE and its denominator complete, and leaves what the number
says to the reader. Same stance as check_semantic_coverage.py, for the same reason.

Exit codes: 0 clean, 1 debt, 2 CANNOT DECIDE (register missing or unparseable, primitives unimportable,
roster unresolvable, a declared corpus with a zero denominator). A gate that could not run must never
report success -- this repo has five recorded instances of exactly that.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "merlin" / "python"))
# This gate's own directory, so its sibling helpers import whether it is RUN as a script or LOADED by
# path. The regression tests load every gate by path -- that is the point, since a gate imported as a
# module is not the thing the hook runs -- and a helper that resolves only under one of the two makes
# the gate untestable in exactly the way that lets a broken gate ship.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_strict(yaml, text: str):
    """Parse the register, REFUSING a mapping that states the same key twice.

    A plain load keeps the last of two identical keys and drops the first without a word, which in a
    register whose whole purpose is that a claim cannot outrun the code is the one failure that
    matters: a row read as `primitive: [pipeline]` while it also says `primitive: null` is a row whose
    text and meaning disagree, and the disagreement is invisible to every reader and to this gate.
    Measured once, on `hazard_resolution_trait`. Raising here routes it to `exit 2` -- a register this
    gate cannot read unambiguously is one it cannot decide on, not one that passes.
    """

    class _Strict(yaml.SafeLoader):
        pass

    def _mapping(loader, node, deep=False):
        seen: set = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.YAMLError(
                    f"line {key_node.start_mark.line + 1}: {key!r} is stated twice in one entry; the "
                    "second silently replaces the first"
                )
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    _Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
    return yaml.load(text, Loader=_Strict)  # noqa: S506 - _Strict derives from SafeLoader


REGISTER = ROOT / "merlin" / "contract" / "schedule_ir_coverage.yaml"
RATCHET = Path(__file__).resolve().parent / "schedule_ir_coverage_ratchet.txt"
_GATE = "schedule-ir-coverage"

#: Computed, never authored. See the register's own header for what each one means.
STATUSES = ("UNEXPRESSED", "EXPRESSED", "EXERCISED", "UNMEASURED")
PROOFS = ("by_construction", "bounded_check", "rtl_gate")
EXO = ("EXO_HAS", "EXO_PARTIAL", "EXO_ABSENT")

#: What a row may say about a prior system. UNASSESSED is the third state and the reason this is a
#: vocabulary rather than a boolean: "that system does not have this" and "nobody has looked" are
#: different claims, and collapsing them is how a denominator drawn from ONE system got read as a
#: statement about the field. An UNASSESSED row owes a reason like any other non-HAS verdict, so the
#: unlooked-at corner is countable instead of invisible.
PRIOR_ART_STATES = ("HAS", "PARTIAL", "ABSENT", "UNASSESSED")

REQUIRED = ("id", "axis", "exo", "forced_by", "generic", "owner")
REQUIRED_BY_STATUS = {
    "UNEXPRESSED": ("surfaced_by", "surfaced_on", "blocks"),
    "EXPRESSED": ("primitive", "test"),
    "EXERCISED": ("primitive", "test"),
    "UNMEASURED": ("missing_input",),
}


def _declared_systems(declared) -> tuple[set[str], dict[str, str], list[str]]:
    """``(systems rows must score, {blanketed system: why}, problems)`` from the register's own list.

    A system may declare a BLANKET of UNASSESSED with a reason, and then no row scores it. That is the
    honest shape for a system nobody has verified against: the fact is one fact -- "there is no vendored
    or pinned copy of it in this checkout, so a verdict would be recalled rather than checked" -- and
    writing it into thirteen rows would multiply one admission into thirteen assertions that each LOOK
    like an independent judgement.

    It also creates the right pressure in the right direction. Vendoring the system means deleting its
    blanket, and the gate then demands a verdict on every row at once, which is the audit that widening
    a denominator is supposed to force.
    """
    systems: set[str] = set()
    blanket: dict[str, str] = {}
    problems: list[str] = []
    for entry in declared:
        if not isinstance(entry, dict) or not entry.get("id"):
            problems.append("prior_art entries need an 'id'")
            continue
        name = str(entry["id"])
        if entry.get("blanket") is None:
            systems.add(name)
            continue
        if entry["blanket"] != "UNASSESSED":
            problems.append(
                f"prior_art {name!r}: a blanket may only be UNASSESSED. A system that HAS or lacks an "
                "axis says so per row, because that is a claim about each axis and not about the system."
            )
        if not entry.get("why"):
            problems.append(f"prior_art {name!r}: a blanket needs a 'why' saying what is missing to score it")
        blanket[name] = str(entry.get("why") or "")
    return systems, blanket, problems


def _prior_art_of(row: dict) -> tuple[dict, dict]:
    """``(state_by_system, evidence_by_system)``, folding the legacy ``exo:`` spelling into the map.

    The register compared against exactly one system, in a field named after it, checked by a constant
    named after it. That is fine as a fact and misleading as a headline: "11 of 13 axes expressed" reads
    as coverage of the field and means coverage of the axes one system's absence suggested. Which
    systems are in the denominator is now DATA -- a top-level list in the register -- so adding a fourth
    costs an edit there and a verdict per row, not a change here.
    """
    states = dict(row.get("prior_art") or {})
    evidence = dict(row.get("prior_art_evidence") or {})
    legacy = row.get("exo")
    if isinstance(legacy, str) and legacy.startswith("EXO_"):
        states.setdefault("exo", legacy[len("EXO_") :])
        if row.get("exo_evidence"):
            evidence.setdefault("exo", row["exo_evidence"])
    return states, evidence


def _prior_art_problems(rid: str, row: dict, systems: set[str]) -> list[str]:
    """Every declared system scored on every row, with a reason for anything short of HAS.

    Requiring a verdict per system is the part with teeth. Declaring a system and letting rows omit it
    widens the denominator on paper while changing nothing, which is worse than not declaring it -- the
    register would then look like it had been checked against three systems and have been checked
    against one.
    """
    states, evidence = _prior_art_of(row)
    problems: list[str] = []
    for name in sorted(states):
        state = states[name]
        if name not in systems:
            problems.append(
                f"{rid}: prior_art scores {name!r}, which the register's top-level prior_art list does "
                "not declare. Declare the system before scoring rows against it."
            )
            continue
        if state not in PRIOR_ART_STATES:
            problems.append(f"{rid}: prior_art {name} is {state!r}, not one of {list(PRIOR_ART_STATES)}")
        elif state != "HAS" and not evidence.get(name):
            problems.append(
                f"{rid}: prior_art {name} is {state} and cites no evidence. A claim about what another "
                "system cannot do -- or that nobody looked -- is exactly the claim that needs one."
            )
    unscored = sorted(s for s in systems if s not in states)
    if unscored:
        problems.append(
            f"{rid}: prior art unscored for {unscored}. Every declared system gets a verdict on every "
            "row, UNASSESSED with a reason included, or the denominator is wider than the measurement."
        )
    return problems


def _slice_count(rid: str, forced: list[str], basis) -> tuple[int, list[str]]:
    """How many INDEPENDENT slices a row's forcing targets amount to, and what is wrong with the claim.

    `forced_by` names TARGETS; the two-slice rule is about independent SLICES, and the two are not the
    same thing. Two names can be one device -- one elaboration declared twice so that neither target's
    kernels are ever attributed to the other -- and counting names there satisfies the rule with a single
    piece of silicon. That is not hypothetical: an axis was promoted to core vocabulary on 2026-09-18 by
    counting `muon` and `radiance`, which resolve to one `RadianceMuonConfig` cluster and derive
    byte-identical machines, and it took a separate audit to notice.

    So the grouping is DECLARED, in `slice_basis`, rather than inferred from names or guessed from a
    shared facts path. A target no entry mentions is its own slice, which is both the honest default and
    the reason every existing row needed no edit.
    """
    problems: list[str] = []
    grouped: dict[str, str] = {}
    names: set[str] = set()
    for entry in basis or ():
        if not isinstance(entry, dict) or not entry.get("slice") or not entry.get("targets") or not entry.get("why"):
            problems.append(f"{rid}: slice_basis entries need a 'slice', a 'targets' list and a 'why'")
            continue
        name = str(entry["slice"])
        if name in names:
            problems.append(f"{rid}: slice_basis declares {name!r} twice")
        names.add(name)
        members = list(entry["targets"])
        if len(members) < 2:
            problems.append(
                f"{rid}: slice_basis {name!r} groups {len(members)} target(s). An entry exists to say "
                "that SEVERAL names are one slice; one name is already one slice."
            )
        for t in members:
            if t not in forced:
                problems.append(f"{rid}: slice_basis {name!r} names {t!r}, which is not in forced_by")
            elif t in grouped:
                problems.append(f"{rid}: {t!r} is in two slices, {grouped[t]!r} and {name!r}")
            else:
                grouped[t] = name
    problems.extend(_undeclared_shared_design(rid, forced, grouped))
    return len(names) + len([t for t in forced if t not in grouped]), problems


def _undeclared_shared_design(rid: str, forced: list[str], grouped: dict[str, str]) -> list[str]:
    """Forcing targets that PROVABLY resolve to one design and are not declared as one slice.

    `slice_basis` is a declaration, so on its own it can be omitted: drop it and two names silently count
    as two slices again, which is the original defect. This closes the half that can be decided without
    the RTL toolchain -- `facts_alias` reads a target's own declaration (a residual's `facts_target`, else
    the registry) and is pure file IO, so it grounds in CI. It catches the variant-of-another-generator
    case: `mx_gemmini` resolves to `gemmini`, so a row forced by both is one slice unless it says why not.

    What it deliberately does NOT catch is two targets with distinct aliases that nonetheless elaborate
    the SAME configuration -- `muon` and `radiance` both being `RadianceMuonConfig`. Deciding that needs
    the facts artifact's `source.config`, which a clone without the toolchain does not have, and a check
    that silently passes when it cannot look is worse than one that does not claim to look. That case is
    held by a test instead.
    """
    try:
        from merlin.targetgen.rtl.facts import facts_alias
    except Exception:  # noqa: BLE001 — no merlin on the path is not a coverage problem
        return []
    designs: dict[str, list[str]] = {}
    for t in forced:
        try:
            designs.setdefault(str(facts_alias(t)), []).append(t)
        except Exception:  # noqa: BLE001 — an unresolvable target is reported by the roster check
            continue
    out: list[str] = []
    for design, members in designs.items():
        if len(members) < 2:
            continue
        slices = {grouped.get(t) for t in members}
        if None in slices or len(slices) > 1:
            out.append(
                f"{rid}: forced_by names {sorted(members)}, which all resolve to the design {design!r}, "
                "but they are not declared as one slice. Either group them in slice_basis with a why, "
                "or say what makes them independent for THIS axis."
            )
    return out


def status_of(row: dict, live_primitives: set[str], measured: dict | None = None) -> str:
    """The state this row's EVIDENCE supports, derived from the live tree.

    Not read from the row: a status a human can write is a status a human can be wrong about, and the
    one thing worth knowing is whether the tree still supports it.

    ``measured`` maps a row id to a :class:`merlin.sched.expressiveness.AxisMeasurement`. When one is
    supplied IT decides, because ``evidence.measured_on`` is otherwise a field a human can type -- the
    one status in this register that could be asserted rather than computed, in a file whose whole
    premise is that a claim cannot outrun the code. Without a measurement the row is read as before,
    so the gate still runs in a checkout with no external corpus; what it does NOT do then is report
    EXERCISED on an unverified `measured_on`, which is why `verdict` also raises that contradiction.
    """
    claimed = _primitives_of(row)
    if not claimed or not claimed <= live_primitives:
        return "UNEXPRESSED"
    evidence = row.get("evidence") or {}
    if not evidence.get("corpus"):
        return "EXPRESSED"
    if measured is not None:
        found = measured.get(row.get("id"))
        if found is not None:
            return found.status
    if row.get("missing_input"):
        return "UNMEASURED"
    # EXERCISED IS NOT REACHABLE FROM THE REGISTER'S OWN TEXT, and that is the whole point of the file.
    # This line used to read `"EXERCISED" if evidence.get("measured_on")`, so the strongest status was
    # conferred by a DATE A HUMAN TYPES -- in a file whose header promises "status is computed, never
    # authored". The test guarding that promise checked only a literal `status:` key, which is the door
    # nobody would use; `measured_on` was the door that was open. A corpus-lift measurement can grant
    # this status by re-deriving it; until one exists in the tree, the answer is EXPRESSED and a row
    # claiming otherwise is reported by `_authored_measurement`.
    return "EXPRESSED"


def _authored_measurement(row: dict) -> str | None:
    """A row asserting it was measured, in a gate that cannot re-derive the measurement.

    Silently ignoring the claim would leave the register reading as though someone had checked. Saying
    so keeps the difference between "measured" and "asserted to have been measured" visible, which is
    the only difference this file exists to keep.
    """
    evidence = row.get("evidence") or {}
    if not evidence.get("measured_on"):
        return None
    return (
        f"{row.get('id')}: declares evidence.measured_on={evidence['measured_on']!r}, but nothing in "
        "this tree re-derives that measurement, so the claim cannot be checked and is not believed. "
        "Remove it, or land the corpus-lift measurement that computes it."
    )


def verdict(
    rows: list[dict],
    live_primitives: set[str],
    roster: set[str],
    proofs: dict[str, str] | None = None,
    measured: dict | None = None,
    systems: set[str] | None = None,
) -> tuple[dict, list[str], int]:
    """``(status_by_id, problems, rc)`` -- the whole decision, as a pure function.

    Pure so a test can assert each problem class directly, without a register, a corpus or a simulator.
    Two gates in this repo measured something, printed it, and were structurally unable to fail on it;
    the fix both times was to move the decision somewhere a test could reach.
    """
    proofs = proofs if proofs is not None else primitive_proofs()
    problems: list[str] = []
    statuses: dict[str, str] = {}
    seen: set[str] = set()
    claimed: set[str] = set()

    for row in rows:
        rid = row.get("id")
        if not rid:
            problems.append("a row has no id")
            continue
        if rid in seen:
            problems.append(f"{rid}: declared twice")
        seen.add(rid)
        for field in REQUIRED:
            if row.get(field) in (None, "", []):
                problems.append(f"{rid}: missing required field {field!r}")
        if row.get("exo") not in EXO:
            problems.append(f"{rid}: exo {row.get('exo')!r} not in {list(EXO)}")
        elif row.get("exo") != "EXO_HAS" and not row.get("exo_evidence"):
            problems.append(f"{rid}: claims an existing language lacks this axis but cites no evidence")
        problems.extend(_prior_art_problems(rid, row, systems if systems is not None else {"exo"}))
        forced = list(row.get("forced_by") or [])
        unknown = [t for t in forced if t not in roster]
        if unknown:
            problems.append(f"{rid}: forced_by names {unknown}, which no target registry declares")
        # `forced_by_unknown` is the third state, and it exists because collapsing it into either of the
        # other two loses the thing worth knowing. A target that contributes nothing to an axis because
        # its data was never AUTHORED is not the same as one measured not to need it -- muon supplies no
        # latency or completion kind for the plain reason that no muon schedule contract exists, and a
        # SIMT device plainly has both. Listing it as forcing the axis overstates the denominator;
        # dropping it silently records "does not need this", which is a stronger claim than anyone
        # checked. Named here with the missing input instead, and NOT counted toward the two-slice rule.
        for entry in row.get("forced_by_unknown") or []:
            if not isinstance(entry, dict) or not entry.get("target") or not entry.get("missing"):
                problems.append(f"{rid}: forced_by_unknown entries need a 'target' and a 'missing'")
                continue
            if entry["target"] not in roster:
                problems.append(f"{rid}: forced_by_unknown names {entry['target']!r}, which no registry declares")
            if entry["target"] in forced:
                problems.append(
                    f"{rid}: {entry['target']!r} is in both forced_by and forced_by_unknown; it either "
                    "forces the axis or its need is undetermined, not both"
                )
        authored = _authored_measurement(row)
        if authored:
            problems.append(authored)
        n_slices, slice_problems = _slice_count(rid, forced, row.get("slice_basis"))
        problems.extend(slice_problems)
        if row.get("generic") is True and n_slices < 2:
            problems.append(
                f"{rid}: generic: true with {n_slices} independent slice(s) across forced_by {forced}. "
                "A generic primitive is added only when two slices need it; one slice is a target "
                "extension. Where several of those names are ONE device, declare it in slice_basis."
            )
        if row.get("proof") is not None:
            problems.append(
                f"{rid}: declares proof {row['proof']!r}. An obligation is read from the primitive that "
                "carries it, never authored here -- a row could otherwise claim a stronger one than the "
                "code discharges, which is the exact claim this register exists to stop being writable."
            )
        for name in sorted(named := _primitives_of(row)):
            declared = proofs.get(name)
            if name in live_primitives and declared not in PROOFS:
                problems.append(f"{rid}: primitive {name!r} declares no obligation in {list(PROOFS)}")
        claimed |= named
        for missing in sorted(named - live_primitives):
            problems.append(f"{rid}: names primitive {missing!r}, which is not exported")

        status = status_of(row, live_primitives, measured)
        statuses[rid] = status
        found = (measured or {}).get(rid)
        for field in REQUIRED_BY_STATUS.get(status, ()):
            # A measurement that COMPUTED the field satisfies the requirement to declare it. Demanding
            # the register also spell a `missing_input` the measurement just derived would make an
            # author retype a machine-readable fact, and a retyped fact is one that can go stale.
            if row.get(field) in (None, "", []) and not getattr(found, field, None):
                problems.append(f"{rid}: computed {status} but declares no {field!r}")
        if found is not None and (row.get("evidence") or {}).get("measured_on") and found.status != "EXERCISED":
            problems.append(
                f"{rid}: declares evidence.measured_on but the measurement says {found.status} "
                f"({found.missing_input or found.triple.text()}). A measurement someone typed is the one "
                "thing this register is built to make impossible; delete the field or fix the corpus."
            )
        if status in ("EXPRESSED", "EXERCISED") and row.get("test"):
            if not _test_exists(row["test"]):
                problems.append(f"{rid}: names test {row['test']!r}, which does not exist")

    for orphan in sorted(live_primitives - claimed):
        problems.append(
            f"primitive {orphan!r} is exported but no row declares it. New vocabulary is declared, "
            "never absorbed: a primitive nobody has to justify is how an IR grows one per kernel."
        )
    return statuses, problems, (1 if problems else 0)


def _primitives_of(row: dict) -> set[str]:
    """The primitives a row claims. One axis is often several moves -- restructuring a loop nest is
    split, reorder, fuse and unroll -- so a single-valued field would force either four near-identical
    rows or a claim narrower than the truth."""
    value = row.get("primitive")
    if not value:
        return set()
    return {value} if isinstance(value, str) else set(value)


def _test_exists(nodeid: str) -> bool:
    path, _, name = nodeid.partition("::")
    file = ROOT / path
    if not file.is_file():
        return False
    return (not name) or (name in file.read_text(encoding="utf-8", errors="replace"))


def live_primitives() -> set[str]:
    """Every primitive the tree actually exports. Reflected, never listed.

    A completeness check whose universe is hand-maintained can only confirm what someone remembered.
    An absent module is an empty set, not an error: the register is meant to be writable before the
    primitives exist, and every row then reads UNEXPRESSED, which is the truth.
    """
    try:
        import merlin.sched.primitives as prims  # noqa: PLC0415
    except ImportError:
        return set()
    return {n for n in (getattr(prims, "__all__", ()) or ()) if getattr(getattr(prims, n, None), "PROOF", None)}


def primitive_proofs() -> dict[str, str]:
    """``{primitive: its declared obligation}``, read from the code that carries it."""
    try:
        import merlin.sched.primitives as prims  # noqa: PLC0415
    except ImportError:
        return {}
    out = {}
    for name in getattr(prims, "__all__", ()) or ():
        fn = getattr(prims, name, None)
        proof = getattr(fn, "PROOF", None)
        if proof is not None:
            out[name] = proof
    return out


def _roster() -> set[str]:
    from _target_roster import target_names  # noqa: PLC0415

    return set(target_names(ROOT))


def _ratchet() -> set[str]:
    if not RATCHET.is_file():
        return set()
    return {
        line.split("#", 1)[0].strip()
        for line in RATCHET.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }


def _unexaminable(reason: str, *, stop_hook: bool) -> int:
    text = f"{_GATE}: {reason}; NOTHING was examined, which is not the same as clean. Fix the tree and re-run."
    if stop_hook:
        print(json.dumps({"decision": "block", "reason": text}))
        return 0
    print(f"[FAIL] {text}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--staged", action="store_true", help="only run when the register or a primitive is staged")
    parser.add_argument("--stop-hook", action="store_true", help="report a refusal as a Stop-hook JSON decision")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument(
        "--measure",
        action="store_true",
        help="run the expressiveness measurement and let IT decide each row's status, instead of "
        "reading evidence.measured_on. Off by default so the pre-commit path needs no external "
        "checkout; on, an unresolvable corpus reports UNMEASURED naming the input it needs.",
    )
    args = parser.parse_args(argv)

    if args.staged:
        try:
            changed = subprocess.run(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
        except (OSError, subprocess.CalledProcessError) as exc:
            return _unexaminable(f"could not list the staged files ({exc})", stop_hook=args.stop_hook)
        touched = any("schedule_ir_coverage" in c or "sched/primitives" in c for c in changed)
        if not touched:
            return 0

    if not REGISTER.is_file():
        return _unexaminable(f"no register at {REGISTER}", stop_hook=args.stop_hook)
    try:
        import yaml  # noqa: PLC0415

        doc = _load_strict(yaml, REGISTER.read_text(encoding="utf-8")) or {}
        rows = list(doc.get("entries") or [])
    except Exception as exc:  # noqa: BLE001
        return _unexaminable(f"the register could not be parsed ({exc})", stop_hook=args.stop_hook)
    if not rows:
        return _unexaminable("the register declares no axis", stop_hook=args.stop_hook)
    try:
        roster = _roster()
    except Exception as exc:  # noqa: BLE001
        return _unexaminable(f"the target roster could not be resolved ({exc})", stop_hook=args.stop_hook)
    if not roster:
        return _unexaminable("the target roster is empty", stop_hook=args.stop_hook)

    measured = None
    if args.measure:
        try:
            from merlin.sched.expressiveness import measure_register, totals  # noqa: PLC0415

            measured = {m.axis_id: m for m in measure_register(rows)}
        except Exception as exc:  # noqa: BLE001
            return _unexaminable(f"the expressiveness measurement could not run ({exc})", stop_hook=args.stop_hook)

    # Which prior systems are in the denominator is the register's own declaration, so widening it is a
    # data edit. Absent the list, the historical single system, so an older register still validates.
    declared = doc.get("prior_art") or [{"id": "exo"}]
    systems, blanket, prior_art_problems = _declared_systems(declared)
    statuses, problems, rc = verdict(rows, live_primitives(), roster, measured=measured, systems=systems)
    problems = prior_art_problems + problems
    if prior_art_problems:
        rc = max(rc, 1)
    ratcheted = [p for p in problems if p in _ratchet()]
    fresh = [p for p in problems if p not in _ratchet()]

    counts = {s: sum(1 for v in statuses.values() if v == s) for s in STATUSES}
    total = len(statuses)
    if args.json:
        payload = {"statuses": statuses, "counts": counts, "problems": problems}
        if measured is not None:
            whole = totals(measured.values())
            payload["triple"] = {
                "expressed": whole.expressed,
                "denominator": whole.denominator,
                "unexercised": whole.unexercised,
                "bounds": list(whole.bounds()),
            }
            payload["axes"] = {
                m.axis_id: {
                    "status": m.status,
                    "triple": m.triple.text(),
                    "corpus": m.corpus,
                    "missing_input": m.missing_input,
                    "measured_on": m.measured_on,
                }
                for m in measured.values()
            }
        print(json.dumps(payload, indent=2))
    else:
        for problem in fresh:
            print(f"[DEBT] {_GATE}: {problem}")
        for problem in ratcheted:
            print(f"[NOTE] {_GATE}: (ratcheted) {problem}")
        # UNMEASURED counts toward EXPRESSED, because a row only reaches the corpus branch of
        # `status_of` once its primitives are live -- it cleared the expressed bar by construction.
        # Counting it out would mean declaring a corpus LOWERS the coverage number, which is the same
        # perverse incentive as gating the ratio: the cheapest way to look better would be to delete
        # the evidence pointer.
        expressed = counts["EXPRESSED"] + counts["EXERCISED"] + counts["UNMEASURED"]
        print(
            f"[{'  ok' if not fresh else 'FAIL'}] {_GATE}: {expressed}/{total} axes expressed "
            f"({counts['EXERCISED']} exercised, {counts['UNEXPRESSED']} unexpressed, "
            f"{counts['UNMEASURED']} unmeasured and charged to the denominator). "
            "The ratio is reported, never gated."
        )
        scored = sorted(systems)
        print(
            f"       prior art: scored against {scored or ['nothing']}"
            + (
                f"; UNASSESSED against {sorted(blanket)} -- so the ratio above is a statement about the "
                "axes those scored systems suggested, not about the field"
                if blanket
                else ""
            )
        )
        if measured is not None:
            whole = totals(measured.values())
            lo, hi = whole.bounds()
            print(
                f"       expressed/denominator/unexercised = {whole.text()} over the CCA fields the "
                f"register's rows name; agreement is somewhere in [{lo:.2f}, {hi:.2f}]. Every "
                "undecidable field is charged to the denominator, so thin evidence reads as a wide "
                "range and never as a high number."
            )
        if args.summary:
            for rid, status in sorted(statuses.items()):
                line = f"    {status:12s} {rid}"
                m = (measured or {}).get(rid)
                if m is not None:
                    line += f"  {m.triple.text()}"
                    if m.missing_input:
                        line += f"  MISSING: {m.missing_input}"
                    elif m.measured_on:
                        line += f"  on {m.measured_on}"
                print(line)
    return 1 if fresh else 0


if __name__ == "__main__":
    raise SystemExit(main())
