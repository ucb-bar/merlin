"""How much of a target's capsule corpus has ever been graded, passed and certified.

WHY THIS IS NOT A PASS COUNT. ``n_passed`` answers "how did this run do on the suite it was
handed". Coverage answers a different question -- "how much of the benchmark has anyone ever
exercised, and to what depth of evidence" -- and the two come apart badly. A target can score
42/42 on every run it ever had while 135 capsules of its declared corpus have never been graded
once. Reporting only the first reads as complete coverage of a corpus nobody has touched.

THREE DENOMINATORS, KEPT APART. A capsule name appearing in a verdict is not automatically
coverage of the target it was graded under:

* **roster** -- the target's declared corpus (``profiles/<target>*.yaml``). This is the
  denominator, because it is what the benchmark claims to test.
* **off-roster, owned elsewhere** -- graded under this target while another target's roster
  declares it. This has really happened: a self-check globbed the wrong suite and graded a
  different device's capsules. Folding those in inflates the denominator with someone else's
  corpus and makes a broken run look broad.
* **off-roster, unowned** -- graded historically, in no current roster. Real evidence about a
  corpus that has since changed, which is not the same as evidence about today's corpus.

Only the first is scored. The other two are counted, attributed and reported, never merged and
never dropped -- a silently unioned denominator is how a suite bug turns into a coverage claim.

THE CERTIFYING BAR IS PER CAPSULE AND DERIVED. Capsules declare their own
``required_oracle_tiers``, and they disagree: some are satisfied by the loop tier, others demand
an elaborated-RTL certificate. So "certified" is judged against each capsule's OWN declared bar,
read from the corpus. One bar applied corpus-wide would either credit a capsule for evidence it
never needed or fault it for evidence it never asked for -- and both errors land in the same
direction the reader is least able to check.

DEPTH IS SEPARATE FROM PASSING. A capsule can pass a certifying tier while its overall verdict
fails (numerics agree, a trace check does not). So the deepest tier ever passed and whether an
overall pass was ever recorded are tracked as two facts. A figure that shows only the first
overstates the evidence; one that shows only the second throws away the tier ladder the whole
grading design exists to provide.

Target-agnostic by construction: the roster, the per-capsule bars and who owns which capsule name
all arrive as arguments. Nothing here knows a device.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .availability import Availability, derived, measured, unavailable

#: Roster file suffix -> the lane it declares. Data, not policy: a caller with a different corpus
#: layout passes its own map. ``""`` is the target's own profile.
DEFAULT_LANES: Mapping[str, str] = {"": "public", ".synth": "synth", ".hidden": "hidden"}

#: Where a graded capsule sits relative to the target's roster.
ON_ROSTER = "on_roster"
OFF_ROSTER_OWNED = "off_roster_owned"
OFF_ROSTER_UNOWNED = "off_roster_unowned"

#: Coverage depth, worst to best. ``NEVER_GRADED`` is a stronger statement than
#: ``GRADED_NEVER_PASSED`` and the two must not share a colour: one says nobody looked, the other
#: says somebody looked and it did not work.
NEVER_GRADED = "never_graded"
GRADED_NEVER_PASSED = "graded_never_passed"
PASSED_BELOW_BAR = "passed_below_bar"
PASSED_AT_BAR = "passed_at_bar"
PASSED_ABOVE_BAR = "passed_above_bar"
#: Passed something, but the capsule declares no readable bar, so "certified" is unjudgeable.
BAR_UNKNOWN = "bar_unknown"
DEPTHS = (NEVER_GRADED, GRADED_NEVER_PASSED, PASSED_BELOW_BAR, PASSED_AT_BAR,
          PASSED_ABOVE_BAR, BAR_UNKNOWN)


def tier_rank(label: str) -> int:
    """Ordinal of an oracle tier label, or -1 when it cannot be read.

    Parsed structurally rather than matched against a list of known tiers, so a corpus that adds
    a tier does not need an edit here. ``"L3"`` and ``"L3-verilator"`` rank the same -- the engine
    suffix names which simulator produced the evidence, not how deep the evidence goes.

    Returns -1 for a label carrying no ordinal at all, which the caller must treat as UNKNOWN
    rather than as tier zero; a bad label ranking as 0 would silently read as shallow evidence.
    """
    body = label.split("-", 1)[0].strip()
    digits = "".join(ch for ch in body if ch.isdigit())
    return int(digits) if digits else -1


def deepest_tier(labels: Iterable[str]) -> str:
    """The deepest readable tier among ``labels``, or "" if none is readable."""
    best, best_rank = "", -1
    for label in labels:
        rank = tier_rank(label)
        if rank > best_rank:
            best, best_rank = label, rank
    return best


@dataclass(frozen=True)
class Observation:
    """One capsule's outcome in one grade."""

    capsule: str
    label: str
    status: str
    tiers: Mapping[str, str]
    failure_plane: str = ""
    failure_category: str = ""


def observations(run_dir: Path) -> tuple[tuple[Observation, ...], str]:
    """Every per-capsule row this run recorded, with a reason when there are none.

    Reads the verdict history rather than the run manifest: the manifest carries the run's final
    score, while the history carries which capsule earned what, which is the only thing coverage
    can be computed from. A malformed verdict is skipped and counted, not raised -- one bad file
    in a 736-file archive must not cost the whole figure -- but the count reaches the caller so
    the loss is reportable.
    """
    hist = run_dir / "qa_history"
    if not hist.is_dir():
        return (), f"{run_dir.name} wrote no qa_history/"
    rows: list[Observation] = []
    files = unreadable = 0
    for path in sorted(hist.glob("verdict_*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            unreadable += 1
            continue
        per = doc.get("per_capsule")
        if not isinstance(per, list):
            continue
        files += 1
        for row in per:
            if not isinstance(row, dict):
                continue
            name = row.get("capsule")
            if not name:
                continue
            tiers = row.get("tiers")
            rows.append(Observation(
                capsule=str(name),
                label=str(row.get("label") or ""),
                status=str(row.get("status") or ""),
                tiers=dict(tiers) if isinstance(tiers, dict) else {},
                failure_plane=str(row.get("failure_plane") or ""),
                failure_category=str(row.get("failure_category") or "")))
    if not rows:
        why = (f"{run_dir.name} has {unreadable} unreadable verdict file(s)" if unreadable
               else f"{run_dir.name} recorded no per-capsule rows in {files} verdict file(s)")
        return (), why
    return tuple(rows), ""


def read_roster(profile_dir: Path, target: str, *,
                lanes: Mapping[str, str] | None = None) -> tuple[dict[str, str], str]:
    """``{capsule name: lane}`` for one target, plus a reason when nothing could be read.

    The roster is the DECLARED corpus, which is why it is the denominator: it states what the
    benchmark claims to test, independently of what any run happened to be handed. Sweeps are not
    expanded here because the corpus generator already materialises them into the ``capsules``
    list -- expanding again would double-count.
    """
    import yaml  # local: keeps the module importable where the report is not being built

    lanes = DEFAULT_LANES if lanes is None else lanes
    out: dict[str, str] = {}
    for suffix, lane in lanes.items():
        path = profile_dir / f"{target}{suffix}.yaml"
        if not path.is_file():
            continue
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            continue
        for entry in (doc.get("capsules") or []):
            if isinstance(entry, dict) and entry.get("name"):
                # First lane to declare a name owns it: a hidden re-declaration of a public
                # capsule must not silently reclassify the public one.
                out.setdefault(str(entry["name"]), lane)
    if not out:
        return {}, f"no roster file for {target!r} under {profile_dir}"
    return out, ""


def read_required_tiers(corpus_dir: Path) -> tuple[dict[str, str], str]:
    """``{capsule name: its deepest declared required tier}`` read from the corpus itself.

    This is the bar a capsule must clear to count as certified, and it is DERIVED from the
    capsule's own ``required_oracle_tiers`` rather than assumed. A capsule declaring no tiers is
    absent from the result, so the caller can mark it unjudgeable instead of defaulting it into
    whichever band happens to be convenient.
    """
    import yaml

    out: dict[str, str] = {}
    seen = 0
    for path in sorted(corpus_dir.rglob("capsule.yaml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            continue
        seen += 1
        if not isinstance(doc, dict):
            continue
        name = doc.get("name") or path.parent.name
        tiers = doc.get("required_oracle_tiers")
        if not isinstance(tiers, list) or not tiers:
            continue
        bar = deepest_tier(str(t) for t in tiers)
        if not bar:
            continue
        # A name declared twice (a hidden mirror of a public capsule) keeps the DEEPEST bar seen:
        # crediting the shallower one would certify a capsule against a weaker demand than some
        # copy of it makes.
        prior = out.get(str(name))
        if prior is None or tier_rank(bar) > tier_rank(prior):
            out[str(name)] = bar
    if not out:
        return {}, f"no capsule under {corpus_dir} declares required_oracle_tiers ({seen} read)"
    return out, ""


@dataclass
class CapsuleCoverage:
    """What the whole archive knows about one capsule."""

    name: str
    lane: str = ""
    placement: str = ON_ROSTER
    #: Which target's roster declares it, when this target's does not.
    owner: str = ""
    #: This capsule's own declared bar, and where that bar came from.
    bar: str = ""
    bar_source: str = ""
    grades: int = 0
    passes: int = 0
    tiers_passed: set[str] = field(default_factory=set)
    planes: Counter = field(default_factory=Counter)
    categories: Counter = field(default_factory=Counter)
    arms: set[str] = field(default_factory=set)
    runs: set[str] = field(default_factory=set)
    #: Tier labels that carried no readable ordinal. Kept so they cannot pass as tier zero.
    unreadable_tiers: set[str] = field(default_factory=set)

    @property
    def ever_passed(self) -> bool:
        """An overall pass was recorded at least once."""
        return self.passes > 0

    @property
    def best_tier(self) -> str:
        return deepest_tier(self.tiers_passed)

    @property
    def best_tier_rank(self) -> int:
        """Deepest tier ever passed, -1 if none was."""
        return tier_rank(self.best_tier) if self.best_tier else -1

    @property
    def bar_rank(self) -> int:
        return tier_rank(self.bar) if self.bar else -1

    @property
    def depth(self) -> str:
        """Which coverage band this capsule falls in, judged against its OWN bar."""
        if not self.grades:
            return NEVER_GRADED
        rank = self.best_tier_rank
        if rank < 0 and not self.ever_passed:
            return GRADED_NEVER_PASSED
        bar = self.bar_rank
        if bar < 0:
            return BAR_UNKNOWN
        if rank > bar:
            return PASSED_ABOVE_BAR
        if rank == bar:
            return PASSED_AT_BAR
        return PASSED_BELOW_BAR

    def to_dict(self) -> dict:
        return {"name": self.name, "lane": self.lane, "placement": self.placement,
                "owner": self.owner, "bar": self.bar, "bar_source": self.bar_source,
                "grades": self.grades, "passes": self.passes,
                "ever_passed": self.ever_passed, "best_tier": self.best_tier,
                "tiers_passed": sorted(self.tiers_passed), "depth": self.depth,
                "planes": dict(self.planes.most_common()),
                "categories": dict(self.categories.most_common()),
                "arms": sorted(self.arms), "n_runs": len(self.runs),
                "unreadable_tiers": sorted(self.unreadable_tiers)}


@dataclass
class TargetCoverage:
    """Corpus coverage for one target, with every denominator kept separate."""

    target: str
    default_bar: str = ""
    capsules: dict[str, CapsuleCoverage] = field(default_factory=dict)
    off_roster: dict[str, CapsuleCoverage] = field(default_factory=dict)
    availability: Availability = field(default_factory=Availability)
    n_runs: int = 0
    n_runs_without_rows: int = 0

    def bands(self) -> dict[str, list[str]]:
        """Roster capsule names grouped by coverage depth."""
        out: dict[str, list[str]] = {d: [] for d in DEPTHS}
        for cov in self.capsules.values():
            out[cov.depth].append(cov.name)
        return {k: sorted(v) for k, v in out.items()}

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.bands().items()}

    @property
    def roster_size(self) -> int:
        return len(self.capsules)

    def bars(self) -> Counter:
        """How many roster capsules demand each bar -- the corpus's own difficulty mix."""
        return Counter(c.bar or "undeclared" for c in self.capsules.values())

    def uncertified_planes(self) -> Counter:
        """Where the graded-but-uncertified roster capsules stop -- the 'why' beside 'how much'.

        Spans every graded capsule short of its own bar, NOT only the ones that never passed
        anything. A capsule that clears the cheap tiers every time and never reaches its declared
        certifying tier is the commonest way coverage stalls, and a tally restricted to total
        failures reports zero for exactly that case.

        Counted over CAPSULES, not over grade rows: a capsule re-graded 300 times would otherwise
        outvote every other capsule and the tally would describe how often a run repeated itself.
        Each capsule contributes its own commonest plane, once.
        """
        tally: Counter = Counter()
        for cov in self.capsules.values():
            if not cov.grades or cov.depth in (PASSED_AT_BAR, PASSED_ABOVE_BAR):
                continue
            if cov.planes:
                tally[cov.planes.most_common(1)[0][0]] += 1
            else:
                tally["unrecorded"] += 1
        return tally

    def passed_without_overall(self) -> list[str]:
        """Capsules that passed a tier but never passed overall -- evidence, not a verdict."""
        return sorted(c.name for c in self.capsules.values()
                      if c.best_tier_rank >= 0 and not c.ever_passed)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "default_bar": self.default_bar,
            "roster_size": self.roster_size,
            "n_runs": self.n_runs,
            "n_runs_without_rows": self.n_runs_without_rows,
            "counts": self.counts(),
            "bands": self.bands(),
            "bars": dict(self.bars().most_common()),
            "uncertified_planes": dict(self.uncertified_planes().most_common()),
            "passed_without_overall": self.passed_without_overall(),
            "off_roster": {
                "owned": sorted(c.name for c in self.off_roster.values()
                                if c.placement == OFF_ROSTER_OWNED),
                "unowned": sorted(c.name for c in self.off_roster.values()
                                  if c.placement == OFF_ROSTER_UNOWNED),
                "owners": {c.name: c.owner for c in sorted(self.off_roster.values(),
                                                           key=lambda x: x.name) if c.owner},
                "grades": {c.name: c.grades for c in sorted(self.off_roster.values(),
                                                            key=lambda x: x.name)},
            },
            "capsules": [c.to_dict() for c in sorted(self.capsules.values(),
                                                     key=lambda c: c.name)],
            "availability": self.availability.to_dict(),
        }


def build(target: str, *, roster: Mapping[str, str],
          runs: Iterable[tuple[str, str, Path]],
          required_tiers: Mapping[str, str] | None = None,
          default_bar: str = "",
          owners: Mapping[str, str] | None = None,
          roster_reason: str = "") -> TargetCoverage:
    """Accumulate corpus coverage for one target.

    ``runs`` is ``(run_id, arm, path)`` -- the arm arrives already resolved, because an arm IS its
    grant set and re-deriving it from a directory name gets arms 3 and 4 wrong (they share a
    subtree). ``required_tiers`` maps a capsule to its own declared bar; ``default_bar`` is the
    target-wide fallback for capsules that declare none. ``owners`` maps a capsule name to the
    target whose roster declares it, so an off-roster capsule can be attributed rather than merely
    excluded.
    """
    cov = TargetCoverage(target=target, default_bar=default_bar)
    required_tiers = required_tiers or {}
    owners = owners or {}

    fallback = 0
    for name, lane in roster.items():
        bar = required_tiers.get(name, "")
        source = "capsule"
        if not bar:
            bar, source = default_bar, ("profile_default" if default_bar else "")
            fallback += 1
        cov.capsules[name] = CapsuleCoverage(name=name, lane=lane, placement=ON_ROSTER,
                                             bar=bar, bar_source=source)
    cov.availability.set("roster", measured("capsule_profiles") if roster
                         else unavailable(roster_reason or f"no roster for {target!r}"))
    if not roster:
        cov.availability.set("bar", unavailable("no roster, so no capsule has a bar"))
    elif fallback and default_bar:
        cov.availability.set("bar", derived(
            f"{fallback} of {len(roster)} roster capsule(s) declare no required_oracle_tiers and "
            f"fall back to the profile's {default_bar}", source="capsule_yaml+profile"))
    elif fallback:
        cov.availability.set("bar", unavailable(
            f"{fallback} of {len(roster)} roster capsule(s) declare no required_oracle_tiers and "
            f"the profile declares no default, so their certification cannot be judged"))
    else:
        cov.availability.set("bar", measured("capsule_yaml"))

    no_rows: list[str] = []
    for run_id, arm, path in runs:
        cov.n_runs += 1
        rows, why = observations(Path(path))
        if not rows:
            cov.n_runs_without_rows += 1
            no_rows.append(why)
            continue
        for obs in rows:
            rec = cov.capsules.get(obs.capsule)
            if rec is None:
                rec = cov.off_roster.get(obs.capsule)
                if rec is None:
                    owner = owners.get(obs.capsule, "")
                    rec = CapsuleCoverage(
                        name=obs.capsule, owner=owner,
                        bar=required_tiers.get(obs.capsule, ""), bar_source="capsule",
                        placement=OFF_ROSTER_OWNED if owner else OFF_ROSTER_UNOWNED)
                    cov.off_roster[obs.capsule] = rec
            rec.grades += 1
            rec.arms.add(arm)
            rec.runs.add(run_id)
            if obs.label and not rec.lane:
                rec.lane = obs.label
            if obs.status == "pass":
                rec.passes += 1
            for tier, outcome in obs.tiers.items():
                if outcome != "pass":
                    continue
                rec.tiers_passed.add(tier)
                if tier_rank(tier) < 0:
                    rec.unreadable_tiers.add(tier)
            if obs.failure_plane:
                rec.planes[obs.failure_plane] += 1
            if obs.failure_category:
                rec.categories[obs.failure_category] += 1

    graded = sum(1 for c in cov.capsules.values() if c.grades)
    if not cov.n_runs:
        cov.availability.set("grades", unavailable(f"no run was offered for {target!r}"))
    elif not graded:
        cov.availability.set("grades", unavailable(
            f"none of {cov.n_runs} run(s) recorded a per-capsule row for a roster capsule"
            + (f"; e.g. {no_rows[0]}" if no_rows else "")))
    elif cov.n_runs_without_rows:
        cov.availability.set("grades", derived(
            f"{cov.n_runs_without_rows} of {cov.n_runs} run(s) contributed no per-capsule row"
            + (f"; e.g. {no_rows[0]}" if no_rows else ""), source="qa_history_verdicts"))
    else:
        cov.availability.set("grades", measured("qa_history_verdicts"))

    # The hidden lane is declared in the roster but the verdict history records only the public
    # label, so hidden coverage cannot be read from this source. Say so rather than letting the
    # hidden capsules sit in `never_graded`, where they would read as an untested holdout when the
    # truth is that this reader cannot see them.
    hidden = [n for n, lane in roster.items() if lane == "hidden"]
    hidden_graded = [n for n in hidden if cov.capsules[n].grades]
    if hidden and not hidden_graded:
        cov.availability.set("hidden_lane", unavailable(
            f"{len(hidden)} hidden capsule(s) are declared but the verdict history labels every "
            f"row public, so their coverage is not readable here"))
    elif hidden:
        cov.availability.set("hidden_lane", measured("qa_history_verdicts"))

    return cov


def owner_map(rosters: Mapping[str, Mapping[str, str]], *, exclude: str = "") -> dict[str, str]:
    """``{capsule name: a target whose roster declares it}``, for attributing off-roster grades.

    A name in several rosters resolves to the first target in sorted order, deterministically --
    the point is to say "this belongs to another target's corpus", not to adjudicate between them.
    """
    out: dict[str, str] = {}
    for target in sorted(rosters):
        if target == exclude:
            continue
        for name in rosters[target]:
            out.setdefault(name, target)
    return out
