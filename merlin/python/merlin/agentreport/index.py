"""Walk a set of run roots and say what each run IS, without reading a single number out of it.

WHY THIS IS SEPARATE FROM THE READERS. Deciding *which* runs a report covers, and deciding *what a
run scored*, are different jobs with different failure modes. The index answers identity — target,
bench, phase, arm, driver, model — and it must answer it for every run on disk, including the ones
that produced nothing, because the denominator is part of the result. The readers answer magnitude,
and they are allowed to refuse.

THE ARM IS NOT IN THE DIRECTORY. Arms 3 and 4 are launched into the SAME run-dir subtree
(``merlin_assisted/``), so the directory cannot distinguish them; ``run_manifest.yaml:arm`` records
that same subtree name and is equally useless. Two things can distinguish them -- the input bundle
(which is what an arm actually IS: the set of paths the agent may read) and the run-id prefix the
launcher stamps. They USUALLY agree. When they do not, this records the disagreement instead of
picking a winner quietly: measured on the corpus, ``merlincirct_atlas_operands_v2`` carries the
arm-3 bundle under the arm-4 prefix, and a report that silently resolved that would attribute one
arm's result to another.

The arm vocabulary itself is a PARAMETER (:class:`ArmSpec`), because it is declared by the launcher,
which lives in the experiment tree that library code is not allowed to reach into.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from merlin.agentreport.availability import Availability, derived, measured, unavailable

#: Bench directory name -> which phase of the study it belongs to. The phase split is a real
#: distinction in this repo (see ``merlin.targetgen.phase_policy``): phase 1 generates a functional
#: compiler, phase 2 optimises a FROZEN phase-1 submission for speed.
PHASE1 = "phase1"
PHASE2 = "phase2"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ArmSpec:
    """One rung of the ladder: its id, the run-id prefix the launcher stamps, and its bundle ids."""

    arm_id: str                       # e.g. "arm4"
    name: str                         # e.g. "merlin_rtlchecks"
    prefix: str                       # e.g. "merlincirct"
    bundle_ids: tuple[str, ...] = ()  # every bundle id that means this arm


@dataclass
class RunRef:
    """Identity of one run directory. Carries no measurement -- only what the run IS."""

    root: Path
    path: Path
    target: str
    bench: str
    phase: str
    run_id: str
    arm: str = UNKNOWN
    arm_source: str = ""              # "bundle_id" | "run_id_prefix" | ""
    arm_conflict: str = ""            # set when bundle and prefix disagree; both spellings recorded
    bundle_id: str = ""
    driver: str = ""
    model: str = ""
    provider: str = ""
    started_at: str = ""
    repo_sha: str = ""
    n_rounds: int = 0
    availability: Availability = field(default_factory=Availability)

    @property
    def key(self) -> str:
        """Stable identity across roots. A run copied between roots keeps one key."""
        return f"{self.target}/{self.bench}/{self.run_id}"

    def to_dict(self) -> dict:
        return {"root": str(self.root), "path": str(self.path), "target": self.target,
                "bench": self.bench, "phase": self.phase, "run_id": self.run_id,
                "arm": self.arm, "arm_source": self.arm_source, "arm_conflict": self.arm_conflict,
                "bundle_id": self.bundle_id, "driver": self.driver, "model": self.model,
                "provider": self.provider, "started_at": self.started_at,
                "repo_sha": self.repo_sha, "n_rounds": self.n_rounds,
                "availability": self.availability.to_dict()}


def arm_from_prefix(run_id: str, arms: Sequence[ArmSpec]) -> str:
    """LONGEST prefix wins, and the prefix is matched WITH its separator.

    The separator is what disambiguates today's ladder: ``merlincirct_x`` does not start with
    ``merlin_``, so arm 3 and arm 4 cannot be confused however the specs are ordered. Matching a bare
    ``merlin`` would confuse them, which is why the token is built here rather than by the caller.

    Longest-match is the guard for the case the separator does NOT settle: an arm whose prefix is an
    underscore-extension of another's (``merlin`` and a future ``merlin_rtl``) matches both, and
    without a length comparison the winner would depend on spec order."""
    best: ArmSpec | None = None
    for spec in arms:
        token = spec.prefix + "_"
        if run_id.startswith(token) and (best is None or len(spec.prefix) > len(best.prefix)):
            best = spec
    return best.arm_id if best else UNKNOWN


def arm_from_bundle(bundle_id: str, arms: Sequence[ArmSpec]) -> str:
    if not bundle_id:
        return UNKNOWN
    for spec in arms:
        if bundle_id in spec.bundle_ids:
            return spec.arm_id
    return UNKNOWN


def resolve_arm(run_id: str, bundle_id: str, arms: Sequence[ArmSpec]) -> tuple[str, str, str]:
    """``(arm, source, conflict)``.

    The bundle wins when the two disagree, because an arm IS its bundle -- the set of paths the agent
    may read -- while the prefix is only what the launcher was asked to call the run. But the
    disagreement is RETURNED, not swallowed: a run whose two identifiers disagree is a data-quality
    fact the report has to be able to show."""
    by_bundle = arm_from_bundle(bundle_id, arms)
    by_prefix = arm_from_prefix(run_id, arms)
    if by_bundle != UNKNOWN and by_prefix != UNKNOWN and by_bundle != by_prefix:
        return by_bundle, "bundle_id", f"bundle_id={by_bundle} run_id_prefix={by_prefix}"
    if by_bundle != UNKNOWN:
        return by_bundle, "bundle_id", ""
    if by_prefix != UNKNOWN:
        return by_prefix, "run_id_prefix", ""
    return UNKNOWN, "", ""


def _yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(errors="ignore")) or {}
    except Exception:  # noqa: BLE001 - a malformed side file must not lose the run
        return {}


def _phase_of(bench: str, phase_map: dict[str, str]) -> str:
    return phase_map.get(bench, UNKNOWN)


def _round_transcripts(run: Path) -> list[Path]:
    rounds = run / "rounds"
    if rounds.is_dir():
        found = sorted(rounds.glob("round_*.transcript.jsonl"))
        if found:
            return found
    flat = run / "transcript.jsonl"
    return [flat] if flat.is_file() else []


def iter_run_dirs(root: Path) -> Iterable[tuple[str, str, Path]]:
    """``(target, bench, run_dir)`` for every run under ``root``.

    Run dirs sit at ``<root>/<target>/<bench>/<group>/<run_id>`` for the capsule bench and at
    ``<root>/<target>/<bench>/<group>/<run_id>`` for the perf bench too, so one shape covers both.
    A run is anything holding a ``rounds/`` directory or a flat transcript -- NOT anything holding a
    ``qa_history/``, because grading-only trees have those and were never an agent session."""
    if not root.is_dir():
        return
    for target_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for bench_dir in sorted(p for p in target_dir.iterdir() if p.is_dir()):
            for group_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
                for run in sorted(p for p in group_dir.iterdir() if p.is_dir()):
                    if (run / "rounds").is_dir() or (run / "transcript.jsonl").is_file():
                        yield target_dir.name, bench_dir.name, run


def build_index(roots: Sequence[Path], arms: Sequence[ArmSpec],
                phase_map: dict[str, str]) -> list[RunRef]:
    """Every run under every root, identified. Never refuses a run -- an unidentifiable one is kept
    with ``arm=UNKNOWN`` so it still counts toward the denominator."""
    out: list[RunRef] = []
    for root in roots:
        root = Path(root)
        for target, bench, run in iter_run_dirs(root):
            env = _yaml(run / "environment.yaml")
            bundle = str(env.get("bundle_id") or "")
            if not bundle:
                bundle = str(_yaml(run / "input_bundle_manifest.yaml").get("bundle_id") or "")
            arm, source, conflict = resolve_arm(run.name, bundle, arms)
            transcripts = _round_transcripts(run)
            ref = RunRef(root=root, path=run, target=target, bench=bench,
                         phase=_phase_of(bench, phase_map), run_id=run.name,
                         arm=arm, arm_source=source, arm_conflict=conflict, bundle_id=bundle,
                         driver=str(env.get("driver") or ""), model=str(env.get("model") or ""),
                         provider=str(env.get("provider") or ""),
                         started_at=str(env.get("started_at") or ""),
                         repo_sha=str(env.get("repo_sha") or ""),
                         n_rounds=len(transcripts))
            if arm == UNKNOWN:
                ref.availability.set("arm", unavailable(
                    f"run id {run.name!r} matches no arm prefix and "
                    f"{'bundle_id ' + bundle if bundle else 'no bundle_id was recorded'}"))
            elif conflict:
                ref.availability.set("arm", derived(
                    f"bundle and prefix disagree ({conflict}); the bundle decides because an arm is "
                    f"its grant set", source=source))
            else:
                ref.availability.set("arm", measured(source))
            ref.availability.set("transcripts", measured("rounds") if transcripts
                                 else unavailable("run has no round transcript and no flat transcript"))
            out.append(ref)
    return out


def write_index(refs: Sequence[RunRef], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in refs], indent=2) + "\n")
    return path
