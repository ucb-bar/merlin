"""Tier A of the phase-2 fast feedback: the whole-model program on the FUNCTIONAL simulator, as a screen.

    screen = structure_screen(elf, groups=..., target=..., out=...)   # ~1.5 min for a ResNet
    calibration = fit_calibration(collect_pairs([store]))             # refit from every measured pair

The functional simulator runs the SAME ELF the board runs, in about a minute and a half, and every
group's own local check runs with it. So it answers "is every group still correct" at once. It also
prints a per-group cycle count, and that number is where this module is careful: measured on 142 group
pairs (the same ELFs on the functional simulator and on the board), the functional simulator ranks
whole models correctly but is BLIND TO DATA MOVEMENT -- the board/simulator ratio is ~4x for a
convolution, ~6x for a matmul (up to 122x), ~89x for a residual add, and of the groups whose board
cycles moved by more than 5% it agreed on direction for barely a third.

So the screen's cycles are labelled a STRUCTURE SCREEN and never feed an objective:

* **Correctness** per group is the program's own local line (``GM_LOCAL`` for an exact group,
  ``GM_BOUND`` for a tolerance group); a group with neither is ``absent``, and a console missing any
  expected group line is refused outright.
* **Calibration** per kind is REFIT from every (functional, board) pair the measurement store has
  accumulated -- never a constant: :func:`collect_pairs` walks the store for jobs that carry both a
  board result and a screen of the same ELF, and :func:`fit_calibration` states, per kind and per
  group, the median board/simulator ratio, its spread, and the pair count.
* **``spike_blind``** marks a group whose kind (or whose own measured ratio) sits above
  ``blind_ratio`` -- a memory-bound group, whose functional cycles say nothing about its board cycles.
  No per-group board estimate is stated: a group's own ratio depends on what it spends its time
  on (an operand gather on the core runs at ~1.3x, the kernel beside it at ~5x), so the class's
  ratio and spread are given instead, for a reader to weigh.

Nothing here names a target: the simulator is reached through the target's backend.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "LABEL",
    "SCHEMA",
    "ScreenRefusal",
    "VALIDATED",
    "UNVALIDATED",
    "collect_pairs",
    "fit_calibration",
    "minimum_decided_pairs",
    "minimum_rank_rate",
    "validate_calibration",
    "pairs_from_consoles",
    "routes_of",
    "screen_console",
    "screen_dir",
    "structure_screen",
]

SCHEMA = "whole_model_structure_screen_v1"
LABEL = "STRUCTURE SCREEN (functional simulator): correctness and compute structure only; never an objective"
SCREEN_FILE = "structure_screen.json"
#: Above this board/simulator ratio a kind's (or group's) functional cycles are blind to its board cost.
BLIND_RATIO = 20.0
VALIDATION_SCHEMA = "whole_model_screen_calibration_validation_v1"
VALIDATED, UNVALIDATED = "validated", "unvalidated"


class ScreenRefusal(ValueError):
    """The console cannot be read as a whole-model screen, and the message says why."""


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -------------------------------------------------------------------------------------- the screen


def screen_console(
    text: str,
    groups: Mapping[str, str],
    *,
    templates: Mapping[str, str] | None = None,
    calibration: Mapping[str, Any] | None = None,
    blind_ratio: float = BLIND_RATIO,
    routes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read a functional-simulator console against the groups the build states (``{group: compare}``).

    ``routes`` (``{group: who answered it}``, :func:`routes_of`) selects the calibration of the
    group's kind UNDER ITS ROUTE when the fit has one; the kind alone is the fallback.

    Refused (``status: refused``) when any expected group line is missing: a partial console has no
    per-group reading. Otherwise every group gets its simulator cycles, its local correctness and, from
    ``calibration``, whether the simulator is blind to it.
    """
    from . import whole_model_verdict as V

    try:
        parsed = V.parse_log(text, templates)
    except V.VerdictRefusal as why:
        return {"schema": SCHEMA, "label": LABEL, "status": "refused", "refusal": str(why)}
    expected, printed = set(map(str, groups)), set(parsed.groups)
    if printed != expected:
        return {
            "schema": SCHEMA,
            "label": LABEL,
            "status": "refused",
            "refusal": (
                f"the console printed {len(printed & expected)} of the {len(expected)} expected group lines "
                f"(missing {sorted(expected - printed, key=int)[:12]}); a partial run is not read"
            ),
        }
    kinds = (calibration or {}).get("kinds") or {}
    per_group = (calibration or {}).get("groups") or {}
    rows = []
    for group in sorted(expected, key=int):
        line = parsed.groups[group]
        if groups[group] in ("bounded", "bounded_int"):
            got = parsed.bounds.get(group)
            state = "absent" if got is None else ("correct" if got[1] == 0 else "wrong")
            detail = None if got is None else {"max_abs": got[0], "over": got[1], "bound": got[2]}
        else:
            got = parsed.local.get(group)
            state = "absent" if got is None else ("correct" if got[0] == 0 else "wrong")
            detail = None if got is None else {"mismatches": got[0], "of": got[1]}
        row: dict[str, Any] = {"group": int(group), "kind": line.kind, "spike_cycles": line.cycles, "local": state}
        if routes:
            row["on"] = routes.get(group)
        if detail:
            # The check's own numbers, kept for every group: an exactness contract grades from them.
            row["check"] = detail
        if detail and state != "correct":
            row["failure"] = detail
        on = (routes or {}).get(group)
        classes = (calibration or {}).get("classes") or {}
        fit_kind = classes.get(_class(line.kind, on)) if on else None
        basis = f"{line.kind!r} answered by {on}" if fit_kind else f"kind {line.kind!r}"
        fit_kind = fit_kind or kinds.get(line.kind)
        fit_group = (per_group.get(_class(group, on)) if on else None) or per_group.get(group)
        reasons = []
        if fit_kind and fit_kind.get("spike_blind"):
            reasons.append(f"{basis}: board/simulator median {fit_kind['median_ratio']}x")
        if fit_group and fit_group["median_ratio"] > blind_ratio:
            reasons.append(f"this group: board/simulator median {fit_group['median_ratio']}x")
        row["spike_blind"] = bool(reasons) if (fit_kind or fit_group) else None
        if reasons:
            row["blind_because"] = reasons
        elif fit_kind:
            # The class's ratio and spread, NOT a per-group estimate: measured on the same program, a
            # group whose time is an operand gather on the core sits near 1.3x while its class sits
            # near 4.6x, so a multiplied-out "board estimate" would be a number nobody measured.
            row["class_ratio"] = {k: fit_kind[k] for k in ("median_ratio", "p10", "p90", "n")}
        rows.append(row)
    wrong = [r["group"] for r in rows if r["local"] != "correct"]
    # THE SCREEN'S RANKING (its class ratios as a board estimate) stands only on a calibration whose refit
    # was validated against held-out board readings; anything else is recorded as unvalidated, and why.
    validation = (calibration or {}).get("validation") or {}
    ranking = {
        "status": VALIDATED if validation.get("status") == VALIDATED else UNVALIDATED,
        "reasons": list(validation.get("reasons") or ())
        or ([] if validation.get("status") == VALIDATED else ["no validated calibration was refit for this screen"]),
    }
    return {
        "schema": SCHEMA,
        "label": LABEL,
        "feeds_objective": False,
        "status": "screened",
        "all_groups_correct": not wrong,
        "groups_not_correct": wrong,
        "whole_window_spike_cycles": parsed.whole_window_cycles,
        "argmax": list(parsed.argmax) if parsed.argmax else None,
        # A model graded on its output tensor states ``(within, of)`` instead of a class.
        "output": list(parsed.output) if parsed.output else None,
        # The whole output's digest and every dispatch's result digest: what a reference arm compares.
        "output_digest": list(parsed.output_digest) if parsed.output_digest else None,
        "words": {g: list(v) for g, v in parsed.words.items()},
        "calibration": {
            "pairs": (calibration or {}).get("pairs"),
            "sources": (calibration or {}).get("sources"),
            "blind_ratio": blind_ratio,
            "validation": validation or None,
        }
        if calibration
        else None,
        "ranking": ranking,
        "groups": rows,
    }


def structure_screen(
    elf: str | Path,
    *,
    groups: Mapping[str, str],
    target: str,
    out: str | Path,
    elf_sha256: str | None = None,
    templates: Mapping[str, str] | None = None,
    simulator: str = "spike",
    timeout: int = 4 * 3600,
    calibration: Mapping[str, Any] | None = None,
    routes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``elf`` on the target's functional simulator and screen it (see :func:`screen_console`).

    The console and the screen are written to ``out`` (``console.txt``, ``structure_screen.json``); put
    ``out`` beside a board result of the same ELF and :func:`collect_pairs` will learn from it.
    """
    from merlin.runtime.backends import base as backends

    elf, out = Path(elf), Path(out)
    observed = _sha256(elf)
    if elf_sha256 and observed != elf_sha256:
        raise ScreenRefusal(f"{elf} is {observed[:12]}, not the {elf_sha256[:12]} named for the screen")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    console = backends.get_backend(target).run_elf(elf, simulator=simulator, timeout=timeout)
    wall = round(time.monotonic() - started, 1)
    (out / "console.txt").write_text(console, encoding="utf-8")
    screen = screen_console(console, groups, templates=templates, calibration=calibration, routes=routes)
    screen.update({"elf_sha256": observed, "simulator": simulator, "wall_s": wall, "target": target})
    (out / SCREEN_FILE).write_text(json.dumps(screen, indent=1) + "\n", encoding="utf-8")
    return screen


# ------------------------------------------------------------------------------------- calibration


def _group_cycles(text: str, templates: Mapping[str, str] | None = None) -> dict[str, tuple[str, int]]:
    from . import whole_model_verdict as V

    parsed = V.parse_log(text, templates)
    return {g: (line.kind, line.cycles) for g, line in parsed.groups.items()}


#: Older service records spell the package's side ``submission``; it is the same side.
_SIDE = {"submission": "package"}


def routes_of(record: Mapping[str, Any]) -> dict[str, str]:
    """``{group: route}`` from a build record (the whole-model build's or the service's).

    A route is who answered the group and, for a kernel, the command shape it lowered to
    (``package:resident_matmul``): a kernel that hands its loop to the hardware sequencer costs the
    functional simulator almost nothing, so two kernels of one kind can differ by 100x in how blind
    the simulator is to them, and the shape is what tells them apart.
    """
    rows = (record.get("attribution") or {}).get("per_group") or record.get("groups") or ()
    routes = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("group") is None:
            continue
        side = _SIDE.get(str(row.get("on")), str(row.get("on")))
        shape = row.get("lowering") or row.get("shape")
        routes[str(row["group"])] = f"{side}:{shape}" if shape and side == "package" else side
    return routes


def pairs_from_consoles(
    spike_console: str,
    board_console: str,
    *,
    source: str,
    templates: Mapping[str, str] | None = None,
    routes: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Per-group (kind, route, simulator cycles, board cycles) for two consoles of the SAME ELF."""
    spike, board = _group_cycles(spike_console, templates), _group_cycles(board_console, templates)
    if set(spike) != set(board):
        raise ScreenRefusal(f"{source}: the two consoles print different group sets")
    return [
        {
            "source": source,
            "group": g,
            "kind": spike[g][0],
            "on": (routes or {}).get(g),
            "spike": spike[g][1],
            "board": board[g][1],
        }
        for g in sorted(spike, key=int)
        if spike[g][1] > 0 and board[g][1] > 0
    ]


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def screen_dir(store_base: str | Path, elf_sha256: str) -> Path:
    """Where the store keeps the screen of one ELF: by the program's content, beside the jobs."""
    return Path(store_base) / "_structure_screens" / str(elf_sha256)


def collect_pairs(store_bases: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Every (simulator, board) group pair the measurement store holds for one ELF measured on both.

    A board result (a job's ``result.json`` whose device is on the board rung and whose run completed)
    counts when the store also holds a screen of the ELF it names, under
    ``<base>/_structure_screens/<elf sha256>/`` (:func:`screen_dir`) -- keyed by the program's bytes, so
    a screen is written once per ELF and never into another run's job directory. Each ELF counts once,
    however many jobs measured it.
    """
    seen: set[str] = set()
    pairs: list[dict[str, Any]] = []
    for base in store_bases:
        for result_path in sorted(Path(base).glob("*/*/result.json")):
            job = result_path.parent
            elf_named = str((_read(result_path).get("build") or {}).get("elf_sha256") or "")
            if not elf_named:
                continue
            screen_path = screen_dir(base, elf_named) / SCREEN_FILE
            if not screen_path.is_file():
                continue
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                screen = json.loads(screen_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            elf = (result.get("build") or {}).get("elf_sha256")
            run = result.get("run") or {}
            if not elf or elf != screen.get("elf_sha256") or elf in seen or not run.get("completed"):
                continue
            if (result.get("device") or {}).get("rung") != "fpga_firesim":
                continue
            uart = Path(str(run.get("uart_log") or ""))
            console = screen_path.parent / "console.txt"
            if not uart.is_file() or not console.is_file():
                continue
            try:
                found = pairs_from_consoles(
                    console.read_text(encoding="utf-8", errors="replace"),
                    uart.read_text(encoding="utf-8", errors="replace"),
                    source=f"{job.parent.name}/{job.name[:16]} elf {elf[:12]}",
                    routes=routes_of(result.get("build") or {}),
                )
            except (ScreenRefusal, ValueError):
                continue
            # What a held-out validation binds each pair to: the executable, the measurement domain (the
            # board's own identity, as the store recorded it) and the bytes the two readings came from.
            binding = {
                "elf_sha256": str(elf),
                "domain": _domain(result.get("device")),
                "evidence_sha256s": [_sha256(result_path), _sha256(uart), _sha256(console)],
            }
            pairs += [{**pair, **binding} for pair in found]
            seen.add(elf)
    return pairs


def _domain(device: Any) -> dict[str, Any] | None:
    """The measurement domain a board reading belongs to: its target, rung and the board's own binary
    digest -- the identity the store's noise derivation keys a machine by -- or None when unrecorded."""
    if not isinstance(device, Mapping) or not device.get("binary_sha256"):
        return None
    return {key: device.get(key) for key in ("target", "rung", "binary_sha256")}


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def fit_calibration(
    pairs: Sequence[Mapping[str, Any]],
    *,
    blind_ratio: float = BLIND_RATIO,
    margin: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Per kind and per group: the board/simulator ratio's median, spread and pair count; the blind flag.

    Also the log-log correlation of the two instruments over every pair, which says how much a
    simulator cycle count tells about a board one at all.

    Every refit is validated (:func:`validate_calibration`) against held-out board-measured groups under
    the machine's derived noise ``margin`` (the measured mode's :func:`noise.margin` document), and the
    result is recorded as ``validation``; with no margin the screen's ranking is ``unvalidated``.
    ``validate`` is off only for the per-fold fits the validation itself makes.
    """
    by_kind: dict[str, list[float]] = {}
    by_class: dict[str, list[float]] = {}
    by_group: dict[str, list[float]] = {}
    xs, ys = [], []
    for pair in pairs:
        ratio = float(pair["board"]) / float(pair["spike"])
        by_kind.setdefault(str(pair["kind"]), []).append(ratio)
        if pair.get("on"):
            by_class.setdefault(_class(pair["kind"], pair["on"]), []).append(ratio)
            by_group.setdefault(_class(pair["group"], pair["on"]), []).append(ratio)
        by_group.setdefault(str(pair["group"]), []).append(ratio)
        xs.append(math.log(float(pair["spike"])))
        ys.append(math.log(float(pair["board"])))

    def summary(ratios: list[float]) -> dict[str, Any]:
        median = statistics.median(ratios)
        return {
            "n": len(ratios),
            "median_ratio": round(median, 3),
            "p10": round(_quantile(ratios, 0.1), 3),
            "p90": round(_quantile(ratios, 0.9), 3),
            "max_ratio": round(max(ratios), 3),
            "spike_blind": median > blind_ratio,
        }

    correlation = None
    if len(xs) >= 3:
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
        sxx = sum((a - mx) ** 2 for a in xs) ** 0.5
        syy = sum((b - my) ** 2 for b in ys) ** 0.5
        correlation = round(sxy / (sxx * syy), 4) if sxx and syy else None
    return {
        "pairs": len(pairs),
        "sources": sorted({str(p.get("source")) for p in pairs}),
        "blind_ratio": blind_ratio,
        "log_log_r": correlation,
        "kinds": {kind: summary(r) for kind, r in sorted(by_kind.items())},
        # THE ROUTE IS PART OF THE KEY. A group the library answers with a sequenced loop is almost
        # free on the functional simulator (the sequencer's work is not an instruction it counts),
        # so one kind can sit at 5x under one route and at 2000x under another; a per-kind ratio
        # averaged over both describes neither.
        "classes": {name: summary(r) for name, r in sorted(by_class.items())},
        "groups": {group: summary(r) for group, r in sorted(by_group.items())},
        **({"validation": validate_calibration(pairs, margin=margin)} if validate else {}),
    }


def _class(kind_or_group: Any, on: Any) -> str:
    return f"{kind_or_group}@{on}"


# ------------------------------------------------------------------------------------ validation
#
# A refit calibration is a fast estimate of board cycles: a group's functional-simulator cycles times
# the board/simulator ratio band of its class. It is only worth exposing if it predicts board readings
# it was not fitted on, and orders candidates the way the board did. Every threshold below is derived:
#
# * the error bound is the machine's own noise margin (a prediction that misses by more than the
#   machine's run-to-run spread is not a prediction at the machine's resolution);
# * every held-out reading the store holds must be predicted (none may be UNKNOWN);
# * the statistical minimum is the smallest count of decided pairs at which a scorer at chance (each
#   pair a fair coin) could agree on all of them with probability no larger than that margin: below it
#   no agreement rate is evidence, so the validation is undeterminable and fails closed;
# * the agreement rate must be one a fair coin would reach on the store's own measured order (its
#   within-workload pairs of board readings) with probability no larger than the margin.


def minimum_decided_pairs(margin: float) -> int:
    """The fewest decided pairs on which unanimous agreement is evidence at significance ``margin``: the
    smallest ``n`` with ``CHANCE ** n <= margin`` for a fair coin."""
    from .rank_validation import CHANCE

    if not math.isfinite(margin) or not 0 < margin < 1:
        raise ValueError("a significance margin must lie strictly between 0 and 1")
    return max(1, math.ceil(math.log(margin) / math.log(CHANCE)))


def minimum_rank_rate(pairs: int, margin: float) -> float:
    """The smallest agreement rate ``k / pairs`` a fair coin reaches with probability at most ``margin``
    (the upper binomial tail), on ``pairs`` measured within-workload pairs."""
    from .rank_validation import CHANCE

    if pairs < minimum_decided_pairs(margin):
        raise ValueError(f"{pairs} measured pair(s) cannot show agreement beyond chance at {margin}")
    log_p = pairs * math.log(CHANCE)  # P(X = pairs): every pair agreed
    tail, k = 0.0, pairs
    while k > 0:
        mass = math.exp(log_p)
        if tail + mass > margin:
            break
        tail += mass
        # P(X = k - 1) from P(X = k) for a fair coin: times k / (pairs - k + 1)
        log_p += math.log(k) - math.log(pairs - k + 1)
        k -= 1
    return (k + 1) / pairs


class _RatioScreen:
    """The calibration's own prediction for a held-out reading: its functional cycles times the board/
    simulator band ``[p10, p90]`` of its kind under its route (else of its kind), as :func:`screen_console`
    selects it."""

    def __init__(self, calibration: Mapping[str, Any], domain_sha256: str) -> None:
        self.calibration, self.domain_sha256 = calibration, domain_sha256

    def predict(self, features: Mapping[str, float | None], *, domain_sha256: str):
        from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

        if domain_sha256 != self.domain_sha256:
            return CycleInterval.unknown("the reading is from another measurement domain than the fit")
        if len(features) != 1:
            return CycleInterval.unknown("a held-out reading names exactly one group's simulator cycles")
        ((pointer, spike),) = features.items()
        kind, on = _parse_pointer(pointer)
        if spike is None or kind is None:
            return CycleInterval.unknown(f"feature {pointer} names no kind's simulator cycles")
        classes, kinds = self.calibration.get("classes") or {}, self.calibration.get("kinds") or {}
        fit = (classes.get(_class(kind, on)) if on else None) or kinds.get(kind)
        if not fit:
            return CycleInterval.unknown(f"no training pair of kind {kind!r}")
        return CycleInterval(
            float(fit["p10"]) * float(spike),
            float(fit["p90"]) * float(spike),
            provenance=(f"board/simulator band [p10, p90] over {fit['n']} training pair(s)",),
        )


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _pointer(kind: str, on: str | None) -> str:
    """The JSON pointer (RFC 6901) of a reading's simulator cycles: its kind, and its route when known."""
    route = f"/routes/{_escape(on)}" if on else ""
    return f"/kinds/{_escape(kind)}{route}/spike_cycles"


def _parse_pointer(pointer: str) -> tuple[str | None, str | None]:
    tokens = [t.replace("~1", "/").replace("~0", "~") for t in pointer.split("/")[1:]]
    if len(tokens) == 3 and tokens[0] == "kinds" and tokens[2] == "spike_cycles":
        return tokens[1], None
    if len(tokens) == 5 and tokens[0] == "kinds" and tokens[2] == "routes" and tokens[4] == "spike_cycles":
        return tokens[1], tokens[3]
    return None, None


def _slices_required() -> int:
    """How many held-out workloads must carry evidence: the existing schedule-rank gate's own minimum (a
    rate measured on one slice says nothing about another)."""
    import inspect

    from . import rank_validation as R

    return int(inspect.signature(R.verdict).parameters["minimum_slices"].default)


def validate_calibration(pairs: Sequence[Mapping[str, Any]], *, margin: Mapping[str, Any] | None) -> dict[str, Any]:
    """Hold out each board-measured group of the store in turn, refit the calibration on the others, and
    predict it (:func:`merlin.perf.fast_estimate_validation.cross_validate`). ``status`` is
    :data:`VALIDATED` only when every derived threshold holds; otherwise :data:`UNVALIDATED`, with why."""
    from merlin.common.digest import is_sha256
    from merlin.common.jsonio import canonical_sha256

    from . import fast_estimate_validation as FV
    from . import rank_validation as R

    def unvalidated(*reasons: str, **known: Any) -> dict[str, Any]:
        return {"schema": VALIDATION_SCHEMA, "status": UNVALIDATED, "reasons": list(reasons), **known}

    value = (margin or {}).get("margin")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) < 1:
        return unvalidated("no machine noise margin was derived for this store, so no error bound exists")
    margin_value = float(value)
    record: dict[str, Any] = {"margin": dict(margin or {})}
    observations: list[Any] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    unbound = 0
    for pair in pairs:
        domain, elf = pair.get("domain"), str(pair.get("elf_sha256") or "")
        evidence = tuple(pair.get("evidence_sha256s") or ())
        if not isinstance(domain, Mapping) or not is_sha256(elf) or not evidence:
            unbound += 1
            continue
        observation = FV.Observation(
            program=elf,
            workload=canonical_sha256(["whole_model_group", str(pair["group"]), str(pair["kind"])]),
            group=str(pair["group"]),
            domain=canonical_sha256(dict(domain)),
            features={_pointer(str(pair["kind"]), str(pair["on"]) if pair.get("on") else None): float(pair["spike"])},
            cycles=float(pair["board"]),
            evidence_sha256s=evidence,
        )
        observations.append(observation)
        by_id[observation.id] = pair
    record.update(observations=len(observations), unbound_pairs=unbound)
    if unbound:
        # The fit uses every pair; one the validation cannot bind is data nothing checked.
        return unvalidated(f"{unbound} pair(s) carry no executable, domain or evidence binding", **record)
    if len({o.domain for o in observations}) > 1:
        return unvalidated("the store's pairs span more than one measurement domain", **record)
    needed = minimum_decided_pairs(margin_value)
    measured_order = len(R.ordered_pairs([R.Program(o.workload, o.id, o.cycles, o.group) for o in observations]))
    record["measured_order_pairs"] = measured_order
    if measured_order < needed:
        return unvalidated(
            f"the store's measured order holds {measured_order} within-workload pair(s); at least {needed} are "
            f"needed before agreement at the machine's margin {margin_value:.6g} is evidence",
            **record,
        )
    domain_sha256 = observations[0].domain

    def fit(train):
        return _RatioScreen(fit_calibration([by_id[o.id] for o in train], validate=False), domain_sha256)

    def run(rate: float, basis: str) -> tuple[dict[str, Any], dict[str, Any]]:
        thresholds = {
            "maximum_relative_error": margin_value,
            "maximum_relative_error_basis": f"the machine's noise margin ({(margin or {}).get('basis')})",
            "minimum_predictions": len(observations),
            "minimum_predictions_basis": "every held-out board reading the store holds",
            "minimum_rank_rate": rate,
            "minimum_rank_rate_basis": basis,
            "minimum_decided": needed,
            "minimum_slice_decided": needed,
            "minimum_decided_basis": "the fewest decided pairs a fair coin agrees on unanimously with probability "
            "at most the margin",
            "minimum_slices": _slices_required(),
        }
        result = FV.cross_validate(
            observations,
            fit,
            maximum_relative_error=thresholds["maximum_relative_error"],
            minimum_predictions=thresholds["minimum_predictions"],
            minimum_rank_rate=thresholds["minimum_rank_rate"],
            minimum_decided=thresholds["minimum_decided"],
            minimum_slice_decided=thresholds["minimum_slice_decided"],
            minimum_slices=thresholds["minimum_slices"],
        )
        return result, thresholds

    validation, thresholds = run(
        minimum_rank_rate(measured_order, margin_value),
        f"agreement a fair coin reaches on the store's {measured_order} measured pair(s) with probability at most "
        "the margin",
    )
    decided = int(validation["ranking"]["overall"]["decided"])
    if decided >= needed and decided < measured_order:
        # The rate is a rate over the pairs the screen decided; hold it to the significance of that count.
        validation, thresholds = run(
            minimum_rank_rate(decided, margin_value),
            f"agreement a fair coin reaches on the {decided} pair(s) the screen decided with probability at most "
            "the margin",
        )
    collisions = [
        row
        for pointer in sorted({next(iter(o.features)) for o in observations})
        for row in FV.feature_collisions([o for o in observations if pointer in o.features], [pointer])
    ]
    record.update(
        thresholds=thresholds,
        absolute_error=validation["absolute_error"],
        ranking=validation["ranking"],
        predictions=len(validation["predictions"]),
        unresolved_predictions=sum(1 for row in validation["predictions"] if not row["prediction"]["resolved"]),
        collisions=len(collisions),
        coefficient_scope=validation["coefficient_scope"],
    )
    if validation["exposable"]:
        return {"schema": VALIDATION_SCHEMA, "status": VALIDATED, "reasons": [], **record}
    return unvalidated(*validation["reasons"], **record)
