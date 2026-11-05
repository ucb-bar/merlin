"""The memory-order tier: does a certified program depend on memory answering in issue order?

WHY. The cert tier runs the elaborated RTL against a memory model that answers every request in the
order it was accepted. A program that writes the same accumulator rows with two loads and nothing
ordering their COMPLETION (only their issue) is correct on that model and wrong on a board whose memory
system reorders independent requests. Measured: an overwrite-then-accumulate residual add passed L2 and
L3, then on FireSim returned the first operand alone in the failing elements. A fence between the two
loads made it pass on the same board, and so did the vendor library's own residual add once fenced.

OPT-IN. The tier costs one in-order run plus N seeded runs of the memory-perturbing engine per capsule it
fires on, so it runs only when ``MERLIN_L3_ORDER_SEEDS`` names a positive seed count. Unset (or 0), a
grade is byte-identical to one from before this tier existed and no record is written.

WHAT IT DOES, both steps fail-closed:

1. A STATIC trigger. When the grade reached a completion-order verdict over the executed command
   stream (:mod:`merlin.targetgen.completion_order`), that verdict is the trigger. Otherwise, the
   accumulator-only check over the decoded instruction trace (:func:`static_check`): two inbound transfers whose
   destinations may overlap in the accumulator, with no barrier between them, are a hazard. A transfer
   whose direction or destination the decoder could not resolve, and a command whose memory effects
   this check does not model (a hardware loop unroller issues its own loads), count as UNRESOLVED.
   Unresolved triggers the sweep exactly as a proven hazard does: "we could not tell" is never "clear".
2. When the static check is not clear, the executable is re-run on the target's memory-perturbing RTL
   engine (:mod:`merlin.targetgen.mem_perturb`), in order and under N seeds, and every run is judged
   against the capsule's own golden. Any run that answers wrong under a legal schedule FAILS the
   certificate, and the record names the seeds that reproduce it.

WHERE THE FACTS COME FROM. The accumulator and accumulate bits of a local address come from the target's
derived ISA constants (:func:`merlin.targetgen.rocc.decode.isa_constants`, owned by the selected
backend) and the block width of a transfer from the derived array geometry
(:func:`merlin.perf.derived_bound.machine_from_facts`). Which commands move data, and in which direction,
is read from the decoded payload's own fields (the shared decoded-ABI field vocabulary of
:mod:`merlin.perf.deps.rocc`); which command is a barrier comes from the shared class-family table
(:mod:`merlin.targetgen.semantic_families`). A target without those facts is recorded as not applicable,
never checked against guessed ones.

WHAT A VERDICT MEANS. ``order_sensitive`` is a statement about the PROGRAM under the ordering freedom the
bus protocol grants, not a claim that a particular chip reorders those two requests. ``order_stable`` is
evidence, not proof: N seeds sample the schedule space, they do not cover it.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Which decoded-ABI classes issue NO memory request of their own -- configuration, cache maintenance,
#: and the preload/compute steps whose operands are already on chip -- is read from the shared class
#: vocabulary every capability manifest maps its RTL funct codes onto
#: (:func:`merlin.targetgen.semantic_families.local_only_classes`), never re-listed here and never a fact
#: about one device. A class outside that set that carries no transfer evidence is OPAQUE -- it may issue
#: loads the trace cannot see -- and is treated as an unresolved load.

#: The shared class family that orders memory (a host fence drains the accelerator's queues).
BARRIER_FAMILY = "synchronization"
#: The shared class family of a data-motion command.
MOVEMENT_FAMILY = "movement"
#: The decoded payload field naming a transfer's off-chip operand.
OFF_CHIP_FIELD = "dram"

#: The pin-registry role of a memory-perturbing elaborated-RTL engine. Looked up per target.
ENGINE_ROLE = "verilator_binary_mem_perturb"

SEEDS_ENV = "MERLIN_L3_ORDER_SEEDS"  # how many seeds; unset or 0 leaves the tier off
LATENCY_ENV = "MERLIN_L3_ORDER_MAX_LATENCY"  # any of these four: one profile instead of the defaults
TAIL_ENV = "MERLIN_L3_ORDER_TAIL_PERMILLE"
REGION_ENV = "MERLIN_L3_ORDER_REGION_LOG2"
SLOW_ENV = "MERLIN_L3_ORDER_SLOW_PERMILLE"
WORKERS_ENV = "MERLIN_L3_ORDER_WORKERS"
#: The knob profiles the seeds are dealt across, round robin. Two, because the two races measured need
#: opposite settings. A back-to-back overwrite/accumulate pair is caught by SHORT uniform latency (5 of 8
#: seeds at 64 cycles, 0 of 8 at 512 cycles with a tail, where a long first load fills the DMA's
#: in-flight window and the second cannot even issue). A pair several loads apart (a hardware loop
#: unroller's residual add) was never flagged by uniform latency up to 2048 cycles, nor by per-region
#: latency, and was flagged only by a heavy per-response TAIL. The tail is bounded because the
#: accelerator's own stall watchdog fires past ~10000 cycles. The fenced control stayed clean at every
#: setting tried.
DEFAULT_PROFILES: tuple[dict, ...] = (
    {"max_latency": 64, "tail_permille": 0},
    {"max_latency": 512, "tail_permille": 100},
)


@dataclass(frozen=True)
class OrderFacts:
    """What the static check needs to know about a target's local addresses, all derived."""

    acc_bit: int  # set in a local address that names the accumulator
    accum_bit: int  # set when the write adds onto the rows instead of overwriting them
    flag_mask: int  # every mode bit a local address may carry beside the row
    block: int | None  # columns per transfer block; None leaves every multi-column span unbounded


@functools.cache
def facts_for(target: str) -> tuple[OrderFacts | None, str]:
    """``(facts, why)``: the derived facts, or None and the reason they could not be derived.

    Cached per target, the absence included: deriving them loads the backend's ISA constants and the RTL
    facts, and a target without them would otherwise pay that failed derivation on every capsule.
    """
    try:
        from .rocc import decode as RD

        isa = RD.isa_constants(target)
    except Exception as exc:  # noqa: BLE001 -- no facts is "not applicable", returned with its reason
        return None, f"no derived ISA constants for {target!r}: {type(exc).__name__}: {exc}"
    bits = {name: isa.get(name) for name in ("ACC_I8", "ACC_ACCUM", "FULL_C_BIT")}
    missing = sorted(name for name, value in bits.items() if not isinstance(value, int) or isinstance(value, bool))
    if missing:
        return None, f"{target!r} derives no {', '.join(missing)}: accumulator rows cannot be told apart"
    try:
        from merlin.perf.decompose import is_unknown
        from merlin.perf.derived_bound import machine_from_facts

        cols = machine_from_facts(target, measure_fill=False).array_cols
        block = None if is_unknown(cols) else int(cols)
    except Exception:  # noqa: BLE001 -- an unknown block width only widens spans to unbounded
        block = None
    acc, accum, full = bits["ACC_I8"], bits["ACC_ACCUM"], bits["FULL_C_BIT"]
    return OrderFacts(acc, accum, acc | accum | full, block), "derived"


@dataclass
class _Load:
    index: int
    cls: str
    rows: tuple[int, int | None] | None  # [lo, hi) accumulator rows (hi None: unbounded), None: unresolved


def _family(cls: str) -> str | None:
    from .semantic_families import from_isa_class

    return from_isa_class(cls)


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _destination(payload: Mapping) -> str:
    """``in`` / ``out`` / ``unknown`` for a transfer, read from which local field its payload resolves."""
    from merlin.perf.deps.rocc import CONSUMES, DEFINES

    if any(_int(payload.get(name)) is not None for name in DEFINES):
        return "in"
    if any(_int(payload.get(name)) is not None for name in CONSUMES):
        return "out"
    return "unknown"


def _as_load(row: Mapping, facts: OrderFacts) -> _Load | None:
    """The accumulator rows a command may write by a memory transfer, None when it provably writes none."""
    from merlin.perf.deps.rocc import DEFINES

    cls = str(row.get("class"))
    index = row.get("index", -1)
    payload = row.get("decoded") if isinstance(row.get("decoded"), Mapping) else {}
    transfer = OFF_CHIP_FIELD in payload or _family(cls) == MOVEMENT_FAMILY
    if not transfer:
        # no transfer evidence: a local-only command issues no request; anything else is opaque
        from .semantic_families import local_only_classes

        return None if cls in local_only_classes() else _Load(index, cls, None)
    direction = _destination(payload)
    if direction == "out":
        return None
    if direction == "unknown":
        return _Load(index, cls, None)
    addr = next(_int(payload.get(name)) for name in DEFINES if _int(payload.get(name)) is not None)
    if not addr & facts.acc_bit:
        return None  # a scratchpad destination: not the accumulator
    base = addr & ~facts.flag_mask
    rows, cols = _int(payload.get("rows")), _int(payload.get("cols"))
    if rows is None or cols is None or facts.block is None:
        return _Load(index, cls, (base, None))
    blocks = max(1, -(-cols // facts.block))
    # A multi-block transfer's stride between blocks is configured elsewhere, so its span is unbounded.
    return _Load(index, cls, (base, base + rows) if blocks == 1 else (base, None))


def _overlap(a: _Load, b: _Load) -> bool | None:
    """True / False when both ranges are known, None when either is not."""
    if a.rows is None or b.rows is None:
        return None
    (alo, ahi), (blo, bhi) = a.rows, b.rows
    if not ((ahi is None or blo < ahi) and (bhi is None or alo < bhi)):
        return False
    return None if (ahi is None or bhi is None) else True


def static_check(trace: Mapping | None, facts: OrderFacts) -> dict:
    """Classify a decoded trace: ``hazard`` / ``unresolved`` / ``clear``, with the pairs that decided it."""
    hazards: list[dict] = []
    unresolved: list[dict] = []
    window: list[_Load] = []
    for row in (trace or {}).get("instructions") or []:
        cls = str(row.get("class"))
        if _family(cls) == BARRIER_FAMILY:
            window = []
            continue
        load = _as_load(row, facts)
        if load is None:
            continue
        for prev in window:
            overlap = _overlap(prev, load)
            pair = {"first": prev.index, "second": load.index, "classes": [prev.cls, load.cls]}
            if overlap is True:
                hazards.append(pair)
            elif overlap is None:
                unresolved.append(pair)
        window.append(load)
    verdict = "hazard" if hazards else ("unresolved" if unresolved else "clear")
    return {
        "verdict": verdict,
        "hazard_pairs": hazards[:16],
        "unresolved_pairs": unresolved[:16],
        "n_hazard_pairs": len(hazards),
        "n_unresolved_pairs": len(unresolved),
    }


def engine_for(target: str) -> tuple[Path | None, str]:
    """The target's declared memory-perturbing engine, verified by content, or (None, why)."""
    from merlin.common import provenance as P

    try:
        arts = [a for a in P.load_artifacts().values() if a.target == target and a.role == ENGINE_ROLE]
    except Exception as exc:  # noqa: BLE001 -- recorded as the reason the engine is unavailable
        return None, f"pin registry unreadable: {type(exc).__name__}: {exc}"
    if not arts:
        return None, f"no artifact with role {ENGINE_ROLE!r} is declared for {target!r}"
    if len(arts) > 1:
        return None, f"{len(arts)} artifacts declare role {ENGINE_ROLE!r} for {target!r}; the engine is ambiguous"
    check = P.verify_artifact(arts[0].name)
    if not check.present or check.matches is not True:
        return None, f"{arts[0].name}: " + ("; ".join(check.gaps) or "not verified")
    return Path(check.path), f"{arts[0].name} ({check.digest[:12]})"


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def seeds_requested() -> int:
    """How many seeds the tier runs; 0 (the default) leaves it off."""
    return max(0, _int_env(SEEDS_ENV) or 0)


def _profiles() -> tuple[dict, ...]:
    """The declared profiles, or the single one the environment names when it names any knob."""
    named = {
        "max_latency": _int_env(LATENCY_ENV),
        "tail_permille": _int_env(TAIL_ENV),
        "region_log2": _int_env(REGION_ENV),
        "slow_permille": _int_env(SLOW_ENV),
    }
    if all(value is None for value in named.values()):
        return DEFAULT_PROFILES
    return ({k: v for k, v in named.items() if v is not None},)


@dataclass
class OrderOutcome:
    record: dict
    failed: bool = False
    seeds: list = field(default_factory=list)

    @property
    def reason(self) -> str:
        names = ", ".join("in_order" if s is None else f"seed {s}" for s in self.seeds)
        return (
            "the certified program answers wrong when independent memory responses complete out of "
            f"order ({names}; {self.record.get('knobs')}): two loads into overlapping accumulator rows "
            "are ordered by issue, not completion -- order them (a fence) or make them not overlap"
        )


def run_order_check(
    *,
    target: str,
    trace: Mapping | None,
    elf: Path,
    judge: Callable[[int, str], str],
    facts: OrderFacts | None = None,
    engine: tuple[Path | None, str] | None = None,
    sweep: Callable[..., dict] | None = None,
    static: Mapping | None = None,
) -> OrderOutcome | None:
    """Run the tier. None only when it is switched off; every other outcome is recorded.

    ``static`` is the trigger verdict (``clear`` / ``hazard`` / ``unresolved``) when the caller already
    has one -- the completion-order check over the executed stream
    (:mod:`merlin.targetgen.completion_order`), which covers every local memory and follows the
    dispatch's real dependency edges. Without it the accumulator-only :func:`static_check` decides."""
    seeds_n = seeds_requested()
    if seeds_n <= 0:
        return None
    if static is None:
        why = "supplied"
        if facts is None:
            facts, why = facts_for(target)
        if facts is None:
            return OrderOutcome({"status": "not_applicable", "detail": why})
        static = static_check(trace, facts)
    static = dict(static)
    if static["verdict"] == "clear":
        return OrderOutcome({"status": "static_clear", "static": static})
    path, engine_why = engine if engine is not None else engine_for(target)
    rec: dict[str, Any] = {"static": static, "engine": engine_why}
    if path is None:
        rec["status"] = "unavailable"
        return OrderOutcome(rec)
    if not Path(elf).is_file():
        rec["status"] = "unavailable"
        rec["engine"] = f"{engine_why}; no executable at {elf}"
        return OrderOutcome(rec)
    profiles = _profiles()
    from . import mem_perturb as MP

    if sweep is None:
        sweep = MP.sweep
    # the engine's recorded invocation: it preloads the image so operands reach the memory model
    extra = [a.replace("{elf}", str(elf)) for a in MP.invocation_for(path)]
    rec["invocation"] = extra
    runs: list[dict] = []
    elapsed = 0.0
    res: dict = {}
    for i, knobs in enumerate(profiles):
        seeds = [s for s in range(1, seeds_n + 1) if (s - 1) % len(profiles) == i]
        if not seeds:
            continue
        # through the environment, so the engine runs with exactly the command line its backend uses
        res = sweep(
            path, elf, seeds, judge, via="env", include_in_order=(i == 0), extra_args=extra,
            workers=_int_env(WORKERS_ENV) or len(seeds) + 1, **knobs,
        )  # fmt: skip
        runs += [{**{k: r[k] for k in ("seed", "rc", "wall_s", "outcome")}, "profile": i} for r in res["runs"]]
        elapsed += float(res["elapsed_s"])
    bad = sorted((r["seed"] for r in runs if r["outcome"] == "fail"), key=lambda s: -1 if s is None else s)
    unjudged = [r["seed"] for r in runs if r["outcome"] not in ("pass", "fail")]
    rec.update(
        {
            "status": "order_sensitive" if bad else ("unjudged" if unjudged else "order_stable"),
            "failing_seeds": bad,
            "unjudged_seeds": unjudged,
            "knobs": {"seeds": seeds_n, "profiles": list(profiles)},
            "runs": runs,
            "elapsed_s": round(elapsed, 2),
            "emulator_sha256": res.get("emulator_sha256"),
            "elf_sha256": res.get("elf_sha256"),
        }
    )
    return OrderOutcome(rec, failed=bool(bad), seeds=bad)


def after_certificate(
    *,
    target: str,
    tiers: Mapping[str, Any],
    rtl_tiers: Any,
    trace: Mapping | None,
    elf: Path,
    judge: Callable[[int, str], str],
    static: Mapping | None = None,
) -> OrderOutcome | None:
    """The tier as the capsule ladder runs it, after a ladder whose cert tier passed.

    The cert tier is the deepest elaborated-RTL tier that passed; with none, nothing was certified and
    there is nothing to qualify. Never raises: a check that crashed is recorded as ``error`` on the
    result, because a silent skip would read as a clean one.
    """
    if seeds_requested() <= 0:
        return None
    passed = [t for t in sorted(tiers) if t in set(rtl_tiers or ()) and getattr(tiers[t], "status", None) == "pass"]
    if not passed:
        return None
    try:
        out = run_order_check(target=target, trace=trace, elf=elf, judge=judge, static=static)
    except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed
        out = OrderOutcome({"status": "error", "detail": f"{type(exc).__name__}: {exc}"})
    if out is not None:
        out.record["cert_tier"] = passed[-1]
    return out


def console_judge(compare: Callable[[dict], bool], parse: Callable[[str], dict]) -> Callable[[int, str], str]:
    """``pass`` / ``fail`` against the golden, ``unjudged:<why>`` when the run produced nothing to compare."""

    def judge(rc: int, text: str) -> str:
        if rc != 0:
            return f"unjudged:rc={rc}"
        try:
            outputs = parse(text)
        except Exception as exc:  # noqa: BLE001 -- no parseable answer is an outcome, not a crash
            return f"unjudged:{type(exc).__name__}"
        return "pass" if compare(outputs) else "fail"

    return judge
