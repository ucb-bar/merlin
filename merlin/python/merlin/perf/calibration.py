"""What an analytical model has to be calibrated against, and which mechanisms nothing can calibrate.

The performance phase does not put a cycle-accurate tier in its inner loop. Candidates are screened
``candidate -> L2 correctness -> analytical score -> roofline -> relative ranking``, and cycle-accurate
measurement appears exactly twice: **before** the run, on a small fixed set of mechanism-calibration
capsules whose measurements fit the model's coefficients, and **after**, as the functional regression
basis. The "before" half is what this module drives. Without it the model that ranks every candidate
schedule is uncalibrated, and a ranking from an uncalibrated model is a guess with a number attached.

WHAT A CALIBRATION IS ALLOWED TO SAY
------------------------------------
Four mechanisms, and each one is reported in three states -- calibrated, uncovered, uncalibratable --
never in two:

* **which engines exist and are observable.** The engine set is the target's OWN declaration
  (:func:`~merlin.perf.occupancy.declared_engines`) widened by what synthesis DETECTED
  (:func:`~merlin.targetgen.rtl.fsm.fsm_inventory`), because a synthesis flow exports only the state
  machines whose re-encoding would pay off: measured on one target, **15 state machines detected and 3
  exported**, with the two controllers whose concurrency was the entire point among the 12 dropped. An
  engine that no trace column is bound to is UNKNOWN here, never idle.
* **the composition operator and eta**, per capsule, with ``overlap_observable`` carried alongside.
  A zero eta from a vector that could not have shown overlap is not evidence of a machine that
  serialises; it is arithmetic about an instrument. The two are indistinguishable unless the flag
  travels with the number, so it does.
* **per-engine busy cycles and the realised/available overlap split**, so a model can see how much of
  the ceiling a schedule already took rather than only the ratio.
* **the memory regime each calibration capsule occupies**, because a coefficient is only fitted in the
  regime its capsules were in. Measured on the interlocked target here: of its 48 public capsules, 46
  are ``fits_double``, 1 is ``fits_single`` and 1 is ``spills`` -- while 90.1% of contraction regions
  across 20 real captured models land in ``spills``. A per-byte coefficient fitted on the 46 has never
  seen re-load traffic, and extending it to the 90% case is extrapolation dressed as a measurement.

WHY THREE STATES, EVERYWHERE
----------------------------
This tree has a recurring bug class in which a thing that could not be measured became a measured
zero. A vector unit exposed by no top-level busy port was left out of the occupancy vector, read as
permanently idle, and moved a corpus idle figure from **76.7% to 46.2%** (on one kernel, 89.9% to
39.2%). A 1 MiB window in our own harness was recorded as a hardware limit. The arithmetic always
produces a number, so the only defence is that every refusal here carries its reason and no refusal is
ever spelled ``0.0``: :func:`audit` walks a finished record and fails it if any of them is.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not run a simulator, name a target, or know how a trace was recorded. Traces arrive as
:class:`MechanismTrace` -- the producer's per-cycle column values plus the two facts that cannot be
derived from them (which engine a column belongs to, and what its ports read when low) -- and the
readings are taken by the existing measurement libraries rather than re-derived here:
:func:`~merlin.perf.falsifier.eta_from_occupancy` for eta,
:func:`~merlin.perf.occupancy.calibrate_state_idle` for the idle encoding,
:func:`~merlin.perf.headroom.composition_operator` for the operator, and
:func:`~merlin.targetgen.memory_regime.capsule_regime` for the regime. Re-deriving any of them would
re-make the mistake each one was written to stop.

TWO INSTRUMENTS, REPORTED SEPARATELY
------------------------------------
A per-cycle trace needs a waveform build or a co-simulation model, and a target with neither leaves
every eta here UNKNOWN however good its RTL is. So there is a second seam: :class:`CounterReading`, the
AGGREGATE totals a target's own COMBINATION performance counters hold. Where the RTL counts the cycles
each subset of its engines was busy, realised overlap is a counter value rather than an inference, and
it comes off the same elaborated RTL that certifies a capsule.

The two are never merged. They are two instruments over two engine axes -- a trace is per-cycle over
the engines the CONTRACT declares, a counter set is aggregate over the engines the target's counter
HEADER names -- and the second fills ``counter_calibration`` alone: ``ran_against_traces`` stays False
without a trace, the capsule cover stays UNKNOWN, and no number crosses between the blocks. See
:data:`INSTRUMENTS_NOT_COMPARABLE`. This module already refuses to compare its own two axes for a
weaker version of the same reason (:data:`KIND_AXIS_NOTE`); an instrument boundary is the stronger one.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "CALIBRATED", "COMPOSITION_TOLERANCE", "COUNTER_INSTRUMENT", "Cell", "CounterReading",
    "DEFAULT_PORT_LOW", "ENGINE_PAIR_AXIS", "EngineInventory",
    "EngineObservability", "IDLE_DECLARED", "IDLE_DERIVED", "IDLE_UNESTABLISHED",
    "IdleCalibration", "INSTRUMENTS_NOT_COMPARABLE", "KIND_AXIS_NOTE", "MEASURED",
    "MEMORY_REGIME_AXIS", "MechanismTrace", "POINTS_PER_CELL", "SCHEMA_VERSION", "TRACE_INSTRUMENT",
    "UNCALIBRATABLE", "UNCOVERED", "UNKNOWN", "audit",
    "busy_vectors", "calibrate", "calibrate_idle", "counter_calibration", "engine_inventory",
    "measured", "required_cells", "select_calibration_set", "unknown",
]

#: Bumped to 2 when the second instrument arrived: a v2 record carries ``counter_calibration``,
#: ``ran_against_counters`` and ``measured_basis`` beside the trace-side blocks a v1 record had. The
#: addition is backward-compatible in shape but not in MEANING -- a reader that checks only
#: ``ran_against_traces`` would call a counter-calibrated v2 record uncalibrated -- so the version says
#: so rather than leaving the reader to notice.
SCHEMA_VERSION = 2

#: The two states any single quantity in a calibration record may be in. There is no third spelling
#: and in particular there is no numeric stand-in for the second: see the module docstring.
MEASURED = "measured"
UNKNOWN = "unknown"

#: The three states a MECHANISM (a cell of the calibration cover) may be in. ``UNCOVERED`` and
#: ``UNCALIBRATABLE`` are deliberately apart: the first is a hole in the corpus that adding a capsule
#: would close, the second is a property of the target or the instrument that no capsule can close.
#: Collapsing them turns "nobody wrote this capsule" into "this machine does not do that".
CALIBRATED = "calibrated"
UNCOVERED = "uncovered"
UNCALIBRATABLE = "uncalibratable"

#: The axes a calibration cover is taken over.
ENGINE_PAIR_AXIS = "engine_pair"
MEMORY_REGIME_AXIS = "memory_regime"

#: How a busy PORT reads when it is low. Spelled exactly as
#: :func:`~merlin.perf.occupancy.calibrate_state_idle` reads its port columns, and overridable per
#: trace, because the calibration and the vector must agree about the same bits: a port read one way
#: while the idle encoding was derived from it read another way pins an encoding that is then applied
#: to a different signal. This is a port's declared assertion polarity, not an assumed ISA constant --
#: the producer states which columns ARE ports, and may restate the low value if its instrument
#: spells it differently.
DEFAULT_PORT_LOW = ("0", "")


def unknown(why: str) -> dict:
    """A quantity that could not be measured. ``value`` is ``None`` and ``why`` is mandatory.

    Constructed through a function rather than written out, so that no call site can produce the one
    shape this module exists to forbid -- a refusal whose value is a number.
    """
    if not why:
        raise ValueError("an UNKNOWN must carry the reason it is unknown; that is the whole point")
    return {"state": UNKNOWN, "value": None, "why": why}


def measured(value, detail: str = "") -> dict:
    """A quantity that was measured. ``None`` is refused: a missing value is an :func:`unknown`."""
    if value is None:
        raise ValueError("measured(None) would report a refusal as a reading; use unknown(why)")
    return {"state": MEASURED, "value": value, "detail": detail}


@dataclass(frozen=True)
class MechanismTrace:
    """One capsule's per-cycle occupancy trace, as the instrument recorded it.

    This is the seam. Everything a reading needs that CANNOT be derived from the samples is declared
    here by the producer, and nothing else is: which columns are top-level busy ports and which are
    internal state registers (the difference decides whether the idle encoding has to be calibrated at
    all), which declared engine each column belongs to (containment in the data cannot tell a unit's
    own load/store halves from an accelerator embedded in the cluster that drives it), what the
    instrument states it did NOT read, and what work the run performed.
    """

    capsule: str
    #: ``{column: [raw value per cycle]}``, in the instrument's own spelling. Strings, not booleans:
    #: a state register carries an encoding and reducing it to a boolean is what has to be calibrated.
    columns: Mapping[str, Sequence[str]]
    #: ``{column: declared engine}``. Declared, never read out of a signal's name.
    binding: Mapping[str, str]
    #: Columns that are top-level busy ports -- one bit, and their own reference for the calibration.
    port_columns: tuple[str, ...] = ()
    #: Columns that are internal state registers, whose idle value must be calibrated or refused.
    state_columns: tuple[str, ...] = ()
    #: Units the instrument states it did not read. Non-empty forces the eta reading to refuse; an
    #: unread unit is UNKNOWN, never idle, and restoring its cycles can move overlap either way.
    unmeasured_units: tuple[str, ...] = ()
    #: The producer's fingerprint for the work this run performed. ``None`` == not stated, and that is
    #: recorded rather than assumed equal, because eta is a ratio and doing less work can raise it.
    work: str | None = None
    #: Can the instrument observe when an engine's work COMPLETED, not merely that it was busy?
    #: Tri-state and never defaulted -- :func:`~merlin.perf.headroom.concurrency_traits` refuses
    #: without it, and defaulting it to True is how an unmeasured trait becomes a satisfied gate.
    completion_observable: bool | None = None
    #: What this instrument's ports read when low, if not the shared spelling.
    port_low: tuple[str, ...] = DEFAULT_PORT_LOW
    provenance: str = ""

    @property
    def sampled_cycles(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0


@dataclass(frozen=True)
class IdleCalibration:
    """Which value of a state column means idle, and on what evidence -- or that nothing pinned it."""

    idle_value: str | None
    basis: str
    paired_with: tuple[str, ...] = ()
    paired_columns: tuple[str, ...] = ()
    checked_traces: int = 0
    detail: str = ""

    @property
    def established(self) -> bool:
        return self.idle_value is not None

    def to_dict(self) -> dict:
        return {"idle_value": self.idle_value, "basis": self.basis,
                "paired_with": list(self.paired_with), "paired_columns": list(self.paired_columns),
                "checked_traces": self.checked_traces, "detail": self.detail}


#: The bases an idle encoding may rest on, weakest last. ``DECLARED`` is an INPUT, not a measurement,
#: and every number downstream of it is stamped with it so the distinction survives into the report --
#: a producer's declaration is acceptable evidence (the engine set and the kinds already arrive that
#: way) but it is not a cycle-exact identity and must not be quoted as one.
IDLE_DERIVED = "derived_from_paired_port"
IDLE_DECLARED = "declared_by_producer"
IDLE_UNESTABLISHED = "unestablished"


def calibrate_idle(traces: Sequence[MechanismTrace], *,
                   declared_idle_value: str | None = None) -> IdleCalibration:
    """Derive the state columns' idle encoding over the WHOLE corpus, or refuse.

    Delegates to :func:`~merlin.perf.occupancy.calibrate_state_idle`, which pins the encoding by the
    cycle-exact identity "the idle value is the one held on precisely the cycles a busy port is low".
    Taken over the whole corpus rather than per trace, because the encoding is a property of the
    design: a program that never exercises the paired unit leaves its state constant and must not be
    allowed to withdraw a calibration the rest of the corpus established.

    ``declared_idle_value`` is the value the PRODUCER states its state registers hold when idle. It is
    used only when the derivation refuses, it is never a default (absent it, unpaired state columns
    stay out of every vector and are reported), and it is stamped as :data:`IDLE_DECLARED` so nothing
    downstream can quote it as measured.
    """
    from merlin.perf.occupancy import calibrate_state_idle

    raw = [{c: list(v) for c, v in t.columns.items()} for t in traces]
    states = sorted({c for t in traces for c in t.state_columns})
    ports = sorted({c for t in traces for c in t.port_columns})
    got = calibrate_state_idle(raw, states, ports)
    if got.get("idle_value") is not None:
        return IdleCalibration(idle_value=str(got["idle_value"]), basis=IDLE_DERIVED,
                               paired_with=tuple(got.get("paired_with") or ()),
                               paired_columns=tuple(got.get("paired_columns") or ()),
                               checked_traces=int(got.get("checked_traces") or 0),
                               detail=str(got.get("detail") or ""))
    if declared_idle_value is not None:
        return IdleCalibration(
            idle_value=str(declared_idle_value), basis=IDLE_DECLARED,
            checked_traces=int(got.get("checked_traces") or 0),
            detail=(f"{got.get('detail') or 'nothing pinned the encoding'}; falling back to the value "
                    "the producer declares for these registers. DECLARED, not derived -- and "
                    "unverifiable until a workload exercises a busy port"))
    return IdleCalibration(idle_value=None, basis=IDLE_UNESTABLISHED,
                           checked_traces=int(got.get("checked_traces") or 0),
                           detail=(f"{got.get('detail') or 'nothing pinned the encoding'}; every "
                                   "unpaired state column therefore stays OUT of the occupancy "
                                   "vector and is reported unread, rather than assumed idle -- which "
                                   "is the reading that flatters the result"))


def busy_vectors(trace: MechanismTrace, idle: IdleCalibration
                 ) -> tuple[dict[str, list[bool]], dict[str, str]]:
    """``(hot, unreadable)`` -- the per-cycle busy vector, and the columns that could not be reduced.

    A port column reduces against its own declared low value. A state column reduces only once the
    idle encoding is established; without it the column is UNREADABLE and is returned as such, so the
    caller reports it instead of contributing zero busy cycles. That substitution is exactly the one
    that moved a corpus idle figure from 76.7% to 46.2%.

    A column that is CONSTANT across the run is still readable -- nothing was observed of it either
    way, and :func:`~merlin.perf.occupancy.joint_counts` already keeps constant columns out of the
    liveness test. Dropping it here instead would hide it from ``sampled_cycles`` as well.
    """
    hot: dict[str, list[bool]] = {}
    unreadable: dict[str, str] = {}
    ports = set(trace.port_columns)
    states = set(trace.state_columns)
    for col, vals in trace.columns.items():
        if col in ports:
            hot[col] = [str(v) not in trace.port_low for v in vals]
        elif col in states:
            if not idle.established:
                unreadable[col] = ("a state register with no calibrated idle encoding: "
                                   f"{idle.detail}")
                continue
            hot[col] = [str(v) != idle.idle_value for v in vals]
        else:
            unreadable[col] = ("the producer declared this column neither a busy port nor a state "
                               "register, so what value means busy is unstated; a column whose "
                               "meaning cannot be established stays out of the joint counts")
    return hot, unreadable


@dataclass(frozen=True)
class EngineObservability:
    """One declared engine: whether anything in the corpus can see it, and if not, why not."""

    engine: str
    kind: str
    contains: tuple[str, ...]
    observable: bool
    columns: tuple[str, ...]
    why: str
    #: WHERE THIS ENGINE CAME FROM -- the contract's declaration, or a derivation over the target's
    #: own RTL, with the evidence either way. An inventory that lists engines without saying which
    #: ones the machine itself evidenced cannot be audited.
    basis: str = ""

    def to_dict(self) -> dict:
        return {"engine": self.engine, "kind": self.kind, "contains": list(self.contains),
                "observable": self.observable, "columns": list(self.columns), "why": self.why,
                "basis": self.basis}


@dataclass(frozen=True)
class EngineInventory:
    """The engine set, from the target's declaration widened by what synthesis detected."""

    declared: dict[str, EngineObservability]
    #: Every control-state register the synthesis extraction FOUND, as
    #: :class:`~merlin.targetgen.rtl.fsm.FsmRegister` dicts. Empty means no extraction was found --
    #: a statement about the extraction, NOT about the design.
    detected: tuple[dict, ...] = ()
    detected_basis: str = ""
    #: Detected control FSMs that no contract entry declares. These are engines that EXIST and that
    #: nothing can bind a trace column to, which is a different failure from an engine with no port.
    detected_undeclared: tuple[str, ...] = ()
    #: Columns carrying busy cycles that are bound to no declared engine. Their cycles are real and
    #: they cannot be attributed, so any engine-axis reading over the same vector is refused.
    unattributed_columns: tuple[str, ...] = ()
    binding_error: str | None = None
    #: How the engine set was arrived at (:func:`merlin.perf.occupancy.engine_set`): the rule, what
    #: the RTL evidenced, and every candidate the derivation REFUSED with its reason.
    derivation: dict = field(default_factory=dict)

    @property
    def observable_engines(self) -> tuple[str, ...]:
        return tuple(sorted(e for e, o in self.declared.items() if o.observable))

    def to_dict(self) -> dict:
        return {
            "declared": {e: o.to_dict() for e, o in sorted(self.declared.items())},
            "n_declared": len(self.declared),
            "observable": list(self.observable_engines),
            "unobservable": {e: o.why for e, o in sorted(self.declared.items())
                             if not o.observable},
            "detected_registers": list(self.detected),
            "n_detected": len(self.detected),
            "detected_basis": self.detected_basis,
            "detected_undeclared": list(self.detected_undeclared),
            "unattributed_columns": list(self.unattributed_columns),
            "binding_error": self.binding_error,
            "derivation": dict(self.derivation),
        }


def engine_inventory(contract: Mapping, traces: Sequence[MechanismTrace], idle: IdleCalibration, *,
                     fsm_registers: Sequence = ()) -> EngineInventory:
    """Which engines exist, which the corpus can actually see, and which are UNKNOWN.

    Three separate reasons an engine ends up unobservable, kept apart because they call for different
    work: no trace binds a column to it at all (nobody instrumented it), a column is bound but is a
    state register with no calibrated encoding (no top-level busy port exposes it -- the class of unit
    whose invisibility inflated an idle figure by thirty points), or the binding itself disagrees with
    the contract (the trace and the contract do not describe the same device).

    ``fsm_registers`` is the synthesis inventory. It is used to widen the engine set, never to narrow
    it: a register the extraction found that no bound column reaches is reported, because on the
    interlocked target here the contract declares ONE engine while the extraction detected FIFTEEN
    control FSMs (3 exported) -- and ``LoadController.control_state`` /
    ``StoreController.control_state`` / ``ExecuteController.control_state``, whose concurrency is the
    entire measurement, are among the ones the contract does not name.
    """
    from merlin.perf.occupancy import engine_set, unit_bindings

    # The contract's compute units WIDENED by the engines the target's own RTL evidences: a
    # declaration naming one arithmetic unit leaves an overlap term unidentifiable on a machine whose
    # controllers plainly run at once (see occupancy.engine_set for the derivation and its refusals).
    engines, derivation = engine_set(contract, fsm_registers=fsm_registers)

    readable_by_engine: dict[str, set[str]] = {}
    unreadable_by_engine: dict[str, dict[str, str]] = {}
    unattributed: set[str] = set()
    binding_error: str | None = None

    for tr in traces:
        hot, unreadable = busy_vectors(tr, idle)
        try:
            unit_of, unbound = unit_bindings(sorted(tr.columns), tr.binding, engines)
        except ValueError as exc:
            # The trace names an engine the contract does not declare. That is not a column we can
            # place somewhere else: it means the instrument and the contract disagree about what the
            # device IS, and every engine-axis reading over this vector is void. Recorded once, and
            # the affected columns are treated as unattributed rather than silently reassigned.
            binding_error = str(exc)
            unattributed |= {c for c in tr.columns if any(hot.get(c, ()))}
            continue
        for col, eng in unit_of.items():
            if col in hot:
                readable_by_engine.setdefault(eng, set()).add(col)
            elif col in unreadable:
                unreadable_by_engine.setdefault(eng, {})[col] = unreadable[col]
        unattributed |= {c for c in unbound if any(hot.get(c, ()))}

    declared: dict[str, EngineObservability] = {}
    for name, spec in sorted(engines.items()):
        cols = tuple(sorted(readable_by_engine.get(name, ())))
        if cols:
            why = (f"{len(cols)} trace column(s) bound to it reduce to a busy vector "
                   f"(idle basis: {idle.basis})")
            observable = True
        elif name in unreadable_by_engine:
            observable = False
            why = ("bound only to column(s) no instrument could reduce to busy: "
                   + "; ".join(f"{c}: {r}" for c, r in sorted(unreadable_by_engine[name].items())))
        else:
            observable = False
            why = ("no trace column is bound to this engine, so nothing observed it. UNKNOWN, not "
                   "idle: an engine no instrument read contributes no busy cycles, which inflates "
                   "the idle figure and makes its overlap unobservable by construction")
        declared[name] = EngineObservability(
            engine=name, kind=str(spec.get("kind") or ""),
            contains=tuple(spec.get("contains") or ()), observable=observable,
            columns=cols, why=why, basis=str(spec.get("basis") or ""))

    detected = tuple({"module": r.module, "register": r.register, "qualified": r.qualified,
                      "states": r.states, "exported": r.exported} for r in fsm_registers)
    if detected:
        basis = (f"{len(detected)} control-state register(s) the synthesis extraction DETECTED, "
                 f"{sum(1 for d in detected if d['exported'])} of which it exported a transition "
                 "table for. The inventory is the detected set: an export answers whether "
                 "re-encoding would pay off, not whether the controller is observable")
    else:
        basis = ("no synthesis FSM extraction was found. That is a statement about the extraction, "
                 "NOT about the design -- this target's engine set beyond its declaration is UNKNOWN")
    # A detected register counts as REACHED when some column already bound to a declared engine is
    # plausibly that register. The match is the register's own structural one (leaf plus a containing
    # instance), because a synthesis export names the module CLASS while a trace names the INSTANCE
    # path; comparing the two strings directly reports every controller as unreached.
    bound_columns = {c for o in declared.values() for c in o.columns}
    undeclared = tuple(r.qualified for r in fsm_registers
                       if not any(r.matches_signal(c) for c in bound_columns))
    return EngineInventory(declared=declared, detected=detected, detected_basis=basis,
                           detected_undeclared=undeclared,
                           unattributed_columns=tuple(sorted(unattributed)),
                           binding_error=binding_error, derivation=derivation)


@dataclass(frozen=True)
class Cell:
    """One mechanism the cover has to hit, and whether anything can hit it."""

    axis: str
    key: str
    state: str
    why: str
    #: Capsules the selector chose for this cell, in the order it chose them.
    capsules: tuple[str, ...] = ()
    #: Capsules that instantiate the cell but were not selected.
    candidates: tuple[str, ...] = ()
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"axis": self.axis, "key": self.key, "state": self.state, "why": self.why,
                "capsules": list(self.capsules), "n_candidates": len(self.candidates),
                "candidates": list(self.candidates), **({"detail": self.detail} if self.detail else {})}


def required_cells(inventory: EngineInventory, corpus_regimes: Mapping) -> list[Cell]:
    """The mechanisms an analytical model needs calibrated, DERIVED -- never a written-down list.

    Two axes, and the choice of both is forced rather than picked:

    * **one cell per unordered pair of DECLARED engines.** A pair is what overlap is a property of, so
      a model with an overlap term has one coefficient per pair. Nested engines are pairs too: a
      systolic array embedded in the cluster that drives it is two engines whose concurrency is the
      measurement, and folding it away reports zero overlap on the one device where the question
      matters. A pair whose either half is unobservable is :data:`UNCALIBRATABLE` and says which half.
    * **one cell per memory regime PRESENT in the corpus.** A coefficient is fitted in a regime, not
      in general: one fitted only on work that fits the operand store twice over has never seen
      re-load traffic, so extending it to ``spills`` is extrapolation. A corpus whose capacity cannot
      be derived yields a single ``UNCALIBRATABLE`` cell rather than a regime-free pass, because an
      unmeasurable capacity reported as satisfied is how a range-check abort once reached a simulator
      with nothing recorded.
    """
    from merlin.targetgen import memory_regime as MR

    cells: list[Cell] = []
    names = sorted(inventory.declared)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            oa, ob = inventory.declared[a], inventory.declared[b]
            key = f"{a}|{b}"
            blocked = [o for o in (oa, ob) if not o.observable]
            if blocked:
                cells.append(Cell(
                    ENGINE_PAIR_AXIS, key, UNCALIBRATABLE,
                    why=("; ".join(f"{o.engine} is not observable: {o.why}" for o in blocked)
                         + ". No capsule can calibrate an overlap term for this pair, because the "
                           "instrument cannot see one of its halves -- and a zero read off a vector "
                           "missing an engine is not a measurement of serialisation")))
            else:
                cells.append(Cell(ENGINE_PAIR_AXIS, key, UNCOVERED,
                                  why="both halves are observable; awaiting a capsule that makes "
                                      "both live in one run"))
    if len(names) < 2:
        cells.append(Cell(
            ENGINE_PAIR_AXIS, "<no pair>", UNCALIBRATABLE,
            why=(f"the target declares {len(names)} engine(s), so there is no pair to overlap. Any "
                 "overlap term in a model of this target is unidentifiable from the declaration "
                 "alone -- which is a fact about the declaration, since a control-FSM inventory may "
                 "well show engines the contract does not name"),
            detail={"declared": names, "detected_undeclared": list(inventory.detected_undeclared)}))

    by_regime = dict((corpus_regimes or {}).get("by_regime") or {})
    capacity = (corpus_regimes or {}).get("capacity_rows")
    if not capacity:
        cells.append(Cell(
            MEMORY_REGIME_AXIS, MR.UNKNOWN, UNCALIBRATABLE,
            why=("the target declares no operand-store capacity we can derive, so no capsule's "
                 "regime can be established and a coefficient's domain of validity is unknown. "
                 "Never folded into a fitting regime")))
        return cells
    for regime in MR.ORDER:
        members = sorted(by_regime.get(regime) or ())
        if members:
            cells.append(Cell(MEMORY_REGIME_AXIS, regime, UNCOVERED,
                              why=f"{len(members)} capsule(s) in the corpus occupy this regime",
                              candidates=tuple(members)))
        else:
            cells.append(Cell(
                MEMORY_REGIME_AXIS, regime, UNCALIBRATABLE,
                why=("no capsule in the corpus occupies this regime, so nothing here can fit a "
                     "coefficient in it. A coefficient fitted in the regimes that ARE present does "
                     "not transfer to this one, and the model must report UNKNOWN there rather than "
                     "extending its fit")))
    unsized = sorted(by_regime.get(MR.UNKNOWN) or ())
    if unsized:
        cells.append(Cell(
            MEMORY_REGIME_AXIS, MR.UNKNOWN, UNCALIBRATABLE,
            why=(f"{len(unsized)} capsule(s) declare no shape we could size, so which regime they "
                 "occupy is unknown; they are excluded from every regime cell rather than counted in "
                 "the weakest one"), candidates=tuple(unsized)))
    return cells


#: Said on every corpus operator, because the two etas in this record are two INSTRUMENTS.
#: :func:`~merlin.perf.headroom.composition_operator` groups by resource KIND, and every unit a
#: contract declares under ``compute_units`` is an arithmetic engine -- so on a target whose declared
#: engines share a kind the kind axis collapses them into ONE group and the operator is correctly
#: unavailable. The per-capsule eta is on the DECLARED ENGINE axis, which can see two engines of one
#: kind running together. Same formula and same denominator shape, different instrument.
KIND_AXIS_NOTE = (
    "the corpus operator resolves on the resource-KIND axis (composition_operator groups by kind); "
    "the per-capsule eta resolves on the DECLARED ENGINE axis. Where two declared engines share a "
    "kind the kind axis collapses them into one group and the operator is unavailable by "
    "construction. These are two instruments, not two readings, and must not be compared")


#: The eta thresholds a composition operator is classified against. Declared once so the trace seam and
#: the counter seam classify the same eta the same way; changing it in one place would let the two
#: instruments disagree about the operator while agreeing about the number.
COMPOSITION_TOLERANCE = 0.05

#: The two instruments this module accepts, named so a record says which one produced it.
TRACE_INSTRUMENT = "per_cycle_trace"
COUNTER_INSTRUMENT = "hardware_combination_counters"

#: Why a counter reading and a trace reading are never merged, averaged or compared.
#:
#: A trace is a per-cycle vector: it can say WHEN each engine was busy, so overlap is read off the
#: timeline and the vector's own live-column count says whether overlap could have been seen at all. A
#: combination counter is an AGGREGATE: the hardware has already reduced the timeline to seven totals,
#: overlap is a counter value rather than an inference, and no per-cycle question can be asked of it
#: afterwards. The two also disagree about what an engine IS -- a trace binds columns to the engines the
#: CONTRACT declares, while the counter block's engines are factored out of the target's own counter
#: names -- so an eta from one is not the same quantity as an eta from the other even when the numbers
#: are close. This module already refuses to compare its own two axes for a weaker version of this
#: reason (see :data:`KIND_AXIS_NOTE`); the same rule applies here, more strongly.
INSTRUMENTS_NOT_COMPARABLE = (
    "a counter reading is an AGGREGATE from the target's own combination counters and a trace reading "
    "is a per-cycle vector over the engines the CONTRACT declares. They are two instruments over two "
    "engine axes, not two readings of one quantity: they are reported side by side and never merged, "
    "averaged, cross-checked as agreement, or substituted for one another")


@dataclass(frozen=True)
class CounterReading:
    """One workload's joint occupancy, as the target's OWN hardware counted it.

    The SECOND seam, and deliberately a separate one. :class:`MechanismTrace` needs a per-cycle
    instrument -- a waveform build or a co-simulation model -- and on a target that has neither, every
    eta in a calibration record stays UNKNOWN however good the RTL is. A target whose RTL carries
    combination performance counters needs neither: the hardware already counts the cycles in which
    each SUBSET of its engines was busy, and a combination counter IS a joint-occupancy reading.

    What the producer declares here, because none of it is derivable from the values:

    * ``values`` -- the counter readings a bracketed run actually printed back. A counter the bracket
      configured but whose value did not come back must be ABSENT from this mapping, never zero:
      :func:`~merlin.perf.hw_counters.eta_from_counters` refuses on a missing counter, and a zero
      would report overlap that was never measured as overlap that did not happen.
    * ``counters`` -- the :class:`~merlin.perf.hw_counters.OccupancyCounters` block the target's own
      shipped header was factored into. The engine axis comes from there, not from this module, and
      not from the contract.
    * ``kind_of`` -- each counter engine's resource KIND, if the producer can state it. A kind cannot
      be derived from a counter's spelling (deriving "LD means movement" from the token is exactly the
      overfit the repo's cardinal rule forbids), so absent it the kind axis refuses rather than
      guesses.
    """

    workload: str
    #: ``{counter name: cycles}``. A missing counter is UNKNOWN; it is never defaulted to 0.
    values: Mapping[str, int]
    #: The target's derived combination-counter block (``hw_counters.OccupancyCounters``).
    counters: object
    #: The run's total cycle window, if the harness measured one. ``None`` == not stated; the idle
    #: residual is then simply not emitted rather than computed against a guessed window.
    total_cycles: int | None = None
    #: ``{counter engine: resource kind}``, declared by the producer. Absent == the kind axis refuses.
    kind_of: Mapping[str, str] | None = None
    #: Whether the harness could observe when an engine's work COMPLETED. Tri-state, never defaulted.
    completion_observable: bool | None = None
    provenance: str = ""


#: How many capsules a fitted cell gets by default. TWO, because the repo's standing rule is at least
#: two points per fitted parameter: one point cannot separate a rate from a fixed fill/drain
#: intercept, and the one place a single point was used it could not (a layer at 751.1 cycles per tile
#: pass, from which no intercept is recoverable). A cell that can only supply one capsule is fitted
#: with one and SAYS the coefficient is an extrapolation there.
POINTS_PER_CELL = 2


def select_calibration_set(cells: Sequence[Cell], *,
                           readings: Mapping[str, Mapping] | None = None,
                           regime_by_capsule: Mapping[str, Mapping] | None = None,
                           traces: Sequence[MechanismTrace] = (),
                           points_per_cell: int = POINTS_PER_CELL) -> list[Cell]:
    """Fill each cell with the cheapest capsules that actually instantiate it.

    Two different selection rules, because the two axes are cheap in different ways:

    * an **engine-pair** cell is filled from the runs whose vector shows BOTH halves live, ordered by
      SAMPLED CYCLES ascending. Cycle-accurate cost on this rig is dominated by output size, so the
      shortest run that exhibits the mechanism is the one to spend the tier on;
    * a **memory-regime** cell is filled from its members at the EXTREMES of working-set size (the
      smallest and the largest). Two points at the ends of the regime is what lets an affine
      coefficient be fitted across it; two points in the middle would fit the same intercept twice.

    A cell nothing can fill stays :data:`UNCOVERED` (a corpus hole), and a cell that was already
    :data:`UNCALIBRATABLE` is left exactly as it is -- selection cannot repair a mechanism the
    instrument cannot see.
    """
    readings = readings or {}
    regime_by_capsule = regime_by_capsule or {}
    cycles = {t.capsule: t.sampled_cycles for t in traces}
    out: list[Cell] = []
    for cell in cells:
        if cell.state == UNCALIBRATABLE:
            out.append(cell)
            continue
        if cell.axis == ENGINE_PAIR_AXIS:
            a, b = cell.key.split("|", 1)
            live = [name for name, r in sorted(readings.items())
                    if a in (r.get("live_engines") or ()) and b in (r.get("live_engines") or ())]
            live.sort(key=lambda n: (cycles.get(n, 1 << 62), n))
            picked = tuple(live[:max(1, points_per_cell)])
            why = (f"{len(live)} run(s) show both halves live; taking the {len(picked)} with the "
                   "fewest sampled cycles, since cycle-accurate cost here tracks output size"
                   if picked else
                   "no supplied run makes both halves live in the same vector, so this pair's "
                   "overlap term is UNCOVERED -- a hole in the corpus, not a machine that serialises")
            out.append(Cell(cell.axis, cell.key, CALIBRATED if picked else UNCOVERED, why,
                            capsules=picked, candidates=tuple(live), detail=cell.detail))
            continue
        members = list(cell.candidates)
        sized = [(m, (regime_by_capsule.get(m) or {}).get("rows")) for m in members]
        known = sorted(((m, r) for m, r in sized if isinstance(r, int)), key=lambda kv: (kv[1], kv[0]))
        # Both ENDS of the regime first, then an even spread between them. Two points in the middle
        # of a regime fit the same intercept twice and recover no slope, which is the single-point
        # failure one step disguised.
        k = max(1, points_per_cell)
        n = len(known)
        if n == 0:
            picked: tuple[str, ...] = ()
        elif k == 1 or n == 1:
            picked = (known[0][0],)
        elif n <= k:
            picked = tuple(m for m, _r in known)
        else:
            idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
            picked = tuple(known[i][0] for i in idx)
        if not picked:
            out.append(Cell(cell.axis, cell.key, UNCOVERED,
                            why=("the corpus lists capsule(s) in this regime but none of them could "
                                 "be sized, so none is established to occupy it"),
                            candidates=tuple(members), detail=cell.detail))
            continue
        extrap = (" only ONE point is available, so a coefficient fitted here cannot separate a rate "
                  "from a fixed intercept and is an EXTRAPOLATION in this regime."
                  if len(picked) < 2 else "")
        out.append(Cell(cell.axis, cell.key, CALIBRATED,
                        why=(f"{len(picked)} point(s) at the extremes of the regime's working-set "
                             f"range ({[r for _m, r in known if _m in picked]} rows)." + extrap),
                        capsules=picked, candidates=tuple(members), detail=cell.detail))
    return out


def _regime_of(capsule: str, regime_by_capsule: Mapping[str, Mapping] | None) -> dict:
    """The regime a capsule occupies, or an UNKNOWN naming why it could not be established."""
    got = (regime_by_capsule or {}).get(capsule)
    if not got:
        return unknown(f"no memory regime was established for {capsule!r}; a coefficient fitted on "
                       "this run has no known domain of validity")
    from merlin.targetgen import memory_regime as MR
    if got.get("regime") in (None, MR.UNKNOWN):
        return unknown(f"the regime of {capsule!r} is unknown: {got.get('why') or 'no reason given'}")
    return measured(dict(got), detail="derived from the capsule's own declared inputs against the "
                                      "target's derived operand-store capacity")


def _reading(trace: MechanismTrace, inventory: EngineInventory, idle: IdleCalibration) -> dict:
    """One capsule's mechanism reading: eta, the overlap split, per-engine busy, and the caveats.

    eta comes from :func:`~merlin.perf.falsifier.eta_from_occupancy`, on the target's DECLARED engine
    axis, so that this number and the perf ledger's falsifier verdict are the same quantity and the
    same denominator (the second-largest per-engine busy count -- the most any single pair could ever
    overlap). ``overlap_observable`` travels beside it because a zero eta from a vector with fewer
    than two live engines is arithmetic, not evidence.
    """
    from merlin.perf.falsifier import ENGINE_AXIS, eta_from_occupancy
    from merlin.perf.occupancy import declared_engines, joint_counts, unit_bindings

    hot, unreadable = busy_vectors(trace, idle)
    engines = {e: {"kind": o.kind, "contains": o.contains} for e, o in inventory.declared.items()}
    out: dict = {"capsule": trace.capsule, "sampled_cycles": trace.sampled_cycles,
                 "work": trace.work, "idle_basis": idle.basis,
                 "unreadable_columns": dict(sorted(unreadable.items())),
                 "instrument_unmeasured_units": list(trace.unmeasured_units),
                 "completion_observable": trace.completion_observable,
                 "provenance": trace.provenance}
    try:
        unit_of, unbound = unit_bindings(sorted(hot), trace.binding, engines)
    except ValueError as exc:
        out["eta"] = unknown(f"the trace binds a column to an engine the contract does not declare: "
                             f"{exc}. The trace and the contract disagree about what the device is, "
                             "so every engine-axis reading over this vector is void")
        out["busy_cycles"] = unknown("no engine-axis attribution exists for this vector")
        out["overlap"] = {"realised_cycles": unknown("eta refused"),
                          "available_cycles": unknown("eta refused"),
                          "unrealised_cycles": unknown("eta refused")}
        out["overlap_observable"] = None
        out["live_engines"] = ()
        return out

    kinds = {c: engines[e]["kind"] for c, e in unit_of.items() if e in engines}
    jc = joint_counts(hot, kinds, unit_of)
    obs = eta_from_occupancy(trace.capsule, hot, unit_of=unit_of, kinds=kinds, work=trace.work,
                             unmeasured=trace.unmeasured_units)

    eng_hot = {e: [any(hot[c][i] for c in jc["joint_columns"] if unit_of.get(c) == e)
                   for i in range(jc["sampled_cycles"])]
               for e in sorted({unit_of[c] for c in jc["joint_columns"] if c in unit_of})}
    busy = {e: sum(v) for e, v in eng_hot.items()}
    # Live at ENGINE level, the same rule joint_counts applies to columns and for the same reason: an
    # engine busy on every sampled cycle is constant, and nothing was observed of it either way.
    live = tuple(e for e, n in busy.items() if 0 < n < jc["sampled_cycles"]) or \
        tuple(e for e, n in busy.items() if n > 0)

    out["axis"] = ENGINE_AXIS
    out["eta"] = (measured(obs.eta, detail=obs.detail) if obs.eta is not None
                  else unknown(obs.detail or "the vector supports no eta reading"))
    out["overlap_observable"] = bool(jc["overlap_observable"])
    out["live_engines"] = live
    out["busy_cycles"] = (measured(dict(sorted(busy.items())),
                                   detail="per DECLARED engine, after sub-signal subsumption")
                          if busy else unknown("no column reduced to a busy vector"))
    if obs.measured:
        unreal = obs.available_cycles - obs.realised_cycles
        out["overlap"] = {
            "realised_cycles": measured(obs.realised_cycles,
                                        detail="cycles with >=2 declared engines busy together"),
            "available_cycles": measured(obs.available_cycles,
                                         detail="the second-largest per-engine busy count -- the "
                                                "ceiling on any single pair, and the denominator "
                                                "headroom/composition_operator use"),
            # Not clipped at zero. With three or more engines overlapping in disjoint pairs the
            # numerator counts all pairs while the denominator is the top pair's ceiling, so eta can
            # exceed 1 and this can go negative. That is a true statement about the vector (more than
            # one pair overlapped) and is reported as such.
            "unrealised_cycles": measured(unreal, detail=("negative when eta > 1, i.e. when >=3 "
                                                          "engines overlapped in disjoint pairs")),
        }
    else:
        why = obs.detail or "the vector supports no overlap reading"
        out["overlap"] = {"realised_cycles": unknown(why), "available_cycles": unknown(why),
                          "unrealised_cycles": unknown(why)}
    out["joint"] = {k: jc[k] for k in ("sampled_cycles", "joint_columns", "subsumed_columns",
                                       "overlap_observable", "live_columns", "idle_cycles",
                                       "overlap_any", "overlap_across_kinds",
                                       "overlap_across_kinds_is_lower_bound", "undeclared_columns",
                                       "unbound_columns")}
    out["unbound_columns_with_work"] = sorted(c for c in unbound if any(hot.get(c, ())))
    return out


def _composition(readings: Mapping[str, Mapping], inventory: EngineInventory) -> dict:
    """The corpus composition operator, on BOTH axes, with the difference between them stated.

    Two entries, because the library's operator and this record's eta are two instruments:

    ``kind_axis``
        :func:`~merlin.perf.headroom.composition_operator` verbatim. It groups by resource KIND, and
        every unit a contract declares under ``compute_units`` is an arithmetic engine -- so wherever
        the declared engines share a ResourceKind the kind axis collapses them into ONE group and the
        operator is correctly unavailable. That refusal is recorded, not worked around: inventing a
        distinct kind per engine to force a finer grouping would manufacture exactly the pair the
        grouping refuses to assume.
    ``engine_axis``
        the same arithmetic and the same thresholds (:class:`~merlin.perf.headroom.Composition`,
        ``eta <= tol -> SUM``, ``eta >= 1-tol -> MAX``, else ``PARTIAL``) and the same corpus
        aggregation (sum the realised counts, sum the available ceilings), applied on the DECLARED
        ENGINE axis -- where the per-capsule etas already live, and which can see two engines of one
        kind running together. Nothing about the quantity changes; only the grouping does, and that is
        why it is a separate entry rather than a correction to the first.

    UNKNOWN propagates in both, the way the library propagates it: one supplied run with no overlap
    reading leaves the corpus operator unestablished rather than partially derived, because dropping
    it would silently reweight the corpus towards the runs that happened to be measurable.
    """
    from merlin.perf.decompose import ActivitySource, Resource, ResourceKind, Unavailable
    from merlin.perf.headroom import Composition, composition_operator

    tol = COMPOSITION_TOLERANCE
    kinds_declared = sorted({o.kind for o in inventory.declared.values()})
    unreadable = {name for name, r in readings.items()
                  if (r.get("overlap") or {}).get("realised_cycles", {}).get("state") != MEASURED
                  or (r.get("busy_cycles") or {}).get("state") != MEASURED}
    out: dict = {"n_runs": len(readings), "runs_without_a_reading": sorted(unreadable),
                 "declared_kinds": kinds_declared, "kind_axis_note": KIND_AXIS_NOTE}

    if not readings or unreadable:
        why = (("no per-cycle trace was supplied, so nothing constrains the operator"
                if not readings else
                f"run(s) {sorted(unreadable)} yielded no overlap reading; UNKNOWN propagates -- one "
                "unmeasured run leaves the corpus operator unestablished rather than partially "
                "derived, since dropping it reweights the corpus towards whatever was measurable"))
        out["kind_axis"] = {"operator": unknown(why), "eta": unknown(why)}
        out["engine_axis"] = {"operator": unknown(why), "eta": unknown(why),
                              "realised_cycles": unknown(why), "available_cycles": unknown(why)}
        return out

    realised = sum(int(r["overlap"]["realised_cycles"]["value"]) for r in readings.values())
    available = sum(int(r["overlap"]["available_cycles"]["value"]) for r in readings.values())
    if available == 0:
        why = ("no supplied run has any overlappable time (every run's second-busiest engine is busy "
               "0 cycles), so the operator is 0/0 -- undefined, not SUM")
        out["engine_axis"] = {"operator": unknown(why), "eta": unknown(why),
                             "realised_cycles": measured(realised), "available_cycles": measured(0)}
    else:
        eta = realised / available
        op = (Composition.SUM if eta <= tol
              else Composition.MAX if eta >= 1.0 - tol else Composition.PARTIAL)
        out["engine_axis"] = {
            "operator": measured(op.value, detail=f"over {len(readings)} calibration run(s), "
                                                  f"tolerance {tol}"),
            # Not clipped. With >=3 engines overlapping in disjoint pairs the numerator counts all
            # pairs while the denominator is the top pair's ceiling, so eta can exceed 1. That is a
            # true statement about the vector and MAX is the right classification of it.
            "eta": measured(eta, detail="realised / available overlap on the DECLARED ENGINE axis; "
                                        "may exceed 1 with >=3 engines overlapping in disjoint pairs "
                                        "and is reported, not clipped"),
            "realised_cycles": measured(realised), "available_cycles": measured(available)}

    sources: list[ActivitySource] = []
    overlaps: dict[str, int] = {}
    for name, r in sorted(readings.items()):
        # Every declared compute unit is an arithmetic engine by construction -- that is what the
        # contract's compute-unit list IS -- so its ResourceKind is COMPUTE. The collapse this causes
        # is the point of reporting both axes.
        resources = tuple(Resource(name=e, kind=ResourceKind.COMPUTE, busy_cycles=int(n))
                          for e, n in sorted(r["busy_cycles"]["value"].items()))
        if not resources:
            continue
        sources.append(ActivitySource(
            workload=name, total_cycles=int(r.get("sampled_cycles") or 0), resources=resources,
            partitioned=False,      # a per-cycle JOINT vector is not a partition; asserting it is
                                    # what licenses an overlap reading at all
            completion_observable=r.get("completion_observable"),
            provenance=str(r.get("provenance") or "")))
        overlaps[name] = int(r["overlap"]["realised_cycles"]["value"])
    got = composition_operator(sources, observed_overlap_cycles=overlaps) if sources else None
    if got is None:
        why = "no run yielded a per-engine busy attribution, so there is no activity source"
        out["kind_axis"] = {"operator": unknown(why), "eta": unknown(why)}
    elif isinstance(got, Unavailable):
        why = f"{got.what}: missing {list(got.missing)}" + (f" ({got.detail})" if got.detail else "")
        out["kind_axis"] = {"operator": unknown(why), "eta": unknown(why)}
    else:
        op, eta = got
        out["kind_axis"] = {
            "operator": measured(op.value, detail=f"over {len(sources)} calibration run(s)"),
            "eta": measured(eta, detail="realised / available overlap on the resource-KIND axis")}
    return out


def counter_calibration(readings: Sequence[CounterReading]) -> dict:
    """The corpus composition operator from AGGREGATE hardware combination counters.

    The second instrument, kept structurally apart from :func:`_composition` rather than folded into
    it. Both produce an operator and an eta, and that is exactly why they must not share a code path:
    the numbers would then look like two readings of one quantity, and the first thing a reader would
    do is compare them. They are over different engine axes and different instruments; see
    :data:`INSTRUMENTS_NOT_COMPARABLE`.

    Every reading is taken by :mod:`merlin.perf.hw_counters`, which already knows how a counter set
    lies -- that a per-engine busy total is the single counter PLUS every combination containing it
    (reading the singles as whole-engine totals understates the busiest engine, which is eta's
    denominator, and so inflates eta), and that a missing counter makes the whole reading UNKNOWN
    rather than a lower bound reported as a total. Re-deriving either here would re-make the mistake
    each rule was written to stop.

    UNKNOWN propagates exactly as it does on the trace side: one run without a reading leaves the
    corpus operator unestablished rather than partially derived, because dropping it reweights the
    corpus towards whatever happened to be measurable.
    """
    from merlin.perf.decompose import ActivitySource, Resource, ResourceKind, Unavailable
    from merlin.perf.headroom import Composition, composition_operator
    from merlin.perf.hw_counters import eta_from_counters, observations_from_counters

    out: dict = {"instrument": COUNTER_INSTRUMENT, "n_runs": len(readings),
                 "not_comparable_with_traces": INSTRUMENTS_NOT_COMPARABLE,
                 "engine_axis_source": ("factored out of the target's OWN shipped counter header, "
                                        "not read from the capability contract"),
                 "runs": []}
    if not readings:
        why = ("no counter reading was supplied, so the target's own combination counters constrain "
               "nothing here")
        out["engine_axis"] = {"operator": unknown(why), "eta": unknown(why),
                              "realised_cycles": unknown(why), "available_cycles": unknown(why)}
        out["kind_axis"] = {"operator": unknown(why), "eta": unknown(why)}
        out["engines"] = []
        out["runs_without_a_reading"] = []
        return out

    # Readings taken over DIFFERENT engine sets are two instruments, not one corpus: summing their
    # totals would build a corpus figure whose denominator is over engines the numerator never saw.
    engine_sets = {tuple(getattr(r.counters, "engines", ())) for r in readings}
    mixed = len(engine_sets) != 1
    out["engines"] = [] if mixed else sorted(next(iter(engine_sets)))
    per_run: dict[str, dict] = {}
    for r in sorted(readings, key=lambda x: x.workload):
        got = eta_from_counters(dict(r.values), r.counters)
        obs = observations_from_counters(
            dict(r.values), r.counters, total_cycles=r.total_cycles,
            source=r.provenance or COUNTER_INSTRUMENT,
            kind_of=dict(r.kind_of) if r.kind_of else None)
        entry: dict = {
            "workload": r.workload,
            "counters": r.counters.to_dict() if hasattr(r.counters, "to_dict") else {},
            "counter_values": {k: int(v) for k, v in sorted(dict(r.values).items())},
            "total_cycles": r.total_cycles,
            "completion_observable": r.completion_observable,
            "provenance": r.provenance,
            "observations": obs,
        }
        if got.get("state") == "measured":
            entry["busy_cycles"] = measured(
                {k: int(v) for k, v in sorted(got["busy_cycles"].items())},
                detail="per COUNTER-DERIVED engine: the single counter plus every combination "
                       "containing it, which is exact because the increment conditions partition "
                       "busy time")
            entry["realised_cycles"] = measured(
                int(got["realised_cycles"]),
                detail="cycles the hardware itself counted with >=2 engines busy together -- "
                       "measured, not inferred from buckets")
            entry["available_cycles"] = measured(
                int(got["available_cycles"]),
                detail="min(total - busiest, total // 2); equals the second-largest per-engine total "
                       "for two engines, which is the falsifier's denominator")
            entry["eta"] = measured(float(got["eta"]),
                                    detail="realised / available overlap on the counter-derived "
                                           "engine axis")
            entry["counter_set_complete"] = bool(got.get("complete"))
        else:
            why = str(got.get("why") or "the counter set supports no eta reading")
            entry["busy_cycles"] = unknown(why)
            entry["realised_cycles"] = unknown(why)
            entry["available_cycles"] = unknown(why)
            entry["eta"] = unknown(why)
            entry["counter_set_complete"] = None
        per_run[r.workload] = entry
        out["runs"].append(entry)

    unreadable = sorted(w for w, e in per_run.items() if e["eta"]["state"] != MEASURED)
    if mixed:
        why = ("the supplied runs were counted over DIFFERENT engine sets, so their totals are not "
               "over one axis and summing them would build a corpus figure out of two instruments")
    elif unreadable:
        why = (f"run(s) {unreadable} yielded no counter reading; UNKNOWN propagates -- one unmeasured "
               "run leaves the corpus operator unestablished rather than partially derived, since "
               "dropping it reweights the corpus towards whatever was measurable")
    else:
        why = ""

    if why:
        out["engine_axis"] = {"operator": unknown(why), "eta": unknown(why),
                              "realised_cycles": unknown(why), "available_cycles": unknown(why)}
        out["kind_axis"] = {"operator": unknown(why), "eta": unknown(why)}
        out["runs_without_a_reading"] = unreadable
        return out

    out["runs_without_a_reading"] = []
    realised = sum(int(e["realised_cycles"]["value"]) for e in per_run.values())
    available = sum(int(e["available_cycles"]["value"]) for e in per_run.values())
    if available == 0:
        zero = ("no supplied run has any overlappable time (every run's second-busiest engine is busy "
                "0 cycles), so the operator is 0/0 -- undefined, not SUM")
        out["engine_axis"] = {"operator": unknown(zero), "eta": unknown(zero),
                              "realised_cycles": measured(realised), "available_cycles": measured(0)}
    else:
        eta = realised / available
        op = (Composition.SUM if eta <= COMPOSITION_TOLERANCE
              else Composition.MAX if eta >= 1.0 - COMPOSITION_TOLERANCE else Composition.PARTIAL)
        out["engine_axis"] = {
            "operator": measured(op.value, detail=f"over {len(per_run)} counter-bracketed run(s), "
                                                  f"tolerance {COMPOSITION_TOLERANCE}"),
            "eta": measured(eta, detail="realised / available overlap summed over the corpus, on the "
                                        "COUNTER-DERIVED engine axis"),
            "realised_cycles": measured(realised), "available_cycles": measured(available)}

    # The KIND axis needs each counter engine's resource kind, and a kind is NOT derivable from a
    # counter's spelling -- reading "LD" as movement is the overfit the cardinal rule forbids. So it is
    # declared by the producer or the axis refuses.
    declared_kinds = {}
    for r in readings:
        for engine, kind in (r.kind_of or {}).items():
            declared_kinds[str(engine)] = str(kind)
    missing_kinds = sorted(e for e in out["engines"] if e not in declared_kinds)
    if missing_kinds:
        why_k = (f"no resource kind is declared for counter engine(s) {missing_kinds}. A kind cannot "
                 "be derived from a counter's name, and the capability contract does not declare "
                 "these engines, so the kind axis refuses rather than inventing a grouping")
        out["kind_axis"] = {"operator": unknown(why_k), "eta": unknown(why_k)}
        out["declared_kinds"] = dict(sorted(declared_kinds.items()))
        return out

    sources, overlaps = [], {}
    for workload, e in sorted(per_run.items()):
        resources = tuple(Resource(name=n, kind=ResourceKind(declared_kinds[n]), busy_cycles=int(v))
                          for n, v in sorted(e["busy_cycles"]["value"].items()))
        sources.append(ActivitySource(
            workload=workload, total_cycles=int(e.get("total_cycles") or 0), resources=resources,
            # A combination-counter set does NOT partition the timeline once each single is summed
            # with the combinations containing it -- which is what licenses the overlap reading.
            partitioned=False,
            completion_observable=e.get("completion_observable"),
            provenance=str(e.get("provenance") or COUNTER_INSTRUMENT)))
        overlaps[workload] = int(e["realised_cycles"]["value"])
    got = composition_operator(sources, observed_overlap_cycles=overlaps)
    if isinstance(got, Unavailable):
        why_k = f"{got.what}: missing {list(got.missing)}" + (f" ({got.detail})" if got.detail else "")
        out["kind_axis"] = {"operator": unknown(why_k), "eta": unknown(why_k)}
    else:
        op, eta = got
        out["kind_axis"] = {
            "operator": measured(op.value, detail=f"over {len(sources)} counter-bracketed run(s)"),
            "eta": measured(eta, detail="realised / available overlap on the resource-KIND axis")}
    out["declared_kinds"] = dict(sorted(declared_kinds.items()))
    return out


def _measurement_basis(n_traces: int, n_counter_runs: int) -> str:
    """The one sentence that says which instruments actually ran, naming each one it did not.

    Written as a function because the failure this whole module guards against is a record whose
    headline outlives its evidence: a plan that reads as a calibration. Every branch below names what
    was NOT supplied as well as what was, so no reader has to infer the absence from a missing field.
    """
    if n_traces and n_counter_runs:
        return (f"{n_traces} per-cycle trace(s) through the MechanismTrace seam AND {n_counter_runs} "
                f"aggregate hardware-counter run(s) through the CounterReading seam. Two instruments, "
                f"reported separately: {INSTRUMENTS_NOT_COMPARABLE}")
    if n_traces:
        return (f"{n_traces} per-cycle trace(s) supplied through the MechanismTrace seam. No hardware "
                "combination counters were supplied, so counter_calibration is UNKNOWN throughout")
    if n_counter_runs:
        return (f"{n_counter_runs} aggregate hardware-counter run(s) through the CounterReading seam. "
                "NO per-cycle trace was supplied, so the trace-side composition, every per-capsule "
                "eta and the whole capsule cover stay UNKNOWN and ran_against_traces is False. The "
                "counter block below is a real measurement of the composition operator on the "
                "COUNTER-DERIVED engine axis, and it is not a per-cycle trace: "
                f"{INSTRUMENTS_NOT_COMPARABLE}")
    return ("NO per-cycle trace and NO hardware counter reading were supplied. The engine inventory "
            "and the regime cover below are derived and real; every eta, overlap split and per-engine "
            "busy count is UNKNOWN and no mechanism is calibrated. This record is a PLAN, not a "
            "calibration")


def calibrate(*, target: str, contract: Mapping,
              traces: Sequence[MechanismTrace] = (),
              counter_readings: Sequence[CounterReading] = (),
              corpus_regimes: Mapping | None = None,
              regime_by_capsule: Mapping[str, Mapping] | None = None,
              fsm_registers: Sequence = (),
              declared_idle_value: str | None = None,
              points_per_cell: int = POINTS_PER_CELL,
              provenance: Mapping | None = None,
              notes: str = "") -> dict:
    """Measure the mechanisms an analytical model needs, and report what nothing could measure.

    ``traces`` may be empty. That is the PLAN mode and it is a real answer: the engine inventory and
    the regime cover are derivable without running anything, so a run with no traces reports which
    mechanisms are uncalibratable on this target at all, which capsules the cover would spend the
    cycle-accurate tier on, and every eta as UNKNOWN. It never reports a calibration that did not
    happen -- ``ran_against_traces`` is False and every reading is an :func:`unknown`.

    ``counter_readings`` is the SECOND, independent seam (:class:`CounterReading`), for a target whose
    RTL counts its own engine combinations. It fills ``counter_calibration`` and nothing else: it does
    not enter the capsule cover, the engine inventory, the idle calibration or ``composition``, and
    ``ran_against_traces`` stays False when no trace was supplied however many counter runs there
    were. The two instruments are reported side by side and never merged -- see
    :data:`INSTRUMENTS_NOT_COMPARABLE`.
    """
    from merlin.targetgen import memory_regime as MR

    idle = calibrate_idle(traces, declared_idle_value=declared_idle_value)
    inventory = engine_inventory(contract, traces, idle, fsm_registers=fsm_registers)
    readings = {t.capsule: _reading(t, inventory, idle) for t in traces}

    cells = required_cells(inventory, corpus_regimes or {})
    filled = select_calibration_set(cells, readings=readings, regime_by_capsule=regime_by_capsule,
                                   traces=traces, points_per_cell=points_per_cell)
    selected = sorted({c for cell in filled for c in cell.capsules})

    fitted = sorted({cell.key for cell in filled
                     if cell.axis == MEMORY_REGIME_AXIS and cell.state == CALIBRATED})
    not_fitted = sorted({cell.key for cell in filled
                         if cell.axis == MEMORY_REGIME_AXIS and cell.state != CALIBRATED})
    measured_regimes = sorted({(readings[c].get("regime") or {}).get("value", {}).get("regime")
                               for c in readings
                               if (readings[c].get("regime") or {}).get("state") == MEASURED})

    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mechanism_calibration",
        "target": target,
        "notes": notes,
        "provenance": dict(provenance) if provenance else None,
        "idle_encoding": idle.to_dict(),
        "engine_inventory": inventory.to_dict(),
        "memory_regimes": {
            "capacity_rows": (corpus_regimes or {}).get("capacity_rows"),
            "by_regime": {k: sorted(v) for k, v in
                          sorted(((corpus_regimes or {}).get("by_regime") or {}).items())},
            "largest_working_set": (corpus_regimes or {}).get("largest_working_set"),
            "regime_order_weakest_first": list(MR.ORDER),
        },
        "calibration_set": {
            "points_per_cell": points_per_cell,
            "selection_rule": (select_calibration_set.__doc__ or "").strip().split("\n\n")[0],
            "cells": [c.to_dict() for c in filled],
            "selected_capsules": selected,
            "n_calibrated": sum(1 for c in filled if c.state == CALIBRATED),
            "n_uncovered": sum(1 for c in filled if c.state == UNCOVERED),
            "n_uncalibratable": sum(1 for c in filled if c.state == UNCALIBRATABLE),
            "uncalibratable": {c.key: c.why for c in filled if c.state == UNCALIBRATABLE},
            "uncovered": {c.key: c.why for c in filled if c.state == UNCOVERED},
        },
        "capsules": [dict(readings[c], regime=_regime_of(c, regime_by_capsule))
                     for c in sorted(readings)],
        "composition": _composition(readings, inventory),
        "coefficient_domain": {
            "regimes_with_points": fitted,
            "regimes_without_points": not_fitted,
            "regimes_measured_on_traces": [r for r in measured_regimes if r],
            "transfer_warning": ("a coefficient is fitted in the regimes listed under "
                                 "regimes_with_points and nowhere else. A model asked to score work "
                                 "in a regime not listed there must report UNKNOWN rather than "
                                 "extend the fit: measured on the interlocked target here, the "
                                 "corpus is 46/48 fits_double while 90.1% of contraction regions "
                                 "across 20 real captures land in spills"),
        },
        "counter_calibration": counter_calibration(counter_readings),
        "ran_against_traces": bool(traces),
        "n_traces": len(traces),
        "ran_against_counters": bool(counter_readings),
        "n_counter_runs": len(counter_readings),
        "measured_basis": {
            TRACE_INSTRUMENT: bool(traces),
            COUNTER_INSTRUMENT: bool(counter_readings),
            "any": bool(traces or counter_readings),
            "note": INSTRUMENTS_NOT_COMPARABLE,
        },
        "measurement_basis": _measurement_basis(len(traces), len(counter_readings)),
    }
    record["audit"] = audit(record)
    return record


def audit(record: Mapping) -> dict:
    """Walk a finished record and fail it if any refusal was spelled as a number.

    The one invariant worth machine-checking, because it is the invariant this whole module is for: an
    entry in state :data:`UNKNOWN` must carry ``value: None`` and a non-empty ``why``, and an entry in
    state :data:`MEASURED` must carry a value. Returns the violations rather than raising, so a report
    that breaks the rule is still written -- with the breach recorded in it.
    """
    problems: list[str] = []

    def walk(node, path: str) -> None:
        if isinstance(node, Mapping):
            state = node.get("state")
            if state == UNKNOWN:
                if node.get("value") is not None:
                    problems.append(f"{path}: state=unknown but carries a value "
                                    f"{node.get('value')!r} -- an unmeasurable thing reported as "
                                    "measured")
                if not node.get("why"):
                    problems.append(f"{path}: state=unknown with no reason; a refusal without its "
                                    "reason is indistinguishable from a measured zero")
            elif state == MEASURED and node.get("value") is None:
                problems.append(f"{path}: state=measured with value None")
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(record, "")
    return {"ok": not problems, "violations": problems}
