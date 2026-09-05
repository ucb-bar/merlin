"""What a cycle-accurate certification COSTS, fitted from the runs this repo has already paid for.

A capsule derived at an application's real shape is worthless if nobody can afford to certify it.
GSIM L3 is ~23x verilator and 143 of 183 capsules demand L3, so "derive a bigger capsule" and "the
corpus still runs" are in direct tension — and the sweet spot between "too small to generalize" and
"too big to simulate" is exactly what nobody can pick by eye.

So it is measured. Every graded run already records ``sim_active_s`` per capsule
(``capsule_grade``'s timing block), and the corpus records each capsule's declared operands. Joining
the two gives a cost model per target, and the shape of it is the useful part:

    gemmini, n=32 measured runs:  seconds ~= 114.8 + 0.0605 * elements   (R^2 = 0.70)

A ~115 second FLOOR that a capsule pays for existing, and ~0.06 s per operand element on top. The
floor dominates below ~1900 elements, and today's capsules are 256-512 — so the corpus is paying
almost the whole cost of a certification to exercise a 16x16 tile, and could grow roughly sevenfold
before size is what it is paying for. That is a fact about this hardware and this oracle, not a
guess, and it is why sizing belongs here rather than in a constant someone picked.

HOW WELL IT PREDICTS, measured rather than hoped. Leave-one-out over those 32 runs -- refit without
each capsule, then predict it -- gives a median absolute error of 17.5%, p90 31%, worst case 51%,
with 31 of 32 inside 50%. So it is a sizing instrument, not a stopwatch: budget with margin and
expect a capsule sized to 300 s to sometimes land near 350. That is the honest reading of an R^2 of
0.70, and it is why the fit reports ``r2`` and its sample count rather than presenting a number.

TWO REFUSALS, both deliberate:

* **No measured history, no fit.** A target nobody has certified yields ``None``, and the caller
  must then decline to promote a capsule to the cert tier rather than size it from a default. A
  default here would be a number nobody measured driving a decision somebody quotes.
* **No extrapolation.** A fit built on 256-4096 elements says nothing about 400,000. Predictions
  outside the measured range (with a small margin) return ``None``, because the honest answer to
  "how long would a capsule 100x bigger than anything we have run take" is that we do not know.

WHICH SIZE METRIC. Measured, not assumed: on the first 15-run subset the largest single operand
predicted cost at R^2 0.914 against 0.908 for total operand elements, and declared OUTPUT elements is
degenerate
because a capsule records its inputs and not its result shape. That last point matters, because the
memory-regime module reasons that a deep-K sweep is cheap on the grounds that cost tracks output
size (``memory_regime.deep_k_rows``). It is cheapER -- doubling K moved A3 only 5 s -- but the
largest operand does grow with K, so deep-K is not free and this module does not pretend it is.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CostFit", "fit_for", "max_elements_within", "predict_seconds", "capsule_elements"]

#: How far past the largest measured capsule a prediction is still honest, as a multiple. A fit is a
#: local linearisation of a simulator's behaviour, not a law; beyond this the answer is "unknown".
_EXTRAPOLATION_MARGIN = 2.0

#: Fewest measured capsules a fit may rest on. Two points define a line through anything.
_MIN_SAMPLES = 5


@dataclass(frozen=True)
class CostFit:
    """``seconds ~= intercept_s + per_element_s * elements``, with the evidence it rests on."""

    target: str
    intercept_s: float
    per_element_s: float
    r2: float
    n_samples: int
    elements_min: int
    elements_max: int
    metric: str = "written_output_elements"
    sources: tuple[str, ...] = ()
    #: The engines whose samples this fit rests on. More than one means the fit is a MIXTURE: two
    #: elaborated-RTL engines answer the same capsule at the same fidelity roughly 26x apart, so a
    #: line through both prices a capsule at neither engine's cost. Empty string = engine unrecorded.
    engines: tuple[str, ...] = ()

    @property
    def mixed_engines(self) -> bool:
        """Whether this fit averages measurements from more than one named engine.

        A caller sizing a budget for a SPECIFIC engine must not use a mixed fit -- refit with
        ``fit_for(..., engine=...)``. Reported rather than refused, because a history that predates the
        discriminator is legitimately unattributed and must keep working.
        """
        return len({e for e in self.engines if e}) > 1

    @property
    def floor_dominates_below(self) -> int:
        """Elements below which the fixed cost exceeds the size-dependent cost.

        The number that says how much bigger a capsule can get before it is paying for its size
        rather than for existing at all -- i.e. where representativeness is nearly free.
        """
        if self.per_element_s <= 0:
            return 0
        return int(self.intercept_s / self.per_element_s)

    def to_dict(self) -> dict:
        return {
            "target": self.target, "intercept_s": round(self.intercept_s, 3),
            "per_element_s": round(self.per_element_s, 6), "r2": round(self.r2, 4),
            "n_samples": self.n_samples,
            "measured_range_elements": [self.elements_min, self.elements_max],
            "metric": self.metric, "sources": list(self.sources),
            "floor_dominates_below_elements": self.floor_dominates_below,
        }


def _linalg_result_elements(mlir_text: str) -> int:
    """Elements in the entry function's result types, for a linalg-on-tensors capsule.

    Parsed through the repo's own linalg frontend rather than scanned: a shape read off a text line is
    exactly the brittleness the no-regex rule exists to prevent, and a wrong extent here would misprice
    a certification. The frontend is also what knows which dialects to register -- a bare builtin
    context fails on the first `tensor.empty` with "does not have a custom format", so registering them
    here would be a second, drifting copy of that list.

    Returns 0 when the module cannot be parsed, so the caller reports the capsule as unpriceable rather
    than as free.
    """
    try:
        from merlin.frontends.linalg_mlir import parse_mlir_text
    except ImportError:
        return 0
    try:
        module = parse_mlir_text(mlir_text)
    except Exception:                              # noqa: BLE001 -- unparseable is not zero-cost
        return 0
    total = 0
    for op in module.walk():
        if op.name != "func.func":
            continue
        ftype = op.properties.get("function_type") or op.attributes.get("function_type")
        outputs = getattr(ftype, "outputs", None)
        entries = outputs.data if outputs is not None and hasattr(outputs, "data") else ()
        for t in entries:
            shape = getattr(getattr(t, "shape", None), "data", None)
            if not shape:
                continue
            n = 1
            for d in shape:
                n *= int(getattr(d, "data", d))
            total += n
    return total


def capsule_output_elements(interface_mlir_text: str) -> int:
    """Total elements a capsule WRITES OUT, across every terminal write in its interface.

    THE MEASURED COST DRIVER, and the reason this exists beside `capsule_elements`. A calibration
    ladder holding the lhs at one tile while the weight grew 64x measured cycle-accurate seconds
    scaling with the written output and not with the operands: x1.98 then x2.06 against output x2 and
    x2, r2 0.9998, at a near-constant 0.347 s per element with no fixed floor. Independently confirmed
    outside the ladder on a two-commit resident-reuse capsule at 0.3409 s/element.

    A TERMINAL WRITE is a command whose ``dst`` no later command reads -- i.e. a program output rather
    than an intermediate. Deriving it that way, instead of looking for a ``COMMIT``, is what makes the
    metric work for the whole corpus: an epilogue-only capsule (``BIAS_ADD``) and a movement capsule
    (``MOVEMENT``) write their result directly with no commit at all, and asking only about commits
    priced 85 of 295 L3-demanding capsules at zero -- reporting "commits nothing measurable" for
    capsules that plainly write a 16x16 tensor.

    Summed rather than maximised, because a capsule that writes twice pays for both. A contraction's
    extent comes from the grammar's own ``_commit_out_shape`` (so a pooled commit contributes its
    POOLED extent, and this file does not become a second copy of that formula); every other terminal
    write takes the extent of the operand it reads, which is what an elementwise stage produces.
    """
    from merlin.targetgen.contract import interface_emit as IE

    cb = IE.parse_interface_mlir(interface_mlir_text)
    commands = list(cb.get("commands") or ())
    tensors = cb.get("tensors") or {}
    if not commands:
        # A LINALG-ON-TENSORS capsule, not a merlin_iface program: the two grammars are both first
        # class here, and 54 of 295 L3-demanding capsules are the linalg kind. Its outputs are the
        # entry function's RESULT types, read with a real parser rather than by scanning the text.
        return _linalg_result_elements(interface_mlir_text)

    def _read_names(cmd: dict) -> set[str]:
        ops = dict(cmd.get("operands") or {})
        ops.pop("dst", None)
        return {str(v) for v in ops.values() if isinstance(v, str)}

    total = 0
    for i, cmd in enumerate(commands):
        dst = (cmd.get("operands") or {}).get("dst")
        if not dst:
            continue                               # a lifetime op writes nothing
        if any(dst in _read_names(later) for later in commands[i + 1:]):
            continue                               # an intermediate, not an output
        if cmd.get("opcode") == "COMMIT":
            src = (cmd.get("operands") or {}).get("src")
            if src:
                m, n = IE._commit_out_shape(cb, src, cmd.get("attributes") or {})
                total += int(m) * int(n)
                continue
        # Every other terminal write produces the extent of what it read.
        for key in ("src", "lhs", "q", "ifm", "a"):
            name = (cmd.get("operands") or {}).get(key)
            spec = tensors.get(name) if name else None
            shape = (spec or {}).get("shape")
            if shape:
                n = 1
                for d in shape:
                    n *= int(d)
                total += n
                break
    return total


# --- the measured cost law -------------------------------------------------------------------------
# Four cycle-accurate runs at 512/1024/2048/4096 written elements took 177.2 / 351.6 / 723.1 / 1682.6
# seconds. Seconds per element is NOT constant across that: 0.346, 0.343, 0.353, 0.411 -- flat over
# the first three rungs and 16% higher on the fourth. The honest form is a mild power law,
#
#     seconds = 0.20509 * output^1.0782          (log-log r2 0.9976, every point within 5.4%)
#
# and the exponent matters because the error compounds where it is most expensive to be wrong: at the
# corpus's largest capsule (262,144 written elements) a flat 0.347 s/element predicts 25.3h where the
# power law predicts 39.6h, +57%. UNDER-predicting is the dangerous direction -- it is how a run gets
# committed to that never finishes -- so the power law is what callers get, and anything past the
# measured range is flagged as extrapolation rather than quietly returned.
MEASURED_COEFFICIENT_S = 0.20509
MEASURED_EXPONENT = 1.0782
#: Largest written-element count any of the calibration runs actually reached.
MEASURED_MAX_OUTPUT_ELEMENTS = 4096
#: Largest SINGLE OPERAND any calibration run carried -- the ladder's deepest weight was 256x256.
#:
#: The law is silent beyond this, and the silence matters. Time tracked the written output across the
#: ladder while operands grew 64x, which is why output is the metric; but every one of those runs moved
#: at most 65,536 operand elements. A class carrying a 65.5M-element operand is a THOUSAND times past
#: that, and no measurement here says what moving it costs. Treating output as the only bound there
#: would price an enormous transfer at zero, so a caller must refuse rather than extrapolate.
MEASURED_MAX_OPERAND_ELEMENTS = 65536
#: Seconds per element over the FLAT part of the ladder (512..2048). A mid-range figure for a reader,
#: never the thing to price a large capsule with -- see the exponent above.
MEASURED_S_PER_OUTPUT_ELEMENT = 0.347


def predict_seconds_from_output(output_elements: int) -> "tuple[float | None, bool]":
    """``(seconds, extrapolated)`` for a capsule writing ``output_elements``.

    ``extrapolated`` is True past the largest calibration run, and the caller is expected to say so:
    nine of the corpus's L3-demanding capsules are beyond it and account for 90% of the predicted
    bill, so silently treating the law as valid out there would put most of the total on an unstated
    guess.
    """
    if not output_elements or output_elements <= 0:
        return None, False
    secs = MEASURED_COEFFICIENT_S * float(output_elements) ** MEASURED_EXPONENT
    return secs, output_elements > MEASURED_MAX_OUTPUT_ELEMENTS


def capsule_elements(capsule_yaml: dict) -> int:
    """The size metric: the largest single declared operand, in elements.

    Chosen by measurement rather than by argument -- see the module docstring. Operands with no
    shape contribute nothing instead of raising: a capsule that declares one is not thereby
    unmeasurable, it just does not move this metric.
    """
    biggest = 0
    for operand in (capsule_yaml.get("inputs") or ()):
        n = 1
        for dim in (operand.get("shape") or ()):
            try:
                n *= int(dim)
            except (TypeError, ValueError):        # a symbolic dim contributes no size
                n = 0
                break
        biggest = max(biggest, n)
    return biggest


#: What a CERTIFICATION costs is the cycle-accurate tier's own time, never the sum over tiers. The two
#: differ by orders of magnitude on the same capsule -- measured on PC00_k64, spike at L2 took 0.009s
#: while verilator at L3 took 698.2s -- so a model fitted on the sum is fitted on whichever tiers
#: happened to run and cannot answer "how big may this capsule be and still be certifiable".
_CYCLE_ACCURATE_ONLY = "cycle_accurate_tier"
_SUMMED_LEGACY = "summed_over_tiers(legacy score file, no per-tier block)"


def _cycle_accurate_seconds(timing: dict) -> tuple[float | None, str]:
    """``(seconds, basis)`` for the cycle-accurate tier of one capsule's timing entry.

    A thin projection of :func:`_cycle_accurate_pick`. The two exist so the engine can be read off the
    SAME tier the seconds came from, without a second copy of the selection rule that could drift from
    this one -- which on this module would mean pricing one engine's run with another's name.
    """
    seconds, basis, _engine = _cycle_accurate_pick(timing)
    return seconds, basis


def _cycle_accurate_pick(timing: dict) -> tuple[float | None, str, str]:
    """``(seconds, basis, engine)`` for the cycle-accurate tier of one capsule's timing entry.

    Prefers the per-tier block and selects the tier that DECLARES itself cycle-accurate, rather than
    assuming a tier name means an oracle kind (a target may certify on any rung its contract
    declares). Falls back to the summed scalar only for score files written before the per-tier block
    existed, and says so in the basis so a fit over mixed provenance is visible rather than implied.
    """
    by_tier = timing.get("by_tier")
    if isinstance(by_tier, dict) and by_tier:
        best = None
        for name, rec in by_tier.items():
            if not isinstance(rec, dict):
                continue
            # `cycle_accurate` is the property the cost question is about; `derived_from_rtl` is
            # accepted as the older spelling of the same claim.
            if not (rec.get("cycle_accurate") is True or rec.get("derived_from_rtl") is True):
                continue
            secs = rec.get("sim_active_s")
            if isinstance(secs, (int, float)) and secs > 0:
                # Deepest reported wins if several qualify; a longer one is the binding cost.
                if best is None or secs > best[0]:
                    # The engine rides in the BASIS, which is the string every caller already keeps
                    # beside the number. A fit whose samples came from two engines that differ by 26x
                    # is then readable off its own sources instead of being a silent average.
                    eng = str(rec.get("engine") or "").strip()
                    basis = f"{_CYCLE_ACCURATE_ONLY}:{name}"
                    best = (float(secs), f"{basis}@{eng}" if eng else basis, eng)
        if best:
            return best
        return None, "no cycle-accurate tier ran for this capsule", ""
    # ⚠️ A SUMMED SAMPLE IS REFUSED, not used as a fallback. Without a per-tier block there is no way
    # to know a cycle-accurate tier ran at all, and on a target whose graded history is functional-only
    # the summed figure is milliseconds. Measured while this fallback was live: atlas fitted 13 samples
    # of ~0.01s from legacy score files and priced a 1000-element capsule at 0.008 SECONDS -- the
    # "zero reads as free" error this module exists to prevent, reintroduced by its own compatibility
    # path. An old score file is not evidence about certification cost; it is evidence that somebody
    # graded something.
    if timing.get("sim_active_s"):
        return None, ("summed over tiers with no per-tier block, so no cycle-accurate time can be "
                      "attributed; re-grade to contribute a sample"), ""
    return None, "no positive sim_active_s", ""


def _per_tier_from_result(doc: dict) -> dict:
    """A capsule_result's ``tiers`` reshaped into the ``by_tier`` block a score file carries.

    The per-capsule result is the PRIMARY record and always has per-tier timing; a score file's
    ``timing_diagnostic`` is a roll-up of it. Reading both means a cost sample comes from any capsule
    run, not only from a graded batch -- which is what a single calibration run produces, and it was
    otherwise invisible to this model.
    """
    out = {}
    for name, rec in (doc.get("tiers") or {}).items():
        if not isinstance(rec, dict):
            continue
        tm = rec.get("timing")
        if not isinstance(tm, dict):
            continue
        out[str(name)] = {"sim_active_s": tm.get("sim_active_s"),
                          "build_s": tm.get("build_s"),
                          "oracle_wait_s": tm.get("oracle_wait_s"),
                          "cycle_accurate": rec.get("cycle_accurate"),
                          "derived_from_rtl": rec.get("derived_from_rtl"),
                          # WHICH ENGINE PRODUCED THIS SECOND. Two elaborated-RTL engines answer the same
                          # capsule at the same fidelity and are NOT interchangeable as cost samples:
                          # measured on gemmini against the identical ELF, GSIM answers in 3.31 s where
                          # Verilator takes 86.83 s. A fit over a mixture of the two prices a capsule at
                          # neither engine's cost, and until this field was carried the mixture was
                          # invisible -- the record had the engine and this reshaping dropped it.
                          "engine": rec.get("engine"),
                          "evidence": rec.get("evidence")}
    return out


#: The key an unattributed sample is filed under. Deliberately not "verilator" or any engine name: the
#: 804 cycle-accurate records on disk today carry no engine at all, and guessing one from the console
#: FILENAME is the trap this discriminator exists to close -- `capsule_runner` takes `sim_name` from the
#: contract's static `tier_sim` map, which a run-time engine substitution does not update, so GSIM
#: consoles were written under Verilator's name. An unknown engine stays unknown.
UNKNOWN_ENGINE = ""


def _timing_records(target: str, root: Path | None = None,
                    extra_roots=()) -> "dict[tuple[str, str], tuple[float, str]]":
    """``(capsule, engine) -> (cycle_accurate_seconds, source)`` from every run this target has on disk.

    KEYED ON THE PAIR, not on the capsule. Keyed on the name alone, a capsule certified on two engines
    kept exactly ONE sample -- whichever file sorted last silently won -- so the same capsule's 3.31 s
    GSIM run and 86.83 s Verilator run could never both be retained, and the one that survived was
    decided by filesystem order. The engine is part of what identifies a measurement, because the two
    engines answer the same capsule at the same fidelity roughly 26x apart.

    An engine that was not recorded is filed under :data:`UNKNOWN_ENGINE` rather than assigned one, so a
    history with no discriminator behaves exactly as it did before and a mixed history stops colliding.

    Two record kinds are read, because they are written by different paths: a score file's
    ``timing_diagnostic`` (the batch grader's roll-up) and a ``capsule_result.json``'s ``tiers``
    (the per-capsule primary record, which a single-capsule run writes and a score file does not
    exist for). Later files win on a repeat, which is what "the most recent measurement" means when a
    capsule has been certified more than once. A run that never reached a cycle-accurate tier
    contributes NOTHING rather than its functional time -- a fit that absorbed those would read a
    near-zero cost for a capsule nobody certified.
    """
    from merlin.common.paths import artifacts_dir, runs_dir

    # ⚠️ BOTH BASES MUST BE TARGET-SCOPED. `runs_dir()` alone globs every target's runs, so every
    # target -- including one that does not exist -- got the same fit, and atlas capsules were priced
    # from gemmini measurements. That silently broke this module's own stated refusal, that a target
    # with no certification history has no basis for sizing. The layout is out/runs/<target>/<suite>/.
    bases = [Path(root)] if root else [artifacts_dir() / "capsule-bench" / str(target),
                                       runs_dir() / str(target)]
    bases += [Path(r) for r in extra_roots]
    out: "dict[tuple[str, str], tuple[float, str]]" = {}
    for base in bases:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            if path.name != "capsule_result.json" and not path.name.startswith("score"):
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):   # unreadable is not a measurement
                continue
            block = doc.get("timing_diagnostic")
            if isinstance(block, dict) and block:
                for name, timing in block.items():
                    if not isinstance(timing, dict):
                        continue
                    seconds, basis, engine = _cycle_accurate_pick(timing)
                    if seconds is not None:
                        out[(str(name), engine)] = (seconds, f"{path}#{basis}")
                continue
            name = doc.get("capsule")
            per_tier = _per_tier_from_result(doc)
            if name and per_tier:
                seconds, basis, engine = _cycle_accurate_pick({"by_tier": per_tier})
                if seconds is not None:
                    out[(str(name), engine)] = (seconds, f"{path}#{basis}")
    return out


def _capsule_sizes(corpus_roots) -> dict[str, int]:
    """``capsule -> written output elements`` for every capsule under the given roots.

    WRITTEN OUTPUT, not the largest operand. The operand metric this module shipped with does not
    predict: over 72 real certifications it fits at r2 0.199, and sizing against it left a held-out
    median error of 47%. Output fits at 0.92 on the same corpus and 0.9976 on a deliberate ladder,
    and an independent refit on another target's history agreed (output 0.924, max operand 0.226,
    work M*K*N 0.655).

    The mechanism is why: reduction DEPTH is nearly free while parallel EXTENT is what costs. Measured
    on the corpus, the same shape at K=16 and K=128 -- writing the same 256 elements either way --
    took 121.1s and 161.5s, so eight times the reduction bought a third more time. A metric keyed on
    the largest operand reads that as an eightfold size increase and misprices it accordingly.

    Falls back to the operand metric when a capsule's interface cannot be read, so an unparseable
    capsule contributes a worse sample rather than none, and the fit records which metric it used.
    """
    import yaml

    sizes: dict[str, int] = {}
    for root in corpus_roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for cy in base.rglob("capsule.yaml"):
            try:
                doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            name = str(doc.get("name") or cy.parent.name)
            size = 0
            ifc = cy.parent / str(doc.get("interface_mlir") or "capsule.interface.mlir")
            if ifc.is_file():
                try:
                    size = capsule_output_elements(ifc.read_text(encoding="utf-8"))
                except Exception:                  # noqa: BLE001 -- fall back, do not drop
                    size = 0
            if size <= 0:
                size = capsule_elements(doc)
            if size > 0:
                sizes[name] = size
    return sizes


def fit_for(target: str, *, corpus_roots=None, timing_root=None,
            extra_timing_roots=(), engine: str | None = None) -> "CostFit | None":
    """The cost model for ``target``, or ``None`` when nothing has been measured.

    ``None`` is a real answer and the caller must honour it: a target with no certification history
    has no basis for sizing a capsule to a time budget, and the correct response is to leave its
    capsules at the shallow tier rather than to certify a size nobody has evidence for.

    ``engine`` restricts the fit to one engine's samples. Sizing a budget for a named engine over a
    history containing two is asking for a number that describes neither: measured on the identical
    ELF, GSIM answers a capsule in 3.31 s where Verilator takes 86.83 s. Pass :data:`UNKNOWN_ENGINE`
    to fit only the records that predate the discriminator. Left unset, every sample is used and the
    engines that contributed are recorded on the fit, so a mixture is visible
    (:attr:`CostFit.mixed_engines`) instead of implied -- unattributed history stays usable, which
    matters because as of today NO record on disk carries an engine at all.
    """
    from merlin.common.paths import merlin_dir

    timings = _timing_records(target, timing_root, extra_roots=extra_timing_roots or ())
    if not timings:
        return None
    roots = list(corpus_roots) if corpus_roots else [merlin_dir() / "contract" / "capsules"]
    sizes = _capsule_sizes(roots)

    xs: list[int] = []
    ys: list[float] = []
    sources: set[str] = set()
    engines: set[str] = set()
    for (name, rec_engine), (seconds, source) in sorted(timings.items()):
        if engine is not None and rec_engine != engine:
            continue
        size = sizes.get(name)
        if not size:
            continue
        xs.append(size)
        ys.append(seconds)
        sources.add(source)
        engines.add(rec_engine)
    if len(xs) < _MIN_SAMPLES or len(set(xs)) < 2:
        return None                                # a line through one x tells you nothing

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - (sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / ss_tot) if ss_tot else 0.0
    return CostFit(target=str(target), intercept_s=intercept, per_element_s=slope, r2=r2,
                   n_samples=n, elements_min=min(xs), elements_max=max(xs),
                   sources=tuple(sorted(sources)), engines=tuple(sorted(engines)))


def predict_seconds(fit: "CostFit | None", elements: int) -> "float | None":
    """Predicted certification seconds, or ``None`` when the question is outside the evidence.

    Refuses below zero elements and above the measured range times the margin. A prediction the fit
    cannot support is not a large number, it is an absence -- reporting one anyway is how a capsule
    nobody could afford gets scheduled on the strength of arithmetic.
    """
    if fit is None or elements <= 0:
        return None
    if elements > fit.elements_max * _EXTRAPOLATION_MARGIN:
        return None
    return fit.intercept_s + fit.per_element_s * float(elements)


def max_elements_within(fit: "CostFit | None", budget_s: float) -> "int | None":
    """The largest capsule whose predicted certification fits ``budget_s``.

    ``None`` when there is no fit, when the budget cannot even cover the fixed floor (no capsule of
    any size fits, which is a statement about the budget rather than about the shape), or when the
    answer would lie outside the measured range -- in which case it is clamped to the range and the
    caller gets the largest size the evidence actually supports.
    """
    if fit is None or budget_s <= 0:
        return None
    if budget_s <= fit.intercept_s:
        return None                                # the floor alone exceeds the budget
    if fit.per_element_s <= 0:
        return int(fit.elements_max)               # size did not move cost over the measured range
    raw = int((budget_s - fit.intercept_s) / fit.per_element_s)
    ceiling = int(fit.elements_max * _EXTRAPOLATION_MARGIN)
    return max(1, min(raw, ceiling))

# ---------------------------------------------------------------------------------------------------
# sizing by WORK rather than by shape
# ---------------------------------------------------------------------------------------------------
# A shape metric is a proxy, and a weak one: measured over 72 real gemmini certifications, seconds vs
# `max_operand_elements` has r2 0.20, and the best of five shape candidates (`output_elements`) only
# reaches 0.33. An RTL simulator's time is not spent on a tensor's declared extent, it is spent
# advancing cycles -- so the honest independent variable is the cycle count, and it happens to be
# something we can obtain almost free. Measured on the calibration ladder: the FUNCTIONAL tier costs
# 0.006-0.008s and reports a cycle count, and the cycle-accurate tier's count tracks it at a roughly
# stable ratio. So a capsule that has only ever run at L2 can still be sized for L3.
#
# This does not replace the shape fit; a capsule that has never run at all has no cycles either, and
# the shape fit is the only thing that can speak for it. Both are offered, and each says what it rests
# on rather than pretending to one authority.


@dataclass(frozen=True)
class CycleCostFit:
    """``seconds ~= intercept_s + per_cycle_s * cycles`` on the cycle-accurate tier."""

    target: str
    intercept_s: float
    per_cycle_s: float
    r2: float
    n_samples: int
    cycles_min: int
    cycles_max: int
    #: median ``cycle_accurate_cycles / functional_cycles`` over capsules that ran both, or None.
    functional_ratio: float | None = None
    n_ratio_samples: int = 0
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"target": self.target, "intercept_s": round(self.intercept_s, 3),
                "per_cycle_s": round(self.per_cycle_s, 6), "r2": round(self.r2, 4),
                "n_samples": self.n_samples,
                "measured_range_cycles": [self.cycles_min, self.cycles_max],
                "functional_to_cycle_accurate_ratio": (round(self.functional_ratio, 3)
                                                       if self.functional_ratio else None),
                "n_ratio_samples": self.n_ratio_samples, "sources": list(self.sources)}


def _cycle_records(target: str, root: Path | None = None, extra_roots=()) -> dict[str, dict]:
    """``capsule -> {seconds, cycles, functional_cycles, source}`` for cycle-accurate runs.

    Only a tier that DECLARES itself cycle-accurate contributes seconds and cycles; a functional
    tier's cycle count is kept separately, as the cheap predictor, and never as the cost itself.
    """
    from merlin.common.paths import artifacts_dir, runs_dir

    # Target-scoped for the reason spelled out in `_timing_records`: an unscoped runs root hands
    # every target the same measurements.
    bases = [Path(root)] if root else [artifacts_dir() / "capsule-bench" / str(target),
                                       runs_dir() / str(target)]
    bases += [Path(r) for r in extra_roots]
    out: dict[str, dict] = {}
    for base in bases:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("capsule_result.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            name = doc.get("capsule")
            if not name:
                continue
            secs = cycles = func_cycles = None
            for rec in (doc.get("tiers") or {}).values():
                if not isinstance(rec, dict):
                    continue
                tm = rec.get("timing") if isinstance(rec.get("timing"), dict) else {}
                accurate = rec.get("cycle_accurate") is True or rec.get("derived_from_rtl") is True
                c = rec.get("cycles")
                if accurate:
                    sv = tm.get("sim_active_s")
                    if isinstance(sv, (int, float)) and sv > 0 and isinstance(c, int) and c > 0:
                        if secs is None or sv > secs:
                            secs, cycles = float(sv), int(c)
                elif isinstance(c, int) and c > 0 and func_cycles is None:
                    func_cycles = int(c)
            if secs is not None:
                out[str(name)] = {"seconds": secs, "cycles": cycles,
                                  "functional_cycles": func_cycles, "source": str(path)}
    return out


def fit_cycles_for(target: str, *, timing_root=None, extra_timing_roots=()) -> "CycleCostFit | None":
    """Seconds-per-cycle for ``target``'s cycle-accurate tier, or None when too little was measured.

    ``None`` is a real answer, honoured the same way :func:`fit_for`'s is: a target nobody has timed
    cannot have its capsules sized to a budget, and the correct response is to leave them shallow
    rather than certify a size on a guess.
    """
    recs = _cycle_records(target, timing_root, extra_timing_roots)
    xs = [r["cycles"] for r in recs.values()]
    ys = [r["seconds"] for r in recs.values()]
    ratios = [r["cycles"] / r["functional_cycles"] for r in recs.values()
              if r.get("functional_cycles")]
    if len(xs) < _MIN_SAMPLES or len(set(xs)) < 2:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    icept = my - slope * mx
    sst = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - (sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys)) / sst) if sst else 0.0
    med = None
    if ratios:
        rs = sorted(ratios)
        med = rs[len(rs) // 2] if len(rs) % 2 else (rs[len(rs) // 2 - 1] + rs[len(rs) // 2]) / 2
    return CycleCostFit(target=str(target), intercept_s=icept, per_cycle_s=slope, r2=r2,
                        n_samples=n, cycles_min=min(xs), cycles_max=max(xs),
                        functional_ratio=med, n_ratio_samples=len(ratios),
                        sources=tuple(sorted({r["source"] for r in recs.values()})))


def predict_seconds_from_cycles(fit: "CycleCostFit | None", cycles: int) -> "float | None":
    """Certification seconds for a capsule expected to run ``cycles``, or None with no fit."""
    if fit is None or not cycles or cycles <= 0:
        return None
    return fit.intercept_s + fit.per_cycle_s * float(cycles)


def predict_seconds_from_functional_cycles(fit: "CycleCostFit | None",
                                           functional_cycles: int) -> "tuple[float | None, str]":
    """Estimate the cycle-accurate cost from a FUNCTIONAL run's cycle count.

    This is the cheap path the ladder exists to justify: the functional tier costs milliseconds and
    reports a cycle count, so a capsule can be priced for certification without ever being certified.
    Returns ``(seconds, basis)`` and refuses -- rather than guessing -- when no ratio was measured.
    """
    if fit is None:
        return None, "no cycle cost fit for this target"
    if not functional_cycles or functional_cycles <= 0:
        return None, "no functional cycle count to scale"
    if not fit.functional_ratio:
        return None, ("no capsule has run at BOTH tiers on this target, so the functional-to-"
                      "cycle-accurate cycle ratio is unmeasured and cannot be assumed")
    est = fit.intercept_s + fit.per_cycle_s * functional_cycles * fit.functional_ratio
    return est, (f"functional cycles x measured ratio {fit.functional_ratio:.2f} "
                 f"(n={fit.n_ratio_samples}), then {fit.per_cycle_s:.4f} s/cycle "
                 f"over a {fit.intercept_s:.0f}s floor")


def max_cycles_within(fit: "CycleCostFit | None", budget_s: float) -> "int | None":
    """The most cycles a capsule may run and still certify inside ``budget_s``.

    Clamped to the measured range for the same reason :func:`max_elements_within` is: past the
    evidence the line is an opinion.
    """
    if fit is None or budget_s <= 0 or fit.per_cycle_s <= 0:
        return None
    if budget_s <= fit.intercept_s:
        return None
    raw = int((budget_s - fit.intercept_s) / fit.per_cycle_s)
    return max(1, min(raw, int(fit.cycles_max * _EXTRAPOLATION_MARGIN)))

