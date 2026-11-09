"""Tier result record and its stable serialized measurement fields."""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TierResult:
    tier: str
    status: str  # pass | fail | skipped | unavailable
    mandatory: bool
    reason: str | None = None
    cache_unavailable: str = ""  # why this tier's certificate could not even be LOOKED UP.
    # Distinct from a lookup that missed: both used to render as an
    # absence, which is how a certificate cache that had never once
    # fired for two of three targets went unnoticed. Rides the
    # result so a reader sees it without re-deriving anything.
    cycles: int | None = None
    derived_from_rtl: bool = False
    cycle_accurate: bool = False
    evidence: str | None = None
    timing: dict | None = None  # {build_s, sim_active_s, oracle_wait_s} — active vs waiting
    gflops: float | None = None  # perf (SIMT); None -> omitted so systolic output is unchanged
    pct_fp_peak: float | None = None
    timing_observations: list | None = None
    # FINER TIMING THE ORACLE COULD ACTUALLY SEE: a list of
    # {quantity, value, unit, concurrent, note} entries (per-unit
    # busy, per-op latency, per-cycle activity). Every functional
    # run already executes on a real oracle, and discarding what it
    # timed throws away calibration evidence the run already paid
    # for. An adapter with NO timing capability emits nothing here
    # — not the key, not a list of zeros, because "not reported"
    # and "cost nothing" are different facts. None -> omitted, so
    # an adapter that carries none is byte-identical to before.
    counters: dict | None = None  # COUNTS the oracle reported alongside the cycle total — bytes
    # moved, reuse hits, commits. Deliberately NOT `utilization`:
    # those are fractions of a cycle window and these are integers
    # of a different kind, and filing a byte count under a name
    # that means "fraction of time" is how a number gets read as
    # something it is not. A model that computes these and an
    # adapter that drops them are indistinguishable downstream,
    # which is what happened: the arc oracle has reported movement
    # and residency counts all along and the adapter returned only
    # cycles. None -> omitted, so a target reporting none is
    # byte-identical to before.
    utilization: dict | None = None  # WHERE THE TIME WENT, as fractions of the oracle's own cycle
    # window (warp occupancy, per-unit busy, memory conflicts).
    # Latency says a kernel is slow; this says why, which is the
    # difference between an actionable result and a number. Shape is
    # the target's to define; None -> omitted, so a target whose
    # oracle reports no counters is byte-identical to before.
    oracle_ceiling: dict | None = None
    # WHY THIS TIER WAS NOT BOUGHT, when a per-capsule oracle-tier
    # ceiling declined it: the cap, where it came from (declared /
    # derived from measured cost / unpriced), the budget that set it,
    # the `extends` sibling the claim rests on, and the affordability
    # calculation. A capped tier is ALWAYS recorded -- `skipped` with
    # this block -- never omitted: a tier with no record is not
    # evidence, and an absent key is what made two model capsules read
    # as though they had never been graded.
    # See merlin.targetgen.tier_policy.oracle_ceiling.
    not_applicable: bool = False  # tier honestly N/A for this capsule's datatype (e.g. the integer
    # L0/L1 floor on a float datapath) — a legitimate skip, not a
    # not_run_is_not_pass violation (unlike an unavailable RTL oracle)
    budget_deferred: bool = False  # tier APPLIED and was DELIBERATELY NOT PAID FOR: this capsule is
    # outside the derived covering set and the certify budget is gone.
    # Never a pass, never a fail — the capsule was screened, not
    # certified, and it says so by name.
    fidelity: str | None = None  # WHAT THE ORACLE ACTUALLY WAS, in the oracle's own words:
    # elaborated_rtl | rtl_derived_model | functional_model. The tier
    # NAME cannot carry this — one target's L3 is Verilator and
    # another's is a model — so a reader of the record could not tell
    # a hardware verdict from a model one.

    concurrency: dict | None = None  # WHAT ELSE THE HOST WAS DOING WHILE THIS WAS MEASURED --
    # {workers, nproc, load_avg, serial}, stamped at the moment of
    # measurement. CYCLES are concurrency-invariant (verified:
    # identical serial and at 16 workers); WALL TIMES are not (the
    # same query measured 3.7 s serial and 23.4 s at 16 workers, a
    # 6.3x spread). So every ``timing`` block is only interpretable
    # next to the concurrency it was taken at, and a block without
    # one is not comparable with any other. Absent -> the run
    # predates the stamp; see merlin.perf.observations.
    # concurrency_of, which marks that rather than assuming it.
    submission: dict | None = None  # WHICH SUBMISSION these numbers are about, stated rather than
    # inferred from where the file sits. Cycles are a property of
    # the submission, not of the capsule: the SAME capsule has
    # measured 1090 / 3078 / 8889 across three submissions of one
    # task, an 8.2x spread on identical inputs. A path heuristic
    # that mis-keys them pools three different programs into one
    # "latency", so the field is explicit and the heuristic becomes
    # the fallback it should always have been.
    timing_capability: dict | None = None
    # WHAT THE INSTRUMENT COULD AND COULD NOT SEE, beside the
    # observations it produced: which units went unread, whether
    # its buckets partition the timeline, its alias accounting, and
    # everything it refused. An unread signal is UNMEASURED, never
    # zero, and this is where that distinction is recorded.

    engine: str | None = None  # WHICH SIMULATOR ANSWERED, as a first-class field.
    # It was recorded nowhere: the only trace of it was the `evidence`
    # FILENAME (`<engine>_console.log`), so telling a GSIM cert from a
    # Verilator one meant string-parsing a path, and every consumer
    # that did not (the verdict, the perf ledger, cert_cost) simply
    # could not see which engine produced the seconds it was fitting.
    # A cert tier is a fidelity, not a simulator -- which is exactly
    # why the simulator has to be written down.
    engine_selection: dict | None = None
    # HOW that engine was chosen and what it was chosen OVER, from
    # `rtl_engine_policy.select`: {engine, fidelity, reason,
    # passed_over, considered}. Falling back to the slow engine was
    # SILENT -- the run recorded a cert and not the fact that the fast
    # one was unavailable, so nobody learned there was anything to fix.
    sim_provenance: dict | None = None
    # WHICH BUILD of that simulator: binary digest, and the build
    # receipt binding it to an RTL revision when one exists. A result
    # that claims a hardware verdict must record which hardware
    # revision it came from, and a prebuilt simulator binary is the
    # one input to that verdict nothing else in the record identifies.
    console_log: str | None = None  # The sim console FILE (name under the capsule's artifacts dir) and
    console_bytes: int | None = None  # its size. `evidence` already names it, but only by convention;
    # a perf-tuning pass reading these records should not have to
    # reconstruct the filename to find the log it needs.

    carried: dict | None = None  # THIS TIER WAS NOT EXECUTED IN THIS RUN. Present only on a record
    # whose verdict was CARRIED from an earlier execution of the same
    # program on the same instrument (merlin.targetgen.tier_cache):
    # it names the execution identity and instrument digest the
    # verdict was earned under, when, and by which run. A reused
    # certificate reported as a fresh measurement is not a saving, it
    # is a false claim about what this run did -- so the record says
    # so, and `to_dict` additionally stamps `measured_now: false` on
    # a carried record and `true` on every executed one, so the
    # distinction survives into anything that reads only the JSON.
    toolchain: str | None = None  # WHICH PROGRAM was graded, as reported by the adapter. A block-
    # scaled MX capsule is graded on the harness's own reference MX
    # kernel rather than the submission, so a pass there measures the
    # fixture. Recording it keeps a score decomposable instead of
    # silently overstating the backend by the size of the MX set.

    def to_dict(self) -> dict:
        d = {
            "status": self.status,
            "mandatory": self.mandatory,
            "not_run_is_not_pass": True,
            "reason": self.reason,
            "cycles": self.cycles,
            "derived_from_rtl": self.derived_from_rtl,
            "cycle_accurate": self.cycle_accurate,
            "evidence": self.evidence,
            "timing": self.timing,
        }
        if self.cache_unavailable:
            d["cache_unavailable"] = self.cache_unavailable
        if self.not_applicable:
            d["not_applicable"] = True
        if self.budget_deferred:
            d["budget_deferred"] = True
        if self.oracle_ceiling:
            d["oracle_ceiling"] = dict(self.oracle_ceiling)
        if self.fidelity:
            d["fidelity"] = self.fidelity
        # perf fields ride the result ONLY when populated (SIMT) — keeps systolic output byte-identical.
        if self.gflops is not None:
            d["gflops"] = self.gflops
        if self.pct_fp_peak is not None:
            d["pct_fp_peak"] = self.pct_fp_peak
        if self.counters:
            d["counters"] = dict(self.counters)
        if self.utilization:
            d["utilization"] = dict(self.utilization)
        if self.timing_observations:
            d["timing_observations"] = list(self.timing_observations)
        if self.timing_capability:
            d["timing_capability"] = dict(self.timing_capability)
        # Concurrency and submission ride the record ONLY when they were established. An absent
        # field is "unrecorded", which is the honest state for every run taken before the stamp
        # existed; a defaulted one would be a claim.
        if self.concurrency:
            d["concurrency"] = dict(self.concurrency)
        if self.submission:
            d["submission"] = dict(self.submission)
        # unchanged. This is what separates "the submission passed" from "the harness fixture passed":
        # a block-scaled MX capsule is graded on the reference MX kernel, not the submitted backend.
        if self.toolchain:
            d["toolchain"] = self.toolchain
        # SIMULATOR TELEMETRY. Omitted when unset, so a record from an adapter that reports none is
        # byte-identical to before; present, it is what a perf-tuning pass reads to know which engine
        # produced these cycles and seconds, which build of it, and where its console went.
        if self.engine:
            d["engine"] = self.engine
        if self.engine_selection:
            d["engine_selection"] = dict(self.engine_selection)
        if self.sim_provenance:
            d["sim_provenance"] = dict(self.sim_provenance)
        if self.console_log:
            d["console_log"] = self.console_log
        if self.console_bytes is not None:
            d["console_bytes"] = self.console_bytes
        # WAS THIS MEASURED NOW? Stamped on EVERY record, not only on a carried one: a reader that has
        # to infer "measured" from the ABSENCE of a key cannot tell a fresh record from one written by
        # a version that did not know about carrying. Both states are asserted.
        d["measured_now"] = self.carried is None
        if self.carried:
            d["carried"] = dict(self.carried)
        return d
