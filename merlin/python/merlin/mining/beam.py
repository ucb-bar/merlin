"""Beam-search of forks — the engine that replaces hand-hunting levers. Each generation: expand
every surviving parent into forks (via a proposer over the S4 divergences), certify the whole
generation IN PARALLEL + isolated, rank (correctness -> structural_match toward the expert ->
cycles), keep top-k as parents for the next depth. Forks accumulate under artifacts/targets/; the
full lineage + scores are written to beam_tree.yaml (inspectable, resumable).

Target-agnostic by construction: parameterized by a `proposer` (deterministic gap-router by
default; an LLM tuning-agent is a drop-in) and a `certify_fn`. RVV is the first instantiation.

Ranking note: without a cycle-accurate target (K1/FireSim) the driver is structural_match (toward
the curated expert fingerprint) gated by correctness; spike cycles are a weak tiebreak only.
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable

# Board noise floor is >=1.9% (measured on the K1); a fork must beat its parent by more than this to
# count as a real win rather than measurement noise. Overridable per-run (env or run_beam param).
_DEFAULT_NOISE_MARGIN = 0.02
_NOISE_MARGIN_ENV = "MERLIN_BEAM_NOISE_MARGIN"

from ..common.yaml import write_yaml
from ..kernels.compare import RvvFingerprint, compare_fingerprints
from ..kernels.knobs import propose_forks
from .fork_from_action import propose_forks_from_cca
from .from_strategy import mint_fork
from .registry import load_rvv_package
from .runner import certify_rvv
from .baseline import UNRECORDED as _BASELINE_UNRECORDED, attainment as _baseline_attainment
from .prior import landing_prior_fn, seam_evidence_from_nodes
from .select import proposal_key, select_proposals
from .sweep import rank_results, run_sweep


# the frozen-baseline content files (the CONTROL). baseline_runs/ is a per-run measurement cache, NOT
# part of the frozen definition, so it is excluded from the freeze digest.
_SEED_FROZEN_FILES = ("schedule.mlir", "knobs.yaml", "manifest.yaml")


def _seed_digest(pkg_dir: str | Path) -> str:
    """A content digest over the frozen-baseline definition files. The beam forks FROM the baseline
    into fresh dirs and never mutates it; this digest is asserted byte-unchanged pre/post run (BB0),
    turning 'the control stayed frozen' from a convention into a checked invariant."""
    h = hashlib.sha256()
    for name in _SEED_FROZEN_FILES:
        p = Path(pkg_dir) / name
        h.update(name.encode())
        h.update(p.read_bytes() if p.is_file() else b"\0<absent>")
    return h.hexdigest()


def _escalations(action, achieved_cca, knobs: dict) -> list:
    """The next-stronger fork proposals for the axes a fork PROMISED but did NOT achieve. Mirrors
    ``mine.audit_achievement``: for each residual axis, route the next class up the FLAG→KNOB→HEURISTIC
    →PASS→CODEGEN ladder (`route_escalated`) and map it to a ForkProposal. This is what turns the beam
    from a knob-tuner into an escalation engine — when a knob leaves the divergence open, reach for the
    stronger implementation (heuristic/pass/codegen) instead of silently accepting the miss."""
    from ..kernels import action_catalog as ac
    from ..kernels.cca_compare import Divergence
    from .fork_from_action import action_to_fork

    props = []
    for axis in ac.achieved_residual(action, achieved_cca):
        want = (action.intended_facet or {}).get(axis)
        d = Divergence(axis=axis, expert=want, ours=ac._facet_value(achieved_cca, axis),
                       backend=action.backend, evidence=list(action.evidence))
        esc = ac.route_escalated(d, action.action_class)
        if esc is not None:
            props.append(action_to_fork(esc, knobs))
    return props


def _cca_divergences(run_dir: Path, expert_cca, op_key: dict, compare_fn: "Callable | None" = None) -> list:
    """Lift OUR emitted CCA from a run's objdump.txt (no toolchain re-run) and diff it against the
    expert -> the CCA Divergences that drive the CCA-native proposer. [] if no objdump.

    ``compare_fn(ours) -> [Divergence]`` replaces the single-expert diff. It exists because a whole
    model is not one kernel and one expert cannot answer every axis: an expert GEMM has no activation,
    so ``compute.activation_vectorization`` is UNCOMPARABLE against it and raises nothing however
    large the model's activation cost is. MEASURED on small_llama fp32: scalar `exp` is 16.48% of real
    model work (``__ieee754_expf`` 11.91% + ``expf`` 4.57%), and the matmul-teacher-only beam reported
    that axis as uncomparable -- the single largest fp32-specific cost was invisible to the search.
    Consulting every family teacher took the same model from 5 divergences to 9.
    """
    objd = Path(run_dir) / "generated" / "objdump.txt"
    if not objd.is_file():
        return []
    from ..kernels import cca, cca_compare
    from ..kernels.decode import rvv
    ours = cca.lift_asm(rvv.decode_text(objd.read_text()), op=str(op_key.get("op", "matmul")),
                        source="ours", undefined_symbols=_undef_syms(run_dir))
    if compare_fn is not None:
        return list(compare_fn(ours))
    return cca_compare.compare(expert_cca, ours)


def _undef_syms(run_dir: "Path") -> "tuple[str, ...] | None":
    """Undefined symbols of a run's emitted object, so the CCA envelope facet can NAME the runtime
    helpers the kernel escapes to. Without this the beam sees only that a call exists, not that it
    is `memrefCopy`, and the routed PASS that removes it never fires."""
    from ..kernels.decode.objdump import undefined_symbols
    obj = Path(run_dir) / "generated" / "model.o"
    return undefined_symbols(obj) if obj.is_file() else None


def _emitted_digest(run_dir: "Path") -> str | None:
    """Digest of a fork's EMITTED code (the objdump instruction stream), or None if absent.

    The inert-lever guard. A lever can look wired at every layer -- a distinct schedule, a clean
    UnsupportedAxis, a plausible route -- and still produce byte-identical output, in which case
    crediting its measurement is crediting noise. Two shipped levers were inert this way: ``KC``
    (the schedule contained no K-blocking at all) and ``MR`` under ``unroll_m`` (the schedule text
    differed by the unroll factor while retired instructions stayed flat across MR 2..7).

    Digesting the MNEMONIC STREAM rather than the raw text so that addresses, register allocation
    noise and symbol offsets do not mask a genuine no-op as a change.
    """
    objd = run_dir / "generated" / "objdump.txt"
    if not objd.is_file():
        return None
    from ..kernels.decode import rvv as _rvv
    stream = _rvv.decode_text(objd.read_text())
    body = "\n".join(f"{i.raw.mnemonic} {','.join(i.raw.operands)}" for i in stream.insns)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _correctness_residual(result: dict) -> float | None:
    """The worst relative error a certify result reports, or None when it reports none.

    None means UNKNOWN, never 0.0 -- a candidate whose error was not measured must not sort ahead of
    one measured at 3%. Takes the MAX across the reported metrics rather than a mean, for the same
    reason the per-element gate exists: an aggregate can look fine while individual elements are far
    out (measured: 1209% off per element at a passing cos).
    """
    c = result.get("correctness") or {}
    vals = [v for k, v in c.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and "rel" in k]
    return float(max(vals)) if vals else None


def _score(result: dict, run_dir: Path, curated: RvvFingerprint, op_key: dict,
           target: str | None = None) -> dict:
    """Attach gate_ok + structural_match + divergences to a certify result."""
    gate_ok = bool((result.get("correctness") or {}).get("gate_ok"))
    # WHICH substrate may produce which number is DECLARED per target (kernels.measurement), not chosen
    # here. The substrate labels used to be literals with an honest comment saying the choice was
    # "INHERENT to this path, not a derivable per-target fact" -- true for one target and wrong for
    # five, since a target with no such substrate then silently yields no number at all.
    #
    # The distinction the declaration preserves: more than one substrate emits a `cycles` field while
    # only one is authoritative (the other is an rdtime-derived ESTIMATE), so picking by field name
    # gets the estimate. `pick` refuses to fall back for exactly that reason.
    from ..kernels import measurement as _meas
    # The target is NAMED by the result or by the caller -- never defaulted. Defaulting it read one
    # target's declared authority for another, which does not fail closed: it produces a NUMBER, from
    # whichever substrate that other target happens to declare, attributed to this run. An unnamed
    # target is UNKNOWN, and the run says so.
    _target = result.get("target") or target
    _auth = (_meas.authority_for(_target) if _target else _meas.MeasurementAuthority(
        target="<unspecified>", declared=False,
        lookup_error="neither the certify result nor the caller named a target"))
    measurements = result.get("measurement", [])
    cycles, _cyc_from = _meas.pick(measurements, _auth, "cycles")
    k1_wall, _wall_from = _meas.pick(measurements, _auth, "wall")
    if not _auth.declared:
        # Undeclared is UNKNOWN, and unknown must not silently become somebody else's substrate. The
        # gaps ride along so a run that measured nothing says so instead of reporting zeros.
        _auth_gaps = list(_auth.gaps())
    else:
        _auth_gaps = []
    sm, divs = 0.0, []
    objd = run_dir / "generated" / "objdump.txt"
    if objd.is_file():
        gen_fp = RvvFingerprint.from_objdump(objd.read_text(), op_key, run_dir.name)
        cmp = compare_fingerprints(curated, gen_fp)
        sm, divs = cmp["structural_match"], cmp["divergences"]
    else:  # mock / no-objdump path: trust result fields if present
        sm, divs = result.get("structural_match", 0.0), result.get("divergences", [])
    # The WHOLE-MODEL objective, gated on numerics. A kernel-scoped score can look excellent while
    # the model runs at a few percent of peak, because most of the arithmetic was never claimed by
    # the schedule at all -- measured across several models, 86-89% of linalg ops never reached the
    # vectorized path. None when either factor is unknown, because an unknown factor is not 1.0, and
    # None for a fast wrong answer rather than a number with a caveat attached.
    from merlin.llvmlower import codegen_env as _codegen_env
    cov = (result.get("coverage") or {})
    objective = _meas.whole_model_objective(cov.get("claimed_mac_fraction"),
                                            cov.get("attainment"), numerics_ok=gate_ok)
    # The conditions and protocol the WALL was measured under, carried onto the node so the tree
    # records them per fork rather than leaving them in the certify record. Read from the same
    # measurement entry the wall came from, so they always describe THIS number.
    _cond, _proto = None, None
    for _m in measurements:
        if _m.get("wall_ns") is not None and _m.get("wall_ns") == k1_wall:
            _cond = _m.get("board_conditions")
            if _m.get("iters") is not None:
                _proto = {"warmup": _m.get("warmup"), "iters": _m.get("iters")}
            break
    out = {"gate_ok": gate_ok, "structural_match": sm, "cycles": cycles,
           "k1_wall_ns": k1_wall, "divergences": divs,
           "whole_model_objective": objective,
           # How WRONG it is, not just that it is wrong. Without this a repair search has nothing to
           # climb: every incorrect candidate looks identical to every other. The WORST reported
           # relative error, because a mean hides the per-element failure an aggregate cos already
           # hides (a kernel measured 1209% off on individual elements while cos looked fine).
           "correctness_residual": _correctness_residual(result),
           # Which substrate each number came from, so a reader never has to infer it. A number whose
           # provenance is inferred is a number that gets attributed to the wrong device.
           "cycles_from": _cyc_from, "wall_from": _wall_from,
           # The ENVIRONMENT the compiler ran under. A configuration that does not determine the
           # binary is not a configuration: two nodes with byte-identical knobs.yaml emitted
           # different binaries (210dbfe9a01c44aa vs 2efd837676ff75cd) and ran 1.61x apart, and
           # nothing recorded could explain it because nothing recorded this.
           "codegen_env": _codegen_env.snapshot(),
           "codegen_env_digest": _codegen_env.digest(),
           # Under WHAT the wall was measured. Two runs of the byte-identical frozen seed measured
           # 1.9915x apart with nothing in either artifact able to show it; a wall without its
           # conditions cannot be compared to a wall measured at another time.
           "board_conditions": _cond, "measurement_protocol": _proto}
    if _auth_gaps:
        out["measurement_gaps"] = _auth_gaps
    return out


def _baseline_identity(value) -> dict[str, Any] | None:
    """What the expert baseline says about itself, for the run record. A bare number says nothing,
    which is itself the fact worth writing down."""
    from .baseline import ExpertBaseline
    b = ExpertBaseline.of(value)
    if b is None:
        return None
    return {"wall_ns": b.wall_ns, "workload": b.workload, "dtype": b.dtype,
            "substrate": b.substrate, "revision": b.revision,
            "provenance_recorded": b.provenance_recorded}


def _resolve_margin(noise_margin: float | None) -> float:
    """The noise-floor margin a fork must beat its parent by. Explicit param wins; else the env
    override; else the measured board default (2% >= the >=1.9% K1 floor)."""
    if noise_margin is not None:
        return float(noise_margin)
    return float(os.environ.get(_NOISE_MARGIN_ENV, _DEFAULT_NOISE_MARGIN))


def _margin_improved(speedup: float | None, parent_speedup: float | None, margin: float) -> bool:
    """True only when the fork's speedup exceeds its parent's by MORE than the noise margin — the
    honest 'this is a real win, not board noise' test. Missing either speedup -> not a credited win."""
    if speedup is None or parent_speedup is None:
        return False
    return speedup > parent_speedup * (1.0 + margin)


def _ranked_speedup(speedup: float | None, parent_speedup: float | None,
                    margin: float) -> float | None:
    """The speedup rank_results should USE for this fork — the raw measurement gated by the noise
    floor: a fork faster than its parent but within ``margin`` (the board noise floor) is pinned to
    the parent's speed, so it ranks as a TIE rather than a noise-promoted win. A genuine win
    (> parent * (1+margin)) keeps its measured speedup; a genuine regression (< parent) keeps its
    (lower) measured speedup so it correctly sorts below the parent.

    Inert forks (byte-identical emitted code) are handled separately — excluded from the survivor set
    and demoted by the ``not_inert`` tiebreak in rank_results — and in a real run their measured
    speedup is within noise of the parent anyway, so this same clamp pins them to the parent's speed."""
    if speedup is None or parent_speedup is None:
        return speedup
    if speedup > parent_speedup * (1.0 + margin):
        return speedup                     # genuine win -> credit the real measurement
    if speedup >= parent_speedup:
        return parent_speedup              # within the noise floor above parent -> tie (no win credit)
    return speedup                         # genuine regression -> keep (sorts below the parent)


def run_beam(seed_pkg: str | Path, model_dir: str | Path, curated_text: str, op_key: dict, *,
             runs_root: str | Path, out_root: str | Path = "out/artifacts/targets",
             width: int = 3, depth: int = 2, top_k: int = 2, target: str = "rvv",
             timestamp: str = "run", targets: tuple[str, ...] = ("spike",),
             baseline_run_dir: str | Path | None = None,
             certify_fn: Callable = certify_rvv, proposer: Callable | None = None,
             expert_cca=None, compare_fn: Callable | None = None,
             pass_slot_fn: Callable | None = None,
             loader: Callable = load_rvv_package, minter: Callable = mint_fork,
             max_workers: int | None = None, sweep_fn: Callable = run_sweep,
             expert_wall_ns: float | None = None, validate_fn: Callable | None = None,
             noise_margin: float | None = None,
             prior_fn: Callable[[Any], float | None] | None = None
             ) -> dict[str, Any]:
    """Run the beam. Returns {best, nodes, deferred, tree_path}. ``curated_text`` is the expert
    kernel C source for this op (the structural target); ``op_key`` = {op,dtype,shape_regime}.

    Target-plugin seams (all injectable; defaults are the RVV instantiation — see
    mining/TARGET_PLUGIN.md): ``loader(package_dir) -> pkg`` (pkg must expose ``.run_id`` and
    ``.knobs``), ``minter(parent, overrides, ...) -> Path`` (render+write a fork package),
    ``proposer(divergences, knobs) -> [ForkProposal]`` (the lever/tuning policy), and
    ``certify_fn(**job) -> result`` (build+run+gate). A new target reuses this engine unchanged by
    supplying its own four callables.

    Two-phase objective (P1): whole-model certification per fork is slow, so the EXPLORE generations
    score forks with the cheap ``certify_fn`` (a fast proxy model), and only the surviving top-k are
    RE-CERTIFIED with ``validate_fn`` (the full whole-model bundle) before they are promoted to
    parents and before the final ``best`` is picked. ``validate_fn=None`` (default) keeps the
    single-phase behavior — survivors are promoted on their explore scores. ``validate_fn`` has the
    same signature/return contract as ``certify_fn`` (both are ``(**job) -> result`` seams).

    Noise-floor gate (P3): a fork counts as a WIN only if its speedup beats its parent's by more than
    ``noise_margin`` (default 2% >= the measured >=1.9% K1 floor; env ``MERLIN_BEAM_NOISE_MARGIN``);
    sub-margin deltas rank as ties. INERT forks (emitted code byte-identical to the parent) are
    excluded from the survivor set and can never be credited a win."""
    runs_root = Path(runs_root)
    model_dir_name = Path(model_dir).name
    margin = _resolve_margin(noise_margin)
    curated = RvvFingerprint.from_curated(curated_text, op_key, "curated")
    # CCA mode: when an expert CCA is supplied, drive the search from OUR-vs-EXPERT CCA divergences
    # via the CCA-native proposer (whose proposals carry their CompilerAction, so the per-fork audit
    # fires). Otherwise the legacy motif-string fingerprint router (backward compatible).
    cca_mode = expert_cca is not None or compare_fn is not None
    if proposer is None:
        proposer = propose_forks_from_cca if cca_mode else propose_forks

    def certify_and_score(pkg_dir: Path, run_id: str, parent_rid: str | None,
                          lever: str, evidence: list[str], d: int) -> dict:
        res = certify_fn(package_dir=str(pkg_dir), model_dir=str(model_dir),
                         runs_root=str(runs_root), run_id=run_id, targets=targets,
                         baseline_run_dir=(str(baseline_run_dir) if baseline_run_dir else None))
        sc = _score(res, runs_root / run_id, curated, op_key, target=target)
        return {"run_id": run_id, "package_dir": str(pkg_dir), "parent_run_id": parent_rid,
                "lever": lever, "evidence": evidence, "depth": d, **sc}

    nodes: list[dict] = []
    deferred: list[dict] = []           # recorded lever-2/3 work-items the beam can't auto-apply
    # How many past generations deferred each proposal for width. Band and arrival order are both
    # deterministic, so without this a proposal that lost the width cut once loses identically every
    # time -- never built, so never measured, so never able to out-rank anything. That makes the
    # reachable lever set `width`, not `width x depth`. Charged by (family, targets), since the
    # proposer rebuilds its proposal objects each generation.
    starved: Counter = Counter()
    node_by_rid: dict[str, dict] = {}   # run_id -> node, for parent-speedup lookup (the margin gate)
    # run_id -> the CompilerActions applied along that fork's lineage. Kept OUT of the node because
    # nodes are serialized to beam_tree.yaml; the node carries the seam names for inspectability and
    # this map carries the objects the selector's legality check needs.
    applied_by_rid: dict[str, list[Any]] = {}

    # BB0 freeze-assert: snapshot the frozen baseline BEFORE any fork is minted. The beam forks into
    # fresh dirs (from_strategy.mint_fork writes a NEW package; the seed is read-only), so this digest
    # MUST match at the end — a mismatch means something mutated the control and the run is void.
    seed_digest_pre = _seed_digest(seed_pkg)

    # generation 0: the seed (e.g. hand_v0)
    seed = loader(seed_pkg)
    seed_node = certify_and_score(Path(seed_pkg), f"{seed.run_id}__beam", None, "seed",
                                  ["baseline"], 0)
    nodes.append(seed_node)
    # REPAIR MODE: an incorrect seed still seeds the search, with correctness as the objective.
    #
    # It used to yield no parents at all, so the beam stopped after generation 0 and returned nothing
    # -- for any model whose BASELINE is numerically wrong, no lever could ever be tried. That is not
    # hypothetical and not rare: deepjscc went from cos 0.9176 to BIT-EXACT purely by switching to
    # per-op register blocking, i.e. the fix was a lever already in this search space. Measured on
    # lstmnetvit int8, the frozen seed reports w8a8_rel 0.250 (25% off) at cos 0.985 -- so the beam
    # reported "0 forks, best=seed, gate_ok=False" and the levers were never tried.
    #
    # Nothing can be credited a win from here: rank_results sorts correctness FIRST, so any candidate
    # that achieves gate_ok outranks every incorrect one regardless of speed.
    repair_mode = not seed_node["gate_ok"]
    parents = [(seed, seed_node)]

    # The seed's real-silicon wall time is the baseline every fork's REAL speedup is measured against
    # (when the beam ran the k1 target). fail-closed: no baseline wall -> no real speedup credit.
    #
    # In repair mode there IS no legitimate baseline: the seed computes the wrong answer, so a ratio
    # against its wall would be a speedup over a program that does not work. Suppressed outright --
    # the deliverable of a repair run is WHICH LEVER restores correctness, and the speed search is a
    # separate run seeded from the repaired package.
    seed_k1_wall = None if repair_mode else seed_node.get("k1_wall_ns")

    def _real_speedup(node: dict) -> float | None:
        w = node.get("k1_wall_ns")
        return round(seed_k1_wall / w, 3) if seed_k1_wall and w else None

    def _attainment_vs_expert(node: dict) -> float | None:
        """The REAL scoreboard: fork wall vs the EXPERT (XNNPACK) wall for this workload. >= 1.0 means
        we matched/beat the expert; 0.56 means 56% of it (1.8x slower).

        Refuses the comparison when the baseline's DECLARED identity disagrees with this run's -- two
        recorded int8 runs were scored against their fp32 sibling's wall time and both reported
        beating the expert, which was the baseline's dtype rather than our code. A bare number still
        works and is stamped `expert_baseline_provenance: unrecorded`, so it can be excluded from
        anything that claims to be a verified comparison."""
        w = node.get("k1_wall_ns")
        value, problems, recorded = _baseline_attainment(
            expert_wall_ns, w, workload=str(model_dir_name), dtype=op_key.get("dtype"))
        if problems:
            node["attainment_problems"] = list(problems)
        node["expert_baseline_provenance"] = "recorded" if recorded else _BASELINE_UNRECORDED
        return value

    seed_node["speedup"] = 1.0 if seed_k1_wall else None
    seed_node["attainment_vs_expert"] = _attainment_vs_expert(seed_node)
    seed_node["explore_speedup"] = seed_node["speedup"]   # the seed IS the explore-phase baseline
    node_by_rid[seed_node["run_id"]] = seed_node

    # P1 two-phase: validate_seed_wall is the validation-phase baseline (survivors' validated speedup
    # is measured against a VALIDATED seed, never against the cheap explore-proxy seed wall).
    validate_seed_wall: float | None = None

    def _validated_speedup(node: dict) -> float | None:
        w = node.get("k1_wall_ns")
        return round(validate_seed_wall / w, 3) if validate_seed_wall and w else None

    def _validate_node(node: dict) -> dict:
        """Re-certify a node with the full ``validate_fn`` and OVERWRITE its explore-phase scores in
        place (into a ``__validate`` sibling run dir so the explore run is left intact). Marks the
        node ``validated`` so the final ``best`` is picked from the validated pool only."""
        vrun_id = f"{node['run_id']}__validate"
        res = validate_fn(package_dir=node["package_dir"], model_dir=str(model_dir),
                          runs_root=str(runs_root), run_id=vrun_id, targets=targets,
                          baseline_run_dir=(str(baseline_run_dir) if baseline_run_dir else None))
        node.update(_score(res, runs_root / vrun_id, curated, op_key, target=target))
        node["validated"] = True
        return node

    # Validate the seed once up front (when two-phase) to fix the validation baseline wall; the
    # explore forks still measure against the proxy seed_k1_wall captured above (phase-consistent).
    if validate_fn is not None and seed_node["gate_ok"]:
        _validate_node(seed_node)
        validate_seed_wall = seed_node.get("k1_wall_ns")
        seed_node["speedup"] = 1.0 if validate_seed_wall else None
        seed_node["attainment_vs_expert"] = _attainment_vs_expert(seed_node)
        seed_node["ranked_speedup"] = seed_node["speedup"]

    counter = 0
    pending_escalations: list[tuple] = []    # (pkg, ForkProposal, parent_run_id) queued for the next gen
    for d in range(1, depth + 1):
        jobs, meta = [], []
        # Drain the previous generation's escalations FIRST: a fork that left a residual escalates the
        # unmet axis to the next-stronger class, minted on top of that fork.
        for esc_pkg, esc_prop, esc_parent_rid in pending_escalations:
            counter += 1
            fork_dir = minter(esc_pkg, esc_prop.overrides, version=d, depth=d,
                                 timestamp=f"{timestamp}_esc{counter}", source_evidence=esc_prop.evidence,
                                 lever=esc_prop.lever, target=target, out_root=out_root,
                                 generated_by_agent=False)
            jobs.append({"package_dir": str(fork_dir), "model_dir": str(model_dir),
                         "runs_root": str(runs_root), "run_id": fork_dir.name, "targets": targets,
                         "baseline_run_dir": (str(baseline_run_dir) if baseline_run_dir else None)})
            meta.append((fork_dir, esc_parent_rid, esc_prop))
        pending_escalations = []
        for parent_pkg, parent_node in parents:
            # feed the proposer the matching divergences: OUR-vs-EXPERT CCA divergences in CCA mode
            # (lifted from the parent's emitted asm), else the fingerprint divergences.
            divs = (_cca_divergences(runs_root / parent_node["run_id"], expert_cca, op_key,
                                     compare_fn=compare_fn)
                    if cca_mode else parent_node["divergences"])
            props = proposer(divs, parent_pkg.knobs)
            # SELECTION: spend this generation's width on EVIDENCE, not on the order the divergence
            # list happened to arrive in. Drops proposals illegal on this parent (a conflict, or a
            # requirement the lineage does not satisfy -- which would build and do nothing, and be
            # blamed on the action), ranks measured-helps > unmeasured > measured-refuted, and
            # round-robins by action family so width buys distinct IDEAS. Nothing is silently lost:
            # everything unbuilt lands in `deferred` with its reason.
            # The run learns from ITSELF as it goes. Every candidate built so far -- including ones
            # that were never measured -- says whether its action does what it promises, and that
            # axis costs only a build. The run's own evidence is about THIS compiler and THIS target,
            # so it outranks the injected corpus prior once there is enough of it; below that
            # threshold the caller's prior still answers.
            _own = landing_prior_fn(seam_evidence_from_nodes(nodes))

            def _prior(proposal, _own=_own, _outer=prior_fn):
                p = _own(proposal)
                if p is not None:
                    return p
                return _outer(proposal) if _outer is not None else None

            forkable, rejected = select_proposals(
                props, width=width,
                applied_actions=applied_by_rid.get(parent_node["run_id"], ()),
                prior_fn=_prior, starved_fn=lambda p: starved[proposal_key(p)])
            # age only the width losers: an illegal-on-parent rejection is a permanent verdict on
            # this lineage, not a queue position, so aging it would push a proposal that can never
            # be built ahead of ones that can.
            for r in rejected:
                if r.reason == "over_width":
                    starved[(r.family, r.targets)] += 1
            deferred.extend({"parent": parent_node["run_id"], "lever": p.lever,
                             "targets": p.targets, "note": p.note, "evidence": p.evidence}
                            for p in props if not p.forkable)
            deferred.extend({"parent": parent_node["run_id"], **r.to_dict()} for r in rejected)
            for p in forkable:
                counter += 1
                fork_dir = minter(parent_pkg, p.overrides, version=d, depth=d,
                                     timestamp=f"{timestamp}_{counter}", source_evidence=p.evidence,
                                     lever=p.lever, target=target, out_root=out_root,
                                     generated_by_agent=False)
                jobs.append({"package_dir": str(fork_dir), "model_dir": str(model_dir),
                             "runs_root": str(runs_root), "run_id": fork_dir.name,
                             "targets": targets,
                             "baseline_run_dir": (str(baseline_run_dir) if baseline_run_dir else None)})
                meta.append((fork_dir, parent_node["run_id"], p))
        if not jobs:
            break
        # sweep_fn certifies this generation's forks. Default: the in-process ThreadPool run_sweep
        # (max_workers=1 serializes it — REQUIRED for a k1 target on the single board). A chia-driven
        # sweep (BB3, chia-venv) is injected here to fan forks out as @ChiaFunction(resources={"k1":1})
        # Ray tasks — the single-slot k1 resource being the board gate across a Ray cluster (the
        # host-wide k1.board_lock flock is the stronger cross-process serialization either way).
        results = sweep_fn(jobs, certify_fn=certify_fn, max_workers=max_workers)
        gen_nodes = []
        for (fork_dir, parent_rid, p), res in zip(meta, results):
            sc = _score(res, runs_root / fork_dir.name, curated, op_key, target=target)
            _p_action = getattr(p, "action", None)
            _lineage = list(applied_by_rid.get(parent_rid, ())) + ([_p_action] if _p_action else [])
            applied_by_rid[fork_dir.name] = _lineage
            node = {"run_id": fork_dir.name, "package_dir": str(fork_dir),
                    "parent_run_id": parent_rid, "lever": p.lever, "evidence": p.evidence,
                    "targets_decision": p.targets, "depth": d,
                    # the action lineage as SEAM NAMES -- the selector reads the objects from
                    # applied_by_rid; this is the inspectable record that survives into the tree file
                    "applied_seams": [getattr(a, "target_seam", "?") for a in _lineage],
                    **sc}
            node["speedup"] = _real_speedup(node)     # real K1 speedup vs the seed (None if no k1)
            node["explore_speedup"] = node["speedup"]  # frozen explore-phase speed (survives validation)
            node["attainment_vs_expert"] = _attainment_vs_expert(node)   # vs XNNPACK (the real target)
            # INERT-LEVER GUARD: did this fork's emitted code actually differ from its parent's? A
            # lever that changes nothing must not be credited with whatever the board measured --
            # that is measurement noise wearing a lever's name.
            node["emitted_digest"] = _emitted_digest(runs_root / fork_dir.name)
            _parent_digest = _emitted_digest(runs_root / parent_rid) if parent_rid else None
            node["inert"] = bool(node["emitted_digest"] and _parent_digest
                                 and node["emitted_digest"] == _parent_digest)
            # NOISE-FLOOR MARGIN GATE: a fork is a WIN only if it beats its parent by more than the
            # measured board noise floor; sub-margin deltas (and inert forks) are pinned to the
            # parent's speed so rank_results treats them as ties, never noise-promoted wins.
            _parent_node = node_by_rid.get(parent_rid)
            _parent_spd = _parent_node.get("explore_speedup") if _parent_node else None
            node["parent_speedup"] = _parent_spd
            node["margin_improved"] = _margin_improved(node["speedup"], _parent_spd, margin)
            node["ranked_speedup"] = _ranked_speedup(node["speedup"], _parent_spd, margin)
            node_by_rid[node["run_id"]] = node
            # AUDIT: when the proposal carries its CompilerAction (CCA-native proposer) and the fork
            # emitted an objdump, record the per-step SearchStep — did the fork's asm actually ACHIEVE
            # the promised facet? (else escalate) + real-vs-fake speedup. The LLM-digestible step record.
            # The REAL K1 speedup is passed so the SearchStep's fail-closed real-vs-fake gate fires on
            # measured silicon, not a proxy.
            action = getattr(p, "action", None)
            objd = runs_root / fork_dir.name / "generated" / "objdump.txt"
            if action is not None and objd.is_file():
                from ..kernels.cca import lift_asm
                from ..kernels.decode import rvv as _rvv
                from ..kernels.search_step import make_step
                achieved_cca = lift_asm(_rvv.decode_text(objd.read_text()),
                                        op=str(op_key.get("op", "matmul")), source="fork",
                                        undefined_symbols=_undef_syms(runs_root / fork_dir.name))
                step = make_step(action, achieved_cca, correctness_ok=sc["gate_ok"],
                                 speedup=node.get("speedup"))
                node["search_step"] = step.to_dict()
                # ESCALATION: the fork left a residual (promise unmet) -> route the next-stronger class
                # for each open axis + queue it for the next generation (built on THIS fork).
                if step.residual:
                    fork_pkg = loader(str(fork_dir))
                    esc_props = _escalations(action, achieved_cca, fork_pkg.knobs)
                    node["escalations"] = [
                        {"axis": e.targets, "class": getattr(e.action, "action_class", None),
                         "seam": getattr(e.action, "target_seam", None), "forkable": e.forkable}
                        for e in esc_props]
                    pending_escalations.extend(
                        (fork_pkg, e, fork_dir.name) for e in esc_props if e.forkable)
                    # The rungs no knob or feature can express -- where the ladder used to stop and
                    # record a work-item. When a slot is supplied, run it on each one whose seam names
                    # a module it can overlay; everything else stays a recorded work-item, including
                    # the reason it is not actionable. Off by default: a slot turn costs an agent and
                    # a build, so it is never entered implicitly.
                    if pass_slot_fn is not None:
                        for e in esc_props:
                            if e.forkable or getattr(e, "action", None) is None:
                                continue
                            try:
                                outcome = pass_slot_fn(e.action, parent_run_id=fork_dir.name)
                            except Exception as exc:  # noqa: BLE001 - a slot failure is recorded,
                                # never fatal: the search's own results must survive it.
                                outcome = {"error": f"{type(exc).__name__}: {exc}"}
                            if outcome is not None:
                                node.setdefault("pass_slot", []).append(outcome)
            gen_nodes.append(node)
            nodes.append(node)
        # P3: exclude INERT forks (byte-identical emitted code to the parent) from the survivor set --
        # a no-op lever must never be promoted to a parent and propagate as if it were progress.
        survivors = [n for n in rank_results(gen_nodes)
                     if n["gate_ok"] and not n.get("inert")][:top_k]
        if repair_mode and not survivors:
            # Nothing is correct yet. Carry forward the candidates that got CLOSER than their parent,
            # so the search can climb toward correctness over generations instead of stopping after
            # one. Strictly closer: a candidate that did not improve the residual is not progress, and
            # an UNKNOWN residual is not an improvement (None is not 0).
            _par = seed_node.get("correctness_residual")
            closer = [n for n in gen_nodes
                      if not n.get("inert")
                      and n.get("correctness_residual") is not None
                      and (_par is None or n["correctness_residual"] < _par)]
            survivors = sorted(closer, key=lambda n: n["correctness_residual"])[:top_k]
        # P1 two-phase: re-certify the survivor set with the full validate_fn and re-rank on the
        # VALIDATED scores before promoting parents / picking best (explore was a cheap proxy).
        if validate_fn is not None and survivors:
            for n in survivors:
                _validate_node(n)
                n["speedup"] = _validated_speedup(n)
                n["attainment_vs_expert"] = _attainment_vs_expert(n)
                _pn = node_by_rid.get(n.get("parent_run_id"))
                _pspd = _pn.get("speedup") if _pn else None   # parent's VALIDATED speed (seed is validated)
                n["parent_speedup"] = _pspd
                n["margin_improved"] = _margin_improved(n["speedup"], _pspd, margin)
                n["ranked_speedup"] = _ranked_speedup(n["speedup"], _pspd, margin)
            survivors = [n for n in rank_results(survivors) if n["gate_ok"]][:top_k]
        parents = [(loader(n["package_dir"]), n) for n in survivors]
        if not parents:
            break

    # BB0 freeze-assert: the frozen baseline must be byte-identical to its pre-run snapshot.
    seed_digest_post = _seed_digest(seed_pkg)
    if seed_digest_post != seed_digest_pre:
        raise RuntimeError(
            f"frozen baseline {seed_pkg} was MUTATED during the beam run "
            f"(pre={seed_digest_pre[:12]} != post={seed_digest_post[:12]}); the control is void")

    # P1 two-phase: the final best is picked from the VALIDATED pool only (explore scores are a proxy
    # and are not comparable to validated walls). Falls back to the full ranking when single-phase.
    if validate_fn is not None:
        validated = [n for n in nodes if n.get("validated") and n["gate_ok"]]
        ranked = rank_results(validated) or rank_results([n for n in nodes if n["gate_ok"]]) or nodes
    else:
        ranked = rank_results([n for n in nodes if n["gate_ok"]]) or nodes
    best = ranked[0] if ranked else None
    # WAS THE SEARCH STARVED? A proposal deferred `over_width` is not a proposal the search
    # rejected -- it is one it never got to try, and it stays invisible unless someone opens the
    # tree and counts. In the run this census was added for, 41 of 44 deferrals were `over_width`,
    # including every cap refinement: only the smallest stack-cap rung was ever built, while the
    # largest is worth 1.34x. That reads as "the search converged" and is actually "the budget ran
    # out". Report it beside the result so the two cannot be confused.
    _by_reason: dict[str, int] = {}
    for _d in deferred:
        _by_reason[str(_d.get("reason") or "not_forkable")] = (
            _by_reason.get(str(_d.get("reason") or "not_forkable"), 0) + 1)
    _starved = sorted({str(_d.get("family") or _d.get("lever") or "?")
                       for _d in deferred if _d.get("reason") == "over_width"})
    deferral_census = {"total": len(deferred), "by_reason": _by_reason,
                       "over_width": _by_reason.get("over_width", 0),
                       "over_width_families": _starved,
                       "width_was_binding": _by_reason.get("over_width", 0) > 0,
                       "note": ("proposals deferred over_width were never built; a non-zero count "
                                "means the result is bounded by --width, not by the search space")}
    # DID THE FORKS ACTUALLY DIFFER? A lever that emits byte-identical code did nothing, and the
    # node already records `inert` and `emitted_digest` -- but only per node, so a search where most
    # levers are no-ops looks exactly like one where they all worked and none helped. On the run this
    # was added for, 88 forks produced 21 distinct binaries and 34 nodes were flagged inert: the
    # int8 quant pass had rewritten every linalg.matmul into a linalg.generic, so all 39 named-op
    # schedule levers matched an empty handle and silently did nothing while reporting as applied.
    _digests = [n.get("emitted_digest") for n in nodes if n.get("emitted_digest")]
    _distinct = len(set(_digests))
    _inert = sum(1 for n in nodes if n.get("inert") is True)
    effectiveness_census = {
        "nodes": len(nodes), "with_emitted_digest": len(_digests),
        "distinct_emitted": _distinct,
        "duplicate_emission_rate": (round(1.0 - _distinct / len(_digests), 3) if _digests else None),
        "inert_nodes": _inert,
        "note": ("a high duplicate_emission_rate or inert_nodes count means the levers were not "
                 "reaching the IR -- the search space is not the constraint, the lever set is"),
    }
    tree = {"target": target, "seed": str(seed_pkg), "op_key": op_key,
            "deferral_census": deferral_census,
            "effectiveness_census": effectiveness_census,
            "baseline_frozen": {"digest": seed_digest_pre, "verified_unchanged": True},
            "repair_mode": repair_mode,
            "seed_correctness_residual": seed_node.get("correctness_residual"),
            "width": width, "depth": depth, "top_k": top_k,
            "expert_wall_ns": getattr(expert_wall_ns, "wall_ns", expert_wall_ns),
            "expert_baseline": (_baseline_identity(expert_wall_ns)), "noise_margin": margin,
            "two_phase": validate_fn is not None,
            "best": {k: best.get(k) for k in ("run_id", "structural_match", "speedup",
                                              "attainment_vs_expert", "cycles", "lever")}
                    if best else None,
            "nodes": nodes, "deferred_work_items": deferred}
    tree_path = runs_root / "beam_tree.yaml"
    write_yaml(tree_path, tree, header="RVV beam-search tree (mining.beam.run_beam)")
    return {"best": best, "nodes": nodes, "deferred": deferred, "tree_path": str(tree_path),
            "deferral_census": deferral_census,
            "effectiveness_census": effectiveness_census,
            # Surfaced, not just written to the tree: a caller that cannot tell a repair run from a
            # speed run would read "no speedup" as a failed search rather than as the one honest
            # answer available from an incorrect baseline.
            "repair_mode": repair_mode,
            "seed_correctness_residual": seed_node.get("correctness_residual")}
