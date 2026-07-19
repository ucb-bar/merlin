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
from pathlib import Path
from typing import Any, Callable

from ..common.yaml import write_yaml
from ..kernels.compare import RvvFingerprint, compare_fingerprints
from ..kernels.rvv_knobs import propose_forks
from .fork_from_action import propose_forks_from_cca
from .from_strategy import mint_fork
from .registry import load_rvv_package
from .runner import certify_rvv
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


def _cca_divergences(run_dir: Path, expert_cca, op_key: dict) -> list:
    """Lift OUR emitted CCA from a run's objdump.txt (no toolchain re-run) and diff it against the
    expert CCA -> the CCA Divergences that drive the CCA-native proposer. [] if no objdump."""
    objd = Path(run_dir) / "generated" / "objdump.txt"
    if not objd.is_file():
        return []
    from ..kernels import cca, cca_compare
    from ..kernels.decode import rvv
    ours = cca.lift_asm(rvv.decode_text(objd.read_text()), op=str(op_key.get("op", "matmul")),
                        source="ours", undefined_symbols=_undef_syms(run_dir))
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


def _score(result: dict, run_dir: Path, curated: RvvFingerprint, op_key: dict) -> dict:
    """Attach gate_ok + structural_match + divergences to a certify result."""
    gate_ok = bool((result.get("correctness") or {}).get("gate_ok"))
    cycles = next((m.get("cycles") for m in result.get("measurement", [])
                   if m.get("target") == "spike"), None)
    # real-silicon wall time (K1) when the beam ran the k1 target — the REAL speedup signal (vs the
    # structural_match proxy / spike functional cycles). None when k1 was not a target.
    k1_wall = next((m.get("wall_ns") for m in result.get("measurement", [])
                    if m.get("target") == "k1"), None)
    sm, divs = 0.0, []
    objd = run_dir / "generated" / "objdump.txt"
    if objd.is_file():
        gen_fp = RvvFingerprint.from_objdump(objd.read_text(), op_key, run_dir.name)
        cmp = compare_fingerprints(curated, gen_fp)
        sm, divs = cmp["structural_match"], cmp["divergences"]
    else:  # mock / no-objdump path: trust result fields if present
        sm, divs = result.get("structural_match", 0.0), result.get("divergences", [])
    return {"gate_ok": gate_ok, "structural_match": sm, "cycles": cycles,
            "k1_wall_ns": k1_wall, "divergences": divs}


def run_beam(seed_pkg: str | Path, model_dir: str | Path, curated_text: str, op_key: dict, *,
             runs_root: str | Path, out_root: str | Path = "out/artifacts/targets",
             width: int = 3, depth: int = 2, top_k: int = 2, target: str = "rvv",
             timestamp: str = "run", targets: tuple[str, ...] = ("spike",),
             baseline_run_dir: str | Path | None = None,
             certify_fn: Callable = certify_rvv, proposer: Callable | None = None,
             expert_cca=None, loader: Callable = load_rvv_package, minter: Callable = mint_fork,
             max_workers: int | None = None, sweep_fn: Callable = run_sweep,
             expert_wall_ns: float | None = None
             ) -> dict[str, Any]:
    """Run the beam. Returns {best, nodes, deferred, tree_path}. ``curated_text`` is the expert
    kernel C source for this op (the structural target); ``op_key`` = {op,dtype,shape_regime}.

    Target-plugin seams (all injectable; defaults are the RVV instantiation — see
    rvvgen/TARGET_PLUGIN.md): ``loader(package_dir) -> pkg`` (pkg must expose ``.run_id`` and
    ``.knobs``), ``minter(parent, overrides, ...) -> Path`` (render+write a fork package),
    ``proposer(divergences, knobs) -> [ForkProposal]`` (the lever/tuning policy), and
    ``certify_fn(**job) -> result`` (build+run+gate). A new target reuses this engine unchanged by
    supplying its own four callables."""
    runs_root = Path(runs_root)
    curated = RvvFingerprint.from_curated(curated_text, op_key, "curated")
    # CCA mode: when an expert CCA is supplied, drive the search from OUR-vs-EXPERT CCA divergences
    # via the CCA-native proposer (whose proposals carry their CompilerAction, so the per-fork audit
    # fires). Otherwise the legacy motif-string fingerprint router (backward compatible).
    cca_mode = expert_cca is not None
    if proposer is None:
        proposer = propose_forks_from_cca if cca_mode else propose_forks

    def certify_and_score(pkg_dir: Path, run_id: str, parent_rid: str | None,
                          lever: str, evidence: list[str], d: int) -> dict:
        res = certify_fn(package_dir=str(pkg_dir), model_dir=str(model_dir),
                         runs_root=str(runs_root), run_id=run_id, targets=targets,
                         baseline_run_dir=(str(baseline_run_dir) if baseline_run_dir else None))
        sc = _score(res, runs_root / run_id, curated, op_key)
        return {"run_id": run_id, "package_dir": str(pkg_dir), "parent_run_id": parent_rid,
                "lever": lever, "evidence": evidence, "depth": d, **sc}

    nodes: list[dict] = []
    deferred: list[dict] = []           # recorded lever-2/3 work-items the beam can't auto-apply

    # BB0 freeze-assert: snapshot the frozen baseline BEFORE any fork is minted. The beam forks into
    # fresh dirs (from_strategy.mint_fork writes a NEW package; the seed is read-only), so this digest
    # MUST match at the end — a mismatch means something mutated the control and the run is void.
    seed_digest_pre = _seed_digest(seed_pkg)

    # generation 0: the seed (e.g. hand_v0)
    seed = loader(seed_pkg)
    seed_node = certify_and_score(Path(seed_pkg), f"{seed.run_id}__beam", None, "seed",
                                  ["baseline"], 0)
    nodes.append(seed_node)
    parents = [(seed, seed_node)] if seed_node["gate_ok"] else []

    # The seed's real-silicon wall time is the baseline every fork's REAL speedup is measured against
    # (when the beam ran the k1 target). fail-closed: no baseline wall -> no real speedup credit.
    seed_k1_wall = seed_node.get("k1_wall_ns")

    def _real_speedup(node: dict) -> float | None:
        w = node.get("k1_wall_ns")
        return round(seed_k1_wall / w, 3) if seed_k1_wall and w else None

    def _attainment_vs_expert(node: dict) -> float | None:
        """The REAL scoreboard the user cares about: fork wall vs the EXPERT (XNNPACK) wall for this
        workload. >= 1.0 means we matched/beat XNNPACK; 0.56 means 56% of XNNPACK (1.8x slower). This
        is what re-targets the beam at XNNPACK instead of the naive baseline. None if no expert wall."""
        w = node.get("k1_wall_ns")
        return round(expert_wall_ns / w, 3) if expert_wall_ns and w else None

    seed_node["speedup"] = 1.0 if seed_k1_wall else None
    seed_node["attainment_vs_expert"] = _attainment_vs_expert(seed_node)

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
            divs = (_cca_divergences(runs_root / parent_node["run_id"], expert_cca, op_key)
                    if cca_mode else parent_node["divergences"])
            props = proposer(divs, parent_pkg.knobs)
            forkable = [p for p in props if p.forkable][:width]
            deferred.extend({"parent": parent_node["run_id"], "lever": p.lever,
                             "targets": p.targets, "note": p.note, "evidence": p.evidence}
                            for p in props if not p.forkable)
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
            sc = _score(res, runs_root / fork_dir.name, curated, op_key)
            node = {"run_id": fork_dir.name, "package_dir": str(fork_dir),
                    "parent_run_id": parent_rid, "lever": p.lever, "evidence": p.evidence,
                    "targets_decision": p.targets, "depth": d, **sc}
            node["speedup"] = _real_speedup(node)     # real K1 speedup vs the seed (None if no k1)
            node["attainment_vs_expert"] = _attainment_vs_expert(node)   # vs XNNPACK (the real target)
            # INERT-LEVER GUARD: did this fork's emitted code actually differ from its parent's? A
            # lever that changes nothing must not be credited with whatever the board measured --
            # that is measurement noise wearing a lever's name.
            node["emitted_digest"] = _emitted_digest(runs_root / fork_dir.name)
            _parent_digest = _emitted_digest(runs_root / parent_rid) if parent_rid else None
            node["inert"] = bool(node["emitted_digest"] and _parent_digest
                                 and node["emitted_digest"] == _parent_digest)
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
            gen_nodes.append(node)
            nodes.append(node)
        survivors = [n for n in rank_results(gen_nodes) if n["gate_ok"]][:top_k]
        parents = [(loader(n["package_dir"]), n) for n in survivors]
        if not parents:
            break

    # BB0 freeze-assert: the frozen baseline must be byte-identical to its pre-run snapshot.
    seed_digest_post = _seed_digest(seed_pkg)
    if seed_digest_post != seed_digest_pre:
        raise RuntimeError(
            f"frozen baseline {seed_pkg} was MUTATED during the beam run "
            f"(pre={seed_digest_pre[:12]} != post={seed_digest_post[:12]}); the control is void")

    ranked = rank_results([n for n in nodes if n["gate_ok"]]) or nodes
    best = ranked[0] if ranked else None
    tree = {"target": target, "seed": str(seed_pkg), "op_key": op_key,
            "baseline_frozen": {"digest": seed_digest_pre, "verified_unchanged": True},
            "width": width, "depth": depth, "top_k": top_k,
            "expert_wall_ns": expert_wall_ns,
            "best": {k: best.get(k) for k in ("run_id", "structural_match", "speedup",
                                              "attainment_vs_expert", "cycles", "lever")}
                    if best else None,
            "nodes": nodes, "deferred_work_items": deferred}
    tree_path = runs_root / "beam_tree.yaml"
    write_yaml(tree_path, tree, header="RVV beam-search tree (rvvgen.beam.run_beam)")
    return {"best": best, "nodes": nodes, "deferred": deferred, "tree_path": str(tree_path)}
