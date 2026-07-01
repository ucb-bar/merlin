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

from pathlib import Path
from typing import Any, Callable

from ..common.yaml import write_yaml
from ..kernels.compare import RvvFingerprint, compare_fingerprints
from ..kernels.rvv_knobs import propose_forks
from .from_strategy import mint_fork
from .registry import load_rvv_package
from .runner import certify_rvv
from .sweep import rank_results, run_sweep


def _score(result: dict, run_dir: Path, curated: RvvFingerprint, op_key: dict) -> dict:
    """Attach gate_ok + structural_match + divergences to a certify result."""
    gate_ok = bool((result.get("correctness") or {}).get("gate_ok"))
    cycles = next((m.get("cycles") for m in result.get("measurement", [])
                   if m.get("target") == "spike"), None)
    sm, divs = 0.0, []
    objd = run_dir / "generated" / "objdump.txt"
    if objd.is_file():
        gen_fp = RvvFingerprint.from_objdump(objd.read_text(), op_key, run_dir.name)
        cmp = compare_fingerprints(curated, gen_fp)
        sm, divs = cmp["structural_match"], cmp["divergences"]
    else:  # mock / no-objdump path: trust result fields if present
        sm, divs = result.get("structural_match", 0.0), result.get("divergences", [])
    return {"gate_ok": gate_ok, "structural_match": sm, "cycles": cycles, "divergences": divs}


def run_beam(seed_pkg: str | Path, model_dir: str | Path, curated_text: str, op_key: dict, *,
             runs_root: str | Path, out_root: str | Path = "artifacts/targets",
             width: int = 3, depth: int = 2, top_k: int = 2, target: str = "rvv",
             timestamp: str = "run", targets: tuple[str, ...] = ("spike",),
             baseline_run_dir: str | Path | None = None,
             certify_fn: Callable = certify_rvv, proposer: Callable = propose_forks,
             loader: Callable = load_rvv_package, minter: Callable = mint_fork
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

    # generation 0: the seed (e.g. hand_v0)
    seed = loader(seed_pkg)
    seed_node = certify_and_score(Path(seed_pkg), f"{seed.run_id}__beam", None, "seed",
                                  ["baseline"], 0)
    nodes.append(seed_node)
    parents = [(seed, seed_node)] if seed_node["gate_ok"] else []

    counter = 0
    for d in range(1, depth + 1):
        jobs, meta = [], []
        for parent_pkg, parent_node in parents:
            props = proposer(parent_node["divergences"], parent_pkg.knobs)
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
        results = run_sweep(jobs, certify_fn=certify_fn)
        gen_nodes = []
        for (fork_dir, parent_rid, p), res in zip(meta, results):
            sc = _score(res, runs_root / fork_dir.name, curated, op_key)
            node = {"run_id": fork_dir.name, "package_dir": str(fork_dir),
                    "parent_run_id": parent_rid, "lever": p.lever, "evidence": p.evidence,
                    "targets_decision": p.targets, "depth": d, **sc}
            gen_nodes.append(node)
            nodes.append(node)
        survivors = [n for n in rank_results(gen_nodes) if n["gate_ok"]][:top_k]
        parents = [(loader(n["package_dir"]), n) for n in survivors]
        if not parents:
            break

    ranked = rank_results([n for n in nodes if n["gate_ok"]]) or nodes
    best = ranked[0] if ranked else None
    tree = {"target": target, "seed": str(seed_pkg), "op_key": op_key,
            "width": width, "depth": depth, "top_k": top_k,
            "best": {k: best[k] for k in ("run_id", "structural_match", "cycles", "lever")}
                    if best else None,
            "nodes": nodes, "deferred_work_items": deferred}
    tree_path = runs_root / "beam_tree.yaml"
    write_yaml(tree_path, tree, header="RVV beam-search tree (rvvgen.beam.run_beam)")
    return {"best": best, "nodes": nodes, "deferred": deferred, "tree_path": str(tree_path)}
