"""The agent-facing surface over the frozen gemmini backend: four verbs, compact JSON, no MCP.

    agent_compile.py inspect  <workload.mlir>
    agent_compile.py choices  <workload.mlir>
    agent_compile.py build    <workload.mlir> --recipe '{"activation_residency":"panel",...}'
    agent_compile.py evaluate <candidate_id>

WHAT THIS IS TESTING. The agent produces a RECIPE and nothing else -- never source, never an
instruction. The compiler turns the decision into code and the RTL turns the code into cycles, so the
only thing being measured about the agent is the quality and cost of its decisions. Every verb is
therefore deliberately narrow, and the observation is deliberately small: the point of the ablation is
partly that a compiler-mediated loop needs FEWER tokens, which a verb that dumps MLIR would quietly
destroy. Full artifacts are available, but only when asked for (``build --dump``).

THREE HONESTY RULES BAKED IN, each from a measured failure in this tree:

* **No predicted cycles.** The calibrated gemmini cost model is FALSIFIED against measured cycles
  (A2: 174 predicted vs 302 measured; PK03_k128: 1103 vs 604; w1_small: 1056 vs 780) -- errors in both
  directions, outside its own declared ``max_abs_pct`` of 34.9%. So ``build`` reports the emitted-code
  delta, which it can observe, and refuses to report a cycle estimate it cannot stand behind.
* **Legality comes from the compiler, not from the caller.** ``choices`` intersects what the fork can
  EMIT (its own ``--list-recipe-choices``-equivalent catalog) with what the machine can HOLD (capacity,
  from RTL-derived facts), and every value carries its own verdict so a refusal is never inferred from
  an omission.
* **A cycle number names its engine.** GSIM here simulates ``GemminiGsimSerialClkConfig`` and was
  MEASURED to disagree with stock ``GemminiRocketConfig`` under Verilator (302 vs 303, 604 vs 610), so
  every evaluate result carries the engine and config and may not be quoted as Verilator-equivalent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / "merlin" / "python").is_dir():
            return cand
    raise SystemExit("could not locate repo root")


REPO = _repo_root()
sys.path.insert(0, str(REPO / "merlin" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from merlin.common.artifacts import cache_dir            # noqa: E402
import _track as T                                       # noqa: E402
from merlin.common.paths import artifacts_dir            # noqa: E402

FORK = artifacts_dir() / "targets/gemmini/gemmini_xdsl_recipe_v0"
PKG = FORK / "mlir_oot"
STORE = cache_dir("recipe_select_candidates")

GSIM_EMU = ("/scratch/agustin/tmp/gsim_cert_serialclk_v1/"
            "emu_gemmini_gsim_serialclk_v1_filtered_final")
GSIM_CONFIG = "chipyard.harness.TestHarness.GemminiGsimSerialClkConfig"
ENGINE_NOTE = ("cycles describe " + GSIM_CONFIG + "; measured to disagree with stock "
               "GemminiRocketConfig under Verilator (302 vs 303, 604 vs 610), so they are NOT "
               "quotable as Verilator-equivalent")


def _recipe_mod():
    sys.path.insert(0, str(PKG / "lowering"))
    import recipe                                                     # noqa: PLC0415
    return recipe


def machine(target: str = "gemmini") -> dict:
    """The machine facts a recipe decision needs, DERIVED from the target's own RTL fact bundle."""
    from merlin.perf.workload_gen import tile_geometry
    from merlin.targetgen.rtl import facts as rtl_facts

    geom = tile_geometry(target)
    body = rtl_facts.load_facts(target)
    mems = (body.get("facts") or body).get("memories") or []

    def cap(name: str) -> int | None:
        m = next((x for x in mems if x.get("name") == name), None)
        return int(m["bytes"]) if m and m.get("bytes") else None

    spad_b, acc_b = cap("scratchpad"), cap("accumulator")
    if not spad_b or not acc_b:
        raise SystemExit("operand/accumulator capacity is not derivable from the RTL facts; "
                         "refusing to assume one")
    dim = geom.rows
    return {
        "dim": dim, "mesh": [geom.rows, geom.cols],
        "operand_store_rows": spad_b // dim,
        "accumulator_rows": acc_b // (dim * 4),
        "facts_source": (body.get("facts") or body).get("source"),
    }


def shape_of(mlir: Path) -> dict:
    """M, N, K and the operand dtypes, read from the interface program the compiler itself parses.

    The operands are followed through the COMMAND GRAPH rather than looked up by tensor name: the
    contraction names its lhs and its resident rhs, and the resident is traced back to the RES_PACK
    that produced it. Keying on ``"A0"``/``"W"`` would work only for workloads that happen to spell
    them that way.
    """
    r = subprocess.run([sys.executable, "gemmini_opt.py", "--emit-command-buffer=/dev/stdout",
                        str(mlir)], cwd=str(PKG), capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise SystemExit(f"the compiler could not parse {mlir}: {r.stderr[-800:]}")
    cb = json.loads(r.stdout)
    tensors = cb.get("tensors") or {}
    cmds = cb.get("commands") or []

    resident_src = {c["operands"]["dst"]: c["operands"]["src"]
                    for c in cmds if c.get("opcode") == "RES_PACK"}
    mm = next((c for c in cmds if c.get("opcode") in ("MATMUL_RESIDENT", "ATTENTION_QK")), None)
    if mm is None:
        raise SystemExit(json.dumps({"error": "no_contraction",
                                     "detail": "this workload declares no matmul to tune"}))
    ops = mm["operands"]
    lhs_name = ops.get("lhs") or ops.get("q")
    rhs_ref = ops.get("rhs") or ops.get("k")
    w_name = resident_src.get(rhs_ref, rhs_ref)
    lhs, w = tensors[lhs_name], tensors[w_name]
    m, k = lhs["shape"]
    kw, n = w["shape"]
    if kw != k:
        raise SystemExit(json.dumps({"error": "contraction_mismatch",
                                     "detail": f"lhs K={k} but weight K={kw}"}))
    return {"M": m, "N": n, "K": k, "operand_dtype": lhs.get("dtype"),
            "weight_dtype": w.get("dtype"), "lhs": lhs_name, "weight": w_name,
            "transposed_rhs": mm.get("opcode") == "ATTENTION_QK"}


def emit(mlir: Path, recipe: dict | None) -> str:
    env = dict(os.environ)
    env.pop("MERLIN_CODEGEN_RECIPE", None)
    if recipe is not None:
        env["MERLIN_CODEGEN_RECIPE"] = json.dumps(recipe)
    r = subprocess.run([sys.executable, "gemmini_opt.py", "--convert-iface-to-gemmini",
                        "--emit-target-artifact", str(mlir)],
                       cwd=str(PKG), capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        raise SystemExit(json.dumps({"error": "emit_failed", "detail": r.stderr[-800:]}))
    return r.stdout


def classes(artifact: str) -> tuple[dict[str, int], list[str]]:
    """Instruction-class counts AND the ordered funct sequence.

    Both are needed: a deletion shows in the counts, a reordering only in the order -- and judging a
    reordering by counts reports a real change as an inert lever.
    """
    sys.path.insert(0, str(PKG))
    from lowering.isa import FUNCT                                    # noqa: PLC0415
    names: dict[int, str] = {}
    for nm, f in FUNCT.items():
        names.setdefault(f, nm)
    names[0] = "CONFIG"
    counts: dict[str, int] = {}
    order: list[str] = []
    for line in artifact.splitlines():
        if "llvm.inline_asm" not in line:
            continue
        if '"fence"' in line:
            counts["FENCE"] = counts.get("FENCE", 0) + 1
            order.append("FENCE")
            continue
        body = line.partition(".insn ")[2].split('"', 1)[0]
        parts = [p.strip() for p in body.split(",")]
        if len(parts) < 3 or not parts[2].startswith("0x"):
            continue
        nm = names.get(int(parts[2], 16), f"funct_{parts[2]}")
        counts[nm] = counts.get(nm, 0) + 1
        order.append(nm)
    return counts, order


# --------------------------------------------------------------------------- verbs

def v_inspect(mlir: Path) -> dict:
    sh = shape_of(mlir)
    mc = machine()
    d = mc["dim"]
    mt, nt, kt = -(-sh["M"] // d), -(-sh["N"] // d), -(-sh["K"] // d)
    return {
        "workload": str(mlir), **sh, "tiles": {"Mt": mt, "Nt": nt, "Kt": kt},
        "macs": sh["M"] * sh["N"] * sh["K"], "machine": mc,
        "capacity_relation": (f"the activation grid (Mt*Kt={mt * kt} tiles) and the weight grid "
                              f"(Kt*Nt={kt * nt} tiles) are staged together, so "
                              f"Kt*(Mt+Nt)={kt * (mt + nt)} must not exceed "
                              f"{mc['operand_store_rows'] // d}"),
        "reuse_available": {
            "activation_transfers_now": kt * nt * mt,
            "activation_transfers_if_resident": mt * kt,
            "note": ("the saving a residency recipe can buy is Mt*Kt*(Nt-1) transfers, so it is zero "
                     "when Nt==1 and grows with the N sweep"),
        },
    }


def v_choices(mlir: Path) -> dict:
    R = _recipe_mod()
    sh = shape_of(mlir)
    mc = machine()
    cat = R.catalog(m=sh["M"], n=sh["N"], k=sh["K"], dim=mc["dim"],
                    spad_rows=mc["operand_store_rows"], acc_rows=mc["accumulator_rows"])
    cat["authority"] = ("values come from the compiler's own catalog (what it can emit) intersected "
                        "with the RTL-derived capacity (what the machine can hold); a value with "
                        "legal=false carries the reason and must not be built")
    return cat


def v_build(mlir: Path, recipe_spec: str | None, dump: bool) -> dict:
    R = _recipe_mod()
    recipe = R.Recipe.parse(recipe_spec).as_dict()
    sh = shape_of(mlir)
    mc = machine()
    # LEGALITY IS `blocks`, NOT `fit`. `fit` answers the pre-blocking question -- does the WHOLE shape
    # fit both stores at once -- and for every ResNet-50 and TinyLlama shape the answer is no. Gating
    # on it here refused shapes the compiler can now emit by cutting them, which is a refusal the
    # agent cannot act on and a workload the arm would silently never cover.
    plan = R.blocks(R.Recipe(**recipe), m=sh["M"], n=sh["N"], k=sh["K"], dim=mc["dim"],
                    spad_rows=mc["operand_store_rows"], acc_rows=mc["accumulator_rows"])
    if not plan.ok:
        return {"built": False, "recipe": recipe, "failure": "illegal_for_this_shape",
                "reason": plan.reason}
    f = R.fit(R.Recipe(**recipe), m=sh["M"], n=sh["N"], k=sh["K"], dim=mc["dim"],
              spad_rows=mc["operand_store_rows"], acc_rows=mc["accumulator_rows"])

    art = emit(mlir, None if recipe == R.DEFAULTS else recipe)
    base = emit(mlir, None)
    counts, order = classes(art)
    bcounts, border = classes(base)
    digest = hashlib.sha256(art.encode()).hexdigest()
    cid = hashlib.sha256((str(mlir) + json.dumps(recipe, sort_keys=True)).encode()).hexdigest()[:16]

    slot = STORE / cid
    slot.mkdir(parents=True, exist_ok=True)
    (slot / "artifact.mlir").write_text(art, encoding="utf-8")
    (slot / "candidate.json").write_text(json.dumps(
        {"candidate_id": cid, "workload": str(mlir), "recipe": recipe,
         "artifact_digest": digest}, indent=1), encoding="utf-8")

    delta = {k: counts.get(k, 0) - bcounts.get(k, 0)
             for k in set(counts) | set(bcounts) if counts.get(k, 0) != bcounts.get(k, 0)}
    out = {
        "built": True, "candidate_id": cid, "recipe": recipe,
        "is_default": recipe == R.DEFAULTS,
        "artifact_digest": digest[:16], "n_instructions": sum(counts.values()),
        "instr_counts": counts,
        # The block plan is part of what was BUILT, not a detail of how: it decides how many times
        # each operand is re-fetched, and a candidate row that does not carry it cannot be joined
        # back to the schedule that produced its cycles.
        "blocks": {"block_m": plan.bm, "block_n": plan.bn, "block_k": plan.bk,
                   "n_blocks": plan.n_blocks, "derived": plan.derived},
        "fits_without_cutting": f.ok,
        "why_cutting_is_needed": "" if f.ok else f.reason,
        "vs_default": {
            "count_delta": delta,
            "reordered_only": (not delta) and order != border,
            "identical": digest == hashlib.sha256(base.encode()).hexdigest(),
        },
        "predicted_cycles": None,
        "why_no_prediction": ("the calibrated cost model is falsified against measured cycles "
                              "(-42% on A2, +83% on PK03_k128, -26% on w1_small; its own declared "
                              "max_abs_pct is 34.9%), so no estimate is offered -- run evaluate"),
    }
    if dump:
        out["artifact"] = art
    return out


def v_evaluate(cid: str, engine: str, timeout: int) -> dict:
    slot = STORE / cid
    meta_p = slot / "candidate.json"
    if not meta_p.exists():
        return {"candidate_id": cid, "correct": None, "cycles": None,
                "failure": "unknown_candidate_id: build it first"}
    meta = json.loads(meta_p.read_text())
    cached = slot / f"result_{engine}.json"
    if cached.exists():
        r = json.loads(cached.read_text())
        r["served_from_cache"] = True          # a re-evaluated digest is charged nothing
        return r

    from merlin.targetgen import oot_runner as OOT
    os.environ.setdefault("MERLIN_GEMMINI_GSIM_EMU", GSIM_EMU)
    os.environ.setdefault("MERLIN_GEMMINI_GSIM_MAXCYCLES", "100000000")
    prev = os.environ.get("MERLIN_CODEGEN_RECIPE")
    R = _recipe_mod()
    if meta["recipe"] == R.DEFAULTS:
        os.environ.pop("MERLIN_CODEGEN_RECIPE", None)
    else:
        os.environ["MERLIN_CODEGEN_RECIPE"] = json.dumps(meta["recipe"])
    t0 = time.time()
    try:
        res = OOT.certify(FORK, Path(meta["workload"]), runs_root=T.RUNS,
                          run_id=f"ac_{cid}_{engine}", simulator=engine, target="gemmini",
                          timeout=timeout)
        failure = None
    except Exception as exc:
        res, failure = {}, f"{type(exc).__name__}: {exc}"
    wall = time.time() - t0
    if prev is None:
        os.environ.pop("MERLIN_CODEGEN_RECIPE", None)
    else:
        os.environ["MERLIN_CODEGEN_RECIPE"] = prev

    oracle = (res or {}).get("oracle") or {}
    out = {
        "candidate_id": cid, "recipe": meta["recipe"],
        "correct": (res or {}).get("status") == "pass",
        "cycles": oracle.get("cycles"),
        "engine": engine, "oracle_kind": oracle.get("kind"),
        "derived_from_rtl": oracle.get("derived_from_rtl"),
        "config": GSIM_CONFIG if engine == "gsim" else "GemminiRocketConfig",
        "engine_note": ENGINE_NOTE if engine == "gsim" else None,
        "eval_seconds": round(wall, 2), "concurrency_observed": 1,
        "failure": failure or ((res or {}).get("status") if
                               (res or {}).get("status") != "pass" else None),
        "served_from_cache": False,
    }
    cached.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _workload(arg: str) -> Path:
    """Absolutise the workload path. Every compiler invocation runs with ``cwd=PKG``, so a
    caller-relative path would be resolved against the package root and vanish -- the same misrooting
    ``oot_runner._resolve_argv`` absolutises against."""
    p = Path(arg).expanduser()
    p = p if p.is_absolute() else (Path.cwd() / p)
    p = p.resolve()
    if not p.exists():
        raise SystemExit(json.dumps({"error": "workload_not_found", "path": str(p)}))
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    for v in ("inspect", "choices"):
        s = sub.add_parser(v)
        s.add_argument("workload")
    b = sub.add_parser("build")
    b.add_argument("workload")
    b.add_argument("--recipe", default=None)
    b.add_argument("--dump", action="store_true", help="include the full emitted artifact")
    e = sub.add_parser("evaluate")
    e.add_argument("candidate_id")
    e.add_argument("--engine", default="gsim")
    e.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args(argv)

    if a.verb == "inspect":
        out = v_inspect(_workload(a.workload))
    elif a.verb == "choices":
        out = v_choices(_workload(a.workload))
    elif a.verb == "build":
        out = v_build(_workload(a.workload), a.recipe, a.dump)
    else:
        out = v_evaluate(a.candidate_id, a.engine, a.timeout)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
