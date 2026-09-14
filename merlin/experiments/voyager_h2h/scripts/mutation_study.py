"""Mutation study: which realistic compiler faults does Voyager's own verification catch, and which
does merlin's bit-exact schedule check catch?

Each fault is seeded ONCE, into Voyager's own bufferized program (the FX graph ``compile()`` leaves
executable, or its memory plan), by ``_mutation_worker.py`` running in Voyager's environment. The
same faulty program is then judged by both sides:

Voyager, by its own criteria, at the stages it applies them (all read from the pinned compiler):

* ``pre_tiling``  -- ``test/test_codegen.py``'s numeric check as the CNN / BERT / ViT harnesses run it:
  on the graph after ``transform()``, BEFORE ``compile()`` tiles and bufferizes. A bufferization fault
  cannot reach it; the verdict is the unmutated program's, recorded once per workload.
* ``post_bufferization`` -- the same criterion (``assert_close(rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)``),
  on the faulty bufferized graph run eagerly in Voyager's stack, which is where its LLM harness
  (``test/utils/models/llama.py``) and ``voyager_export.py --run-lowered`` take ``new_output``. Warn-only
  in every harness (``run_ci.py`` lists ``numeric_drift`` as "not gated").
* ``ci_gate`` -- ``test/run_ci.py``'s actual gate: FAIL on a compile error or when ``model.txt`` differs
  from the previous run's. Evaluated as "the clean compiler was the previous run".

merlin, by ``merlin.baselines.voyager_ir.replay`` (counting-semaphore discipline) ->
``voyager_schedule.lower_gemm`` / ``lower_conv`` -> ``execute``, compared bit-exactly against the exact
integer result of the original operator on the same integer operands Voyager's program reads.

Negative controls (semantics-preserving rewrites) must pass every check that is worth having; the
identity control checks the harness itself.

Run from the repo root with merlin's venv::

    .venv/bin/python merlin/experiments/voyager_h2h/scripts/mutation_study.py

Writes a ``compare/gemmini`` product (``results.json`` with provenance, ``table.md``, and each
workload's full evidence under ``work/``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from merlin.baselines.voyager import geometry_for
from merlin.baselines.voyager_ir import SemaphoreViolation, UnsupportedConstruct, load_model, replay
from merlin.baselines.voyager_schedule import execute, lower_conv, lower_gemm
from merlin.common import provenance
from merlin.common.artifacts import new_product
from merlin.common.paths import build_dir, merlin_dir, repo_root

HERE = Path(__file__).resolve().parent
FIXTURES = ("split_k_256x512x256", "conv3x3_s1_28x28x64x64", "conv3x3_s2_28x28x64x128",
            "conv1x1_28x28x64x256")
#: The Voyager sources whose behaviour this study's Voyager verdicts depend on (criteria, stages,
#: eager kernels, emitter, planner), each checked by CONTENT against the pinned commit.
VOYAGER_READS = (
    "test/test_codegen.py", "test/run_ci.py",
    "test/utils/models/torchvision_models.py", "test/utils/models/vit.py",
    "test/utils/models/bert.py", "test/utils/models/mobilebert.py", "test/utils/models/llama.py",
    "src/voyager_compiler/__init__.py",
    "src/voyager_compiler/codegen/transform/bufferize/ops.py",
    "src/voyager_compiler/codegen/transform/bufferize/emit.py",
    "src/voyager_compiler/codegen/transform/bufferize/memory_planning.py",
)


# ------------------------------------------------------------------------------------------------
# The exact reference: the original operator on the program's own integer operands
# ------------------------------------------------------------------------------------------------

def exact_reference(workload: dict, operands: dict, dram: dict) -> np.ndarray:
    """int64 result of the operator the workload names, independent of any schedule, laid out as
    the schedule's 2-D output view ([M, N] for a GEMM, [N*OH*OW, Cout] for an NHWC convolution)."""
    lhs = operands[dram["lhs"]].astype(np.int64)
    weight = operands[dram["weight"]].astype(np.int64)
    bias = operands[dram["bias"]].astype(np.int64) if "bias" in dram else None
    if workload["kind"] == "linear":
        out = lhs.reshape(-1, lhs.shape[-1]) @ weight
        return out + bias if bias is not None else out
    k, stride = workload["k"], workload.get("stride", 1)
    pad = workload.get("padding", k // 2)       # voyager_export.build_workload's default
    n, h, w, cin = lhs.shape
    if weight.shape != (k, k, cin, workload["Cout"]):
        raise ValueError(f"weight {weight.shape} is not HWIO ({k}, {k}, {cin}, {workload['Cout']})")
    padded = np.pad(lhs, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    oh, ow = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
    out = np.zeros((n, oh, ow, weight.shape[-1]), dtype=np.int64)
    for fy in range(k):
        for fx in range(k):
            window = padded[:, fy:fy + stride * oh:stride, fx:fx + stride * ow:stride]
            out += np.einsum("nhwc,co->nhwo", window, weight[fy, fx])
    if bias is not None:
        out += bias
    return out.reshape(-1, weight.shape[-1])


def reference_agreement(workload: dict, exact: np.ndarray, scale: float,
                        voyager_reference: np.ndarray) -> dict:
    """How closely ``relu?(exact * scale)`` reproduces Voyager's own (bf16) quantized reference: a
    check that the operands, their layout and the exact reference describe Voyager's computation."""
    value = exact.astype(np.float64) * scale
    if workload.get("relu"):
        value = np.maximum(value, 0.0)
    if workload["kind"] == "conv":
        n, c, oh, ow = voyager_reference.shape
        value = value.reshape(n, oh, ow, c).transpose(0, 3, 1, 2)
    ref = voyager_reference.astype(np.float64)
    diff = np.abs(value.reshape(ref.shape) - ref)
    return {"max_abs": float(diff.max()),
            "max_rel_to_scale": float(diff.max() / (np.abs(ref).max() or 1.0)),
            "rel_l2": float(np.linalg.norm(diff) / (np.linalg.norm(ref) or 1.0))}


# ------------------------------------------------------------------------------------------------
# merlin's check on one (possibly faulty) program
# ------------------------------------------------------------------------------------------------

def merlin_check(model_json: Path, workload: dict, operands: dict, geometry) -> dict:
    try:
        return _merlin_check(model_json, workload, operands, geometry)
    except Exception as exc:  # noqa: BLE001 -- a crash is recorded as such, never as a pass
        return {"verdict": "crashed", "detail": f"{type(exc).__name__}: {str(exc)[:300]}"}


def _merlin_check(model_json: Path, workload: dict, operands: dict, geometry) -> dict:
    try:
        trace = replay(load_model(model_json))
    except SemaphoreViolation as exc:
        return {"verdict": "flagged_semaphore", "detail": str(exc)[:300]}
    except UnsupportedConstruct as exc:
        return {"verdict": "refused", "stage": "replay", "detail": str(exc)[:300]}
    lower = lower_gemm if workload["kind"] == "linear" else lower_conv
    try:
        schedule = lower(trace, geometry)
    except UnsupportedConstruct as exc:
        return {"verdict": "refused", "stage": "lower", "detail": str(exc)[:300]}
    dram = schedule.dram_nodes
    views = {role: operands[node].reshape(schedule.shapes[role]) for role, node in dram.items()
             if role in ("lhs", "weight", "bias")}
    extra = {"bias": views["bias"]} if "bias" in views else {}
    got = execute(schedule, views["lhs"], views["weight"], **extra).astype(np.int64)
    want = exact_reference(workload, operands, dram)
    if got.shape != want.shape:
        return {"verdict": "flagged_mismatch", "detail": f"shape {got.shape} vs {want.shape}"}
    diff = got - want
    wrong = int(np.count_nonzero(diff))
    norm = float(np.linalg.norm(want.astype(np.float64)))
    return {"verdict": "flagged_mismatch" if wrong else "pass",
            "mismatched_elements": wrong, "elements": int(want.size),
            "max_abs_int": int(np.abs(diff).max()),
            "rel_l2": float(np.linalg.norm(diff.astype(np.float64)) / norm) if norm else 0.0,
            "ops": len(schedule.ops)}


# ------------------------------------------------------------------------------------------------
# Verdict synthesis
# ------------------------------------------------------------------------------------------------

def voyager_verdicts(row: dict, side: dict) -> dict:
    pre = side["pre_tiling_check"]["result"]
    post = row.get("post_bufferization_check", {}).get("result")
    unmutated_post = side["post_bufferization_check_unmutated"]["result"]
    raised = (row.get("eager") or {}).get("raised")
    delta = row.get("error_vs_unmutated_lowered") or {}
    changed_output = bool(raised) or delta.get("changed_elements", 0) > 0
    txt_changed = (row.get("emit") or {}).get("model_txt_differs_from_baseline")
    return {
        # As shipped for CNN/BERT/ViT: the check runs before the fault exists.
        "pre_tiling": pre,
        # The criterion on the faulty bufferized program (LLM harness / --run-lowered placement).
        "post_bufferization": "raised" if raised else post,
        # Generous reading: the warn-only criterion fires AND the faulty program's output differs from
        # the correct program's (a warning the unmutated program also raises carries no information).
        "post_bufferization_informative": bool(raised) or (post == "warning" and changed_output),
        "unmutated_post_bufferization": unmutated_post,
        "output_changed": changed_output,
        # memory_planning._check_invariants on the faulty plan: logs [MEM_OVERLAP], never raises.
        "plan_invariant": ("error" if row.get("plan_invariant_error") else
                           "warning" if row.get("plan_invariant_warnings") else "clean"),
        # run_ci.py: FAIL on compile error / missing model.txt / model.txt text differing from the
        # previous run. Numeric warnings are listed "not gated".
        "ci_gate": ("FAIL(model.txt MISMATCH)" if txt_changed else "PASS(model.txt MATCH)")
        if txt_changed is not None else "n/a",
    }


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def build_table(results: list[dict]) -> list[str]:
    lines = ["| workload | mutation | class | Voyager pre-tiling (CI as shipped) | Voyager "
             "post-bufferization (warn-only) | out-of-tol elems (unmutated) | max abs / rel L2 vs "
             "correct program | Voyager plan invariant (warn-only) | run_ci gate | merlin exact "
             "check | merlin int error (elems, max) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for wl in results:
        base_oot = wl["unmutated_post_outside_tolerance"]
        for row in wl["rows"]:
            if row["status"] != "applied":
                continue
            v, m = row["voyager"], row["merlin"]
            err = row.get("error_vs_unmutated_lowered") or {}
            ref = row.get("error_vs_voyager_reference") or {}
            post = v["post_bufferization"] + ("" if v["output_changed"] else " (output identical)")
            lines.append(
                f"| {wl['name']} | {row['id']} | {row['class']} | {v['pre_tiling']} | {post} | "
                f"{_fmt(ref.get('outside_tolerance'))} ({_fmt(base_oot)}) | "
                f"{_fmt(err.get('max_abs'))} / {_fmt(err.get('rel_l2'))} | {v['plan_invariant']} | "
                f"{v['ci_gate']} | "
                f"{m['verdict']} | {_fmt(m.get('mismatched_elements'))}, {_fmt(m.get('max_abs_int'))} |")
    return lines


# ------------------------------------------------------------------------------------------------

def _run_worker(voyager_python: Path, fixture: Path, out: Path, env: dict, only) -> int:
    cmd = [str(voyager_python), str(HERE / "_mutation_worker.py"), "--fixture", str(fixture),
           "--out", str(out)]
    if only:
        cmd += ["--only", *only]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "worker.log", "w") as log:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                              cwd=str(repo_root())).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixtures", nargs="*", default=list(FIXTURES))
    parser.add_argument("--only", nargs="*", help="mutation ids (default: the whole catalogue)")
    parser.add_argument("--target", default="gemmini",
                        help="target whose derived geometry the merlin check lays schedules out on")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8, help="total CPU threads across workers")
    parser.add_argument("--reuse", type=Path,
                        help="an existing work/ dir: skip the Voyager side and re-judge its output")
    args = parser.parse_args(argv)

    os.environ.setdefault("MERLIN_EXT_VOYAGER_COMPILER",
                          str(build_dir() / "external" / "voyager-compiler"))
    voyager_python = build_dir() / "voyager-venv" / "bin" / "python"
    fixtures_root = merlin_dir() / "tests" / "data" / "voyager_ir"
    pin = provenance.verify("voyager_compiler", reads=VOYAGER_READS)
    read_status = [provenance.source_status("voyager_compiler", rel) for rel in VOYAGER_READS]
    reads_pinned = all(str(s.status) == str(provenance.PINNED) for s in read_status)
    geometry = geometry_for(args.target)

    product = new_product("compare", version=1, target=args.target, update_latest=False,
                          notes="voyager_h2h mutation study: Voyager's own checks vs merlin's "
                                "bit-exact schedule check on faults seeded into Voyager's "
                                "bufferized programs")
    work = args.reuse or (product.path / "work")
    if not args.reuse:
        per_job = max(1, args.threads // max(1, min(args.jobs, len(args.fixtures))))
        env = dict(os.environ, OMP_NUM_THREADS=str(per_job), MKL_NUM_THREADS=str(per_job))
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            codes = dict(zip(args.fixtures, pool.map(
                lambda name: _run_worker(voyager_python, fixtures_root / name, work / name, env,
                                         args.only), args.fixtures)))
        failed = {k: v for k, v in codes.items() if v}
        if failed:
            print(f"Voyager-side worker failed: {failed} (see work/<fixture>/worker.log)")
            return 1

    results = []
    for name in args.fixtures:
        side = json.loads((work / name / "voyager_side.json").read_text())
        workload = side["workload"]
        operands = dict(np.load(work / name / "operands.npz"))
        baseline = merlin_check(work / name / "baseline" / "model.json", workload, operands,
                                geometry)
        scale = next(iter(side["dequantize_scales"].values()), None)
        agreement = None
        try:
            trace = replay(load_model(work / name / "baseline" / "model.json"))
            lower = lower_gemm if workload["kind"] == "linear" else lower_conv
            dram = lower(trace, geometry).dram_nodes
            if scale is not None:
                agreement = reference_agreement(
                    workload, exact_reference(workload, operands, dram), scale,
                    np.load(work / name / "voyager_reference.npy"))
        except UnsupportedConstruct as exc:
            agreement = {"error": str(exc)}
        unmutated_oot = next((r.get("error_vs_voyager_reference", {}).get("outside_tolerance")
                              for r in side["mutations"] if r["id"] == "C0_identity"), None)
        rows = []
        for row in side["mutations"]:
            if row["status"] == "applied" and (row.get("emit") or {}).get("ok"):
                row["merlin"] = merlin_check(work / name / row["id"] / "model.json", workload,
                                             operands, geometry)
                row["voyager"] = voyager_verdicts(row, side)
            elif row["status"] == "applied":
                row["merlin"] = {"verdict": "no_ir", "detail": (row.get("emit") or {}).get("error")}
                row["voyager"] = voyager_verdicts(row, side)
            rows.append(row)
        results.append({"name": name, "workload": workload,
                        "fixture_ir_identical": side["fixture_ir_identical"],
                        "pre_tiling_check": side["pre_tiling_check"],
                        "post_bufferization_check_unmutated":
                            side["post_bufferization_check_unmutated"],
                        "unmutated_post_outside_tolerance": unmutated_oot,
                        "merlin_check_unmutated": baseline,
                        "exact_reference_vs_voyager_reference": agreement,
                        "operand_containers": side["operand_containers"],
                        "dequantize_scales": side["dequantize_scales"],
                        "top1_agreement": None,
                        "top1_note": "single-layer workload: no class output, so top-1 is not "
                                     "defined; error is reported elementwise",
                        "rows": rows})

    # Scorecard over applied mutations.
    score: dict[str, dict] = {}
    for wl in results:
        for row in wl["rows"]:
            if row["status"] != "applied" or "merlin" not in row:
                continue
            s = score.setdefault(row["class"], {"n": 0, "voyager_pre_tiling_flags": 0,
                                                "voyager_post_informative": 0,
                                                "voyager_post_warns": 0, "ci_gate_fails": 0,
                                                "voyager_plan_invariant_warns": 0,
                                                "merlin_flags": 0, "merlin_refuses": 0})
            s["n"] += 1
            v, m = row["voyager"], row["merlin"]
            s["voyager_pre_tiling_flags"] += v["pre_tiling"] != "match"
            s["voyager_post_warns"] += v["post_bufferization"] in ("warning", "raised")
            s["voyager_post_informative"] += bool(v["post_bufferization_informative"])
            s["ci_gate_fails"] += v["ci_gate"].startswith("FAIL")
            s["voyager_plan_invariant_warns"] += v["plan_invariant"] == "warning"
            s["merlin_flags"] += m["verdict"].startswith("flagged")
            s["merlin_refuses"] += m["verdict"] == "refused"

    voyager_root = Path(os.environ["MERLIN_EXT_VOYAGER_COMPILER"])
    record = provenance.record(
        pins={"voyager_compiler": pin},
        sources=[HERE / "mutation_study.py", HERE / "_mutation_worker.py",
                 HERE / "voyager_export.py",
                 merlin_dir() / "python" / "merlin" / "baselines" / "voyager_ir.py",
                 merlin_dir() / "python" / "merlin" / "baselines" / "voyager_schedule.py",
                 voyager_root / "test" / "test_codegen.py", voyager_root / "test" / "run_ci.py",
                 *[fixtures_root / n / "model.json" for n in args.fixtures]],
        extra={"geometry": {"target": args.target, **geometry.__dict__},
               "voyager_python": str(voyager_python),
               "voyager_read_set": [{"rel": s.rel, "status": str(s.status), "reason": s.reason}
                                    for s in read_status],
               "voyager_read_set_all_pinned_by_content": reads_pinned})
    doc = {"study": "voyager_h2h mutation study", "target": args.target,
           "voyager_criteria": {
               "numeric": "test/test_codegen.py: assert_close(new, old, rtol=OUTPUT_RTOL, "
                          "atol=OUTPUT_ATOL), warn-only (medusa_* excepted)",
               "stages": {"cnn_bert_vit": "after transform(), before compile()",
                          "llm": "after compile() (bufferized graph)"},
               "ci_gate": "test/run_ci.py: FAIL iff compile error or model.txt text mismatch vs "
                          "previous run; numeric_drift/unverified are listed 'not gated'"},
           "scorecard": score, "workloads": results, "provenance": record}
    (product.path / "results.json").write_text(json.dumps(doc, indent=1, default=str))
    table = ["# Voyager mutation study", "",
             "Each fault is seeded into Voyager's own bufferized program and judged by Voyager's "
             "criteria (read from the pinned compiler) and by merlin's exact schedule check. "
             "`post-bufferization` applies Voyager's `assert_close(rtol=5e-2, atol=1e-4)` to the "
             "faulty program run in Voyager's stack; `(output identical)` means the eager program's "
             "output did not change at all. Parentheses in the tolerance column: the same count "
             "for the correct program.", ""]
    table += build_table(results)
    table += ["", "## Scorecard (applied mutations)", "", "```",
              json.dumps(score, indent=1), "```", "",
              f"voyager_compiler pin ok={pin.ok}; the {len(VOYAGER_READS)} Voyager sources the "
              f"verdicts depend on are byte-identical to the pinned commit: {reads_pinned}; "
              f"merlin commit {record['merlin']['commit'][:12]} "
              f"(dirty files at run time: {record['merlin']['dirty_files']})"]
    (product.path / "table.md").write_text("\n".join(table) + "\n")
    product.add_artifact("results.json")
    product.add_artifact("table.md")
    product.add_artifact("work")
    product.write_manifest()
    print(json.dumps({"product": str(product.path), "scorecard": score}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
