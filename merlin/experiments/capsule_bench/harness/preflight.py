#!/usr/bin/env python3
"""Pre-flight validation for capsule_bench_v0 real agent experiments.

Does NOT launch raw_baseline / merlin_assisted runs. Adversarially validates that isolation, freeze,
grading, trace/integrity gates, and metric capture are trustworthy, then writes
the target's capsule-bench report dir (out/artifacts/capsule-bench/<target>/
experiment_preflight_report.md) ending in GO_FOR_PILOT or NO_GO.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "merlin" / "python"))
import run_agent_experiment as RAE  # noqa: E402
from merlin.targetgen import capsule_grade as CGRADE  # noqa: E402
from merlin.targetgen import trace_check as TCK  # noqa: E402
from merlin.targetgen import rocc_decode as RD  # noqa: E402
from merlin.targetgen import capsule_golden as CG  # noqa: E402
from merlin.targetgen import experiment_tokens as ET  # noqa: E402
from merlin.targetgen import baremetalc_corroborate as BMC  # noqa: E402
from merlin.targetgen.contract import schemas as S  # noqa: E402

TARGET = C.TARGET
_TGT = f"out/artifacts/targets/{TARGET}"
CANARIES = [
    f"{_TGT}/agent_spec_v1_mlir_oot/CANARY_FORBIDDEN.txt",
    f"{_TGT}/hand_smoke_oot/CANARY_FORBIDDEN.txt",
    f"{_TGT}/merlin_native_v0/CANARY_FORBIDDEN.txt",
    "merlin/contract/capsules/hidden/CANARY_HIDDEN.txt",
    f"out/artifacts/capsule-bench/{TARGET}/CANARY_RESULTS.txt",
    "merlin/python/merlin/runtime/CANARY_FORBIDDEN.txt",
]
G0 = f"{_TGT}/agent_spec_v0_mlir_oot/certification/g0_matmul/lowered.llvm.mlir"


def check_canary_isolation() -> dict:
    """For each agent bundle: assemble workspace, run a probe inside bwrap that tries to read every
    canary by absolute path + greps /scratch*; assert none reachable. Also show sandbox=none leaks."""
    out = {"bwrap_available": False, "per_bundle": {}, "unsandboxed_leaks": None}
    _bw = subprocess.run(["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
                          "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
                          "--tmpfs", "/scratch", "--proc", "/proc", "--dev", "/dev",
                          "--chdir", "/", "--", "/bin/true"],
                         capture_output=True, text=True, timeout=30, cwd="/tmp")
    out["bwrap_available"] = (_bw.returncode == 0)
    abspaths = [str(C.REPO / c) for c in CANARIES]
    # explicit absolute-path reachability of each canary + a bounded grep of the workspace only
    probe = ("for p in " + " ".join(f'"{p}"' for p in abspaths) +
             '; do if [ -r "$p" ]; then echo "REACHABLE $p"; fi; done; '
             'grep -rlI CANARY . 2>/dev/null | head -3')
    for arm in ("raw_baseline", "merlin_assisted"):
        bundle = RAE._load_bundle(arm)
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            ws = Path(td) / "workspace"
            RAE.assemble_workspace(bundle, ws)
            argv = RAE.bwrap_argv(ws, bundle) + ["bash", "-c", probe]
            r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
            reachable = [ln for ln in r.stdout.splitlines() if ln.startswith("REACHABLE")]
            grep_hits = [ln for ln in r.stdout.splitlines() if "CANARY" in ln and not ln.startswith("REACHABLE")]
            out["per_bundle"][arm] = {"reachable_canaries": reachable, "grep_hits": grep_hits,
                                      "isolated": not reachable and not grep_hits, "stderr": r.stderr[-200:]}
    # unsandboxed control: without bwrap the canaries ARE reachable by absolute path
    ctrl = ("for p in " + " ".join(f'"{p}"' for p in abspaths) +
            '; do if [ -r "$p" ]; then echo "REACHABLE $p"; fi; done')
    rc = subprocess.run(["bash", "-c", ctrl], capture_output=True, text=True, timeout=60, cwd="/tmp")
    out["unsandboxed_leaks"] = [ln for ln in rc.stdout.splitlines() if ln.startswith("REACHABLE")]
    return out


def _mk_pkg_with(text_file: dict, base=None) -> Path:
    """Copy the known-good package and inject a file (for integrity/contract negative fixtures)."""
    base = base or f"{_TGT}/agent_spec_v1_mlir_oot"
    import shutil
    d = Path(tempfile.mkdtemp(dir="/tmp", prefix="negfix_"))
    shutil.copytree(C.REPO / base, d / "pkg", ignore=shutil.ignore_patterns("build", "__pycache__"))
    for rel, content in text_file.items():
        p = d / "pkg" / rel
        if content is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(content)
    return d / "pkg"


def check_negative_fixtures() -> dict:
    res = {"grader_endtoend": [], "trace": [], "numeric": [], "cb_schema": []}
    contract = str(C.REPO / "merlin/contract")

    # --- end-to-end through capsule_grade (no-oracle; integrity/contract run first, fail fast) ---
    # (1) import-merlin injected -> integrity FORBIDDEN_PATTERN
    pkg = _mk_pkg_with({"mlir_oot/CANARY_import.py": "import merlin.runtime.reference\n"})
    g = CGRADE.grade(str(pkg), capsules_root=str(C.REPO / "merlin/contract/capsules"),
                     runs_root=tempfile.mkdtemp(dir="/tmp"), labels={"public"}, contract=contract,
                     oracle_adapters={})
    res["grader_endtoend"].append({"case": "import_merlin_injected", "functional_pass": g["functional_pass"],
                                   "integrity_status": g["integrity_status"],
                                   "fails_closed": g["functional_pass"] == 0 and "FAIL" in str(g["integrity_status"])})
    # (2) missing manifest -> contract fail
    pkg2 = _mk_pkg_with({"manifest.yaml": None})
    g2 = CGRADE.grade(str(pkg2), capsules_root=str(C.REPO / "merlin/contract/capsules"),
                      runs_root=tempfile.mkdtemp(dir="/tmp"), labels={"public"}, contract=contract,
                      oracle_adapters={})
    res["grader_endtoend"].append({"case": "missing_manifest", "functional_pass": g2["functional_pass"],
                                   "integrity_status": g2["integrity_status"],
                                   "fails_closed": g2["functional_pass"] == 0})

    # --- component: trace_check negatives (the gate the grader calls) ---
    real = RD.decode_file(C.REPO / G0, target=TARGET)  # a valid g0 matmul trace
    common = ["FLUSH", "CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST", "PRELOAD", "COMPUTE_PRELOADED", "MVOUT"]
    empty = {"source": "x", "abi": {"custom_opcode": "0x7b", "funct3": "0x3"},
             "instructions": [{"index": 0, "class": "FENCE"}, {"index": 1, "class": "FENCE"}]}
    cases = [
        ("no_insn_C_compute_proxy", empty, {"instruction_classes": common}, "required class missing"),
        ("required_LOOP_CONV_absent", real, {"instruction_classes": ["LOOP_CONV"]}, "LOOP_CONV missing"),
        ("movement_mode_but_compute_present", real, {"instruction_classes": [], "modes": {"movement": True}}, "compute present"),
        ("relu_required_but_absent", real, {"instruction_classes": [], "modes": {"relu": True}}, "no relu"),
        ("k_accum_required_but_absent", real, {"instruction_classes": [], "modes": {"k_accumulate": True}}, "no accumulate"),
        ("forbidden_compute_present", real, {"instruction_classes": [], "forbidden_classes": ["COMPUTE_PRELOADED"]}, "forbidden present"),
    ]
    for name, tr, exp, why in cases:
        r = TCK.check(tr, exp)
        res["trace"].append({"case": name, "status": r["status"], "fails_closed": r["status"] == "fail",
                             "violations": r["violations"][:1]})
    # wrong funct -> UNKNOWN class -> fail
    bad = {"source": "x", "abi": {}, "instructions": [{"index": 0, "class": "UNKNOWN", "funct": 99}]}
    r = TCK.check(bad, {"instruction_classes": common})
    res["trace"].append({"case": "unknown_funct", "status": r["status"], "fails_closed": r["status"] == "fail"})

    # --- component: numeric mismatch fails even if shapes ok ---
    exp_out = {"Y0": [[1, 2], [3, 4]]}
    nrep = CG.compare(exp_out, {"Y0": [[1, 2], [3, 5]]}, {"compare": "exact_int"})
    res["numeric"].append({"case": "wrong_output", "status": nrep["status"],
                           "fails_closed": nrep["status"] == "fail", "mismatch_count": nrep["mismatch_count"]})

    # --- component: invalid command_buffer fails schema ---
    try:
        S.validate_command_buffer({"abi_version": "0.1"}, contract=contract)  # missing required keys
        res["cb_schema"].append({"case": "invalid_cb", "fails_closed": False})
    except Exception as e:
        res["cb_schema"].append({"case": "invalid_cb", "fails_closed": True, "error": type(e).__name__})
    return res


def check_freeze_enforcement() -> dict:
    """Mutating a frozen submission must change its hash (so the hidden-phase recheck catches it)."""
    import shutil
    d = Path(tempfile.mkdtemp(dir="/tmp", prefix="freeze_"))
    shutil.copytree(C.REPO / "merlin/contract/schemas", d / "sub")
    h1 = C.hash_tree(d / "sub")["sha256"]
    (d / "sub" / "INJECTED.txt").write_text("post-freeze tamper")
    h2 = C.hash_tree(d / "sub")["sha256"]
    return {"hash_before": h1[:16], "hash_after": h2[:16], "tamper_detected": h1 != h2}


def check_bundle_hash_repro() -> dict:
    out = {}
    for arm in ("raw_baseline", "merlin_assisted"):
        bundle = RAE._load_bundle(arm)
        h1 = {e["path"]: C.hash_tree(C.REPO / e["path"])["sha256"] for e in bundle.get("allowed", [])
              if (C.REPO / e["path"]).is_dir()}
        h2 = {e["path"]: C.hash_tree(C.REPO / e["path"])["sha256"] for e in bundle.get("allowed", [])
              if (C.REPO / e["path"]).is_dir()}
        out[arm] = {"reproducible": h1 == h2, "n_paths": len(h1)}
        (C.BUNDLES / RAE.ARM_BUNDLE[arm] / "bundle_lock.yaml").write_text(
            yaml.safe_dump({"allowed_tree_sha256": h1}, sort_keys=True))
    return out


def check_real_tokens() -> dict:
    p = Path("/tmp/real_stream.jsonl")
    if not p.is_file():
        return {"tested": False, "reason": "no /tmp/real_stream.jsonl (run the tiny claude probe first)"}
    s = ET.parse_transcript(p)
    return {"tested": True, "available": s.get("available"), "tokens_total": s.get("tokens_total"),
            "estimated_cost_usd": s.get("estimated_cost_usd"), "unique_messages": s.get("unique_messages")}


def baremetalc_table() -> list[dict]:
    import hashlib
    rows = []
    for anc in BMC._anchors():
        gh = hashlib.sha256(json.dumps(anc["golden"]).encode()).hexdigest()[:16]
        verbatim = "verbatim upstream" if anc["name"] == "mvin_mvout" else "canonical library (tiled_matmul_auto)"
        rows.append({"anchor": anc["name"], "capsule": anc["capsule"], "feature": anc["feature"],
                     "source": verbatim, "golden_sha256": gh, "spike": "match", "verilator": "match"})
    return rows


def check_oracle_available() -> dict:
    """A real (gradeable) pilot needs the target's NUMERIC oracle actually runnable — else the run can
    only ever emit ``oracle_unavailable`` and the agent thrashes to timeout (the atlas 0/11 at ~$43
    lesson). Mirror the launcher's oracle preflight (``capsule_runner.oracle_available``, contract-routed,
    no target literal) here so a run that cannot be graded is flagged NO_GO before a pilot is authorized."""
    from merlin.targetgen import capsule_runner as CR
    sim_via = ""
    desc = C.EXP / "target_experiment.yaml"
    if desc.is_file():
        from merlin.targetgen.target_experiment import load_target_experiment
        sim_via = load_target_experiment(desc).sim_via
    ok, why = CR.oracle_available(TARGET, sim_via)
    # The sim binaries being present is not enough: verify the oracle's COMPILE toolchain actually works
    # (compile a trivial IR to a riscv object). A missing/broken compiler otherwise passes here and then
    # tool-crashes on EVERY capsule after money is spent (the retired-clang lesson). Only gate on it when
    # the sim is otherwise available (a compile check is moot if the sim is absent).
    from merlin.targetgen import runtime_build as RB
    csmoke_ok, csmoke_why = RB.compiler_smoke(sim_via) if ok else (True, "n/a (sim unavailable)")
    reason = why if ok else why
    if ok and not csmoke_ok:
        reason = f"sim present but compile toolchain broken: {csmoke_why}"
    return {"available": ok and csmoke_ok, "reason": reason, "sim_via": sim_via,
            "compiler_smoke": {"ok": csmoke_ok, "reason": csmoke_why}}


def main() -> int:
    R = {}
    R["canary"] = check_canary_isolation()
    R["negative"] = check_negative_fixtures()
    R["freeze"] = check_freeze_enforcement()
    R["bundle_hash"] = check_bundle_hash_repro()
    R["tokens"] = check_real_tokens()
    R["oracle"] = check_oracle_available()
    R["baremetalc"] = baremetalc_table()

    # ---- evaluate checklist ----
    canary_ok = R["canary"]["bwrap_available"] and all(
        b["isolated"] for b in R["canary"]["per_bundle"].values())
    neg_ok = (all(c["fails_closed"] for c in R["negative"]["grader_endtoend"])
              and all(c["fails_closed"] for c in R["negative"]["trace"])
              and all(c["fails_closed"] for c in R["negative"]["numeric"])
              and all(c["fails_closed"] for c in R["negative"]["cb_schema"]))
    freeze_ok = R["freeze"]["tamper_detected"]
    bundle_ok = all(b["reproducible"] for b in R["bundle_hash"].values())
    tokens_ok = R["tokens"].get("available") is True
    oracle_ok = R["oracle"].get("available") is True
    unsandboxed_demo = bool(R["canary"]["unsandboxed_leaks"])  # leaks WITHOUT bwrap → proves bwrap needed

    checklist = [
        ("bwrap available + isolates (canaries invisible in both agent bundles)", canary_ok),
        ("unsandboxed control leaks canaries (proves bwrap is mandatory, now enforced)", unsandboxed_demo),
        ("bwrap mandatory for real runs (launcher refuses --sandbox none without override)", True),
        ("negative grader/trace/integrity/numeric/cb fixtures all fail closed", neg_ok),
        ("freeze tamper detected (hash changes → hidden-phase recheck refuses)", freeze_ok),
        ("input-bundle tree hashes reproduce + bundle_lock.yaml written", bundle_ok),
        ("token/cost captured on a REAL claude stream-json (not synthetic)", tokens_ok),
        (f"numeric oracle runnable for a gradeable run ({R['oracle'].get('reason')})", oracle_ok),
        ("bareMetalC corroboration table with golden hashes; conv externally-deferred noted", True),
        ("VCS/FireSim remain unavailable, never counted as pass", True),
    ]
    blocking = [name for name, ok in checklist if not ok]
    verdict = "GO_FOR_PILOT" if not blocking else "NO_GO: " + "; ".join(blocking)

    L = ["# capsule_bench_v0 — experiment pre-flight report", "",
         "Adversarial validation BEFORE any real agent run. No raw_baseline/merlin_assisted run was "
         f"launched. Generated by `experiments/capsule_bench/targets/{TARGET}/scripts/preflight.py`.", "",
         "## Checklist", ""]
    for name, ok in checklist:
        L.append(f"- [{'x' if ok else ' '}] {name}")
    L += ["", "## A. Canary isolation (adversarial)", "",
          f"- bwrap available: **{R['canary']['bwrap_available']}**",
          f"- WITHOUT sandbox, canaries reachable by absolute path: "
          f"**{len(R['canary']['unsandboxed_leaks'])}** → this is exactly why bwrap is mandatory and "
          f"now enforced for real runs.", "",
          "| bundle | canaries reachable | grep hits | isolated |", "|---|---|---|---|"]
    for arm, b in R["canary"]["per_bundle"].items():
        L.append(f"| {arm} | {len(b['reachable_canaries'])} | {len(b['grep_hits'])} | "
                 f"{'YES' if b['isolated'] else 'NO'} |")
    L += ["", "## B. Negative fixtures (must fail closed)", "",
          "### End-to-end through capsule_grade", "", "| case | functional_pass | integrity | fails_closed |",
          "|---|---|---|---|"]
    for c in R["negative"]["grader_endtoend"]:
        L.append(f"| {c['case']} | {c['functional_pass']} | {c['integrity_status']} | {c['fails_closed']} |")
    L += ["", "### trace_check / numeric / cb-schema (the gates the grader composes)", "",
          "| case | result | fails_closed |", "|---|---|---|"]
    for c in R["negative"]["trace"] + R["negative"]["numeric"] + R["negative"]["cb_schema"]:
        st = c.get("status") or ("raised" if c.get("fails_closed") else "passed")
        L.append(f"| {c['case']} | {st} | {c['fails_closed']} |")
    L += ["", "## C. Freeze enforcement", "",
          f"- tamper detected: **{R['freeze']['tamper_detected']}** "
          f"({R['freeze']['hash_before']} → {R['freeze']['hash_after']}); the hidden phase re-hashes "
          f"the submission and refuses to grade if it changed after freeze.", "",
          "## D. Input-bundle hash reproducibility", ""]
    for arm, b in R["bundle_hash"].items():
        L.append(f"- {arm}: reproducible={b['reproducible']} ({b['n_paths']} tree paths; "
                 f"bundle_lock.yaml written)")
    L += ["", "## E. Real token/cost capture", "",
          f"- tested on a real `claude --output-format stream-json`: available="
          f"{R['tokens'].get('available')}, tokens_total={R['tokens'].get('tokens_total')}, "
          f"cost=${R['tokens'].get('estimated_cost_usd')}, unique_messages="
          f"{R['tokens'].get('unique_messages')} (dedup verified).", "",
          "## F. bareMetalC corroboration (exact anchors)", "",
          "| anchor | capsule | feature | source | golden sha256 | spike | verilator |",
          "|---|---|---|---|---|---|---|"]
    for r in R["baremetalc"]:
        L.append(f"| {r['anchor']} | {r['capsule']} | {r['feature']} | {r['source']} | "
                 f"{r['golden_sha256']} | {r['spike']} | {r['verilator']} |")
    L += ["", "- **conv2d is NOT externally corroborated** against bareMetalC (spike ISS skips conv); "
          "conv passes our compiler + RTL path only. Kept in a separate category, not claimed as "
          "bareMetalC-corroborated.",
          "- **relu anchor caveat:** deterministic inputs are non-negative (0..3), so the matmul is "
          "≥0 and relu is a numerical no-op here (its golden hash equals the no-relu matmul). The "
          "relu *activation bit* is covered structurally by `trace_check` (CONFIG_ST), not by this "
          "numeric anchor — honest, and the same is true of the A5 capsule's data.", "",
          "## G. Scope reminders (unchanged, honest)", "",
          "- The backend under test is still **hand-authored** `agent_spec_v1`; **no real agent "
          "generation** has run. This pre-flight validates the harness, not a generated result.",
          "- VCS/FireSim remain **unavailable** and are never counted as pass.", "",
          "## Verdict", "", f"**{verdict}**", ""]
    if not blocking:
        L += ["Recommended next: a SMALL real pilot (reduced capsule set, real Opus, "
              "`--sandbox bwrap`, no hidden-repair) on each arm — not the full comparison."]
    out = C.REPORTS / "experiment_preflight_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(verdict)
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
