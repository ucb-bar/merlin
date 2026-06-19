"""Run the RTL-derived FileCheck assertions against an agent's emitted artifacts (Pillar 1 runner).

Ties the deterministic RTL facts + the FileCheck compiler to a candidate capsule run:

  1. render the decoded RoCC trace to a canonical text (counts + ABI + per-instruction lines),
  2. compile the FileCheck assertions for the capsule (:mod:`rtl_check_compiler`),
  3. invoke the **FileCheck LLVM binary** over (i) the gemmini-dialect MLIR and (ii) the rendered trace,
  4. additionally run the Python :func:`rtl_checks.screen` for numeric bounds FileCheck can't express
     (scratchpad/accumulator capacity, multi-matmul tile lower bound),

and return a combined result whose ``verdict`` a caller may use to SKIP the expensive spike/verilator/VCS
oracle on a hard reject — turning a multi-minute failed RTL run into an instant FileCheck diagnostic.

Frozen runner/grader/contract are never touched; this runs *around* them.

CLI::

    python -m merlin.targetgen.rtl_check_runner <run_capsule_dir | runs_root> [--write] [--quantify]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from . import rtl_check_compiler as CC
from . import rtl_checks as RC

_REPO = Path(__file__).resolve().parents[4]
_FACTS = _REPO / "merlin/targets/gemmini/contracts/rtl_facts/facts.json"
_CAPSULE_ROOTS = [_REPO / "bench_contract" / "capsules",
                  _REPO / "experiments" / "gemmini_perf_bench" / "kernels"]
_FILECHECK_CANDIDATES = [
    "/scratch2/agustin/LHWIR/third_party/circt/install/bin/FileCheck",
    str(_REPO / "third_party/llvm-build/bin/FileCheck"),
    "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/FileCheck",
]
_COMPUTE_CLASSES = {"COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"}


def find_filecheck() -> str | None:
    for c in _FILECHECK_CANDIDATES:
        if Path(c).is_file():
            return c
    return shutil.which("FileCheck")


_OPFORM_RE = re.compile(r"=\s*gemmini\.\w+|gemmini\.(res_pack|matmul|commit|evict)\b")


def _is_op_form(mlir: str) -> bool:
    """True when the gemmini dialect is emitted as real ops (`%x = gemmini.<op> ...`), vs the
    attribute-encoded `gemmini.program = [...]` form the op-name FileCheck patterns don't apply to."""
    return bool(_OPFORM_RE.search(mlir)) and "gemmini.program" not in mlir


def _legal_funct(facts_rec: dict) -> set[int]:
    facts = facts_rec.get("facts", facts_rec)
    for i in (facts.get("interfaces") or []):
        if i.get("name") == "funct_decode_table":
            return set(i.get("legal_funct") or [])
    return set(range(0, 26))  # RTL-derived default


def render_trace(trace: dict, facts_rec: dict) -> str:
    """Canonical text the TRACE FileCheck lines are matched against."""
    instrs = trace.get("instructions", [])
    hist: dict[str, int] = {}
    for i in instrs:
        hist[i.get("class")] = hist.get(i.get("class"), 0) + 1
    n_mvin = hist.get("MVIN", 0)
    n_mvout = hist.get("MVOUT", 0)
    n_compute = sum(hist.get(c, 0) for c in _COMPUTE_CLASSES)
    legal = _legal_funct(facts_rec)
    n_illegal = sum(1 for i in instrs
                    if isinstance(i.get("funct"), int) and i["funct"] not in legal)
    abi = trace.get("abi") or {}
    custom = abi.get("custom_opcode", "0x7b")
    funct3 = abi.get("funct3", "0x3")
    L = [f"# {CC.RENDER_SCHEMA}",
         f"ABI custom={custom} funct3={funct3}",
         f"MVIN_COUNT {n_mvin}",
         f"MVOUT_COUNT {n_mvout}",
         f"COMPUTE_COUNT {n_compute}",
         f"ILLEGAL_FUNCT_COUNT {n_illegal}",
         f"COMPUTE_PRESENT {'yes' if n_compute else 'no'}",
         f"MVIN_PRESENT {'yes' if n_mvin else 'no'}"]
    for i in instrs:
        f = i.get("funct")
        L.append(f"INSTR {i.get('index')} {i.get('class')} funct={f if f is not None else '-'}")
    return "\n".join(L) + "\n"


def run_filecheck(fc: str, check_text: str, input_text: str, prefix: str) -> tuple[bool, str]:
    """Run FileCheck(check_text) over input_text with the given --check-prefix. (ok, diagnostics)."""
    with tempfile.NamedTemporaryFile("w", suffix=".checks", delete=False) as cf:
        cf.write(check_text)
        check_path = cf.name
    try:
        p = subprocess.run([fc, f"--check-prefix={prefix}", "--allow-unused-prefixes", check_path],
                           input=input_text, capture_output=True, text=True)
        return (p.returncode == 0, (p.stderr or p.stdout).strip())
    finally:
        Path(check_path).unlink(missing_ok=True)


def _capsule_index() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for root in _CAPSULE_ROOTS:
        if root.is_dir():
            for cy in root.rglob("capsule.yaml"):
                idx.setdefault(cy.parent.name, cy)
    return idx


def _load_capsule(name: str, index: dict[str, Path]) -> dict | None:
    p = index.get(name)
    return yaml.safe_load(p.read_text()) if p else None


def screen_run(run_capsule_dir: Path, facts_rec: dict, index: dict[str, Path],
               fc: str | None, write: bool = False) -> dict | None:
    """Run the full RTL-check suite (FileCheck dialect+trace + Python numeric screen) on one run dir."""
    gen = run_capsule_dir / "generated"
    trace_p = gen / "instruction_trace.json"
    dialect_p = gen / "lowered.target.mlir"
    if not trace_p.is_file():
        return None
    trace = json.loads(trace_p.read_text())
    capsule = _load_capsule(_capsule_name_for(run_capsule_dir), index)
    compiled = CC.compile_checks(facts_rec, capsule or {})
    res: dict[str, Any] = {"capsule": (capsule or {}).get("name") or run_capsule_dir.name,
                           "filecheck": {}, "screen": None}

    if fc:
        if compiled["trace"]:
            ok, diag = run_filecheck(fc, compiled["trace"], render_trace(trace, facts_rec), "TRACE")
            res["filecheck"]["trace"] = {"ok": ok, "diag": diag}
        if compiled["dialect"] and dialect_p.is_file():
            mlir = dialect_p.read_text()
            # The agent legitimately emits >1 MLIR surface form: op-form (`%x = gemmini.<op> ...`) and
            # an attribute-encoded form (`gemmini.program = [...]`). The op-name patterns only apply to
            # op-form; on other forms the check is SKIPPED (honest), not failed — the format-agnostic
            # TRACE check over the decoded RoCC stream carries the structural verdict either way.
            if _is_op_form(mlir):
                ok, diag = run_filecheck(fc, compiled["dialect"], mlir, "DIALECT")
                res["filecheck"]["dialect"] = {"ok": ok, "diag": diag}
            else:
                res["filecheck"]["dialect"] = {"ok": None, "skipped": "non-op-form MLIR"}
    # Python numeric/lower-bound checks (capacity, multi-matmul tile bound) the RTL facts feed.
    rc_facts = CC._facts_to_rc(facts_rec)
    rep = RC.screen(trace, capsule, rc_facts)
    res["screen"] = rep.to_dict()

    # VERDICT rides only the format-agnostic, RTL-grounded checks: the TRACE FileCheck (over the decoded
    # RoCC stream) + the Python numeric screen. The DIALECT FileCheck is ADVISORY-only (still reported as
    # feedback) — corroboration against 383 real agent runs showed the agent emits several legal MLIR
    # surface forms, so op-name patterns over lowered.target.mlir false-fail on passing code. The decoded
    # trace is canonical and format-independent, so it never had a false positive.
    fc_fail = (res["filecheck"].get("trace") or {}).get("ok") is False
    res["verdict"] = "reject" if (fc_fail or rep.verdict == "reject") else (
        "warn" if rep.verdict == "warn" else "ok")
    if write:
        (run_capsule_dir / "rtl_checks.json").write_text(json.dumps(res, indent=2))
    return res


def _capsule_name_for(d: Path) -> str:
    r = d / "capsule_result.json"
    if r.is_file():
        try:
            return json.loads(r.read_text()).get("capsule") or d.name
        except Exception:
            pass
    return d.name


def prescreen(run_capsule_dir: Path) -> dict | None:
    """Opt-in cost gate: compile+run the RTL checks; caller may skip the oracle on verdict=='reject'."""
    facts = json.loads(_FACTS.read_text())
    return screen_run(Path(run_capsule_dir), facts, _capsule_index(), find_filecheck(), write=False)


def iter_run_dirs(root: Path):
    if (root / "generated" / "instruction_trace.json").is_file():
        yield root
        return
    for t in root.rglob("generated/instruction_trace.json"):
        yield t.parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="a capsule run dir or a runs/ tree")
    ap.add_argument("--write", action="store_true", help="write rtl_checks.json beside capsule_result.json")
    ap.add_argument("--quantify", action="store_true",
                    help="summarize how many runs the pre-screen would reject (oracle skips)")
    a = ap.parse_args(argv)
    fc = find_filecheck()
    if not fc:
        print("WARNING: FileCheck binary not found; running Python screen only")
    facts = json.loads(_FACTS.read_text())
    index = _capsule_index()
    rejects = warns = oks = n = 0
    for d in sorted(iter_run_dirs(Path(a.root))):
        r = screen_run(d, facts, index, fc, write=a.write)
        if r is None:
            continue
        n += 1
        v = r["verdict"]
        rejects += v == "reject"; warns += v == "warn"; oks += v == "ok"
        if not a.quantify:
            fcs = " ".join(f"{k}={'ok' if vv['ok'] else 'FAIL'}" for k, vv in r["filecheck"].items())
            print(f"  {v:6s} {r['capsule']:34s} filecheck[{fcs}] screen={r['screen']['verdict']}")
    print(f"\n{n} runs: {rejects} reject, {warns} warn, {oks} ok "
          f"(reject => oracle run can be skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
