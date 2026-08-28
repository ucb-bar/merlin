#!/usr/bin/env python3
"""THE evaluation path. Every arm calls this -- the direct agent loop and AutoComp alike.

Two arms scored by different oracles cannot be compared, and the difference would masquerade as a
result about the methods. So the oracle, the fidelity ladder, the metrics and the redaction all live
here, and an arm supplies only its kernel.

FIDELITY LADDER. Certifying a kernel on cycle-accurate RTL costs ~115 s; the fast functional model
costs ~4 s. A search that certified every candidate would spend its budget on cycles rather than on
search, so:

    fast  -- the functional model. Correctness + latency + utilization, in seconds. Every candidate
             the search touches is evaluated here.
    cert  -- the cycle-accurate engine. Reserved for a kernel the search already accepted, because
             this is the number that may be quoted.

The two are not interchangeable and the result says which one produced it. A `fast` cycle count is a
model's estimate; only `cert` is cycle-accurate. Reporting one as the other is the exact error that
makes a perf claim unfalsifiable.

WHAT COMES BACK. Latency AND utilization, because latency alone cannot say why a kernel is slow. A
kernel at 3% FP utilization and 50% warp occupancy is issuing the wrong work; one at 90% FP with 10%
occupancy is starved. Those need opposite fixes, and a cycle count distinguishes neither.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

#: Tiers whose PASS actually compared output against a reference. A tier that only proves the kernel
#: ran to completion is an execution cert and may never set "correct".
CERTIFYING_TIERS = frozenset({"L2"})

#: Environment each fidelity needs. Declared as data so a run records WHICH ladder it used, and so a
#: missing dependency is visible rather than silently degrading to the other tier.
FIDELITY_ENV = {
    # Functional model only. Skipping the RTL tier is what makes a search affordable.
    "fast": {"MERLIN_MUON_SKIP_RTL_L3": "1"},
    # Cycle-accurate. GSIM is ~23x faster than the Verilator path in practice (115 s vs ~45 min),
    # which is what makes a per-candidate cert tier possible at all rather than a once-per-run luxury.
    "cert": {
        "MERLIN_MUON_SKIP_RTL_L3": "0",
        "MERLIN_MUON_GSIM_EMU":
            "/scratch/agustin/projects/gsim/build/radiance_gsim/emu_radiance_gsimconfig",
        "MERLIN_MUON_GSIM_MAXCYCLES": "2000000",
        # ⚠️ Without this the derived ISA encoding is absent and EVERY fork-free compile fails closed.
        # A prior sweep scored 6/38 for exactly this reason, and the six "passes" were fixtures that
        # need no oracle -- i.e. it looks like a bad submission, not a missing variable.
        "MERLIN_MLC_DIR": "/scratch2/agustin/mvp-lhwir/modeling",
    },
}


def repo_root() -> Path:
    for anc in HERE.parents:
        if (anc / "merlin" / "python" / "merlin").is_dir():
            return anc
    raise SystemExit("could not locate the repo root")


def missing_requirements(fidelity: str) -> list[str]:
    """Paths a fidelity needs that are not present. Empty means it can actually run.

    Checked up front so a cert request FAILS LOUDLY instead of quietly producing a functional-model
    number that would then be quoted as cycle-accurate.
    """
    out: list[str] = []
    for key, val in FIDELITY_ENV.get(fidelity, {}).items():
        if val.startswith("/") and not Path(val).exists():
            out.append(f"{key}={val}")
    return out


def evaluate(
    capsule_dir: Path,
    kernel_file: Path,
    *,
    shim_pkg: Path,
    runs_root: Path,
    target: str = "radiance",
    fidelity: str = "fast",
    method: str = "",
    config_id: str = "C0",
    timeout: int = 900,
) -> dict:
    """Grade one kernel. Returns {"full": ..., "redacted": ..., "fidelity": ...}.

    Runs in a SUBPROCESS: the oracle loads a target backend and a simulator, and a crash there must
    cost one evaluation rather than the campaign. It also lets each call carry its own fidelity
    environment without mutating the caller's.
    """
    if fidelity not in FIDELITY_ENV:
        raise ValueError(f"unknown fidelity {fidelity!r}; expected one of {sorted(FIDELITY_ENV)}")
    repo = repo_root()

    code = f'''
import json, pathlib
from merlin.runtime.backends.base import get_backend
from merlin.benchharness import evaluation as EV
MR = get_backend("muon").muon_capsule_runner
d = pathlib.Path({str(capsule_dir)!r})
try:
    r = MR.run_capsule(MR.load_capsule(d), {str(shim_pkg)!r},
                       runs_root={str(runs_root)!r}, target={target!r}, timeout={timeout})
except Exception as e:
    r = {{"status": "error", "tiers": {{}}, "numeric": {{}},
          "failure": {{"plane": "oracle", "category": "tool_crash",
                      "detail": f"{{type(e).__name__}}: {{e}}"}}}}
ev = EV.from_capsule_result(r, task_id=d.name, config_id={config_id!r}, target={target!r},
                            certifying_tiers=frozenset({sorted(CERTIFYING_TIERS)!r}),
                            method={method!r},
                            artifact_provenance="agent_kernel_in_harness_shim")
print("<<<KVC>>>" + json.dumps({{"full": ev.to_dict(), "redacted": ev.redact()}}))
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "merlin" / "python")
    env["MERLIN_KVC_KERNEL_FILE"] = str(kernel_file)
    env.setdefault("MERLIN_KVC_REFERENCE_TOOL",
                   str(repo / "out/artifacts/targets/muon/reference_v0/muon-opt"))
    env.setdefault("TMPDIR", "/scratch/agustin/tmp")
    env.update(FIDELITY_ENV[fidelity])

    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env=env, cwd=str(repo), timeout=timeout + 300)
        for line in r.stdout.splitlines():
            if line.startswith("<<<KVC>>>"):
                out = json.loads(line[len("<<<KVC>>>"):])
                out["fidelity"] = fidelity
                # The oracle's own name for what ran, so a fast estimate can never be quoted as a
                # cycle-accurate measurement by a reader of the record alone.
                out["full"]["fidelity"] = fidelity
                out["redacted"]["fidelity"] = fidelity
                return out
        detail = (r.stderr or r.stdout)[-800:]
    except subprocess.TimeoutExpired:
        detail = f"evaluation exceeded {timeout + 300}s"

    err = {"status": "error", "correct": False, "fidelity": fidelity, "failure_detail": detail}
    return {"full": dict(err), "redacted": dict(err), "fidelity": fidelity}


def feedback_text(redacted: dict) -> str:
    """The verdict as prose, for a search whose feedback channel is a string rather than JSON.

    AutoComp carries per-candidate diagnostics as text, so the same facts have to survive that shape.
    Correctness first, then latency, then where the time went -- the order an optimizer should read.
    """
    lines: list[str] = []
    ok = redacted.get("correct")
    lines.append(f"correct: {ok}   (oracle fidelity: {redacted.get('fidelity')})")
    if not ok:
        for k in ("failure_plane", "failure_category", "failure_detail"):
            if redacted.get(k):
                lines.append(f"{k}: {redacted[k]}")
        if redacted.get("mismatch_count") is not None:
            lines.append(f"mismatched elements: {redacted['mismatch_count']} "
                         f"(max abs {redacted.get('max_abs_error')}, "
                         f"max rel {redacted.get('max_rel_error')})")
    if redacted.get("cycles") is not None:
        lines.append(f"cycles: {redacted['cycles']}   %FP-peak: {redacted.get('pct_fp_peak')}")
    util = redacted.get("utilization") or {}
    if util:
        lines.append("utilization (fraction of the run's cycles):")
        for k, v in util.items():
            lines.append(f"    {k}: {'unmeasured' if v is None else f'{v:.4f}'}")
        lines.append("Low FP utilization with healthy occupancy means the wrong work is being "
                     "issued; high utilization with low occupancy means it is starved.")
    return "\n".join(lines)
