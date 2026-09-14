"""T2 — the non-terminal verilator barrier (the abc7 regression: a failed/timed-out L3 must NOT end the
run; it feeds back and re-iterates until L3-pass OR max_rounds). Tests the extracted decision function
(run_baseline_qa_loop._l3_barrier_decision) AND simulates the full barrier loop orchestration that uses it.
Deterministic, no agent, no sim."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_baseline_qa_loop as L

results = []


def _ok(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def _simulate(l3_sequence, max_rounds, start_rnd=1):
    """Replays the EXACT barrier loop structure from run_baseline_qa_loop using the real decision fn.
    l3_sequence[i] = whether verilator all_pass on attempt i. Returns (outcome, attempts, fixrounds, rnd)."""
    rnd = start_rnd
    attempts = 0
    fixrounds = 0
    while True:
        attempts += 1
        all_pass = l3_sequence[min(attempts - 1, len(l3_sequence) - 1)]
        d = L._l3_barrier_decision(all_pass, rnd, max_rounds)
        if d == "done":
            return ("converged", attempts, fixrounds, rnd)
        if d == "budget":
            return ("max_rounds", attempts, fixrounds, rnd)
        # iterate: a fix round runs (launch_agent), rnd advances — exactly the live loop
        fixrounds += 1
        rnd += 1


def main():
    # --- decision function, all branches ---
    _ok("decision: L3 pass -> done", L._l3_barrier_decision(True, 0, 6) == "done")
    _ok("decision: L3 fail, rounds left -> iterate (NON-terminal)", L._l3_barrier_decision(False, 2, 6) == "iterate")
    _ok("decision: L3 fail, budget hit -> budget (honest stop)", L._l3_barrier_decision(False, 6, 6) == "budget")
    _ok("decision: L3 fail at rnd>max -> budget", L._l3_barrier_decision(False, 7, 6) == "budget")

    # --- Scenario A: the abc7 regression — fail then pass MUST recover (not end at first failure) ---
    out, att, fr, rnd = _simulate([False, False, True], max_rounds=6, start_rnd=1)
    _ok("recovery: fail,fail,pass -> converged via re-iteration", out == "converged" and att == 3 and fr == 2,
        f"outcome={out} attempts={att} fixrounds={fr}")

    # --- Scenario B: happy path — first L3 pass converges immediately ---
    out, att, fr, _ = _simulate([True], max_rounds=6, start_rnd=1)
    _ok("happy: first L3 pass -> converged, no fix rounds", out == "converged" and att == 1 and fr == 0,
        f"outcome={out} attempts={att}")

    # --- Scenario C: persistent L3 failure -> honest max_rounds exit (NOT a crash / barrier-timeout end) ---
    out, att, fr, rnd = _simulate([False], max_rounds=3, start_rnd=1)
    _ok("budget: persistent L3 fail -> stop at max_rounds (honest, not terminal-at-first-fail)",
        out == "max_rounds" and rnd == 3 and fr >= 1, f"outcome={out} rnd={rnd} fixrounds={fr}")

    # --- Scenario D: a single L3 TIMEOUT (all_pass False) behaves like any failure: re-iterate, not end ---
    out, att, fr, _ = _simulate([False, True], max_rounds=6, start_rnd=2)
    _ok("timeout-then-pass -> recovers (timeout is non-terminal)", out == "converged" and fr == 1,
        f"outcome={out} fixrounds={fr}")

    n = sum(1 for _, ok in results if ok)
    print(f"\nT2 non-terminal barrier: {n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
