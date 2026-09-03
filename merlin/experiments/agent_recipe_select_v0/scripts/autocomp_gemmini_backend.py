"""Serve AutoComp's ``EvalBackend`` interface from THIS study's oracle, so both arms are scored alike.

WHY THE NATIVE BACKEND CANNOT BE USED, and this is not a preference. AutoComp ships
`GemminiEvalBackend`, and its performance metric is `latency_spike_raw` -- **spike cycles**
(`autocomp/backend/gemmini/gemmini_eval.py:526`). Spike does not model Gemmini timing at all: its
cycle counts plateau at ~120 from 4K to 2M MACs, because a RoCC matmul retires in ~0 modelled cycles.
So its "latency" is not a performance number for this target, and an arm optimising against it is
optimising against noise. It is also pointed at a different checkout (`chipyard-mx`, MX-Gemmini) than
the config this study measures.

WHAT THIS SUBSTITUTES. The same oracle the recipe arm uses: the agent's C is compiled with the
identical toolchain and flags (`baremetalc_corroborate._build_cmd`, which mirrors
`contract.compile.link_elf`), run on **GSIM** elaborated RTL, and graded bit-exactly against a
host-computed golden over the SAME deterministic stimulus. So workload, correctness oracle, timing
oracle and golden are held constant across arms; what differs is the search space, which is the thing
the ablation is about.

⚠️ **WHAT IS NOT HELD CONSTANT, and must be reported rather than implied away.** The two arms do not
search comparable spaces: the recipe arm chooses among 20 compiler-defined points, while AutoComp
rewrites arbitrary C. That is the ABLATION, not a flaw -- but it means "one candidate" is not the same
unit on both sides, so a per-candidate comparison is invalid and only spend-to-quality curves are
meaningful. Likewise a compiler-constructed candidate cannot be malformed, so a 0% invalid rate on
that side is a property of the mechanism, not evidence the model is better at writing code.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _track as T                                                    # noqa: E402

#: merlin's interpreter, resolved EXPLICITLY. AutoComp runs under its own venv, and an evaluator that
#: inherited `sys.executable` from its caller once graded a known-good reference kernel as wrong.
MERLIN_PY = T.REPO / ".venv" / "bin" / "python"
EVALUATOR = Path(__file__).resolve().parent / "eval_candidate.py"


def clean_code(text: str) -> str:
    """Strip a markdown fence off an LLM reply, structurally (no regex, per the repo rule).

    Implemented here rather than imported from AutoComp's `gemmini_eval`, because importing that
    module pulls in its spike-based backend and its hardcoded `chipyard-mx` path -- the very things
    this file exists to avoid.
    """
    t = text.strip()
    if "```" not in t:
        return t
    parts = t.split("```")
    # A fenced reply is ``prose ``` [lang]\ncode ``` prose``; take the longest fenced block, which is
    # the code even when the model also fences a short snippet in its explanation.
    blocks = [parts[i] for i in range(1, len(parts), 2)]
    if not blocks:
        return t
    best = max(blocks, key=len)
    first, _, rest = best.partition("\n")
    return (rest if first.strip().lower() in ("c", "cpp", "c++", "") else best).strip()


class GsimGemminiEvalBackend:
    """AutoComp's 4-method backend contract, served by GSIM + a host golden.

    Deliberately NOT a subclass of AutoComp's ``EvalBackend``: keeping it duck-typed means this module
    imports nothing from AutoComp and stays unit-testable without it installed -- the same choice the
    repo's existing muon bridge documents.
    """

    #: The stimulus tensor names, which fix the golden. Shared with the recipe arm's capsules so both
    #: arms are graded on identical operand values.
    NAME_A = "A0"
    NAME_B = "W"

    def __init__(self, *, M: int, K: int, N: int, workdir: Path, simulator: str = "gsim",
                 timeout: int = 3600):
        self.M, self.K, self.N = M, K, N
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.simulator = simulator
        self.timeout = timeout
        self._n = 0
        #: Every candidate ever seen, INCLUDING the ones AutoComp's search discards. Its own
        #: trajectory records only survivors, and a rejected candidate still cost tokens.
        self.trajectory: list[dict] = []
        if not MERLIN_PY.exists():
            raise SystemExit(f"merlin interpreter not found at {MERLIN_PY}")

    # -- AutoComp contract ------------------------------------------------------------------

    def harness_overhead(self, *_a, **_k) -> int:
        """Cycles the measurement harness itself costs. Zero here: the cycle window is the kernel's
        own `read_cycles` bracket, so no overhead is included to subtract."""
        return 0

    def get_backend_specific_rules(self) -> list[str]:
        return [
            f"The kernel computes C = A x B with A {self.M}x{self.K} int8, B {self.K}x{self.N} "
            f"int8, and C {self.M}x{self.N} int32.",
            "A and B must be filled with the exact deterministic pattern already in the seed "
            "kernel; changing the input values changes the answer and will be graded incorrect.",
            "Correctness is BIT-EXACT against a host-computed reference. There is no tolerance.",
            "Cycles are measured on elaborated RTL, not on a functional model, so instruction "
            "count is not the objective -- overlapped transfers can be free while a barrier "
            "cannot.",
            "The host Rocket core has no FPU: any floating-point operation traps on real RTL.",
            "The timing bracket uses 'unsigned long', not 'uint64_t', so it does not depend on "
            "<stdint.h>. Keep it that way if you restructure the includes.",
            "You MUST keep the read_cycles() timing bracket and the printf of "
            "'METRIC cycles <n>' that appear in the seed, and keep the 'OUT C' tensor dump and "
            "'DONE' line. A candidate with no METRIC cycles line has no measured cost and is "
            "rejected outright -- it does not score as fast.",
        ]

    def get_hw_feedback(self, prob, code_strs) -> list[list[str]]:
        """No per-candidate hardware counters are wired for this arm, so this is EMPTY rather than
        invented. AutoComp's `give_hw_feedback` must therefore be left off, and that is recorded as a
        deviation instead of being papered over with a plausible-looking string."""
        return [[] for _ in code_strs]

    def evaluate_code(self, prob, code_strs: list[str], simulator: str | None = None) -> list[dict]:
        sim = simulator or self.simulator
        out: list[dict] = []
        for code in code_strs:
            self._n += 1
            name = f"ac_cand_{self._n:04d}"
            src = self.workdir / f"{name}.src.c"
            src.write_text(clean_code(code), encoding="utf-8")
            argv = [str(MERLIN_PY), str(EVALUATOR), "--source", str(src), "--name", name,
                    "--workdir", str(self.workdir), "--M", str(self.M), "--K", str(self.K),
                    "--N", str(self.N), "--name-a", self.NAME_A, "--name-b", self.NAME_B,
                    "--simulator", sim, "--timeout", str(self.timeout)]
            t0 = time.time()
            try:
                # +180s so the evaluator's own timeout fires first and reports a REASON, rather than
                # this wrapper killing it and leaving an unattributable failure.
                r = subprocess.run(argv, capture_output=True, text=True,
                                   timeout=self.timeout + 180)
                res = json.loads((r.stdout or "").strip().splitlines()[-1])
            except Exception as exc:
                res = {"compiled": False, "correct": False, "cycles": None,
                       "detail": f"evaluator failed: {type(exc).__name__}: {exc}"}
            dur = time.time() - t0
            rec = {"index": self._n, "name": name, "source_chars": len(code),
                   "compiled": res.get("compiled"), "correct": res.get("correct"),
                   "cycles": res.get("cycles"),
                   "latency": res["cycles"] if (res.get("correct")
                                                and isinstance(res.get("cycles"), int))
                   else float("inf"),
                   "stderr": res.get("detail") or "",
                   "eval_seconds": round(dur, 2), "engine": sim,
                   "engine_config": res.get("engine_config", T.GSIM_CONFIG)}
            self.trajectory.append(rec)
            # AutoComp's contract: `latency` is the score, inf means unusable, and `stderr` is its
            # diagnostic channel — so the reason rides there and the model can repair from it.
            out.append({"correct": bool(rec["correct"]), "compiled": bool(rec["compiled"]),
                        "test_results": [{"correct": bool(rec["correct"]),
                                          "latency": rec["latency"]}],
                        "stderr": rec["stderr"], "latency": rec["latency"]})
        return out

    # -- seed ------------------------------------------------------------------------------

    def seed_source(self) -> str:
        """The canonical gemmini C kernel AutoComp starts from, cycle-instrumented.

        Produced across the interpreter seam, because the template and its stimulus live in merlin's
        venv while this class may be running under AutoComp's.

        ⚠️ The agent can still delete the timing bracket, and if it does the candidate is REFUSED for
        having no measured cost rather than scored 0. That is a real failure mode of a code-writing
        arm — breaking its own measurement contract — and one the recipe arm cannot exhibit, because
        it never touches the harness. It is reported, not designed away.
        """
        r = subprocess.run([str(MERLIN_PY), str(EVALUATOR), "--emit-seed",
                            "--M", str(self.M), "--K", str(self.K), "--N", str(self.N),
                            "--name-a", self.NAME_A, "--name-b", self.NAME_B],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not r.stdout.strip():
            raise SystemExit(f"could not emit the seed kernel: {r.stderr[-800:]}")
        return r.stdout
