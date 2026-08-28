#!/usr/bin/env python3
"""Make AutoComp grade through THIS study's evaluator instead of its own.

AutoComp is the credible kernel-generation baseline: a real published search, already Radiance-capable,
already run against these models. Reimplementing its algorithm would produce a strawman opponent and a
result about my scaffolding. So the search stays exactly as it is, and only the thing it measures with
is replaced.

That substitution is the whole point. If AutoComp scored kernels on its own harness while the compiler
arm scored on ours, a difference between them could be the oracle rather than the method, and no
amount of statistics would separate the two afterwards. One evaluator, one fidelity ladder, one
definition of correct.

WHAT AUTOCOMP GETS BACK. Its contract is a per-candidate dict of {correct, compiled, latency, stderr}.
`stderr` is its diagnostic channel -- the search feeds it back when it retries a failed candidate -- so
the full verdict goes there as prose: correctness, latency, and utilization. An optimizer told only
"slower" cannot act; one told "3% FP utilization at 50% occupancy" can.

LATENCY SIGN. AutoComp minimises `latency`, so a lower number must mean a better kernel. Cycles from
the oracle already have that sense; nothing is inverted here, and an incorrect candidate reports no
latency at all rather than a fast one.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kvc_eval  # noqa: E402  (path set above)


class KvcMuonEvalBackend:
    """AutoComp's ``EvalBackend`` interface, served by this study's evaluator.

    Deliberately does NOT subclass AutoComp's class: importing it would drag the whole framework in
    just to satisfy an interface that is four methods wide, and the bridge has to be importable for
    testing without AutoComp installed.
    """

    def __init__(self, capsule_dir: Path, *, shim_pkg: Path, runs_root: Path,
                 target: str = "radiance", fidelity: str = "fast", hw_config=None,
                 method: str = "autocomp") -> None:
        self.capsule_dir = Path(capsule_dir)
        self.shim_pkg = Path(shim_pkg)
        self.runs_root = Path(runs_root)
        self.target = target
        self.fidelity = fidelity
        self.hw_config = hw_config
        self.method = method
        #: Every candidate ever evaluated, in order. The study reports the cost of the failures too,
        #: so the search's own filtering must not be the only record of what was tried.
        self.trajectory: list[dict] = []

    def __repr__(self) -> str:
        return (f"KvcMuonEvalBackend(target={self.target!r}, fidelity={self.fidelity!r}, "
                f"capsule={self.capsule_dir.name!r})")

    def harness_overhead(self, prob) -> int:
        """Cycles to subtract as harness cost.

        Zero: the oracle reports the kernel's own cycles, and every arm is measured through the same
        harness, so subtracting an estimate would add noise without improving comparability.
        """
        return 0

    def get_backend_specific_rules(self) -> list[str]:
        """What the code generator must know about the artifact it is producing."""
        return [
            "Emit LLVM-dialect MLIR, not C++ and not assembly.",
            "One `llvm.func` named `radiance_kernel`; parameters are plain pointers in the order "
            "[weight tensors] ++ [inputs in command order] ++ [outputs in command order].",
            "Read with llvm.getelementptr + llvm.load, write with llvm.store.",
            "Warps, barriers and scheduling come from the runtime -- emit plain scalar compute.",
            "Do not emit .insn, raw .word, DRAM addresses, or a halt.",
            "Correctness gates everything: an incorrect kernel scores no latency at all.",
        ]

    def evaluate_code(self, prob, code_strs: list[str], simulator: str | None = None) -> list[dict]:
        """Grade each candidate. Returns AutoComp's per-candidate stat dicts."""
        out: list[dict] = []
        for i, code in enumerate(code_strs):
            out.append(self._evaluate_one(code, index=i))
        return out

    def _evaluate_one(self, code: str, *, index: int) -> dict:
        import tempfile

        if not (code or "").strip():
            # An empty candidate is a generation failure, not a wrong answer. Saying so lets the
            # search retry with context instead of scoring it as an incorrect kernel.
            rec = {"correct": False, "compiled": False, "test_results": {0: False},
                   "stderr": "candidate was empty: no kernel text was produced"}
            self.trajectory.append({"index": index, **rec})
            return rec

        tmp = Path(tempfile.mkdtemp(prefix="kvc-ac-", dir="/scratch/agustin/tmp"))
        kernel = tmp / "kernel.llvm.mlir"
        kernel.write_text(code)

        res = kvc_eval.evaluate(
            self.capsule_dir, kernel,
            shim_pkg=self.shim_pkg, runs_root=self.runs_root,
            target=self.target, fidelity=self.fidelity, method=self.method,
        )
        red, full = res["redacted"], res["full"]
        correct = bool(red.get("correct"))

        stat: dict = {
            "correct": correct,
            "compiled": bool(red.get("compile_ok")),
            "test_results": {0: correct},
            # The diagnostic channel the search reads on a retry. Sent for a PASS too, since the
            # optimization stage needs the utilization of a working kernel to know what to improve.
            "stderr": kvc_eval.feedback_text(red),
        }
        if correct and red.get("cycles") is not None:
            stat["latency"] = max(int(red["cycles"]), 1)

        self.trajectory.append({
            "index": index, "correct": correct, "fidelity": res["fidelity"],
            "cycles": red.get("cycles"), "utilization": red.get("utilization"),
            "source_chars": len(code), "verdict": full.get("verdict"),
        })
        return stat
