"""Reuse the frozen guarded VLEN corpus with explicit copy-loop lowering.

Only the native lowering invocation changes; graph, schedule, reference,
guards, ELF harness and vector-length checks are the existing corpus.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import prove_im2col_stream as corpus

from rvv.transforms import static_copy


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--compiler-python", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    original_run = corpus.run
    reports = []

    def run(argv, directory, name, timeout=60):
        if name.startswith("lower"):
            pipeline = next(
                str(v) for v in argv if str(v).startswith("--pass-pipeline=")
            )
            prefix = "--pass-pipeline=builtin.module("
            if not pipeline.startswith(prefix) or not pipeline.endswith(")"):
                raise ValueError("explicit whole-module pipeline required")
            report = static_copy.lower_file(
                argv[1],
                pipeline=pipeline[len(prefix) : -1],
                mlir_opt=argv[0],
                compiler_python=options.compiler_python,
                work=directory / (name + "-copy-loops"),
                enabled=True,
            )
            output = Path(argv[argv.index("-o") + 1])
            shutil.copyfile(report["llvm_dialect"], output)
            reports.append(report)
            return None
        return original_run(argv, directory, name, timeout)

    corpus.run = run
    corpus.main()
    output = Path(remaining[remaining.index("--output") + 1])
    inputs = [Path(__file__), Path(static_copy.__file__), Path(corpus.__file__)]
    report = {
        "schema": "rvv.static_copy_vlen_proof.v1",
        "pass": True,
        "input_sha256": {str(p.resolve()): corpus.sha(p) for p in inputs},
        "corpus_report_sha256": corpus.sha(output / "report.json"),
        "copy_reports": reports,
        "timing_accuracy_certified": False,
        "physical_utilization_measured": False,
    }
    (output / "copy-proof.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
