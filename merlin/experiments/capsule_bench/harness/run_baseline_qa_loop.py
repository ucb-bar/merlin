#!/usr/bin/env python3
"""Raw-baseline capsule_bench_v0 PILOT run with a redacted QA-gate iterate-to-pass loop.

A fresh, sandboxed Claude agent authors `submission/` from the allowed bundle ONLY, with goldens
withheld (masked). Between rounds an operator-side QA gate (`qa_check.py`) grades the current
submission against the public pilot capsules and writes a REDACTED verdict (pass/fail + failure
plane + trace violations — never expected/golden values) into the agent's `qa/verdict.json`. The
agent reads it and iterates. The loop ends when all 4 pilot capsules pass or a round/wall cap is hit.

Design = multi-round relaunch (robust; no daemon). Each round is a fresh agent context that resumes
from its own `submission/` + `docs/iteration_notes.md` + `qa/verdict.json`. Process telemetry
(wall/cost/tokens/tool-calls) is SUMMED across rounds = total effort to pass.

After convergence: copy the final submission into the run dir, freeze, and run the official
public+hidden grading record via `grade_agent_run.py` (against the pilot capsule subset).

This driver builds the QA substrate (UNCOUNTED). Only the agent's autonomous authoring counts.

Usage:
  run_baseline_qa_loop.py --run-id rb_pilot_0001 [--model claude-opus-4-8] [--max-rounds 6]
                          [--round-timeout 3600] [--no-oracle] [--skip-hidden] [--sandbox bwrap|none]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import _common as C
import run_agent_experiment as RX  # reuse bundle/workspace/bwrap primitives
from merlin_experiments.phase1 import treatments as T

SCRIPTS = Path(__file__).resolve().parent  # retained native brokers and tools


sys.path.insert(0, str(C.REPO / "merlin" / "python"))
from merlin.common.paths import ext_path  # noqa: E402

ARM = "raw_baseline"  # default arm; overridable via --arm (the QA loop is arm-agnostic)
# Loop+gate capsule set: the target's PUBLIC capsules, DERIVED from the descriptor's capsule_corpus
# (+ sibling corpora) and materialized per-target — NOT a committed gemmini set. So an atlas run grades
# atlas fp8/bf16 capsules (arc L3, float-tolerance), a gemmini run its i8 set (spike L2, exact_int); no
# target leak. `None` = "resolve lazily from the descriptor" in the authoring engine. The old module hook
# remains for legacy callers; fullsuite now passes its public-root selection explicitly as Treatment.
PILOT_SUBSET = None


def _te():
    """This experiment's target descriptor (honors MERLIN_TARGET_EXPERIMENT via C.EXP)."""
    from merlin.targetgen.target_experiment import load_target_experiment

    return load_target_experiment(C.DESCRIPTOR)


def _manifest():
    """This target's capability manifest (endpoint kind + sim tiers) — the second input render_prompt
    needs. Derived from the committed target_contract, so any target's runner works unchanged."""
    from merlin.targetgen.target_experiment import load_capability_manifest

    return load_capability_manifest(C.TARGET)


def main(argv: list[str] | None = None, *, treatment: T.Treatment | None = None) -> int:
    from merlin_experiments.phase1 import controller, session
    from merlin_experiments.phase1.options import parse_options

    from merlin.common.paths import env as machine_env

    options = parse_options(argv, default_arm=ARM)
    # Refuse invalid options before resolving legacy machine defaults.
    refusal = session.validate_options(options)
    if refusal is not None:
        return refusal
    bundle_id = options.bundle or RX.ARM_BUNDLE[options.arm]
    conda = ext_path("chipyard") / ".conda-env"
    return controller.run(
        C.CONTEXT,
        options,
        bundle_manifest=C.BUNDLES / bundle_id / "input_bundle_manifest.yaml",
        bundle_id=bundle_id,
        oracle_timing=SCRIPTS / ".oracle_timing.json",
        launcher_argv=tuple(sys.argv[1:] if argv is None else argv),
        language=os.environ.get("PILOT_LANG", ""),
        treatment=treatment,
        public_root=PILOT_SUBSET,
        machine_defaults={
            "AWS_BEARER_TOKEN_BEDROCK": machine_env("AWS_BEARER_TOKEN_BEDROCK") or "",
        },
        library_paths=(C.REPO / ".compat_lib", conda / "lib", conda / "riscv-tools/lib"),
        source_entrypoint=Path(__file__),
        require_native_source=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
