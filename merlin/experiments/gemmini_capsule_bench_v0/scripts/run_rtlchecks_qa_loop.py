"""Isolated RTL-checks agentic track — the merlin_assisted loop + advisory RTL-derived checks.

This launcher reuses the **unmodified** baseline QA loop (`run_baseline_qa_loop.py`) and merlin_assisted
arm, changing only two things, both in-process (no existing file is edited):

  1. swaps the merlin_assisted *bundle* to ``merlin_assisted_rtlchecks_public_v0`` (identical allowed/denied
     + a TASK addendum describing the rtl_checks feedback), so the served workspace is an exact mirror of
     merlin_assisted plus the addendum;
  2. injects :mod:`qa_check_rtlchecks` in place of ``qa_check``, so each round's redacted verdict gains an
     advisory ``rtl_checks`` block (FileCheck over the emitted MLIR + decoded trace; bounds from the
     CIRCT-extracted RTL facts). The block does NOT gate pass/fail.

Result: a clean A/B — run this with the same task/model/run accounting as merlin_assisted; the ONLY
difference the agent sees is the extra RTL-grounded feedback. Use a distinct ``--run-id`` (outputs land in
``runs/merlin_assisted/<run-id>``; this track marks them with ``run_dir/TRACK_RTLCHECKS``).

Usage (mirror the baseline loop's flags)::

    run_rtlchecks_qa_loop.py --run-id rtlchecks_0001 --model claude-opus-4-8 [--max-rounds 6] ...
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import qa_check_rtlchecks                       # wraps the real qa_check
sys.modules["qa_check"] = qa_check_rtlchecks    # the loop's local `import qa_check` resolves to the wrapper

import run_agent_experiment as RX               # noqa: E402
# Serve the rtlchecks bundle for the merlin_assisted arm (identical tools + the rtl_checks addendum).
RX.ARM_BUNDLE["merlin_assisted"] = "merlin_assisted_rtlchecks_public_v0"

import run_baseline_qa_loop as L                # noqa: E402  (imported AFTER the swaps above)

_RTLCHECKS_BUNDLE = "merlin_assisted_rtlchecks_public_v0"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--arm" not in argv:                     # this track is always the merlin arm + checks
        argv += ["--arm", "merlin_assisted"]
    assert RX.ARM_BUNDLE["merlin_assisted"] == _RTLCHECKS_BUNDLE, "bundle swap did not take"
    assert sys.modules.get("qa_check") is qa_check_rtlchecks, "qa_check injection did not take"
    return L.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
