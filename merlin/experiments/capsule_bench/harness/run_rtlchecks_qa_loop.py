"""Isolated RTL-checks agentic track — the merlin_assisted loop + advisory RTL-derived checks.

This launcher reuses the baseline QA loop and merlin_assisted arm through an explicit
source-attributed Treatment, changing only two things:

  1. swaps the merlin_assisted *bundle* to ``merlin_assisted_rtlchecks_public_v0`` (identical allowed/denied
     + a TASK addendum describing the rtl_checks feedback), so the served workspace is an exact mirror of
     merlin_assisted plus the addendum;
  2. selects :mod:`qa_check_rtlchecks` callbacks, so each round's redacted verdict gains an
     advisory ``rtl_checks`` block (FileCheck over the emitted MLIR + decoded trace; bounds from the
     CIRCT-extracted RTL facts). The block does NOT gate pass/fail.

Result: a clean A/B — run this with the same task/model/run accounting as merlin_assisted; the ONLY
difference the agent sees is the extra RTL-grounded feedback. Use a distinct ``--run-id`` (outputs land in
``runs/merlin_assisted/<run-id>``; this track marks them with ``run_dir/TRACK_RTLCHECKS``).

Usage (mirror the baseline loop's flags)::

    run_rtlchecks_qa_loop.py --run-id rtlchecks_0001 --model claude-opus-4-8 [--max-rounds 6] ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _descriptor_for_invocation(argv0: str) -> Path | None:
    """Return the descriptor adjacent to a target-local launcher alias, if one exists.

    ``_common`` intentionally defaults when no descriptor is selected, which is useful for the canonical
    harness entrypoint but dangerous for an alias under ``targets/<name>/scripts``: invoking that alias
    used to stage the default target while its pathname advertised another one.  Derive from the
    filesystem layout before importing ``_common``; an explicit environment selection still wins.
    """
    invoked = Path(argv0).expanduser().absolute()
    candidate = invoked.parent.parent / "target_experiment.yaml"
    return candidate.resolve() if candidate.is_file() else None


if not os.environ.get("MERLIN_TARGET_EXPERIMENT", "").strip():
    _invoked_descriptor = _descriptor_for_invocation(sys.argv[0])
    if _invoked_descriptor is not None:
        os.environ["MERLIN_TARGET_EXPERIMENT"] = str(_invoked_descriptor)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _common as C  # noqa: E402
import run_baseline_qa_loop as L  # noqa: E402
from merlin_experiments.phase1.feedback import rtlchecks  # noqa: E402

_RTLCHECKS_BUNDLE = "merlin_assisted_rtlchecks_public_v0"


def main(argv: list[str] | None = None) -> int:
    from merlin.common.paths import ext_path

    try:
        arguments = rtlchecks.prepare_arguments(
            list(sys.argv[1:] if argv is None else argv), bundles=C.BUNDLES, default_bundle=_RTLCHECKS_BUNDLE
        )
    except ValueError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 4
    candidates = (
        C.REPO / "third_party/llvm-build/bin/FileCheck",
        ext_path("chipyard") / ".conda-env/riscv-tools/bin/FileCheck",
    )
    return L.main(arguments, treatment=rtlchecks.treatment(C.CONTEXT, filecheck_candidates=candidates))


if __name__ == "__main__":
    raise SystemExit(main())
