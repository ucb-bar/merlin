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

import os
import sys
from pathlib import Path

import yaml


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

import qa_check_rtlchecks                       # wraps the real qa_check
sys.modules["qa_check"] = qa_check_rtlchecks    # the loop's local `import qa_check` resolves to the wrapper

import run_agent_experiment as RX               # noqa: E402
# Serve the rtlchecks bundle for the merlin_assisted arm (identical tools + the rtl_checks addendum).
RX.ARM_BUNDLE["merlin_assisted"] = "merlin_assisted_rtlchecks_public_v0"

import run_baseline_qa_loop as L                # noqa: E402  (imported AFTER the swaps above)
import _common as C                             # noqa: E402

_RTLCHECKS_BUNDLE = "merlin_assisted_rtlchecks_public_v0"


def _option_values(argv: list[str], option: str) -> list[str | None]:
    """Return every spelling of an argparse value option, preserving malformed missing values."""
    values: list[str | None] = []
    for i, token in enumerate(argv):
        if token == option:
            values.append(argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else None)
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    return values


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    requested_bundles = _option_values(argv, "--bundle")
    if (len(requested_bundles) > 1 or any(value is None for value in requested_bundles)):
        print(f"REFUSING: Arm-4 requires at most one well-formed --bundle; "
              f"received {requested_bundles!r}", file=sys.stderr)
        return 4
    selected_bundle = requested_bundles[0] if requested_bundles else _RTLCHECKS_BUNDLE
    if (not selected_bundle or Path(selected_bundle).name != selected_bundle
            or not selected_bundle.startswith("merlin_assisted_rtlchecks_")):
        print(f"REFUSING: invalid Arm-4 bundle identity {selected_bundle!r}", file=sys.stderr)
        return 4
    manifest_path = C.BUNDLES / selected_bundle / "input_bundle_manifest.yaml"
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 — an unreadable treatment must fail before authoring
        print(f"REFUSING: cannot read Arm-4 bundle manifest {manifest_path}: {exc}", file=sys.stderr)
        return 4
    if (manifest.get("bundle_id") != selected_bundle
            or manifest.get("arm") != "merlin_rtlchecks"):
        print(f"REFUSING: {selected_bundle!r} is not a generated merlin_rtlchecks bundle", file=sys.stderr)
        return 4
    requested_arms = _option_values(argv, "--arm")
    if any(value != "merlin_assisted" for value in requested_arms):
        print("REFUSING: the Arm-4 RTL-checks wrapper requires --arm merlin_assisted", file=sys.stderr)
        return 4
    if not requested_arms:                       # this track is always the merlin arm + checks
        argv += ["--arm", "merlin_assisted"]
    # Reassert immediately before the baseline parser applies an allowed, identical --bundle.  This
    # makes an earlier in-process mutation fail closed too, while still permitting launchers to pin the
    # canonical bundle explicitly in their auditable command line.
    RX.ARM_BUNDLE["merlin_assisted"] = selected_bundle
    assert RX.ARM_BUNDLE["merlin_assisted"] == selected_bundle, "bundle swap did not take"
    assert sys.modules.get("qa_check") is qa_check_rtlchecks, "qa_check injection did not take"
    return L.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
