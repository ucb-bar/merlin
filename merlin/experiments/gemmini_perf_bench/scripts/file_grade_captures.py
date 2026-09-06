#!/usr/bin/env python3
"""File engine cross-validation captures from a FINISHED functional grade.

WHY THIS EXISTS. A performance campaign cannot launch without a full public+hidden GSIM equivalence
certificate, and building one costs hours of REFERENCE-engine simulation in front of a run that cannot
start until it finishes. Meanwhile the functional grade requires no certificate at all -- it grades on
the candidate engine and trusts it -- and it has already built, for every capsule, the exact three
inputs a capture needs: the command buffer, the lowered module, and the ELF compiled from them. The
expensive half was being paid twice: once, silently, by nobody, and again by the campaign.

This walks a completed grade, runs the reference leg on the SAME artifacts through the same
``capture_case`` a certificate uses, and files the result in the content-addressed capture store. The
certificate build then consults that store before paying for either engine, so the work is spent
overlapped with grading instead of serialised in front of the campaign.

IT PROVES NOTHING NEW AND IS NOT A CERTIFICATE. Every capture it files is the same document
``capture_case`` produces, keyed on the same (ELF, engine pins), and re-validated by the certificate
that adopts it. Running this is an optimisation; skipping it changes only how long the campaign waits.

FAIL-OPEN, PER CAPSULE. A capsule whose capture cannot be taken is reported and skipped: an incomplete
store costs time, while a missing one costs only the time it would have saved. The engine pins,
however, fail CLOSED -- a capture filed under a pin set that does not describe the engines that produced
it is worse than no capture at all.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import produce_gsim_certificate as PRODUCER      # noqa: E402
from merlin.perf import capture_store as STORE   # noqa: E402
from merlin.perf.engine_pins import engine_pins  # noqa: E402


def _artifact_paths(target: str) -> PRODUCER.ArtifactPaths:
    """The five engine artifacts, resolved and digest-verified from the pin registry."""
    pins = engine_pins(target)
    return PRODUCER.ArtifactPaths(
        gsim_firrtl=Path(pins["gsim_firrtl"]["path"]),
        verilator_firrtl=Path(pins["verilator_firrtl"]["path"]),
        gsim_model=Path(pins["gsim_model"]["path"]),
        gsim_binary=Path(pins["gsim_binary"]["path"]),
        verilator_binary=Path(pins["verilator_binary"]["path"]))


def _graded_capsules(run_dir: Path) -> list[tuple[str, Path]]:
    """``(capsule name, generated dir)`` for every capsule this grade actually lowered.

    A capsule is included only when the grade left BOTH artifacts a capture consumes; one that declined
    to lower emitted neither, and there is nothing to cross-validate about a program that was not built.
    """
    found: list[tuple[str, Path]] = []
    for generated in sorted(run_dir.rglob("generated")):
        if not generated.is_dir():
            continue
        if not (generated / "command_buffer.json").is_file():
            continue
        if not (generated / "lowered.llvm.mlir").is_file():
            continue
        found.append((generated.parent.name, generated))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="a finished capsule-bench run directory")
    ap.add_argument("--capsule-root", required=True, type=Path,
                    help="corpus root holding <capsule>/capsule.yaml")
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--reference-timeout", type=int, default=25200,
                    help="the reference engine is more than an order of magnitude slower; one deadline "
                         "sized for the candidate kills exactly the deep cases a capture is wanted for")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    artifacts = _artifact_paths(args.target)          # fails closed, and says which pin
    pins = artifacts.pinned()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    filed = hit = skipped = 0
    for name, generated in _graded_capsules(Path(args.run_dir))[:args.limit]:
        manifest = Path(args.capsule_root) / name / "capsule.yaml"
        if not manifest.is_file():
            print(f"  {name}: no manifest under {args.capsule_root}; skipped", flush=True)
            skipped += 1
            continue
        try:
            before = STORE.census(args.target)["entries"]
            PRODUCER.capture_case(
                target=args.target, capsule_manifest=manifest, artifact_dir=generated,
                workdir=work / name, artifacts=artifacts, timeout=args.timeout,
                reference_timeout=args.reference_timeout)
            after = STORE.census(args.target)["entries"]
            if after > before:
                filed += 1
                print(f"  {name}: filed", flush=True)
            else:
                hit += 1
                print(f"  {name}: already stored", flush=True)
        except Exception as exc:                       # noqa: BLE001 - per-capsule, fail open
            skipped += 1
            print(f"  {name}: skipped ({type(exc).__name__}: {str(exc)[:110]})", flush=True)
    print(f"\nfiled {filed}, already stored {hit}, skipped {skipped}; "
          f"store now {STORE.census(args.target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
