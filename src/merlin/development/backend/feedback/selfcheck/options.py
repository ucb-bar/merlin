"""Command-line options for an explicit selfcheck request."""

from __future__ import annotations

import argparse
from pathlib import Path

from merlin.development.backend.context import (
    add_context_arguments,
)
from merlin.development.backend.feedback.certification.promotion_admission import REQUIRED_SIM


def parser():
    from merlin.development.backend.feedback.selfcheck.adapters import SIM_CHOICES

    ap = argparse.ArgumentParser(description="Agent self-check runner (redacted; spike/verilator/vcs).")
    add_context_arguments(ap)
    ap.add_argument("--suite", choices=("core", "scalability", "programs"), default="core")
    ap.add_argument("--capsules-root", type=Path, help="explicit public corpus, never rediscovered when supplied")
    ap.add_argument("--contract", type=Path, help="explicit grading contract root")
    ap.add_argument("--rtl-facts", type=Path, help="Selected facts in the verified frozen input snapshot")
    ap.add_argument("--submission", default="submission", help="path to your package (with manifest.yaml)")
    # DEFAULTS TO THE GRADE'S OWN SELECTION, not to one engine. The capsules declare their mandatory
    # tiers (a functional screen, an elaborated-RTL cert tier, a provided-host tier), and the ladder
    # runs cheapest-measured-first with fail-fast, so the screen still refutes a broken submission at
    # screen cost -- what changes is that a capsule which PASSES the screen goes on to certify in the
    # same request. A single engine name ("spike") is a legitimate fast screen, but it CANNOT certify:
    # each mandatory tier it does not run reports unavailable and the capsule is not a pass.
    # Measured on one run: 166 of 217 self-check capsule results were incomplete because each request
    # ran a single engine, and pass/fail was learned only from the background grade.
    ap.add_argument(
        "--sim",
        choices=list(SIM_CHOICES),
        default=None,
        help=f"default {REQUIRED_SIM!r}: every tier the official grade runs; one engine name is a narrower screen",
    )
    ap.add_argument(
        "--capsules",
        default="all",
        help="'all', 'failing' (only what did not pass last time — fast iteration), 'changed' (build and "
        "screen every public capsule, reuse exact-byte RTL certificates, simulate only those whose bytes are "
        "neither certified nor known to fail), or comma-separated capsule names",
    )
    # 0 = derive from this host (see capsule_grade.default_grade_workers): the per-capsule builds and
    # sims are independent, and a literal 8 left most of a 48-core machine idle through every sweep.
    ap.add_argument(
        "--workers", type=int, default=0, help="parallel per-capsule workers; 0 (default) sizes them to this host"
    )
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="", help="optional: also write the redacted JSON here")
    ap.add_argument("--progress-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--artifact-root", default="", help=argparse.SUPPRESS)
    ap.add_argument("--model-gate-voucher", action="append", default=[], help=argparse.SUPPRESS)
    ap.add_argument(
        "--caller-layout",
        default="",
        help="INSTEAD of grading, inspect a submission-relative command buffer's selected physical pointer layout",
    )
    # THE FLAG select_tiers WAS WRITTEN FOR. It existed as a function with six tests pinning it and no
    # way to reach it: argparse never accepted --tiers, so the broker's promotion jobs -- which always
    # forward `--tiers <cert_tier>` -- died on "unrecognized arguments: --tiers L3" before doing any
    # work, and the broker reported that as "no verdict produced". Measured on
    # merlincirct_arm4_func_20260901_v4: 19 promotion jobs, every one killed by argparse.
    ap.add_argument(
        "--tiers",
        default="",
        help="comma-separated tiers to grade (default: this target's cheap loop ladder), e.g. L3 for a cert run",
    )
    ap.add_argument(
        "--shape-coverage",
        action="store_true",
        help="INSTEAD of the capsule suite, probe whether your backend LOWERS the same "
        "contraction at one tile and at two tiles in each of M, K and N. Costs no "
        "simulator and no oracle (it only runs your emit path), so it is cheap to run "
        "often. The public capsules are a fixed set of shapes; passing all of them "
        "says nothing about whether you generalize past them, and a backend that "
        "hardcodes those shapes scores 100%% here and fails the held-out grade.",
    )
    ap.add_argument(
        "--offload-census",
        action="store_true",
        help="INSTEAD of the capsule suite, report what your package puts on the "
        "accelerator: per public program, the work that reached a unit, what was "
        "placed on the host, what was declined, and what was emitted with no "
        "statement at all. Runs only your emit path, so it is cheap. A passing "
        "program whose work all ran on the host is not what this target is for.",
    )
    ap.add_argument(
        "--model-layers",
        action="store_true",
        help="INSTEAD of the capsule suite, grade your package at the functional tier against "
        "capsules generated from the layers REAL MODELS form on this target, at the models' own "
        "extents. Not part of your score. A backend that passes the suite and fails these runs "
        "toy shapes only. Honors --capsules: name one layer to check one layer.",
    )
    ap.add_argument(
        "--candidate-diagnostics",
        action="store_true",
        help="INSTEAD of grading, inspect own public source/emission/host work; no simulator or baseline",
    )
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="compile original public scale/frontier tensor sources and link a no-data probe; no execution",
    )
    ap.add_argument(
        "--scale-probe",
        action="store_true",
        help="with --compile-only, inspect public scale/tail LLVM emission before object compilation",
    )
    ap.add_argument("--compiler-bundle-manifest", type=Path, help=argparse.SUPPRESS)
    ap.add_argument(
        "--dynamic-census", action="store_true", help="append bounded own-ELF activity after functional screen"
    )

    return ap
