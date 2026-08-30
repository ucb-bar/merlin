#!/usr/bin/env python3
"""Gate: every path a bundle GRANTS must be a path the workspace can DELIVER.

A bundle manifest is the claim about what an arm can read; the workspace stager is the delivery. When
the two disagree the run does not fail -- it succeeds, quietly, with the arm credited for a tool it
never carried. That is the worst shape a defect can take here, because the number it produces gets
cited. Measured before this gate existed: five of six targets were assembled without the RTL-facts grant
that DEFINES the CIRCT rung, and four targets named at least one further grant that resolved to nothing,
one of them a target's ISA headers.

Two statuses fail, for two different reasons:

  missing   nothing is there, on any machine. Always an error.
  external  it is there only by following a tracked symlink out of the repo, into a path that exists on
            one machine. It delivers for the author and dangles for everyone else, and a dangling entry
            is skipped just as silently as an absent one. Ratcheted: the listed ones are known debt and
            the list MAY ONLY SHRINK.

``derived`` is fine and is reported, not failed: an RTL-facts grant is GENERATED from the target's own
RTL and gitignored on purpose. Committing it would be the wrong fix for its absence.

Usage: check_bundle_grants.py [--json] [--write-ratchet]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin" / "python"))

from merlin.common.paths import merlin_dir, repo_root  # noqa: E402
from merlin.targetgen import bundle_grants as BG  # noqa: E402
from merlin.targetgen.target_experiment import load_target_experiment  # noqa: E402

RATCHET = Path(__file__).with_name("bundle_grants_ratchet.txt")


def descriptors() -> list[Path]:
    """Every target descriptor in the repo -- globbed, never listed, so a new target is covered the day
    it lands rather than the day someone remembers to add it here."""
    return sorted(merlin_dir().glob("experiments/*/targets/*/target_experiment.yaml"))


def load_ratchet() -> set[str]:
    if not RATCHET.is_file():
        return set()
    return {ln.strip() for ln in RATCHET.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-ratchet", action="store_true",
                    help="rewrite the ratchet from what is on disk (only ever to SHRINK it)")
    a = ap.parse_args()

    findings: dict[str, dict[str, list[list[str]]]] = {}
    reviewed = 0
    for d in descriptors():
        try:
            te = load_target_experiment(d)
        except Exception as e:  # a descriptor that will not load is its own, separate gate
            print(f"[skip] {d.relative_to(repo_root())}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        reviewed += BG.grant_count(te)
        bad = BG.audit(te)
        if bad:
            findings[te.target] = {b: [list(x) for x in v] for b, v in bad.items()}

    # A grant appears once per arm that carries it; the DEFECT is the path, so collapse to paths.
    missing: set[tuple[str, str]] = set()
    external: set[str] = set()
    derived: set[str] = set()
    for target, bundles in findings.items():
        for _bundle, entries in bundles.items():
            for path, status in entries:
                if status == BG.MISSING:
                    missing.add((target, path))
                elif status == BG.EXTERNAL:
                    external.add(path)
                else:
                    derived.add(path)

    if a.json:
        print(json.dumps({"missing": sorted(list(m) for m in missing),
                          "external": sorted(external), "derived": sorted(derived)}, indent=2))

    ratchet = load_ratchet()
    if a.write_ratchet:
        keep = sorted(external & ratchet) if ratchet else sorted(external)
        RATCHET.write_text(
            "# Grants that resolve ONLY by following a tracked symlink out of the repo, into a path that\n"
            "# exists on one machine. They deliver for whoever made the link and dangle for everyone\n"
            "# else -- and a dangling grant is skipped as silently as an absent one.\n"
            "# This list MAY ONLY SHRINK. Fix one by vendoring the content, or by declaring the external\n"
            "# root in the descriptor so it is resolved at stage time instead of frozen into a symlink.\n"
            + "".join(f"{p}\n" for p in keep))
        print(f"wrote {len(keep)} entries to {RATCHET.name}")
        return 0

    rc = 0
    if missing:
        rc = 1
        print("GRANTS THAT DELIVER NOTHING (a manifest claiming a tool the arm cannot read):",
              file=sys.stderr)
        for target, path in sorted(missing):
            print(f"  {target:26s} {path}", file=sys.stderr)
    unratcheted = external - ratchet
    if unratcheted:
        rc = 1
        print("\nNEW machine-local grants (resolve here, dangle on every other clone):", file=sys.stderr)
        for path in sorted(unratcheted):
            for src, dest in sorted(BG.escaping_members(path).items()):
                print(f"  {src}\n      -> {dest}", file=sys.stderr)
        print("  Vendor the content, or declare the external root in the descriptor so it is resolved\n"
              "  at stage time. Do not add these to the ratchet.", file=sys.stderr)
    stale = ratchet - external
    if stale:
        rc = 1
        print("\nRATCHET ENTRIES NO LONGER NEEDED (the debt is paid -- delete these lines):",
              file=sys.stderr)
        for path in sorted(stale):
            print(f"  {path}", file=sys.stderr)
    if not rc and not a.json:
        print(f"bundle grants ok: {reviewed} grants across {len(descriptors())} targets "
              f"(generated ladder + materialized manifests) — 0 undeliverable, "
              f"{len(derived)} generated-on-demand, "
              f"{len(external)} ratcheted machine-local")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
