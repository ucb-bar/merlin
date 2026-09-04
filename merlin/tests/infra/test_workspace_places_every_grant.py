"""Every allowed grant must be reachable in the workspace AT THE PATH THE AGENT IS TOLD.

The bwrap mount and the workspace assembly answer two different questions, and only the second one is
the agent's. `readiness_check` F3 proves a grant RESOLVES and F4 proves the RESOLVED DESTINATION is
visible inside the sandbox -- both passed for atlas while the arm that exists to compile from RTL facts
could not find them.

Measured on merlincirct_atlasp1arm4 (2026-09-04): the manifest granted
`merlin/targets/atlas/contracts/rtl_facts/`, TASK.md told the agent to derive the ISA/mesh/datapath from
it, and `ALLOWED_MERLIN_TOOLS.md` listed it by name. A target's package does not always live under
`merlin/targets/`, so the registry deliberately redirects that grant to the target's real facts home --
the bwrap mount resolved it correctly and the bytes WERE in the sandbox. But the workspace assembler
computed `<repo>/<declared path>`, found nothing, and hit a bare `continue`: no entry, no warning. The
agent looked where it was told, found nothing, and shipped a backend using no RTL facts at all. All five
RTL-workflow conformance checks failed and it read as the agent ignoring its tooling.
"""
from __future__ import annotations

import sys

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen.sandbox.bwrap import resolve_grant

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))


def _bundles():
    root = merlin_dir() / "experiments/capsule_bench/targets"
    return sorted(root.glob("*/input_bundles/*/input_bundle_manifest.yaml"))


#: Grants that resolve to nothing TODAY, recorded so the count may only fall. Both are a `task/`
#: directory a bundle names for a target that does not ship one; neither target is currently launchable,
#: which is why nothing caught them. Delete an entry when the path appears -- never add one.
_KNOWN_UNRESOLVABLE = frozenset({
    "experiments/capsule_bench/targets/mx_gemmini/task/",
    "experiments/capsule_bench/targets/saturn_opu/task/",
})


def test_every_allowed_grant_resolves_to_something_that_exists():
    """The assembler's own precondition: a grant it cannot place is a tool the agent will not find."""
    missing: list[str] = []
    n = 0
    for man in _bundles():
        doc = yaml.safe_load(man.read_text(encoding="utf-8")) or {}
        for entry in (doc.get("allowed") or []):
            rel = str((entry or {}).get("path") or "")
            if not rel:
                continue
            n += 1
            if rel in _KNOWN_UNRESOLVABLE:
                continue
            if not resolve_grant(rel, repo_root()).exists():
                missing.append(f"{man.parent.name}: {rel}")
    assert not missing, (
        f"{len(missing)} granted path(s) resolve to nothing, so the workspace cannot place them and the "
        f"agent cannot find the tool it was told to use: {sorted(set(missing))[:5]}")
    assert n > 0, "no grants were checked; this assertion would pass vacuously"


def test_the_assembler_resolves_through_the_shared_resolver():
    """Pins the FIX, not just the outcome.

    `resolve_grant`'s own docstring says every consumer of a manifest must go through it, and names two
    earlier consumers that resolved `<repo>/<rel>` only and silently skipped what they could not find.
    The workspace assembler was a third. Recomputing the path here is what makes a registry-redirected
    grant vanish, so assert the assembler asks the resolver rather than rebuilding the path itself.
    """
    src = (_HARNESS / "run_baseline_qa_loop.py").read_text(encoding="utf-8")
    body = src[src.index("def assemble_copy_workspace"):]
    body = body[:body.index("\ndef ")] if "\ndef " in body[10:] else body
    assert "resolve_grant" in body, (
        "assemble_copy_workspace no longer resolves grants through the shared resolver; a grant whose "
        "home the target registry owns will silently not be placed in the workspace")
    assert 'src = C.REPO / entry["path"]' not in body, (
        "the assembler is recomputing <repo>/<declared path> again, which drops registry-redirected "
        "grants without a warning")


def test_a_grant_that_cannot_be_placed_is_reported_not_dropped():
    """Silence is what made this cost a round: a missing tool reads as a tool that does not exist."""
    src = (_HARNESS / "run_baseline_qa_loop.py").read_text(encoding="utf-8")
    assert "unresolvable_grants" in src, (
        "the workspace report no longer records grants it could not place, so a dropped tool is "
        "indistinguishable from a tool the arm was never granted")
