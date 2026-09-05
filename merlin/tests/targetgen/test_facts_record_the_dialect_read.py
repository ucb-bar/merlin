"""RTL facts must record the HW dialect they actually read.

The regression this guards: `inputs` recorded only the SoC dialect (`_soc_hw_path`), which feeds the
legacy accumulator port-parse and is absent for most targets. So the facts named a file that was not
on disk -- `hw_sha: "missing"` -- while saying nothing about the CORE dialect that mlc discovery and
the pipeline-depth walk had genuinely consumed. A wrong name is worse than a blank field: it reads as
provenance. Until this field is real, a performance term cannot state which elaboration it holds for,
so its validity domain is asserted rather than evidenced.
"""
from __future__ import annotations

import pytest

from merlin.common import provenance
from merlin.targetgen.rtl import circt_introspect, mlc_bridge

_SENTINELS = {"unresolved", "missing"}


def _targets_with_a_core_dialect() -> list[str]:
    return [t for t in ("atlas", "gemmini") if mlc_bridge.core_hw_mlir(t) is not None]


def test_the_recorded_core_dialect_is_the_one_mlc_resolves() -> None:
    """The recorded name must be the file mlc hands back -- not a path built by convention."""
    targets = _targets_with_a_core_dialect()
    if not targets:
        pytest.skip("no target resolves a core HW dialect here (mlc unavailable)")
    for target in targets:
        rec = circt_introspect._core_hw_input(target)
        resolved = mlc_bridge.core_hw_mlir(target)
        assert rec["core_hw_mlir"] == resolved.name
        assert rec["core_hw_sha"] not in _SENTINELS, f"{target}: resolved dialect but no digest"
        assert len(rec["core_hw_sha"]) == 16
        assert len(rec["core_hw_sha256"]) == 64


def test_it_is_target_agnostic() -> None:
    """Two targets of different archetypes must both resolve, to distinct dialects.

    A single-target pass would not distinguish "derives the dialect" from "knows one path"."""
    targets = _targets_with_a_core_dialect()
    if len(targets) < 2:
        pytest.skip("need two targets resolving a core HW dialect")
    recs = {t: circt_introspect._core_hw_input(t) for t in targets}
    names = {r["core_hw_mlir"] for r in recs.values()}
    shas = {r["core_hw_sha"] for r in recs.values()}
    assert len(names) == len(targets), f"targets share a dialect name: {recs}"
    assert len(shas) == len(targets), f"targets share a digest: {recs}"


def test_an_unresolvable_dialect_is_named_not_faked() -> None:
    """No dialect must yield the `unresolved` sentinel -- never a digest, never an empty string.

    The distinction between "mlc resolved nothing" and "the file is resolved but absent" is load
    bearing: only the second is fixable by rebuilding, and collapsing them onto one value loses that."""
    rec = circt_introspect._core_hw_input("a-target-that-does-not-exist")
    assert rec == {"core_hw_mlir": "unresolved", "core_hw_sha": "unresolved",
                   "core_hw_sha256": "unresolved"}


def test_the_recorded_digest_agrees_with_the_pinned_artifact() -> None:
    """The facts' digest must be a prefix of the pinned artifact's, or the two disagree about identity.

    This is the join that makes a validity domain evidenced: the pin says which elaboration exists, the
    facts say which one was read, and a result is citable only when they are the same file."""
    if mlc_bridge.core_hw_mlir("atlas") is None:
        pytest.skip("atlas core dialect not resolvable here")
    check = provenance.verify_artifact("atlas_core_hw_dialect")
    if not check.present:
        pytest.skip("pinned dialect artifact absent on this host")
    recorded = circt_introspect._core_hw_input("atlas")["core_hw_sha"]
    assert check.digest.startswith(recorded), (
        f"facts recorded {recorded!r} but the pin is {check.digest!r} -- "
        "the facts and the pin name different files")
    recorded_full = circt_introspect._core_hw_input("atlas")["core_hw_sha256"]
    assert check.digest == recorded_full
