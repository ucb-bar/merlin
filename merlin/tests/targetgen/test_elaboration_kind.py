"""Why a target's port facts are unavailable, told apart from each other.

`explicit_completion` — the trait PS needs, and PC needed — is derived by reading module ports out of a
target's own elaboration. When that read fails the reason decides the repair, and one message covered
three different situations:

* the facts name a `.fir` and the file is not on this host → find the file;
* the facts name no `.fir` at all because the elaboration is recorded as CIRCT **hw dialect**, and the
  named `hw.mlir` EXISTS → teach the reader that dialect;
* the facts name no elaboration artifact of either kind → re-extract.

Measured: atlas's facts point at `atlas_hw.mlir`, 4.3 MB and present, and the message said "the
elaborated FIRRTL this target's facts name could not be located" — sending a reader to look for a file
that was never named. gemmini's facts DO name a `.fir`, which is why its ports derive and atlas's do
not, and that difference is the whole story.

Two fact-bundle shapes are read rather than one assumed, because both occur here: `facts.source` is a
mapping with a `fir` key on one target and a bare string on another, and `facts.interfaces` is a
mapping on one and a list of records on another. Assuming either turns a target with a recorded
elaboration into one reporting none — the direction that hides a fixable gap.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.targetgen.rtl.ports import elaboration_kind, port_facts

_TARGETS = ("gemmini", "atlas", "radiance", "mx_gemmini")


def test_every_target_resolves_to_a_named_kind_without_raising():
    """Shape robustness. An AttributeError here would report a good elaboration as absent."""
    for t in _TARGETS:
        kind, detail = elaboration_kind(t)
        assert kind in ("fir", "hw_mlir", "none"), f"{t}: unexpected kind {kind!r}"
        assert detail, f"{t}: a kind with no detail explains nothing"


def test_the_target_whose_facts_name_a_fir_derives_its_ports():
    """gemmini is the control: its facts name a .fir, so the read succeeds and the trait is decidable."""
    kind, named = elaboration_kind("gemmini")
    assert kind == "fir" and named.endswith(".fir")
    pf = port_facts("gemmini", fields=("completed",))
    assert pf.get("status") == "derived"
    decoupled = ((pf.get("fields") or {}).get("completed") or {}).get("decoupled") or ()
    assert decoupled, "gemmini's completion channels no longer read as decoupled"


def test_a_hw_dialect_elaboration_says_so_instead_of_could_not_be_located():
    """⚠️ The actionable half: the artifact EXISTS, and the reader is what does not read it.

    So the message must not claim the file is missing. If this target's kind changes to `fir`, or the
    reader gains the hw dialect, `explicit_completion` becomes derivable there and PS's remaining gap
    closes with it.
    """
    kind, detail = elaboration_kind("atlas")
    if kind != "hw_mlir":
        pytest.skip(f"atlas's elaboration is now recorded as {kind!r}; re-read this test")
    assert "hw dialect" in detail and "parses .fir only" in detail
    why = str(port_facts("atlas", fields=("completed",)).get("why") or "")
    assert "could not be located" not in why, (
        "the message claims the named file is missing; atlas's facts name no .fir at all, and the "
        "hw.mlir they DO name is present on this host")
    # The named artifact is real. Checked so a change in the fact bundle cannot make this test pass
    # while describing something absent.
    named = [w for w in detail.replace("(", " ").replace(")", " ").replace(",", " ").split()
             if w.endswith(".mlir") and w.startswith("/")]
    if named:
        assert Path(named[0]).is_file(), f"{named[0]} is named but absent; the diagnosis changed"


def test_no_named_artifact_is_a_third_state_not_folded_into_the_others():
    """radiance names neither kind. "Re-extract" is a different repair from "teach the reader"."""
    kind, detail = elaboration_kind("radiance")
    if kind != "none":
        pytest.skip(f"radiance now records a {kind!r} elaboration")
    assert "no elaboration artifact" in detail
    why = str(port_facts("radiance", fields=("completed",)).get("why") or "")
    assert "hw dialect" not in why, "a target naming nothing must not be reported as a dialect gap"
