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

⚠️ WHAT IT WOULD TAKE TO READ THE hw DIALECT, measured so nobody re-derives it. xdsl DOES ship `hw`,
`comb` and `seq`, so a structural parse looked like the answer — and it is not: parsing
`atlas_hw.mlir` fails in 0.1 s with "Operation builtin.unregistered does not have a custom format",
because the file also uses CIRCT dialects xdsl has no definitions for and an unregistered op in custom
assembly cannot be parsed. The repo's own hw-dialect reader (`rtl.extract_module`) walks module
DECLARATIONS and INSTANCES, not port lists. So a port reader for this dialect is new extractor work —
a hand tokenizer over `hw.module @Name(in %a : i1, ...)` signatures — in the same area whose docstring
already records a port pattern silently skipping what it could not read.

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


def test_a_hw_dialect_elaboration_is_read_rather_than_reported_unknown():
    """The close: atlas's ports now DERIVE from the artifact its facts always named.

    The file was there the whole time -- 4.5 MB of CIRCT hw dialect -- and the reader parsed `.fir`
    only, so the trait that needs those ports reported UNKNOWN against it. Both dialects are read now,
    and `dialect` travels in the record because they are not interchangeable evidence: hw-dialect
    bundles are flattened, so a field's leaves are reconstructed from name segments.
    """
    kind, detail = elaboration_kind("atlas")
    if kind != "hw_mlir":
        pytest.skip(f"atlas's elaboration is now recorded as {kind!r}; re-read this test")
    named = [w.strip() for w in detail.split(",") if w.strip().startswith("/")]
    assert named and Path(named[0]).is_file(), (
        f"the named artifact {named[:1]} is not on this host; the diagnosis changed")

    pf = port_facts("atlas", fields=("completed", "busy"))
    assert pf["status"] == "derived", f"atlas's ports should read now: {pf.get('why')}"
    assert pf["dialect"] == "hw"
    assert pf["n_modules"] > 0


def test_the_hw_reader_refuses_an_incomplete_signature_rather_than_answering():
    """A module whose port list cannot be read leaves the field list incomplete.

    Answering from an incomplete list is how "no completion port" gets concluded from a parse gap --
    the failure this file's own history records. So an unreadable signature makes the whole read
    UNAVAILABLE, and the modules it could not read are named.
    """
    from merlin.targetgen.rtl.ports import hw_module_ports

    ports, bad = hw_module_ports(
        "hw.module @Good(in %clock : i1, out done : i1) {\n}\n"
        "hw.module @Truncated(in %clock : i1\n")
    assert "Good" in ports and "Truncated" in bad
    assert "matching paren" in bad["Truncated"]

    # And a signature it CAN read regroups the flattened bundle back into a field with leaves.
    ports, bad = hw_module_ports(
        "hw.module @Ctrl(in %completed_ready : i1, out completed_valid : i1, "
        "out completed_bits : i6) {\n}\n")
    assert not bad
    f = ports["Ctrl"].field_named("completed")
    assert f is not None and set(f.leaves) == {"ready", "valid", "bits"}
    assert f.is_decoupled(), "ready+valid must still read as a handshake once regrouped"


def test_no_named_artifact_is_a_third_state_not_folded_into_the_others():
    """radiance names neither kind. "Re-extract" is a different repair from "teach the reader"."""
    kind, detail = elaboration_kind("radiance")
    if kind != "none":
        pytest.skip(f"radiance now records a {kind!r} elaboration")
    assert "no elaboration artifact" in detail
    why = str(port_facts("radiance", fields=("completed",)).get("why") or "")
    assert "hw dialect" not in why, "a target naming nothing must not be reported as a dialect gap"
