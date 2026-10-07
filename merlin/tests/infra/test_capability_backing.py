"""A narrowing capability claim must cite something, and the citation must still resolve.

These tests are the mutation half of ``build_tools/scripts/check_capability_backing.py``. The gate
itself can be green because every claim is evidenced OR because the reader stopped finding claims, and
those two states are indistinguishable from the exit code -- which is exactly the failure mode the
defect it was written for had. So each test below RE-INTRODUCES a defect in a contract it wrote itself
and asserts the verdict changes, rather than asserting that today's tree passes.

The defect: one target declared ``{family: elementwise_map, composed_with: [contraction]}`` on two
lines of prose. The RTL said the opposite in both directions -- the device runs the STANDALONE add and
cannot fold one into a producing contraction's accumulator -- and because the declaration narrowed, the
consequence was silence: no capsule demanded the work, so nothing failed, so the compiler shipped
refusing 16 of ResNet-50's 71 device groups.
"""

from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import capability_backing as CB

#: A contract document with one compute unit and one family, parameterised on the family entry and the
#: comment above it. Deliberately minimal: what is under test is the reader and the backing rule, not
#: any real target's declarations.
_DOC = """\
name: {name}
compute_units:
  - name: unit0
    dtypes: [int8, int16]
    ops: [matmul]
    semantic_capabilities:
{comment}      - {entry}
"""


def _doc(entry: str, comment_lines: tuple[str, ...] = (), *, name: str = "t") -> str:
    comment = "".join(f"      # {line}\n" for line in comment_lines)
    return _DOC.format(name=name, comment=comment, entry=entry)


def _claims(entry: str, comment=(), *, pins=None):
    return CB.claims_in_document(
        _doc(entry, comment),
        target="t",
        source="<test>/contract.yaml",
        pins=pins or {},
    )


# ---------------------------------------------------------------------------------------------------
# THE MUTATION: the defect itself, re-introduced
# ---------------------------------------------------------------------------------------------------


def test_the_defect_shape_is_reported_as_unbacked() -> None:
    """The exact line and the exact justification that cost a compiler generation."""
    got = _claims(
        "{family: elementwise_map, dtypes: [int8, int16], composed_with: [contraction]}",
        (
            "Epilogue-only: per-channel scale + requant apply on the way out of the accumulator, so an",
            "elementwise map exists ONLY fused behind a contraction.",
        ),
    )
    claim = next(c for c in got if c.shape == "composed_with")
    assert claim.detail == "elementwise_map"
    assert not claim.backed
    assert claim.citations == ()
    assert "no citation" in claim.why()


def test_the_same_line_passes_once_it_cites_an_rtl_source_the_pin_reads() -> None:
    """...and the fix is a citation, not a rewording. Same restriction, same prose, plus the file the
    verdict was read out of -- and that file is in the read set of a pin this target declares."""
    got = _claims(
        "{family: elementwise_map, dtypes: [int8, int16], composed_with: [contraction]}",
        ("Read off `TheUnroller.scala`'s accumulate literals, not asserted.",),
        pins={"a_pin": ("src/main/scala/x/TheUnroller.scala",)},
    )
    claim = next(c for c in got if c.shape == "composed_with")
    assert claim.backed
    assert [(c.token, c.kind) for c in claim.citations] == [("TheUnroller.scala", "rtl_pin")]


def test_an_rtl_citation_outside_every_declared_pin_read_set_does_not_back_the_claim() -> None:
    """THE HOLE THAT BIT HARDEST, and the reason this is not just "does the comment name a file".

    A claim may cite exactly the right Chisel file while the pin that names the checkout never READS
    it. `verify` then reports the pin clean over an edit that changed the verdict. Both systolic pins
    in this repo were in that state when this gate was written.
    """
    got = _claims(
        "{family: elementwise_map, dtypes: [int8, int16], composed_with: [contraction]}",
        ("Read off `TheUnroller.scala`'s accumulate literals.",),
        pins={"a_pin": ("src/main/scala/x/SomethingElse.scala",)},
    )
    claim = next(c for c in got if c.shape == "composed_with")
    assert not claim.backed
    assert [c.kind for c in claim.citations] == ["unpinned_rtl"]
    assert "add it to that pin's read set" in claim.citations[0].detail


def test_a_citation_that_no_longer_resolves_is_reported_rather_than_ignored() -> None:
    """A citation cannot rot into decoration. A test that moved is a claim nobody re-checked."""
    got = _claims(
        "{family: elementwise_map, dtypes: [int8, int16], composed_with: [contraction]}",
        ("Pinned by merlin/tests/targetgen/test_a_file_that_does_not_exist.py.",),
    )
    claim = next(c for c in got if c.shape == "composed_with")
    assert not claim.backed
    assert [c.kind for c in claim.citations] == ["unresolved_repo_path"]


def test_a_backed_claim_with_one_broken_citation_is_still_reported() -> None:
    """Backed is not a pass for the citation that broke: `rotted` is its own axis in the ledger."""
    doc = CB.audit_document(
        _doc(
            "{family: elementwise_map, dtypes: [int8, int16], composed_with: [contraction]}",
            (
                "Derived by merlin.targetgen.eligibility and pinned by",
                "merlin/tests/targetgen/test_a_file_that_does_not_exist.py.",
            ),
        ),
        target="t",
        source="<test>/contract.yaml",
    )
    assert doc["unbacked"] == []
    assert doc["rotted"] == ["t rotted:composed_with:elementwise_map"]
    assert any("ROTTED" in p for p in doc["problems"])


# ---------------------------------------------------------------------------------------------------
# the reader itself -- a claim it cannot see is a claim nothing will ever ask about
# ---------------------------------------------------------------------------------------------------


def test_a_citation_wrapped_in_backticks_and_inflected_is_still_found() -> None:
    """A real regression: prose writes ``` `StoreController.scala`'s ``` and an end-strip tokenizer
    leaves the backtick embedded, so the citation is invisible and the claim reads as bare prose.
    A too-narrow tokenizer that drops a valid spelling is the failure this repo bans regex for."""
    for spelling in (
        "`TheUnroller.scala`'s accumulate literal",
        "(TheUnroller.scala),",
        "see src/main/scala/x/TheUnroller.scala.",
        "[TheUnroller.scala]",
    ):
        got = _claims(
            "{family: elementwise_map, dtypes: [int8, int16], composed_with: [contraction]}",
            (spelling,),
            pins={"a_pin": ("src/main/scala/x/TheUnroller.scala",)},
        )
        claim = next(c for c in got if c.shape == "composed_with")
        assert claim.backed, f"citation invisible when spelled {spelling!r}"


def test_a_blank_line_ends_the_comment_block() -> None:
    """Prose about the SECTION is not evidence for the line. Without this, any restriction inherits
    whatever citation happens to appear earlier in the file."""
    text = (
        "name: t\ncompute_units:\n  - name: unit0\n    dtypes: [int8, int16]\n"
        "    semantic_capabilities:\n"
        "      # Derived by merlin.targetgen.eligibility.\n"
        "\n"
        "      - {family: elementwise_map, dtypes: [int8], composed_with: [contraction]}\n"
    )
    got = CB.claims_in_document(text, target="t", source="<test>/c.yaml")
    claim = next(c for c in got if c.shape == "composed_with")
    assert not claim.backed


def test_an_empty_composed_with_is_not_a_restriction() -> None:
    """An empty list is the REFUSAL of a restriction -- how the fixed contract says "this family
    composes with nothing, and the fused form is refused by being declared nowhere". Counting it as a
    narrowing would demand evidence for the absence of a claim."""
    got = _claims("{family: elementwise_map, dtypes: [int8, int16], composed_with: []}")
    assert [c.shape for c in got] == []


def test_a_unit_with_no_semantic_capabilities_is_the_most_total_narrowing_there_is() -> None:
    """Zero declared families means every region routed at the unit is refused `undeclared_family`,
    which reads in every aggregate as work the hardware cannot do. It has a line to attach to -- the
    unit's own -- so it is a claim, not an absence this gate has to let pass."""
    text = "name: t\ncompute_units:\n  - name: unit0\n    dtypes: [int8]\n    ops: [matmul]\n"
    got = CB.claims_in_document(text, target="t", source="<test>/c.yaml")
    assert [(c.shape, c.detail) for c in got] == [("no_semantic_capabilities", "unit0")]


def test_a_family_narrower_than_its_own_unit_is_a_claim_and_an_equal_one_is_not() -> None:
    narrower = _claims("{family: contraction, dtypes: [int8]}")
    assert [c.shape for c in narrower] == ["family_dtypes"]
    equal = _claims("{family: contraction, dtypes: [int8, int16]}")
    assert [c.shape for c in equal] == []


def test_an_unmaterializable_entry_cites_through_its_own_prose() -> None:
    """Its reason lives in the VALUE, not in a `#` line above the key. A reader that looked only at
    comments reported "no citation at all" on entries that name a source file by line range."""
    text = (
        "name: t\ncompute_units: []\n"
        "unmaterializable_families:\n"
        "  elementwise_map: >-\n"
        "    no probe path: see src/merlin/targetgen/eligibility.py for the predicate.\n"
    )
    got = CB.claims_in_document(text, target="t", source="<test>/c.yaml")
    assert [(c.shape, c.detail, c.backed) for c in got] == [("unmaterializable", "elementwise_map", True)]


def test_an_unreadable_contract_raises_rather_than_reporting_no_claims() -> None:
    """Fail closed. "Zero claims" and "I could not read it" must never be the same answer -- that
    equivalence is how a narrowing goes unasked about."""
    with pytest.raises(CB.BackingError):
        CB.claims_in_document("name: [unclosed\n", target="t", source="<test>/c.yaml")


# ---------------------------------------------------------------------------------------------------
# the ledger, held in both directions
# ---------------------------------------------------------------------------------------------------


def _ledger_path():
    return repo_root() / "build_tools" / "scripts" / "capability_backing_ratchet.txt"


def _ledger() -> set[str]:
    path = _ledger_path()
    if not path.is_file():
        return set()
    return {
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }


def test_the_gate_is_green_on_this_tree() -> None:
    """Every narrowing claim is either evidenced or ratcheted, and no ratchet line is stale.

    Both halves. A ledger held in one direction only grows entries that outlive their gaps and becomes
    a permanent allowlist -- the gap closes, nobody notices, and the line forgives the next regression
    of the same claim.
    """
    ledger = _ledger()
    problems, stale = [], []
    for target in CB.gated_targets():
        doc = CB.audit(target, ratchet=ledger)
        problems += doc["problems"]
        stale += doc["stale_ratchet_entries"]
    assert not problems, "\n".join(problems)
    assert not stale, f"ledger lines that are no longer gaps: {stale}"


def test_every_ledger_line_names_a_real_target_and_a_declared_shape() -> None:
    """A ledger entry that names nothing cannot be closed, and cannot be shown to be stale either."""
    targets = set(CB.gated_targets())
    for entry in _ledger():
        target, _, rest = entry.partition(" ")
        axis, _, claim_key = rest.partition(":")
        shape = claim_key.partition(":")[0]
        assert target in targets, f"{entry}: no such target"
        assert axis in ("unbacked", "rotted"), f"{entry}: unknown axis {axis!r}"
        assert shape in CB.NARROWING_SHAPES, f"{entry}: unknown shape {shape!r}"


def test_the_scope_of_the_gate_is_declared_rather_than_implied() -> None:
    """A narrowing shape this gate does not decide has to be NAMED, or a green run reads as "every
    narrowing in this repo is evidenced" -- which is a check that cannot fail."""
    assert CB.DEFERRED_SHAPES, "deferred shapes must be enumerated, not left implicit"
    assert not (set(CB.DEFERRED_SHAPES) & set(CB.NARROWING_SHAPES))
    assert all(why.strip() for why in CB.DEFERRED_SHAPES.values())
    assert all(why.strip() for why in CB.NARROWING_SHAPES.values())


def test_a_target_that_ships_only_a_residual_is_still_surveyed() -> None:
    """The registry's target list needs a `target_contract.yaml`. A residual-only target declares
    capabilities the deriver reads all the same, and the one in this tree carries the narrowest rank
    set and two `unmaterializable_families` entries."""
    from merlin.targetgen import target_registry as tr

    assert set(tr.all_targets()) <= set(CB.gated_targets())
    residual_only = set(CB.gated_targets()) - set(tr.all_targets())
    for target in residual_only:
        assert CB.contract_documents(target), f"{target} was surveyed but has no contract document"


def test_the_cli_agrees_with_the_library_and_reports_the_scope() -> None:
    """The script is what pre-commit and CI run, so its exit code is part of the property. Its
    `--shapes` mode prints both tables, which is how a reader finds out what the green run means.

    The first assertion is that the two halves are looking at the SAME checkout, because they resolve
    it differently and nothing used to notice when they diverged. The script seats its own tree's
    `src` at the head of `sys.path`, so it always audits the clone it was launched from.
    This test imports `merlin` however the interpreter resolves it -- and in a git worktree sharing a
    venv with the clone it was branched from, the editable install points at the OTHER clone. Both
    halves then run green about two different trees, and the exit code cannot tell you which one the
    contracts came from. Comparing the roots turns that into a named environment failure.
    """
    import subprocess
    import sys

    script = repo_root() / "build_tools" / "scripts" / "check_capability_backing.py"
    tree = subprocess.run([sys.executable, str(script), "--tree"], capture_output=True, text=True, timeout=300)
    assert tree.returncode == 0, tree.stdout + tree.stderr
    assert tree.stdout.strip() == str(repo_root()), (
        f"the gate script audits {tree.stdout.strip()!r} but this test imported merlin from "
        f"{repo_root()!r}; they would both report green about different checkouts. Set "
        f"PYTHONPATH=<this tree>/src so the library half reads the tree under test."
    )
    got = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert got.returncode == 0, got.stdout + got.stderr
    shapes = subprocess.run([sys.executable, str(script), "--shapes"], capture_output=True, text=True, timeout=300)
    assert shapes.returncode == 0
    for key in list(CB.NARROWING_SHAPES) + list(CB.DEFERRED_SHAPES):
        assert key in shapes.stdout, f"{key} missing from --shapes"
    # And with the ledger ignored it FAILS, which is what proves the ledger is load-bearing rather
    # than an empty file the gate happens to read.
    clean = subprocess.run([sys.executable, str(script), "--no-ratchet"], capture_output=True, text=True, timeout=300)
    assert clean.returncode == 1, "the ratchet forgives nothing, so it is not holding anything back"


def test_a_dotted_module_citation_is_not_truncated_to_whatever_package_exists() -> None:
    """A real regression in this reader. It walked UP the dotted path until something imported, so
    `merlin.compile.linalg_lower` -- which does not exist; the module is `merlin.targetgen.
    linalg_lower` -- read as a citation because `merlin.compile` does. Any dotted token starting with
    a real package then backed a claim, and a wrong module name could not fail."""
    bad = _claims(
        "{family: elementwise_map, dtypes: [int8, int16], composed_with: [contraction]}",
        ("Derived by merlin.compile.linalg_lower.",),
    )
    claim = next(c for c in bad if c.shape == "composed_with")
    assert not claim.backed
    assert [c.kind for c in claim.citations] == ["missing_module"]
    # One trailing segment is still allowed off, because a comment names a function as often as a
    # module -- but only down to a module, never down to a bare package.
    good = _claims(
        "{family: elementwise_map, dtypes: [int8, int16], composed_with: [contraction]}",
        ("Derived by merlin.targetgen.eligibility.is_eligible.",),
    )
    claim = next(c for c in good if c.shape == "composed_with")
    assert claim.backed
    assert claim.citations[0].detail == "merlin.targetgen.eligibility"


# ---------------------------------------------------------------------------------------------------
# the gate script itself, in process -- what the hook and CI run
# ---------------------------------------------------------------------------------------------------


def _gate_script():
    import importlib.util
    import sys

    path = repo_root() / "build_tools" / "scripts" / "check_capability_backing.py"
    spec = importlib.util.spec_from_file_location("check_capability_backing", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_gate_fails_on_a_reintroduced_unbacked_narrowing(tmp_path, monkeypatch) -> None:
    """THE GATE'S OWN FAILING DIRECTION, driven through its ``main``.

    One synthetic target whose only contract document carries the defect line with prose and no
    citation. Nothing else can satisfy the gate: the ledger is ignored, the target declares no pin, and
    the comment names no file, test or module -- so a pass could only mean the gate stopped reading
    claims. The same document with an in-repo citation passes, so a gate that failed unconditionally
    would not satisfy this test either.
    """
    gate = _gate_script()
    contract = tmp_path / "contract.yaml"
    monkeypatch.setattr(CB, "gated_targets", lambda: ("t",))
    monkeypatch.setattr(CB, "contract_documents", lambda target: ((contract, contract.read_text()),))
    monkeypatch.setattr(CB, "_pin_read_paths", lambda target: {})
    entry = "{family: elementwise_map, dtypes: [int8, int16], composed_with: [contraction]}"

    contract.write_text(_doc(entry, ("An elementwise map exists ONLY fused behind a contraction.",)))
    assert gate.main(["--no-ratchet"]) == 1

    contract.write_text(_doc(entry, ("Derived by merlin.targetgen.eligibility.is_eligible.",)))
    assert gate.main(["--no-ratchet"]) == 0
