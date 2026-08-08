"""What the packager may and may not call "the same program".

A board variant we cannot simulate is not therefore unknown: if it runs the same instruction sequence as
a variant that WAS gated, the gate covers its arithmetic. That claim is only worth anything if the
comparison is strict about what it forgives, so this pins both directions -- what counts as the same
program, and what must be reported as a mismatch.
"""
from __future__ import annotations

import importlib.util
import sys

from merlin.common.paths import repo_root


def _load_packager():
    """Load the delivery packager (a build_tools script, not an installed module) by path.

    Registered in ``sys.modules`` before ``exec_module``: the script declares dataclasses under
    postponed annotations, and skipping the registration makes their construction fail with a bare
    ``AttributeError: 'NoneType'``.
    """
    p = repo_root() / "build_tools" / "scripts" / "make_delivery.py"
    spec = importlib.util.spec_from_file_location("make_delivery", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_an_identical_stream_is_the_same_program_with_nothing_relaxed():
    md = _load_packager()
    s = ["addi", "vle32.v", "vfmacc.vv", "vse32.v", "ret"]
    assert md.twin_equivalence(s, list(s)) == (True, 0)


def test_a_relaxed_address_materialisation_is_still_the_same_program():
    """The measured case. Between the 50 MHz and 500 MHz gemmelos sets, `spectformer_h1` differed by
    exactly three deleted `auipc` and `whisper_h2` by one inserted `auipc`, both in the last 0.2 % of
    the stream with every other instruction identical. `auipc` materialises a PC-relative address, and a
    shifted layout that brings a symbol into range of a shorter form collapses the pair. Calling that a
    mismatch reported five images as unverified when their arithmetic was known good."""
    md = _load_packager()
    base = ["addi", "vfmacc.vv", "auipc", "addi", "ld", "vse32.v", "ret"]
    dropped = ["addi", "vfmacc.vv", "addi", "ld", "vse32.v", "ret"]
    assert md.twin_equivalence(base, dropped) == (True, 1)
    assert md.twin_equivalence(dropped, base) == (True, 1)          # symmetric


def test_any_other_difference_is_a_mismatch():
    """The tolerance has to stay narrow or the check stops meaning anything. A REPLACE is a mismatch
    even when an `auipc` is involved, and an insert/delete carrying anything besides an address
    materialisation is a real difference in the program."""
    md = _load_packager()
    base = ["addi", "vfmacc.vv", "auipc", "vse32.v"]
    assert md.twin_equivalence(base, ["addi", "vfmacc.vv", "lui", "vse32.v"]) == (False, 0)
    # a dropped arithmetic instruction is never forgivable
    assert md.twin_equivalence(base, ["addi", "auipc", "vse32.v"]) == (False, 0)
    # nor is an insert that mixes an auipc with real work
    assert md.twin_equivalence(base, ["addi", "vfmacc.vv", "auipc", "vfmacc.vv", "vse32.v"])[0] is False
    # a different vector op at the same position is the case this exists to catch
    assert md.twin_equivalence(["vfmacc.vv"], ["vfmacc.vf"]) == (False, 0)


def test_the_relaxable_set_is_deliberately_tiny():
    """If this set grows, the twin claim weakens for every package that has ever used it, so growing it
    should be a decision someone makes on purpose rather than a drive-by."""
    md = _load_packager()
    assert md.RELAXABLE_MNEMONICS == frozenset({"auipc"})
