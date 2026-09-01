"""The CIRCT hw.module slicer reads the instance graph structurally, and fails closed.

`rtl/extract_module.py` cuts a self-contained `hw.module` subtree out of an elaborated SoC HW MLIR.
It used to find module declarations and instance edges with two regexes, and both were silent-drop
hazards whose symptom is not an error but a SMALLER accelerator: a module missing from the span
table has its body swallowed into the preceding module's span, and a missed `hw.instance` target is
never pulled into the closure, so the emitted slice quietly references an undeclared module.

The `sym`-form case below is a REAL regression from the local CIRCT corpus, not a hypothetical:
`hw.instance "verification_assert" sym @verification_assert @FP8Unpack_Verification_Assert()` was
dropped by the retired `hw\\.instance\\s+\\S+\\s+@(\\w+)` pattern, and the extracted `@FP8Unpack`
slice therefore instantiated a module it did not carry -- while reporting zero unresolved refs.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.rtl import extract_module as em


def _soc(*module_blocks: str) -> str:
    return "module attributes {circt.loweringOptions = \"x\"} {\n" + "".join(module_blocks) + "}\n"


_LEAF = "  hw.module private @Leaf(in %a : i1, out b : i1) {\n    hw.output %a : i1\n  }\n"


def test_extracts_the_transitive_closure_and_reports_unresolved_refs():
    src = _soc(
        "  hw.module @Top(in %a : i1, out b : i1) {\n"
        "    %l.b = hw.instance \"l\" @Leaf(a: %a: i1) -> (b: i1)\n"
        "    %e.b = hw.instance \"e\" @Elsewhere(a: %a: i1) -> (b: i1)\n"
        "    hw.output %l.b : i1\n  }\n",
        _LEAF,
        "  hw.module @Unrelated(in %a : i1, out b : i1) {\n    hw.output %a : i1\n  }\n",
    )
    out, included, missing = em.extract(src, "Top")
    assert set(included) == {"Top", "Leaf"}
    assert missing == ["Elsewhere"]
    assert "@Unrelated" not in out
    assert out.startswith("module {\n") and "hw.module private @Leaf" in out


def test_inner_sym_instance_target_is_followed():
    """REGRESSION (real corpus): `hw.instance "n" sym @s @Module(...)`. The retired pattern's
    `\\s+\\S+\\s+@` shape could not step over the inner symbol, so the target was silently absent
    from the closure and from `missing` alike."""
    src = _soc(
        "  hw.module @Top(in %a : i1, out b : i1) {\n"
        "    hw.instance \"verification_assert\" sym @verification_assert @Leaf(a: %a: i1) "
        "-> (b: i1) {doNotPrint}\n"
        "    hw.output %a : i1\n  }\n",
        _LEAF,
    )
    _out, included, missing = em.extract(src, "Top")
    assert set(included) == {"Top", "Leaf"} and missing == []


def test_quoted_instance_name_containing_a_space_is_followed():
    """`\\S+` for the instance name could not cross a space inside the quoted name."""
    src = _soc(
        "  hw.module @Top(in %a : i1, out b : i1) {\n"
        "    hw.instance \"my inst\" @Leaf(a: %a: i1) -> (b: i1)\n"
        "    hw.output %a : i1\n  }\n",
        _LEAF,
    )
    _out, included, missing = em.extract(src, "Top")
    assert set(included) == {"Top", "Leaf"} and missing == []


def test_symbol_with_dollar_or_dot_is_not_truncated():
    """`[A-Za-z0-9_]+` truncated a legal bare symbol, so the DEFINITION and the REFERENCE ended up
    with different names and a present module was reported unresolved."""
    src = _soc(
        "  hw.module @Top(in %a : i1, out b : i1) {\n"
        "    hw.instance \"l\" @Leaf$impl.v2(a: %a: i1) -> (b: i1)\n"
        "    hw.output %a : i1\n  }\n",
        "  hw.module private @Leaf$impl.v2(in %a : i1, out b : i1) {\n    hw.output %a : i1\n  }\n",
    )
    _out, included, missing = em.extract(src, "Top")
    assert set(included) == {"Top", "Leaf$impl.v2"} and missing == []


def test_extern_and_generated_declarations_are_indexed():
    src = _soc(
        "  hw.module @Top(in %a : i1, out b : i1) {\n"
        "    %m.b = hw.instance \"m\" @Blackbox(a: %a: i1) -> (b: i1)\n"
        "    hw.output %m.b : i1\n  }\n",
        "  hw.module.extern @Blackbox(in %a : i1, out b : i1)\n",
    )
    _out, included, missing = em.extract(src, "Top")
    assert set(included) == {"Top", "Blackbox"} and missing == []
    assert em._module_spans(src)["Blackbox"][2] is True     # flagged body-less


def test_unreadable_module_declaration_raises():
    """A `hw.module` line with no readable symbol used to be skipped, which merged its body into
    the PREVIOUS module's span -- corrupting the slice without a word."""
    src = _soc("  hw.module private (in %a : i1)\n", _LEAF)
    with pytest.raises(em.ExtractModuleError) as exc:
        em.extract(src, "Leaf")
    assert "hw.module" in str(exc.value)


def test_unreadable_instance_raises():
    src = _soc(
        "  hw.module @Top(in %a : i1, out b : i1) {\n"
        "    hw.instance \"l\" (a: %a: i1) -> (b: i1)\n"
        "    hw.output %a : i1\n  }\n",
        _LEAF,
    )
    with pytest.raises(em.ExtractModuleError) as exc:
        em.extract(src, "Top")
    assert "hw.instance" in str(exc.value)


def test_missing_root_still_raises_keyerror():
    with pytest.raises(KeyError):
        em.extract(_soc(_LEAF), "Nope")


def test_included_order_is_reproducible():
    """`included` used to vary run-to-run with set iteration order; the emitted text never did."""
    src = _soc(
        "  hw.module @Top(in %a : i1, out b : i1) {\n"
        "    %x.b = hw.instance \"x\" @Leaf(a: %a: i1) -> (b: i1)\n"
        "    %y.b = hw.instance \"y\" @Other(a: %a: i1) -> (b: i1)\n"
        "    hw.output %a : i1\n  }\n",
        _LEAF,
        "  hw.module private @Other(in %a : i1, out b : i1) {\n    hw.output %a : i1\n  }\n",
    )
    assert em.extract(src, "Top")[1] == em.extract(src, "Top")[1] == ["Top", "Other", "Leaf"]
