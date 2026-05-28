"""Property-based invariants for ``classify_inventory``.

Hypothesis generates arbitrary subsets of the source-finding kinds the
scanners can emit, builds a minimal ``SourceInventory``, and exercises the
classifier. The invariants are conservative: things that must hold for
every input shape, however weird.
"""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip(
    "hypothesis",
    reason="hypothesis is pinned in pyproject.toml but missing from the active env",
)
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from targetgen.intake import (  # noqa: E402
    chipyard_scanner,
    chisel_scanner,
    classify_inventory,
    cmake_scanner,
    docs_scanner,
    hal_scanner,
    llvm_scanner,
    mlir_scanner,
    rtl_scanner,
    systemc_scanner,
)
from targetgen.intake.classifier import SOURCE_TO_TARGETGEN  # noqa: E402
from targetgen.model import SourceInventory  # noqa: E402

VALID_TARGETGEN_STYLES = frozenset({"runtime_hal", "structured_text_isa", "post_global_plugin", "llvm_ukernel"})

ALL_SCANNER_KINDS: tuple[str, ...] = tuple(
    sorted(
        set(
            mlir_scanner.DETECTED_KINDS
            + cmake_scanner.DETECTED_KINDS
            + llvm_scanner.DETECTED_KINDS
            + hal_scanner.DETECTED_KINDS
            + chipyard_scanner.DETECTED_KINDS
            + chisel_scanner.DETECTED_KINDS
            + rtl_scanner.DETECTED_KINDS
            + systemc_scanner.DETECTED_KINDS
            + docs_scanner.DETECTED_KINDS
        )
    )
)


def _inventory_from_kinds(kinds: list[str]) -> SourceInventory:
    return SourceInventory(
        target="propcheck",
        repositories=[],
        findings=[],
        detected_source_kinds=sorted(set(kinds)),
        missing_information=[],
    )


@given(st.sets(st.sampled_from(ALL_SCANNER_KINDS), max_size=8))
@settings(
    max_examples=200,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_classifier_never_crashes(kinds: set[str]) -> None:
    classify_inventory(_inventory_from_kinds(list(kinds)))


@given(st.sets(st.sampled_from(ALL_SCANNER_KINDS), max_size=8))
@settings(max_examples=200, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_primary_integration_in_canonical_styles(kinds: set[str]) -> None:
    cls = classify_inventory(_inventory_from_kinds(list(kinds)))
    assert cls.primary_integration in VALID_TARGETGEN_STYLES


@given(st.sets(st.sampled_from(ALL_SCANNER_KINDS), max_size=8))
@settings(max_examples=200, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_primary_in_targetgen_styles(kinds: set[str]) -> None:
    cls = classify_inventory(_inventory_from_kinds(list(kinds)))
    assert cls.primary_integration in cls.targetgen_styles


@given(st.sets(st.sampled_from(ALL_SCANNER_KINDS), max_size=8))
@settings(max_examples=200, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_targetgen_styles_are_subset_of_canonical(kinds: set[str]) -> None:
    cls = classify_inventory(_inventory_from_kinds(list(kinds)))
    assert set(cls.targetgen_styles).issubset(VALID_TARGETGEN_STYLES)


@given(st.sets(st.sampled_from(ALL_SCANNER_KINDS), max_size=8))
@settings(max_examples=200, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_confidence_in_unit_interval(kinds: set[str]) -> None:
    cls = classify_inventory(_inventory_from_kinds(list(kinds)))
    assert 0.0 <= cls.confidence <= 1.0


@given(st.sets(st.sampled_from(ALL_SCANNER_KINDS), max_size=8))
@settings(max_examples=200, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_targetgen_styles_never_empty(kinds: set[str]) -> None:
    cls = classify_inventory(_inventory_from_kinds(list(kinds)))
    assert len(cls.targetgen_styles) >= 1


def test_empty_inventory_defaults_to_llvm_ukernel() -> None:
    cls = classify_inventory(_inventory_from_kinds([]))
    assert cls.source_styles == []
    assert cls.targetgen_styles == ["llvm_ukernel"]
    assert cls.primary_integration == "llvm_ukernel"
    assert cls.confidence == 0.0


def test_source_to_targetgen_table_is_complete_and_canonical() -> None:
    """Every entry in the source→targetgen table maps to canonical styles."""
    for source_style, mapped in SOURCE_TO_TARGETGEN.items():
        assert mapped, f"{source_style} maps to no targetgen styles"
        for tg in mapped:
            assert tg in VALID_TARGETGEN_STYLES, (source_style, tg)
