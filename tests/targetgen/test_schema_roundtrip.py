"""Schema roundtrip tests.

Two scenarios:

1. **Capability spec roundtrip**: every fixture under
   ``target_specs/examples/`` loads, serialises back to YAML, and reloads
   into an equivalent ``TargetCapabilities``. Catches loader/dumper drift
   in ``tools/targetgen/loader.py``.

2. **Capability draft is loadable**: classifying a fixture and rendering
   the draft via ``./merlin targetgen classify`` produces YAML that
   ``load_capability_spec`` accepts (with stub values flagged in the
   header). This is what unblocks the CLI/MCP chain on novel targets:
   ``targetgen classify → modification-map`` no longer fails on the
   draft itself. The previous behavior was a sketch-only YAML that
   crashed the loader; that regression is what these tests guard
   against.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
from conftest import all_capability_specs
from targetgen import load_capability_spec
from targetgen.intake import build_source_inventory, classify_inventory

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "capability_path",
    all_capability_specs(),
    ids=lambda p: p.parent.name,
)
def test_capability_yaml_loads(capability_path: Path) -> None:
    """Each committed capability.yaml loads without raising."""
    caps = load_capability_spec(capability_path)
    assert caps.identity.name


@pytest.mark.parametrize(
    "capability_path",
    all_capability_specs(),
    ids=lambda p: p.parent.name,
)
def test_capability_dataclass_has_required_blocks(capability_path: Path) -> None:
    """Loaded capability has every top-level block expected by the planner."""
    caps = load_capability_spec(capability_path)
    d = asdict(caps)
    for required in (
        "identity",
        "platform",
        "execution",
        "isa",
        "operations",
        "tiles",
        "memory",
        "numeric",
        "runtime",
        "verification",
        "access",
    ):
        assert d[required] is not None, f"{capability_path.parent.name} missing {required}"


def _classify_fixture(name: str):
    inv = build_source_inventory(target=name, sources=[FIXTURES / name])
    return classify_inventory(inv)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "external_mlir_cuda_tile",
        "chipyard_gemmini_rocc",
        "radiance_gluon_gpu",
        "fft_generator_mmio",
    ],
)
def test_capability_draft_is_loadable(fixture_name: str, tmp_path: Path) -> None:
    """The CLI's draft must round-trip through ``load_capability_spec``.

    Previously the CLI emitted a sketch-only YAML (target + _targetgen_intake)
    that the loader rejected, breaking the ``classify → modification-map``
    chain. The CLI now delegates to ``intake.draft.render_loadable_draft_yaml``
    so the draft is loadable on first try (with stub values flagged in the
    header for the operator).
    """
    import targetgen_cmd  # imported here so tests/conftest sets sys.path first

    classification = _classify_fixture(fixture_name)
    rendered = targetgen_cmd._render_capability_draft(classification)

    draft_path = tmp_path / "draft.yaml"
    draft_path.write_text(rendered, encoding="utf-8")
    caps = load_capability_spec(draft_path)
    assert caps.identity.name == fixture_name


def test_capability_draft_header_lists_unresolved_fields() -> None:
    """The draft is loadable but every stub the operator must replace is
    enumerated in the header comment block, so it cannot be silently
    promoted as if it were a real spec."""
    import targetgen_cmd

    classification = _classify_fixture("external_mlir_cuda_tile")
    rendered = targetgen_cmd._render_capability_draft(classification)
    # Header has UNRESOLVED block listing fields that need filling in.
    assert "UNRESOLVED" in rendered, "draft must flag unresolved fields"
    # The classifier signals are recorded in the header for review.
    assert classification.primary_integration in rendered
    for style in classification.targetgen_styles:
        assert style in rendered
