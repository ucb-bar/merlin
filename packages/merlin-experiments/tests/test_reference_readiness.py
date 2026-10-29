"""Complete site/region coverage cannot be replaced by a successful sample."""

from copy import deepcopy

import pytest
from merlin_experiments.phase1.feedback.reference_readiness import assess, require_ready


def complete_report():
    projection = dict(
        semantic_region_status="established",
        source_binary_correspondence="established",
        compatible_comparison="established",
        unclassified_instruction_classes=[],
    )
    return dict(
        input=dict(path="reference-kernels/model.elf", sha256="a" * 64),
        total=2,
        coverage=dict(accelerator_sites=7, selected_region_sites=7),
        regions=[
            dict(
                stable_id=f"region-{index}",
                accelerator_sites=sites,
                mechanism_projection=projection,
                cca=dict(
                    provenance=dict(
                        axis_evidence={
                            "dispatch.loop_offloaded": dict(status="inferred", evidence=["selected-isa-role-roster"])
                        }
                    )
                ),
            )
            for index, sites in enumerate((3, 4))
        ],
    )


def test_operator_readiness_requires_every_region_and_all_declared_instruction_sites():
    report = complete_report()
    result = assess([report])
    assert result["launch_ready"] and require_ready(result)
    assert result["performance_qualification"] is False
    incomplete = deepcopy(report)
    incomplete["regions"] = incomplete["regions"][:1]
    result = assess([incomplete])
    assert not result["launch_ready"]
    assert {row["kind"] for row in result["failures"]} == {
        "incomplete_or_duplicate_region_coverage",
        "unaccounted_instruction_sites",
    }
    with pytest.raises(ValueError, match="unresolved"):
        require_ready(result)


def test_partial_projection_remains_in_denominator_and_duplicates_never_cover_missing_regions():
    report = complete_report()
    report["regions"][1]["mechanism_projection"] = dict(status="partial", semantic_region_status="unknown")
    result = assess([report])
    assert result["region_count"] == 2 and result["ready_regions"] == 1
    assert result["missing"]["complete_upstream_semantic_region"] == 1
    report = complete_report()
    report["regions"] = [report["regions"][0]] * 2
    assert not assess([report])["launch_ready"]
    assert not assess([complete_report(), complete_report()])["launch_ready"]


def test_missing_coverage_fields_or_no_regions_never_mean_ready():
    report = complete_report()
    report.pop("coverage")
    assert not assess([report])["launch_ready"]
    assert not assess([])["launch_ready"]
