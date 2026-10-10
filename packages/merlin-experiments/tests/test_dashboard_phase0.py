"""The Phase 0 corpus-and-coverage view reads derivations, the written corpus and coverage as written.

Hidden capsules are counted and never named unless the page is built in operator-private mode; a gap the
coverage records is highlighted; a directory with no Phase 0 records renders "not recorded".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dashboard_fixtures import phase0_generation
from merlin_experiments.cli import main
from merlin_experiments.tracking import phase0, phase0_html, write_dashboard

SECRET = "HE00_secret_layer"


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*") if p.is_file()}


def test_inventory_requirements_and_cohorts_come_from_the_records(tmp_path):
    run = phase0_generation(tmp_path)
    s = phase0.summary(run)
    assert [d["name"] for d in s["derivations"]] == ["derive"]
    public = s["corpus"]["public"]
    assert public["n"] == 4 and public["by"]["family"] == {"contraction": 3, "movement": 1}
    assert public["tree"]["contraction"] == {"matmul": 1, "conv2d": 2}
    assert public["windows"] == {"k3x3/s1x1/d1x1/pad1x1": 1}
    assert s["corpus"]["hidden"]["n"] == 1 and s["corpus"]["hidden"]["rows"] is None
    cohorts = s["corpus"]["manifest"]["cohorts"]
    assert len(cohorts["phase1"]["members"]) == 3 and cohorts["phase2"]["members"] == ["_perf/PW00_conv"]
    p1 = s["coverage"]["phase1"]
    assert p1["cells"]["uncovered"] == ["movement/i8/aligned"]
    assert p1["axes"]["conv_geometry"]["uncovered"] == ["k7x7/s2x2/d1x1/pad3x3"]
    assert s["coverage"]["phase2"]["form"]["missing"] == ["c2"]
    guard = s["derivations"][0]["requirements"]["heldout_guard"]
    assert guard["gemm_shapes"] == 62 and "heldout_layer_shapes_sha256" not in guard


def test_public_page_counts_hidden_material_and_names_none_of_it(tmp_path):
    run = phase0_generation(tmp_path)
    before = _snapshot(run)
    page = phase0_html.render(phase0.summary(run))
    assert _snapshot(run) == before  # read-only
    assert SECRET not in page and "HE01_other" not in page and "e" * 64 not in page
    assert "Public mode" in page and "hidden capsules in the written corpus" in page
    for expected in (
        "movement/i8/aligned",
        "&#10005; gap",
        "conv2d_k3x3_bias",
        "matmul_m1_raw",
        "Phase 1 cohort",
        "phase1 &middot; 3 members",
        "<svg",
        "k7x7/s2x2/d1x1/pad3x3",
        "PW",
    ):
        assert expected in page, expected
    assert 'class="gap"' in page and 'class="hit"' in page
    assert "http://" not in page and "https://" not in page and "<link" not in page


def test_operator_private_page_names_hidden_capsules_and_says_so(tmp_path):
    run = phase0_generation(tmp_path)
    page = phase0_html.render(phase0.summary(run, operator_private=True))
    assert "OPERATOR-PRIVATE mode" in page and SECRET in page and "e" * 64 in page


def test_a_directory_without_phase0_records_is_not_recorded(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    s = phase0.summary(empty)
    assert s["derivations"] == [] and s["corpus"] is None and s["coverage"] is None
    page = phase0_html.render(s)
    assert "not recorded" in page
    assert {r["state"] for r in s["inventory"]} == {"absent"}


def test_a_derivation_root_without_coverage_shows_requirements_with_coverage_not_recorded(tmp_path):
    run = phase0_generation(tmp_path)
    s = phase0.summary(run / "derive")
    assert s["coverage"] is None and s["derivations"][0]["requirements"]["typed"]["n"] == 1
    page = phase0_html.render(s)
    assert "tall_skinny" in page and 'class="gap"' not in page and "&#10005; gap" not in page


def test_dashboard_cli_writes_a_phase0_page(tmp_path, capsys):
    run = phase0_generation(tmp_path)
    out = tmp_path / "pages" / "p0.html"
    assert main(["dashboard", "--phase0", str(run), "--out", str(out)]) == 0
    assert out.is_file() and SECRET not in out.read_text(encoding="utf-8")
    assert main(["dashboard", "--phase0", str(run), "--out", str(out), "--operator-private"]) == 0
    assert SECRET in out.read_text(encoding="utf-8")
    assert main(["dashboard", "--phase0", str(run), "--out", str(run / "inside.html")]) == 2  # never into inputs


@pytest.mark.parametrize("root", ["/work/merlin/tasks/capsule-audit/.tmp/r6/final2"])
def test_a_real_derivation_renders_when_present(root, tmp_path):
    if not Path(root).is_dir():
        pytest.skip("host-specific derivation not present")
    result = write_dashboard(phase0=Path(root), out=tmp_path / "real.html")
    page = Path(result["dashboard"]).read_text(encoding="utf-8")
    assert "Requirements vs covered" in page and "Typed required instances" in page
