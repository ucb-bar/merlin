"""The run explorer indexes every phase's runs, checks the lineage the records bind, and compares runs.

The fixture target has a sealed Phase 0 release, two Phase 1 runs (one admitted under that release's
seal, one naming a seal no release has) and two paired Phase 2 experiments (one bound to the Phase 1
run's frozen submission, one bound to a different digest).  The broken and inconsistent links must be
shown -- the mismatch loudly -- and nothing private appears unless the page is operator-private.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from dashboard_fixtures import HOUR, T0, TARGET, phase0_generation, phase1_run, write
from merlin_experiments.cli import main
from merlin_experiments.spec import SpecError
from merlin_experiments.tracking import explorer, write_dashboard

DIGEST = "d" * 64
NOTE = "reviewed the hidden roster by hand"


@pytest.fixture()
def out_root(tmp_path, monkeypatch):
    root = tmp_path / "out"
    for sub in ("runs", "artifacts", "build"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    return root


def release(out_root: Path) -> Path:
    from merlin.common.artifacts import declared_home

    root = declared_home("phase0-releases") / TARGET / "phase0-20260920T000000Z-abc1234"
    write(
        root / "private" / "preparation.json",
        {
            "target": TARGET,
            "payload_sha256": "p" * 64,
            "prepared_at": "2026-09-20T00:00:00+00:00",
            "admission": {"hidden_admitted": 4, "public_capsules": 12},
            "generation_lineage": {"capsules": ["a", "b", "c"], "coverage": {"phase1": {"status": "complete"}}},
        },
    )
    write(
        root / "private" / "seal.json",
        {"review_digest": DIGEST, "review": {"reviewed_by": "operator", "note": NOTE, "at": "2026-09-20"}},
    )
    corpus = phase0_generation(root / "scratch") / "phase0" / "capsules"
    (root / "payload").mkdir(parents=True)
    corpus.rename(root / "payload" / "corpus")
    return root


def paired(out_root: Path, name: str, functional_run_id: str, sha: str, ratio_cycles: int = 80) -> Path:
    root = out_root / "runs" / TARGET / "phase2" / name
    write(
        root / "state" / ("checkpoint.0000." + "0" * 64 + ".json"), {"index": 0, "stage": "predeclared", "evidence": {}}
    )
    cell = out_root / "runs" / TARGET / "perf" / f"{name}__trial_00__tuning"
    write(
        root / "state" / ("checkpoint.0001." + "1" * 64 + ".json"),
        {
            "index": 1,
            "stage": "measurement:trial_00:tuning",
            "evidence": {"path": str(cell / "campaign_manifest.json")},
        },
    )
    write(
        cell / "campaign_manifest.json",
        {
            "status": "GO",
            "functional_run_id": functional_run_id,
            "functional_submission_sha256": sha,
            "completion": {"complete": True, "expected": 4, "reported": 4},
        },
    )
    write(
        cell / "paired_cycles.json",
        [
            {
                "family": "PK",
                "capsule": "case",
                "replicate": r,
                "baseline_cycles": 100,
                "candidate_cycles": ratio_cycles,
                "comparable": True,
                "baseline_over_candidate": 100 / ratio_cycles,
            }
            for r in ("r000", "r001")
        ],
    )
    return root


@pytest.fixture()
def chain(out_root):
    rel = release(out_root)
    good = phase1_run(out_root, t0=T0)
    env = yaml.safe_load((good / "environment.yaml").read_text())
    env["corpus_review"] = {"review_digest": DIGEST, "release": str(rel)}
    (good / "environment.yaml").write_text(yaml.safe_dump(env))
    stray = phase1_run(out_root, t0=T0 + 10 * HOUR)
    env = yaml.safe_load((stray / "environment.yaml").read_text())
    env["corpus_review"] = {"review_digest": "9" * 64, "release": "/nowhere"}
    (stray / "environment.yaml").write_text(yaml.safe_dump(env))
    frozen = json.loads((good / "freeze.json").read_text())["submission_sha256"]
    bound = paired(out_root, "exp_bound", good.name, frozen)
    drifted = paired(out_root, "exp_drifted", good.name, "0" * 64, ratio_cycles=50)
    orphan = paired(out_root, "exp_orphan", "no_such_run", frozen)
    return {"release": rel, "good": good, "stray": stray, "bound": bound, "drifted": drifted, "orphan": orphan}


def test_lineage_links_what_the_records_bind_and_flags_the_rest(chain):
    collected = explorer.collect(TARGET, T0 + 20 * HOUR, operator_private=False)
    graph = explorer.lineage(collected)
    states = {(e["from"][1], e["to"][1]): e["state"] for e in graph["edges"]}
    assert states[(chain["release"].name, chain["good"].name)] == "ok"
    assert states[(chain["good"].name, "exp_bound")] == "ok"
    assert states[(chain["good"].name, "exp_drifted")] == "mismatch"
    notes = {(n["run"], n["state"]) for n in graph["notes"]}
    assert (chain["stray"].name, "broken") in notes and ("exp_orphan", "broken") in notes


def test_explorer_page_indexes_runs_links_their_pages_and_shouts_mismatches(chain, out_root, tmp_path):
    out = tmp_path / "dash" / "explorer.html"
    result = write_dashboard(target=TARGET, explorer=True, out=out, now=T0 + 20 * HOUR)
    assert result["kind"] == "explorer"
    page = out.read_text(encoding="utf-8")
    for expected in (
        "Lineage across phases",
        "lineage mismatch",
        "exp_drifted",
        chain["good"].name,
        chain["release"].name,
        "operator",
        'class="sortable"',
        "data-filter",
        "Phase 2 runs",
        f'href="{chain["good"].name}.html"',
        "1.250x",
        "matches no release",
    ):
        assert expected in page, expected
    assert NOTE not in page and "http://" not in page and page.count("<script>") == 1
    siblings = {p.name for p in out.parent.iterdir()}
    assert {f"{chain['good'].name}.html", "exp_bound.html", explorer.page_name("0", chain["release"].name)} <= siblings
    assert "HE00_secret_layer" not in (out.parent / explorer.page_name("0", chain["release"].name)).read_text()
    private = tmp_path / "private" / "explorer.html"
    write_dashboard(target=TARGET, explorer=True, out=private, operator_private=True, now=T0 + 20 * HOUR)
    assert NOTE in private.read_text(encoding="utf-8")


def test_compare_two_phase1_runs(chain, tmp_path):
    out = tmp_path / "cmp.html"
    assert write_dashboard(compare=(chain["good"], chain["stray"]), out=out, now=T0 + 20 * HOUR)["kind"] == "compare"
    page = out.read_text(encoding="utf-8")
    for expected in (
        "Capsules passed against hours since each run",
        "Per-capsule tier",
        "cap_mm",
        "Tokens and time",
        "40097",
    ):
        assert expected in page, expected


def test_compare_two_paired_experiments_and_refuse_mixed_phases(chain, tmp_path):
    out = tmp_path / "cmp2.html"
    write_dashboard(compare=(chain["bound"], chain["drifted"]), out=out)
    page = out.read_text(encoding="utf-8")
    assert "1.250x" in page and "2.000x" in page and "+60.0%" in page
    with pytest.raises(SpecError):
        write_dashboard(compare=(chain["good"], chain["bound"]), out=tmp_path / "x.html")


def test_compare_two_phase0_runs_diffs_the_corpus(tmp_path):
    a = phase0_generation(tmp_path / "a")
    b = phase0_generation(tmp_path / "b")
    (b / "phase0" / "capsules" / "isa" / "SY_move" / "capsule.yaml").unlink()
    doc = yaml.safe_load((b / "phase0" / "capsules" / "layers" / "MF_mm_tall" / "capsule.yaml").read_text())
    doc["operation"]["attributes"]["epilogue"] = ["relu"]
    (b / "phase0" / "capsules" / "layers" / "MF_mm_tall" / "capsule.yaml").write_text(yaml.safe_dump(doc))
    out = tmp_path / "cmp0.html"
    assert main(["dashboard", "--compare", str(a), str(b), "--out", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    assert "isa/SY_move" in page and "epilogue: acc_scale -&gt; relu" in page and "HE00_secret_layer" not in page


def test_an_empty_target_explorer_says_not_recorded(out_root, tmp_path):
    out = tmp_path / "empty.html"
    write_dashboard(target=TARGET, explorer=True, out=out)
    page = out.read_text(encoding="utf-8")
    assert "not recorded" in page and "Phase 1 runs" in page
